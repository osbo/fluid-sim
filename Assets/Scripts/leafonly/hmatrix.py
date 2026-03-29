"""Weak-admissibility H-matrix partition on a uniform leaf grid (static, vectorized)."""

from typing import Tuple

import torch

from .config import HMATRIX_ETA, MAX_NUM_LEAVES


def standard_admissible_unique_blocks(num_units: int, eta: float, device, dtype=torch.float32) -> torch.Tensor:
    """
    Vectorized weak admissibility (same as numpy reference). Returns all unique (r0, c0, S) tiles
    in leaf-index space, shape [M, 3] int64, sorted lexicographically.
    """
    nu = int(num_units)
    if nu <= 0:
        return torch.zeros(0, 3, device=device, dtype=torch.long)
    r = torch.arange(nu, device=device, dtype=torch.long).view(nu, 1)
    c = torch.arange(nu, device=device, dtype=torch.long).view(1, nu)
    S_max = torch.ones((nu, nu), device=device, dtype=torch.long)
    max_level = int(nu.bit_length() - 1) if nu > 1 else 0
    eta_t = torch.tensor(float(eta), device=device, dtype=dtype)
    for Lv in range(1, max_level + 1):
        S = 1 << Lv
        r0 = (r // S) * S
        c0 = (c // S) * S
        d = (r0 - c0).abs().to(dtype) - float(S)
        S_f = torch.tensor(float(S), device=device, dtype=dtype)
        admissible = (d > 0) & (S_f <= eta_t * d)
        S_max = torch.where(admissible, torch.full_like(S_max, S), S_max)
    final_r0 = (r // S_max) * S_max
    final_c0 = (c // S_max) * S_max
    flat = torch.stack([final_r0.reshape(-1), final_c0.reshape(-1), S_max.reshape(-1).long()], dim=-1)
    return torch.unique(flat, dim=0)


def off_blocks_strict_upper(num_units: int, eta: float, device, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    H-matrix off-diagonal tiles: unique admissible blocks excluding on-diagonal 1×1 leaves
    (those are handled by the diagonal stack). Keeps r0 <= c0 convention (one tile per symmetric
    coupling; apply uses M and M.T on row/col strips).

    Returns (r0, c0, S) each int64 [M_off].
    """
    u = standard_admissible_unique_blocks(num_units, eta, device, dtype)
    if u.numel() == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z, z
    r0, c0, S = u[:, 0], u[:, 1], u[:, 2]
    on_diag_unit = (r0 == c0) & (S == 1)
    upper = r0 <= c0
    sel = upper & ~on_diag_unit
    r0 = r0[sel]
    c0 = c0[sel]
    S = S[sel]
    return r0, c0, S


def precompute_hmatrix_off_buffers(num_units=None, eta=None, device=None, dtype=torch.float32):
    """Static off-block origins and sizes for MAX_NUM_LEAVES (compile-time constant layout)."""
    if num_units is None:
        num_units = MAX_NUM_LEAVES
    if eta is None:
        eta = HMATRIX_ETA
    dev = device or torch.device("cpu")
    r0, c0, S = off_blocks_strict_upper(num_units, eta, dev, dtype=dtype)
    return r0, c0, S


def hm_leaf_mean_pool_weights(
    r0: torch.Tensor, c0: torch.Tensor, S: torch.Tensor, num_leaves: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Static (M, K) weights: for h_k shape (B, K, L, C), einsum('mk,bklc->bmlc', W_row, h_k) gives the row-strip
    mean pool for each H off-block; same for W_col / column strip. Built once at module init (Python loop, small M).
    """
    M = int(r0.shape[0])
    K = int(num_leaves)
    W_row = torch.zeros(M, K, dtype=torch.float32)
    W_col = torch.zeros(M, K, dtype=torch.float32)
    r0_l = r0.long().cpu().tolist()
    c0_l = c0.long().cpu().tolist()
    S_l = S.long().cpu().tolist()
    for i in range(M):
        rr, cc, ss = r0_l[i], c0_l[i], S_l[i]
        inv = 1.0 / float(ss)
        W_row[i, rr : rr + ss] = inv
        W_col[i, cc : cc + ss] = inv
    return W_row, W_col


def hmatrix_off_masks_and_feats(
    num_off_blocks: int,
    leaf_size: int,
    device,
    dtype=torch.float32,
    num_extra: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Legacy helper: dense all-ones mask and zero edge_feats of shape (M_off, T, T), T = L + num_extra.

    Training/inference use ``build_leaf_block_connectivity``, which builds masks and edge_feats from
    the graph (reachability + per-cell mean edge features) instead of calling this.
    """
    L = int(leaf_size)
    T = L + int(num_extra)
    if num_off_blocks <= 0:
        zm = torch.zeros(0, T, T, device=device, dtype=dtype)
        zf = torch.zeros(0, T, T, 4, device=device, dtype=dtype)
        return zm, zf
    m = torch.ones(num_off_blocks, T, T, device=device, dtype=dtype)
    f = torch.zeros(num_off_blocks, T, T, 4, device=device, dtype=dtype)
    return m, f


def num_hmatrix_off_blocks(num_units=None, eta=None) -> int:
    r0, _, _ = precompute_hmatrix_off_buffers(num_units=num_units, eta=eta, device=torch.device("cpu"))
    return int(r0.shape[0])


# CPU indices for apply/unpack (static layout); import from here in InspectModel / tests.
HM_R0_CPU, HM_C0_CPU, HM_S_CPU = precompute_hmatrix_off_buffers(device=torch.device("cpu"))
NUM_HMATRIX_OFF_BLOCKS = int(HM_R0_CPU.shape[0])
# Row/column strip mean weights for building pooled H off-diagonal token streams; sum weights below for FMM moments.
HM_POOL_W_ROW_CPU, HM_POOL_W_COL_CPU = hm_leaf_mean_pool_weights(
    HM_R0_CPU, HM_C0_CPU, HM_S_CPU, MAX_NUM_LEAVES
)


def hm_leaf_sum_pool_weights(
    r0: torch.Tensor, c0: torch.Tensor, S: torch.Tensor, num_leaves: int
) -> tuple[torch.Tensor, torch.Tensor]:
    M = int(r0.shape[0])
    K = int(num_leaves)
    W_row = torch.zeros(M, K, dtype=torch.float32)
    W_col = torch.zeros(M, K, dtype=torch.float32)
    r0_l, c0_l, S_l = r0.long().cpu().tolist(), c0.long().cpu().tolist(), S.long().cpu().tolist()
    for i in range(M):
        rr, cc, ss = r0_l[i], c0_l[i], S_l[i]
        W_row[i, rr : rr + ss] = 1.0
        W_col[i, cc : cc + ss] = 1.0
    return W_row, W_col


# Export sum weights for FMM-style pooling
HM_SUM_W_ROW_CPU, HM_SUM_W_COL_CPU = hm_leaf_sum_pool_weights(
    HM_R0_CPU, HM_C0_CPU, HM_S_CPU, MAX_NUM_LEAVES
)
# Plain Python ints for strip bounds (no Tensor.item / Dynamo graph breaks in apply).
HM_R0_LIST = [int(x) for x in HM_R0_CPU.tolist()]
HM_C0_LIST = [int(x) for x in HM_C0_CPU.tolist()]
HM_S_LIST = [int(x) for x in HM_S_CPU.tolist()]
# Vectorized prolongation: leaf indices and repeat counts (avoid M_h slice-+= autograd nodes).
HM_PROLONG_ROW_LEAF_IDX = torch.cat(
    [torch.arange(int(r0), int(r0) + int(s), dtype=torch.long) for r0, s in zip(HM_R0_LIST, HM_S_LIST)]
)
HM_PROLONG_COL_LEAF_IDX = torch.cat(
    [torch.arange(int(c0), int(c0) + int(s), dtype=torch.long) for c0, s in zip(HM_C0_LIST, HM_S_LIST)]
)
HM_PROLONG_REPEAT_PER_BLOCK = torch.tensor(HM_S_LIST, dtype=torch.long)
# For each prolongation slot t in 0..T-1, which off-block row (0..M_h-1) does y_r[:,i,:,:] come from?
HM_PROLONG_GATHER_IDX = torch.cat(
    [torch.full((int(s),), int(i), dtype=torch.long) for i, s in enumerate(HM_S_LIST)]
)
