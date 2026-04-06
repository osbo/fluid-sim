#!/usr/bin/env python3
"""
Cross-check LeafOnly GPU inference / apply against Unity [LeafOnlyParity] logs.

Unity: enable FluidSimulator.debugLeafOnlyParityLog, assign FluidSimulator.leafOnlyEmbedShader
(LeafOnlyEmbed.compute), ensure weights load from StreamingAssets; run one pressure solve. Copy console
lines starting with [LeafOnlyParity]. This script prints the same tensor names and summary stats
(min, max, mean, sum_abs, l2) from PyTorch on a recorded frame (default: dataset index 0 =
lexicographically first frame under --data_folder). Python lines use phase=pytorch; Unity uses phase=unity.

Embedding parity: compare tensor ``token_after_enc`` (Unity runs embed + enc_input_proj on GPU after the
input tensors). PyTorch reference is ``model.embed`` then ``model.enc_input_proj`` (see end of main()).

Layer 0 (non-highway): compare ``h_diag_after_layer0`` and ``off_stream_after_layer0`` (Unity CPU eager-softmax
mirror in ``FluidLeafOnlyCpuLayer1Parity.cs`` vs PyTorch ``_layer0_transformer_reference_tensors`` with
``eager_attention=True``).

Dense diagonal attention RPE: ``diag_edge_feats`` (K×L×L×4), same as ``build_diag_dense_edge_feats_from_positions``
(no attn_mask; channels 0–2 = position deltas, 3 = mean in-leaf matrix entry per cell). Unity builds this on GPU
(``LeafOnlyDiagEdgeFeats.compute``) from ``leafXPacked`` + CSR ``A``.

Static H off-block partition: ``hmatrix_meta`` + ``hmatrix_off_r0_c0_s_flat`` (length ``3 * M_off``), matching
``leafonly/hmatrix.py`` / ``LeafOnlyHMatrixStatic.cs`` (``MAX_NUM_LEAVES``, ``HMATRIX_ETA``); no per-frame data.

Dense off-diagonal H-tile RPE: ``off_edge_feats`` (``M_off×L×L×4``), ``build_hmatrix_off_dense_rpe_from_positions``;
Unity ``LeafOnlyOffEdgeFeats.compute``. Logs: ``off_edge_feats_meta``, full stats, ``_head[32]``, ``_at{mid}_head[48]``,
``_at{n-32}_head[32]``.

Known Unity vs dataset differences (see --match_unity_shader):
  • Zeroing normalized x[:, 5:8] in PyTorch vs filled diffusion from diffusion_gradient.bin.
  • LeafOnlyInputs.compute uses scale_A = min(max row |A| sum, max col |A| sum) (same as FluidGraphDataset).

Run (from Assets/Scripts):
  python3 InspectParity.py --frame 0
  python3 InspectParity.py --frame 0 --match_unity_shader --weights ../StreamingAssets/leaf_only_weights_sim_8192.bytes
  python3 InspectParity.py --frame 0 --print_unity_inputs --inputs_parity_only

Optional: compare apply step with r=all_ones (default) or random (--r random).
  --print_unity_inputs emits the same tensor names as Unity FluidLeafOnlyInputs parity logs (x_leaf, ...).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_script_dir = Path(__file__).resolve().parent
# Same file Unity loads via StreamingAssets (FluidSimulator.leafOnlyWeightsStreamingAssetsName).
_DEFAULT_WEIGHTS_PATH = _script_dir.parent / "StreamingAssets" / "leaf_only_weights_sim_8192.bytes"
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import LeafOnly as _leaf_only_script  # noqa: E402

from leafonly.architecture import (  # noqa: E402
    LeafOnlyNet,
    apply_block_diagonal_M_physical,
    default_attention_layout,
    pool_leaf_edge_feats,
    unpack_precond,
)
from leafonly.checkpoint import leaf_only_arch_from_checkpoint, load_leaf_only_weights  # noqa: E402
from leafonly.config import (  # noqa: E402
    HMATRIX_ETA,
    LEAF_SIZE,
    MAX_MIXED_SIZE,
    MAX_NUM_LEAVES,
    problem_padded_num_nodes,
)


def _leaf_align_n(num_nodes_real: int, mode: str) -> int:
    """Unity buffers use ceil(numNodes/LEAF_SIZE) leaves; InspectModel uses floor (truncate)."""
    L = int(LEAF_SIZE)
    n = int(num_nodes_real)
    if mode == "ceil":
        return ((n + L - 1) // L) * L
    if mode == "floor":
        return (n // L) * L
    raise ValueError(mode)
from leafonly.data import NODE_DTYPE, FluidGraphDataset, build_leaf_block_connectivity  # noqa: E402


def _summarize(name: str, t: torch.Tensor) -> None:
    x = t.detach().float().cpu().reshape(-1)
    n = int(x.numel())
    if n == 0:
        print(f"[LeafOnlyParity] phase=pytorch tensor={name} n=0 (empty)")
        return
    xmin = float(x.min())
    xmax = float(x.max())
    xmean = float(x.mean())
    sum_abs = float(x.abs().sum())
    l2 = float(torch.sqrt((x * x).sum()))
    print(
        f"[LeafOnlyParity] phase=pytorch tensor={name} n={n} min={xmin:.9g} max={xmax:.9g} "
        f"mean={xmean:.9g} sum_abs={sum_abs:.9g} l2={l2:.9g}"
    )


def _head(name: str, t: torch.Tensor, k: int) -> None:
    x = t.detach().float().cpu().reshape(-1)[:k]
    if x.numel() == 0:
        print(f"[LeafOnlyParity] phase=pytorch tensor={name}_head[0]=")
        return
    parts = ",".join(f"{float(v):.9g}" for v in x)
    print(f"[LeafOnlyParity] phase=pytorch tensor={name}_head[{int(x.numel())}]={parts}")


def _parity_head_at(name: str, t: torch.Tensor, start: int, k: int) -> None:
    flat = t.detach().float().cpu().reshape(-1)
    n = int(flat.numel())
    if n == 0 or start < 0 or start >= n:
        print(f"[LeafOnlyParity] phase=pytorch tensor={name}_at{start}_head[0]=")
        return
    c = min(k, n - start)
    parts = ",".join(f"{float(flat[start + i]):.9g}" for i in range(c))
    print(f"[LeafOnlyParity] phase=pytorch tensor={name}_at{start}_head[{c}]={parts}")


def _layer0_transformer_reference_tensors(
    model: LeafOnlyNet,
    h: torch.Tensor,
    ef_diag: torch.Tensor,
    ef_off: torch.Tensor,
    edge_index: torch.Tensor,
    edge_values: torch.Tensor,
    positions: torch.Tensor,
    *,
    eager_attention: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Match ``_apply_transformer_stacks`` for layer index 0 with ``highway_ffn=False`` and ``M_off>0``:
    diag block then strip + off block. Uses ``eager_attention=True`` so logits match manual softmax (Unity CPU).
    """
    if bool(getattr(model, "highway_ffn", False)):
        raise ValueError("layer0 parity helper requires checkpoint with highway_ffn=0 / use_highways=False")
    B, N, C_h = h.shape
    L = int(model.leaf_size)
    K = N // L
    M_off = int(model.num_h_off)

    ef = ef_diag
    if int(model.diag_token_pool) > 1 and bool(model.diag_dense_attention) and ef is not None:
        ef = pool_leaf_edge_feats(ef, L, int(model.diag_token_pool))

    h_diag = model._pool_full_leaf_to_diag_tokens(h, B, K, L, C_h)
    h_lv_diag = h_diag.view(B, K, int(model.leaf_apply_size), C_h)
    diag_prep_block = None if model.diag_dense_attention else h_lv_diag.mean(dim=2, keepdim=True)
    diag_prep_matrix = h.mean(dim=1, keepdim=True).unsqueeze(1).expand(-1, K, -1, -1)
    attn_kw_diag = dict(
        edge_index=edge_index,
        edge_values=edge_values,
        positions=positions,
        save_attention=False,
        attn_mask=None,
        edge_feats=ef,
        prep_block_node=diag_prep_block,
        prep_matrix_node=diag_prep_matrix,
        eager_attention=eager_attention,
    )
    blk_d = model.blocks[0]
    x_mid_d = blk_d.forward_attn(h_diag, **attn_kw_diag)
    h_diag = blk_d.forward_mlp(x_mid_d)

    if M_off <= 0:
        return h_diag, None

    oe = ef_off
    if bool(model.off_diag_dense_attention) and int(model.off_token_pool) > 1:
        oe = pool_leaf_edge_feats(oe, L, int(model.off_token_pool))
    Ls_o = int(model.leaf_apply_off)
    if oe.dim() == 4:
        oe5 = oe.unsqueeze(0).expand(B, M_off, -1, -1, -1)
    else:
        oe5 = oe.contiguous()
    off_edge_feats = oe5.reshape(B * M_off, 1, Ls_o, Ls_o, 4)

    strip0 = model._build_off_strip(h, B, K, L, C_h, M_off)
    Bm = B * M_off
    off_prep_block = (
        None
        if model.off_diag_dense_attention
        else strip0.view(Bm, 1, Ls_o, C_h).mean(dim=2, keepdim=True)
    )
    off_prep_matrix = strip0.mean(dim=1, keepdim=True).unsqueeze(1).unsqueeze(1)
    attn_kw_off = dict(
        edge_index=edge_index,
        edge_values=edge_values,
        positions=positions,
        save_attention=False,
        attn_mask=None,
        edge_feats=off_edge_feats,
        prep_block_node=off_prep_block,
        prep_matrix_node=off_prep_matrix,
        eager_attention=eager_attention,
    )
    strip = model._build_off_strip(
        model._diag_tokens_to_full_leaf(h_diag, B, K, L, C_h), B, K, L, C_h, M_off
    )
    off_in = strip
    blk_o = model.off_diag_blocks[0]
    x_mid_o = blk_o.forward_attn(off_in, **attn_kw_off)
    off_stream = blk_o.forward_mlp(x_mid_o)
    return h_diag, off_stream


def _unity_stats_and_global(
    num_nodes: int,
    mass: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match LeafOnlyPreconditioner FinalizeNodeStats + globalFeatBuf (diffusion globals zero)."""
    n = int(num_nodes)
    sign_m = np.sign(mass.astype(np.float64))
    log_m = sign_m * np.log1p(np.abs(mass.astype(np.float64)))
    mass_mean = float(log_m.sum() / n)
    mass_var = max(0.0, float((log_m**2).mean() - mass_mean**2))
    mass_std = math.sqrt(mass_var) + 1e-6

    diag = np.zeros(n, dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        if int(r) == int(c):
            diag[int(r)] = float(v)

    diag_mean = float(diag.sum() / n)
    diag_var = max(0.0, float((diag**2).mean() - diag_mean**2))
    diag_std = math.sqrt(diag_var) + 1e-6

    row_sum = np.zeros(n, dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        row_sum[int(r)] += abs(float(v))
    rmx = float(row_sum.max()) if n else 1.0
    if rmx <= 0.0:
        rmx = 1.0

    stats = np.array(
        [mass_mean, mass_std, diag_mean, diag_std, rmx, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    global_feat = np.zeros(12, dtype=np.float32)
    global_feat[0] = math.log2(max(1, n))
    global_feat[1] = 0.0
    global_feat[2] = mass_mean
    global_feat[3] = mass_std
    global_feat[4] = diag_mean
    global_feat[5] = diag_std
    # [6..11] left 0 — matches Unity globalFeatBuf
    return stats, global_feat


def _read_raw_coo(frame_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fp = Path(frame_path)
    rows = np.fromfile(fp / "edge_index_rows.bin", dtype=np.uint32)
    cols = np.fromfile(fp / "edge_index_cols.bin", dtype=np.uint32)
    vals = np.fromfile(fp / "A_values.bin", dtype=np.float32)
    return rows, cols, vals


def _scale_a_and_csr_scaled(
    rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, num_nodes: int
) -> tuple[float, np.ndarray]:
    """Match LeafOnlyInputs.compute / FluidGraphDataset: min(max row abs-sum, max col abs-sum)."""
    n = int(num_nodes)
    row_sums = np.zeros(max(n, 1), dtype=np.float64)
    col_sums = np.zeros(max(n, 1), dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        ri, ci = int(r), int(c)
        if 0 <= ri < n:
            row_sums[ri] += abs(float(v))
        if 0 <= ci < n:
            col_sums[ci] += abs(float(v))
    max_row = float(row_sums.max()) if n else 1.0
    max_col = float(col_sums.max()) if n else 1.0
    scale_a = float(min(max_row, max_col))
    if not math.isfinite(scale_a) or scale_a <= 1e-30:
        scale_a = 1.0
    csr_scaled = (vals.astype(np.float64) / scale_a).astype(np.float32)
    return scale_a, csr_scaled


def _unity_leaf_moments_np(
    num_nodes: int,
    mass: np.ndarray,
    diffusion_grad: np.ndarray,
    diag: np.ndarray,
    scale_a: float,
) -> np.ndarray:
    """Match LeafOnly_WriteScaleAndMoments in LeafOnlyInputs.compute (16 floats; tail unused on GPU)."""
    n = int(num_nodes)
    if n <= 0:
        out = np.zeros(16, dtype=np.float32)
        out[0] = 1.0
        return out

    inv_n = 1.0 / float(n)
    m = mass[:n].astype(np.float64)
    m_sym = np.sign(m) * np.log1p(np.abs(m))
    sum_m = float(m_sym.sum())
    sum_m2 = float((m_sym * m_sym).sum())
    mass_mean = sum_m * inv_n
    mass_var = max(0.0, sum_m2 * inv_n - mass_mean * mass_mean)
    mass_std = math.sqrt(mass_var) + 1e-8

    g = diffusion_grad[:n].astype(np.float64)
    d_sym = np.sign(g) * np.log1p(np.abs(g))
    sum_d = d_sym.sum(axis=0)
    sum_d2 = (d_sym * d_sym).sum(axis=0)
    d_mean = sum_d * inv_n
    d_var = np.maximum(0.0, sum_d2 * inv_n - d_mean * d_mean)
    d_std = np.sqrt(d_var) + 1e-8

    diag_f = diag[:n].astype(np.float64)
    s = scale_a if math.isfinite(scale_a) and scale_a > 1e-30 else 1.0
    dn = diag_f / s
    dg_mean = float(dn.mean())
    dg_var = max(0.0, float((dn * dn).mean() - dg_mean * dg_mean))
    dg_std = math.sqrt(dg_var) + 1e-8

    moments = np.zeros(16, dtype=np.float32)
    moments[0] = np.float32(mass_mean)
    moments[1] = np.float32(mass_std)
    moments[2] = np.float32(d_mean[0])
    moments[3] = np.float32(d_std[0])
    moments[4] = np.float32(d_mean[1])
    moments[5] = np.float32(d_std[1])
    moments[6] = np.float32(d_mean[2])
    moments[7] = np.float32(d_std[2])
    moments[8] = np.float32(dg_mean)
    moments[9] = np.float32(dg_std)
    moments[10] = np.float32(math.log2(max(1.0, float(n))))
    moments[11] = 0.0
    return moments


def _global_features_from_moments(moments: np.ndarray) -> np.ndarray:
    """Match LeafOnlyInputs.compute global_features layout after moments write."""
    gf = np.zeros(12, dtype=np.float32)
    gf[0] = moments[10]
    gf[1] = 0.0
    gf[2] = moments[0]
    gf[3] = moments[1]
    gf[4] = moments[8]
    gf[5] = moments[9]
    gf[6] = moments[2]
    gf[7] = moments[3]
    gf[8] = moments[4]
    gf[9] = moments[5]
    gf[10] = moments[6]
    gf[11] = moments[7]
    return gf


def _build_x_leaf_features_np(
    num_nodes: int,
    n_pad: int,
    n_take: int,
    positions: np.ndarray,
    layer: np.ndarray,
    mass: np.ndarray,
    diffusion_grad: np.ndarray,
    diag: np.ndarray,
    moments: np.ndarray,
    scale_a: float,
    zero_diffusion: bool,
) -> np.ndarray:
    """Row-major (n_pad * 9,) matching LeafOnly_BuildX9 (same normalization as dataset)."""
    mm, ms = float(moments[0]), float(moments[1])
    d0m, d0s = float(moments[2]), float(moments[3])
    d1m, d1s = float(moments[4]), float(moments[5])
    d2m, d2s = float(moments[6]), float(moments[7])
    s = scale_a if math.isfinite(scale_a) and scale_a > 1e-30 else 1.0

    x = np.zeros((n_pad, 9), dtype=np.float32)
    for i in range(n_take):
        pos = positions[i].astype(np.float64) / 1024.0
        layer_n = float(layer[i]) / 4.0
        m = float(mass[i])
        m_sym = float(np.sign(m)) * math.log1p(abs(m))
        m_n = (m_sym - mm) / ms

        g = diffusion_grad[i].astype(np.float64)
        d_sym = np.sign(g) * np.log1p(np.abs(g))
        d_n = (d_sym - np.array([d0m, d1m, d2m], dtype=np.float64)) / np.array(
            [d0s, d1s, d2s], dtype=np.float64
        )

        diag_n = float(diag[i]) / s
        x[i, 0] = np.float32(pos[0])
        x[i, 1] = np.float32(pos[1])
        x[i, 2] = np.float32(pos[2])
        x[i, 3] = np.float32(layer_n)
        x[i, 4] = np.float32(m_n)
        x[i, 5] = np.float32(d_n[0])
        x[i, 6] = np.float32(d_n[1])
        x[i, 7] = np.float32(d_n[2])
        x[i, 8] = np.float32(diag_n)

    if zero_diffusion:
        x[:, 5:8] = 0.0
    return x.reshape(-1)


def _parity_summarize_phase(phase: str, name: str, data: np.ndarray) -> None:
    x = np.asarray(data, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n == 0:
        print(f"[LeafOnlyParity] phase={phase} tensor={name} n=0 (empty)")
        return
    xmin = float(x.min())
    xmax = float(x.max())
    mean = float(x.mean())
    sum_abs = float(np.abs(x).sum())
    l2 = float(np.sqrt((x * x).sum()))
    print(
        f"[LeafOnlyParity] phase={phase} tensor={name} n={n} min={xmin:.9g} max={xmax:.9g} "
        f"mean={mean:.9g} sum_abs={sum_abs:.9g} l2={l2:.9g}"
    )


def _parity_head_phase(phase: str, name: str, data: np.ndarray, k: int = 16) -> None:
    x = np.asarray(data, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n == 0:
        print(f"[LeafOnlyParity] phase={phase} tensor={name}_head[0]=")
        return
    c = min(k, n)
    parts = ",".join(f"{float(x[i]):.9g}" for i in range(c))
    print(f"[LeafOnlyParity] phase={phase} tensor={name}_head[{c}]={parts}")


def _parity_head_phase_at(phase: str, name: str, data: np.ndarray, start: int, k: int) -> None:
    x = np.asarray(data, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n == 0 or start < 0 or start >= n:
        print(f"[LeafOnlyParity] phase={phase} tensor={name}_at{start}_head[0]=")
        return
    c = min(int(k), n - start)
    parts = ",".join(f"{float(x[start + i]):.9g}" for i in range(c))
    print(f"[LeafOnlyParity] phase={phase} tensor={name}_at{start}_head[{c}]={parts}")


def _print_off_edge_feats_parity_lines(phase: str, flat: np.ndarray) -> None:
    from leafonly.hmatrix import HM_R0_CPU

    x = np.asarray(flat, dtype=np.float64).reshape(-1)
    n = int(x.size)
    m = int(HM_R0_CPU.shape[0])
    print(
        f"[LeafOnlyParity] phase={phase} tensor=off_edge_feats_meta M_off={m} "
        f"L={int(LEAF_SIZE)} n={n}"
    )
    _parity_summarize_phase(phase, "off_edge_feats", x)
    _parity_head_phase(phase, "off_edge_feats", x, 32)
    mid = (n // 2 // 4) * 4
    _parity_head_phase_at(phase, "off_edge_feats", x, mid, 48)
    _parity_head_phase_at(phase, "off_edge_feats", x, max(0, n - 32), 32)


def print_off_edge_feats_parity_from_x_flat(
    *,
    phase: str,
    x_flat: np.ndarray,
    n_pad: int,
    rows: np.ndarray,
    cols: np.ndarray,
    csr_scaled: np.ndarray,
    num_nodes_real: int,
) -> None:
    """CPU reference matching Unity off shader (positions = x[:, :3], CSR ÷ scale_A)."""
    import torch

    from leafonly.data import build_hmatrix_off_dense_rpe_from_positions
    from leafonly.hmatrix import HM_C0_CPU, HM_R0_CPU, HM_S_CPU

    k_leaves = n_pad // int(LEAF_SIZE)
    n_edge_active = min(int(num_nodes_real), int(n_pad))
    em = (rows < n_edge_active) & (cols < n_edge_active)
    r_e = rows[em]
    c_e = cols[em]
    v_e = csr_scaled[em]
    if r_e.size == 0:
        ei = torch.zeros((2, 0), dtype=torch.long)
        ev = torch.zeros(0, dtype=torch.float32)
    else:
        ei = torch.from_numpy(np.stack([r_e.astype(np.int64), c_e.astype(np.int64)], axis=0)).long()
        ev = torch.from_numpy(v_e.astype(np.float32)).float()
    x2 = np.asarray(x_flat, dtype=np.float32).reshape(n_pad, 9)
    pos = torch.from_numpy(np.ascontiguousarray(x2[:, :3])).float()
    ef = build_hmatrix_off_dense_rpe_from_positions(
        pos,
        HM_R0_CPU,
        HM_C0_CPU,
        HM_S_CPU,
        int(k_leaves),
        int(LEAF_SIZE),
        torch.float32,
        edge_index=ei,
        edge_values=ev,
        with_block_key_column=False,
    )
    flat = ef.reshape(-1).detach().float().cpu().numpy().astype(np.float64)
    _print_off_edge_feats_parity_lines(phase, flat)


def print_hmatrix_static_parity_lines(phase: str) -> None:
    """Same static off-tile triples as ``leafonly.hmatrix`` / Unity ``LeafOnlyHMatrixStatic``."""
    from leafonly.hmatrix import HM_C0_CPU, HM_R0_CPU, HM_S_CPU

    m = int(HM_R0_CPU.shape[0])
    r0 = HM_R0_CPU.numpy().astype(np.int64)
    c0 = HM_C0_CPU.numpy().astype(np.int64)
    s = HM_S_CPU.numpy().astype(np.int64)
    flat = np.stack([r0, c0, s], axis=1).reshape(-1).astype(np.float64)
    eta_s = f"{float(HMATRIX_ETA):.9g}"
    print(
        f"[LeafOnlyParity] phase={phase} tensor=hmatrix_meta M_off={m} "
        f"MAX_NUM_LEAVES={MAX_NUM_LEAVES} ETA={eta_s}"
    )
    _parity_summarize_phase(phase, "hmatrix_off_r0_c0_s_flat", flat)
    _parity_head_phase(phase, "hmatrix_off_r0_c0_s_flat", flat, 48)


def print_unity_style_input_parity_lines(
    *,
    phase: str,
    num_nodes_real: int,
    n_pad: int,
    n_take: int,
    align_label: str,
    rows: np.ndarray,
    cols: np.ndarray,
    vals_raw: np.ndarray,
    frame_path: Path,
    match_unity_shader: bool,
) -> None:
    """Emit the same [LeafOnlyParity] tensor lines as FluidLeafOnlyInputs.LogLeafOnlyParityToConsole."""
    n_floor = (min(num_nodes_real, int(MAX_MIXED_SIZE)) // int(LEAF_SIZE)) * int(LEAF_SIZE)
    print(
        f"[LeafOnlyParity] phase={phase} num_nodes_real={num_nodes_real} align={align_label} n_pad={n_pad} "
        f"n_take={n_take} num_leaves={n_pad // int(LEAF_SIZE)} problem_padded_floor={n_floor} "
        f"MAX_MIXED_SIZE={MAX_MIXED_SIZE}"
    )

    scale_a, csr_scaled = _scale_a_and_csr_scaled(rows, cols, vals_raw, num_nodes_real)
    nnz = int(vals_raw.shape[0])

    raw_nodes = np.fromfile(Path(frame_path) / "nodes.bin", dtype=NODE_DTYPE)[:num_nodes_real]
    positions = np.asarray(raw_nodes["position"], dtype=np.float32)
    layer = np.asarray(raw_nodes["layer"], dtype=np.float32)
    mass = np.asarray(raw_nodes["mass"], dtype=np.float32)
    dg_path = Path(frame_path) / "diffusion_gradient.bin"
    if dg_path.exists():
        diffusion_full = np.fromfile(dg_path, dtype=np.float32).reshape(num_nodes_real, 3)
    else:
        diffusion_full = np.zeros((num_nodes_real, 3), dtype=np.float32)

    diag = np.zeros(num_nodes_real, dtype=np.float32)
    for r, c, v in zip(rows, cols, vals_raw):
        if int(r) == int(c):
            diag[int(r)] = np.float32(v)

    diffusion_for_moments = (
        np.zeros((num_nodes_real, 3), dtype=np.float32) if match_unity_shader else diffusion_full
    )
    moments = _unity_leaf_moments_np(num_nodes_real, mass, diffusion_for_moments, diag, scale_a)
    gf = _global_features_from_moments(moments)

    x_flat = _build_x_leaf_features_np(
        num_nodes_real,
        n_pad,
        n_take,
        positions,
        layer,
        mass,
        diffusion_full,
        diag,
        moments,
        scale_a,
        zero_diffusion=match_unity_shader,
    )

    _parity_summarize_phase(phase, "x_leaf", x_flat)
    _parity_head_phase(phase, "x_leaf", x_flat, 16)
    _parity_summarize_phase(phase, "global_features", gf)
    _parity_head_phase(phase, "global_features", gf, 12)
    _parity_summarize_phase(phase, "scale_A", np.array([scale_a], dtype=np.float32))
    _parity_summarize_phase(phase, "leaf_moments", moments)
    _parity_head_phase(phase, "leaf_moments", moments, 16)

    if nnz > 0:
        _parity_summarize_phase(phase, "csr_values_scaled", csr_scaled)
        _parity_head_phase(phase, "csr_values_scaled", csr_scaled, 16)
    else:
        print(f"[LeafOnlyParity] phase={phase} tensor=csr_values_scaled n=0 (empty)")

    print(f"[LeafOnlyParity] phase={phase} tensor=meta nnz={nnz} (CSR nonzeros after build)")

    print_off_edge_feats_parity_from_x_flat(
        phase=phase,
        x_flat=x_flat,
        n_pad=n_pad,
        rows=rows,
        cols=cols,
        csr_scaled=csr_scaled,
        num_nodes_real=num_nodes_real,
    )

    n_edge_active = min(num_nodes_real, n_pad)
    em = (rows < n_edge_active) & (cols < n_edge_active)
    r2 = rows[em]
    c2 = cols[em]
    v2 = csr_scaled[em]
    compact_nnz = int(r2.shape[0])
    print(f"[LeafOnlyParity] phase={phase} tensor=gcn_meta compact_nnz={compact_nnz} n_edge_active={n_edge_active}")
    if compact_nnz > 0:
        head_e = min(16, compact_nnz)
        parts = ";".join(
            f"({int(r2[i])},{int(c2[i])},{float(v2[i]):.9g})" for i in range(head_e)
        )
        print(f"[LeafOnlyParity] phase={phase} tensor=gcn_edges_head[{head_e}]={parts}")

    print_hmatrix_static_parity_lines(phase)


def main() -> None:
    p = argparse.ArgumentParser(description="Print LeafOnly PyTorch parity stats for Unity comparison.")
    p.add_argument(
        "--data_folder",
        type=Path,
        default=None,
        help="Folder containing recorded runs (rglob nodes.bin). Default: ../StreamingAssets/TestData",
    )
    p.add_argument("--frame", type=int, default=0, help="Dataset index (sorted frame paths).")
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=f"Checkpoint .bytes (default: {_DEFAULT_WEIGHTS_PATH})",
    )
    p.add_argument(
        "--match_unity_shader",
        action="store_true",
        help="Zero x[:,5:8] and compute moments with zero diffusion (compare to Unity with cleared gradients).",
    )
    p.add_argument(
        "--align",
        choices=("floor", "ceil"),
        default="ceil",
        help="Leaf padding for PyTorch N: ceil matches Unity buffer leaf count when N%%128!=0; "
        "floor matches InspectModel problem_padded_num_nodes (truncate).",
    )
    p.add_argument(
        "--r",
        choices=("ones", "random"),
        default="ones",
        help="Test vector for apply_block_diagonal_M_physical.",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed when --r random.")
    p.add_argument("--no_apply", action="store_true", help="Skip M@r apply summaries.")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (may hide bugs).")
    p.add_argument(
        "--print_unity_inputs",
        action="store_true",
        help="Print [LeafOnlyParity] lines matching Unity (x_leaf, global_features, scale_A, csr, gcn, …).",
    )
    p.add_argument(
        "--inputs_parity_only",
        action="store_true",
        help="Load only the dataset frame and print --print_unity_inputs (no checkpoint or model).",
    )
    args = p.parse_args()
    if args.inputs_parity_only:
        args.print_unity_inputs = True

    data_folder = args.data_folder
    if data_folder is None:
        data_folder = _script_dir.parent / "StreamingAssets" / "TestData"
    data_folder = Path(data_folder).resolve()
    weights_path = Path(args.weights).resolve() if args.weights else _DEFAULT_WEIGHTS_PATH.resolve()

    dataset = FluidGraphDataset([data_folder])
    if len(dataset) == 0:
        raise SystemExit(f"No frames under {data_folder}")
    fi = max(0, min(int(args.frame), len(dataset) - 1))
    batch = dataset[fi]
    frame_path = batch.get("frame_path", "")
    num_nodes_real = int(batch["num_nodes"])
    if args.align == "floor":
        n_pad = int(problem_padded_num_nodes(num_nodes_real))
    else:
        n_pad = _leaf_align_n(num_nodes_real, "ceil")
    n_pad = min(n_pad, int(MAX_MIXED_SIZE))
    n_inspect_floor = int(problem_padded_num_nodes(num_nodes_real))
    n_take = min(num_nodes_real, n_pad)

    print(f"[LeafOnlyParity] phase=pytorch frame_index={fi} frame_path={frame_path}")

    if not args.print_unity_inputs:
        print_hmatrix_static_parity_lines("pytorch")

    if args.print_unity_inputs:
        rows_u, cols_u, vals_u = _read_raw_coo(Path(frame_path))
        print_unity_style_input_parity_lines(
            phase="pytorch",
            num_nodes_real=num_nodes_real,
            n_pad=n_pad,
            n_take=n_take,
            align_label=args.align,
            rows=rows_u,
            cols=cols_u,
            vals_raw=vals_u,
            frame_path=Path(frame_path),
            match_unity_shader=args.match_unity_shader,
        )

    if args.inputs_parity_only:
        return

    if not weights_path.is_file():
        raise SystemExit(f"Weights not found: {weights_path}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt = leaf_only_arch_from_checkpoint(weights_path)
    if ckpt is None:
        raise SystemExit(f"Bad checkpoint header: {weights_path}")
    if int(ckpt["leaf_size"]) != int(LEAF_SIZE):
        raise SystemExit(f"Checkpoint leaf_size {ckpt['leaf_size']} != LEAF_SIZE {LEAF_SIZE}")

    ck_hw = int(ckpt.get("highway_ffn_mlp", 0))
    cli_hw = bool(_leaf_only_script.DEFAULT_USE_HIGHWAYS)
    if ck_hw == 1 and not cli_hw:
        raise SystemExit(_leaf_only_script.CHECKPOINT_ERR_HIGHWAY_IN_FILE_NEED_CLI_ON)
    if ck_hw == 0 and cli_hw:
        raise SystemExit(_leaf_only_script.CHECKPOINT_ERR_NO_HIGHWAY_IN_FILE_NEED_CLI_OFF)
    ck_ffn = int(ckpt.get("ffn_concat_width", 3 if ck_hw else 1))

    print(f"[LeafOnlyParity] phase=pytorch device={device} match_unity_shader={args.match_unity_shader}")
    if not args.print_unity_inputs:
        print(
            f"[LeafOnlyParity] phase=pytorch num_nodes_real={num_nodes_real} align={args.align} n_pad={n_pad} "
            f"num_leaves={n_pad // int(LEAF_SIZE)} problem_padded_floor={n_inspect_floor} MAX_MIXED_SIZE={MAX_MIXED_SIZE}"
        )

    x = batch["x"].unsqueeze(0).to(device).clone()
    ei, ev = batch["edge_index"], batch["edge_values"]
    n_edge_active = min(num_nodes_real, n_pad)
    em = (ei[0] < n_edge_active) & (ei[1] < n_edge_active)
    edge_index_leaf = ei[:, em].to(device)
    edge_values_leaf = ev[em].to(device)

    if args.match_unity_shader:
        x[:, :, 5:8] = 0.0
        rows = ei[0].numpy()
        cols = ei[1].numpy()
        vals = ev.numpy()
        em_np = em.numpy()
        rows = rows[em_np]
        cols = cols[em_np]
        vals = vals[em_np]
        mass = np.fromfile(Path(frame_path) / "nodes.bin", dtype=NODE_DTYPE)["mass"][:num_nodes_real]
        stats_n = min(num_nodes_real, n_pad)
        stats_u, gf_u = _unity_stats_and_global(stats_n, mass[:stats_n], rows, cols, vals)
        gf_t = torch.from_numpy(gf_u).to(device=device, dtype=x.dtype)
        _summarize("statsBuffer_like", torch.from_numpy(stats_u))
        _head("statsBuffer_like", torch.from_numpy(stats_u), 8)
        _summarize("globalFeatBuf_like", gf_t)
        _head("globalFeatBuf_like", gf_t, 12)
        global_features = gf_t.unsqueeze(0)
    else:
        global_features = batch["global_features"].to(device)
        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)  # (1, 12)

    x_leaf = x[:, :n_take, :].clone()
    if n_pad > n_take:
        x_leaf = F.pad(x_leaf, (0, 0, 0, n_pad - n_take), value=0.0)

    _, ef_diag, _, ef_off = build_leaf_block_connectivity(
        edge_index_leaf,
        edge_values_leaf,
        x_leaf[0, :, :3],
        int(LEAF_SIZE),
        device,
        x_leaf.dtype,
        off_diag_dense_attention=True,
        diag_dense_attention=True,
    )
    _summarize("diag_edge_feats", ef_diag.reshape(-1))
    _head("diag_edge_feats", ef_diag.reshape(-1), 32)

    # ``--print_unity_inputs`` already printed off_edge_feats from CPU x_flat; skip duplicate block.
    if not args.print_unity_inputs:
        off_flat = ef_off.reshape(-1).detach().float().cpu().numpy().astype(np.float64)
        _print_off_edge_feats_parity_lines("pytorch", off_flat)

    model = LeafOnlyNet(
        input_dim=int(ckpt["input_dim"]),
        d_model=int(ckpt["d_model"]),
        leaf_size=int(ckpt["leaf_size"]),
        num_layers=int(ckpt["num_layers"]),
        num_heads=int(ckpt["num_heads"]),
        use_gcn=bool(int(ckpt["use_gcn"])),
        attention_layout=default_attention_layout(int(ckpt["leaf_size"])),
        off_diag_dense_attention=True,
        diag_dense_attention=True,
        use_highways=cli_hw,
        ffn_concat_width=int(ck_ffn) if cli_hw else None,
    ).to(device)
    load_leaf_only_weights(model, weights_path)
    model.eval()
    if args.compile:
        model = torch.compile(model)

    with torch.inference_mode():
        packed = model(
            x_leaf,
            edge_index_leaf,
            edge_values_leaf,
            global_features=global_features,
        )

    _summarize("packed_precond", packed)
    _head("packed_precond", packed, 16)

    La_d = int(model.leaf_apply_size)
    La_o = int(model.leaf_apply_off)
    num_leaves = n_pad // int(LEAF_SIZE)
    diag_b, off_b, node_u, node_v, jac = unpack_precond(
        packed, n_pad, leaf_size=LEAF_SIZE, leaf_apply_size=La_d, leaf_apply_off=La_o
    )
    _summarize("precondDiag", diag_b)
    _head("precondDiag", diag_b, 16)
    if off_b is not None and off_b.numel() > 0:
        _summarize("precondOff", off_b)
        _head("precondOff", off_b, 16)
    _summarize("precondU", node_u)
    _head("precondU", node_u, 16)
    _summarize("precondV", node_v)
    _head("precondV", node_v, 16)
    if jac is not None:
        _summarize("precondJac", jac)
        _head("precondJac", jac, 16)

    # Token embedding path (after enc_input_proj in PyTorch = after NormAndProject in Unity shader).
    h0_ref = None
    off0_ref = None
    layer0_err: str | None = None
    with torch.inference_mode():
        h0 = model.embed(x_leaf, edge_index_leaf, edge_values_leaf, global_features=global_features)
        h_enc = model.enc_input_proj(h0)
        if int(ckpt["num_layers"]) >= 1 and not cli_hw:
            try:
                h0_ref, off0_ref = _layer0_transformer_reference_tensors(
                    model,
                    h_enc,
                    ef_diag,
                    ef_off,
                    edge_index_leaf,
                    edge_values_leaf,
                    x_leaf[0, :, :3],
                    eager_attention=True,
                )
            except Exception as ex:
                layer0_err = f"{type(ex).__name__}: {ex}"

    _summarize("token_after_enc", h_enc)
    _head("token_after_enc", h_enc, 16)
    _head("token_after_enc", h_enc, 32)

    if layer0_err is not None:
        print(f"[LeafOnlyParity] phase=pytorch tensor=h_diag_after_layer0 skipped ({layer0_err})")
    elif h0_ref is not None:
        flat_d = h0_ref.reshape(-1)
        _summarize("h_diag_after_layer0", flat_d)
        _head("h_diag_after_layer0", flat_d, 16)
        _head("h_diag_after_layer0", flat_d, 32)
        n_h = int(flat_d.numel())
        if n_h > 64:
            _parity_head_at("h_diag_after_layer0", flat_d, n_h // 2, 48)
        if off0_ref is not None:
            flat_o = off0_ref.reshape(-1)
            _summarize("off_stream_after_layer0", flat_o)
            _head("off_stream_after_layer0", flat_o, 16)
            _head("off_stream_after_layer0", flat_o, 32)
            n_o = int(flat_o.numel())
            if n_o > 64:
                _parity_head_at("off_stream_after_layer0", flat_o, n_o // 2, 48)
        print(
            f"[LeafOnlyParity] phase=pytorch tensor=layer1_ref_meta eager_attention=1 "
            f"K={n_pad // int(LEAF_SIZE)} L_diag={int(model.leaf_apply_size)} "
            f"L_off={int(model.leaf_apply_off)} M_off={int(model.num_h_off)}"
        )

    if args.no_apply:
        return

    A = torch.zeros(n_pad, n_pad, device=device, dtype=x.dtype)
    A[edge_index_leaf[0], edge_index_leaf[1]] = edge_values_leaf
    diag_a = torch.diagonal(A)
    jacobi_inv = torch.zeros(1, n_pad, device=device, dtype=x.dtype)
    jacobi_inv[0, :n_edge_active] = torch.where(
        diag_a[:n_edge_active].abs() > 1e-12,
        1.0 / diag_a[:n_edge_active],
        torch.zeros_like(diag_a[:n_edge_active]),
    )

    if args.r == "ones":
        r = torch.ones(1, n_pad, 1, device=device, dtype=x.dtype)
    else:
        g = torch.Generator(device=device)
        g.manual_seed(int(args.seed))
        r = torch.randn(1, n_pad, 1, device=device, dtype=x.dtype, generator=g)

    _summarize("apply_r", r)
    _head("apply_r", r, 16)

    with torch.inference_mode():
        z = apply_block_diagonal_M_physical(
            packed,
            r,
            leaf_size=LEAF_SIZE,
            leaf_apply_size=La_d,
            leaf_apply_off=La_o,
            jacobi_inv_diag=jacobi_inv,
        )
    _summarize("apply_z", z)
    _head("apply_z", z, 16)


if __name__ == "__main__":
    main()
