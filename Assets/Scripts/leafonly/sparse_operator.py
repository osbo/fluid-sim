"""Padded graph Laplacian as sparse COO + block-diagonal batched apply (CUDA / MPS / CPU)."""

import os

import torch

# Max bytes for B separate dense (N×N) factors on MPS; above this, batched A@Z uses CPU sparse.mm.
_MPS_AZ_DENSE_CAP_BYTES = int(os.environ.get("LEAFONLY_MPS_AZ_DENSE_CAP_BYTES", str(512 * 1024 * 1024)))


def padded_operator_sparse(edge_index, edge_values, n_phys: int, n_pad: int, dtype, device):
    """Physical Laplacian on [:n_phys,:n_phys] plus identity on the padded tail (matches dense A layout)."""
    rows, cols = edge_index[0].long(), edge_index[1].long()
    v = edge_values.to(dtype=dtype)
    tail = torch.arange(n_phys, n_pad, device=device, dtype=torch.long)
    tail_ones = torch.ones(n_pad - n_phys, device=device, dtype=dtype)
    ri = torch.cat([rows, tail])
    ci = torch.cat([cols, tail])
    vv = torch.cat([v, tail_ones])
    return torch.sparse_coo_tensor(torch.stack([ri, ci]), vv, (n_pad, n_pad), device=device, dtype=dtype).coalesce()


def diag_from_sparse_coo(indices: torch.Tensor, values: torch.Tensor, n_pad: int, dtype, device) -> torch.Tensor:
    diag = torch.zeros(n_pad, dtype=dtype, device=device)
    r, c = indices[0], indices[1]
    m = r == c
    if m.any():
        diag.index_add_(0, r[m], values[m].to(dtype=dtype))
    return diag


def extend_sparse_identity_tail(sp: torch.Tensor, n_src: int, n_dst: int) -> torch.Tensor:
    """Grow a padded operator from n_src×n_src to n_dst×n_dst with identity on new diagonal tail."""
    sp = sp.coalesce()
    if n_dst == n_src:
        return sp
    idx, val = sp.indices(), sp.values()
    tail = torch.arange(n_src, n_dst, device=sp.device, dtype=idx.dtype)
    extra = torch.stack([tail, tail])
    ones = torch.ones(n_dst - n_src, device=sp.device, dtype=val.dtype)
    new_i = torch.cat([idx, extra], dim=1)
    new_v = torch.cat([val, ones])
    return torch.sparse_coo_tensor(new_i, new_v, (n_dst, n_dst), dtype=val.dtype, device=sp.device).coalesce()


def batched_A_mul_Z_blockdiag(sp_list: list, Z: torch.Tensor) -> torch.Tensor:
    """Apply diag(A_1,…,A_B) to Z via one sparse.mm (avoids dense N² batched bmm).

    PyTorch does not implement ``sparse.mm`` for SparseMPS (NotImplementedError on ``aten::addmm``).
    On MPS we run the block-diagonal ``sparse.mm`` on CPU and move the result back.
    """
    B, N, K = Z.shape
    device, dtype = Z.device, Z.dtype
    rows_c, cols_c, vals_c = [], [], []
    for b in range(B):
        sp = sp_list[b]
        if sp.device != device:
            sp = sp.to(device=device)
        if sp.dtype != dtype:
            sp = torch.sparse_coo_tensor(
                sp.indices(), sp.values().to(dtype=dtype), sp.shape, device=device, dtype=dtype
            ).coalesce()
        else:
            sp = sp.coalesce()
        idx = sp.indices()
        rows_c.append(idx[0] + b * N)
        cols_c.append(idx[1] + b * N)
        vals_c.append(sp.values())
    rows = torch.cat(rows_c)
    cols = torch.cat(cols_c)
    vals = torch.cat(vals_c)
    z_flat = Z.reshape(B * N, K)
    if device.type == "mps":
        cpu = torch.device("cpu")
        a_big = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), vals, (B * N, B * N), device=cpu, dtype=dtype
        ).coalesce()
        out = torch.sparse.mm(a_big, z_flat.cpu())
        return out.to(device=device, dtype=dtype).view(B, N, K)
    a_big = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, (B * N, B * N), device=device, dtype=dtype
    ).coalesce()
    out = torch.sparse.mm(a_big, z_flat)
    return out.view(B, N, K)


def sparse_A_mul_Z(A_sp: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """A_sp (N×N) @ Z (B,N,K) → (B,N,K); B must be 1.

    SparseMPS has no ``sparse.mm``; densifies **each call** — for Jacobi / many matvecs use ``make_sparse_az_fn``.
    """
    B, N, K = Z.shape
    if B != 1:
        raise ValueError(f"sparse_A_mul_Z expects B=1, got B={B}")
    if A_sp.device.type == "mps":
        return (A_sp.to_dense() @ Z.squeeze(0)).unsqueeze(0)
    return torch.sparse.mm(A_sp, Z.squeeze(0)).unsqueeze(0)


def make_sparse_az_fn(A_sp: torch.Tensor):
    """Return ``f(Zt)`` with ``Zt`` shape ``(1,N,K)`` → ``A @ Zt``. On MPS, densifies ``A`` once (hot Jacobi path)."""
    if A_sp.device.type == "mps":
        A_dense = A_sp.to_dense()

        def f(Zt: torch.Tensor) -> torch.Tensor:
            return (A_dense @ Zt.squeeze(0)).unsqueeze(0)

        return f

    def f(Zt: torch.Tensor) -> torch.Tensor:
        return sparse_A_mul_Z(A_sp, Zt)

    return f


def make_batched_sparse_az_fn(
    sp_list: list,
    *,
    B: int,
    N: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """
    Return ``f(Zt)`` for ``Zt`` shape ``(B,N,K)`` → block-diagonal ``A @ Zt``.

    On MPS, if ``B*N*N`` dense storage is below ``LEAFONLY_MPS_AZ_DENSE_CAP_BYTES`` (default 512 MiB),
    densifies each ``A_b`` once per outer training step (cached across Jacobi matvecs). Otherwise uses
    CPU ``sparse.mm`` on the block-diagonal COO (slower but avoids OOM).
    """
    elem = torch.empty((), dtype=dtype, device=device).element_size()
    dense_bytes = B * N * N * elem
    if device.type == "mps" and dense_bytes <= _MPS_AZ_DENSE_CAP_BYTES:
        cache: list = [None]

        def f(Zt: torch.Tensor) -> torch.Tensor:
            if cache[0] is None:
                cache[0] = [sp_list[b].to_dense() for b in range(B)]
            dlist = cache[0]
            return torch.stack([dlist[b] @ Zt[b] for b in range(B)], dim=0)

        return f

    def f(Zt: torch.Tensor) -> torch.Tensor:
        return batched_A_mul_Z_blockdiag(sp_list, Zt)

    return f
