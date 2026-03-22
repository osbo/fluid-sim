"""Inspect LeafOnly preconditioner: load weights, build M, compare A·M with AMG. Run with python3 InspectModel.py."""
import sys
import time
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Sparse CSR.*", category=UserWarning)

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.sparse import csr_matrix

try:
    import pyamg
    HAS_AMG = True
except ImportError:
    HAS_AMG = False
    print("Warning: 'pyamg' not installed. AMG baseline will be skipped.")

try:
    import pyamgx
    HAS_AMGX = True
except (ImportError, OSError) as e:
    HAS_AMGX = False
    print("Warning: 'pyamgx' not available. AMGX (GPU) comparison will be skipped.", e)
import torch

import torch.nn.functional as F

from leafonly import (
    LeafOnlyNet,
    load_leaf_only_weights,
    apply_block_diagonal_M,
    default_attention_layout,
    unpack_precond,
    next_valid_size,
    FluidGraphDataset,
    build_leaf_block_connectivity,
    pool_precomputed_leaf_connectivity,
)
from leafonly.eval import _timed_ms
from leafonly.checkpoint import read_leaf_only_header
from leafonly.config import ATTN_POOL_FACTOR, LEAF_SIZE

# Saturated bases for legend (match _leaf_block_partition_diagram)
_LEAF_DIAG_BASE = np.array([0.20, 0.35, 0.75], dtype=np.float64)
_LEAF_OFF_BASE = np.array([0.90, 0.40, 0.12], dtype=np.float64)


def _leaf_block_partition_diagram(n: int, leaf_size: int) -> np.ndarray:
    """
    (n, n, 3) RGB in [0, 1]: each L×L leaf block is a dark border band (scaled base) inside the
    true cell boundary, then a solid lighter interior (base blended toward white). Distance is
    measured in pixels from the nearest block edge (Chebyshev-style min distance to a side).
    """
    L = int(leaf_size)
    if n % L != 0:
        raise ValueError(f"n={n} must be divisible by leaf_size={L}")
    ii = np.arange(n, dtype=np.float64)[:, None]
    jj = np.arange(n, dtype=np.float64)[None, :]
    bi = (ii // L).astype(np.int64)
    bj = (jj // L).astype(np.int64)
    on_diag = bi == bj

    li, lj = ii % L, jj % L
    dist_to_edge = np.minimum(np.minimum(li, L - 1 - li), np.minimum(lj, L - 1 - lj))
    # Band width ~10% of L, at least 1 px; shrink if L is tiny so an interior can remain.
    border_w = min(max(1, int(round(0.10 * L))), max(1, (L - 1) // 2))
    in_border = dist_to_edge < float(border_w)

    white = np.ones(3, dtype=np.float64)
    diag_dark = np.clip(_LEAF_DIAG_BASE * 0.48, 0.0, 1.0)
    diag_light = np.clip(0.40 * _LEAF_DIAG_BASE + 0.60 * white, 0.0, 1.0)
    off_dark = np.clip(_LEAF_OFF_BASE * 0.48, 0.0, 1.0)
    off_light = np.clip(0.40 * _LEAF_OFF_BASE + 0.60 * white, 0.0, 1.0)

    sel_dark = np.where(on_diag[:, :, np.newaxis], diag_dark, off_dark)
    sel_light = np.where(on_diag[:, :, np.newaxis], diag_light, off_light)
    rgb = np.where(in_border[:, :, np.newaxis], sel_dark, sel_light)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _amg_solver(A_sparse_scipy):
    if not HAS_AMG:
        return None
    dtype = np.float64
    if A_sparse_scipy.dtype != dtype:
        A_sparse_scipy = A_sparse_scipy.astype(dtype)
    return pyamg.smoothed_aggregation_solver(A_sparse_scipy)


def get_dense_amg(A_sparse_scipy, viz_limit=200, maxiter=1, tol=1e-6, progress_interval=200, ml=None):
    if not HAS_AMG:
        n = min(A_sparse_scipy.shape[0], viz_limit)
        return np.eye(n)
    dtype = np.float64
    if A_sparse_scipy.dtype != dtype:
        A_sparse_scipy = A_sparse_scipy.astype(dtype)
    if ml is None:
        print("\nComputing AMG baseline (building hierarchy + M for plot)...")
        ml = pyamg.smoothed_aggregation_solver(A_sparse_scipy)
    else:
        print("\nBuilding dense M for plot (column solves)...")
    limit = min(A_sparse_scipy.shape[0], viz_limit)
    M_dense = np.zeros((limit, limit), dtype=dtype)
    for i in range(limit):
        e_i = np.zeros(A_sparse_scipy.shape[0], dtype=dtype)
        e_i[i] = 1.0
        x0 = np.zeros(A_sparse_scipy.shape[0], dtype=dtype)
        M_dense[:, i] = ml.solve(e_i, x0=x0, maxiter=maxiter, cycle='V', tol=tol)[:limit]
        if progress_interval and (i + 1) % progress_interval == 0:
            print(f"  AMG column {i + 1}/{limit}")
    return M_dense


def pcg_gpu(A, b, apply_precond, tol=1e-8, max_iter=500, device=None, debug=False, check_freq=3):
    if device is None:
        device = b.device

    n = b.shape[0]
    x = torch.zeros(n, 1, device=device, dtype=b.dtype)

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
    else:
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()

    r = b - (A @ x.squeeze(-1)).unsqueeze(-1)
    z = apply_precond(r)
    p = z.clone()

    rho = (r * z).sum()
    b_norm_sq = (b * b).sum()
    tol_sq_val = (tol * tol) * b_norm_sq.item()

    iters = 0
    if tol_sq_val > 0 or b_norm_sq.item() > 0:
        for k in range(max_iter):
            Ap = (A @ p.squeeze(-1)).unsqueeze(-1)
            pAp = (p * Ap).sum()

            alpha = rho / pAp
            x = x + alpha * p
            r = r - alpha * Ap

            z = apply_precond(r)
            rho_new = (r * z).sum()

            beta = rho_new / rho
            p = z + beta * p
            rho = rho_new

            if (k + 1) % check_freq == 0 or k == max_iter - 1:
                r_sq = (r * r).sum()
                if r_sq.item() <= tol_sq_val:
                    if debug:
                        print(f"    [PCG GPU] converged at iter {k+1}")
                    iters = k + 1
                    break
        if iters == 0:
            iters = max_iter

    if device.type == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        wall_ms = start_event.elapsed_time(end_event)
    else:
        if device.type == "mps":
            torch.mps.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000
    return x, iters, wall_ms


def pcg_gpu_cudagraph(
    A_gpu,
    b_gpu,
    precond_packed,
    jacobi_inv_diag,
    tol=1e-8,
    max_iter=500,
    device=None,
    check_freq=3,
    leaf_size=None,
    leaf_apply_size=None,
):
    if device is None:
        device = b_gpu.device
    if leaf_size is None:
        leaf_size = LEAF_SIZE
    if leaf_apply_size is None:
        leaf_apply_size = leaf_size

    if precond_packed.dim() == 1:
        precond_packed = precond_packed.unsqueeze(0)
    jacobi_inv_diag_2d = jacobi_inv_diag if jacobi_inv_diag.dim() == 2 else jacobi_inv_diag.unsqueeze(0)

    x_static = torch.zeros_like(b_gpu)
    r_static = torch.zeros_like(b_gpu)
    p_static = torch.zeros_like(b_gpu)
    z_static = torch.zeros_like(b_gpu)
    rho_static = torch.zeros(1, device=device, dtype=b_gpu.dtype)

    x_static.zero_()
    r_static.copy_(b_gpu)
    z_initial = apply_block_diagonal_M(
        precond_packed,
        r_static.unsqueeze(0),
        leaf_size=leaf_size,
        leaf_apply_size=leaf_apply_size,
        jacobi_inv_diag=jacobi_inv_diag_2d,
    )
    z_static.copy_(z_initial.squeeze(0))
    p_static.copy_(z_static)
    rho_static.fill_((r_static * z_static).sum())

    tol_sq_val = (tol * tol) * (b_gpu * b_gpu).sum().item()

    def one_iter():
        Ap = A_gpu @ p_static.squeeze(-1)
        Ap = Ap.unsqueeze(-1)
        pAp = (p_static * Ap).sum()
        alpha = rho_static / pAp
        x_static.add_(p_static * alpha)
        r_static.sub_(Ap * alpha)
        z_new = apply_block_diagonal_M(
            precond_packed,
            r_static.unsqueeze(0),
            leaf_size=leaf_size,
            leaf_apply_size=leaf_apply_size,
            jacobi_inv_diag=jacobi_inv_diag_2d,
        )
        z_static.copy_(z_new.squeeze(0))
        rho_new = (r_static * z_static).sum()
        beta = rho_new / rho_static
        rho_static.fill_(rho_new)
        p_static.mul_(beta).add_(z_static)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        one_iter()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_ev.record()
    iters = 0
    for k in range(max_iter):
        g.replay()
        iters = k + 1
        if (k + 1) % check_freq == 0 or k == max_iter - 1:
            r_sq = (r_static * r_static).sum()
            if r_sq.item() <= tol_sq_val:
                break
    end_ev.record()
    torch.cuda.synchronize()
    wall_ms = start_ev.elapsed_time(end_ev)
    return x_static.clone(), iters, wall_ms


def pcg_cpu(A, b, apply_precond, tol=1e-8, max_iter=500, debug=False):
    n = b.shape[0]
    x = np.zeros((n, 1), dtype=np.float64)
    r = b - A @ x
    z = apply_precond(r)
    p = z.copy()
    rho = float(np.dot(r.ravel(), z.ravel()))
    b_norm_sq = float(np.dot(b.ravel(), b.ravel()))
    b_norm = b_norm_sq ** 0.5 if b_norm_sq > 0 else 0.0
    if b_norm_sq <= 0:
        return x, 0, 0.0
    tol_sq = (tol * tol) * b_norm_sq
    r_norm_0 = float(np.dot(r.ravel(), r.ravel())) ** 0.5
    if debug:
        print(f"    [PCG CPU] init: ||b||={b_norm:.2e}, ||r0||={r_norm_0:.2e}, rho0={rho:.2e}, tol*||b||={tol*b_norm:.2e}")
    t0 = time.perf_counter()
    iters = max_iter
    for k in range(max_iter):
        Ap = A @ p
        pAp = float(np.dot(p.ravel(), Ap.ravel()))
        if pAp <= 1e-14:
            if debug:
                print(f"    [PCG CPU] iter {k}: pAp={pAp:.2e} <= 0, stopping")
            iters = k
            break
        alpha = rho / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        r_sq = float(np.dot(r.ravel(), r.ravel()))
        r_norm = r_sq ** 0.5
        if debug and (k < 5 or (k + 1) % 50 == 0 or k == max_iter - 1):
            rel = r_norm / b_norm if b_norm > 0 else float('nan')
            print(f"    [PCG CPU] iter {k+1}: ||r||={r_norm:.2e}, rel_res={rel:.2e}, alpha={alpha:.2e}")
        if r_sq <= tol_sq:
            if debug:
                print(f"    [PCG CPU] converged at iter {k+1}, ||r||={r_norm:.2e}")
            return x, k + 1, (time.perf_counter() - t0) * 1000
        z = apply_precond(r)
        rho_new = float(np.dot(r.ravel(), z.ravel()))
        beta = rho_new / rho
        p = z + beta * p
        rho = rho_new
    if debug:
        print(f"    [PCG CPU] stopped at iter {iters}, ||r||={r_norm:.2e}, rel_res={r_norm/b_norm:.2e}")
    return x, iters, (time.perf_counter() - t0) * 1000


def main():
    script_dir = _script_dir
    # Persist torch.compile / Inductor artifacts across runs (speeds repeat InspectModel invocations).
    _inductor_cache = script_dir / ".torch_inductor_cache"
    _inductor_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_inductor_cache.resolve()))
    _triton_cache = script_dir / ".triton_cache"
    _triton_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(_triton_cache.resolve()))

    data_folder = script_dir.parent / "StreamingAssets" / "TestData"
    leaf_only_weights_path = script_dir / "leaf_only_weights.bytes"
    out_path = script_dir / "inspect_model_plot.png"
    frame_idx = 600

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
    print(f"Using device: {device}")
    check_freq = 3

    print(f"Loading data from {data_folder}")
    dataset = FluidGraphDataset([Path(data_folder)])
    if len(dataset) == 0:
        raise SystemExit("No frames found (need nodes.bin, edge_index_*.bin, A_values.bin).")
    frame_idx = min(frame_idx, len(dataset) - 1)
    batch = dataset[frame_idx]

    x = batch['x'].unsqueeze(0).to(device)
    num_nodes_real = int(batch['num_nodes'])
    print(f"System N={num_nodes_real}")

    if not leaf_only_weights_path.exists():
        raise SystemExit(f"Leaf-only weights not found: {leaf_only_weights_path}. Run LeafOnly.py first.")
    (
        d_model_lo,
        leaf_size_lo,
        input_dim_lo,
        num_layers_lo,
        num_heads_lo,
        use_gcn_lo,
        _num_gcn_hdr,
        _header_bytes,
        leaf_apply_ckpt,
    ) = read_leaf_only_header(leaf_only_weights_path)
    if LEAF_SIZE != leaf_size_lo:
        print(
            f"  Note: leafonly.config.LEAF_SIZE={LEAF_SIZE} != checkpoint leaf_size={leaf_size_lo}; "
            "using checkpoint leaf size for padding and preconditioner layout."
        )
    leaf_L = int(leaf_size_lo)

    n_requested = 256
    n_pad = next_valid_size(n_requested, leaf_L)
    if n_pad != n_requested:
        print(f"  Padding view: {n_requested} nodes -> {n_pad} (power-of-2 leaf blocks of {leaf_L})")

    ei, ev = batch["edge_index"], batch["edge_values"]
    # Single mask for the n_requested subgraph — reused for A_viz, GPU edges, and connectivity.
    em = (ei[0] < n_requested) & (ei[1] < n_requested)
    A_small = torch.sparse_coo_tensor(ei[:, em], ev[em], (n_requested, n_requested)).coalesce().to_dense().numpy()
    A_viz = np.zeros((n_pad, n_pad), dtype=np.float64)
    A_viz[:n_requested, :n_requested] = A_small
    if n_pad > n_requested:
        A_viz[n_requested:, n_requested:] = np.eye(n_pad - n_requested, dtype=np.float64)
    viz_n = n_pad

    print("\nLeafOnly (GPU)...")
    model_leaf = LeafOnlyNet(
        input_dim=input_dim_lo,
        d_model=d_model_lo,
        leaf_size=leaf_size_lo,
        num_layers=num_layers_lo,
        num_heads=num_heads_lo,
        use_gcn=bool(use_gcn_lo),
        attention_layout=default_attention_layout(leaf_size_lo),
    ).to(device)
    model_leaf = torch.compile(model_leaf)
    load_leaf_only_weights(model_leaf, leaf_only_weights_path)
    model_leaf.eval()
    leaf_apply_L = int(model_leaf.leaf_apply_size)
    if leaf_apply_L != int(leaf_apply_ckpt):
        raise RuntimeError(f"header leaf_apply_size {leaf_apply_ckpt} != model.leaf_apply_size {leaf_apply_L}")
    pool_to_full = int(leaf_L) // leaf_apply_L

    with torch.inference_mode():
        x_leaf = x[:, :n_requested, :].clone()
        if n_pad > n_requested:
            x_leaf = F.pad(x_leaf, (0, 0, 0, n_pad - n_requested), value=0.0)
        edge_index_leaf = ei[:, em].to(device)
        edge_values_leaf = batch["edge_values"][em].to(device)

        global_feat = batch.get("global_features")
        if global_feat is not None:
            global_feat = global_feat.to(device)
            if global_feat.dim() == 1:
                global_feat = global_feat.unsqueeze(0)

        positions_leaf = x_leaf[0, :, :3]
        # Full graph connectivity once, then pooled masks for attention (matches training cache).
        pre_leaf_connectivity = pool_precomputed_leaf_connectivity(
            build_leaf_block_connectivity(
                edge_index_leaf,
                edge_values_leaf,
                positions_leaf,
                leaf_L,
                device,
                x_leaf.dtype,
            ),
            leaf_L,
            ATTN_POOL_FACTOR,
        )
        pre_leaf_connectivity = tuple(t.contiguous() for t in pre_leaf_connectivity)

        _last_out = [None]

        def _fwd():
            _last_out[0] = model_leaf(
                x_leaf,
                edge_index=edge_index_leaf,
                edge_values=edge_values_leaf,
                global_features=global_feat,
                precomputed_leaf_connectivity=pre_leaf_connectivity,
            )

        # Prime torch.compile / CUDA allocator; Inductor kernels also land in TORCHINDUCTOR_CACHE_DIR (see above).
        for _ in range(15):
            _fwd()
        if device.type == "cuda":
            torch.cuda.synchronize()

        inference_ms = _timed_ms(_fwd, device, warmup=0, repeat=10)
        precond_out = _last_out[0]
        diag_blocks, off_diag_blocks, jacobi_scale = unpack_precond(
            precond_out, n_pad, leaf_size=leaf_L, leaf_apply_size=leaf_apply_L
        )
        jacobi_inv_diag = torch.ones(1, n_pad, device=device, dtype=diag_blocks.dtype)
        diag_A_tensor = torch.from_numpy(np.diag(A_viz).astype(np.float32)).to(device)
        diag_mask = diag_A_tensor.abs() > 1e-6
        jacobi_inv_diag[0, diag_mask] = 1.0 / diag_A_tensor[diag_mask]
        num_leaves = n_pad // leaf_L
        M_neural_gpu = torch.zeros((n_pad, n_pad), dtype=torch.float32, device=device)
        for b in range(num_leaves):
            r0, r1 = b * leaf_L, (b + 1) * leaf_L
            blk = diag_blocks[0, b]
            if pool_to_full > 1:
                blk = blk.repeat_interleave(pool_to_full, dim=0).repeat_interleave(pool_to_full, dim=1)
            M_neural_gpu[r0:r1, r0:r1] = blk

        P = (num_leaves * (num_leaves - 1)) // 2
        if P > 0 and off_diag_blocks is not None:
            r_idx, c_idx = torch.triu_indices(num_leaves, num_leaves, offset=1, device=device)
            for p in range(P):
                r, c = r_idx[p].item(), c_idx[p].item()
                r0, r1 = r * leaf_L, (r + 1) * leaf_L
                c0, c1 = c * leaf_L, (c + 1) * leaf_L
                oblk = off_diag_blocks[0, p]
                if pool_to_full > 1:
                    oblk = oblk.repeat_interleave(pool_to_full, dim=0).repeat_interleave(pool_to_full, dim=1)
                M_neural_gpu[r0:r1, c0:c1] = oblk
                M_neural_gpu[c0:c1, r0:r1] = oblk.transpose(-1, -2)

        if jacobi_scale is not None:
            M_neural_gpu += torch.diag((jacobi_scale[0] * jacobi_inv_diag[0]).to(M_neural_gpu.dtype))
    print(
        f"  LeafOnly: {num_leaves} leaves, {n_pad}x{n_pad} M (learned blocks {leaf_apply_L}×{leaf_apply_L}, "
        f"Kronecker prolongation ×{pool_to_full} for visualization)"
    )

    d = diag_blocks.detach()
    if d.dim() == 3:
        d = d.unsqueeze(0)
    frobs = (d ** 2).sum(dim=(-2, -1)).sqrt()
    flat = d.reshape(-1)
    print(f"  Diagonal blocks: Frob mean={frobs.mean().item():.6f} std={frobs.std().item():.6f}")
    print(f"  Elements: mean={flat.mean().item():.6e} std={flat.std().item():.6e} min={flat.min().item():.6e} max={flat.max().item():.6e}")

    A_scipy = csr_matrix(A_viz.astype(np.float64))
    ml_amg = None
    amg_setup_ms = 0.0
    if HAS_AMG:
        t0 = time.perf_counter()
        ml_amg = _amg_solver(A_scipy)
        amg_setup_ms = (time.perf_counter() - t0) * 1000

    M_amg = get_dense_amg(A_scipy, viz_limit=viz_n, tol=1e-6, progress_interval=200, ml=ml_amg)

    A_viz_n = A_viz[:viz_n, :viz_n]
    assert A_viz_n.shape[0] == viz_n and A_viz_n.shape[1] == viz_n
    M_gpu = M_neural_gpu[:viz_n, :viz_n]
    M_amg_n = M_amg[:viz_n, :viz_n]

    pcg_tol = 1e-8
    pcg_max_iter = 5000
    np.random.seed(123)
    b_np = np.random.randn(viz_n).astype(np.float64)
    b_np = b_np / (np.linalg.norm(b_np) + 1e-12)
    b_np = b_np.reshape(-1, 1)

    A_scipy_csr = csr_matrix(A_viz_n.astype(np.float32))

    def apply_M_block_diag(r):
        r_batched = r.unsqueeze(0)
        out = apply_block_diagonal_M(
            precond_out,
            r_batched,
            leaf_size=leaf_L,
            leaf_apply_size=leaf_apply_L,
            jacobi_inv_diag=jacobi_inv_diag,
        )
        return out.squeeze(0)

    with torch.inference_mode():
        if device.type == "cuda":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                csr_indptr = torch.tensor(
                    np.ascontiguousarray(A_scipy_csr.indptr.astype(np.int32)),
                    dtype=torch.int32,
                    device=device,
                ).contiguous()
                csr_indices = torch.tensor(
                    np.ascontiguousarray(A_scipy_csr.indices.astype(np.int32)),
                    dtype=torch.int32,
                    device=device,
                ).contiguous()
                csr_data = torch.tensor(
                    np.ascontiguousarray(A_scipy_csr.data.astype(np.float32)),
                    dtype=torch.float32,
                    device=device,
                ).contiguous()
                A_gpu = torch.sparse_csr_tensor(
                    csr_indptr, csr_indices, csr_data, size=(viz_n, viz_n)
                )
        else:
            A_gpu = torch.from_numpy(A_viz_n.astype(np.float32)).to(device).contiguous()
        b_gpu = torch.from_numpy(b_np).float().to(device).contiguous()

        x_gpu_none, iters_none_gpu, solve_none_gpu_ms = pcg_gpu(
            A_gpu,
            b_gpu,
            lambda r: r,
            tol=pcg_tol,
            max_iter=pcg_max_iter,
            device=device,
        )

        if device.type == "cuda":
            try:
                x_gpu, iters_leaf, solve_leaf_ms = pcg_gpu_cudagraph(
                    A_gpu,
                    b_gpu,
                    precond_out,
                    jacobi_inv_diag,
                    tol=pcg_tol,
                    max_iter=pcg_max_iter,
                    device=device,
                    check_freq=check_freq,
                    leaf_size=leaf_L,
                    leaf_apply_size=leaf_apply_L,
                )
            except Exception as e:
                print(f"  LeafOnly (CUDA graph failed, using regular solve): {e}")
                x_gpu, iters_leaf, solve_leaf_ms = pcg_gpu(
                    A_gpu,
                    b_gpu,
                    apply_M_block_diag,
                    tol=pcg_tol,
                    max_iter=pcg_max_iter,
                    device=device,
                    check_freq=check_freq,
                )
        else:
            x_gpu, iters_leaf, solve_leaf_ms = pcg_gpu(
                A_gpu,
                b_gpu,
                apply_M_block_diag,
                tol=pcg_tol,
                max_iter=pcg_max_iter,
                device=device,
                check_freq=check_freq,
            )

    x_cpu_none, iters_none_cpu, solve_none_cpu_ms = pcg_cpu(
        A_viz_n.astype(np.float64), b_np, lambda r: r, tol=pcg_tol, max_iter=pcg_max_iter
    )

    setup_none_ms = 0.0
    total_none_gpu_ms = setup_none_ms + solve_none_gpu_ms
    total_none_cpu_ms = setup_none_ms + solve_none_cpu_ms

    total_leaf_ms = inference_ms + solve_leaf_ms

    solve_amg_ms = 0.0
    iters_amg = 0
    if ml_amg is not None:
        def apply_M_amg(r):
            r_flat = np.asarray(r, dtype=np.float64).ravel()
            x0 = np.zeros(A_scipy.shape[0], dtype=np.float64)
            z_full = ml_amg.solve(r_flat, x0=x0, maxiter=1, cycle='V', tol=1e-6)
            return z_full[:viz_n].reshape(-1, 1).astype(np.float64)
        x_cpu, iters_amg, solve_amg_ms = pcg_cpu(A_viz_n.astype(np.float64), b_np, apply_M_amg, tol=pcg_tol, max_iter=pcg_max_iter)
    total_amg_ms = amg_setup_ms + solve_amg_ms

    solve_amgx_ms = 0.0
    iters_amgx = 0
    amgx_setup_ms = 0.0
    if HAS_AMGX and device.type == 'cuda':
        A_amgx_csr = csr_matrix(A_viz_n.astype(np.float64))
        A_amgx_csr.indptr = np.ascontiguousarray(A_amgx_csr.indptr.astype(np.int32))
        A_amgx_csr.indices = np.ascontiguousarray(A_amgx_csr.indices.astype(np.int32))
        A_amgx_csr.data = np.ascontiguousarray(A_amgx_csr.data.astype(np.float64))

        b_flat = np.ascontiguousarray(b_np.ravel().astype(np.float64))
        x_init = np.zeros(viz_n, dtype=np.float64)

        try:
            _stderr_fd = os.dup(2)
            _devnull = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(_devnull, 2)
                pyamgx.initialize()
            finally:
                os.dup2(_stderr_fd, 2)
                os.close(_stderr_fd)
                os.close(_devnull)

            cfg = pyamgx.Config().create_from_dict({
                "config_version": 2,
                "determinism_flag": 1,
                "exception_handling": 1,
                "solver": {
                    "solver": "PCG",
                    "max_iters": pcg_max_iter,
                    "convergence": "RELATIVE_INI",
                    "tolerance": float(pcg_tol),
                    "monitor_residual": 1,
                    "preconditioner": {
                        "solver": "AMG",
                        "algorithm": "AGGREGATION",
                        "selector": "SIZE_2",
                        "cycle": "V",
                        "smoother": "MULTICOLOR_GS",
                        "presweeps": 1,
                        "postsweeps": 1
                    }
                }
            })
            rsc = pyamgx.Resources().create_simple(cfg)
            A_amgx = pyamgx.Matrix().create(rsc)
            A_amgx.upload_CSR(A_amgx_csr)
            b_amgx = pyamgx.Vector().create(rsc)
            b_amgx.upload(b_flat)
            x_amgx = pyamgx.Vector().create(rsc)
            x_amgx.upload(x_init)

            solver = pyamgx.Solver().create(rsc, cfg)

            t0 = time.perf_counter()
            solver.setup(A_amgx)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            amgx_setup_ms = (time.perf_counter() - t0) * 1000

            num_amgx_warmup = 3
            for _ in range(num_amgx_warmup):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                x_amgx.upload(x_init)
                solver.solve(b_amgx, x_amgx)
            if device.type == 'cuda':
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            x_amgx.upload(x_init)
            solver.solve(b_amgx, x_amgx)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            solve_amgx_ms = (time.perf_counter() - t0) * 1000

            iters_amgx = solver.iterations_number

            solver.destroy()
            x_amgx.destroy()
            b_amgx.destroy()
            A_amgx.destroy()
            rsc.destroy()
            cfg.destroy()
            _stderr_fd = os.dup(2)
            _devnull = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(_devnull, 2)
                pyamgx.finalize()
            finally:
                os.dup2(_stderr_fd, 2)
                os.close(_stderr_fd)
                os.close(_devnull)
        except Exception as e:
            if HAS_AMGX:
                print(f"Warning: AMGX failed: {e}")
            iters_amgx = 0
            solve_amgx_ms = 0.0
            amgx_setup_ms = 0.0
            try:
                _stderr_fd = os.dup(2)
                _devnull = os.open(os.devnull, os.O_WRONLY)
                try:
                    os.dup2(_devnull, 2)
                    pyamgx.finalize()
                finally:
                    os.dup2(_stderr_fd, 2)
                    os.close(_stderr_fd)
                    os.close(_devnull)
            except Exception:
                pass
    total_amgx_ms = amgx_setup_ms + solve_amgx_ms

    print("\nPCG solve A x = b (relative residual tol={:.0e})".format(pcg_tol))
    print("Unpreconditioned (CG):")
    print(f"  Setup: {setup_none_ms:.2f} ms, solve (GPU): {solve_none_gpu_ms:.2f} ms, {iters_none_gpu} iterations, total: {total_none_gpu_ms:.2f} ms")
    print(f"  Setup: {setup_none_ms:.2f} ms, solve (CPU): {solve_none_cpu_ms:.2f} ms, {iters_none_cpu} iterations, total: {total_none_cpu_ms:.2f} ms")
    print("LeafOnly:")
    print(f"  Inference: {inference_ms:.2f} ms, solve: {solve_leaf_ms:.2f} ms, {iters_leaf} iterations, total: {total_leaf_ms:.2f} ms")
    print("AMG (CPU):")
    print(f"  Setup: {amg_setup_ms:.2f} ms, solve: {solve_amg_ms:.2f} ms, {iters_amg} iterations, total: {total_amg_ms:.2f} ms")
    if HAS_AMGX and device.type == 'cuda' and (iters_amgx > 0 or amgx_setup_ms > 0):
        print("AMGX (GPU) [PCG + AMG]:")
        print(f"  Setup: {amgx_setup_ms:.2f} ms, solve: {solve_amgx_ms:.2f} ms, {iters_amgx} iterations, total: {total_amgx_ms:.2f} ms")

    A_inv_viz = np.linalg.inv(A_viz_n)
    diag_ainv = np.diag(A_inv_viz)
    print(f"\nTrue inverse A^{{-1}} diagonal (viz {viz_n}x{viz_n}): min={diag_ainv.min():.6f}, max={diag_ainv.max():.6f}, mean={diag_ainv.mean():.6f}")

    cond_A = np.linalg.cond(A_viz_n)
    print(f"Condition number (block A): {cond_A:.2e}")
    print(f"Leaf boundaries: every {leaf_L}")

    PLOT_MATRICES = True

    if PLOT_MATRICES:
        M_neural_n = M_gpu.detach().cpu().numpy()
        methods = [("LeafOnly", M_neural_n), ("AMG", M_amg_n)]
        n_cols = 5
        n_rows = 1 + len(methods)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 + 3 * n_rows), constrained_layout=True)

        log_ainv = np.log10(np.abs(A_inv_viz) + 1e-9)
        log_m_leaf = np.log10(np.abs(M_neural_n) + 1e-9)
        log_m_amg = np.log10(np.abs(M_amg_n) + 1e-9)
        vmin_log = min(log_ainv.min(), log_m_leaf.min(), log_m_amg.min())
        vmax_log = max(log_ainv.max(), log_m_leaf.max(), log_m_amg.max())

        axes[0, 0].imshow(np.log10(np.abs(A_viz_n) + 1e-9), cmap='magma', aspect='auto')
        axes[0, 0].set_title(f"A (input) log10 [leaf {leaf_L}x{leaf_L}]")
        plt.colorbar(axes[0, 0].images[0], ax=axes[0, 0])
        im_ainv = axes[0, 1].imshow(log_ainv, cmap='magma', aspect='auto', vmin=vmin_log, vmax=vmax_log)
        axes[0, 1].set_title(f"A^{{-1}} (viz {viz_n}x{viz_n}) log10")
        plt.colorbar(im_ainv, ax=axes[0, 1])
        ax_blk = axes[0, 2]
        rgb_layout = _leaf_block_partition_diagram(viz_n, leaf_L)
        ax_blk.imshow(rgb_layout, aspect="auto", interpolation="nearest")
        ax_blk.set_title(
            f"Leaf partition (on-diag {leaf_L}² vs off-diag), n={viz_n}, K={viz_n // leaf_L}"
        )
        leg = ax_blk.legend(
            handles=[
                Patch(facecolor=_LEAF_DIAG_BASE, edgecolor="0.3", linewidth=0.5, label=f"On-diagonal leaf ({leaf_L}×{leaf_L})"),
                Patch(facecolor=_LEAF_OFF_BASE, edgecolor="0.3", linewidth=0.5, label="Off-diagonal"),
            ],
            loc="upper right",
            fontsize=8,
            framealpha=0.92,
        )
        ax_a = axes[0, 3]
        try:
            evals_A = np.linalg.eigvals(A_viz_n)
            ax_a.scatter(evals_A.real, evals_A.imag, alpha=0.7, s=12, c='C0', edgecolors='none')
            ax_a.axhline(0.0, color='k', linestyle='-', alpha=0.3)
            ax_a.set_xlabel('Re(λ)')
            ax_a.set_ylabel('Im(λ)')
            ax_a.set_title(f"Eigenvalues of A (n={viz_n})")
            ax_a.set_aspect('equal', adjustable='box')
            r_min, r_max = evals_A.real.min(), evals_A.real.max()
            i_max = np.abs(evals_A.imag).max()
            margin = 0.1 * max(r_max - r_min, 2 * i_max, 1.0) or 0.2
            ax_a.set_xlim(r_min - margin, r_max + margin)
            ax_a.set_ylim(-max(i_max, margin), max(i_max, margin))
        except Exception as e:
            ax_a.text(0.5, 0.5, f"eig failed:\n{e}", transform=ax_a.transAxes, ha='center', va='center', fontsize=9)
            ax_a.set_title("Eigenvalues of A (failed)")
        ax_a_t = axes[0, 4]
        ax_a_t.axis('off')
        ax_a_t.text(0.1, 0.8, "Unpreconditioned A", fontsize=12, fontfamily='monospace')
        ax_a_t.text(0.1, 0.65, f"Cond(A): {cond_A:.2e}", fontsize=12, fontfamily='monospace')

        am_log_min, am_log_max = None, None
        for _name, M in methods:
            AM = A_viz_n @ M
            am_abs_log = np.log10(np.abs(AM) + 1e-9)
            lo, hi = am_abs_log.min(), am_abs_log.max()
            am_log_min = lo if am_log_min is None else min(am_log_min, lo)
            am_log_max = hi if am_log_max is None else max(am_log_max, hi)
        if am_log_min is None:
            am_log_min, am_log_max = -8.0, 0.0

        for idx, (name, M) in enumerate(methods):
            row = 1 + idx
            im_m = axes[row, 0].imshow(np.log10(np.abs(M) + 1e-9), cmap='magma', aspect='auto', vmin=vmin_log, vmax=vmax_log)
            axes[row, 0].set_title(f"{name} M (log10)")
            plt.colorbar(im_m, ax=axes[row, 0])
            abs_err = np.abs(M - A_inv_viz)
            log_err = np.log10(abs_err + 1e-12)
            im_diff = axes[row, 1].imshow(log_err, cmap='magma', aspect='auto')
            axes[row, 1].set_title(f"{name} |M − A^{{-1}}| (log10)")
            plt.colorbar(im_diff, ax=axes[row, 1])
            AM = A_viz_n @ M
            am_abs_log = np.log10(np.abs(AM) + 1e-9)
            im_am = axes[row, 2].imshow(am_abs_log, cmap='magma', aspect='auto', vmin=am_log_min, vmax=am_log_max)
            axes[row, 2].set_title(f"{name} A·M (log10 |·|)")
            plt.colorbar(im_am, ax=axes[row, 2])

            ax_d = axes[row, 3]
            try:
                evals = np.linalg.eigvals(AM)
                ax_d.scatter(evals.real, evals.imag, alpha=0.7, s=12, c='C0', edgecolors='none')
                ax_d.axvline(1.0, color='r', linestyle='--', alpha=0.8)
                ax_d.axhline(0.0, color='k', linestyle='-', alpha=0.3)
                ax_d.set_xlabel('Re(λ)')
                ax_d.set_ylabel('Im(λ)')
                ax_d.set_title(f"{name} Eigenvalues of A·M (n={viz_n})")
                ax_d.set_aspect('equal', adjustable='box')
                r_min, r_max = evals.real.min(), evals.real.max()
                i_max = np.abs(evals.imag).max()
                margin = 0.1 * max(r_max - r_min, 2 * i_max, 1.0) or 0.2
                ax_d.set_xlim(min(r_min, 1.0) - margin, max(r_max, 1.0) + margin)
                ax_d.set_ylim(-max(i_max, margin), max(i_max, margin))
            except Exception as e:
                ax_d.text(0.5, 0.5, f"eig failed:\n{e}", transform=ax_d.transAxes, ha='center', va='center', fontsize=9)
                ax_d.set_title(f"{name} Eigenvalues (failed)")

            ax_t = axes[row, 4]
            ax_t.axis('off')
            cond_AM = np.linalg.cond(AM)
            err_fro = np.linalg.norm(AM - np.eye(viz_n)) / np.linalg.norm(np.eye(viz_n))
            msg = [
                f"Method: {name}",
                f"Cond(AM): {cond_AM:.2e}",
                f"Improvement: {cond_A/cond_AM:.2f}x",
                f"Frobenius Err: {err_fro:.4f}",
            ]
            color = 'green' if cond_A / cond_AM > 1.0 else 'red'
            for i, line in enumerate(msg):
                ax_t.text(0.1, 0.8 - i * 0.15, line, fontsize=12, color='black' if i != 2 else color, fontfamily='monospace')

        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved to: {out_path}")
    else:
        print("\nSkipped matrix dense plotting/eigenvalues to save time.")


if __name__ == "__main__":
    main()
