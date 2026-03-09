"""Inspect LeafOnly preconditioner: load weights, build M, compare A·M with AMG. Run with python3 InspectModel.py."""
import sys
import time
import os
import warnings
from pathlib import Path

# PyTorch sparse CSR is still in beta; suppress the warning when building A_gpu for PCG
warnings.filterwarnings("ignore", message=".*Sparse CSR.*", category=UserWarning)

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import numpy as np
import matplotlib.pyplot as plt
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

from LeafOnly import (
    LeafOnlyNet,
    read_leaf_only_header,
    load_leaf_only_weights,
    apply_block_structured_M,
    build_hodlr_off_diag_structure,
    build_off_diag_super_connectivity,
    LEAF_SIZE,
    VIEW_SIZE,
    next_valid_size,
    FluidGraphDataset,
)


def print_hodlr_m_stats(diag_blocks, off_diag_list, off_diag_struct, model_leaf=None):
    """Print per-level statistics of the neural M (diag + off-diag by HODLR level) for normalized*scale analysis."""
    import math
    # --- Level 0: diagonal blocks ---
    d = diag_blocks.detach()
    if d.dim() == 3:
        d = d.unsqueeze(0)
    B, num_leaves, L, _ = d.shape
    frobs = (d ** 2).sum(dim=(-2, -1)).sqrt()
    flat = d.reshape(-1)
    print("\n  [HODLR M stats] Level 0 (diagonal blocks)")
    print(f"    shape {tuple(d.shape)}, leaf_size={L}")
    print(f"    Frobenius per block: mean={frobs.mean().item():.6f} std={frobs.std().item():.6f} min={frobs.min().item():.6f} max={frobs.max().item():.6f}")
    print(f"    elements: mean={flat.mean().item():.6e} std={flat.std().item():.6e} min={flat.min().item():.6e} max={flat.max().item():.6e}")

    if not off_diag_list or not off_diag_struct:
        return
    by_level = {}
    for idx, spec in enumerate(off_diag_struct):
        L = spec["level"]
        by_level.setdefault(L, []).append((idx, spec))

    level_scale_params = getattr(model_leaf, "level_scale_params", None)
    for level in sorted(by_level.keys()):
        entries = by_level[level]
        scales = []
        u_frobs, v_frobs, block_frobs = [], [], []
        block_means, block_stds, block_mins, block_maxs = [], [], [], []
        for idx, spec in entries:
            U = off_diag_list[idx][0]
            V = off_diag_list[idx][1]
            if isinstance(U, (list, tuple)):
                U, V = U[0], V[0]
            U = U.detach().float()
            V = V.detach().float()
            if U.dim() == 3:
                U, V = U[0], V[0]
            side, rank = U.shape[0], U.shape[1]
            uv = U @ V.T
            u_frobs.append(U.norm().item())
            v_frobs.append(V.norm().item())
            block_frobs.append(uv.norm().item())
            flat_uv = uv.reshape(-1)
            block_means.append(flat_uv.mean().item())
            block_stds.append(flat_uv.std().item())
            block_mins.append(flat_uv.min().item())
            block_maxs.append(flat_uv.max().item())
        scale_str = ""
        if level_scale_params is not None:
            p = level_scale_params.detach().cpu()
            log_scale = (p[0] + (level - 1.0) * p[1]).item()
            scale_val = math.exp(log_scale)
            scale_str = f" scale(level)={scale_val:.6e} (log={log_scale:.4f})"
        n_blk = len(entries)
        rs, re = entries[0][1]["row_start"], entries[0][1]["row_end"]
        side = re - rs
        rk = entries[0][1]["rank"]
        print(f"\n  [HODLR M stats] Level {level} (off-diag) n_blocks={n_blk} side={side} rank={rk}{scale_str}")
        print(f"    U Frobenius: mean={np.mean(u_frobs):.6f} std={np.std(u_frobs):.6f}")
        print(f"    V Frobenius: mean={np.mean(v_frobs):.6f} std={np.std(v_frobs):.6f}")
        print(f"    UV^T block Frobenius: mean={np.mean(block_frobs):.6f} std={np.std(block_frobs):.6f}")
        print(f"    UV^T elements: mean={np.mean(block_means):.6e} std={np.mean(block_stds):.6e} min={np.min(block_mins):.6e} max={np.max(block_maxs):.6e}")


def _amg_solver(A_sparse_scipy):
    """Build AMG hierarchy only (setup). Returns solver or None if no pyamg."""
    if not HAS_AMG:
        return None
    dtype = np.float64
    if A_sparse_scipy.dtype != dtype:
        A_sparse_scipy = A_sparse_scipy.astype(dtype)
    return pyamg.smoothed_aggregation_solver(A_sparse_scipy)


def get_dense_amg(A_sparse_scipy, viz_limit=200, maxiter=1, tol=1e-6, progress_interval=200, ml=None):
    """Build dense M by columns via AMG solves (for plot only). If ml is provided, reuse it (no setup)."""
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
    """
    Optimized PCG on GPU with accurate CUDA Event timing (CUDA) or wall-clock (MPS).
    Includes initial residual calculation and preconditioning inside the timed block.
    """
    if device is None:
        device = b.device
        
    n = b.shape[0]
    x = torch.zeros(n, 1, device=device, dtype=b.dtype)
    
    # Timing: CUDA Events on CUDA, wall-clock on MPS/CPU
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
    else:
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()

    # Initial setup is now correctly timed
    r = b - (A @ x.squeeze(-1)).unsqueeze(-1)
    z = apply_precond(r)
    p = z.clone()
    
    rho = (r * z).sum()
    b_norm_sq = (b * b).sum()
    tol_sq = (tol * tol) * b_norm_sq

    iters = 0
    if b_norm_sq.item() > 0:
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

            # Periodic sync to check convergence
            if (k + 1) % check_freq == 0 or k == max_iter - 1:
                r_sq = (r * r).sum()
                if r_sq.item() <= tol_sq.item():
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
    diag_blocks,
    off_diag_list,
    off_diag_struct,
    tol=1e-8,
    max_iter=500,
    device=None,
    check_freq=3,
):
    """
    PCG on GPU using CUDA Graphs: capture one iteration and replay to eliminate Python dispatch.
    Preconditioner is inlined (diag_blocks + off_diag_list) so the graph has no Python calls.
    Returns (x, iters, wall_ms) with wall_ms measuring only GPU time (replays).
    """
    if device is None:
        device = b_gpu.device
    n = b_gpu.shape[0]
    viz_n = n
    num_leaves = n // LEAF_SIZE

    # diag_blocks may be (B, num_leaves, L, L) or (num_leaves, L, L); in-graph we need (num_leaves, L, L)
    diag_blocks_0 = diag_blocks[0] if diag_blocks.dim() == 4 else diag_blocks

    # Static tensors (fixed addresses for capture; no new allocations inside graph)
    x_static = torch.zeros_like(b_gpu)
    r_static = torch.zeros_like(b_gpu)
    p_static = torch.zeros_like(b_gpu)
    z_static = torch.zeros_like(b_gpu)
    z_off_static = torch.zeros_like(b_gpu)
    rho_static = torch.zeros(1, device=device, dtype=b_gpu.dtype)

    # Initial state in Python (once): use LeafOnly's level-wise batched apply for first z = M @ r
    x_static.zero_()
    r_static.copy_(b_gpu)
    diag_4d = diag_blocks if diag_blocks.dim() == 4 else diag_blocks.unsqueeze(0)
    z_initial = apply_block_structured_M(
        diag_4d, off_diag_list, r_static.unsqueeze(0), off_diag_struct,
        leaf_size=LEAF_SIZE,
    )
    z_static.copy_(z_initial.squeeze(0))
    p_static.copy_(z_static)
    rho_static.fill_((r_static * z_static).sum())

    b_norm_sq = (b_gpu * b_gpu).sum()
    tol_sq = (tol * tol) * b_norm_sq

    # One iteration: read x,r,p,rho; write x,r,p,rho (in-place where possible)
    def one_iter():
        Ap = A_gpu @ p_static.squeeze(-1)
        Ap = Ap.unsqueeze(-1)
        pAp = (p_static * Ap).sum()
        alpha = rho_static / pAp
        x_static.add_(p_static * alpha)
        r_static.sub_(Ap * alpha)
        # Inline M: z_static = M @ r_static (reuse z_off_static to avoid allocation in graph)
        r_blocks = r_static.view(num_leaves, LEAF_SIZE, 1)
        z_diag = torch.bmm(diag_blocks_0, r_blocks).view(viz_n, 1)
        z_off_static.zero_()
        if off_diag_list and off_diag_struct:
            for idx, spec in enumerate(off_diag_struct):
                U = off_diag_list[idx][0][0]
                V = off_diag_list[idx][1][0]
                rs, re = spec["row_start"], spec["row_end"]
                cs, ce = spec["col_start"], spec["col_end"]
                r_c = r_static[cs:ce]
                z_off_static[rs:re].add_(U @ (V.T @ r_c))
                r_r = r_static[rs:re]
                z_off_static[cs:ce].add_(V @ (U.T @ r_r))
        z_static.copy_(z_diag + z_off_static)
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
            if r_sq.item() <= tol_sq.item():
                break
    end_ev.record()
    torch.cuda.synchronize()
    wall_ms = start_ev.elapsed_time(end_ev)
    return x_static.clone(), iters, wall_ms


def pcg_cpu(A, b, apply_precond, tol=1e-8, max_iter=500, debug=False):
    """PCG on CPU: solve A x = b. Precond(r) must return z = M@r. Returns (x, iters, wall_ms)."""
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

    print(f"Loading data from {data_folder}")
    dataset = FluidGraphDataset([Path(data_folder)])
    if len(dataset) == 0:
        raise SystemExit("No frames found (need nodes.bin, edge_index_*.bin, A_values.bin).")
    frame_idx = min(frame_idx, len(dataset) - 1)
    batch = dataset[frame_idx]

    x = batch['x'].unsqueeze(0).to(device)
    num_nodes_real = int(batch['num_nodes'])
    print(f"System N={num_nodes_real}")

    # n_requested = min(VIEW_SIZE, num_nodes_real)
    n_requested = 256
    n_pad = next_valid_size(n_requested, LEAF_SIZE)
    if n_pad != n_requested:
        print(f"  Padding view: {n_requested} nodes -> {n_pad} (power-of-2 * {LEAF_SIZE})")

    # A_viz: n_pad x n_pad block for preconditioner comparison (real A in [:n_requested,:n_requested], identity for padded dofs)
    ei, ev = batch['edge_index'], batch['edge_values']
    em = (ei[0] < n_requested) & (ei[1] < n_requested)
    A_small = torch.sparse_coo_tensor(ei[:, em], ev[em], (n_requested, n_requested)).coalesce().to_dense().numpy()
    A_viz = np.zeros((n_pad, n_pad), dtype=np.float64)
    A_viz[:n_requested, :n_requested] = A_small
    if n_pad > n_requested:
        A_viz[n_requested:, n_requested:] = np.eye(n_pad - n_requested, dtype=np.float64)
    viz_n = n_pad

    # LeafOnly: setup (load) + one forward (get M), no compilation
    print("\nLeafOnly (GPU)...")
    if not leaf_only_weights_path.exists():
        raise SystemExit(f"Leaf-only weights not found: {leaf_only_weights_path}. Run LeafOnly.py first.")
    header_lo = read_leaf_only_header(leaf_only_weights_path)
    d_model_lo, leaf_size_lo, input_dim_lo, num_layers_lo, num_heads_lo = header_lo[:5]
    use_gcn_lo = header_lo[5] if len(header_lo) > 5 else True  # old checkpoints have no use_gcn → True
    model_leaf = LeafOnlyNet(
        input_dim=input_dim_lo, d_model=d_model_lo, leaf_size=LEAF_SIZE, num_layers=num_layers_lo,
        num_heads=num_heads_lo, use_gcn=bool(use_gcn_lo),
    ).to(device)
    # Default compile (reduce-overhead causes CUDA graph capture to fail on LeafOnly's index_put_)
    model_leaf = torch.compile(model_leaf)
    load_leaf_only_weights(model_leaf, leaf_only_weights_path)

    x_leaf = x[:, :n_requested, :].clone()
    if n_pad > n_requested:
        x_leaf = F.pad(x_leaf, (0, 0, 0, n_pad - n_requested), value=0.0)
    em = (ei[0] < n_requested) & (ei[1] < n_requested)
    edge_index_leaf = ei[:, em].to(device)
    edge_values_leaf = batch['edge_values'][em].to(device)
    scale_A = batch.get('scale_A')
    if scale_A is not None and not isinstance(scale_A, torch.Tensor):
        scale_A = torch.tensor(scale_A, device=device, dtype=x_leaf.dtype)

    # Precompute off-diag masks once (same as LeafOnly training) so forward never builds them
    off_diag_struct = build_hodlr_off_diag_structure(
        n_pad, LEAF_SIZE, rank_base=getattr(model_leaf, "rank_base", 16)
    )
    precomputed_masks = [
        build_off_diag_super_connectivity(
            edge_index_leaf, spec["row_start"], spec["row_end"], spec["col_start"], spec["col_end"],
            device, LEAF_SIZE,
        )
        for spec in off_diag_struct
    ] if off_diag_struct else []

    # Pass global_features so lift FiLM matches training (required for v10+ weights)
    global_feat = batch.get('global_features')
    if global_feat is not None:
        global_feat = global_feat.to(device)
        if global_feat.dim() == 1:
            global_feat = global_feat.unsqueeze(0)

    with torch.no_grad():
        # Warm up inference (torch.compile + CUDA); match training warmup so timing is comparable
        for _ in range(15):
            _ = model_leaf(
                x_leaf, edge_index=edge_index_leaf, edge_values=edge_values_leaf, scale_A=scale_A,
                precomputed_masks=precomputed_masks, global_features=global_feat,
            )
        if device.type == 'cuda':
            torch.cuda.synchronize()
            inf_start = torch.cuda.Event(enable_timing=True)
            inf_end = torch.cuda.Event(enable_timing=True)
            inf_start.record()
            diag_blocks, off_diag_list = model_leaf(
                x_leaf, edge_index=edge_index_leaf, edge_values=edge_values_leaf, scale_A=scale_A,
                precomputed_masks=precomputed_masks, global_features=global_feat,
            )
            inf_end.record()
            torch.cuda.synchronize()
            inference_ms = inf_start.elapsed_time(inf_end)
        else:
            t0 = time.perf_counter()
            diag_blocks, off_diag_list = model_leaf(
                x_leaf, edge_index=edge_index_leaf, edge_values=edge_values_leaf, scale_A=scale_A,
                precomputed_masks=precomputed_masks, global_features=global_feat,
            )
            inference_ms = (time.perf_counter() - t0) * 1000
    # Keep preconditioner outputs on GPU so PCG CUDAGRAPH solver uses them without CPU round-trip
    diag_blocks = diag_blocks.to(device)
    if off_diag_list:
        off_diag_list = [
            (tuple(t.to(device) for t in pair[0]), tuple(t.to(device) for t in pair[1]))
            if isinstance(pair[0], (list, tuple)) and isinstance(pair[1], (list, tuple))
            else pair
            for pair in off_diag_list
        ]
    num_leaves = n_pad // LEAF_SIZE
    # Build dense M on GPU to avoid CPU-GPU sync in a loop (keeps inference timing accurate)
    M_neural_gpu = torch.zeros((n_pad, n_pad), dtype=torch.float32, device=device)
    for b in range(num_leaves):
        r0, r1 = b * LEAF_SIZE, (b + 1) * LEAF_SIZE
        M_neural_gpu[r0:r1, r0:r1] = diag_blocks[0, b]
    if off_diag_list and getattr(model_leaf, 'off_diag_struct', None):
        for idx, spec in enumerate(model_leaf.off_diag_struct):
            U = off_diag_list[idx][0][0]   # (1, side, rank)
            V = off_diag_list[idx][1][0]
            rs, re = spec["row_start"], spec["row_end"]
            cs, ce = spec["col_start"], spec["col_end"]
            UVt = (U @ V.T).squeeze(0)
            VUt = (V @ U.T).squeeze(0)
            M_neural_gpu[rs:re, cs:ce] = UVt
            M_neural_gpu[cs:ce, rs:re] = VUt
    print(f"  LeafOnly: {num_leaves} leaves, {n_pad}x{n_pad} M")
    print_hodlr_m_stats(diag_blocks, off_diag_list, getattr(model_leaf, "off_diag_struct", None), model_leaf)

    # AMG on the same n_pad x n_pad block (CPU)
    A_scipy = csr_matrix(A_viz.astype(np.float64))
    ml_amg = None
    amg_setup_ms = 0.0
    if HAS_AMG:
        t0 = time.perf_counter()
        ml_amg = _amg_solver(A_scipy)
        amg_setup_ms = (time.perf_counter() - t0) * 1000

    # Dense M only for plot (not part of timing)
    M_amg = get_dense_amg(A_scipy, viz_limit=viz_n, tol=1e-6, progress_interval=200, ml=ml_amg)

    # All solves use the subset block A_viz_n (viz_n x viz_n), not the full dataset A.
    A_viz_n = A_viz[:viz_n, :viz_n]
    assert A_viz_n.shape[0] == viz_n and A_viz_n.shape[1] == viz_n, "Solve must be on subset A (viz_n x viz_n)"
    M_gpu = M_neural_gpu[:viz_n, :viz_n]
    M_amg_n = M_amg[:viz_n, :viz_n]

    # PCG solve A x = b on block A_viz_n (viz_n x viz_n)
    pcg_tol = 1e-8
    pcg_max_iter = 5000
    np.random.seed(123)
    b_np = np.random.randn(viz_n).astype(np.float64)
    b_np = b_np / (np.linalg.norm(b_np) + 1e-12)
    b_np = b_np.reshape(-1, 1)

    # On CUDA use sparse CSR (cuSPARSE); MPS does not support sparse_csr_tensor so use dense
    A_scipy_csr = csr_matrix(A_viz_n.astype(np.float32))
    if device.type == "cuda":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Sparse CSR beta warning
            A_gpu = torch.sparse_csr_tensor(
                torch.tensor(A_scipy_csr.indptr, dtype=torch.int32, device=device),
                torch.tensor(A_scipy_csr.indices, dtype=torch.int32, device=device),
                torch.tensor(A_scipy_csr.data, dtype=torch.float32, device=device),
                size=(viz_n, viz_n),
            )
    else:
        A_gpu = torch.from_numpy(A_viz_n.astype(np.float32)).to(device)
    b_gpu = torch.from_numpy(b_np).float().to(device)

    # Unpreconditioned (CG): identity preconditioner
    with torch.no_grad():
        x_gpu_none, iters_none_gpu, solve_none_gpu_ms = pcg_gpu(A_gpu, b_gpu, lambda r: r, tol=pcg_tol, max_iter=pcg_max_iter, device=device)
    x_cpu_none, iters_none_cpu, solve_none_cpu_ms = pcg_cpu(A_viz_n.astype(np.float64), b_np, lambda r: r, tol=pcg_tol, max_iter=pcg_max_iter)

    setup_none_ms = 0.0
    total_none_gpu_ms = setup_none_ms + solve_none_gpu_ms
    total_none_cpu_ms = setup_none_ms + solve_none_cpu_ms

    # Use LeafOnly's level-wise batched apply (same fast path as LeafOnly.py)
    with torch.no_grad():
        def apply_M_block_sparse(r):
            r_batched = r.unsqueeze(0)  # (1, viz_n, 1)
            out = apply_block_structured_M(
                diag_blocks, off_diag_list, r_batched,
                getattr(model_leaf, "off_diag_struct", None),
                leaf_size=LEAF_SIZE,
            )
            return out.squeeze(0)  # (viz_n, 1)

    # On CUDA use graph solve (no Python overhead); otherwise or on graph failure use regular pcg_gpu
    if device.type == "cuda":
        try:
            x_gpu, iters_leaf, solve_leaf_ms = pcg_gpu_cudagraph(
                A_gpu,
                b_gpu,
                diag_blocks,
                off_diag_list,
                getattr(model_leaf, "off_diag_struct", None),
                tol=pcg_tol,
                max_iter=pcg_max_iter,
                device=device,
                check_freq=1,  # exact iteration count for e2e benchmark
            )
        except Exception as e:
            print(f"  LeafOnly (CUDA graph failed, using regular solve): {e}")
            x_gpu, iters_leaf, solve_leaf_ms = pcg_gpu(A_gpu, b_gpu, apply_M_block_sparse, tol=pcg_tol, max_iter=pcg_max_iter, device=device)
    else:
        x_gpu, iters_leaf, solve_leaf_ms = pcg_gpu(A_gpu, b_gpu, apply_M_block_sparse, tol=pcg_tol, max_iter=pcg_max_iter, device=device)

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

    # AMGX (GPU): PCG with preconditioner. Use same matrix as other solvers (A_viz_n).
    solve_amgx_ms = 0.0
    iters_amgx = 0
    amgx_setup_ms = 0.0
    if HAS_AMGX and device.type == 'cuda':
        # 1. Explicitly cast to strictly 32-bit ints and 64-bit floats.
        # This prevents the silent Cython dtype mismatch that mangles the CSR structure.
        A_amgx_csr = csr_matrix(A_viz_n.astype(np.float64))
        A_amgx_csr.indptr = np.ascontiguousarray(A_amgx_csr.indptr.astype(np.int32))
        A_amgx_csr.indices = np.ascontiguousarray(A_amgx_csr.indices.astype(np.int32))
        A_amgx_csr.data = np.ascontiguousarray(A_amgx_csr.data.astype(np.float64))

        b_flat = np.ascontiguousarray(b_np.ravel().astype(np.float64))
        x_init = np.zeros(viz_n, dtype=np.float64)

        try:
            # Suppress AMGX deprecation messages (initialize_plugins/finalize_plugins) from stderr
            _stderr_fd = os.dup(2)
            _devnull = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(_devnull, 2)
                pyamgx.initialize()
            finally:
                os.dup2(_stderr_fd, 2)
                os.close(_stderr_fd)
                os.close(_devnull)

            # 2. Configure PCG to use an actual AMG preconditioner
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

            # Warm up AMGX: run a few solves so timed solve doesn't include one-off costs
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
    print(f"Leaf boundaries: every {LEAF_SIZE}")

    # Convert to NumPy only for plotting
    PLOT_MATRICES = True  # Set to True when you actually want the image

    if PLOT_MATRICES:
        M_neural_n = M_gpu.cpu().numpy()
        methods = [("LeafOnly", M_neural_n), ("AMG", M_amg_n)]
        n_cols = 5
        n_rows = 1 + len(methods)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 + 3 * n_rows), constrained_layout=True)

        # Shared color scale for A^{-1}, LeafOnly M, and AMG M (log10 magnitude)
        log_ainv = np.log10(np.abs(A_inv_viz) + 1e-9)
        log_m_leaf = np.log10(np.abs(M_neural_n) + 1e-9)
        log_m_amg = np.log10(np.abs(M_amg_n) + 1e-9)
        vmin_log = min(log_ainv.min(), log_m_leaf.min(), log_m_amg.min())
        vmax_log = max(log_ainv.max(), log_m_leaf.max(), log_m_amg.max())

        # Row 0: A, A^{-1}
        axes[0, 0].imshow(np.log10(np.abs(A_viz_n) + 1e-9), cmap='magma', aspect='auto')
        axes[0, 0].set_title(f"A (input) log10 [leaf {LEAF_SIZE}x{LEAF_SIZE}]")
        plt.colorbar(axes[0, 0].images[0], ax=axes[0, 0])
        im_ainv = axes[0, 1].imshow(log_ainv, cmap='magma', aspect='auto', vmin=vmin_log, vmax=vmax_log)
        axes[0, 1].set_title(f"A^{{-1}} (viz {viz_n}x{viz_n}) log10")
        plt.colorbar(im_ainv, ax=axes[0, 1])
        # Row 0, col 2: empty (no A·M for unpreconditioned)
        axes[0, 2].axis('off')
        # Row 0, col 3: eigenvalues of unpreconditioned A (align with method rows)
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
        # Row 0, col 4: condition number (align with method rows text)
        ax_a_t = axes[0, 4]
        ax_a_t.axis('off')
        ax_a_t.text(0.1, 0.8, "Unpreconditioned A", fontsize=12, fontfamily='monospace')
        ax_a_t.text(0.1, 0.65, f"Cond(A): {cond_A:.2e}", fontsize=12, fontfamily='monospace')

        # Shared color scale for all A·M (log10 |·|) panels
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
            # |M - A^{-1}| (absolute error), log10 to show detail
            abs_err = np.abs(M - A_inv_viz)
            log_err = np.log10(abs_err + 1e-12)
            im_diff = axes[row, 1].imshow(log_err, cmap='magma', aspect='auto')
            axes[row, 1].set_title(f"{name} |M − A^{{-1}}| (log10)")
            plt.colorbar(im_diff, ax=axes[row, 1])
            AM = A_viz_n @ M
            # Absolute value, same colormap and shared scale as the other A·M panel(s)
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
