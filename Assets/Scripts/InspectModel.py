"""Inspect LeafOnly preconditioner: load weights, build M, compare A·M with AMG. Run with python3 InspectModel.py."""
import sys
import time
from pathlib import Path

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
except ImportError:
    HAS_AMGX = False
    print("Warning: 'pyamgx' not installed. AMGX (GPU) comparison will be skipped.")
HAS_AMGX = False  # temporarily disabled (fix later)

import torch

import torch.nn.functional as F

from LeafOnly import (
    LeafOnlyNet,
    read_leaf_only_header,
    load_leaf_only_weights,
    apply_block_structured_M,
    LEAF_SIZE,
    VIEW_SIZE,
    next_valid_size,
    FluidGraphDataset,
)


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


def pcg_gpu(A, b, apply_precond, tol=1e-8, max_iter=500, device=None, debug=False):
    """PCG on GPU: solve A x = b. Precond(r) must return z = M@r (apply preconditioner matrix M to r). Returns (x, iters, wall_ms)."""
    if device is None:
        device = b.device
    n = b.shape[0]
    x = torch.zeros(n, 1, device=device, dtype=b.dtype)
    r = b - (A @ x.squeeze(-1)).unsqueeze(-1)
    z = apply_precond(r)
    p = z.clone()
    rho = (r * z).sum().item()
    b_norm_sq = (b * b).sum().item()
    b_norm = (b_norm_sq ** 0.5) if b_norm_sq > 0 else 0.0
    if b_norm_sq <= 0:
        return x, 0, 0.0
    tol_sq = (tol * tol) * b_norm_sq
    r_norm_0 = (r * r).sum().item() ** 0.5
    if debug:
        print(f"    [PCG GPU] init: ||b||={b_norm:.2e}, ||r0||={r_norm_0:.2e}, rho0={rho:.2e}, tol*||b||={tol*b_norm:.2e}")
    t0 = time.perf_counter()
    iters = max_iter
    for k in range(max_iter):
        Ap = (A @ p.squeeze(-1)).unsqueeze(-1)
        pAp = (p * Ap).sum().item()
        if pAp <= 1e-14:
            if debug:
                print(f"    [PCG GPU] iter {k}: pAp={pAp:.2e} <= 0, stopping")
            iters = k
            break
        alpha = rho / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        r_sq = (r * r).sum().item()
        r_norm = r_sq ** 0.5
        if debug and (k < 5 or (k + 1) % 50 == 0 or k == max_iter - 1):
            rel = r_norm / b_norm if b_norm > 0 else float('nan')
            print(f"    [PCG GPU] iter {k+1}: ||r||={r_norm:.2e}, rel_res={rel:.2e}, alpha={alpha:.2e}")
        if r_sq <= tol_sq:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            if debug:
                print(f"    [PCG GPU] converged at iter {k+1}, ||r||={r_norm:.2e}")
            return x, k + 1, (time.perf_counter() - t0) * 1000
        z = apply_precond(r)
        rho_new = (r * z).sum().item()
        beta = rho_new / rho
        p = z + beta * p
        rho = rho_new
    if device.type == 'cuda':
        torch.cuda.synchronize()
    if debug:
        print(f"    [PCG GPU] stopped at iter {iters}, ||r||={r_norm:.2e}, rel_res={r_norm/b_norm:.2e}")
    return x, iters, (time.perf_counter() - t0) * 1000


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

    n_requested = min(VIEW_SIZE, num_nodes_real)
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
        num_heads=num_heads_lo, n_nodes=n_pad, use_gcn=bool(use_gcn_lo),
    ).to(device)
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

    with torch.no_grad():
        _ = model_leaf(x_leaf, edge_index=edge_index_leaf, edge_values=edge_values_leaf, scale_A=scale_A)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        diag_blocks, off_diag_list = model_leaf(
            x_leaf, edge_index=edge_index_leaf, edge_values=edge_values_leaf, scale_A=scale_A
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - t0) * 1000
    num_leaves = n_pad // LEAF_SIZE
    M_neural = np.zeros((n_pad, n_pad), dtype=np.float64)
    for b in range(num_leaves):
        r0, r1 = b * LEAF_SIZE, (b + 1) * LEAF_SIZE
        M_neural[r0:r1, r0:r1] = diag_blocks[0, b].cpu().numpy()
    if off_diag_list and getattr(model_leaf, 'off_diag_struct', None):
        for idx, spec in enumerate(model_leaf.off_diag_struct):
            U = off_diag_list[idx][0][0].cpu().numpy()
            V = off_diag_list[idx][1][0].cpu().numpy()
            rs, re = spec["row_start"], spec["row_end"]
            cs, ce = spec["col_start"], spec["col_end"]
            M_neural[rs:re, cs:ce] = U @ V.T
            M_neural[cs:ce, rs:re] = V @ U.T
    print(f"  LeafOnly: {num_leaves} leaves, {n_pad}x{n_pad} M")

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
    M_neural_n = M_neural[:viz_n, :viz_n]
    M_amg_n = M_amg[:viz_n, :viz_n]

    # PCG solve A x = b on block A_viz_n (viz_n x viz_n)
    pcg_tol = 1e-8
    pcg_max_iter = 5000
    np.random.seed(123)
    b_np = np.random.randn(viz_n).astype(np.float64)
    b_np = b_np / (np.linalg.norm(b_np) + 1e-12)
    b_np = b_np.reshape(-1, 1)

    A_gpu = torch.from_numpy(A_viz_n).float().to(device)
    b_gpu = torch.from_numpy(b_np).float().to(device)

    # Unpreconditioned (CG): identity preconditioner, setup 0
    with torch.no_grad():
        x_gpu_none, iters_none_gpu, solve_none_gpu_ms = pcg_gpu(A_gpu, b_gpu, lambda r: r, tol=pcg_tol, max_iter=pcg_max_iter, device=device)
    x_cpu_none, iters_none_cpu, solve_none_cpu_ms = pcg_cpu(A_viz_n.astype(np.float64), b_np, lambda r: r, tol=pcg_tol, max_iter=pcg_max_iter)
    setup_none_ms = 0.0
    total_none_gpu_ms = setup_none_ms + solve_none_gpu_ms
    total_none_cpu_ms = setup_none_ms + solve_none_cpu_ms

    # SAI training minimizes || M A z - z ||^2, so M ≈ A^{-1}. Apply z = M@r. Use dense M@r (one matvec) for speed;
    # block-structured apply_block_structured_M does 15+ small kernels per apply and is ~10x slower per iteration.
    M_gpu = torch.from_numpy(M_neural_n).float().to(device)
    with torch.no_grad():
        def apply_M_neural(r):
            return M_gpu @ r
        x_gpu, iters_leaf, solve_leaf_ms = pcg_gpu(A_gpu, b_gpu, apply_M_neural, tol=pcg_tol, max_iter=pcg_max_iter, device=device)
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
        # Same matrix and RHS as other solvers (A_viz_n, b_np)
        A_amgx_csr = csr_matrix(A_viz_n.astype(np.float64))
        b_flat = np.ascontiguousarray(b_np.ravel().astype(np.float64))
        x_init = np.zeros(viz_n, dtype=np.float64)
        try:
            pyamgx.initialize()
            # CG = unpreconditioned conjugate gradient; x/b set via upload (like pyamgx tests/demo)
            cfg = pyamgx.Config().create_from_dict({
                "config_version": 2,
                "determinism_flag": 1,
                "exception_handling": 1,
                "solver": {
                    "solver": "CG",
                    "max_iters": pcg_max_iter,
                    "convergence": "RELATIVE_INI",
                    "tolerance": float(pcg_tol),
                    "monitor_residual": 0,
                    "preconditioner": {"solver": "NOSOLVER"},
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
            t0 = time.perf_counter()
            solver.solve(b_amgx, x_amgx, zero_initial_guess=True)
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
            pyamgx.finalize()
        except Exception as e:
            if HAS_AMGX:
                print(f"Warning: AMGX failed: {e}")
            iters_amgx = 0
            solve_amgx_ms = 0.0
            amgx_setup_ms = 0.0
            try:
                pyamgx.finalize()
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
        print("AMGX (GPU) [PCG, no precond]:")
        print(f"  Setup: {amgx_setup_ms:.2f} ms, solve: {solve_amgx_ms:.2f} ms, {iters_amgx} iterations, total: {total_amgx_ms:.2f} ms")

    A_inv_viz = np.linalg.inv(A_viz_n)
    diag_ainv = np.diag(A_inv_viz)
    print(f"\nTrue inverse A^{{-1}} diagonal (viz {viz_n}x{viz_n}): min={diag_ainv.min():.6f}, max={diag_ainv.max():.6f}, mean={diag_ainv.mean():.6f}")

    cond_A = np.linalg.cond(A_viz_n)
    print(f"Condition number (block A): {cond_A:.2e}")
    print(f"Leaf boundaries: every {LEAF_SIZE}")

    methods = [("LeafOnly", M_neural_n), ("AMG", M_amg_n)]
    n_cols = 4
    n_rows = 1 + len(methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 + 3 * n_rows), constrained_layout=True)
    # Row 0: A, A^{-1}
    axes[0, 0].imshow(np.log10(np.abs(A_viz_n) + 1e-9), cmap='magma', aspect='auto')
    axes[0, 0].set_title(f"A (input) log10 [leaf {LEAF_SIZE}x{LEAF_SIZE}]")
    plt.colorbar(axes[0, 0].images[0], ax=axes[0, 0])
    axes[0, 1].imshow(np.log10(np.abs(A_inv_viz) + 1e-9), cmap='magma', aspect='auto')
    axes[0, 1].set_title(f"A^{{-1}} (viz {viz_n}x{viz_n}) log10")
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1])
    # Row 0, col 2: eigenvalues of unpreconditioned A
    ax_a = axes[0, 2]
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
    # Row 0, col 3: condition number and Frobenius (unpreconditioned: no M, so no Frobenius err of I-AM)
    ax_a_t = axes[0, 3]
    ax_a_t.axis('off')
    ax_a_t.text(0.1, 0.8, "Unpreconditioned A", fontsize=12, fontfamily='monospace')
    ax_a_t.text(0.1, 0.65, f"Cond(A): {cond_A:.2e}", fontsize=12, fontfamily='monospace')

    for idx, (name, M) in enumerate(methods):
        row = 1 + idx
        axes[row, 0].imshow(np.log10(np.abs(M) + 1e-9), cmap='magma', aspect='auto')
        axes[row, 0].set_title(f"{name} M (log10)")
        plt.colorbar(axes[row, 0].images[0], ax=axes[row, 0])
        AM = A_viz_n @ M
        am_min, am_max = np.percentile(AM, [2, 98])
        if am_max - am_min < 1e-8:
            am_min, am_max = 0.0, 1.0
        axes[row, 1].imshow(AM, cmap='RdBu_r', vmin=am_min, vmax=am_max, aspect='auto')
        axes[row, 1].set_title(f"{name} A·M")
        plt.colorbar(axes[row, 1].images[0], ax=axes[row, 1])

        ax_d = axes[row, 2]
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

        ax_t = axes[row, 3]
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


if __name__ == "__main__":
    main()
