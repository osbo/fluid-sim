"""Inspect LeafOnly preconditioner: load weights, build M, compare A·M with AMG. Run with python3 InspectModel.py.

Also reports classical PCG baselines aligned with Yang et al., *Learning Sparse Approximate Inverse
Preconditioners for Conjugate Gradient Solvers on GPUs* (NeurIPS 2025;
`arXiv:2510.27517 <https://arxiv.org/abs/2510.27517>`_, code
`LearningSparsePreconditioner4GPU <https://github.com/Adversarr/LearningSparsePreconditioner4GPU>`_):
Diag (Jacobi), IC-class (``ilupp.IChol0`` or scipy ``spilu`` fallback), PyAMG / AMGX. On CUDA, the benchmark always runs a second AMGX session: PCG with ``MULTICOLOR_DILU`` (scalar CSR; AMGX ``MULTICOLOR_ILU`` GPU path is 4×4-only).

Use ``--test_only`` to run the PCG benchmark path and print only that summary (no plots, profiling, or dense AMG build).

H-matrix rank profiler (see ``_hmatrix_rank_profiler_bands``): builds the true dense A^-1, extracts
blocks aligned with ``HM_R0_CPU`` / ``HM_C0_CPU`` / ``HM_S_CPU``, runs SVD on sampled blocks, and
plots singular-value decay vs. ``leaf_apply_diag`` / ``leaf_apply_off``. If σ_k does not decay by
those indices, increase apply rank or decrease ``HMATRIX_ETA`` for smaller tiles.
"""
import argparse
import sys
import time
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Sparse CSR.*", category=UserWarning)

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

# Used before main() so optional-dependency import warnings stay quiet in test-only runs.
_TEST_ONLY_CLI = "--test_only" in sys.argv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, norm as sparse_norm, spilu

try:
    import pyamg
    HAS_AMG = True
except ImportError:
    HAS_AMG = False
    if not _TEST_ONLY_CLI:
        print("Warning: 'pyamg' not installed. AMG baseline will be skipped.")

try:
    import pyamgx
    HAS_AMGX = True
except (ImportError, OSError) as e:
    HAS_AMGX = False
    if not _TEST_ONLY_CLI:
        print("Warning: 'pyamgx' not available. AMGX (GPU) comparison will be skipped.", e)
import torch

import torch.nn.functional as F

from leafonly import (
    LeafOnlyNet,
    load_leaf_only_weights,
    apply_block_diagonal_m_into,
    block_diagonal_m_apply_workspace,
    build_sparse_bsr_preconditioner,
    default_attention_layout,
    unpack_precond,
    FluidGraphDataset,
    build_leaf_block_connectivity,
)
from leafonly.eval import _timed_ms
from leafonly.checkpoint import read_leaf_only_header
from leafonly.config import HMATRIX_ETA, LEAF_SIZE, MAX_MIXED_SIZE
from leafonly.hmatrix import (
    HM_C0_CPU,
    HM_R0_CPU,
    HM_S_CPU,
    NUM_HMATRIX_OFF_BLOCKS,
    standard_admissible_unique_blocks,
)


def _torch_bsr_to_scipy_csr_cpu(M: torch.Tensor) -> csr_matrix:
    """Expand CPU ``torch.sparse_bsr`` to SciPy CSR (``to_sparse_coo`` is broken on BSR in some PyTorch builds)."""
    crow = M.crow_indices().numpy()
    ccol = M.col_indices().numpy()
    vals = M.values().numpy()
    nrows, ncols = int(M.shape[0]), int(M.shape[1])
    n_br = crow.shape[0] - 1
    nnz_blk = ccol.shape[0]
    vt = M.values()
    if vt.dim() == 3:
        bh, bw = int(vt.shape[1]), int(vt.shape[2])
    else:
        # 1D packed blocks; PyTorch versions without ``Tensor.blocksize()`` (e.g. some 2.2–2.4 builds).
        bh = nrows // n_br if n_br else 0
        if n_br <= 0 or bh * n_br != nrows:
            raise ValueError(f"cannot infer BSR block height from shape=({nrows},{ncols}), n_block_rows={n_br}")
        ne = int(vt.numel())
        if nnz_blk == 0 or ne % nnz_blk != 0:
            raise ValueError("BSR values length inconsistent with col_indices")
        epb = ne // nnz_blk
        if epb % bh != 0:
            raise ValueError(f"BSR elements-per-block {epb} not divisible by inferred bh={bh}")
        bw = epb // bh
        if ncols % bw != 0:
            raise ValueError(f"BSR ncol {ncols} not divisible by inferred bw={bw}")
        vals = vals.reshape(nnz_blk, bh, bw)
    gr, gc = np.mgrid[0:bh, 0:bw]
    gr = gr.ravel()
    gc = gc.ravel()
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    dat_parts: list[np.ndarray] = []
    for br in range(n_br):
        r0 = br * bh
        for k in range(int(crow[br]), int(crow[br + 1])):
            c0 = int(ccol[k]) * bw
            row_parts.append(gr + r0)
            col_parts.append(gc + c0)
            dat_parts.append(vals[k].ravel())
    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    data = np.concatenate(dat_parts)
    return csr_matrix((data, (rows, cols)), shape=(nrows, ncols))


def _torch_sparse_precond_to_scipy_csr(M: torch.Tensor) -> csr_matrix:
    """Convert CPU ``torch.sparse_csr`` or ``torch.sparse_bsr`` to SciPy ``csr_matrix`` for analysis."""
    M = M.detach().cpu()
    if M.layout == torch.sparse_csr:
        crow = M.crow_indices().numpy()
        col = M.col_indices().numpy()
        val = M.values().numpy()
        return csr_matrix((val, col, crow), shape=tuple(M.shape))
    if M.layout == torch.sparse_bsr:
        return _torch_bsr_to_scipy_csr_cpu(M)
    raise TypeError(f"expected sparse_csr or sparse_bsr, got {M.layout}")


def verify_leafonly_preconditioner_spd(
    M_sparse_torch: torch.Tensor,
    viz_n: int,
    *,
    sym_rtol: float = 1e-3,
    min_eig_floor: float = -5e-2,
    n_dense_max: int = 3000,
    quad_samples: int = 48,
    print_fn=print,
) -> None:
    """
    Check whether the LeafOnly sparse preconditioner ``M`` (CSR or BSR, the PCG operator) is numerically
    symmetric and whether its symmetrization ``S = (M + M^T) / 2`` is positive definite enough for CG theory.

    - Symmetry: relative Frobenius ``||M - S||_F / ||M||_F``.
    - Spectrum of ``S``: full ``eigvalsh`` if ``viz_n <= n_dense_max``, else a few Lanczos steps via
      ``eigsh`` (may miss pathological interior modes).
    - Heuristic: fraction of random samples with ``v^T M v <= 0`` (necessary for true PD of the action
      on those directions).
    """
    M_t = M_sparse_torch.detach().cpu()
    if M_t.layout not in (torch.sparse_csr, torch.sparse_bsr):
        print_fn(f"  LeafOnly SPD check: skipped (expected sparse CSR/BSR, got {M_t.layout})")
        return

    M_sp = _torch_sparse_precond_to_scipy_csr(M_t)
    S = (M_sp + M_sp.T) * 0.5
    diff = M_sp - S
    try:
        nM = sparse_norm(M_sp, "fro")
        nd = sparse_norm(diff, "fro")
    except Exception:
        nM = float(np.sqrt(M_sp.power(2).sum()))
        nd = float(np.sqrt(diff.power(2).sum()))
    sym_rel = float(nd / (float(nM) + 1e-30))

    torch.manual_seed(0)
    neg_q = 0
    dtype = M_t.values().dtype
    for _ in range(quad_samples):
        v = torch.randn(viz_n, 1, dtype=dtype)
        # ``torch.sparse.mm`` does not support BSR on some PyTorch builds (IndexError on dim -2).
        v_np = v.detach().numpy()
        Mv_np = M_sp @ v_np
        q = float((v_np * Mv_np).sum())
        if q <= 0:
            neg_q += 1

    layout_tag = "BSR" if M_t.layout == torch.sparse_bsr else "CSR"
    print_fn(
        f"  LeafOnly preconditioner SPD check ({layout_tag} in PCG, n={viz_n}): "
        f"||M-S||_F/||M||_F={sym_rel:.3e} with S=(M+M^T)/2; "
        f"v^T M v ≤ 0 for {neg_q}/{quad_samples} random v"
    )

    lam_min = lam_max = float("nan")
    if viz_n <= n_dense_max:
        w = np.linalg.eigvalsh(S.toarray())
        lam_min, lam_max = float(w[0]), float(w[-1])
        cond_est = lam_max / max(abs(lam_min), 1e-30)
        print_fn(
            f"    Symmetrized S: λ_min={lam_min:.6e}, λ_max={lam_max:.6e}, λ_max/|λ_min|≈{cond_est:.3e}"
        )
    else:
        maxiter = min(8000, max(200, 40 * viz_n))
        try:
            lam_min = float(eigsh(S, k=1, which="SA", tol=1e-3, maxiter=maxiter, return_eigenvectors=False)[0])
        except Exception as ex:
            print_fn(f"    eigsh(S, which='SA') failed: {ex}")
        try:
            lam_max = float(eigsh(S, k=1, which="LA", tol=1e-3, maxiter=maxiter, return_eigenvectors=False)[0])
        except Exception as ex:
            print_fn(f"    eigsh(S, which='LA') failed: {ex}")
        if not np.isnan(lam_min) and not np.isnan(lam_max):
            print_fn(f"    Symmetrized S (Lanczos): λ_min≈{lam_min:.6e}, λ_max≈{lam_max:.6e}")

    sym_ok = sym_rel < sym_rtol
    eig_ok = not np.isnan(lam_min) and lam_min > min_eig_floor
    verdict = sym_ok and eig_ok
    quad_note = "all random v^T M v > 0" if neg_q == 0 else f"{neg_q} samples with v^T M v ≤ 0 (indefinite directions possible)"
    print_fn(
        "    Verdict: "
        + ("PASS " if verdict else "FAIL ")
        + f"for symmetrized S being near-SPD (sym_rel<{sym_rtol:g}, λ_min>{min_eig_floor:g}); "
        + quad_note
        + ". Standard PCG assumes a fixed SPD preconditioner; nonsymmetric M can still work in practice."
    )


def _draw_hmatrix_weak_partition_ax(ax, grid_size: int, leaf_size: int, eta: float) -> None:
    """
    Visualize the same weak-admissibility partition as ``leafonly.hmatrix.standard_admissible_unique_blocks``
    (LeafOnly / neural off-diagonal structure). Row 0 at top via ``ylim(grid, 0)``.
    """
    L = int(leaf_size)
    n = int(grid_size)
    if n % L != 0:
        raise ValueError(f"grid_size={n} must be divisible by leaf_size={L}")
    num_units = n // L
    u = standard_admissible_unique_blocks(num_units, float(eta), torch.device("cpu"), dtype=torch.float32)
    blocks = u.numpy()
    max_log = int(np.log2(num_units)) if num_units > 1 else 0
    green_cmap = plt.get_cmap("Greens")

    for row in blocks:
        r_leaf, c_leaf, size_leaves = int(row[0]), int(row[1]), int(row[2])
        y = r_leaf * L
        x = c_leaf * L
        size_px = size_leaves * L

        on_diag_unit = r_leaf == c_leaf and size_leaves == 1
        if on_diag_unit:
            face = "#e8a0c8"
            edge = "white"
            lw = 1.0
        else:
            k = int(np.log2(size_leaves)) if size_leaves > 1 else 0
            face = green_cmap(0.35 + 0.55 * (k / max(max_log, 1)))
            edge = "#1a3d1a"
            lw = 1.2

        ax.add_patch(
            Rectangle(
                (x, y),
                size_px,
                size_px,
                linewidth=lw,
                edgecolor=edge,
                facecolor=face,
                alpha=0.92,
            )
        )
        if size_px >= 64 and not on_diag_unit:
            ax.text(
                x + size_px / 2,
                y + size_px / 2,
                f"{size_px}",
                color="white",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                alpha=0.85,
            )

    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    tick_step = max(L * 4, 1)
    ticks = np.arange(0, n + 1, tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


def _hmatrix_rank_profiler_bands(
    A_inv: np.ndarray,
    leaf_L: int,
    num_leaves: int,
    leaf_apply_diag: int,
    leaf_apply_off: int,
    max_samples: int = 4,
    rng: np.random.Generator | None = None,
):
    """
    Extract true A^-1 blocks matching the LeafOnly HM tile layout; SVD each sampled block.

    Bands: on-diagonal leaf blocks, then off-diagonal blocks grouped by tile size S (leaf units).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    L = int(leaf_L)
    nu = int(num_leaves)
    n = int(A_inv.shape[0])
    if A_inv.shape != (n, n) or nu * L > n:
        raise ValueError("A_inv and leaf grid mismatch")

    bands: list[dict] = []

    on_idx = np.arange(nu)
    if len(on_idx) > max_samples:
        on_idx = rng.choice(on_idx, size=max_samples, replace=False)
    on_curves: list[np.ndarray] = []
    for li in on_idx:
        sl = slice(int(li) * L, (int(li) + 1) * L)
        B = A_inv[sl, sl]
        _, s, _ = np.linalg.svd(B, full_matrices=False)
        s = np.maximum(s, 0.0)
        on_curves.append(s / (s[0] + 1e-30))
    bands.append(
        {
            "key": "on_diag",
            "title": f"On-diagonal leaf\n{L}×{L}",
            "rank_line": int(leaf_apply_diag),
            "rank_label": f"leaf_apply_diag={leaf_apply_diag}",
            "curves": on_curves,
            "num_blocks_layout": int(nu),
        }
    )

    off_by_S: dict[int, list[tuple[int, int, int]]] = {}
    for i in range(int(NUM_HMATRIX_OFF_BLOCKS)):
        r0 = int(HM_R0_CPU[i].item())
        c0 = int(HM_C0_CPU[i].item())
        S = int(HM_S_CPU[i].item())
        if r0 + S > nu or c0 + S > nu:
            continue
        off_by_S.setdefault(S, []).append((r0, c0, S))

    for S in sorted(off_by_S.keys(), reverse=True):
        pairs = off_by_S[S]
        pix = S * L
        idxs = np.arange(len(pairs))
        if len(idxs) > max_samples:
            idxs = rng.choice(idxs, size=max_samples, replace=False)
        curves: list[np.ndarray] = []
        for j in idxs:
            r0, c0, _ = pairs[int(j)]
            rs = slice(r0 * L, (r0 + S) * L)
            cs = slice(c0 * L, (c0 + S) * L)
            B = A_inv[rs, cs]
            _, s, _ = np.linalg.svd(B, full_matrices=False)
            s = np.maximum(s, 0.0)
            curves.append(s / (s[0] + 1e-30))
        bands.append(
            {
                "key": f"off_S{S}",
                "title": f"Off-diagonal S={S}\n{pix}×{pix}",
                "rank_line": int(leaf_apply_off),
                "rank_label": f"leaf_apply_off={leaf_apply_off}",
                "curves": curves,
                "num_blocks_layout": len(pairs),
            }
        )

    return bands


def _print_hmatrix_rank_profiler(bands: list[dict], eta: float, max_plot_bands: int = 5) -> None:
    print("\n=== H-Matrix Rank Profiler ===")
    print(
        "True dense A^-1 only (no M); blocks from HM_R0_CPU / HM_C0_CPU (weak admissibility). "
        "Plots σ_k/σ_0 vs index; reports σ[r]/σ[0] at r = leaf_apply_* and σ_last/σ[0]. "
        "If σ_k/σ_0 at k = leaf_apply_* is not small, raise leaf_apply_* or decrease HMATRIX_ETA (smaller tiles)."
    )
    print(f"HMATRIX_ETA={eta}")
    if len(bands) > max_plot_bands:
        print(
            f"(inspect_model_plot.png row shows first {max_plot_bands} bands: on-diagonal then largest off-diagonal S; "
            f"{len(bands) - max_plot_bands} more band(s) listed below only.)\n"
        )
    else:
        print()
    for b in bands:
        curves: list[np.ndarray] = b["curves"]
        title_one_line = b["title"].replace("\n", " ")
        if not curves:
            print(f"{title_one_line}: (no blocks)\n")
            continue
        rl = int(b["rank_line"])
        ratios_past_cap: list[float] = []
        tail_ratios: list[float] = []
        for s_arr in curves:
            tail_ratios.append(float(s_arr[-1] / (s_arr[0] + 1e-30)))
            if len(s_arr) > rl:
                ratios_past_cap.append(float(s_arr[rl] / (s_arr[0] + 1e-30)))
        smax = max(len(c) for c in curves)
        nlay = int(b["num_blocks_layout"])
        print(f"Band: {title_one_line}")
        print(f"  Blocks in layout: {nlay}, sampled for SVD: {len(curves)}")
        print(f"  Max singular value count in band: {smax}")
        if ratios_past_cap:
            n_short = len(curves) - len(ratios_past_cap)
            extra = f" ({n_short} sample(s) with ≤{rl} SVs omitted)" if n_short else ""
            print(
                f"  σ[{rl}]/σ[0] past rank cap ({b['rank_label']}){extra}: "
                f"min={np.min(ratios_past_cap):.3e} mean={np.mean(ratios_past_cap):.3e} max={np.max(ratios_past_cap):.3e}"
            )
        else:
            print(
                f"  σ[{rl}]/σ[0]: N/A — each block has ≤{rl} singular values "
                f"(indices 0…{smax - 1}); rank cap does not truncate (see σ_last/σ[0])."
            )
        print(
            f"  σ_last/σ[0] (tail): "
            f"min={np.min(tail_ratios):.3e} mean={np.mean(tail_ratios):.3e} max={np.max(tail_ratios):.3e}"
        )
        print()


def _plot_hmatrix_rank_profiler_row(axes_row, bands_head: list[dict]) -> None:
    """One subplot per column; up to len(axes_row) bands (on-diag first, then off sizes descending)."""
    for ax in axes_row:
        ax.clear()
    n_col = len(axes_row)
    for j in range(n_col):
        ax = axes_row[j]
        if j >= len(bands_head):
            ax.axis("off")
            continue
        b = bands_head[j]
        curves = b["curves"]
        rl = int(b["rank_line"])
        if not curves:
            ax.text(0.5, 0.5, "No blocks", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(b["title"], fontsize=9)
            continue
        for k, srel in enumerate(curves):
            x = np.arange(len(srel))
            ax.semilogy(x, np.maximum(srel, 1e-16), alpha=0.75, linewidth=1.2, label=f"sample {k + 1}")
        if rl > 0 and any(len(c) > rl for c in curves):
            ax.axvline(rl, color="crimson", linestyle="--", linewidth=1.5, alpha=0.9, label=f"rank cap ({rl})")
        ax.set_xlabel("Singular value index")
        ax.set_ylabel(r"$\sigma_k / \sigma_0$")
        ax.set_title(f"H-matrix rank profiler\n{b['title']}", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)


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


def pcg_gpu_cudagraph_bsr(
    A_gpu,
    b_gpu,
    M_sparse_bsr,
    tol=1e-8,
    max_iter=500,
    device=None,
    check_freq=3,
):
    """
    PCG using sparse block-row ``M`` (BSR from ``build_sparse_bsr_preconditioner``) for ``z = M @ r``,
    with one iteration captured in a CUDA Graph.

    ``M_sparse_bsr`` must keep the same shape and block sparsity pattern across replays. ``A_gpu`` should
    be CSR for best performance. After capture, state is reset so replays start from the true initial iterate.
    """
    if device is None:
        device = b_gpu.device
    if device.type != "cuda":
        raise ValueError("pcg_gpu_cudagraph_bsr requires CUDA")

    dtype = b_gpu.dtype
    x_static = torch.zeros_like(b_gpu)
    r_static = b_gpu.clone()
    z_static = torch.zeros_like(b_gpu)
    z_static.copy_(M_sparse_bsr @ r_static)
    p_static = z_static.clone()
    rho_static = torch.zeros(1, device=device, dtype=dtype)
    rho_static.fill_((r_static * z_static).sum())

    tol_sq_val = (tol * tol) * (b_gpu * b_gpu).sum().item()

    def one_iter():
        Ap = A_gpu @ p_static
        pAp = (p_static * Ap).sum()
        alpha = rho_static / pAp
        x_static.add_(p_static * alpha)
        r_static.sub_(Ap * alpha)
        z_static.copy_(M_sparse_bsr @ r_static)
        rho_new = (r_static * z_static).sum()
        beta = rho_new / rho_static
        rho_static.fill_(rho_new)
        p_static.mul_(beta).add_(z_static)

    n = int(b_gpu.shape[0])
    # cuSPARSE may allocate on first SpMV per matrix; do that outside capture, immediately before graph.
    torch.cuda.synchronize()
    z_static.copy_(M_sparse_bsr @ r_static)
    if getattr(A_gpu, "layout", None) == torch.sparse_csr:
        _ = torch.sparse.mm(A_gpu, p_static.view(n, 1))
    else:
        _ = A_gpu @ p_static
    torch.cuda.synchronize()

    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            one_iter()
    except RuntimeError:
        return pcg_gpu(
            A_gpu,
            b_gpu,
            lambda r: M_sparse_bsr @ r,
            tol=tol,
            max_iter=max_iter,
            device=device,
            check_freq=check_freq,
        )

    # ``one_iter`` ran once during capture; restore true PCG initial state before replay.
    x_static.zero_()
    r_static.copy_(b_gpu)
    z_static.copy_(M_sparse_bsr @ r_static)
    p_static.copy_(z_static)
    rho_static.fill_((r_static * z_static).sum())

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


def pcg_gpu_cudagraph_matrix_free(
    A_gpu,
    b_gpu,
    apply_precond_into,
    tol=1e-8,
    max_iter=500,
    device=None,
    check_freq=3,
):
    """
    Matrix-free PCG ``z = M @ r`` with one iteration captured in a CUDA Graph.
    ``apply_precond_into(r, z_out)`` must write in-place into ``z_out`` without allocations.
    ``A_gpu`` should be CSR for best performance. After capture, state is reset so replays
    start from the true initial iterate.
    """
    if device is None:
        device = b_gpu.device
    if device.type != "cuda":
        raise ValueError("pcg_gpu_cudagraph_matrix_free requires CUDA")

    dtype = b_gpu.dtype
    x_static = torch.zeros_like(b_gpu)
    r_static = b_gpu.clone()
    z_static = torch.zeros_like(b_gpu)
    apply_precond_into(r_static, z_static)
    p_static = z_static.clone()
    rho_static = torch.zeros(1, device=device, dtype=dtype)
    rho_static.fill_((r_static * z_static).sum())

    tol_sq_val = (tol * tol) * (b_gpu * b_gpu).sum().item()

    def one_iter():
        Ap = A_gpu @ p_static
        pAp = (p_static * Ap).sum()
        alpha = rho_static / pAp
        x_static.add_(p_static * alpha)
        r_static.sub_(Ap * alpha)
        apply_precond_into(r_static, z_static)
        rho_new = (r_static * z_static).sum()
        beta = rho_new / rho_static
        rho_static.fill_(rho_new)
        p_static.mul_(beta).add_(z_static)

    n = int(b_gpu.shape[0])
    # cuSPARSE may allocate on first SpMV per matrix; do that outside capture, immediately before graph.
    torch.cuda.synchronize()
    apply_precond_into(r_static, z_static)
    if getattr(A_gpu, "layout", None) == torch.sparse_csr:
        _ = torch.sparse.mm(A_gpu, p_static.view(n, 1))
    else:
        _ = A_gpu @ p_static
    torch.cuda.synchronize()

    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            one_iter()
    except RuntimeError:
        def _apply_precond_fallback(r: torch.Tensor) -> torch.Tensor:
            z = torch.empty_like(r)
            return apply_precond_into(r, z)

        return pcg_gpu(
            A_gpu,
            b_gpu,
            _apply_precond_fallback,
            tol=tol,
            max_iter=max_iter,
            device=device,
            check_freq=check_freq,
        )

    # ``one_iter`` ran once during capture; restore true PCG initial state before replay.
    x_static.zero_()
    r_static.copy_(b_gpu)
    apply_precond_into(r_static, z_static)
    p_static.copy_(z_static)
    rho_static.fill_((r_static * z_static).sum())

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


def _cpu_jacobi_apply_fn(A_dense_f64: np.ndarray):
    inv = 1.0 / np.maximum(np.abs(np.diag(A_dense_f64).astype(np.float64)), 1e-30)

    def apply_j(r: np.ndarray) -> np.ndarray:
        return (np.asarray(r, dtype=np.float64).ravel() * inv).reshape(-1, 1)

    return apply_j


def _gpu_jacobi_apply_fn(A_dense_f64: np.ndarray, device: torch.device):
    inv = (1.0 / np.maximum(np.abs(np.diag(A_dense_f64).astype(np.float64)), 1e-30)).astype(np.float32)
    inv_t = torch.from_numpy(inv).to(device=device)

    def apply_j(r: torch.Tensor) -> torch.Tensor:
        return r * inv_t.unsqueeze(1)

    return apply_j


def _build_cpu_ic_apply(A_dense_f64: np.ndarray):
    """
    Incomplete-Cholesky-class preconditioner for SPD-ish systems: prefer ``ilupp.IChol0Preconditioner``,
    else scipy ``spilu`` (ILU, common fallback when IChol is unavailable).
    Returns (setup_ms, apply_fn_or_none, backend_label_or_none).
    """
    A_csr = csr_matrix(A_dense_f64)
    A_csr.sort_indices()
    t0 = time.perf_counter()
    try:
        import ilupp

        prec = ilupp.IChol0Preconditioner(A_csr)
        setup_ms = (time.perf_counter() - t0) * 1000

        def apply_ic(r: np.ndarray) -> np.ndarray:
            z = prec @ np.asarray(r, dtype=np.float64).ravel()
            return z.reshape(-1, 1)

        return setup_ms, apply_ic, "ilupp IChol0"
    except Exception:
        pass
    try:
        t0 = time.perf_counter()
        lu = spilu(
            A_csr.tocsc(),
            drop_tol=1e-8,
            fill_factor=30,
            permc_spec="COLAMD",
            diag_pivot_thresh=0.0,
        )
        setup_ms = (time.perf_counter() - t0) * 1000

        def apply_ilu(r: np.ndarray) -> np.ndarray:
            return lu.solve(np.asarray(r, dtype=np.float64).ravel()).reshape(-1, 1)

        return setup_ms, apply_ilu, "scipy spilu (ILU fallback)"
    except Exception:
        return (time.perf_counter() - t0) * 1000, None, None


def _amgx_safe_destroy(obj) -> None:
    if obj is None:
        return
    try:
        obj.destroy()
    except Exception:
        pass


def _run_amgx_pcg_session(
    A_amgx_csr: csr_matrix,
    b_flat: np.ndarray,
    x_init: np.ndarray,
    pcg_tol: float,
    pcg_max_iter: int,
    preconditioner_block: dict,
    *,
    num_warmup: int = 3,
    warn_fn=None,
    session_label: str = "AMGX",
) -> tuple[float, float, int]:
    """One full pyamgx initialize → PCG → finalize cycle. Returns (setup_ms, solve_ms, iterations)."""
    cfg = None
    rsc = None
    A_amgx = None
    b_amgx = None
    x_amgx = None
    solver = None
    initialized = False

    def _finalize_amgx_lib() -> None:
        _stderr_fd = os.dup(2)
        _devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(_devnull, 2)
            pyamgx.finalize()
        finally:
            os.dup2(_stderr_fd, 2)
            os.close(_stderr_fd)
            os.close(_devnull)

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
        initialized = True

        cfg = pyamgx.Config().create_from_dict(
            {
                "config_version": 2,
                "determinism_flag": 1,
                "exception_handling": 1,
                "solver": {
                    "solver": "PCG",
                    "max_iters": pcg_max_iter,
                    "convergence": "RELATIVE_INI",
                    "tolerance": float(pcg_tol),
                    "monitor_residual": 1,
                    "preconditioner": preconditioner_block,
                },
            }
        )
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
        torch.cuda.synchronize()
        setup_ms = (time.perf_counter() - t0) * 1000

        for _ in range(num_warmup):
            torch.cuda.synchronize()
            x_amgx.upload(x_init)
            solver.solve(b_amgx, x_amgx)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        x_amgx.upload(x_init)
        solver.solve(b_amgx, x_amgx)
        torch.cuda.synchronize()
        solve_ms = (time.perf_counter() - t0) * 1000
        iters = int(solver.iterations_number)

        _amgx_safe_destroy(solver)
        solver = None
        _amgx_safe_destroy(x_amgx)
        x_amgx = None
        _amgx_safe_destroy(b_amgx)
        b_amgx = None
        _amgx_safe_destroy(A_amgx)
        A_amgx = None
        _amgx_safe_destroy(rsc)
        rsc = None
        _amgx_safe_destroy(cfg)
        cfg = None
        _finalize_amgx_lib()
        initialized = False
        return setup_ms, solve_ms, iters
    except Exception as e:
        if warn_fn is not None:
            warn_fn(f"Warning: AMGX ({session_label}) failed: {e}")
        _amgx_safe_destroy(solver)
        _amgx_safe_destroy(x_amgx)
        _amgx_safe_destroy(b_amgx)
        _amgx_safe_destroy(A_amgx)
        _amgx_safe_destroy(rsc)
        _amgx_safe_destroy(cfg)
        if initialized:
            try:
                _finalize_amgx_lib()
            except Exception:
                pass
        return 0.0, 0.0, 0


def main():
    parser = argparse.ArgumentParser(description="Inspect LeafOnly preconditioner and compare solvers.")
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only print the PCG benchmark summary; skip plots, spectral/HM profiling, and dense AMG build.",
    )
    parser.add_argument(
        "--leafonly_pcg",
        choices=("bsr", "matrix_free"),
        default="bsr",
        help=(
            "LeafOnly PCG preconditioner apply: bsr = materialized BSR + SpMV (default); "
            "matrix_free = apply_block_diagonal_m_into with preallocated workspace (scales for large H off-diagonal)."
        ),
    )
    args = parser.parse_args()
    test_only = bool(args.test_only)

    def _info(*a, **k):
        if not test_only:
            print(*a, **k)

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
    _info(f"Using device: {device}")
    check_freq = 3

    _info(f"Loading data from {data_folder}")
    dataset = FluidGraphDataset([Path(data_folder)])
    if len(dataset) == 0:
        raise SystemExit("No frames found (need nodes.bin, edge_index_*.bin, A_values.bin).")
    frame_idx = min(frame_idx, len(dataset) - 1)
    batch = dataset[frame_idx]

    x = batch['x'].unsqueeze(0).to(device)
    num_nodes_real = int(batch['num_nodes'])
    _info(f"System N={num_nodes_real}")

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
        leaf_apply_diag_ckpt,
        leaf_apply_off_ckpt,
        _attention_layout_code,
    ) = read_leaf_only_header(leaf_only_weights_path)
    if int(leaf_size_lo) != int(LEAF_SIZE):
        raise ValueError(
            f"Checkpoint leaf_size={leaf_size_lo} != leafonly.config.LEAF_SIZE={LEAF_SIZE}. "
            "Set LEAF_SIZE in leafonly/config.py to match the weights header so MAX_MIXED_SIZE and MAX_NUM_LEAVES stay consistent."
        )
    leaf_L = int(LEAF_SIZE)
    n_pad = int(MAX_MIXED_SIZE)
    n_requested = (min(MAX_MIXED_SIZE, num_nodes_real) // leaf_L) * leaf_L
    # Physical linear system for PCG / baselines / plots (no identity tail): leaf-aligned, ≤ MAX_MIXED_SIZE.
    viz_n = n_requested
    if n_requested < num_nodes_real:
        _info(
            f"  Leaf-aligned subgraph: {n_requested} nodes (frame has {num_nodes_real}; "
            f"truncated to full {leaf_L}-node leaves); LeafOnly forward padded to {n_pad} (MAX_MIXED_SIZE)"
        )
    elif n_pad > n_requested:
        _info(
            f"  Physical system {n_requested}×{n_requested}; LeafOnly forward padded to {n_pad} (MAX_MIXED_SIZE)"
        )

    ei, ev = batch["edge_index"], batch["edge_values"]
    # Single mask for the n_requested subgraph — reused for A, GPU edges, and connectivity.
    em = (ei[0] < n_requested) & (ei[1] < n_requested)
    A_small = torch.sparse_coo_tensor(ei[:, em], ev[em], (n_requested, n_requested)).coalesce().to_dense().numpy()
    A_phys_f64 = A_small.astype(np.float64, copy=False)
    # Diagonal of expanded training-style A (identity on padded tail) for Jacobi inside full n_pad apply.
    jacobi_diag_np = np.ones(n_pad, dtype=np.float32)
    jacobi_diag_np[:n_requested] = np.diag(A_small).astype(np.float32, copy=False)

    _info("\nLeafOnly (GPU)...")
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
    leaf_apply_diag_L = int(model_leaf.leaf_apply_size)
    leaf_apply_off_L = int(model_leaf.leaf_apply_off)
    if leaf_apply_diag_L != int(leaf_apply_diag_ckpt):
        raise RuntimeError(f"header leaf_apply_diag {leaf_apply_diag_ckpt} != model {leaf_apply_diag_L}")
    if leaf_apply_off_L != int(leaf_apply_off_ckpt):
        raise RuntimeError(f"header leaf_apply_off {leaf_apply_off_ckpt} != model {leaf_apply_off_L}")
    pool_diag_to_full = int(leaf_L) // leaf_apply_diag_L
    pool_off_to_full = int(leaf_L) // leaf_apply_off_L

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
        # Diagonal leaf masks/feats; off-diagonal H tiles use static buffers inside LeafOnlyNet.forward.
        pre_leaf_connectivity = build_leaf_block_connectivity(
            edge_index_leaf,
            edge_values_leaf,
            positions_leaf,
            leaf_L,
            device,
            x_leaf.dtype,
        )
        pre_leaf_connectivity = tuple(
            t.contiguous() if isinstance(t, torch.Tensor) else t for t in pre_leaf_connectivity
        )

        _last_out = [None]

        def _fwd():
            _last_out[0] = model_leaf(
                x_leaf,
                edge_index=edge_index_leaf,
                edge_values=edge_values_leaf,
                global_features=global_feat,
                precomputed_leaf_connectivity=pre_leaf_connectivity,
            )

        # Jacobi diagonal for BSR / padded apply (from true A diag only; independent of precond unpack).
        jacobi_inv_diag = torch.ones(1, n_pad, device=device, dtype=x_leaf.dtype)
        diag_A_tensor = torch.from_numpy(jacobi_diag_np).to(device)
        diag_mask = diag_A_tensor.abs() > 1e-6
        jacobi_inv_diag[0, diag_mask] = 1.0 / diag_A_tensor[diag_mask]

        # Prime torch.compile / CUDA allocator; Inductor kernels also land in TORCHINDUCTOR_CACHE_DIR (see above).
        for _ in range(15):
            _fwd()
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Time pure forward only (same idea as leafonly.eval ``--evaluate_gradients``). Running
        # ``build_sparse_bsr_preconditioner`` (when ``--leafonly_pcg bsr``) or other large materialization
        # first allocates big buffers and can leave the GPU in a different memory state, skewing this line.
        inference_ms = _timed_ms(_fwd, device, warmup=0, repeat=10)
        precond_out = _last_out[0]

        if args.leafonly_pcg == "bsr":
            # First BSR build: fills ``_BSR_TOPOLOGY_CACHE`` for the later PCG path (not part of inference_ms).
            build_sparse_bsr_preconditioner(
                precond_out.detach().contiguous(),
                viz_n,
                leaf_L,
                leaf_apply_diag_L,
                leaf_apply_off_L,
                jacobi_inv_diag[:, :viz_n].contiguous(),
                device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()

        diag_blocks, off_diag_blocks, node_U, node_V, jacobi_scale = unpack_precond(
            precond_out,
            n_pad,
            leaf_size=leaf_L,
            leaf_apply_size=leaf_apply_diag_L,
            leaf_apply_off=leaf_apply_off_L,
        )
        num_leaves = n_pad // leaf_L
        M_neural_gpu = torch.zeros((n_pad, n_pad), dtype=torch.float32, device=device)
        for b in range(num_leaves):
            r0, r1 = b * leaf_L, (b + 1) * leaf_L
            blk = diag_blocks[0, b]
            if pool_diag_to_full > 1:
                blk = blk.repeat_interleave(pool_diag_to_full, dim=0).repeat_interleave(pool_diag_to_full, dim=1)
            M_neural_gpu[r0:r1, r0:r1] = blk

        if NUM_HMATRIX_OFF_BLOCKS > 0 and off_diag_blocks is not None and node_U is not None and node_V is not None:
            for i in range(NUM_HMATRIX_OFF_BLOCKS):
                r0i = int(HM_R0_CPU[i].item())
                c0i = int(HM_C0_CPU[i].item())
                si = int(HM_S_CPU[i].item())
                br0, br1 = r0i * leaf_L, (r0i + si) * leaf_L
                bc0, bc1 = c0i * leaf_L, (c0i + si) * leaf_L
                C_m = off_diag_blocks[0, i]
                U_strip = node_U[0, r0i : r0i + si].reshape(si * leaf_L, leaf_apply_off_L)
                V_strip = node_V[0, c0i : c0i + si].reshape(si * leaf_L, leaf_apply_off_L)
                U_C = torch.matmul(U_strip, C_m)
                oblk_dense = torch.matmul(U_C, V_strip.transpose(0, 1))
                M_neural_gpu[br0:br1, bc0:bc1] = oblk_dense
                M_neural_gpu[bc0:bc1, br0:br1] = oblk_dense.transpose(-1, -2)

        if jacobi_scale is not None:
            M_neural_gpu += torch.diag((jacobi_scale[0] * jacobi_inv_diag[0]).to(M_neural_gpu.dtype))
    _info(
        f"  LeafOnly: static grid {n_pad}×{n_pad} ({num_leaves} leaves); "
        f"assembled M sliced to physical {viz_n}×{viz_n} for benchmarks "
        f"(diag {leaf_apply_diag_L}×{leaf_apply_diag_L} ×{pool_diag_to_full}, "
        f"off {leaf_apply_off_L}×{leaf_apply_off_L} ×{pool_off_to_full})"
    )

    d = diag_blocks.detach()
    if d.dim() == 3:
        d = d.unsqueeze(0)
    frobs = (d ** 2).sum(dim=(-2, -1)).sqrt()
    flat = d.reshape(-1)
    _info(f"  Diagonal blocks: Frob mean={frobs.mean().item():.6f} std={frobs.std().item():.6f}")
    _info(f"  Elements: mean={flat.mean().item():.6e} std={flat.std().item():.6e} min={flat.min().item():.6e} max={flat.max().item():.6e}")

    A_scipy = csr_matrix(A_phys_f64)
    ml_amg = None
    amg_setup_ms = 0.0
    if HAS_AMG:
        t0 = time.perf_counter()
        ml_amg = _amg_solver(A_scipy)
        amg_setup_ms = (time.perf_counter() - t0) * 1000

    if not test_only:
        M_amg = get_dense_amg(A_scipy, viz_limit=viz_n, tol=1e-6, progress_interval=200, ml=ml_amg)

    A_viz_n = A_phys_f64
    assert A_viz_n.shape[0] == viz_n and A_viz_n.shape[1] == viz_n
    M_gpu = M_neural_gpu[:viz_n, :viz_n]
    if not test_only:
        M_amg_n = M_amg[:viz_n, :viz_n]

    pcg_tol = 1e-8
    pcg_max_iter = 5000
    np.random.seed(123)
    b_np = np.random.randn(viz_n).astype(np.float64)
    b_np = b_np / (np.linalg.norm(b_np) + 1e-12)
    b_np = b_np.reshape(-1, 1)

    A_scipy_csr = csr_matrix(A_viz_n.astype(np.float32))

    precond_s = precond_out.detach().contiguous()
    jacobi_s = jacobi_inv_diag.detach().contiguous()
    jacobi_s_phys = jacobi_s[:, :viz_n].contiguous()

    A_f64 = A_viz_n.astype(np.float64)
    leaf_rel_res = float("nan")
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

        apply_diag_gpu = _gpu_jacobi_apply_fn(A_f64, device)
        _, iters_diag_gpu, solve_diag_gpu_ms = pcg_gpu(
            A_gpu,
            b_gpu,
            apply_diag_gpu,
            tol=pcg_tol,
            max_iter=pcg_max_iter,
            device=device,
            check_freq=check_freq,
        )

        if test_only:
            print(f"  LeafOnly PCG mode: {args.leafonly_pcg}")
        else:
            _info(f"  LeafOnly PCG mode: {args.leafonly_pcg}")

        if args.leafonly_pcg == "bsr":
            M_sparse_bsr = build_sparse_bsr_preconditioner(
                precond_s,
                viz_n,
                leaf_L,
                leaf_apply_diag_L,
                leaf_apply_off_L,
                jacobi_s_phys,
                device,
            )
            verify_leafonly_preconditioner_spd(M_sparse_bsr, viz_n, print_fn=_info)

            if device.type == "cuda":
                x_gpu, iters_leaf, solve_leaf_ms = pcg_gpu_cudagraph_bsr(
                    A_gpu,
                    b_gpu,
                    M_sparse_bsr,
                    tol=pcg_tol,
                    max_iter=pcg_max_iter,
                    device=device,
                    check_freq=check_freq,
                )
                _info(
                    "  LeafOnly PCG: CUDAGraph + BSR preconditioner (z.copy_(M @ r))"
                    + (f"; n={viz_n}" if viz_n != n_pad else "")
                )
            else:

                def apply_pcg_fast(r):
                    return M_sparse_bsr @ r

                x_gpu, iters_leaf, solve_leaf_ms = pcg_gpu(
                    A_gpu,
                    b_gpu,
                    apply_pcg_fast,
                    tol=pcg_tol,
                    max_iter=pcg_max_iter,
                    device=device,
                    check_freq=check_freq,
                )
                _info("  LeafOnly PCG: BSR preconditioner + pcg_gpu (no CUDA graph on this device)")
        else:
            precond_ws = block_diagonal_m_apply_workspace(
                num_leaves=viz_n // leaf_L,
                leaf_size=leaf_L,
                K_dim=1,
                M_h=NUM_HMATRIX_OFF_BLOCKS,
                La_o=leaf_apply_off_L,
                device=device,
                dtype=b_gpu.dtype,
            )

            def apply_pcg_fast_into(r: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
                return apply_block_diagonal_m_into(
                    precond_s,
                    r.reshape(1, viz_n, 1),
                    out.reshape(1, viz_n, 1),
                    jacobi_s_phys,
                    precond_ws,
                    leaf_size=leaf_L,
                    leaf_apply_size=leaf_apply_diag_L,
                    leaf_apply_off=leaf_apply_off_L,
                ).reshape(viz_n, 1)

            if device.type == "cuda":
                x_gpu, iters_leaf, solve_leaf_ms = pcg_gpu_cudagraph_matrix_free(
                    A_gpu,
                    b_gpu,
                    apply_pcg_fast_into,
                    tol=pcg_tol,
                    max_iter=pcg_max_iter,
                    device=device,
                    check_freq=check_freq,
                )
                _info(
                    "  LeafOnly PCG: CUDAGraph + matrix-free apply_block_diagonal_m_into preconditioner"
                    + (f"; n={viz_n}" if viz_n != n_pad else "")
                )
            else:

                def apply_pcg_fast(r):
                    z = torch.empty_like(r)
                    return apply_pcg_fast_into(r, z)

                x_gpu, iters_leaf, solve_leaf_ms = pcg_gpu(
                    A_gpu,
                    b_gpu,
                    apply_pcg_fast,
                    tol=pcg_tol,
                    max_iter=pcg_max_iter,
                    device=device,
                    check_freq=check_freq,
                )
                _info("  LeafOnly PCG: matrix-free preconditioner + pcg_gpu (no CUDA graph on this device)")

        _bf = b_gpu.squeeze(-1)
        _Ax = A_gpu @ x_gpu.squeeze(-1)
        leaf_rel_res = float(((_Ax - _bf).norm() / (_bf.norm() + 1e-30)).item())

    x_cpu_none, iters_none_cpu, solve_none_cpu_ms = pcg_cpu(
        A_f64, b_np, lambda r: r, tol=pcg_tol, max_iter=pcg_max_iter
    )

    setup_diag_cpu_ms = 0.0
    setup_diag_gpu_ms = 0.0
    apply_diag_cpu = _cpu_jacobi_apply_fn(A_f64)
    _, iters_diag_cpu, solve_diag_cpu_ms = pcg_cpu(
        A_f64, b_np, apply_diag_cpu, tol=pcg_tol, max_iter=pcg_max_iter
    )

    ic_setup_ms, ic_apply, ic_backend = _build_cpu_ic_apply(A_f64)
    if ic_apply is not None:
        _, iters_ic_cpu, solve_ic_cpu_ms = pcg_cpu(A_f64, b_np, ic_apply, tol=pcg_tol, max_iter=pcg_max_iter)
    else:
        iters_ic_cpu = 0
        solve_ic_cpu_ms = 0.0

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
            return z_full.reshape(-1, 1).astype(np.float64)
        x_cpu, iters_amg, solve_amg_ms = pcg_cpu(A_viz_n.astype(np.float64), b_np, apply_M_amg, tol=pcg_tol, max_iter=pcg_max_iter)
    total_amg_ms = amg_setup_ms + solve_amg_ms

    amgx_ilu_setup_ms = 0.0
    amgx_ilu_solve_ms = 0.0
    amgx_ilu_iters = 0
    amgx_amg_setup_ms = 0.0
    amgx_amg_solve_ms = 0.0
    amgx_amg_iters = 0
    if HAS_AMGX and device.type == 'cuda':
        A_amgx_csr = csr_matrix(A_viz_n.astype(np.float64))
        A_amgx_csr.indptr = np.ascontiguousarray(A_amgx_csr.indptr.astype(np.int32))
        A_amgx_csr.indices = np.ascontiguousarray(A_amgx_csr.indices.astype(np.int32))
        A_amgx_csr.data = np.ascontiguousarray(A_amgx_csr.data.astype(np.float64))

        b_flat = np.ascontiguousarray(b_np.ravel().astype(np.float64))
        x_init = np.zeros(viz_n, dtype=np.float64)

        # MULTICOLOR_ILU GPU LU only supports 4×4 blocks in AMGX; scalar CSR → MULTICOLOR_DILU.
        amgx_ilu_setup_ms, amgx_ilu_solve_ms, amgx_ilu_iters = _run_amgx_pcg_session(
            A_amgx_csr,
            b_flat,
            x_init,
            pcg_tol,
            pcg_max_iter,
            {
                "scope": "precond",
                "solver": "MULTICOLOR_DILU",
                "coloring_level": 2,
                "matrix_coloring_scheme": "MIN_MAX",
                "max_uncolored_percentage": 0.0,
            },
            warn_fn=_info,
            session_label="PCG+MULTICOLOR_DILU",
        )
        amgx_amg_setup_ms, amgx_amg_solve_ms, amgx_amg_iters = _run_amgx_pcg_session(
            A_amgx_csr,
            b_flat,
            x_init,
            pcg_tol,
            pcg_max_iter,
            {
                "solver": "AMG",
                "algorithm": "AGGREGATION",
                "selector": "SIZE_2",
                "cycle": "V",
                "smoother": "MULTICOLOR_GS",
                "presweeps": 1,
                "postsweeps": 1,
            },
            warn_fn=_info,
            session_label="PCG+AMG",
        )
    total_amgx_ilu_ms = amgx_ilu_setup_ms + amgx_ilu_solve_ms
    total_amgx_amg_ms = amgx_amg_setup_ms + amgx_amg_solve_ms

    total_diag_cpu_ms = setup_diag_cpu_ms + solve_diag_cpu_ms
    total_diag_gpu_ms = setup_diag_gpu_ms + solve_diag_gpu_ms
    total_ic_cpu_ms = ic_setup_ms + solve_ic_cpu_ms

    print("\nPCG solve A x = b (relative residual tol={:.0e})".format(pcg_tol))
    print("Unpreconditioned (CG):")
    print(f"  Setup: {setup_none_ms:.2f} ms, solve (GPU): {solve_none_gpu_ms:.2f} ms, {iters_none_gpu} iterations, total: {total_none_gpu_ms:.2f} ms")
    print(f"  Setup: {setup_none_ms:.2f} ms, solve (CPU): {solve_none_cpu_ms:.2f} ms, {iters_none_cpu} iterations, total: {total_none_cpu_ms:.2f} ms")
    print("Diag (Jacobi; paper Table 7: Eigen CPU / custom GPU):")
    print(f"  Setup: {setup_diag_cpu_ms:.2f} ms, solve (CPU): {solve_diag_cpu_ms:.2f} ms, {iters_diag_cpu} iterations, total: {total_diag_cpu_ms:.2f} ms")
    print(f"  Setup: {setup_diag_gpu_ms:.2f} ms, solve (GPU): {solve_diag_gpu_ms:.2f} ms, {iters_diag_gpu} iterations, total: {total_diag_gpu_ms:.2f} ms")
    print("IC (paper: Eigen CPU / cuSPARSE GPU; here ilupp or scipy CPU; GPU line uses AMGX MULTICOLOR_DILU for scalar CSR):")
    if ic_apply is not None:
        print(
            f"  Setup: {ic_setup_ms:.2f} ms, solve (CPU): {solve_ic_cpu_ms:.2f} ms, {iters_ic_cpu} iterations, "
            f"total: {total_ic_cpu_ms:.2f} ms  [{ic_backend}]"
        )
    else:
        print(
            f"  Setup: {ic_setup_ms:.2f} ms, solve (CPU): {solve_ic_cpu_ms:.2f} ms, {iters_ic_cpu} iterations "
            f"(IC factorization unavailable; try pip install ilupp)"
        )
    if HAS_AMGX and device.type == "cuda":
        print(
            f"  Setup: {amgx_ilu_setup_ms:.2f} ms, solve (GPU): {amgx_ilu_solve_ms:.2f} ms, {amgx_ilu_iters} iterations, "
            f"total: {total_amgx_ilu_ms:.2f} ms  [AMGX MULTICOLOR_DILU]"
        )
    print("LeafOnly:")
    print(
        f"  Inference: {inference_ms:.2f} ms, solve: {solve_leaf_ms:.2f} ms, {iters_leaf} iterations, "
        f"total: {total_leaf_ms:.2f} ms; true rel residual ||Ax-b||/||b||={leaf_rel_res:.3e} (PCG tol {pcg_tol:.0e})"
    )
    print("AMG (CPU):")
    print(f"  Setup: {amg_setup_ms:.2f} ms, solve: {solve_amg_ms:.2f} ms, {iters_amg} iterations, total: {total_amg_ms:.2f} ms")
    if HAS_AMGX and device.type == "cuda":
        print("AMGX (GPU) [PCG + AMG]:")
        print(
            f"  Setup: {amgx_amg_setup_ms:.2f} ms, solve: {amgx_amg_solve_ms:.2f} ms, {amgx_amg_iters} iterations, "
            f"total: {total_amgx_amg_ms:.2f} ms"
        )

    if test_only:
        return

    A_inv_viz = np.linalg.inv(A_viz_n)
    diag_ainv = np.diag(A_inv_viz)
    print(f"\nTrue inverse A^{{-1}} diagonal (viz {viz_n}x{viz_n}): min={diag_ainv.min():.6f}, max={diag_ainv.max():.6f}, mean={diag_ainv.mean():.6f}")

    cond_A = np.linalg.cond(A_viz_n)
    print(f"Condition number (block A): {cond_A:.2e}")
    print(f"Leaf boundaries: every {leaf_L}")

    num_leaves_viz = viz_n // leaf_L
    hm_rank_bands = _hmatrix_rank_profiler_bands(
        A_inv_viz,
        leaf_L,
        num_leaves_viz,
        leaf_apply_diag_L,
        leaf_apply_off_L,
        max_samples=4,
        rng=np.random.default_rng(123),
    )
    _print_hmatrix_rank_profiler(hm_rank_bands, HMATRIX_ETA)

    PLOT_MATRICES = True

    if PLOT_MATRICES:
        M_neural_n = M_gpu.detach().cpu().numpy()
        methods = [("LeafOnly", M_neural_n), ("AMG", M_amg_n)]
        n_cols = 5
        n_rows = 1 + len(methods) + 1
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
        _draw_hmatrix_weak_partition_ax(ax_blk, viz_n, leaf_L, HMATRIX_ETA)
        ax_blk.set_title(
            f"Weak-admissible H-matrix (η={HMATRIX_ETA}), LeafOnly layout\n"
            f"Grid {viz_n}×{viz_n}, leaf {leaf_L}×{leaf_L}, K={viz_n // leaf_L} units"
        )
        _green_cmap = plt.get_cmap("Greens")
        ax_blk.legend(
            handles=[
                Patch(
                    facecolor="#e8a0c8",
                    edgecolor="white",
                    linewidth=1.0,
                    label=f"On-diagonal unit leaf ({leaf_L}×{leaf_L} px)",
                ),
                Patch(
                    facecolor=_green_cmap(0.65),
                    edgecolor="#1a3d1a",
                    linewidth=1.2,
                    label="Merged admissible tile",
                ),
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

        _plot_hmatrix_rank_profiler_row(axes[n_rows - 1, :], hm_rank_bands[:n_cols])

        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved to: {out_path}")
    else:
        print("\nSkipped matrix dense plotting/eigenvalues to save time.")


if __name__ == "__main__":
    main()
