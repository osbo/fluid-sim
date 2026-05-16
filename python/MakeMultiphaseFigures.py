"""Generate paper figures from the multiphase_v2 benchmark data.

Subfigures (one per --fig flag, or --fig all):
    hero        — 3-panel strip: density rho, right-hand side b, pressure + velocity quiver.
    variety     — 4x6 grid sampling topology / contrast / orientation axes, with N, ratio, kappa.
    matrices    — 1×3 three-view: log₁₀|A|, peak-relative log₁₀|A⁻¹|, peak-relative log₁₀|M|
                  (top-left 256² by default). Companion to the rank-headroom figure;
                  pinned to the hero frame so the matrix structure corresponds to the initial
                  problem the reader has already seen.
    convergence — Two-column-wide multi-method panel on a *different* frame from hero/matrices:
                  top strip = density rho and RHS b; below, 5 method rows (unpreconditioned,
                  Jacobi, IC, AMG, ours) × 5 |r_k| snapshots at geometrically-spaced PCG
                  iterations (log-spaced from 1 to ~0.9*K of each method's own run). All
                  residual panels share one log colorbar so spatial spread of the error
                  (information propagation) is directly comparable across methods.
    residual-3d-compare — 3D Fig-1c-style panel: top problem row (rho, |b|, |p|, |v| + white
                  traces, NeuralSPAI M) and method rows (Jacobi, AMG, NeuralSPAI) with
                  particle-based |r_k| snapshots at geometric iterations until convergence.

Run on the slurm GPU node so the `fluid` conda env (numpy/scipy/pyamg/ilupp/matplotlib/torch)
is available. matrices and convergence require --ours-weights (trained .bytes matching the
frame scale).

Usage:
    python3 MakeMultiphaseFigures.py --fig all \
        --data-root data \
        --out-dir ../../Paper/figures \
        --ours-weights /path/to/v2_2048_d128_L3_hw.bytes
"""

import argparse
import csv
import math
import os
import sys
import time
import warnings
from typing import Optional
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*torch\._prims_common\.check.*", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm, Normalize, BoundaryNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def _apply_paper_style() -> None:
    """Serif fonts and sizes aligned with ``plot_eval_results.py`` / ``plot_training.py`` (paper.tex figures)."""
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "figure.titlesize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )


def _apply_paper_style_compact() -> None:
    """Larger relative fonts for figures that paper.tex renders at single-column width.

    The hero, variety, and convergence figures are saved at roughly half the canvas size
    of a full-width figure (~5--7 in instead of ~11--14 in). When LaTeX scales the PNG
    down to single-column width (~3.3 in), font sizes shrink by the same factor. Bumping
    the matplotlib font sizes here keeps the final on-page text at a readable ~6.5--8 pt.
    """
    _apply_paper_style()
    matplotlib.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13.5,
            "figure.titlesize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


def _savefig_paper(fig, out_path):
    fig.savefig(out_path, dpi=int(matplotlib.rcParams.get("savefig.dpi", 300)), bbox_inches="tight")


SCRIPTS_DIR = Path(__file__).resolve().parent

NODE_DTYPE = np.dtype([
    ("position", "3<f4"),
    ("velocity", "3<f4"),
    ("face_vels", "6<f4"),
    ("mass", "<f4"),
    ("layer", "<u4"),
    ("morton", "<u4"),
    ("active", "<u4"),
])

# Scale -> leaf size (must match train_configs.py); used when loading Ours weights.
_SCALE_LEAF_SIZE = {1024: 128, 2048: 128, 4096: 128, 8192: 128, 16384: 256}


# ============================================================================
# Frame loading
# ============================================================================
def load_frame(frame_dir):
    """Load a multiphase_v2 frame as a uniform grid + sparse SPD operator A."""
    p = Path(frame_dir)
    nodes = np.fromfile(p / "nodes.bin", dtype=NODE_DTYPE)
    N = len(nodes)
    xs = nodes["position"][:, 0].astype(np.int32)
    ys = nodes["position"][:, 1].astype(np.int32)
    W = int(xs.max()) + 1
    H = int(ys.max()) + 1
    # Training frames can omit a few bbox cells (holes) so N < W*H; duplicates would break scatter.
    pairs = np.column_stack((xs, ys))
    n_unique = len(np.unique(pairs, axis=0))
    if n_unique != N:
        raise ValueError(
            f"{p}: expected one node per (x,y) cell, got N={N}, unique (x,y)={n_unique}, bbox W×H={W}×{H}={W * H}"
        )

    rows = np.fromfile(p / "edge_index_rows.bin", dtype=np.uint32).astype(np.int64)
    cols = np.fromfile(p / "edge_index_cols.bin", dtype=np.uint32).astype(np.int64)
    vals = np.fromfile(p / "A_values.bin", dtype=np.float32).astype(np.float64)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    A = 0.5 * (A + A.T)  # symmetrize tiny floating-point asymmetry

    mass_flat = nodes["mass"].astype(np.float32)
    mass_grid = np.zeros((H, W), dtype=np.float32)
    mass_grid[ys, xs] = mass_flat

    rho_light = float(np.percentile(mass_flat, 1))
    rho_heavy = float(np.percentile(mass_flat, 99))
    thr = math.sqrt(max(rho_light, 1e-6) * max(rho_heavy, 1e-6))
    barrier = mass_grid > thr

    return {
        "frame_dir": str(p),
        "N": N, "W": W, "H": H,
        "mass_flat": mass_flat,
        "mass_grid": mass_grid,
        "barrier": barrier,
        "rho_light": rho_light,
        "rho_heavy": rho_heavy,
        "ratio": rho_heavy / max(rho_light, 1e-12),
        "A": A,
        "xs": xs, "ys": ys,
    }


def gridify(field_flat, info):
    g = np.zeros((info["H"], info["W"]), dtype=field_flat.dtype)
    g[info["ys"], info["xs"]] = field_flat
    return g


def load_frame_3d(frame_dir):
    """Load a 3D multiphase_v2 frame (positions integer (x, y, z) on a uniform cubic grid)."""
    p = Path(frame_dir)
    nodes = np.fromfile(p / "nodes.bin", dtype=NODE_DTYPE)
    N = len(nodes)
    xs = nodes["position"][:, 0].astype(np.int32)
    ys = nodes["position"][:, 1].astype(np.int32)
    zs = nodes["position"][:, 2].astype(np.int32)
    if int(zs.max()) == 0:
        raise ValueError(f"{p}: positions appear 2D (z all zero); use load_frame() instead.")
    W = int(xs.max()) + 1
    H = int(ys.max()) + 1
    D = int(zs.max()) + 1
    mass_flat = nodes["mass"].astype(np.float32)
    mass_grid = np.zeros((W, H, D), dtype=np.float32)
    mass_grid[xs, ys, zs] = mass_flat
    rho_light = float(np.percentile(mass_flat, 1))
    rho_heavy = float(np.percentile(mass_flat, 99))
    thr = math.sqrt(max(rho_light, 1e-6) * max(rho_heavy, 1e-6))
    barrier = mass_grid > thr
    rows = np.fromfile(p / "edge_index_rows.bin", dtype=np.uint32).astype(np.int64)
    cols = np.fromfile(p / "edge_index_cols.bin", dtype=np.uint32).astype(np.int64)
    vals = np.fromfile(p / "A_values.bin", dtype=np.float32).astype(np.float64)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    A = 0.5 * (A + A.T)
    return {
        "frame_dir": str(p),
        "N": N, "W": W, "H": H, "D": D,
        "xs": xs, "ys": ys, "zs": zs,
        "mass_flat": mass_flat,
        "mass_grid": mass_grid,
        "barrier": barrier,
        "rho_light": rho_light,
        "rho_heavy": rho_heavy,
        "ratio": rho_heavy / max(rho_light, 1e-12),
        "A": A,
    }


def buoyancy_rhs(info, g=1.0):
    """Right-hand side b = -d(rho*g)/dy projected to range(A) (constant nullspace removed)."""
    rho = info["mass_grid"].astype(np.float64)
    drhody = np.zeros_like(rho)
    drhody[1:-1, :] = 0.5 * (rho[2:, :] - rho[:-2, :])
    b_grid = -g * drhody
    b = np.zeros(info["N"], dtype=np.float64)
    b[:] = b_grid[info["ys"], info["xs"]]
    b -= b.mean()
    return b


def buoyancy_rhs_3d(info, g=1.0):
    """3D gravity RHS b = -d(rho*g)/dz; projected to range(A)."""
    xs = info["xs"]; ys = info["ys"]; zs = info["zs"]
    rho_3d = info["mass_grid"].astype(np.float64)   # (W, H, D)
    drhodz = np.zeros_like(rho_3d)
    drhodz[:, :, 1:-1] = 0.5 * (rho_3d[:, :, 2:] - rho_3d[:, :, :-2])
    b_grid = -g * drhodz
    b = b_grid[xs, ys, zs].astype(np.float64)
    b -= b.mean()
    return b


# ============================================================================
# Spectral / preconditioner utilities
# ============================================================================
def condition_number(A, tol=5e-3, max_k=4):
    """kappa(A) via Lanczos. Project away the rank-1 constant nullspace."""
    N = A.shape[0]
    try:
        lmax = float(eigsh(A, k=1, which="LA", tol=tol, return_eigenvectors=False, maxiter=2000)[0])
    except Exception as e:
        print(f"  eigsh LA failed: {e}", file=sys.stderr)
        return float("nan"), float("nan"), float("nan")
    try:
        vals = eigsh(A, k=max_k, which="SA", tol=tol, return_eigenvectors=False, maxiter=4000)
        vals = np.sort(np.real(vals))
        # discard eigenvalues consistent with the rank-1 constant nullspace
        cutoff = max(1e-8 * lmax, 1e-12)
        keep = vals[vals > cutoff]
        if keep.size == 0:
            lmin = float(vals[-1])
        else:
            lmin = float(keep[0])
    except Exception as e:
        print(f"  eigsh SA failed: {e}", file=sys.stderr)
        lmin = float("nan")
    if not math.isfinite(lmin) or lmin <= 0:
        return lmax, float("nan"), float("nan")
    return lmax, lmin, lmax / lmin


def amg_solver(A_csr):
    import pyamg
    return pyamg.smoothed_aggregation_solver(A_csr.astype(np.float64))


def _subsample_sorted(keys: list[int], max_n: int) -> list[int]:
    if len(keys) <= max_n:
        return keys
    idx = np.linspace(0, len(keys) - 1, num=max_n).round().astype(int)
    out = [keys[int(i)] for i in idx]
    # preserve uniqueness while keeping order
    seen = set()
    uniq = []
    for k in out:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


# Candidate PCG iterations at which we may store r (pruned to conv grid rows after the solve).
_PCG_SNAP_CANDIDATES = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128,
    160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072,
    3584, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 14336, 16384, 20480, 24576, 32768, 40960, 49152,
    65536,
)

# Default number of |r_k| tiles in the lower two rows (5 + 5).
_CONV_RESIDUAL_GRID_SLOTS = 10


def pcg_preconditioned_full(
    A,
    b,
    apply_M,
    *,
    rtol: float = 1e-8,
    max_iter: int = 10000,
    snap_candidates: tuple[int, ...] = _PCG_SNAP_CANDIDATES,
    snap_iters_explicit: Optional[list[int]] = None,
    max_snapshots: int = 12,
) -> tuple[dict[int, np.ndarray], dict]:
    """Preconditioned CG (same dynamics as before). Prints iteration log; returns sparse residual snaps.

    Stops when ||r||_2 / ||b||_2 <= rtol. ``snap_iters_explicit`` overrides candidate grid.
    """
    b = np.asarray(b, dtype=np.float64).ravel()
    N = A.shape[0]
    if b.shape[0] != N:
        raise ValueError("b shape mismatch")
    b_norm = float(np.linalg.norm(b)) + 1e-30
    x = np.zeros_like(b)
    r = b - A @ x
    z = apply_M(r)
    p = z.copy()
    rho = float(r @ z)
    snaps: dict[int, np.ndarray] = {}
    want = sorted(set(int(k) for k in (snap_iters_explicit or snap_candidates) if k >= 1))
    converged = False
    K = 0

    def _log_line(k: int, rn: float, rel: float) -> None:
        if k <= 30 or (k % 25 == 0) or rel <= rtol:
            print(f"  [PCG] iter={k:5d}  ||r||_2={rn:.6e}  rel={rel:.6e}")

    rn0 = float(np.sqrt(float(r @ r)))
    print(
        f"  [PCG] N={N}  ||b||_2={b_norm:.6e}  rtol={rtol:g}  max_iter={max_iter}  "
        f"initial rel={rn0 / b_norm:.6e}"
    )
    for k in range(1, max_iter + 1):
        Ap = A @ p
        pAp = float(p @ Ap)
        if pAp <= 0:
            print(f"  [PCG] iter={k}: non-positive pAp, stopping", file=sys.stderr)
            K = k
            snaps[k] = r.copy()
            break
        alpha = rho / pAp
        x += alpha * p
        r -= alpha * Ap
        rn = float(np.sqrt(float(r @ r)))
        rel = rn / b_norm
        if snap_iters_explicit is not None:
            if k in want:
                snaps[k] = r.copy()
        elif k in want:
            snaps[k] = r.copy()
        _log_line(k, rn, rel)
        if rel <= rtol:
            converged = True
            K = k
            snaps[k] = r.copy()
            print(f"  [PCG] converged at iter={k}  rel={rel:.6e}  (rtol={rtol:g})")
            break
        z = apply_M(r)
        rho_new = float(r @ z)
        if rho == 0:
            print(f"  [PCG] iter={k}: rho=0, stopping", file=sys.stderr)
            K = k
            snaps[k] = r.copy()
            break
        beta = rho_new / rho
        p = z + beta * p
        rho = rho_new
        K = k
    else:
        rn = float(np.sqrt(float(r @ r)))
        rel = rn / b_norm
        print(f"  [PCG] reached max_iter={max_iter} without rtol  final rel={rel:.6e}", file=sys.stderr)
        snaps[max_iter] = r.copy()
        K = max_iter

    if not snaps:
        snaps[K] = r.copy()

    # Prune snapshot dict for plotting (keep first / spread / last).
    keys_sorted = sorted(snaps.keys())
    if snap_iters_explicit is None and len(keys_sorted) > max_snapshots:
        keys_sorted = _subsample_sorted(keys_sorted, max_snapshots)

    snaps_plot = {k: snaps[k] for k in keys_sorted if k in snaps}
    meta = {
        "iters": K,
        "converged": converged,
        "final_rel": float(np.linalg.norm(snaps_plot[max(snaps_plot)]) / b_norm),
        "b_norm": b_norm,
    }
    return snaps_plot, meta


# ============================================================================
# Ours: neural preconditioner from checkpoint .bytes
# ============================================================================
def _read_checkpoint_max_mixed_size(weights_path) -> Optional[int]:
    """Return max_mixed_size stored in the checkpoint metadata, or None if unavailable."""
    try:
        if str(SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_DIR))
        from leafonly.checkpoint import apply_leaf_only_runtime_from_checkpoint
        meta = apply_leaf_only_runtime_from_checkpoint(str(weights_path))
        if meta and "max_mixed_size" in meta:
            return int(meta["max_mixed_size"])
    except Exception:
        pass
    return None


class OursPreconditioner:
    """Loads a trained checkpoint and exposes apply_M(r) on the CPU side."""

    def __init__(self, info, weights_path, leaf_size=None, max_mixed_size=None):
        self.info = info
        self.weights_path = Path(weights_path)
        N = info["N"]
        if leaf_size is None:
            leaf_size = _SCALE_LEAF_SIZE.get(N, 128)
        if max_mixed_size is None:
            # Prefer the value the checkpoint was trained with (ensures H-matrix layout matches).
            # Fall back to rounding N up to the next leaf-aligned multiple.
            _ckpt_meta = _read_checkpoint_max_mixed_size(self.weights_path)
            if _ckpt_meta is not None:
                max_mixed_size = _ckpt_meta
            else:
                max_mixed_size = ((N + leaf_size - 1) // leaf_size) * leaf_size
        self._bootstrap(leaf_size, max_mixed_size)

        import torch
        import torch.nn.functional as F

        from leafonly.architecture import (
            LeafOnlyNet,
            apply_block_diagonal_m_into,
            block_diagonal_m_apply_workspace,
            default_attention_layout,
            warmup_hmatrix_prolong_gpu,
        )
        from leafonly.checkpoint import (
            leaf_only_arch_from_checkpoint,
            load_leaf_only_weights,
        )
        from leafonly.config import (
            LEAF_APPLY_SIZE,
            LEAF_APPLY_SIZE_OFF,
            LEAF_SIZE,
            problem_padded_num_nodes,
        )
        from leafonly.data import FluidGraphDataset, build_leaf_block_connectivity
        from leafonly.hmatrix import NUM_HMATRIX_OFF_BLOCKS

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        self.device = device

        # FluidGraphDataset is the canonical loader for x / global_features.
        parent = Path(info["frame_dir"]).parent
        ds = FluidGraphDataset([parent])
        batch = None
        for i in range(len(ds)):
            if Path(ds.frame_paths[i]) == Path(info["frame_dir"]):
                batch = ds[i]
                break
        if batch is None:
            raise RuntimeError(f"Frame {info['frame_dir']} not found via FluidGraphDataset")

        num_nodes_real = int(batch["num_nodes"])
        viz_n = problem_padded_num_nodes(num_nodes_real)
        leaf_L = int(LEAF_SIZE)
        leaf_apply_diag_L = int(LEAF_APPLY_SIZE)
        leaf_apply_off_L = int(LEAF_APPLY_SIZE_OFF)
        num_leaves = viz_n // leaf_L

        ei, ev = batch["edge_index"], batch["edge_values"]
        em = (ei[0] < viz_n) & (ei[1] < viz_n)

        x_base = batch["x"].unsqueeze(0).to(device)
        n_feat = x_base.shape[1]
        if n_feat < viz_n:
            x_leaf = F.pad(x_base, (0, 0, 0, viz_n - n_feat), value=0.0)
        else:
            x_leaf = x_base[:, :viz_n, :]

        global_feat = batch.get("global_features")
        if global_feat is not None:
            global_feat = global_feat.to(device)
            if global_feat.dim() == 1:
                global_feat = global_feat.unsqueeze(0)

        edge_index_gpu = ei[:, em].to(device)
        edge_values_gpu = ev[em].to(device)
        positions = x_leaf[0, :, :3]

        # Jacobi mask for the in-graph diagonal regularizer used by apply_block_diagonal_m_into.
        diag_t = batch["edge_values"][batch["edge_index"][0] == batch["edge_index"][1]].to(device)
        # safer: reconstruct from edge_index_gpu / edge_values_gpu
        diag_mask = edge_index_gpu[0] == edge_index_gpu[1]
        diag_vec = torch.zeros(viz_n, device=device, dtype=torch.float32)
        diag_vec.scatter_(0, edge_index_gpu[0][diag_mask].long(), edge_values_gpu[diag_mask].float())
        jacobi_inv = torch.ones(1, viz_n, device=device, dtype=torch.float32)
        ok = diag_vec.abs() > 1e-6
        jacobi_inv[0, ok] = 1.0 / diag_vec[ok]
        self._jacobi_s = jacobi_inv.contiguous()

        pre_connectivity = build_leaf_block_connectivity(
            edge_index_gpu,
            edge_values_gpu,
            positions,
            leaf_L,
            device,
            x_leaf.dtype,
            off_diag_dense_attention=True,
            diag_dense_attention=True,
        )
        pre_connectivity = tuple(
            t.contiguous() if isinstance(t, torch.Tensor) else t for t in pre_connectivity
        )

        ckpt_arch = leaf_only_arch_from_checkpoint(self.weights_path)
        if ckpt_arch is None:
            raise RuntimeError(f"Cannot read architecture from {self.weights_path}")
        ck_hw = int(ckpt_arch.get("highway_ffn_mlp", 0))
        ck_ffn = int(ckpt_arch.get("ffn_concat_width", 3 if ck_hw else 1))

        model = LeafOnlyNet(
            input_dim=9,
            d_model=int(ckpt_arch["d_model"]),
            leaf_size=leaf_L,
            num_layers=int(ckpt_arch["num_layers"]),
            num_heads=int(ckpt_arch["num_heads"]),
            use_gcn=bool(ckpt_arch["use_gcn"]),
            attention_layout=default_attention_layout(leaf_L),
            off_diag_dense_attention=True,
            diag_dense_attention=True,
            use_highways=bool(ck_hw),
            ffn_concat_width=ck_ffn if ck_hw else None,
        ).to(device)
        load_leaf_only_weights(model, str(self.weights_path))
        model.eval()

        warmup_hmatrix_prolong_gpu(device)

        ws = block_diagonal_m_apply_workspace(
            num_leaves=num_leaves,
            leaf_size=leaf_L,
            K_dim=1,
            M_h=NUM_HMATRIX_OFF_BLOCKS,
            La_o=leaf_apply_off_L,
            device=device,
            dtype=torch.float32,
        )

        with torch.inference_mode():
            # warmup forward, then freeze precond features
            for _ in range(3):
                out = model(
                    x_leaf,
                    edge_index=edge_index_gpu,
                    edge_values=edge_values_gpu,
                    global_features=global_feat,
                    precomputed_leaf_connectivity=pre_connectivity,
                )
            self._precond_s = out.detach().contiguous()

        self._torch = torch
        self._apply_into = apply_block_diagonal_m_into
        self._ws = ws
        self._r_buf = torch.zeros(1, viz_n, 1, device=device, dtype=torch.float32)
        self._z_buf = torch.zeros(1, viz_n, 1, device=device, dtype=torch.float32)
        self._viz_n = viz_n
        self._real_n = num_nodes_real
        self._leaf_L = leaf_L
        self._leaf_apply_diag_L = leaf_apply_diag_L
        self._leaf_apply_off_L = leaf_apply_off_L
        self._batch = batch
        self.device = device

    def _bootstrap(self, leaf_size, max_mixed_size):
        import importlib.util
        import types
        leafonly_dir = SCRIPTS_DIR / "leafonly"
        if str(SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_DIR))
        cfg_path = leafonly_dir / "config.py"
        pkg = "leafonly"
        if pkg not in sys.modules:
            m_pkg = types.ModuleType(pkg)
            m_pkg.__path__ = [str(leafonly_dir)]
            sys.modules[pkg] = m_pkg
        cfg_name = "leafonly.config"
        spec = importlib.util.spec_from_file_location(cfg_name, cfg_path)
        cfg_mod = importlib.util.module_from_spec(spec)
        sys.modules[cfg_name] = cfg_mod
        spec.loader.exec_module(cfg_mod)
        cfg_mod.apply_runtime_sizes(leaf_size, max_mixed_size)

    def apply_M(self, r_np):
        """numpy r (N,) -> z (N,) using the learned preconditioner."""
        torch = self._torch
        viz_n = self._viz_n
        r_pad = np.zeros(viz_n, dtype=np.float32)
        r_pad[:self._real_n] = r_np.astype(np.float32, copy=False)
        self._r_buf.copy_(torch.from_numpy(r_pad).to(self.device).reshape(1, viz_n, 1))
        self._apply_into(
            self._precond_s,
            self._r_buf,
            self._z_buf,
            self._jacobi_s,
            self._ws,
            leaf_size=self._leaf_L,
            leaf_apply_size=self._leaf_apply_diag_L,
            leaf_apply_off=self._leaf_apply_off_L,
        )
        z = self._z_buf.reshape(viz_n).detach().cpu().numpy().astype(np.float64)
        return z[:self._real_n]

    def dense_A_numpy(self) -> np.ndarray:
        """Dense Poisson operator on the padded leaf grid (same subgraph as InspectModel)."""
        torch = self._torch
        batch = self._batch
        viz_n = self._viz_n
        ei, ev = batch["edge_index"], batch["edge_values"]
        em = (ei[0] < viz_n) & (ei[1] < viz_n)
        A_small = torch.sparse_coo_tensor(
            ei[:, em].contiguous().cpu(),
            ev[em].contiguous().cpu(),
            (viz_n, viz_n),
        ).coalesce().to_dense().numpy()
        return A_small.astype(np.float64, copy=False)

    def assembled_dense_M_numpy(self) -> np.ndarray:
        """Dense assembled learned preconditioner (InspectModel ``M_neural`` assembly)."""
        return _assemble_neural_m_dense_from_precond(
            self._precond_s,
            self._jacobi_s,
            self._viz_n,
            self._leaf_L,
            self._leaf_apply_diag_L,
            self._leaf_apply_off_L,
            self.device,
        )


def _assign_dense_block_clamped(M, oblk, br0, br1, bc0, bc1, dim_max):
    r0c, r1c = max(br0, 0), min(br1, dim_max)
    c0c, c1c = max(bc0, 0), min(bc1, dim_max)
    if r0c >= r1c or c0c >= c1c:
        return
    M[r0c:r1c, c0c:c1c] = oblk[(r0c - br0) : (r1c - br0), (c0c - bc0) : (c1c - bc0)]


def _offdiag_strip_from_packed_leaves(nodes, r0i, si, num_leaves, leaf_L, La_o):
    import torch

    device, dtype = nodes.device, nodes.dtype
    out = torch.zeros((si * leaf_L, La_o), device=device, dtype=dtype)
    for j in range(si):
        gi = r0i + j
        if 0 <= gi < num_leaves:
            out[j * leaf_L : (j + 1) * leaf_L, :] = nodes[gi]
    return out


def _assemble_neural_m_dense_from_precond(
    precond_s,
    jacobi_inv_diag,
    viz_n: int,
    leaf_L: int,
    leaf_apply_diag_L: int,
    leaf_apply_off_L: int,
    device,
) -> np.ndarray:
    import torch
    from leafonly.architecture import unpack_precond
    from leafonly import hmatrix as _hm

    pool_diag_to_full = leaf_L // leaf_apply_diag_L
    num_leaves = viz_n // leaf_L
    pre = precond_s.to(device=device).contiguous()
    jac = jacobi_inv_diag.to(device=device).contiguous()

    diag_blocks, off_diag_blocks, node_U, node_V, jacobi_scale = unpack_precond(
        pre, viz_n, leaf_size=leaf_L, leaf_apply_size=leaf_apply_diag_L, leaf_apply_off=leaf_apply_off_L
    )

    M_neural_gpu = torch.zeros((viz_n, viz_n), dtype=torch.float32, device=device)
    for b in range(num_leaves):
        r0, r1 = b * leaf_L, (b + 1) * leaf_L
        blk = diag_blocks[0, b]
        if pool_diag_to_full > 1:
            blk = blk.repeat_interleave(pool_diag_to_full, dim=0).repeat_interleave(pool_diag_to_full, dim=1)
        M_neural_gpu[r0:r1, r0:r1] = blk

    if int(_hm.NUM_HMATRIX_OFF_BLOCKS) > 0 and off_diag_blocks is not None and node_U is not None and node_V is not None:
        for i in range(int(_hm.NUM_HMATRIX_OFF_BLOCKS)):
            r0i = int(_hm.HM_R0_CPU[i].item())
            c0i = int(_hm.HM_C0_CPU[i].item())
            si = int(_hm.HM_S_CPU[i].item())
            br0, br1 = r0i * leaf_L, (r0i + si) * leaf_L
            bc0, bc1 = c0i * leaf_L, (c0i + si) * leaf_L
            C_m = off_diag_blocks[0, i]
            U_strip = _offdiag_strip_from_packed_leaves(
                node_U[0], r0i, si, num_leaves, leaf_L, leaf_apply_off_L
            )
            V_strip = _offdiag_strip_from_packed_leaves(
                node_V[0], c0i, si, num_leaves, leaf_L, leaf_apply_off_L
            )
            U_C = torch.matmul(U_strip, C_m)
            oblk_dense = torch.matmul(U_C, V_strip.transpose(0, 1))
            _assign_dense_block_clamped(M_neural_gpu, oblk_dense, br0, br1, bc0, bc1, viz_n)
            _assign_dense_block_clamped(
                M_neural_gpu, oblk_dense.transpose(-1, -2), bc0, bc1, br0, br1, viz_n
            )

    if jacobi_scale is not None:
        M_neural_gpu += torch.diag((jacobi_scale[0] * jac[0]).to(M_neural_gpu.dtype))
    return M_neural_gpu.detach().cpu().numpy().astype(np.float64)


def _dense_inv_numpy_or_torch(A: np.ndarray, device) -> np.ndarray:
    """Dense inverse (InspectModel-style): CUDA fp64, MPS fp32, else NumPy."""
    import torch

    A = np.asarray(A, dtype=np.float64)
    if device.type == "cuda":
        t = torch.as_tensor(A, dtype=torch.float64, device=device)
        out = torch.linalg.inv(t).cpu().numpy()
        torch.cuda.synchronize()
        return out
    if device.type == "mps":
        t = torch.as_tensor(A, dtype=torch.float32, device=device)
        out = torch.linalg.inv(t).cpu().numpy().astype(np.float64)
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        return out
    return np.linalg.inv(A)


def _finalize_conv_residual_snaps(snaps: dict[int, np.ndarray], K_final: int, n_slots: int) -> dict[int, np.ndarray]:
    """Pick ``n_slots`` residual frames: geometric spread in iteration, snapped to available keys."""
    avail = sorted(snaps.keys())
    if not avail:
        return {}
    if len(avail) <= n_slots:
        return {k: snaps[k] for k in avail}
    hi = max(int(K_final), 1)
    targets = np.unique(np.round(np.geomspace(1, float(hi), num=n_slots)).astype(int))
    chosen: list[int] = []
    for t in targets:
        closest = min(avail, key=lambda k: abs(k - int(t)))
        chosen.append(int(closest))
    chosen = sorted(set(chosen))
    if int(K_final) not in chosen:
        chosen.append(int(K_final))
        chosen = sorted(set(chosen))
    if len(chosen) > n_slots:
        chosen = _subsample_sorted(chosen, n_slots)
    return {k: snaps[k] for k in chosen}


def _select_method_snap_iters(
    snaps: dict[int, np.ndarray],
    K_final: int,
    n_panels: int = 5,
    *,
    end_frac: float = 0.9,
) -> list[int]:
    """Per-method snap selection: geometric (log-spaced) sampling from iter 1 to ``end_frac*K``.

    Geometric spacing weights the early iterations more heavily — where the residual still
    carries structure and the per-iteration change is large — while still spanning to a frame
    near the end. We deliberately stop at ``end_frac*K`` (default 0.9) so the final panel does
    not show the converged near-zero residual (which is just black). Falls back gracefully when
    K is small: targets collapse to integers, duplicates are filled from available iterations.
    Each target is snapped to the closest available recorded iter.
    """
    avail = sorted(snaps.keys())
    if not avail:
        return []
    K = max(int(K_final), 1)
    K_end = max(1, int(round(K * float(end_frac))))
    if K_end < 2:
        targets = np.array([1] * n_panels, dtype=np.float64)
    else:
        targets = np.geomspace(1.0, float(K_end), num=n_panels)
    seen: list[int] = []
    for t in targets:
        ti = int(round(float(t)))
        ti = max(1, min(K, ti))
        if ti not in seen:
            seen.append(ti)
    # Fill with whatever's available if geometric collapse left us short (small K).
    if len(seen) < n_panels:
        for k in avail:
            if k not in seen:
                seen.append(int(k))
            if len(seen) >= n_panels:
                break
    seen = sorted(seen)[:n_panels]
    # Snap each target to the closest available iter.
    out: list[int] = []
    for t in seen:
        closest = min(avail, key=lambda k: abs(int(k) - int(t)))
        if int(closest) not in out:
            out.append(int(closest))
    if len(out) < n_panels:
        for k in avail:
            if int(k) not in out:
                out.append(int(k))
            if len(out) >= n_panels:
                break
    out = sorted(out)[:n_panels]
    return out


# ============================================================================
# Baseline preconditioner builders (apply_M : np.ndarray (N,) -> np.ndarray (N,))
# ============================================================================
def _build_identity_apply(N):
    """Unpreconditioned: z = r."""
    def apply_I(r):
        return np.asarray(r, dtype=np.float64).ravel().copy()
    return apply_I


def _build_jacobi_apply(A):
    """Diagonal (Jacobi) preconditioner: z = D^{-1} r."""
    d = np.asarray(A.diagonal(), dtype=np.float64).ravel()
    inv = 1.0 / np.maximum(np.abs(d), 1e-30)

    def apply_J(r):
        return np.asarray(r, dtype=np.float64).ravel() * inv
    return apply_J


def _build_ic_apply(A):
    """Incomplete-Cholesky-class preconditioner: ``ilupp.IChol0`` if available, else scipy
    ``spilu`` (ILU(0)-ish) as a fallback. Operates on the (possibly indefinite-on-the-null-space)
    SPD operator A; constant nullspace is handled implicitly via b having zero mean."""
    A_csr = A if isinstance(A, sp.csr_matrix) else sp.csr_matrix(A)
    A_csr.sort_indices()
    try:
        import ilupp

        prec = ilupp.IChol0Preconditioner(A_csr.astype(np.float64))

        def apply_ic(r):
            return (prec @ np.asarray(r, dtype=np.float64).ravel()).astype(np.float64)
        return apply_ic, "ilupp IChol0"
    except Exception:
        pass
    from scipy.sparse.linalg import spilu
    lu = spilu(
        A_csr.astype(np.float64).tocsc(),
        drop_tol=1e-8,
        fill_factor=30,
        permc_spec="COLAMD",
        diag_pivot_thresh=0.0,
    )

    def apply_ilu(r):
        return lu.solve(np.asarray(r, dtype=np.float64).ravel())
    return apply_ilu, "scipy spilu"


def _build_amg_apply(A):
    """Smoothed-aggregation AMG, single V-cycle as a preconditioner (maxiter=1)."""
    ml = amg_solver(A)
    N = A.shape[0]

    def apply_amg(r):
        r_flat = np.asarray(r, dtype=np.float64).ravel()
        x0 = np.zeros(N, dtype=np.float64)
        z = ml.solve(r_flat, x0=x0, maxiter=1, cycle='V', tol=1e-6)
        return np.asarray(z, dtype=np.float64).ravel()
    return apply_amg


def _find_neural_spai_checkpoint(scale: int, summary_csv: Path) -> str:
    exp = f"multiphase_v2_{int(scale)}"
    with Path(summary_csv).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("exp_name", "") != exp:
                continue
            method = row.get("method", "")
            status = row.get("status", "")
            ckpt = row.get("checkpoint", "")
            if method in ("Neural", "Neural+CUDA") and status == "ok" and ckpt:
                return ckpt
    raise RuntimeError(
        f"Could not find a NeuralSPAI checkpoint for scale={scale} in {summary_csv}."
    )


class NeuralSPAIPreconditioner:
    """Load LearningSparsePreconditioner4GPU checkpoint and expose apply_M(r)."""

    def __init__(self, info, checkpoint_path: str, lsp_repo: str):
        self.info = info
        self.checkpoint_path = Path(checkpoint_path)
        self.lsp_repo = Path(lsp_repo)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"NeuralSPAI checkpoint not found: {self.checkpoint_path}")
        if not self.lsp_repo.is_dir():
            raise FileNotFoundError(f"LSP repo not found: {self.lsp_repo}")
        self._bootstrap()

    def _bootstrap(self):
        if str(self.lsp_repo) not in sys.path:
            sys.path.insert(0, str(self.lsp_repo))

        import torch
        from torch_geometric.data import Data
        from neural_cg.workspace import SimpleTrainingWorkspace
        from neural_cg.scaled_workspace import ScaledTrainingWorkspace

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        model = None
        for workspace_cls in (SimpleTrainingWorkspace, ScaledTrainingWorkspace):
            try:
                model = workspace_cls.load_from_checkpoint(str(self.checkpoint_path), map_location=device)
                break
            except Exception:
                continue
        if model is None:
            raise RuntimeError(f"Failed to load NeuralSPAI workspace from {self.checkpoint_path}")
        model = model.to(device)
        model.eval()

        A_csr = self.info["A"].tocsr().astype(np.float64)
        coo = A_csr.tocoo()
        N = int(self.info["N"])

        edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device)
        edge_vals = coo.data.astype(np.float32)
        edge_attr = torch.tensor(edge_vals.reshape(-1, 1), dtype=torch.float32, device=device)
        matrix_values = torch.tensor(edge_vals.reshape(-1, 1, 1), dtype=torch.float32, device=device)

        diag = A_csr.diagonal().astype(np.float32).reshape(-1, 1)
        inv_diag = np.ones_like(diag, dtype=np.float32)
        rsqrt_diag = np.ones_like(diag, dtype=np.float32)
        ok = np.abs(diag) > 1e-6
        inv_diag[ok] = 1.0 / diag[ok]
        rsqrt_diag[ok] = 1.0 / np.sqrt(diag[ok])
        mask = np.ones((N, 1), dtype=np.float32)
        ones = np.ones((N, 1), dtype=np.float32)

        sample = Data(
            x=torch.tensor(ones, dtype=torch.float32, device=device),
            edge_index=edge_index,
            edge_attr=edge_attr,
            matrix_values=matrix_values,
            mask=torch.tensor(mask, dtype=torch.float32, device=device),
            diagonal=torch.tensor(diag, dtype=torch.float32, device=device),
            inv_diag=torch.tensor(inv_diag, dtype=torch.float32, device=device),
            rsqrt_diag=torch.tensor(rsqrt_diag, dtype=torch.float32, device=device),
            residual=torch.tensor(ones, dtype=torch.float32, device=device),
        )

        with torch.no_grad():
            M = model.precondition(sample, A_csr.toarray())
        self._M = np.asarray(M, dtype=np.float64)

    def apply_M(self, r):
        rr = np.asarray(r, dtype=np.float64).ravel()
        return (self._M @ rr).astype(np.float64, copy=False)


def _pressure_velocity_fields(info):
    """Return pressure field P and velocity magnitude from buoyancy RHS on this frame."""
    b = buoyancy_rhs(info)
    ml = amg_solver(info["A"])
    p = ml.solve(b, tol=1e-8, accel="cg", maxiter=2000).astype(np.float64)
    p -= p.mean()

    rho = info["mass_grid"].astype(np.float64) + 1e-6
    P = gridify(p, info)
    dpdx = np.zeros_like(P)
    dpdy = np.zeros_like(P)
    dpdx[:, 1:-1] = 0.5 * (P[:, 2:] - P[:, :-2])
    dpdy[1:-1, :] = 0.5 * (P[2:, :] - P[:-2, :])
    vx = -dpdx / rho
    vy = -dpdy / rho
    return b, P, vx, vy, np.hypot(vx, vy)


def _shared_log_abs_limits(
    matrices,
    *,
    vmin_pct: float = 1.0,
    vmax_pct: float = 99.5,
    diag_band_exclude_for_scale: Optional[int] = None,
    subtract_log_max: bool = False,
) -> tuple[float, float]:
    """Shared (lo, hi) log₁₀|·| range across multiple matrices (off-diagonal band optional).

    When ``subtract_log_max`` is True, each matrix is shifted in log space so its own max
    maps to 0 *before* the percentile is taken. This normalizes the inputs onto a common
    "peak-relative" scale, so e.g. ``A^{-1}`` (large absolute entries) and learned ``M``
    (smaller absolute entries) can share a colorbar that emphasizes structure, not magnitude.
    """
    all_vals = []
    for M in matrices:
        M = np.asarray(M, dtype=np.float64)
        M_abs = np.abs(M)
        logm = np.log10(M_abs + 1e-9)
        if subtract_log_max:
            logm = logm - float(np.log10(max(float(M_abs.max()), 1e-30)))
        if diag_band_exclude_for_scale is not None and int(diag_band_exclude_for_scale) > 0:
            c = logm.shape[0]
            b = int(diag_band_exclude_for_scale)
            off = np.abs(np.arange(c, dtype=np.int32)[:, None] - np.arange(c, dtype=np.int32)[None, :]) > b
            vals = logm[off]
            if vals.size >= 64:
                all_vals.append(vals)
                continue
        all_vals.append(logm.ravel())
    cat = np.concatenate(all_vals)
    lo = float(np.percentile(cat, vmin_pct))
    hi = float(np.percentile(cat, vmax_pct))
    if hi <= lo + 1e-12:
        hi = lo + 1e-12
    return lo, hi


def _imshow_matrix_log10_abs(
    ax,
    M: np.ndarray,
    title: str,
    *,
    crop: int = 128,
    vmin_pct: float = 1.0,
    vmax_pct: float = 99.5,
    diag_band_exclude_for_scale: Optional[int] = None,
    vmin_log: Optional[float] = None,
    vmax_log: Optional[float] = None,
    subtract_log_max: bool = False,
    rank_normalize: bool = False,
    rank_from_nonzero: bool = False,
    rank_nonzero_color_floor: float = 0.1,
    cmap: str = "magma",
):
    """log₁₀|·| heatmap of the top-left ``crop``×``crop`` block (square aspect).

    Color limits use robust percentiles so a dominant diagonal does not crush the rest.
    If ``diag_band_exclude_for_scale`` is set, entries with ``|i-j| <= band`` are ignored when
    picking vmin/vmax (useful for learned ``M`` so off-diagonal / block structure stays visible).
    Pass ``vmin_log``/``vmax_log`` (in log10 units) to override the percentile pick — used to
    share a single color scale across multiple matrices (e.g. ``A^{-1}`` vs learned ``M``).
    When ``subtract_log_max`` is True, the displayed values are ``log10(|M| / max|M|)`` instead
    of ``log10|M|`` — this peak-normalizes each matrix so two panels can share a single
    structure-revealing colorbar even when their absolute magnitudes differ by orders.

    When ``rank_normalize`` is True, the displayed value is the per-panel percentile rank of
    log10|·| computed from the panel's *own* off-diagonal entries (CDF / histogram-equalize
    tonemap). The mapping is monotonic — entry ordering is preserved — but each panel's
    contrast is rescaled so two structurally-similar matrices with very different magnitude
    distributions (e.g. true ``A^{-1}`` clustered in ~0.3 dex, learned ``M`` spread over
    ~2 dex) render with comparable visual contrast. The shared output range [0, 1] is then
    interpretable as "fraction of off-diagonal entries with smaller |·|".

    When ``rank_from_nonzero`` is True (only meaningful with ``rank_normalize``), the CDF is
    built from *nonzero* entries (zeros stay at rank 0 → colormap black) and the nonzero
    ranks are remapped from ``[0, 1]`` to ``[rank_nonzero_color_floor, 1]`` so the smallest
    nonzero is visibly off-black instead of indistinguishable from a true zero. Use this for
    matrices that are sparse in the structural sense (the operator ``A`` itself) so off-
    diagonal entries get colormap budget instead of being crushed against magma's very dark
    end.
    """
    M = np.asarray(M, dtype=np.float64)
    c = max(1, min(int(crop), M.shape[0], M.shape[1]))
    M_abs = np.abs(M[:c, :c])
    logm = np.log10(M_abs + 1e-9)
    if subtract_log_max:
        logm = logm - float(np.log10(max(float(M_abs.max()), 1e-30)))

    if rank_normalize:
        if rank_from_nonzero:
            # CDF over nonzero entries only. Zeros (structural sparsity) are marked NaN so the
            # colormap's set_bad("black") renders them as solid black background — independent
            # of the colormap's value-0 color (e.g. turbo(0) is dark purple, not black).
            # Nonzero ranks are remapped to [rank_nonzero_color_floor, 1] so the smallest
            # nonzero lands in a visibly-saturated region of the colormap (e.g. blue in turbo)
            # rather than the very-dark low end.
            nz_mask = M_abs > 0
            nz_logm = logm[nz_mask]
            ranks = np.full(logm.shape, np.nan, dtype=np.float64)
            if nz_logm.size >= 2:
                ref_sorted = np.sort(nz_logm.ravel())
                idx = np.searchsorted(ref_sorted, nz_logm, side="right").astype(np.float64)
                nz_ranks = idx / float(ref_sorted.size)
                floor = float(rank_nonzero_color_floor)
                ranks[nz_mask] = floor + (1.0 - floor) * nz_ranks
        else:
            # Empirical CDF built from off-diagonal entries (so the dominant diagonal does not
            # consume rank budget). Monotonic: ordering is preserved, only the spacing changes.
            if diag_band_exclude_for_scale is not None and int(diag_band_exclude_for_scale) > 0:
                b = int(diag_band_exclude_for_scale)
                off = np.abs(np.arange(c, dtype=np.int32)[:, None] - np.arange(c, dtype=np.int32)[None, :]) > b
                ref = logm[off]
            else:
                ref = logm
            ref_sorted = np.sort(ref.ravel())
            if ref_sorted.size < 2:
                ranks = np.zeros_like(logm)
            else:
                # searchsorted gives, for every entry, the count of reference values it exceeds;
                # divide by N to land in [0, 1] (mid-point convention via 'left' + 0.5 / N would
                # only shift the colorbar by half a bin, not worth it here).
                idx = np.searchsorted(ref_sorted, logm.ravel(), side="left").astype(np.float64)
                ranks = (idx / float(ref_sorted.size)).reshape(logm.shape)
        cmap_obj = matplotlib.colormaps[cmap].copy()
        cmap_obj.set_bad("black")
        im = ax.imshow(
            ranks,
            cmap=cmap_obj,
            aspect="equal",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        return im

    if vmin_log is not None and vmax_log is not None:
        lo, hi = float(vmin_log), float(vmax_log)
    elif diag_band_exclude_for_scale is not None and int(diag_band_exclude_for_scale) > 0:
        b = int(diag_band_exclude_for_scale)
        off = np.abs(np.arange(c, dtype=np.int32)[:, None] - np.arange(c, dtype=np.int32)[None, :]) > b
        vals = logm[off]
        if vals.size >= 64:
            lo = float(np.percentile(vals, vmin_pct))
            hi = float(np.percentile(vals, vmax_pct))
        else:
            lo = float(np.percentile(logm, vmin_pct))
            hi = float(np.percentile(logm, vmax_pct))
    else:
        lo = float(np.percentile(logm, vmin_pct))
        hi = float(np.percentile(logm, vmax_pct))
    if hi <= lo + 1e-12:
        lo = float(np.min(logm))
        hi = float(np.max(logm)) + 1e-12
    im = ax.imshow(
        logm,
        cmap=cmap,
        aspect="equal",
        vmin=lo,
        vmax=hi,
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return im


# ============================================================================
# Plot helpers
# ============================================================================
DENSITY_CMAP = "viridis"
RESIDUAL_CMAP = "magma"

# Colorbar geometry (line plots use shrink≈0.85 in paper scripts).
_CBAR_SHRINK = 0.85
_CBAR_PAD = 0.012


def _cbar_linear_paper(fig, im, ax, *, label=None):
    """Linear colorbar: fixed decimal format (no offset like ``+6.09e-6``) to keep layout stable."""
    cb = fig.colorbar(im, ax=ax, shrink=_CBAR_SHRINK, pad=_CBAR_PAD)
    cb.ax.tick_params(labelsize=matplotlib.rcParams["ytick.labelsize"])
    lo, hi = (float(x) for x in im.get_clim())
    span = max(hi - lo, 1e-30)
    if span >= 2:
        dec = "%.1f"
    elif span >= 0.2:
        dec = "%.2f"
    elif span >= 0.02:
        dec = "%.3f"
    elif span >= 2e-3:
        dec = "%.4f"
    else:
        dec = "%.2e"
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(dec))
    cb.ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, min_n_ticks=3))
    if label:
        cb.set_label(label, fontsize=matplotlib.rcParams["axes.labelsize"])
    return cb


def _cbar_log_paper(fig, im, ax, *, label=None):
    cb = fig.colorbar(im, ax=ax, shrink=_CBAR_SHRINK, pad=_CBAR_PAD)
    cb.ax.tick_params(labelsize=matplotlib.rcParams["ytick.labelsize"])
    if label:
        cb.set_label(label, fontsize=matplotlib.rcParams["axes.labelsize"])
    return cb


def _active_mask(info) -> np.ndarray:
    """2-D bool mask of bbox cells that actually carry a node (others are 'no data')."""
    active = np.zeros((info["H"], info["W"]), dtype=bool)
    active[info["ys"], info["xs"]] = True
    return active


def imshow_density(ax, info, vmin=None, vmax=None, title=None):
    rho = info["mass_grid"].astype(np.float64)
    active = _active_mask(info)
    # Inactive bbox cells have rho=0; render them as 'no data' (gray) instead of clipped LogNorm low.
    rho_masked = np.where(active, rho, np.nan)
    if vmin is None:
        vmin = max(float(rho[active].min()), 1e-3) if active.any() else 1e-3
    if vmax is None:
        vmax = max(vmin * 1.01, float(rho[active].max()) if active.any() else vmin * 10.0)
    cmap = matplotlib.colormaps[DENSITY_CMAP].copy()
    cmap.set_bad(color="lightgray")
    im = ax.imshow(
        rho_masked,
        origin="lower",
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",
    )
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    return im


def imshow_barrier(ax, info, title=None):
    im = ax.imshow(
        info["barrier"].astype(np.float32),
        origin="lower",
        cmap="Greys",
        vmin=0.0, vmax=1.0,
        interpolation="nearest",
    )
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    return im


# ============================================================================
# Figure 1: hero strip
# ============================================================================
def make_hero(frame_dir, out_path):
    _apply_paper_style_compact()
    info = load_frame(frame_dir)
    print(f"[hero] frame N={info['N']}  ratio={info['ratio']:.1f}x")

    b = buoyancy_rhs(info)
    ml = amg_solver(info["A"])
    p = ml.solve(b, tol=1e-8, accel="cg", maxiter=2000).astype(np.float64)
    p -= p.mean()

    # central-difference velocity v = -(grad p) / rho, on the grid (active cells only).
    rho = info["mass_grid"].astype(np.float64) + 1e-6
    P = gridify(p, info)
    dpdx = np.zeros_like(P); dpdy = np.zeros_like(P)
    dpdx[:, 1:-1] = 0.5 * (P[:, 2:] - P[:, :-2])
    dpdy[1:-1, :] = 0.5 * (P[2:, :] - P[:-2, :])
    vx = -dpdx / rho
    vy = -dpdy / rho

    # Inactive bbox cells (no node) would otherwise leak into RdBu_r as P=0 → near-white,
    # cutting the active fluid into spurious stripes. Mask them so they render as 'no data'.
    active = _active_mask(info)
    b_grid = gridify(b, info)
    b_masked = np.where(active, b_grid, np.nan)
    P_masked = np.where(active, P, np.nan)

    # Four panels: density, RHS, pressure (alone), velocity (HSV-encoded RGB). The quiver
    # overlay was overwhelming the pressure panel, so velocity gets its own picture; direction
    # is read by hue (color-wheel legend inset) and magnitude by brightness.
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.2), constrained_layout=True)

    im0 = imshow_density(
        axes[0], info,
        title=rf"Density $\rho$",
    )
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    # Panel 1: buoyancy right-hand side b = -∂ρ/∂y (mean-removed; actually solved for).
    b_abs = max(float(np.percentile(np.abs(b), 99)), 1e-30) if b.size else 1.0
    cmap_b = matplotlib.colormaps["RdBu_r"].copy()
    cmap_b.set_bad(color="lightgray")
    im1 = axes[1].imshow(
        b_masked, origin="lower", cmap=cmap_b,
        vmin=-b_abs, vmax=b_abs, interpolation="nearest",
    )
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_title(fr"RHS $b = -\partial_y \rho$")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)

    # Panel 2: pressure on its own (no overlay), shared RdBu_r diverging scale.
    P_abs = max(float(np.percentile(np.abs(p), 99)), 1e-30)
    cmap_p = matplotlib.colormaps["RdBu_r"].copy()
    cmap_p.set_bad(color="lightgray")
    im2 = axes[2].imshow(
        P_masked, origin="lower", cmap=cmap_p,
        vmin=-P_abs, vmax=P_abs, interpolation="nearest",
    )
    axes[2].set_xticks([]); axes[2].set_yticks([])
    axes[2].set_title(r"Pressure $p$")
    fig.colorbar(im2, ax=axes[2], shrink=0.85)

    # Panel 3: velocity v = -∇p/ρ. Background = |v| on a log colormap (compresses the huge
    # dynamic range between bulk flow and jet-near-corner peaks, so we don't get the prior
    # blown-out / pitch-black split). Streamline overlay encodes direction without the visual
    # noise of a dense quiver. Inactive cells render as 'no data' (gray).
    mag = np.hypot(vx, vy)
    mag_masked = np.where(active, mag, np.nan)
    if active.any():
        v_hi = float(np.percentile(mag[active], 99.5))
        v_lo_pos = mag[active & (mag > 0)]
        v_lo = float(np.percentile(v_lo_pos, 5.0)) if v_lo_pos.size else v_hi * 1e-3
    else:
        v_hi = max(float(mag.max()), 1e-30)
        v_lo = v_hi * 1e-3
    v_hi = max(v_hi, 1e-30)
    v_lo = max(min(v_lo, v_hi * 0.5), v_hi * 1e-3)  # at most 3 orders of dynamic range
    cmap_v = matplotlib.colormaps["viridis"].copy()
    cmap_v.set_bad(color="lightgray")
    im3 = axes[3].imshow(
        mag_masked, origin="lower", cmap=cmap_v,
        norm=LogNorm(vmin=v_lo, vmax=v_hi), interpolation="nearest",
    )
    # Streamline overlay: matplotlib needs a regular Y, X grid and finite vx/vy everywhere.
    H, W = vx.shape
    Yg, Xg = np.mgrid[0:H, 0:W]
    vx_safe = np.where(active, vx, 0.0)
    vy_safe = np.where(active, vy, 0.0)
    try:
        axes[3].streamplot(
            Xg, Yg, vx_safe, vy_safe,
            color="white", linewidth=0.7, density=1.4, arrowsize=0.7,
        )
    except Exception as e:
        print(f"  [hero] streamplot failed ({e}); skipping streamlines", file=sys.stderr)
    axes[3].set_xticks([]); axes[3].set_yticks([])
    axes[3].set_title(r"Velocity $|v|=|{-}\nabla p/\rho|$")
    axes[3].set_xlim(-0.5, W - 0.5)
    axes[3].set_ylim(-0.5, H - 0.5)
    fig.colorbar(im3, ax=axes[3], shrink=0.85)

    # No suptitle: paper.tex caption already gives full context (N, contrast, etc.),
    # and dropping it lets the three panels fill the limited vertical budget.
    _savefig_paper(fig, out_path)
    plt.close(fig)
    print(f"[hero] -> {out_path}  (|v| range≈[{v_lo:.3g}, {v_hi:.3g}])")


# ============================================================================
# Figure 2: variety grid (3 rows x 4 cols)
# ============================================================================
def make_variety(data_root, out_path, scales=(2048, 4096, 8192), n_cells=24, condnum=True, seed=0):
    """Sample n_cells frames across topology / contrast axes, both splits, multiple scales.

    Default layout is 4 rows × 6 cols (24 cells). For other ``n_cells`` we keep 6 columns and
    set ``rows = ceil(n_cells / 6)``.
    """
    _apply_paper_style_compact()
    data_root = Path(data_root)
    candidates = []
    for scale in scales:
        for split in ("train", "test"):
            d = data_root / f"multiphase_v2_{scale}" / split
            if not d.is_dir():
                continue
            for fr in sorted(d.iterdir()):
                if (fr / "nodes.bin").exists():
                    candidates.append(fr)
    if not candidates:
        raise SystemExit(f"No multiphase_v2 frames under {data_root}")

    rng = np.random.default_rng(seed)
    picks = []
    seen_keys = set()
    rng.shuffle(candidates)
    for fr in candidates:
        info = load_frame(fr)
        # bucket by (scale, n_barriers_estimate, ratio_decade) so the 12 panels feel different
        nb = int(info["barrier"].sum() > 0) + int(info["barrier"].mean() > 0.05) + int(info["barrier"].mean() > 0.15)
        rdec = int(round(math.log10(max(info["ratio"], 1.001))))
        key = (info["N"], nb, rdec)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        picks.append(info)
        if len(picks) >= n_cells:
            break
    while len(picks) < n_cells and candidates:
        picks.append(load_frame(candidates[len(picks) % len(candidates)]))

    # condition numbers
    kappas = []
    for info in picks:
        if condnum:
            print(f"[variety] kappa for N={info['N']} ...")
            t0 = time.time()
            _, _, k = condition_number(info["A"])
            print(f"           kappa~{k:.2e}  ({time.time()-t0:.1f}s)")
            kappas.append(k)
        else:
            kappas.append(float("nan"))

    cols = 6
    rows = 4 if n_cells == 24 else max(1, int(math.ceil(n_cells / cols)))
    # Doubled per-panel canvas so on-page text reads ~50% the relative size of the previous
    # half-scale layout (i.e. plot cells twice as large for the same absolute pt fonts).
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.3 * rows), constrained_layout=True)
    axes = np.atleast_2d(axes).reshape(rows, cols)

    vmin = max(min(float(p["mass_flat"].min()) for p in picks), 1e-3)
    vmax = max(float(p["mass_flat"].max()) for p in picks)
    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        if idx >= len(picks):
            ax.axis("off")
            continue
        info = picks[idx]
        im = imshow_density(ax, info, vmin=vmin, vmax=vmax)
        kstr = (f"$\\kappa\\!\\approx\\!{kappas[idx]:.1e}$"
                if math.isfinite(kappas[idx]) else "$\\kappa$ n/a")
        # Two-line title so the metadata fits within the narrow single-column panel.
        ax.set_title(
            f"$N\\!=\\!{info['N']}$,  $\\rho_H/\\rho_L\\!=\\!{info['ratio']:.0f}\\times$\n{kstr}",
            fontsize=matplotlib.rcParams["axes.titlesize"] * 0.78,
            linespacing=0.95,
        )

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, location="right")
    cbar.set_label(r"$\rho$", fontsize=matplotlib.rcParams["axes.labelsize"])
    # Suptitle dropped: the paper.tex caption describes the three randomization axes.
    _savefig_paper(fig, out_path)
    plt.close(fig)
    print(f"[variety] -> {out_path}")


# ============================================================================
# ============================================================================
# M zoom strip: rank-normalised log10|M| at several top-left crop sizes
# ============================================================================
def make_m_zoom_strip(
    frame_dir,
    out_path,
    ours_weights,
    *,
    crops=(128, 256, 512, 1024),
    loader_3d: bool = False,
    show_titles: bool = True,
    show_axes: bool = True,
    show_colorbar: bool = True,
    tight_image: bool = False,
):
    """Horizontal strip showing the top-left crop×crop of the learned M at each size in *crops*.

    Each panel uses rank-normalised log10|·| with a shared colorbar so off-diagonal
    multiscale structure is comparable across zoom levels.
    ``loader_3d=True`` uses load_frame_3d; otherwise uses the 2-D load_frame.
    """
    _apply_paper_style_compact()
    load_fn = load_frame_3d if loader_3d else load_frame
    info = load_fn(frame_dir)
    N = info["N"]
    print(f"[m-zoom-strip] frame {frame_dir}  N={N}  ratio={info['ratio']:.1f}x")

    ours = OursPreconditioner(info, ours_weights)
    viz_n = ours._viz_n
    print(f"[m-zoom-strip] viz_n={viz_n}  assembling M …", flush=True)
    t0 = time.time()
    M_dense = ours.assembled_dense_M_numpy()
    print(f"[m-zoom-strip]   assembled M in {time.time() - t0:.1f}s")

    valid_crops = [c for c in crops if c <= viz_n]
    if not valid_crops:
        raise SystemExit(f"All requested crops exceed viz_n={viz_n}")

    n_panels = len(valid_crops)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4.8), constrained_layout=not tight_image)
    if n_panels == 1:
        axes = [axes]

    last_im = None
    for ax, crop in zip(axes, valid_crops):
        band_m = max(8, crop // 8)
        title = rf"$M$  top-left ${crop}\times{crop}$" if show_titles else ""
        im = _imshow_matrix_log10_abs(
            ax,
            M_dense,
            title,
            crop=crop,
            diag_band_exclude_for_scale=band_m,
            rank_normalize=True,
        )
        ax.set_box_aspect(1)
        if not show_axes:
            ax.set_axis_off()
        last_im = im

    if show_colorbar and last_im is not None:
        cb = fig.colorbar(last_im, ax=axes, shrink=0.75, pad=0.02)
        cb.set_label(r"off-diagonal rank of $\log_{10}|M|$", fontsize=11)

    if tight_image:
        # Image-only export: remove margins around the panel(s).
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
        fig.savefig(out_path, dpi=int(matplotlib.rcParams.get("savefig.dpi", 300)), bbox_inches="tight", pad_inches=0.0)
    else:
        _savefig_paper(fig, out_path)
    plt.close(fig)
    print(f"[m-zoom-strip] -> {out_path}")


# ============================================================================
# 3D residual propagation: opacity mapped to |r_k| at PCG step k
# ============================================================================
def _pcg_step_k(A_csr, b, apply_M, k=1):
    """Run exactly k steps of preconditioned CG from x=0; return (x_k, r_k)."""
    import scipy.sparse as _sp_local
    b = np.asarray(b, dtype=np.float64).ravel()
    x = np.zeros_like(b)
    r = b.copy()
    z = apply_M(r)
    p = z.copy()
    rho = float(r @ z)
    for _ in range(k):
        Ap = A_csr @ p
        pAp = float(p @ Ap)
        if pAp <= 0:
            break
        alpha = rho / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        z = apply_M(r)
        rho_new = float(r @ z)
        beta = rho_new / max(rho, 1e-300)
        p = z + beta * p
        rho = rho_new
    return x, r


def _scatter3d_field(ax, px, py, pz, mag, cmap, *,
                     top_frac=0.45, alpha_power=2.4, alpha_scurve=True,
                     elev=22.0, azim=-55.0, size_range=(4.0, 18.0), zorder=1):
    """Depth-sorted 3-D scatter with log-opacity encoding.  Returns (lo, hi) log10 range."""
    import math as _math
    thresh = float(np.percentile(mag, 100.0 * (1.0 - top_frac)))
    visible = mag >= thresh
    mag_v = mag[visible]
    log_v = np.log10(mag_v + 1e-30)
    lo = float(np.percentile(log_v, 2))
    hi = float(log_v.max())
    alpha_lin = np.clip((log_v - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
    if alpha_scurve:
        a = alpha_lin
        alpha_lin_sm = (a * a * (3.0 - 2.0 * a)).astype(np.float32)
    else:
        alpha_lin_sm = alpha_lin
    alpha_vals = np.clip(alpha_lin_sm, 0.0, 1.0) ** max(float(alpha_power), 1e-6)
    rgba = cmap(alpha_lin)
    rgba[:, 3] = alpha_vals
    vx = px[visible]; vy = py[visible]; vz = pz[visible]
    el_r = _math.radians(elev); az_r = _math.radians(azim)
    view_vec = np.array([_math.cos(el_r) * _math.cos(az_r),
                         _math.cos(el_r) * _math.sin(az_r),
                         _math.sin(el_r)], dtype=np.float32)
    depth = vx * view_vec[0] + vy * view_vec[1] + vz * view_vec[2]
    order = np.argsort(depth)
    s_lo, s_hi = size_range
    sizes = s_lo + (s_hi - s_lo) * alpha_vals[order] ** 0.5
    ax.scatter(vx[order], vy[order], vz[order],
               c=rgba[order], s=sizes, linewidths=0, depthshade=False, zorder=zorder)
    return lo, hi


def make_residual_3d(
    frame_dir,
    out_path,
    ours_weights,
    *,
    k: int = 1,
    cmap_name: str = "inferno",
    barrier_alpha: float = 0.14,
    elev: float = 22,
    azim: float = -55,
    dpi: int = 200,
    render_mode: str = "scatter",
    field: str = "residual",
    clean_view: bool = False,
    show_colorbar: bool = True,
    alpha_power: float = 2.4,
    alpha_scurve: bool = True,
    trace_seeds: int = 90,
    trace_steps: int = 48,
    trace_step_size: float = 0.65,
    trace_sigma: float = 1.2,
    zoom: float = 1.0,
    tight_image: bool = False,
):
    """3-D scatter / trace / dual figure for residual or preconditioner-correction fields.

    field="residual"   — existing behaviour: show |r_k| after k PCG steps.
    field="correction" — show |M⁻¹b|: the GNN preconditioner's response to the RHS.
                         Traces are seeded at high-|b| (barrier-interface) nodes and follow
                         the gradient of |M⁻¹b|, making non-local transport visible.
    field="dual"       — overlay both fields in one 3-D axis: cool-blue scatter for |b|
                         (local, interface-bound) and inferno scatter for |M⁻¹b| (global).
                         cmap_name controls the correction colourmap; source uses Blues.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import scipy.sparse as _sp_local

    _apply_paper_style_compact()
    info = load_frame_3d(frame_dir)
    N = info["N"]
    W, H, D = info["W"], info["H"], info["D"]
    print(f"[residual-3d] frame {frame_dir}  N={N}  ratio={info['ratio']:.1f}x  field={field}")

    ours = OursPreconditioner(info, ours_weights)
    viz_n = ours._viz_n

    A_csr = _sp_local.csr_matrix(
        _sp_local.coo_matrix(
            (ours._batch["edge_values"].numpy(),
             (ours._batch["edge_index"][0].numpy(), ours._batch["edge_index"][1].numpy())),
            shape=(N, N),
        )
    )
    A_act = A_csr[:viz_n, :viz_n]
    b_full = buoyancy_rhs_3d(info)[:viz_n]

    # Positions of the active nodes.
    raw_nodes = np.fromfile(Path(frame_dir) / "nodes.bin", dtype=NODE_DTYPE)[:viz_n]
    px = raw_nodes["position"][:, 0].astype(np.float32)
    py = raw_nodes["position"][:, 1].astype(np.float32)
    pz = raw_nodes["position"][:, 2].astype(np.float32)

    field_key = str(field).lower()

    # --- compute the scalar field(s) to visualise ---
    if field_key == "residual":
        print(f"[residual-3d] running {k} PCG step(s) …", flush=True)
        _x, r_k = _pcg_step_k(A_act, b_full, ours.apply_M, k=k)
        scalar_main = np.abs(r_k).astype(np.float64)
        seed_source = scalar_main          # seeds from high-residual nodes
        cmap_main = matplotlib.colormaps[cmap_name]
    elif field_key in ("correction", "dual"):
        print("[residual-3d] applying M⁻¹b …", flush=True)
        correction = np.abs(ours.apply_M(b_full)).astype(np.float64)
        b_mag = np.abs(b_full).astype(np.float64)
        scalar_main = correction
        seed_source = b_mag                # seeds from high-|b| (barrier interface) nodes
        cmap_main = matplotlib.colormaps[cmap_name]
    else:
        raise ValueError(f"Unknown field '{field}' — choose residual | correction | dual")

    fig = plt.figure(figsize=(8.0, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    # Half-cell shifted voxel corners.
    xg, yg, zg = np.indices((W + 1, H + 1, D + 1), dtype=np.float32)
    xg -= 0.5; yg -= 0.5; zg -= 0.5

    # Solid, lit barrier context (contiguous-looking blocks; no voxel edge grid).
    if info["barrier"].any() and barrier_alpha > 0:
        bfc = np.zeros(info["barrier"].shape + (4,), dtype=np.float32)
        bfc[..., :3] = 0.58
        bfc[..., 3] = 1.0
        ax.voxels(
            xg, yg, zg, info["barrier"],
            facecolors=bfc,
            edgecolors=None,
            linewidth=0.0,
            shade=True,
            zorder=0,
        )

    mode = str(render_mode).lower()

    # ------------------------------------------------------------------ dual
    if field_key == "dual":
        # Layer 1 — source |b|: cool blue, shows local barrier-interface forcing.
        cmap_src = matplotlib.colormaps.get_cmap("Blues")
        # Flip so high-|b| = saturated blue (Blues maps 0→white, 1→dark blue).
        lo_b, hi_b = _scatter3d_field(
            ax, px, py, pz, b_mag, cmap_src,
            top_frac=0.15,          # only top 15%: the thin interface shell
            alpha_power=3.0, alpha_scurve=True,
            elev=elev, azim=azim, size_range=(3.0, 12.0), zorder=1,
        )
        print(f"[residual-3d]   |b| log10 range [{lo_b:.2f}, {hi_b:.2f}]")

        # Layer 2 — correction |M⁻¹b|: inferno, shows non-local GNN spread.
        lo_c, hi_c = _scatter3d_field(
            ax, px, py, pz, correction, cmap_main,
            top_frac=0.40,          # top 40%: the global correction cloud
            alpha_power=float(alpha_power), alpha_scurve=bool(alpha_scurve),
            elev=elev, azim=azim, size_range=(4.0, 18.0), zorder=2,
        )
        print(f"[residual-3d]   |M⁻¹b| log10 range [{lo_c:.2f}, {hi_c:.2f}]")

        if show_colorbar:
            # Two mini colourbars stacked vertically.
            sm_src = matplotlib.cm.ScalarMappable(
                cmap=cmap_src,
                norm=matplotlib.colors.Normalize(vmin=lo_b, vmax=hi_b),
            )
            sm_src.set_array([])
            sm_cor = matplotlib.cm.ScalarMappable(
                cmap=cmap_main,
                norm=matplotlib.colors.Normalize(vmin=lo_c, vmax=hi_c),
            )
            sm_cor.set_array([])
            cb1 = fig.colorbar(sm_src, ax=ax, shrink=0.38, pad=0.08, aspect=22, location="right")
            cb1.set_label(r"$\log_{10}\,|b|$", fontsize=10)
            cb2 = fig.colorbar(sm_cor, ax=ax, shrink=0.38, pad=0.16, aspect=22, location="right")
            cb2.set_label(r"$\log_{10}\,|M^{-1}b|$", fontsize=10)

        lo, hi = lo_c, hi_c   # used by title / print below

    # ------------------------------------------------------------------ voxels
    elif mode == "voxels":
        thresh = float(np.percentile(scalar_main, 100.0 * (1 - 0.45)))
        visible = scalar_main >= thresh
        log_s = np.log10(scalar_main[visible] + 1e-30)
        lo = float(np.percentile(log_s, 2)); hi = float(log_s.max())
        alpha_lin = np.clip((log_s - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
        if alpha_scurve:
            a = alpha_lin; alpha_lin = (a * a * (3.0 - 2.0 * a)).astype(np.float32)
        alpha_vals = np.clip(alpha_lin, 0.0, 1.0) ** max(float(alpha_power), 1e-6)
        rgba_vis = cmap_main(alpha_lin); rgba_vis[:, 3] = alpha_vals
        vis_grid = np.zeros((W, H, D), dtype=bool)
        color_grid = np.zeros((W, H, D, 4), dtype=np.float32)
        vx_i = px[visible].astype(np.int32, copy=False)
        vy_i = py[visible].astype(np.int32, copy=False)
        vz_i = pz[visible].astype(np.int32, copy=False)
        vis_grid[vx_i, vy_i, vz_i] = True
        color_grid[vx_i, vy_i, vz_i, :] = rgba_vis
        edge_rgba = np.zeros_like(color_grid); edge_rgba[..., 3] = 0.0
        ax.voxels(xg, yg, zg, vis_grid, facecolors=color_grid,
                  edgecolors=edge_rgba, linewidth=0.0, shade=False, zorder=1)

        if show_colorbar:
            sm = matplotlib.cm.ScalarMappable(cmap=cmap_main,
                                               norm=matplotlib.colors.Normalize(vmin=lo, vmax=hi))
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.08, aspect=30)
            cb.set_label(rf"$\log_{{10}}\,|\cdot|$", fontsize=12)

    # ------------------------------------------------------------------ traces
    elif mode == "traces":
        from scipy import ndimage as _ndi
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        # Build the scalar grid for tracing (scalar_main) and the seed-source grid.
        s_grid = np.zeros((W, H, D), dtype=np.float32)
        src_grid = np.zeros((W, H, D), dtype=np.float32)
        px_i = px.astype(np.int32, copy=False)
        py_i = py.astype(np.int32, copy=False)
        pz_i = pz.astype(np.int32, copy=False)
        s_grid[px_i, py_i, pz_i] = scalar_main.astype(np.float32)
        src_grid[px_i, py_i, pz_i] = seed_source.astype(np.float32)

        log_s_grid = np.log10(np.maximum(s_grid, 1e-30))
        log_s_smooth = _ndi.gaussian_filter(log_s_grid, sigma=max(float(trace_sigma), 1e-6))

        gx, gy, gz = np.gradient(log_s_smooth)
        # Traces flow *toward* higher scalar (uphill gradient) — from seed toward high-correction.
        vx_f = gx.astype(np.float32); vy_f = gy.astype(np.float32); vz_f = gz.astype(np.float32)
        nrm_f = np.sqrt(vx_f**2 + vy_f**2 + vz_f**2) + 1e-12
        vx_f /= nrm_f; vy_f /= nrm_f; vz_f /= nrm_f

        # Streamline-style seeding: start from an even volumetric lattice inside the domain.
        n_seed_target = max(1, int(trace_seeds) * 3)
        domain_vol = max(float(W * H * D), 1.0)
        target_candidates = max(n_seed_target * 4, 64)
        spacing = (domain_vol / float(target_candidates)) ** (1.0 / 3.0)
        nx = max(3, int(np.ceil(W / max(spacing, 1e-6))))
        ny = max(3, int(np.ceil(H / max(spacing, 1e-6))))
        nz = max(3, int(np.ceil(D / max(spacing, 1e-6))))
        x_seed = np.linspace(0.5, max(0.5, W - 1.5), nx, dtype=np.float64)
        y_seed = np.linspace(0.5, max(0.5, H - 1.5), ny, dtype=np.float64)
        z_seed = np.linspace(0.5, max(0.5, D - 1.5), nz, dtype=np.float64)
        gx_s, gy_s, gz_s = np.meshgrid(x_seed, y_seed, z_seed, indexing="ij")
        seed_xyz = np.stack([gx_s.ravel(), gy_s.ravel(), gz_s.ravel()], axis=1)
        # Deterministic shuffle reduces directional striping from strict raster order.
        rng = np.random.default_rng(0)
        seed_xyz = seed_xyz[rng.permutation(seed_xyz.shape[0])]

        def _sample_scalar(F, x, y, z):
            if x < 0 or y < 0 or z < 0 or x > W - 1 or y > H - 1 or z > D - 1:
                return 0.0
            x0 = int(np.floor(x)); y0 = int(np.floor(y)); z0 = int(np.floor(z))
            x1 = min(x0 + 1, W - 1); y1 = min(y0 + 1, H - 1); z1 = min(z0 + 1, D - 1)
            tx = float(x - x0); ty = float(y - y0); tz = float(z - z0)
            c000 = float(F[x0, y0, z0]); c100 = float(F[x1, y0, z0])
            c010 = float(F[x0, y1, z0]); c110 = float(F[x1, y1, z0])
            c001 = float(F[x0, y0, z1]); c101 = float(F[x1, y0, z1])
            c011 = float(F[x0, y1, z1]); c111 = float(F[x1, y1, z1])
            c00 = c000*(1-tx)+c100*tx; c10 = c010*(1-tx)+c110*tx
            c01 = c001*(1-tx)+c101*tx; c11 = c011*(1-tx)+c111*tx
            c0 = c00*(1-ty)+c10*ty;    c1 = c01*(1-ty)+c11*ty
            return c0*(1-tz)+c1*tz

        def _sample_vec(x, y, z):
            vx = _sample_scalar(vx_f, x, y, z)
            vy = _sample_scalar(vy_f, x, y, z)
            vz = _sample_scalar(vz_f, x, y, z)
            nn = (vx*vx + vy*vy + vz*vz)**0.5
            if nn < 1e-8:
                return None
            return (vx/nn, vy/nn, vz/nn)

        # Use an absolute-distance streamline spacing check (continuous coordinates),
        # not voxel occupancy, so spacing is less aggressive on coarse grids.
        min_sep = max(0.18, 0.38 * max(float(trace_step_size), 1e-6))
        min_sep2 = float(min_sep * min_sep)
        cell = min_sep
        spatial: dict[tuple[int, int, int], list[tuple[float, float, float]]] = {}

        def _cell_key(x, y, z):
            return (
                int(np.floor(float(x) / cell)),
                int(np.floor(float(y) / cell)),
                int(np.floor(float(z) / cell)),
            )

        def _near_occupied(x, y, z):
            cx, cy, cz = _cell_key(x, y, z)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        pts = spatial.get((cx + dx, cy + dy, cz + dz))
                        if not pts:
                            continue
                        for px0, py0, pz0 in pts:
                            ddx = float(x) - px0
                            ddy = float(y) - py0
                            ddz = float(z) - pz0
                            if (ddx * ddx + ddy * ddy + ddz * ddz) < min_sep2:
                                return True
            return False

        def _mark_trace(xs, ys, zs):
            for x0, y0, z0 in zip(xs, ys, zs):
                key = _cell_key(x0, y0, z0)
                if key not in spatial:
                    spatial[key] = [(float(x0), float(y0), float(z0))]
                else:
                    spatial[key].append((float(x0), float(y0), float(z0)))

        def _trace(seed_x, seed_y, seed_z):
            xs = [float(seed_x)]; ys = [float(seed_y)]; zs = [float(seed_z)]
            x = float(seed_x); y = float(seed_y); z = float(seed_z)
            # Backward integration only (opposite to local transport direction).
            h = -max(float(trace_step_size), 1e-6)
            for _ in range(int(trace_steps)):
                v = _sample_vec(x, y, z)
                if v is None:
                    break
                x += h*v[0]; y += h*v[1]; z += h*v[2]
                if x < 0 or y < 0 or z < 0 or x > W-1 or y > H-1 or z > D-1:
                    break
                # Stop this streamline when it approaches an existing one.
                if _near_occupied(x, y, z):
                    break
                xs.append(x); ys.append(y); zs.append(z)
            return xs, ys, zs

        log_full = np.log10(np.maximum(scalar_main, 1e-30))
        lo = float(np.percentile(log_full, 2.0))
        hi = float(np.percentile(log_full, 99.5))
        if hi <= lo:
            hi = lo + 1e-6
        trace_norm = matplotlib.colors.Normalize(vmin=lo, vmax=hi)

        def _add_colored_trace(xs, ys, zs):
            if len(xs) < 2:
                return
            pts = np.stack(
                [np.asarray(xs, dtype=np.float64),
                 np.asarray(ys, dtype=np.float64),
                 np.asarray(zs, dtype=np.float64)],
                axis=1,
            )
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            mids = 0.5 * (pts[:-1] + pts[1:])
            seg_vals = np.asarray(
                [np.log10(max(_sample_scalar(s_grid, float(mx), float(my), float(mz)), 1e-30))
                 for (mx, my, mz) in mids],
                dtype=np.float64,
            )
            lc = Line3DCollection(
                segs,
                cmap=cmap_main,
                norm=trace_norm,
                linewidth=1.0,
                alpha=0.64,
                zorder=2,
            )
            lc.set_array(seg_vals)
            ax.add_collection3d(lc)

        n_accepted = 0
        for sx, sy, sz in seed_xyz:
            if n_accepted >= n_seed_target:
                break
            if _near_occupied(float(sx), float(sy), float(sz)):
                continue
            xb, yb, zb = _trace(float(sx), float(sy), float(sz))
            if len(xb) < 4:
                continue
            _add_colored_trace(xb, yb, zb)
            _mark_trace(xb, yb, zb)
            n_accepted += 1

        if show_colorbar:
            sm = matplotlib.cm.ScalarMappable(cmap=cmap_main,
                                               norm=matplotlib.colors.Normalize(vmin=lo, vmax=hi))
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.08, aspect=30)
            field_expr = r"|M^{-1}b|" if field_key == "correction" else rf"|r_{{{k}}}|"
            cb.set_label(rf"$\log_{{10}}\,{field_expr}$", fontsize=12)

    # ------------------------------------------------------------------ scatter (default)
    else:
        lo, hi = _scatter3d_field(
            ax, px, py, pz, scalar_main, cmap_main,
            top_frac=0.45,
            alpha_power=float(alpha_power), alpha_scurve=bool(alpha_scurve),
            elev=elev, azim=azim, size_range=(4.0, 18.0), zorder=1,
        )
        n_vis = int((scalar_main >= float(np.percentile(scalar_main, 55.0))).sum())
        print(f"[residual-3d]   {n_vis}/{viz_n} nodes shown")

        if show_colorbar:
            sm = matplotlib.cm.ScalarMappable(cmap=cmap_main,
                                               norm=matplotlib.colors.Normalize(vmin=lo, vmax=hi))
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.08, aspect=30)
            field_expr = r"|M^{-1}b|" if field_key == "correction" else rf"|r_{{{k}}}|"
            cb.set_label(rf"$\log_{{10}}\,{field_expr}$", fontsize=12)

    # Bounding-box wireframe / axis styling.
    if not clean_view:
        _panel_bbox_lines(ax, W, H, D, color=(0.3, 0.3, 0.3, 0.5), lw=0.6)
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_zlim(0, D)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor((0.85, 0.85, 0.85, 0.5))
        ax.grid(False)
    else:
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_zlim(0, D)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    # Force a noticeably stronger perspective camera.
    try:
        ax.set_proj_type("persp", focal_length=0.42)
    except TypeError:
        ax.set_proj_type("persp")
    # Pull camera closer to amplify perspective on older Matplotlib builds.
    if hasattr(ax, "dist"):
        ax.dist = max(0.5, 7.0 / max(float(zoom), 1e-6))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Optional edge-to-edge framing for figure exports.
    if tight_image:
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        try:
            ax.set_position([0.0, 0.0, 1.0, 1.0])
        except Exception:
            pass

    frame_name = Path(frame_dir).name
    if not clean_view:
        if field_key == "dual":
            ax.set_title(
                rf"$|b|$ (blue) vs $|M^{{-1}}b|$ (orange) — {frame_name}"
                rf"  ($\rho_H/\rho_L\!\approx\!{info['ratio']:.0f}\times$)",
                pad=6,
            )
        elif field_key == "correction":
            ax.set_title(
                rf"Preconditioner correction $|M^{{-1}}b|$ — {frame_name}"
                rf"  ($\rho_H/\rho_L\!\approx\!{info['ratio']:.0f}\times$)",
                pad=6,
            )
        else:
            ax.set_title(
                rf"Residual at $k={k}$ — {frame_name}"
                rf"  ($\rho_H/\rho_L\!\approx\!{info['ratio']:.0f}\times$)",
                pad=6,
            )

    if tight_image:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    else:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[residual-3d] -> {out_path}")
    print(f"[residual-3d]   log10 range [{lo:.2f}, {hi:.2f}]")


def make_residual_3d_propagation(
    frame_dir,
    out_path,
    ours_weights,
    *,
    ks=(1, 2, 4, 8),
    method: str = "ours",
    cmap_name: str = "inferno",
    voxel_quantile: float = 98.5,
    dpi: int = 220,
):
    """Readable multiscale 3D transport panel.

    Columns: PCG iteration k.
    Rows:
      1) XY max-intensity projection of |r_k|
      2) XZ max-intensity projection of |r_k|
      3) YZ max-intensity projection of |r_k|
      4) Sparse 3D voxel view (top-quantile |r_k| only)
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import scipy.sparse as _sp_local

    _apply_paper_style_compact()
    info = load_frame_3d(frame_dir)
    N = info["N"]
    W, H, D = info["W"], info["H"], info["D"]
    print(f"[residual-3d-prop] frame {frame_dir}  N={N}  ratio={info['ratio']:.1f}x")

    ours = OursPreconditioner(info, ours_weights)
    viz_n = ours._viz_n

    A_csr = _sp_local.csr_matrix(
        _sp_local.coo_matrix(
            (
                ours._batch["edge_values"].numpy(),
                (ours._batch["edge_index"][0].numpy(), ours._batch["edge_index"][1].numpy()),
            ),
            shape=(N, N),
        )
    )
    A_act = A_csr[:viz_n, :viz_n]
    b_full = buoyancy_rhs_3d(info)[:viz_n]

    method_key = str(method).lower()
    if method_key == "ours":
        apply_M = ours.apply_M
        method_label = "Ours preconditioned PCG"
    elif method_key == "jacobi":
        d = np.asarray(A_act.diagonal(), dtype=np.float64).ravel()
        inv = 1.0 / np.maximum(np.abs(d), 1e-30)

        def apply_M(r):
            return np.asarray(r, dtype=np.float64).ravel() * inv

        method_label = "Jacobi preconditioned PCG"
    elif method_key in ("none", "identity", "unpreconditioned"):
        def apply_M(r):
            return np.asarray(r, dtype=np.float64).ravel().copy()

        method_label = "Unpreconditioned PCG"
    else:
        raise ValueError(f"Unknown method '{method}' (use ours|jacobi|none)")

    ks = sorted(set(int(k) for k in ks if int(k) >= 1))
    if not ks:
        raise ValueError("ks must contain at least one positive iteration")

    raw_nodes = np.fromfile(Path(frame_dir) / "nodes.bin", dtype=NODE_DTYPE)[:viz_n]
    px = raw_nodes["position"][:, 0].astype(np.int32)
    py = raw_nodes["position"][:, 1].astype(np.int32)
    pz = raw_nodes["position"][:, 2].astype(np.int32)

    residual_grids = {}
    all_pos = []
    for k in ks:
        print(f"[residual-3d-prop] computing r_{k} …", flush=True)
        _x, r_k = _pcg_step_k(A_act, b_full, apply_M, k=k)
        g = np.full((W, H, D), np.nan, dtype=np.float64)
        g[px, py, pz] = np.abs(r_k).astype(np.float64)
        residual_grids[k] = g
        v = g[np.isfinite(g) & (g > 0)]
        if v.size:
            all_pos.append(v)

    if not all_pos:
        raise RuntimeError("No positive residual magnitudes for plotting.")
    cat = np.concatenate(all_pos)
    vmin = max(float(np.percentile(cat, 1.0)), 1e-30)
    vmax = max(float(np.percentile(cat, 99.5)), vmin * 1.01)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps[cmap_name]

    ncol = len(ks)
    fig = plt.figure(figsize=(3.6 * ncol, 12.0), constrained_layout=True)
    gs = GridSpec(4, ncol, figure=fig, height_ratios=(1.0, 1.0, 1.0, 1.25), hspace=0.10, wspace=0.06)

    # Precompute voxel corners centered on integer node coordinates.
    xg, yg, zg = np.indices((W + 1, H + 1, D + 1), dtype=np.float32)
    xg -= 0.5
    yg -= 0.5
    zg -= 0.5

    row_labels = ["XY MIP", "XZ MIP", "YZ MIP", "3D top-quantile voxels"]
    for ci, k in enumerate(ks):
        g = residual_grids[k]
        xy = np.nanmax(g, axis=2)   # (W, H)
        xz = np.nanmax(g, axis=1)   # (W, D)
        yz = np.nanmax(g, axis=0)   # (H, D)

        ax_xy = fig.add_subplot(gs[0, ci])
        ax_xz = fig.add_subplot(gs[1, ci])
        ax_yz = fig.add_subplot(gs[2, ci])
        ax_3d = fig.add_subplot(gs[3, ci], projection="3d")

        im0 = ax_xy.imshow(xy.T, origin="lower", cmap=cmap_name, norm=norm, interpolation="nearest", aspect="equal")
        ax_xz.imshow(xz.T, origin="lower", cmap=cmap_name, norm=norm, interpolation="nearest", aspect="equal")
        ax_yz.imshow(yz.T, origin="lower", cmap=cmap_name, norm=norm, interpolation="nearest", aspect="equal")
        for ax in (ax_xy, ax_xz, ax_yz):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("white")

        if ci == 0:
            ax_xy.set_ylabel(row_labels[0])
            ax_xz.set_ylabel(row_labels[1])
            ax_yz.set_ylabel(row_labels[2])
            ax_3d.set_ylabel(row_labels[3], labelpad=10)
        ax_xy.set_title(rf"$k={k}$")

        # Sparse, readable 3D layer: only highest residual quantile.
        finite_vals = g[np.isfinite(g)]
        q = max(0.0, min(100.0, float(voxel_quantile)))
        thr = float(np.percentile(finite_vals, q)) if finite_vals.size else float("inf")
        vis = np.isfinite(g) & (g >= thr) & (g > 0)

        fc = np.zeros(g.shape + (4,), dtype=np.float32)
        if np.any(vis):
            logv = np.log10(np.maximum(g[vis], 1e-30))
            lo = np.log10(vmin)
            hi = np.log10(vmax)
            t = np.clip((logv - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)
            # Strong S-curve + power to suppress middle-opacity haze.
            t_s = t * t * (3.0 - 2.0 * t)
            alpha = np.clip(t_s, 0.0, 1.0) ** 2.8
            cols = cmap(t)
            cols[:, 3] = alpha
            fc[vis] = cols

        ec = np.zeros_like(fc)
        ec[..., 3] = 0.0
        ax_3d.voxels(xg, yg, zg, vis, facecolors=fc, edgecolors=ec, linewidth=0.0, shade=False)
        ax_3d.set_xlim(-0.5, W - 0.5)
        ax_3d.set_ylim(-0.5, H - 0.5)
        ax_3d.set_zlim(-0.5, D - 0.5)
        ax_3d.set_box_aspect((1, 1, 1))
        ax_3d.set_axis_off()
        ax_3d.view_init(elev=22, azim=-55)
        ax_3d.set_facecolor("white")

    sm = matplotlib.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=fig.axes, shrink=0.42, pad=0.012, location="right")
    cb.set_label(r"$|r_k|$")

    fig.suptitle(
        rf"3D residual transport across PCG iterations — {method_label}  ($\rho_H/\rho_L\!\approx\!{info['ratio']:.0f}\times$)",
        y=0.995,
    )
    _savefig_paper(fig, out_path)
    plt.close(fig)
    print(f"[residual-3d-prop] -> {out_path}")


def _pressure_velocity_fields_3d(info):
    """Return buoyancy RHS, pressure, and velocity components/magnitude on a 3D frame."""
    A = info["A"]
    b = buoyancy_rhs_3d(info)
    ml = amg_solver(A)
    p = ml.solve(b, tol=1e-8, accel="cg", maxiter=2500).astype(np.float64)
    p -= p.mean()

    W, H, D = info["W"], info["H"], info["D"]
    xs, ys, zs = info["xs"], info["ys"], info["zs"]
    rho = info["mass_grid"].astype(np.float64) + 1e-6

    P = np.zeros((W, H, D), dtype=np.float64)
    P[xs, ys, zs] = p

    dpdx = np.zeros_like(P)
    dpdy = np.zeros_like(P)
    dpdz = np.zeros_like(P)
    dpdx[1:-1, :, :] = 0.5 * (P[2:, :, :] - P[:-2, :, :])
    dpdy[:, 1:-1, :] = 0.5 * (P[:, 2:, :] - P[:, :-2, :])
    dpdz[:, :, 1:-1] = 0.5 * (P[:, :, 2:] - P[:, :, :-2])

    vx = -dpdx / rho
    vy = -dpdy / rho
    vz = -dpdz / rho
    vmag = np.sqrt(vx * vx + vy * vy + vz * vz)
    return b, P, vx, vy, vz, vmag


def _scatter3d_particles(
    ax,
    px,
    py,
    pz,
    mag,
    *,
    cmap,
    log_lo,
    log_hi,
    top_frac=0.45,
    alpha_power=2.4,
    elev=22.0,
    azim=-55.0,
    size_range=(3.0, 14.0),
):
    """Draw depth-sorted particle scatter for a scalar node field."""
    mag = np.asarray(mag, dtype=np.float64).ravel()
    pos = mag > 0
    if not np.any(pos):
        return
    thr = float(np.percentile(mag[pos], 100.0 * (1.0 - float(top_frac))))
    vis = pos & (mag >= thr)
    if not np.any(vis):
        return

    vv = mag[vis]
    logv = np.log10(vv + 1e-30)
    t = np.clip((logv - float(log_lo)) / max(float(log_hi) - float(log_lo), 1e-9), 0.0, 1.0).astype(np.float32)
    t_s = t * t * (3.0 - 2.0 * t)
    alpha = np.clip(t_s, 0.0, 1.0) ** max(float(alpha_power), 1e-6)
    rgba = cmap(t)
    rgba[:, 3] = alpha

    vx = px[vis]
    vy = py[vis]
    vz = pz[vis]
    el_r = math.radians(float(elev))
    az_r = math.radians(float(azim))
    view_vec = np.array(
        [math.cos(el_r) * math.cos(az_r), math.cos(el_r) * math.sin(az_r), math.sin(el_r)],
        dtype=np.float32,
    )
    depth = vx * view_vec[0] + vy * view_vec[1] + vz * view_vec[2]
    order = np.argsort(depth)
    s_lo, s_hi = size_range
    sizes = s_lo + (s_hi - s_lo) * (alpha[order] ** 0.55)
    ax.scatter(vx[order], vy[order], vz[order], c=rgba[order], s=sizes, linewidths=0, depthshade=False)


def _draw_velocity_traces_3d(ax, vx, vy, vz, seeds_xyz, *, steps=34, step_size=0.55):
    """Overlay short white streamline-like traces on a 3D velocity field."""
    W, H, D = vx.shape

    def _sample_trilinear(F, x, y, z):
        if x < 0 or y < 0 or z < 0 or x > W - 1 or y > H - 1 or z > D - 1:
            return 0.0
        x0 = int(np.floor(x)); y0 = int(np.floor(y)); z0 = int(np.floor(z))
        x1 = min(x0 + 1, W - 1); y1 = min(y0 + 1, H - 1); z1 = min(z0 + 1, D - 1)
        tx = float(x - x0); ty = float(y - y0); tz = float(z - z0)
        c000 = float(F[x0, y0, z0]); c100 = float(F[x1, y0, z0])
        c010 = float(F[x0, y1, z0]); c110 = float(F[x1, y1, z0])
        c001 = float(F[x0, y0, z1]); c101 = float(F[x1, y0, z1])
        c011 = float(F[x0, y1, z1]); c111 = float(F[x1, y1, z1])
        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        return c0 * (1 - tz) + c1 * tz

    for sx, sy, sz in seeds_xyz:
        xs = [float(sx)]
        ys = [float(sy)]
        zs = [float(sz)]
        x = float(sx)
        y = float(sy)
        z = float(sz)
        for _ in range(int(steps)):
            vx_s = _sample_trilinear(vx, x, y, z)
            vy_s = _sample_trilinear(vy, x, y, z)
            vz_s = _sample_trilinear(vz, x, y, z)
            nrm = float(np.sqrt(vx_s * vx_s + vy_s * vy_s + vz_s * vz_s))
            if nrm < 1e-8:
                break
            x += float(step_size) * vx_s / nrm
            y += float(step_size) * vy_s / nrm
            z += float(step_size) * vz_s / nrm
            if x < 0 or y < 0 or z < 0 or x > W - 1 or y > H - 1 or z > D - 1:
                break
            xs.append(x)
            ys.append(y)
            zs.append(z)
        if len(xs) > 2:
            ax.plot(xs, ys, zs, color=(1.0, 1.0, 1.0, 0.52), linewidth=0.62)


def _draw_pressure_slices_3d(ax, P_grid, *, cmap_name="magma", alpha=0.90):
    """Render 3 orthogonal pressure slices to emphasize field structure."""
    W, H, D = P_grid.shape
    mx, my, mz = W // 2, H // 2, D // 2
    P_abs = np.abs(P_grid).astype(np.float64)
    v = P_abs[P_abs > 0]
    if v.size:
        lo = max(float(np.percentile(v, 1.0)), 1e-30)
        hi = max(float(np.percentile(v, 99.5)), lo * 1.01)
    else:
        lo, hi = 1e-9, 1e-6
    norm = LogNorm(vmin=lo, vmax=hi)
    cmap = matplotlib.colormaps[cmap_name]

    yg, zg = np.meshgrid(np.arange(H), np.arange(D), indexing="ij")
    xg, zg2 = np.meshgrid(np.arange(W), np.arange(D), indexing="ij")
    xg2, yg2 = np.meshgrid(np.arange(W), np.arange(H), indexing="ij")

    c_x = cmap(norm(np.abs(P_grid[mx, :, :])))
    c_y = cmap(norm(np.abs(P_grid[:, my, :])))
    c_z = cmap(norm(np.abs(P_grid[:, :, mz])))
    c_x[..., 3] = alpha
    c_y[..., 3] = alpha
    c_z[..., 3] = alpha

    ax.plot_surface(
        np.full_like(yg, mx, dtype=np.float64),
        yg.astype(np.float64),
        zg.astype(np.float64),
        facecolors=c_x,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax.plot_surface(
        xg.astype(np.float64),
        np.full_like(xg, my, dtype=np.float64),
        zg2.astype(np.float64),
        facecolors=c_y,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax.plot_surface(
        xg2.astype(np.float64),
        yg2.astype(np.float64),
        np.full_like(xg2, mz, dtype=np.float64),
        facecolors=c_z,
        linewidth=0,
        antialiased=False,
        shade=False,
    )


def _draw_velocity_quiver_3d(ax, vx, vy, vz, *, step=4, cmap_name="magma"):
    """Subsampled 3D velocity quiver, color-coded by speed."""
    W, H, D = vx.shape
    xs = np.arange(0, W, max(1, int(step)))
    ys = np.arange(0, H, max(1, int(step)))
    zs = np.arange(0, D, max(1, int(step)))
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    u = vx[gx, gy, gz]
    v = vy[gx, gy, gz]
    w = vz[gx, gy, gz]
    speed = np.sqrt(u * u + v * v + w * w)
    good = speed > 1e-10
    if not np.any(good):
        return

    gxx = gx[good].astype(np.float64)
    gyy = gy[good].astype(np.float64)
    gzz = gz[good].astype(np.float64)
    uu = u[good].astype(np.float64)
    vv = v[good].astype(np.float64)
    ww = w[good].astype(np.float64)
    ss = speed[good].astype(np.float64)

    lo = max(float(np.percentile(ss, 5.0)), 1e-30)
    hi = max(float(np.percentile(ss, 99.5)), lo * 1.01)
    norm = LogNorm(vmin=lo, vmax=hi)
    cols = matplotlib.colormaps[cmap_name](norm(ss))
    cols[:, 3] = 0.86

    nrm = np.sqrt(uu * uu + vv * vv + ww * ww) + 1e-12
    uu /= nrm
    vv /= nrm
    ww /= nrm
    ax.quiver(gxx, gyy, gzz, uu, vv, ww, length=1.55, normalize=False, colors=cols, linewidths=0.55)


def make_residual_3d_compare(
    frame_dir,
    out_path,
    *,
    ours_weights,
    neuralspai_checkpoint,
    lsp_repo="/orcd/home/002/osbo/ondemand/LearningSparsePreconditioner4GPU",
    rtol=1e-8,
    max_iter=6000,
    n_panels=5,
    elev=22.0,
    azim=-55.0,
):
    """3D Fig-1c-style panel with problem row + residual rows for Jacobi/AMG/NeuralSPAI/Ours."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _apply_paper_style_compact()
    info = load_frame_3d(frame_dir)
    print(f"[residual-3d-compare] frame={frame_dir} N={info['N']} ratio={info['ratio']:.1f}x")

    A = info["A"]
    b = buoyancy_rhs_3d(info)
    nodes = np.fromfile(Path(frame_dir) / "nodes.bin", dtype=NODE_DTYPE)
    px = nodes["position"][:, 0].astype(np.float32)
    py = nodes["position"][:, 1].astype(np.float32)
    pz = nodes["position"][:, 2].astype(np.float32)
    rho = np.asarray(info["mass_flat"], dtype=np.float64)
    b_mag = np.abs(b).astype(np.float64)

    print("[residual-3d-compare] solving pressure/velocity via AMG ...", flush=True)
    _, P_grid, vx_grid, vy_grid, vz_grid, vmag_grid = _pressure_velocity_fields_3d(info)
    p_mag = np.abs(P_grid[info["xs"], info["ys"], info["zs"]]).astype(np.float64)
    vmag = np.abs(vmag_grid[info["xs"], info["ys"], info["zs"]]).astype(np.float64)

    if not neuralspai_checkpoint:
        raise SystemExit(
            "make_residual_3d_compare requires --neuralspai-checkpoint "
            "(explicit LearningSparsePreconditioner4GPU checkpoint path)."
        )
    if not ours_weights:
        raise SystemExit(
            "make_residual_3d_compare requires --ours-weights "
            "(LeafOnly/Ours .bytes checkpoint for the same scale/domain)."
        )
    print(f"[residual-3d-compare] loading NeuralSPAI checkpoint {neuralspai_checkpoint}", flush=True)
    neural_spai = NeuralSPAIPreconditioner(info, neuralspai_checkpoint, lsp_repo=lsp_repo)
    print(f"[residual-3d-compare] loading Ours weights {ours_weights}", flush=True)
    ours = OursPreconditioner(info, ours_weights)

    def _infer_ours_effective_n(ours_prec, full_n: int) -> int:
        """Infer input size expected by ``ours_prec.apply_M`` on this frame."""
        n_full = int(full_n)
        # Fast path: full size works.
        try:
            _ = ours_prec.apply_M(np.zeros(n_full, dtype=np.float64))
            return n_full
        except ValueError as e:
            import re as _re

            msg = str(e)
            # Typical failure: "could not broadcast input array from shape (8000,) into shape (7936,)"
            m = _re.search(r"into shape \((\d+),\)", msg)
            if m:
                n = int(m.group(1))
                n = max(1, min(n, n_full))
                return n
            # Fallback to the internal attribute when available.
            n_attr = int(getattr(ours_prec, "_real_n", n_full))
            n_attr = max(1, min(n_attr, n_full))
            return n_attr

    ours_real_n = _infer_ours_effective_n(ours, int(info["N"]))
    print(f"[residual-3d-compare] Ours effective system size={ours_real_n}", flush=True)
    A_ours = A[:ours_real_n, :ours_real_n].tocsr()
    b_ours = np.asarray(b[:ours_real_n], dtype=np.float64)

    methods = [
        {"name": "Jacobi", "apply": _build_jacobi_apply(A), "A": A, "b": b, "lift_n": int(info["N"])},
        {"name": "AMG", "apply": _build_amg_apply(A), "A": A, "b": b, "lift_n": int(info["N"])},
        {"name": "NeuralSPAI", "apply": neural_spai.apply_M, "A": A, "b": b, "lift_n": int(info["N"])},
        {"name": "Ours", "apply": ours.apply_M, "A": A_ours, "b": b_ours, "lift_n": ours_real_n},
    ]
    n_panels = max(3, int(n_panels))
    snap_candidates_long = tuple(sorted(set(_PCG_SNAP_CANDIDATES + tuple(range(1, 40)))))
    method_runs = []
    for method in methods:
        name = str(method["name"])
        apply_M = method["apply"]
        A_use = method["A"]
        b_use = np.asarray(method["b"], dtype=np.float64).ravel()
        lift_n = int(method["lift_n"])
        print(
            f"[residual-3d-compare] PCG method={name} ... "
            f"(system size {A_use.shape[0]})",
            flush=True,
        )
        snaps, meta = pcg_preconditioned_full(
            A_use,
            b_use,
            apply_M,
            rtol=float(rtol),
            max_iter=int(max_iter),
            snap_candidates=snap_candidates_long,
            max_snapshots=10000,
        )
        chosen = _select_method_snap_iters(snaps, int(meta["iters"]), n_panels=n_panels)
        snaps_lifted = {}
        full_n = int(info["N"])
        for k in chosen:
            r_k = np.asarray(snaps[k], dtype=np.float64).ravel()
            if lift_n == full_n:
                snaps_lifted[k] = r_k
            else:
                r_full = np.zeros(full_n, dtype=np.float64)
                n_copy = min(lift_n, r_k.shape[0], full_n)
                r_full[:n_copy] = r_k[:n_copy]
                snaps_lifted[k] = r_full
        method_runs.append(
            {
                "name": name,
                "K": int(meta["iters"]),
                "converged": bool(meta["converged"]),
                "snaps": snaps_lifted,
                "iters_chosen": chosen,
            }
        )
        print(
            f"[residual-3d-compare]   {name}: K={int(meta['iters'])} "
            f"converged={bool(meta['converged'])} snaps={chosen}"
        )

    all_pos = []
    for run in method_runs:
        for r in run["snaps"].values():
            rr = np.abs(np.asarray(r, dtype=np.float64).ravel())
            rr = rr[rr > 0]
            if rr.size:
                all_pos.append(rr)
    if not all_pos:
        raise RuntimeError("No positive residual magnitudes available for residual-3d-compare.")
    cat = np.concatenate(all_pos)
    r_vmin = max(float(np.percentile(cat, 0.5)), 1e-30)
    r_vmax = max(float(np.percentile(cat, 99.5)), r_vmin * 1.01)
    r_log_lo = float(np.log10(r_vmin))
    r_log_hi = float(np.log10(r_vmax))

    magma = matplotlib.colormaps["magma"]

    def _field_log_bounds(v):
        vv = np.asarray(v, dtype=np.float64).ravel()
        vv = vv[vv > 0]
        if vv.size == 0:
            return -8.0, -6.0
        lo = float(np.log10(max(np.percentile(vv, 1.0), 1e-30)))
        hi = float(np.log10(max(np.percentile(vv, 99.5), 1e-29)))
        if hi <= lo:
            hi = lo + 1e-6
        return lo, hi

    rho_lo, rho_hi = _field_log_bounds(rho)
    b_lo, b_hi = _field_log_bounds(b_mag)
    p_lo, p_hi = _field_log_bounds(p_mag)
    v_lo, v_hi = _field_log_bounds(vmag)

    fig_w = 3.2 * n_panels + 1.25
    fig_h = 3.0 * (1 + len(method_runs)) + 0.9
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    outer = GridSpec(1, 2, figure=fig, width_ratios=(1.0, 0.026), wspace=0.015)
    all_grid = GridSpecFromSubplotSpec(
        1 + len(method_runs),
        n_panels,
        outer[0, 0],
        wspace=0.05,
        hspace=0.10,
        height_ratios=(1.0,) + tuple(1.0 for _ in method_runs),
    )

    top_3d = [
        fig.add_subplot(all_grid[0, 0], projection="3d"),
        fig.add_subplot(all_grid[0, 1], projection="3d"),
        fig.add_subplot(all_grid[0, 2], projection="3d"),
        fig.add_subplot(all_grid[0, 3], projection="3d"),
    ]
    ax_m = fig.add_subplot(all_grid[0, 4])

    _scatter3d_particles(top_3d[0], px, py, pz, rho, cmap=magma, log_lo=rho_lo, log_hi=rho_hi, top_frac=0.62, elev=elev, azim=azim)
    _scatter3d_particles(top_3d[1], px, py, pz, b_mag, cmap=magma, log_lo=b_lo, log_hi=b_hi, top_frac=0.48, elev=elev, azim=azim)
    _draw_pressure_slices_3d(top_3d[2], P_grid, cmap_name="magma", alpha=0.90)
    _draw_velocity_quiver_3d(top_3d[3], vx_grid, vy_grid, vz_grid, step=4, cmap_name="magma")
    _scatter3d_particles(top_3d[3], px, py, pz, vmag, cmap=magma, log_lo=v_lo, log_hi=v_hi, top_frac=0.30, elev=elev, azim=azim, size_range=(2.5, 9.0))

    # White velocity traces over the velocity-magnitude panel.
    seed_count = 64
    v_rank = np.argsort(vmag)
    seed_idx = v_rank[-seed_count:] if v_rank.size > seed_count else v_rank
    seeds_xyz = np.stack([px[seed_idx], py[seed_idx], pz[seed_idx]], axis=1)
    _draw_velocity_traces_3d(top_3d[3], vx_grid, vy_grid, vz_grid, seeds_xyz, steps=34, step_size=0.55)

    for ax, ttl in zip(
        top_3d,
        [r"Topology $\rho$", r"RHS $|b|$", r"Pressure $|p|$", r"Velocity $|v|$ + traces"],
    ):
        ax.set_xlim(0, info["W"])
        ax.set_ylim(0, info["H"])
        ax.set_zlim(0, info["D"])
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(ttl, fontsize=12.8, pad=4)
        ax.set_facecolor("white")

    M_dense = ours.assembled_dense_M_numpy()
    mc = max(16, min(256, int(M_dense.shape[0])))
    _imshow_matrix_log10_abs(
        ax_m,
        M_dense,
        rf"Ours $M$" + "\n" + rf"top-left ${mc}\times{mc}$",
        crop=mc,
        cmap="magma",
        rank_normalize=True,
    )
    ax_m.set_box_aspect(1)

    cax = fig.add_subplot(outer[0, 1])
    for mi, run in enumerate(method_runs):
        keys = run["iters_chosen"]
        for j in range(n_panels):
            ax = fig.add_subplot(all_grid[mi + 1, j], projection="3d")
            if j < len(keys):
                k = keys[j]
                rr = np.abs(np.asarray(run["snaps"][k], dtype=np.float64).ravel())
                _scatter3d_particles(
                    ax,
                    px,
                    py,
                    pz,
                    rr,
                    cmap=magma,
                    log_lo=r_log_lo,
                    log_hi=r_log_hi,
                    top_frac=0.45,
                    alpha_power=2.5,
                    elev=elev,
                    azim=azim,
                )
                ax.set_title(rf"$k={k}$", fontsize=11.5, pad=3)
            else:
                ax.set_axis_off()
            ax.set_xlim(0, info["W"])
            ax.set_ylim(0, info["H"])
            ax.set_zlim(0, info["D"])
            ax.set_box_aspect((1, 1, 1))
            ax.set_axis_off()
            ax.view_init(elev=elev, azim=azim)
            if j == 0:
                conv = "" if run["converged"] else " (no conv.)"
                ax.text2D(
                    -0.24,
                    0.52,
                    f"{run['name']}\nK={run['K']}{conv}",
                    transform=ax.transAxes,
                    fontsize=12.5,
                    ha="left",
                    va="center",
                )

    sm = matplotlib.cm.ScalarMappable(cmap="magma", norm=LogNorm(vmin=r_vmin, vmax=r_vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"$|r_k|$")

    fig.suptitle(
        rf"3D multiscale residual transport (particles, opacity-coded)  "
        rf"($\rho_H/\rho_L\!\approx\!{info['ratio']:.0f}\times$)",
        y=0.995,
    )
    _savefig_paper(fig, out_path)
    plt.close(fig)
    print(f"[residual-3d-compare] -> {out_path}")


# Figure 2b: full 3D variety grid (all train + test frames in a 3D bundle)
# ============================================================================
def _panel_bbox_lines(ax, W, H, D, color=(0.45, 0.45, 0.45, 0.55), lw=0.4):
    corners = [
        (0, 0, 0), (W, 0, 0), (W, H, 0), (0, H, 0),
        (0, 0, D), (W, 0, D), (W, H, D), (0, H, D),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        x0, y0, z0 = corners[a]
        x1, y1, z1 = corners[b]
        ax.plot([x0, x1], [y0, y1], [z0, z1], color=color, linewidth=lw)


def make_variety_3d(
    data_root,
    out_path,
    bundle_name="multiphase_v2_3d_8000",
    *,
    rows=12,
    cols=10,
    dpi=130,
):
    """Render every frame in the 3D bundle as a voxel panel in an ``rows x cols`` grid.

    Train frames are filled first (top-left → right → down). Test frames follow
    in a contrasting tint after the train frames are exhausted.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' projection)

    data_root = Path(data_root)
    train_dir = data_root / bundle_name / "train"
    test_dir = data_root / bundle_name / "test"
    train_frames = (
        sorted(f for f in train_dir.iterdir() if (f / "nodes.bin").exists())
        if train_dir.is_dir() else []
    )
    test_frames = (
        sorted(f for f in test_dir.iterdir() if (f / "nodes.bin").exists())
        if test_dir.is_dir() else []
    )
    frames: list[tuple[Path, str]] = [(f, "train") for f in train_frames] + [(f, "test") for f in test_frames]
    if not frames:
        raise SystemExit(f"No frames under {data_root / bundle_name}")

    n_total = rows * cols
    if len(frames) > n_total:
        print(f"[variety-3d] {len(frames)} frames but only {n_total} cells; truncating tail.")
        frames = frames[:n_total]

    train_face = "#3a4060"
    test_face = "#883a3a"
    edge_rgba = (0.0, 0.0, 0.0, 0.18)

    panel_w = 1.55
    panel_h = 1.65
    rc_overrides = {
        "font.family": "serif",
        "font.size": 7,
        "axes.titlesize": 6,
        "figure.titlesize": 12,
    }
    t0 = time.time()
    with matplotlib.rc_context(rc_overrides):
        fig = plt.figure(figsize=(panel_w * cols, panel_h * rows), constrained_layout=False)
        for idx in range(n_total):
            ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
            if idx >= len(frames):
                ax.set_axis_off()
                continue
            fr, split = frames[idx]
            info = load_frame_3d(fr)
            barrier = info["barrier"]
            W, H, D = info["W"], info["H"], info["D"]
            face_rgba = matplotlib.colors.to_rgba(test_face if split == "test" else train_face)

            if barrier.any():
                facecolors = np.zeros(barrier.shape + (4,), dtype=np.float32)
                facecolors[..., 0] = face_rgba[0]
                facecolors[..., 1] = face_rgba[1]
                facecolors[..., 2] = face_rgba[2]
                facecolors[..., 3] = 1.0
                edgecolors = np.zeros_like(facecolors)
                edgecolors[..., 3] = edge_rgba[3]
                ax.voxels(barrier, facecolors=facecolors, edgecolors=edgecolors, linewidth=0.05)

            _panel_bbox_lines(ax, W, H, D)
            ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_zlim(0, D)
            ax.set_box_aspect((1, 1, 1))
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
            ax.view_init(elev=22, azim=-55)
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor((1, 1, 1, 0))
            ax.grid(False)

            tag = fr.name.replace("frame_", "")
            title_color = test_face if split == "test" else "black"
            ax.set_title(
                f"{split[0]}{tag}  $\\rho_H/\\rho_L\\!=\\!{info['ratio']:.0f}\\times$",
                fontsize=5.5,
                color=title_color,
                pad=1.5,
            )
            if (idx + 1) % 24 == 0 or idx + 1 == len(frames):
                print(f"[variety-3d] panel {idx + 1}/{len(frames)}  ({time.time() - t0:.1f}s)")

        fig.suptitle(
            f"multiphase_v2 3D variety — {bundle_name}  "
            f"({len(train_frames)} train + {len(test_frames)} test)",
            y=0.998,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.995))
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    print(f"[variety-3d] -> {out_path}  ({time.time() - t0:.1f}s)")


# ============================================================================
# Figure 3: convergence-in-action
# ============================================================================
def pick_stiff_frame(data_root, scale, n_probe=8, seed=0):
    """Find the highest-kappa frame among a few random candidates at the given scale."""
    data_root = Path(data_root)
    frames = []
    for split in ("test", "train"):
        d = data_root / f"multiphase_v2_{scale}" / split
        if d.is_dir():
            frames.extend(sorted(d.iterdir()))
    rng = np.random.default_rng(seed)
    rng.shuffle(frames)
    frames = [f for f in frames if (f / "nodes.bin").exists()][:n_probe]
    best = None
    best_kappa = -1.0
    for fr in frames:
        info = load_frame(fr)
        _, _, k = condition_number(info["A"], tol=1e-2)
        print(f"  candidate {fr.name}  ratio={info['ratio']:.1f}x  kappa~{k:.2e}")
        if math.isfinite(k) and k > best_kappa:
            best_kappa = k
            best = (fr, info, k)
    if best is None:
        raise SystemExit(f"No usable frame at scale {scale}")
    return best


def make_matrices(
    frame_dir,
    out_path,
    ours_weights,
    *,
    matrix_max_n: int = 65536,
    crop: int = 256,
):
    """Three-view ``A`` / ``A^{-1}`` / ``M`` heatmap. Companion to fig:rank-headroom.

    Renders the top-left ``crop x crop`` of each matrix at log10 |·|; the inverse and learned M
    panels share a peak-relative log scale so their off-diagonal multiscale structure can be
    compared directly. Intended as a single-row figure that sits next to the rank-headroom plot.
    """
    _apply_paper_style_compact()
    if not ours_weights:
        raise SystemExit(
            "The matrices figure requires --ours-weights (a trained .bytes checkpoint for this frame scale)."
        )
    info = load_frame(frame_dir)
    print(f"[matrices] frame {frame_dir}  N={info['N']}  ratio={info['ratio']:.1f}x")

    ours = OursPreconditioner(info, ours_weights)
    viz_n = ours._viz_n

    if viz_n > matrix_max_n:
        raise SystemExit(
            f"[matrices] padded n={viz_n} exceeds --matrices-matrix-max-n={matrix_max_n}; "
            "use a smaller scale or raise the cap."
        )

    t0 = time.time()
    A_dense = ours.dense_A_numpy()
    print(f"[matrices]   dense A in {time.time() - t0:.1f}s")
    t1 = time.time()
    A_inv = _dense_inv_numpy_or_torch(A_dense, ours.device)
    print(f"[matrices]   inv(A) in {time.time() - t1:.1f}s")
    t2 = time.time()
    M_dense = ours.assembled_dense_M_numpy()
    print(f"[matrices]   assembled M in {time.time() - t2:.1f}s")

    mc = min(int(crop), viz_n)
    band_m = max(8, mc // 8)

    fig = plt.figure(figsize=(11.5, 4.0), constrained_layout=True)
    axes = [fig.add_subplot(1, 3, j + 1) for j in range(3)]

    # A: pure binary sparsity pattern (nonzero → white, zero → black). The interesting
    # story for A is the 5-point Laplacian stencil shape, not the coefficient magnitudes,
    # so we deliberately discard all numerical info and show only "is this entry nonzero".
    A_binary = (np.abs(np.asarray(A_dense)[:mc, :mc]) > 0).astype(np.float64)
    im_a = axes[0].imshow(
        A_binary,
        cmap="gray",
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title(rf"$A$ (sparsity pattern)" + "\n" + rf"top-left ${mc}\times{mc}$")
    axes[0].set_box_aspect(1)

    im_ai = _imshow_matrix_log10_abs(
        axes[1],
        A_inv,
        rf"$A^{{-1}}$ (rank-normalized)" + "\n" + rf"top-left ${mc}\times{mc}$",
        crop=mc,
        diag_band_exclude_for_scale=band_m,
        rank_normalize=True,
    )
    axes[1].set_box_aspect(1)
    im_m = _imshow_matrix_log10_abs(
        axes[2],
        M_dense,
        rf"Learned $M$ (rank-normalized)" + "\n" + rf"top-left ${mc}\times{mc}$",
        crop=mc,
        diag_band_exclude_for_scale=band_m,
        rank_normalize=True,
    )
    axes[2].set_box_aspect(1)

    # Single shared colorbar for the two rank-normalized panels (both are on [0, 1]).
    cb_rn = fig.colorbar(im_m, ax=[axes[1], axes[2]], shrink=0.85, pad=0.02)
    cb_rn.ax.tick_params(labelsize=10)
    cb_rn.set_label(r"off-diagonal rank of $\log_{10}|\cdot|$", fontsize=11)

    _savefig_paper(fig, out_path)
    plt.close(fig)
    print(f"[matrices] -> {out_path}")


def make_convergence(
    data_root,
    scale,
    out_path,
    ours_weights,
    *,
    seed=0,
    override_frame=None,
    rtol=1e-8,
    max_iter=5000,
    neuralspai_checkpoint=None,
    lsp_repo="/orcd/home/002/osbo/ondemand/LearningSparsePreconditioner4GPU",
    lsp_summary_csv=None,
):
    """Multi-method convergence figure. Two-column-wide; top row is rho + b (the *problem*),
    then five method rows each showing five |r_k| snapshots taken at geometrically-spaced
    iterations (log-spaced from 1 to ~0.9*K) of *that method's* own PCG run. Early iterations
    get more representation (where structure is richest); the last panel pulls back from K so
    it does not just render as black. Methods (top → bottom): unpreconditioned, Jacobi, IC,
    AMG, ours. A single global log colorbar spans all residual panels so spatial spread of
    error (impulse propagation) is comparable across methods.
    """
    _apply_paper_style_compact()
    if not ours_weights:
        raise SystemExit(
            "The convergence figure requires --ours-weights (a trained .bytes checkpoint for this frame scale)."
        )
    if override_frame is not None:
        info = load_frame(override_frame)
        _, _, kappa = condition_number(info["A"], tol=1e-2)
        frame_path = Path(override_frame)
    else:
        frame_path, info, kappa = pick_stiff_frame(data_root, scale, seed=seed)
    print(f"[conv] picked {frame_path}  kappa~{kappa:.2e}  ratio={info['ratio']:.0f}x")

    A = info["A"]
    b = buoyancy_rhs(info)

    # Build every preconditioner up front (so a setup failure aborts before any solve).
    print(f"[conv] building baselines: identity / Jacobi / IC / AMG / NeuralSPAI …")
    apply_identity = _build_identity_apply(info["N"])
    apply_jacobi = _build_jacobi_apply(A)
    apply_ic, ic_backend = _build_ic_apply(A)
    print(f"[conv]   IC backend: {ic_backend}")
    apply_amg_fn = _build_amg_apply(A)

    print(f"[conv] loading preconditioner weights {ours_weights}")
    ours = OursPreconditioner(info, ours_weights)
    viz_n = ours._viz_n
    if info["N"] != viz_n:
        print(
            f"[conv] note: frame N={info['N']} vs padded operator n={viz_n} "
            "(PCG runs on physical A; ours uses the padded preconditioner).",
            file=sys.stderr,
        )

    if neuralspai_checkpoint is None:
        if lsp_summary_csv is None:
            lsp_summary_csv = SCRIPTS_DIR / "results" / "lsp_scale_infer_summary.csv"
        neuralspai_checkpoint = _find_neural_spai_checkpoint(int(scale), Path(lsp_summary_csv))
    print(f"[conv] loading NeuralSPAI checkpoint {neuralspai_checkpoint}")
    neural_spai = NeuralSPAIPreconditioner(info, neuralspai_checkpoint, lsp_repo=lsp_repo)

    # Methods listed top → bottom. Display labels match the paper text.
    methods = [
        ("Unpreconditioned", apply_identity),
        ("Jacobi", apply_jacobi),
        ("IC", apply_ic),
        ("AMG", apply_amg_fn),
        ("NeuralSPAI", neural_spai.apply_M),
        ("Ours", ours.apply_M),
    ]
    n_methods = len(methods)
    n_panels_per_method = 5

    # Per-method PCG runs. We keep a generous candidate list during the solve, then prune to
    # the (1, 2, 3, mid, near-end) selection after we know each method's own K.
    snap_candidates_long = tuple(sorted(set(_PCG_SNAP_CANDIDATES + tuple(range(1, 32)))))
    method_runs = []  # list of dicts: name, snaps (dict iter→r), K, iters_chosen, converged, final_rel
    for (name, apply_M) in methods:
        print(f"[conv] PCG running: method='{name}' (max_iter={max_iter}, rtol={rtol:g})")
        snaps_full, meta = pcg_preconditioned_full(
            A,
            b,
            apply_M,
            rtol=rtol,
            max_iter=max_iter,
            snap_candidates=snap_candidates_long,
            snap_iters_explicit=None,
            max_snapshots=10_000,  # don't prune yet; we'll pick exactly 5 after.
        )
        K = int(meta["iters"])
        chosen = _select_method_snap_iters(snaps_full, K, n_panels=n_panels_per_method)
        snaps_chosen = {k: snaps_full[k] for k in chosen}
        method_runs.append({
            "name": name,
            "K": K,
            "converged": bool(meta["converged"]),
            "final_rel": float(meta["final_rel"]),
            "iters_chosen": chosen,
            "snaps": snaps_chosen,
        })
        print(
            f"[conv]   '{name}': K={K} converged={meta['converged']} "
            f"final_rel={meta['final_rel']:.2e}  snap iters={chosen}"
        )

    # Shared log color scale across every residual panel (so the spatial extent of the error
    # is directly comparable). Pool all snapshot magnitudes, exclude exact zeros.
    all_pos = []
    for run in method_runs:
        for r in run["snaps"].values():
            v = np.abs(gridify(r, info)).ravel()
            v = v[v > 0]
            if v.size:
                all_pos.append(v)
    if not all_pos:
        raise RuntimeError("no residual snapshot values to plot")
    cat = np.concatenate(all_pos)
    r_vmax = float(np.percentile(cat, 99.5))
    r_vmin = max(float(np.percentile(cat, 0.5)), r_vmax * 1e-8)

    # Layout: 2-column-wide figure. Top strip = (rho, b) only. Below = n_methods × n_panels.
    # Heights: 1 unit for top strip, 1 unit each for method rows.
    _t_top = 13.5
    _t_res = 12.0
    _t_meth = 14.0
    _t_cbar = 11.5
    _t_cbar_lab = 13.0

    cell_in = 2.4  # roughly per-panel size in inches; figure is much bigger than before
    fig_w = cell_in * n_panels_per_method + 1.2  # + colorbar gutter
    fig_h = cell_in * (1 + n_methods) + 0.9
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)

    outer = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=(1.0, 0.025),
        wspace=0.012,
    )
    # Single shared grid: guarantees top-row columns align exactly with residual columns.
    all_grid = GridSpecFromSubplotSpec(
        1 + n_methods,
        n_panels_per_method,
        outer[0, 0],
        wspace=0.04,
        hspace=0.10,
        height_ratios=(1.0,) + tuple(1.0 for _ in range(n_methods)),
    )

    # --- Top strip: rho, b, pressure, velocity, learned M ---
    ax_rho = fig.add_subplot(all_grid[0, 0])
    ax_b = fig.add_subplot(all_grid[0, 1])
    ax_p = fig.add_subplot(all_grid[0, 2])
    ax_v = fig.add_subplot(all_grid[0, 3])
    ax_m = fig.add_subplot(all_grid[0, 4])

    imshow_density(
        ax_rho,
        info,
        title=rf"Density $\rho$  ($N\!=\!{info['N']}$)",
    )
    ax_rho.set_box_aspect(1)

    b_grid = gridify(b, info)
    b_abs = float(np.percentile(np.abs(b_grid), 99.0))
    b_abs = max(b_abs, 1e-30)
    ax_b.imshow(
        b_grid,
        origin="lower",
        cmap="RdBu_r",
        vmin=-b_abs,
        vmax=b_abs,
        interpolation="nearest",
        aspect="equal",
    )
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    ax_b.set_title(r"Input $b$  (RHS)")
    ax_b.set_box_aspect(1)

    b_top, P_top, vx_top, vy_top, mag_top = _pressure_velocity_fields(info)
    _ = b_top  # same RHS; keep for explicit pairing with pressure/velocity helper.
    active = _active_mask(info)

    p_abs = max(float(np.percentile(np.abs(P_top[active]), 99.0)) if active.any() else 1e-6, 1e-6)
    cmap_p = matplotlib.colormaps["RdBu_r"].copy()
    cmap_p.set_bad(color="lightgray")
    ax_p.imshow(
        np.where(active, P_top, np.nan),
        origin="lower",
        cmap=cmap_p,
        vmin=-p_abs,
        vmax=p_abs,
        interpolation="nearest",
        aspect="equal",
    )
    ax_p.set_xticks([])
    ax_p.set_yticks([])
    ax_p.set_title(r"Pressure $p$")
    ax_p.set_box_aspect(1)

    mag_masked = np.where(active, mag_top, np.nan)
    if active.any():
        v_hi = max(float(np.percentile(mag_top[active], 99.5)), 1e-12)
        v_pos = mag_top[active & (mag_top > 0)]
        v_lo = float(np.percentile(v_pos, 5.0)) if v_pos.size else v_hi * 1e-3
        v_lo = max(min(v_lo, v_hi * 0.5), v_hi * 1e-3)
    else:
        v_hi = max(float(np.nanmax(mag_top)), 1e-12)
        v_lo = v_hi * 1e-3
    cmap_v = matplotlib.colormaps["viridis"].copy()
    cmap_v.set_bad(color="lightgray")
    ax_v.imshow(
        mag_masked,
        origin="lower",
        cmap=cmap_v,
        norm=LogNorm(vmin=v_lo, vmax=v_hi),
        interpolation="nearest",
        aspect="equal",
    )
    ax_v.set_xticks([])
    ax_v.set_yticks([])
    ax_v.set_title(r"Velocity $|v|$")
    # White stream arrows to mirror the hero-panel velocity rendering.
    H, W = vx_top.shape
    Yg, Xg = np.mgrid[0:H, 0:W]
    vx_safe = np.where(active, vx_top, 0.0)
    vy_safe = np.where(active, vy_top, 0.0)
    try:
        ax_v.streamplot(
            Xg,
            Yg,
            vx_safe,
            vy_safe,
            color="white",
            linewidth=0.7,
            density=1.4,
            arrowsize=0.7,
        )
    except Exception as e:
        print(f"  [conv] velocity streamplot failed ({e}); skipping arrows", file=sys.stderr)
    ax_v.set_xlim(-0.5, W - 0.5)
    ax_v.set_ylim(-0.5, H - 0.5)
    ax_v.set_box_aspect(1)

    M_dense = ours.assembled_dense_M_numpy()
    mc = max(16, min(256, int(M_dense.shape[0])))
    _imshow_matrix_log10_abs(
        ax_m,
        M_dense,
        rf"Learned $M$" + "\n" + rf"top-left ${mc}\times{mc}$",
        crop=mc,
        cmap="magma",
        rank_normalize=True,
    )
    ax_m.set_box_aspect(1)

    for ax in (ax_rho, ax_b, ax_p, ax_v, ax_m):
        ttl = ax.title
        if ttl.get_text():
            ttl.set_fontsize(_t_top)
            ttl.set_linespacing(0.92)

    # --- Method grid: n_methods rows × n_panels_per_method cols ---
    cax = fig.add_subplot(outer[0, 1])

    im_last = None
    for mi, run in enumerate(method_runs):
        keys = run["iters_chosen"]
        for j in range(n_panels_per_method):
            ax = fig.add_subplot(all_grid[mi + 1, j])
            if j < len(keys):
                k = keys[j]
                r_grid = np.abs(gridify(run["snaps"][k], info)) + 1e-30
                im_last = ax.imshow(
                    r_grid,
                    origin="lower",
                    cmap=RESIDUAL_CMAP,
                    norm=LogNorm(vmin=r_vmin, vmax=r_vmax),
                    interpolation="nearest",
                    aspect="equal",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(fr"$k={k}$", fontsize=_t_res)
            else:
                ax.axis("off")
            ax.set_box_aspect(1)
            # Row label: method name + total iteration count, on the leftmost panel.
            if j == 0:
                K = run["K"]
                conv = "" if run["converged"] else r" (no conv.)"
                ax.set_ylabel(
                    f"{run['name']}\n$K\\!=\\!{K}${conv}",
                    fontsize=_t_meth,
                    rotation=90,
                    labelpad=8,
                )

    if im_last is not None:
        cb = fig.colorbar(im_last, cax=cax)
        cb.ax.tick_params(labelsize=_t_cbar)
        cb.set_label(r"$|r_k|$", fontsize=_t_cbar_lab)

    _savefig_paper(fig, out_path)
    plt.close(fig)
    print(f"[conv] -> {out_path}")


# ============================================================================
# CLI
# ============================================================================
def _default_data_root():
    return SCRIPTS_DIR / "data"


def _resolve_frame(data_root, scale, split, frame_idx):
    d = Path(data_root) / f"multiphase_v2_{scale}" / split
    frames = sorted(f for f in d.iterdir() if (f / "nodes.bin").exists())
    if not frames:
        raise SystemExit(f"No frames in {d}")
    return frames[min(frame_idx, len(frames) - 1)]


def _resolve_frame_3d(data_root, bundle, split, frame_idx):
    d = Path(data_root) / str(bundle) / split
    frames = sorted(f for f in d.iterdir() if (f / "nodes.bin").exists())
    if not frames:
        raise SystemExit(f"No frames in {d}")
    return frames[min(int(frame_idx), len(frames) - 1)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fig",
                   choices=["hero", "variety", "variety-3d", "m-zoom-strip", "residual-3d", "residual-3d-propagation",
                            "residual-3d-compare",
                            "matrices", "convergence", "all"],
                   default="all")
    p.add_argument("--data-root", default=str(_default_data_root()),
                   help="Root containing multiphase_v2_<N>/ subfolders.")
    p.add_argument("--out-dir", default=str(SCRIPTS_DIR.parent.parent / "Paper" / "figures"))

    # Hero (defaults match the matrices/initial-problem frame so both figures show the same system).
    p.add_argument("--hero-scale", type=int, default=16384)
    p.add_argument("--hero-split", choices=["train", "test"], default="train")
    p.add_argument("--hero-frame", type=int, default=73)

    # Variety
    p.add_argument("--variety-scales", type=int, nargs="+",
                   default=[2048, 4096, 8192])
    p.add_argument("--variety-cells", type=int, default=24)
    p.add_argument("--variety-no-condnum", action="store_true")
    p.add_argument("--variety-seed", type=int, default=0)

    # Variety 3D (renders every frame in a 3D bundle; train + test)
    p.add_argument("--variety-3d-bundle", default="multiphase_v2_3d_8000",
                   help="Subfolder name under --data-root containing train/ and test/ for the 3D bundle.")
    p.add_argument("--variety-3d-rows", type=int, default=12)
    p.add_argument("--variety-3d-cols", type=int, default=10)
    p.add_argument("--variety-3d-dpi", type=int, default=130)

    # Matrices (A, A^{-1}, M three-view; pinned to the hero frame so the matrix structure
    # corresponds to the initial problem the reader has already seen).
    p.add_argument("--matrices-scale", type=int, default=16384)
    p.add_argument("--matrices-split", choices=["train", "test"], default="train")
    p.add_argument("--matrices-frame", type=int, default=73)
    p.add_argument("--matrices-frame-dir", default=None,
                   help="Explicit frame directory; overrides --matrices-scale/--matrices-split/--matrices-frame.")
    p.add_argument("--matrices-crop", type=int, default=256,
                   help="Top-left crop size for each matrix panel (default 256).")
    p.add_argument("--matrices-matrix-max-n", type=int, default=65536,
                   help="Refuse to assemble dense matrices when padded n exceeds this.")

    # Convergence — deliberately picks a *different* frame than the matrices/hero so the
    # multi-method residual story is on a fresh problem (one where multiscale information
    # propagation is visually pronounced). Override via --conv-frame / --conv-frame-dir.
    p.add_argument("--conv-scale", type=int, default=16384,
                   help="multiphase_v2_<N> frame scale for the convergence figure (default 16384).")
    p.add_argument("--conv-split", choices=["train", "test"], default="train")
    p.add_argument("--conv-frame", type=int, default=42,
                   help="Frame index within the split (default 42; deliberately differs from "
                        "the hero/matrices frame). Set to -1 to fall back to pick_stiff_frame(--conv-seed).")
    p.add_argument("--conv-seed", type=int, default=0,
                   help="Only used when --conv-frame=-1 (pick_stiff_frame).")
    p.add_argument("--conv-frame-dir", default=None,
                   help="Explicit frame directory; overrides --conv-scale/--conv-split/--conv-frame.")
    p.add_argument("--conv-rtol", type=float, default=1e-8,
                   help="PCG stopping tolerance on ||r||_2 / ||b||_2.")
    p.add_argument("--conv-max-iter", type=int, default=5000,
                   help="Per-method max PCG iterations (unpreconditioned/Jacobi can be slow on stiff frames).")
    p.add_argument("--neuralspai-checkpoint", default=None,
                   help="Optional explicit NeuralSPAI checkpoint for convergence figure.")
    p.add_argument("--lsp-repo", default="/orcd/home/002/osbo/ondemand/LearningSparsePreconditioner4GPU",
                   help="Path to LearningSparsePreconditioner4GPU repo (for NeuralSPAI loading).")
    p.add_argument("--lsp-summary-csv", default=str(SCRIPTS_DIR / "results" / "lsp_scale_infer_summary.csv"),
                   help="CSV used to auto-resolve NeuralSPAI checkpoint by scale when --neuralspai-checkpoint is not set.")
    p.add_argument("--ours-weights", default=None,
                   help="Required for matrices/convergence/m-zoom-strip/residual-3d: trained .bytes checkpoint.")

    # M zoom strip (learned M at multiple crop sizes)
    p.add_argument("--mzoom-frame-dir", default=None,
                   help="Frame directory for m-zoom-strip figure.")
    p.add_argument("--mzoom-crops", type=int, nargs="+", default=[128, 256, 512, 1024],
                   help="Top-left crop sizes to show in the M zoom strip (default 128 256 512 1024).")
    p.add_argument("--mzoom-3d", action="store_true",
                   help="Use 3D frame loader for --mzoom-frame-dir.")
    p.add_argument("--mzoom-no-titles", action="store_true",
                   help="For m-zoom-strip: hide panel titles.")
    p.add_argument("--mzoom-no-axis", action="store_true",
                   help="For m-zoom-strip: hide axes and spines (image only).")
    p.add_argument("--mzoom-no-colorbar", action="store_true",
                   help="For m-zoom-strip: hide shared colorbar.")
    p.add_argument("--mzoom-tight-image", action="store_true",
                   help="For m-zoom-strip: export with zero-ish padding around image content.")

    # 3D residual propagation
    p.add_argument("--residual-3d-frame-dir", default=None,
                   help="Frame directory for make_residual_3d.")
    p.add_argument("--residual-3d-k", type=int, default=1,
                   help="PCG iteration index at which to visualise |r_k| (default 1).")
    p.add_argument("--residual-3d-cmap", default="inferno",
                   help="Matplotlib colormap for residual magnitude (default inferno).")
    p.add_argument("--residual-3d-elev", type=float, default=22.0)
    p.add_argument("--residual-3d-azim", type=float, default=-55.0)
    p.add_argument("--residual-3d-alpha-power", type=float, default=2.4,
                   help="Opacity power after residual normalization; larger suppresses mid-opacity clutter.")
    p.add_argument("--residual-3d-alpha-scurve", action=argparse.BooleanOptionalAction, default=True,
                   help="Apply smoothstep S-curve before opacity power (default true).")
    p.add_argument("--residual-3d-render", choices=["scatter", "voxels", "traces"], default="scatter",
                   help="Render mode for residuals: scatter points, voxels, or trace lines.")
    p.add_argument("--residual-3d-field", choices=["residual", "correction", "dual"],
                   default="residual",
                   help="Scalar field to visualise: residual=|r_k| (default), "
                        "correction=|M⁻¹b| (non-local precond response), "
                        "dual=overlay both with blue/inferno colormaps.")
    p.add_argument("--residual-3d-clean", action="store_true",
                   help="Hide axis, labels, title, and wireframe box for residual-3d export.")
    p.add_argument("--residual-3d-no-colorbar", action="store_true",
                   help="Hide colorbar in residual-3d output.")
    p.add_argument("--residual-3d-trace-seeds", type=int, default=90,
                   help="Number of high-residual seeds for trace mode.")
    p.add_argument("--residual-3d-trace-steps", type=int, default=48,
                   help="Integration steps per direction for each trace seed.")
    p.add_argument("--residual-3d-trace-step-size", type=float, default=0.65,
                   help="Step size in voxel units for trace integration.")
    p.add_argument("--residual-3d-trace-sigma", type=float, default=1.2,
                   help="Gaussian smoothing sigma for residual-derived trace field.")
    p.add_argument("--residual-3d-zoom", type=float, default=1.0,
                   help="Camera zoom factor for residual-3d (>=1 zooms in by reducing camera distance).")
    p.add_argument("--residual-3d-tight-image", action="store_true",
                   help="For residual-3d: export with near-zero outer padding (content fills figure bounds).")
    p.add_argument("--residual-prop-frame-dir", default=None,
                   help="Frame directory for residual-3d-propagation.")
    p.add_argument("--residual-prop-ks", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="PCG iterations to visualize in residual-3d-propagation.")
    p.add_argument("--residual-prop-method", choices=["ours", "jacobi", "none"], default="ours",
                   help="Preconditioner/method for residual-3d-propagation.")
    p.add_argument("--residual-prop-cmap", default="inferno",
                   help="Colormap for residual-3d-propagation.")
    p.add_argument("--residual-prop-voxel-quantile", type=float, default=98.5,
                   help="Only voxels above this residual percentile are shown in 3D row.")
    p.add_argument("--residual-compare-3d-frame-dir", default=None,
                   help="Explicit frame directory for residual-3d-compare.")
    p.add_argument("--residual-compare-3d-bundle", default="multiphase_v2_3d_8000",
                   help="3D data bundle under --data-root used when --residual-compare-3d-frame-dir is not set.")
    p.add_argument("--residual-compare-3d-split", choices=["train", "test"], default="train")
    p.add_argument("--residual-compare-3d-frame", type=int, default=73,
                   help="Default 73 to mirror the paper's Fig. 1c scene selection.")
    p.add_argument("--residual-compare-3d-rtol", type=float, default=1e-8)
    p.add_argument("--residual-compare-3d-max-iter", type=int, default=6000)
    p.add_argument("--residual-compare-3d-panels", type=int, default=5,
                   help="Snapshots per method row for residual-3d-compare.")
    p.add_argument("--residual-compare-3d-elev", type=float, default=22.0)
    p.add_argument("--residual-compare-3d-azim", type=float, default=-55.0)

    args = p.parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = {args.fig} if args.fig != "all" else {"hero", "variety", "matrices", "convergence"}
    # "variety-3d", "m-zoom-strip", "residual-3d", "residual-3d-propagation",
    # and "residual-3d-compare"
    # are not part of "all" (require explicit flags).

    if "hero" in figs:
        fr = _resolve_frame(args.data_root, args.hero_scale, args.hero_split, args.hero_frame)
        make_hero(fr, out_dir / "multiphase_hero.png")

    if "variety" in figs:
        make_variety(
            args.data_root,
            out_dir / "multiphase_variety.png",
            scales=tuple(args.variety_scales),
            n_cells=args.variety_cells,
            condnum=not args.variety_no_condnum,
            seed=args.variety_seed,
        )

    if "variety-3d" in figs:
        make_variety_3d(
            args.data_root,
            out_dir / f"{args.variety_3d_bundle}_variety.png",
            bundle_name=args.variety_3d_bundle,
            rows=args.variety_3d_rows,
            cols=args.variety_3d_cols,
            dpi=args.variety_3d_dpi,
        )

    if "m-zoom-strip" in figs:
        if not args.mzoom_frame_dir:
            raise SystemExit("--fig m-zoom-strip requires --mzoom-frame-dir")
        make_m_zoom_strip(
            args.mzoom_frame_dir,
            out_dir / "multiphase_m_zoom_strip.png",
            args.ours_weights,
            crops=tuple(args.mzoom_crops),
            loader_3d=bool(args.mzoom_3d),
            show_titles=not bool(args.mzoom_no_titles),
            show_axes=not bool(args.mzoom_no_axis),
            show_colorbar=not bool(args.mzoom_no_colorbar),
            tight_image=bool(args.mzoom_tight_image),
        )

    if "residual-3d" in figs:
        if not args.residual_3d_frame_dir:
            raise SystemExit("--fig residual-3d requires --residual-3d-frame-dir")
        k = int(args.residual_3d_k)
        field_arg = str(args.residual_3d_field)
        # "dual" is a field mode, not a render mode — render_mode is ignored for dual.
        out_stem = f"residual_3d_{field_arg}_k{k:03d}" if field_arg != "residual" else f"residual_3d_k{k:03d}"
        make_residual_3d(
            args.residual_3d_frame_dir,
            out_dir / f"{out_stem}.png",
            args.ours_weights,
            k=k,
            cmap_name=args.residual_3d_cmap,
            elev=float(args.residual_3d_elev),
            azim=float(args.residual_3d_azim),
            render_mode=str(args.residual_3d_render),
            field=field_arg,
            clean_view=bool(args.residual_3d_clean),
            show_colorbar=not bool(args.residual_3d_no_colorbar),
            alpha_power=float(args.residual_3d_alpha_power),
            alpha_scurve=bool(args.residual_3d_alpha_scurve),
            trace_seeds=int(args.residual_3d_trace_seeds),
            trace_steps=int(args.residual_3d_trace_steps),
            trace_step_size=float(args.residual_3d_trace_step_size),
            trace_sigma=float(args.residual_3d_trace_sigma),
            zoom=float(args.residual_3d_zoom),
            tight_image=bool(args.residual_3d_tight_image),
        )

    if "residual-3d-propagation" in figs:
        if not args.residual_prop_frame_dir:
            raise SystemExit("--fig residual-3d-propagation requires --residual-prop-frame-dir")
        make_residual_3d_propagation(
            args.residual_prop_frame_dir,
            out_dir / "residual_3d_propagation.png",
            args.ours_weights,
            ks=tuple(int(k) for k in args.residual_prop_ks),
            method=str(args.residual_prop_method),
            cmap_name=str(args.residual_prop_cmap),
            voxel_quantile=float(args.residual_prop_voxel_quantile),
        )

    if "residual-3d-compare" in figs:
        frame_3d = args.residual_compare_3d_frame_dir
        if frame_3d is None:
            frame_3d = str(
                _resolve_frame_3d(
                    args.data_root,
                    args.residual_compare_3d_bundle,
                    args.residual_compare_3d_split,
                    args.residual_compare_3d_frame,
                )
            )
        make_residual_3d_compare(
            frame_3d,
            out_dir / "residual_3d_compare.png",
            ours_weights=args.ours_weights,
            neuralspai_checkpoint=args.neuralspai_checkpoint,
            lsp_repo=args.lsp_repo,
            rtol=float(args.residual_compare_3d_rtol),
            max_iter=int(args.residual_compare_3d_max_iter),
            n_panels=int(args.residual_compare_3d_panels),
            elev=float(args.residual_compare_3d_elev),
            azim=float(args.residual_compare_3d_azim),
        )

    if "matrices" in figs:
        mat_frame = args.matrices_frame_dir
        if mat_frame is None:
            mat_frame = str(_resolve_frame(args.data_root, args.matrices_scale, args.matrices_split, args.matrices_frame))
        make_matrices(
            mat_frame,
            out_dir / "multiphase_matrices.png",
            args.ours_weights,
            matrix_max_n=args.matrices_matrix_max_n,
            crop=args.matrices_crop,
        )

    if "convergence" in figs:
        # Resolution priority: explicit --conv-frame-dir > deterministic split/frame index >
        # pick_stiff_frame (--conv-frame=-1).
        conv_override = args.conv_frame_dir
        if conv_override is None and args.conv_frame is not None and args.conv_frame >= 0:
            conv_override = str(_resolve_frame(args.data_root, args.conv_scale, args.conv_split, args.conv_frame))
        make_convergence(
            args.data_root,
            args.conv_scale,
            out_dir / "multiphase_convergence.png",
            args.ours_weights,
            seed=args.conv_seed,
            override_frame=conv_override,
            rtol=args.conv_rtol,
            max_iter=args.conv_max_iter,
            neuralspai_checkpoint=args.neuralspai_checkpoint,
            lsp_repo=args.lsp_repo,
            lsp_summary_csv=args.lsp_summary_csv,
        )


if __name__ == "__main__":
    main()
