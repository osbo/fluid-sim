"""Generate paper figures from the multiphase_v2 benchmark data.

Subfigures (one per --fig flag, or --fig all):
    hero        — 3-panel strip: density rho, right-hand side b, pressure + velocity quiver.
    variety     — 4x3 grid sampling topology / contrast / orientation axes, with N, ratio, kappa.
    convergence — 3×5 grid: top row = density, b, log₁₀|A|, |A⁻¹|, |M| (top-left 128², no per-panel scales;
                  dense matrices by default up to very large N—use --conv-matrix-max-n to cap if needed);
                  rows 2–3 = ten |r_k| snapshots with one shared log colorbar (PCG is scale-invariant in A, b).

Run on the slurm GPU node so the `fluid` conda env (numpy/scipy/pyamg/matplotlib/torch) is available.
The convergence figure requires --ours-weights (trained .bytes matching the frame scale).

Usage:
    python3 MakeMultiphaseFigures.py --fig all \
        --data-root data \
        --out-dir ../../Paper/figures \
        --ours-weights /path/to/v2_2048_d128_L3_hw.bytes
"""

import argparse
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
class OursPreconditioner:
    """Loads a trained checkpoint and exposes apply_M(r) on the CPU side."""

    def __init__(self, info, weights_path, leaf_size=None, max_mixed_size=None):
        self.info = info
        self.weights_path = Path(weights_path)
        N = info["N"]
        if leaf_size is None:
            leaf_size = _SCALE_LEAF_SIZE.get(N, 128)
        if max_mixed_size is None:
            max_mixed_size = N
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


def _imshow_matrix_log10_abs(
    ax,
    M: np.ndarray,
    title: str,
    *,
    crop: int = 128,
    vmin_pct: float = 1.0,
    vmax_pct: float = 99.5,
    diag_band_exclude_for_scale: Optional[int] = None,
):
    """log₁₀|·| heatmap of the top-left ``crop``×``crop`` block (square aspect).

    Color limits use robust percentiles so a dominant diagonal does not crush the rest.
    If ``diag_band_exclude_for_scale`` is set, entries with ``|i-j| <= band`` are ignored when
    picking vmin/vmax (useful for learned ``M`` so off-diagonal / block structure stays visible).
    """
    M = np.asarray(M, dtype=np.float64)
    c = max(1, min(int(crop), M.shape[0], M.shape[1]))
    logm = np.log10(np.abs(M[:c, :c]) + 1e-9)
    if diag_band_exclude_for_scale is not None and int(diag_band_exclude_for_scale) > 0:
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
        cmap="magma",
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
    _apply_paper_style()
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

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), constrained_layout=True)

    im0 = imshow_density(
        axes[0], info,
        title=rf"Density $\rho$  (range {info['rho_light']:.2g}–{info['rho_heavy']:.2g})",
    )
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    # Middle panel: buoyancy right-hand side b = -∂ρ/∂y (mean-removed; actually solved for).
    b_abs = max(float(np.percentile(np.abs(b), 99)), 1e-30) if b.size else 1.0
    cmap_b = matplotlib.colormaps["RdBu_r"].copy()
    cmap_b.set_bad(color="lightgray")
    im1 = axes[1].imshow(
        b_masked, origin="lower", cmap=cmap_b,
        vmin=-b_abs, vmax=b_abs, interpolation="nearest",
    )
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_title(
        fr"Right-hand side $b = -\partial_y \rho$  "
        fr"($\rho_{{\rm heavy}}/\rho_{{\rm light}}\!=\!{info['ratio']:.0f}\times$)"
    )
    fig.colorbar(im1, ax=axes[1], shrink=0.85)

    P_abs = max(float(np.percentile(np.abs(p), 99)), 1e-30)
    cmap_p = matplotlib.colormaps["RdBu_r"].copy()
    cmap_p.set_bad(color="lightgray")
    im2 = axes[2].imshow(
        P_masked, origin="lower", cmap=cmap_p,
        vmin=-P_abs, vmax=P_abs, interpolation="nearest",
    )
    # Downsample velocity for quiver; skip inactive grid points so we don't draw spurious arrows.
    H, W = P.shape
    step = max(1, min(H, W) // 24)
    ys_q, xs_q = np.mgrid[0:H:step, 0:W:step]
    q_mask = active[ys_q, xs_q]
    axes[2].quiver(
        xs_q[q_mask], ys_q[q_mask],
        vx[ys_q, xs_q][q_mask], vy[ys_q, xs_q][q_mask],
        color="black", scale_units="xy", scale=None, width=0.0025, alpha=0.7,
    )
    axes[2].set_xticks([]); axes[2].set_yticks([])
    axes[2].set_title("Pressure $p$ with velocity $-\\nabla p / \\rho$")
    fig.colorbar(im2, ax=axes[2], shrink=0.85)

    fig.suptitle(f"Multiphase pressure-Poisson frame  ($N\\!=\\!{info['N']}$)")
    _savefig_paper(fig, out_path)
    plt.close(fig)
    print(f"[hero] -> {out_path}")


# ============================================================================
# Figure 2: variety grid (3 rows x 4 cols)
# ============================================================================
def make_variety(data_root, out_path, scales=(2048, 4096, 8192), n_cells=12, condnum=True, seed=0):
    """Sample n_cells frames across topology / contrast axes, both splits, multiple scales."""
    _apply_paper_style()
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

    rows = 3 if n_cells == 12 else max(1, int(math.ceil(n_cells / 4)))
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.2 * rows), constrained_layout=True)
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
        ax.set_title(
            f"$N\\!=\\!{info['N']}$,  $\\rho_H/\\rho_L\\!=\\!{info['ratio']:.0f}\\times$,  {kstr}",
        )

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, location="right")
    cbar.set_label(r"$\rho$", fontsize=matplotlib.rcParams["axes.labelsize"])
    fig.suptitle(
        "Multiphase Poisson systems: every frame independently randomizes topology, contrast, orientation, scale",
    )
    _savefig_paper(fig, out_path)
    plt.close(fig)
    print(f"[variety] -> {out_path}")


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


def make_convergence(
    data_root,
    scale,
    out_path,
    ours_weights,
    *,
    seed=0,
    override_frame=None,
    rtol=1e-8,
    max_iter=10000,
    max_snapshots=36,
    snap_iters_explicit: Optional[list[int]] = None,
    matrix_max_n: int = 65536,
):
    _apply_paper_style()
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

    print(f"[conv] loading preconditioner weights {ours_weights}")
    ours = OursPreconditioner(info, ours_weights)
    viz_n = ours._viz_n
    if info["N"] != viz_n:
        print(
            f"[conv] note: frame N={info['N']} vs padded operator n={viz_n} "
            "(matrix row uses padded graph; PCG uses physical A).",
            file=sys.stderr,
        )

    print(f"[conv] running preconditioned CG (full log below)")
    snaps, meta = pcg_preconditioned_full(
        A,
        b,
        ours.apply_M,
        rtol=rtol,
        max_iter=max_iter,
        snap_iters_explicit=snap_iters_explicit,
        max_snapshots=max_snapshots,
    )
    K_final = int(meta["iters"])
    if snap_iters_explicit is None:
        snaps = _finalize_conv_residual_snaps(snaps, K_final, _CONV_RESIDUAL_GRID_SLOTS)
    else:
        keys_e = sorted(snaps.keys())
        if len(keys_e) > _CONV_RESIDUAL_GRID_SLOTS:
            snaps = {k: snaps[k] for k in _subsample_sorted(keys_e, _CONV_RESIDUAL_GRID_SLOTS)}
    print(
        f"[conv] PCG summary: iters={meta['iters']}  converged={meta['converged']}  "
        f"final rel={meta['final_rel']:.6e}"
    )

    keys = sorted(snaps.keys())
    n_resid = len(keys)
    if n_resid < 1:
        raise RuntimeError("no residual snapshots to plot")

    # Nested layout: 3×5 heatmaps + dedicated colorbar column. Larger canvas, smaller type
    # (top row titles stay two-line; |r_k| panels use a single compact line).
    _t_top = 8.6
    _t_res = 7.85
    _t_sup = 10.6
    _t_cbar = 7.45
    _t_cbar_lab = 9.45
    fig = plt.figure(figsize=(14.25, 7.75), constrained_layout=True)
    outer = GridSpec(1, 2, figure=fig, width_ratios=(1.0, 0.032), wspace=0.011)
    inner = GridSpecFromSubplotSpec(3, 5, outer[0, 0], wspace=0.011, hspace=0.038)
    ax_top = [fig.add_subplot(inner[0, j]) for j in range(5)]
    ax_mid = [fig.add_subplot(inner[1, j]) for j in range(5)]
    ax_bot = [fig.add_subplot(inner[2, j]) for j in range(5)]
    cax = fig.add_subplot(outer[0, 1])
    _eng = fig.get_layout_engine()
    if _eng is not None and hasattr(_eng, "set"):
        try:
            _eng.set(rect=(0.0, 0.0, 1.0, 0.88))
        except (TypeError, ValueError):
            pass

    # --- Top row: frame, b, A, A^-1, M ---
    im_rho = imshow_density(
        ax_top[0],
        info,
        title=rf"Density $\rho$" + "\n" + rf"($N\!=\!{info['N']}$)",
    )
    ax_top[0].set_box_aspect(1)

    b_grid = gridify(b, info)
    b_abs = float(np.percentile(np.abs(b_grid), 99.0))
    b_abs = max(b_abs, 1e-30)
    imb = ax_top[1].imshow(
        b_grid,
        origin="lower",
        cmap="RdBu_r",
        vmin=-b_abs,
        vmax=b_abs,
        interpolation="nearest",
        aspect="equal",
    )
    ax_top[1].set_xticks([])
    ax_top[1].set_yticks([])
    ax_top[1].set_title(r"Input $b$" + "\n" + r"(RHS)", fontsize=_t_top)
    ax_top[1].set_box_aspect(1)

    im_a = im_ai = im_m = None
    if viz_n <= matrix_max_n:
        print(f"[conv] dense matrix panels (n={viz_n} ≤ {matrix_max_n}) …")
        t0 = time.time()
        A_dense = ours.dense_A_numpy()
        mc = min(128, viz_n)
        im_a = _imshow_matrix_log10_abs(
            ax_top[2],
            A_dense,
            rf"$A$ (log$_{{10}}$|·|)" + "\n" + rf"top-left ${mc}\times{mc}$",
        )
        ax_top[2].set_box_aspect(1)
        A_inv = _dense_inv_numpy_or_torch(A_dense, ours.device)
        print(f"[conv]   inv(A) in {time.time() - t0:.1f}s")
        im_ai = _imshow_matrix_log10_abs(
            ax_top[3],
            A_inv,
            rf"$A^{{-1}}$ (log$_{{10}}$|·|)" + "\n" + rf"top-left ${mc}\times{mc}$",
        )
        ax_top[3].set_box_aspect(1)
        t1 = time.time()
        M_dense = ours.assembled_dense_M_numpy()
        print(f"[conv]   assembled M in {time.time() - t1:.1f}s")
        # Exclude a diagonal band from percentile scaling so log₁₀|M| shows block / off-diag pattern.
        band_m = max(8, mc // 8)
        im_m = _imshow_matrix_log10_abs(
            ax_top[4],
            M_dense,
            rf"Learned $M$ (log$_{{10}}$|·|)" + "\n" + rf"top-left ${mc}\times{mc}$",
            vmin_pct=0.5,
            vmax_pct=99.8,
            diag_band_exclude_for_scale=band_m,
        )
        ax_top[4].set_box_aspect(1)
    else:
        msg = (
            f"Dense $A$, $A^{{-1}}$, $M$ heatmaps skipped:\n"
            f"padded $n={viz_n} >$ --conv-matrix-max-n={matrix_max_n}$.\n"
            "Use a smaller --conv-scale for these panels."
        )
        for j in range(2, 5):
            ax_top[j].axis("off")
            ax_top[j].text(
                0.5, 0.5, msg, ha="center", va="center",
                fontsize=_t_res, transform=ax_top[j].transAxes,
            )

    for ax in ax_top:
        ttl = ax.title
        if ttl.get_text():
            ttl.set_fontsize(_t_top)
            ttl.set_linespacing(0.92)

    # --- Rows 2–3: |r_k| on shared log scale ---
    all_pos = []
    for r in snaps.values():
        v = np.abs(gridify(r, info)).ravel()
        v = v[v > 0]
        if v.size:
            all_pos.append(v)
    if not all_pos:
        raise RuntimeError("no residual snapshot values to plot")
    cat = np.concatenate(all_pos)
    r_vmax = float(np.percentile(cat, 99.5))
    r_vmin = max(float(np.percentile(cat, 0.5)), r_vmax * 1e-8)

    resid_axes = ax_mid + ax_bot
    im_last = None
    for idx, k in enumerate(keys):
        if idx >= len(resid_axes):
            break
        ax = resid_axes[idx]
        r_grid = np.abs(gridify(snaps[k], info)) + 1e-30
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
        ax.set_title(fr"$|r_k|$, $k={k}$", fontsize=_t_res)
        ax.set_box_aspect(1)
    for idx in range(n_resid, len(resid_axes)):
        resid_axes[idx].axis("off")

    # No colorbars on the top row: magnitudes are qualitative here, and PCG iterates are
    # invariant to diagonal scaling of (A, b); only the residual row needs a shared scale.

    if im_last is not None:
        cb = fig.colorbar(im_last, cax=cax)
        cb.ax.tick_params(labelsize=_t_cbar)
        cb.set_label(r"$|r_k|$", fontsize=_t_cbar_lab)

    fig.suptitle(
        rf"PCG with learned preconditioner: $N\!=\!{info['N']}$, "
        rf"$\rho_H/\rho_L\!=\!{info['ratio']:.0f}\times$, $\kappa\!\approx\!{kappa:.1e}$",
        fontsize=_t_sup,
        y=0.945,
    )
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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fig", choices=["hero", "variety", "convergence", "all"],
                   default="all")
    p.add_argument("--data-root", default=str(_default_data_root()),
                   help="Root containing multiphase_v2_<N>/ subfolders.")
    p.add_argument("--out-dir", default=str(SCRIPTS_DIR.parent.parent / "Paper" / "figures"))

    # Hero (defaults match the convergence frame so both figures show the same system).
    p.add_argument("--hero-scale", type=int, default=8192)
    p.add_argument("--hero-split", choices=["train", "test"], default="train")
    p.add_argument("--hero-frame", type=int, default=73)

    # Variety
    p.add_argument("--variety-scales", type=int, nargs="+",
                   default=[2048, 4096, 8192])
    p.add_argument("--variety-cells", type=int, default=12)
    p.add_argument("--variety-no-condnum", action="store_true")
    p.add_argument("--variety-seed", type=int, default=0)

    # Convergence — defaults pin to the same frame as the hero figure.
    p.add_argument("--conv-scale", type=int, default=8192,
                   help="multiphase_v2_<N> frame scale for the convergence figure (default 8192).")
    p.add_argument("--conv-split", choices=["train", "test"], default="train",
                   help="Dataset split for the convergence frame (default: train; mirrors --hero-split).")
    p.add_argument("--conv-frame", type=int, default=73,
                   help="Frame index within the split (default 73; mirrors --hero-frame). "
                        "Set to -1 to fall back to pick_stiff_frame(--conv-seed).")
    p.add_argument("--conv-seed", type=int, default=0,
                   help="Only used when --conv-frame=-1 (pick_stiff_frame).")
    p.add_argument("--conv-frame-dir", default=None,
                   help="Explicit frame directory; overrides --conv-scale/--conv-split/--conv-frame.")
    p.add_argument("--conv-rtol", type=float, default=1e-8,
                   help="PCG stopping tolerance on ||r||_2 / ||b||_2.")
    p.add_argument("--conv-max-iter", type=int, default=10000)
    p.add_argument("--conv-max-snap-panels", type=int, default=36,
                   help="Max distinct PCG iterations to retain as residual snapshots during the solve (then 10 are chosen for the 2×5 grid).")
    p.add_argument("--conv-snap-iters", type=int, nargs="*", default=None,
                   help="Optional explicit PCG iterations to plot; default uses an automatic snapshot grid.")
    p.add_argument("--conv-matrix-max-n", type=int, default=65536,
                   help="Skip dense A / A⁻¹ / M heatmaps when padded n exceeds this (default includes N=8192; lower to save time/RAM).")
    p.add_argument("--ours-weights", default=None,
                   help="Required for convergence: trained preconditioner .bytes matching the frame scale.")

    args = p.parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = {args.fig} if args.fig != "all" else {"hero", "variety", "convergence"}

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

    if "convergence" in figs:
        snap_explicit = args.conv_snap_iters
        if snap_explicit is not None and len(snap_explicit) == 0:
            snap_explicit = None
        # Resolution priority: explicit --conv-frame-dir > deterministic split/frame index >
        # pick_stiff_frame (--conv-frame=-1). The default (frame 51 at train/8192) mirrors hero so
        # both figures show the same Poisson system.
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
            max_snapshots=args.conv_max_snap_panels,
            snap_iters_explicit=snap_explicit,
            matrix_max_n=args.conv_matrix_max_n,
        )


if __name__ == "__main__":
    main()
