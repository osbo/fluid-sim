#!/usr/bin/env python3
"""
Visualize cloth dataset frames and characterize matrix hardness.

For each selected frame shows:
  Col 0 – Cloth shape (x/y/z scatter, coloured by z-displacement and E field)
  Col 1 – Stiffness field log(1+E): seam-line barriers visible as bright strips
  Col 2 – Jacobi-PCG hardness: estimated condition number and iteration count
           via power-iteration on the Jacobi-preconditioned system D^{-1}H

Prints a table of N / condition-number / estimated Jacobi-iters per frame.

Usage:
  python python/visualize_cloth.py --data-dir data/cloth_v1_4096/train
  python python/visualize_cloth.py --data-dir data/cloth_v1_4096/train \\
      --n-frames 6 --out figures/cloth_frames.png

  # Compare multiple difficulty levels side-by-side:
  python python/visualize_cloth.py \\
      --data-dir data/cloth_v1_4096/train data/cloth_v1_4096_hard/train \\
      --n-frames 3 --out figures/cloth_difficulty.png

Requires: numpy, matplotlib.  scipy optional (enables exact sparse stats).
"""

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from leafonly.data import NODE_DTYPE


# ── data loading ──────────────────────────────────────────────────────────────

def _read_num_nodes(frame_dir: Path) -> int:
    meta = frame_dir / "meta.txt"
    if meta.exists():
        with open(meta) as f:
            for line in f:
                if "numNodes" in line or "num_nodes" in line:
                    return int(line.split(":")[1].strip())
    nodes_bin = frame_dir / "nodes.bin"
    if nodes_bin.exists():
        return nodes_bin.stat().st_size // 64
    return 0


def load_frame(frame_dir: Path):
    """Returns (nodes, rows, cols, vals, N)."""
    N = _read_num_nodes(frame_dir)
    nodes = np.fromfile(frame_dir / "nodes.bin", dtype=NODE_DTYPE)[:N]
    rows  = np.fromfile(frame_dir / "edge_index_rows.bin", dtype=np.uint32)
    cols  = np.fromfile(frame_dir / "edge_index_cols.bin", dtype=np.uint32)
    vals  = np.fromfile(frame_dir / "A_values.bin",        dtype=np.float32)
    return nodes, rows, cols, vals, N


def find_frames(data_dirs: List[Path]) -> List[Path]:
    frames = []
    for d in data_dirs:
        for nb in sorted(d.rglob("nodes.bin")):
            fd = nb.parent
            if (fd / "A_values.bin").exists():
                frames.append(fd)
    return frames


# ── matrix hardness estimation ────────────────────────────────────────────────

def _jacobi_matvec(rows, cols, vals, diag, x):
    """Apply D^{-1}·H·x via COO scatter."""
    y = np.zeros(len(x), dtype=np.float64)
    np.add.at(y, rows.astype(np.intp), vals.astype(np.float64) * x[cols.astype(np.intp)])
    return y / diag


def estimate_hardness(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    N: int,
    power_iters: int = 40,
    tol_rtol: float = 1e-8,
) -> Tuple[float, float, int]:
    """
    Estimate (λ_max, condition_number, jacobi_pcg_iters) of D^{-1}·H via
    power iteration for λ_max and shifted inverse iteration for λ_min.

    Returns (lam_max, kappa, n_iters_estimate).
    """
    rows_i = rows.astype(np.intp); cols_i = cols.astype(np.intp)
    vals_f = vals.astype(np.float64)
    diag_mask = rows_i == cols_i

    diag = np.zeros(N, dtype=np.float64)
    np.add.at(diag, rows_i[diag_mask], vals_f[diag_mask])
    diag = np.maximum(np.abs(diag), 1e-14)

    rng = np.random.default_rng(0)

    # Power iteration for λ_max(D^{-1}H)
    v = rng.standard_normal(N)
    v /= np.linalg.norm(v)
    lam_max = 1.0
    for _ in range(power_iters):
        w = _jacobi_matvec(rows, cols, vals, diag, v)
        lam_max = float(np.dot(v, w))
        v = w / max(np.linalg.norm(w), 1e-14)

    # Shifted power iteration for λ_min(D^{-1}H):
    # B = λ_max·I - D^{-1}H has eigenvalues λ_max - λ_i ≥ 0; its dominant
    # eigenvalue is λ_max - λ_min, so λ_min = λ_max - power_iterate(B).
    u = rng.standard_normal(N)
    u /= np.linalg.norm(u)
    shift_dom = 0.0
    for _ in range(power_iters):
        w = lam_max * u - _jacobi_matvec(rows, cols, vals, diag, u)
        shift_dom = float(np.dot(u, w))
        u = w / max(np.linalg.norm(w), 1e-14)
    lam_min = max(lam_max - shift_dom, lam_max * 1e-12)

    kappa = lam_max / max(lam_min, 1e-14)
    # CG iteration count to reach rtol: ≈ sqrt(κ)/2 · log(2/rtol)
    n_iters = max(1, int(math.sqrt(kappa) * 0.5 * math.log(2.0 / tol_rtol)))
    return float(lam_max), float(kappa), n_iters


# ── per-frame statistics ──────────────────────────────────────────────────────

def frame_stats(rows, cols, vals, N):
    """Return dict of matrix statistics for the stats table."""
    nnz = len(vals)
    diag_vals = vals[rows == cols]
    od_vals   = vals[rows != cols]
    # Stiffness contrast from off-diagonal values
    od_neg = od_vals[od_vals < 0]
    od_pos = od_vals[od_vals > 0]  # from L² bending term
    return {
        "nnz": nnz,
        "nnz_per_node": nnz / max(N, 1),
        "n_pos_od": len(od_pos),   # count of positive off-diagonal (L² edges)
        "n_neg_od": len(od_neg),
        "diag_min": float(np.min(diag_vals)) if len(diag_vals) else 0.0,
        "diag_max": float(np.max(diag_vals)) if len(diag_vals) else 0.0,
    }


# ── plotting helpers ──────────────────────────────────────────────────────────

_CMAP_Z = "RdBu_r"
_CMAP_E = "inferno"


def _scatter(ax, x, y, c, cmap, title, clabel, vmin=None, vmax=None, s=2):
    sc = ax.scatter(x, y, c=c, cmap=cmap, s=s, vmin=vmin, vmax=vmax, rasterized=True)
    plt.colorbar(sc, ax=ax, label=clabel, fraction=0.04, pad=0.02)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def _plot_frame_row(fig, gs, row_idx, frame_dir, nodes, rows, cols, vals, N,
                    lam_max, kappa, n_iters, stats):
    pos = nodes["position"]  # (N, 3) subarray from structured dtype
    px = pos[:, 0]; py = pos[:, 1]; pz = pos[:, 2]
    E  = nodes["mass"].astype(np.float32)

    # Col 0: cloth shape (z-displacement coloured)
    ax0 = fig.add_subplot(gs[row_idx, 0])
    label = (f"{frame_dir.name}  N={N}")
    _scatter(ax0, px, py, pz, _CMAP_Z, label, "z-disp",
             vmin=float(pz.min()), vmax=float(pz.max()))

    # Col 1: stiffness field (barriers visible)
    ax1 = fig.add_subplot(gs[row_idx, 1])
    log_E = np.log1p(E)
    _scatter(ax1, px, py, log_E, _CMAP_E, "log(1+E)  [stiffness / seams]", "log(1+E)")

    # Col 2: hardness summary panel
    ax2 = fig.add_subplot(gs[row_idx, 2])
    ax2.axis("off")
    lines = [
        f"N        = {N}",
        f"nnz      = {stats['nnz']}  ({stats['nnz_per_node']:.1f}/node)",
        f"L² edges = {stats['n_pos_od']}  (off-diag +)",
        f"diag ∈ [{stats['diag_min']:.2e}, {stats['diag_max']:.2e}]",
        "",
        f"λ_max(D⁻¹H) ≈ {lam_max:.3f}",
        f"κ(D⁻¹H)    ≈ {kappa:.2e}",
        f"Est. Jacobi iters ≈ {n_iters}",
    ]
    ax2.text(0.05, 0.95, "\n".join(lines),
             transform=ax2.transAxes, fontsize=8,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))


# ── main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise cloth implicit-integration dataset frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", nargs="+", required=True,
                   help="One or more split directories (train/ or test/)")
    p.add_argument("--n-frames", type=int, default=4,
                   help="Frames to visualise (evenly spaced across dataset)")
    p.add_argument("--out", type=str, default=None,
                   help="Save figure to file (PNG/PDF).  Default: interactive show")
    p.add_argument("--power-iters", type=int, default=40,
                   help="Power-iteration steps for λ_max estimate")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data_dirs = [Path(d) for d in args.data_dir]

    all_frames = find_frames(data_dirs)
    if not all_frames:
        raise SystemExit(f"No valid frames found in: {[str(d) for d in data_dirs]}")

    n = min(args.n_frames, len(all_frames))
    step = max(1, len(all_frames) // n)
    selected = all_frames[::step][:n]

    fig = plt.figure(figsize=(14, n * 3.6))
    gs  = gridspec.GridSpec(n, 3, figure=fig, hspace=0.45, wspace=0.30,
                            left=0.04, right=0.97, top=0.93, bottom=0.03)

    print(f"\n{'Frame':<20} {'N':>6} {'nnz/node':>9} {'L²-edges':>9} "
          f"{'κ(D⁻¹H)':>12} {'Jacobi iters':>14}")
    print("-" * 74)

    for row_idx, frame_dir in enumerate(selected):
        nodes, rows, cols, vals, N = load_frame(frame_dir)
        lam_max, kappa, n_iters = estimate_hardness(
            rows, cols, vals, N, power_iters=args.power_iters
        )
        stats = frame_stats(rows, cols, vals, N)
        _plot_frame_row(fig, gs, row_idx, frame_dir, nodes,
                        rows, cols, vals, N,
                        lam_max, kappa, n_iters, stats)
        print(f"{frame_dir.name:<20} {N:>6} {stats['nnz_per_node']:>9.1f} "
              f"{stats['n_pos_od']:>9} {kappa:>12.2e} {n_iters:>14}")

    title_dirs = ", ".join(str(d) for d in data_dirs)
    fig.suptitle(f"Cloth implicit dataset  |  {title_dirs}", fontsize=10, fontweight="bold")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"\nSaved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
