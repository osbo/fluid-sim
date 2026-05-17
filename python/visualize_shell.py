#!/usr/bin/env python3
"""
Visualize 3D thin-shell dataset frames and characterize matrix hardness.

For each selected frame shows:
  Col 0 – 3D surface scatter coloured by z-position (reveals shape geometry)
  Col 1 – 3D surface scatter coloured by log(1+E) (material panels / seams)
  Col 2 – Matrix hardness: condition number and Jacobi-PCG iteration estimate

Prints a table of N / nnz / condition number / estimated iterations per frame.

Usage:
  python python/visualize_shell.py --data-dir data/shell_v1_1024/test
  python python/visualize_shell.py --data-dir data/shell_v1_1024/test \\
      --n-frames 8 --elev 30 --azim 60 --out results/shell_v1_1024_frames.png

Requires: numpy, matplotlib.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
        return nodes_bin.stat().st_size // NODE_DTYPE.itemsize
    return 0


def load_frame(frame_dir: Path):
    """Returns (nodes, rows, cols, vals, N)."""
    N     = _read_num_nodes(frame_dir)
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


def _infer_shape(pos: np.ndarray) -> str:
    """Rough shape label from bounding-box geometry."""
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    span   = hi - lo
    r_xy   = math.sqrt(pos[:, 0].var() + pos[:, 1].var())
    r_min  = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2).min()
    if r_min > 0.3 * r_xy:
        return "torus"
    if span[2] > 0.6 * max(span[0], span[1]):
        return "cylinder"
    z_mid  = pos[:, 2].mean()
    if abs(z_mid) > 0.1 * span[2]:
        return "spherical_cap"
    return "saddle"


# ── matrix hardness estimation ─────────────────────────────────────────────────

def _matvec(rows, cols, vals, diag, x):
    """Apply D^{-1}·H·x via COO scatter."""
    y = np.zeros(len(x), dtype=np.float64)
    np.add.at(y, rows.astype(np.intp),
               vals.astype(np.float64) * x[cols.astype(np.intp)])
    return y / diag


def estimate_hardness(rows, cols, vals, N, power_iters: int = 40) -> Tuple[float, float, int]:
    """Return (lam_max, kappa, estimated_jacobi_pcg_iters) for D^{-1}·H."""
    rows_i = rows.astype(np.intp)
    cols_i = cols.astype(np.intp)
    diag_mask = rows_i == cols_i
    diag = np.zeros(N, dtype=np.float64)
    np.add.at(diag, rows_i[diag_mask], vals.astype(np.float64)[diag_mask])
    diag = np.maximum(np.abs(diag), 1e-14)

    rng = np.random.default_rng(0)

    v = rng.standard_normal(N); v /= np.linalg.norm(v)
    lam_max = 1.0
    for _ in range(power_iters):
        w = _matvec(rows, cols, vals, diag, v)
        lam_max = float(np.dot(v, w))
        v = w / max(np.linalg.norm(w), 1e-14)

    u = rng.standard_normal(N); u /= np.linalg.norm(u)
    shift_dom = 0.0
    for _ in range(power_iters):
        w = lam_max * u - _matvec(rows, cols, vals, diag, u)
        shift_dom = float(np.dot(u, w))
        u = w / max(np.linalg.norm(w), 1e-14)
    lam_min = max(lam_max - shift_dom, lam_max * 1e-12)

    kappa   = lam_max / max(lam_min, 1e-14)
    n_iters = max(1, int(math.sqrt(kappa) * 0.5 * math.log(2.0 / 1e-8)))
    return float(lam_max), float(kappa), n_iters


# ── plotting ───────────────────────────────────────────────────────────────────

_CMAP_Z = "RdBu_r"
_CMAP_E = "inferno"


def _scatter3d(ax, pos, c, cmap, title, clabel, elev, azim, vmin=None, vmax=None):
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                    c=c, cmap=cmap, s=3, alpha=0.85,
                    vmin=vmin, vmax=vmax, rasterized=True,
                    depthshade=True)
    plt.colorbar(sc, ax=ax, label=clabel, fraction=0.03, pad=0.06, shrink=0.7)
    ax.set_title(title, fontsize=8)
    ax.view_init(elev=elev, azim=azim)
    lo = pos.min(axis=0); hi = pos.max(axis=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("x", fontsize=6, labelpad=1)
    ax.set_ylabel("y", fontsize=6, labelpad=1)
    ax.set_zlabel("z", fontsize=6, labelpad=1)
    ax.tick_params(labelsize=5)


def _plot_frame_row(fig, gs, row_idx, frame_dir, nodes, rows, cols, vals, N,
                    lam_max, kappa, n_iters, elev, azim):
    pos   = nodes["position"].astype(np.float32)  # (N, 3)
    E     = nodes["mass"].astype(np.float32)       # Young's modulus stored in mass field
    log_E = np.log1p(E)
    shape = _infer_shape(pos)
    diag_mask = rows == cols
    od_mask   = ~diag_mask
    nnz       = len(vals)

    ax0 = fig.add_subplot(gs[row_idx, 0], projection="3d")
    _scatter3d(ax0, pos, pos[:, 2], _CMAP_Z,
               f"{frame_dir.name}  [{shape}]", "z",
               elev=elev, azim=azim)

    ax1 = fig.add_subplot(gs[row_idx, 1], projection="3d")
    _scatter3d(ax1, pos, log_E, _CMAP_E,
               "log(1+E)  [material panels]", "log(1+E)",
               elev=elev, azim=azim + 45)

    ax2 = fig.add_subplot(gs[row_idx, 2])
    ax2.axis("off")
    diag_vals = vals[diag_mask]
    od_neg    = vals[od_mask & (vals < 0)]
    od_pos    = vals[od_mask & (vals > 0)]
    lines = [
        f"Shape (inferred): {shape}",
        f"N        = {N}",
        f"nnz      = {nnz}  ({nnz/max(N,1):.1f}/node)",
        f"L² edges = {len(od_pos)}  (off-diag > 0)",
        f"diag ∈ [{float(diag_vals.min()):.2e}, {float(diag_vals.max()):.2e}]",
        f"E    ∈ [{float(E.min()):.2f}, {float(E.max()):.2f}]",
        "",
        f"λ_max(D⁻¹H) ≈ {lam_max:.3f}",
        f"κ(D⁻¹H)    ≈ {kappa:.2e}",
        f"Est. Jacobi iters ≈ {n_iters}",
    ]
    ax2.text(0.05, 0.95, "\n".join(lines),
             transform=ax2.transAxes, fontsize=8,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))


# ── main ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise 3D thin-shell implicit-integration dataset frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", nargs="+", required=True,
                   help="One or more split directories (train/ or test/)")
    p.add_argument("--n-frames", type=int, default=4,
                   help="Frames to visualise (evenly spaced across dataset)")
    p.add_argument("--out", type=str, default=None,
                   help="Save figure to file (PNG/PDF). Default: interactive show")
    p.add_argument("--elev", type=float, default=25.0, help="3D view elevation (degrees)")
    p.add_argument("--azim", type=float, default=45.0, help="3D view azimuth (degrees)")
    p.add_argument("--power-iters", type=int, default=40,
                   help="Power-iteration steps for λ_max/λ_min estimate")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data_dirs = [Path(d) for d in args.data_dir]

    all_frames = find_frames(data_dirs)
    if not all_frames:
        raise SystemExit(f"No valid frames found in: {[str(d) for d in data_dirs]}")

    n    = min(args.n_frames, len(all_frames))
    step = max(1, len(all_frames) // n)
    selected = all_frames[::step][:n]

    fig = plt.figure(figsize=(15, n * 4.2))
    gs  = gridspec.GridSpec(n, 3, figure=fig, hspace=0.50, wspace=0.25,
                            left=0.03, right=0.97, top=0.93, bottom=0.03)

    print(f"\n{'Frame':<20} {'Shape':>12} {'N':>6} {'nnz/node':>9} "
          f"{'L²-edges':>9} {'κ(D⁻¹H)':>12} {'Jacobi iters':>14}")
    print("-" * 86)

    for row_idx, frame_dir in enumerate(selected):
        nodes, rows, cols, vals, N = load_frame(frame_dir)
        lam_max, kappa, n_iters = estimate_hardness(
            rows, cols, vals, N, power_iters=args.power_iters
        )
        pos   = nodes["position"].astype(np.float32)
        shape = _infer_shape(pos)
        nnz   = len(vals)
        od_pos_count = int((vals[rows != cols] > 0).sum())
        _plot_frame_row(fig, gs, row_idx, frame_dir, nodes,
                        rows, cols, vals, N, lam_max, kappa, n_iters,
                        elev=args.elev, azim=args.azim)
        print(f"{frame_dir.name:<20} {shape:>12} {N:>6} {nnz/max(N,1):>9.1f} "
              f"{od_pos_count:>9} {kappa:>12.2e} {n_iters:>14}")

    title_dirs = ", ".join(str(d) for d in data_dirs)
    fig.suptitle(f"3D thin-shell implicit dataset  |  {title_dirs}",
                 fontsize=10, fontweight="bold")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"\nSaved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
