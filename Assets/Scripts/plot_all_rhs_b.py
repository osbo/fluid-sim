#!/usr/bin/env python3
"""Render a contact sheet of all buoyancy RHS fields b = -d(rho)/dy."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NODE_DTYPE = np.dtype(
    [
        ("position", "3<f4"),
        ("velocity", "3<f4"),
        ("face_vels", "6<f4"),
        ("mass", "<f4"),
        ("layer", "<u4"),
        ("morton", "<u4"),
        ("active", "<u4"),
    ]
)


def load_rhs_grid(frame_dir: Path) -> np.ndarray:
    nodes = np.fromfile(frame_dir / "nodes.bin", dtype=NODE_DTYPE)
    if nodes.size == 0:
        raise ValueError(f"No nodes in {frame_dir}")

    xs = nodes["position"][:, 0].astype(np.int32)
    ys = nodes["position"][:, 1].astype(np.int32)
    w = int(xs.max()) + 1
    h = int(ys.max()) + 1

    rho = np.zeros((h, w), dtype=np.float64)
    rho[ys, xs] = nodes["mass"].astype(np.float64)

    drhody = np.zeros_like(rho)
    drhody[1:-1, :] = 0.5 * (rho[2:, :] - rho[:-2, :])
    b = -drhody
    b -= b.mean()
    return b


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=scripts_dir / "data")
    parser.add_argument("--scale", type=int, default=16384)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all")
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument(
        "--out",
        type=Path,
        default=scripts_dir.parent.parent / "Paper" / "figures" / "all_rhs_b_16384_train.png",
    )
    args = parser.parse_args()

    frame_root = args.data_root / f"multiphase_v2_{args.scale}" / args.split
    frames = sorted(f for f in frame_root.iterdir() if (f / "nodes.bin").is_file())
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    if not frames:
        raise SystemExit(f"No frames found in {frame_root}")

    b_grids: list[np.ndarray] = []
    abs_vals: list[np.ndarray] = []
    for fr in frames:
        b = load_rhs_grid(fr)
        b_grids.append(b)
        abs_vals.append(np.abs(b).ravel())

    all_abs = np.concatenate(abs_vals)
    vmax = float(np.percentile(all_abs, 99.0))
    vmax = max(vmax, 1e-9)
    vmin = -vmax

    n = len(frames)
    cols = max(1, int(args.cols))
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.8 * rows), constrained_layout=True)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = axes.reshape(rows, 1)

    im = None
    for i, (fr, b) in enumerate(zip(frames, b_grids)):
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        im = ax.imshow(b, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(fr.name.replace("frame_", "f"), fontsize=8)

    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, pad=0.01)
        cbar.set_label("Input RHS b")

    fig.suptitle(f"All input b fields: multiphase_v2_{args.scale}/{args.split} ({n} frames)", fontsize=12)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
