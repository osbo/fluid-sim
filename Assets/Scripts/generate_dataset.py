#!/usr/bin/env python3
"""
Generate Morton-sorted coefficient frames for LeafOnly training / inspection.

Writes per-frame binaries compatible with ``leafonly.data.FluidGraphDataset``:
``nodes.bin``, ``edge_index_rows.bin``, ``edge_index_cols.bin``, ``A_values.bin``, ``meta.txt``.

Choose a ``--dataset-type``; outputs default to ``<output-root>/<type_folder>/{train,test}/``.

- **hard-multiphase**: randomized vertical barrier (wall position, thickness, gap topology).
- **thin-shells**: thin C-shaped / cup shells (AMG aliasing stress; ~2–3 cell walls in normalized space).
- **moving-barrier**: same barrier family as hard-multiphase but wall center slides along an episode
  (``--sequence-length`` frames per slide) to mimic moving geometry without rebuilding AMG every frame.

Grid: width * height >= ``n-target``; nodes sorted by Morton code then truncated to ``n-target``.
5-point Laplacian with harmonic-mean density on faces.

If SciPy is installed, COO is normalized via CSR round-trip; otherwise raw COO triplets are written.
"""
import argparse
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    from scipy import sparse as _sparse
except ModuleNotFoundError:  # pragma: no cover
    _sparse = None


def _rng(seed: int) -> Any:
    if hasattr(np.random, "default_rng"):
        return np.random.default_rng(int(seed))
    return np.random.RandomState(int(seed))


_DEFAULT_N_TARGET = 4096

# Default output folder name per --dataset-type (under --output-root).
TYPE_OUTPUT_SUBDIR = {
    "hard-multiphase": "hard_multiphase",
    "thin-shells": "thin_shells",
    "moving-barrier": "moving_barrier",
}

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


def get_morton_code(x: int, y: int) -> int:
    res = 0
    for i in range(16):
        res |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1))
    return int(res)


def _grid_dims(n_target: int) -> Tuple[int, int]:
    w = int(np.ceil(np.sqrt(n_target)))
    h = int(np.ceil(n_target / w))
    return w, h


def _in_wall_barrier(
    x: int,
    y: int,
    grid_w: int,
    grid_h: int,
    wall_x_center: float,
    wall_thickness: float,
    gap_type: str,
) -> bool:
    x_lo = grid_w * (wall_x_center - wall_thickness / 2.0)
    x_hi = grid_w * (wall_x_center + wall_thickness / 2.0)
    if not (x_lo < x < x_hi):
        return False
    if gap_type == "top" and y > grid_h * 0.2:
        return True
    if gap_type == "bottom" and y < grid_h * 0.8:
        return True
    if gap_type == "middle_hole" and (y < grid_h * 0.4 or y > grid_h * 0.6):
        return True
    return False


def _rho_grid_hard_multiphase(
    grid_w: int,
    grid_h: int,
    frame_rng: Any,
    rho_light: float,
    rho_heavy: float,
    wall_x_center: Optional[float] = None,
    wall_thickness: Optional[float] = None,
    gap_type: Optional[str] = None,
) -> List[Tuple[int, int, int, float]]:
    if wall_x_center is None:
        wall_x_center = float(frame_rng.uniform(0.25, 0.75))
    if wall_thickness is None:
        wall_thickness = float(frame_rng.uniform(0.10, 0.25))
    if gap_type is None:
        gap_type = str(frame_rng.choice(["top", "bottom", "middle_hole"]))

    nodes: List[Tuple[int, int, int, float]] = []
    for y in range(grid_h):
        for x in range(grid_w):
            morton = get_morton_code(x, y)
            in_wall = _in_wall_barrier(x, y, grid_w, grid_h, wall_x_center, wall_thickness, gap_type)
            noise = float(frame_rng.normal(0, 0.05))
            rho = (rho_heavy if in_wall else rho_light) * (1.0 + noise)
            nodes.append((x, y, morton, rho))
    return nodes


def _rho_grid_thin_shells(
    grid_w: int,
    grid_h: int,
    frame_rng: Any,
    rho_light: float,
    rho_heavy: float,
    shell_thickness: float,
) -> List[Tuple[int, int, int, float]]:
    cup_x = float(frame_rng.uniform(0.3, 0.7))
    cup_y = float(frame_rng.uniform(0.3, 0.7))
    cup_size = float(frame_rng.uniform(0.15, 0.3))
    opening_dir = str(frame_rng.choice(["up", "down", "left", "right"]))

    nodes: List[Tuple[int, int, int, float]] = []
    for y in range(grid_h):
        for x in range(grid_w):
            morton = get_morton_code(x, y)
            nx = x / grid_w
            ny = y / grid_h

            in_wall = False
            if abs(nx - cup_x) < cup_size and abs(ny - cup_y) < cup_size:
                inner = max(0.0, cup_size - shell_thickness)
                if not (abs(nx - cup_x) < inner and abs(ny - cup_y) < inner):
                    in_wall = True
                    if opening_dir == "up" and ny < cup_y:
                        in_wall = False
                    elif opening_dir == "down" and ny > cup_y:
                        in_wall = False
                    elif opening_dir == "left" and nx < cup_x:
                        in_wall = False
                    elif opening_dir == "right" and nx > cup_x:
                        in_wall = False

            noise = float(frame_rng.normal(0, 0.05))
            rho = (rho_heavy if in_wall else rho_light) * (1.0 + noise)
            nodes.append((x, y, morton, rho))
    return nodes


def _write_frame(
    frame_idx: int,
    out_dir: Path,
    n_target: int,
    nodes_full: List[Tuple[int, int, int, float]],
) -> None:
    nodes_full.sort(key=lambda n: n[2])
    nodes = nodes_full[:n_target]

    frame_dir = Path(out_dir) / f"frame_{frame_idx:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    coord_to_idx = {(n[0], n[1]): i for i, n in enumerate(nodes)}
    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []

    for i, (x, y, _morton, rho) in enumerate(nodes):
        diag_val = 0.0
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if (nx, ny) not in coord_to_idx:
                continue
            j = coord_to_idx[(nx, ny)]
            n_rho = nodes[j][3]
            rho_face = 2.0 * rho * n_rho / (rho + n_rho)
            val = -1.0 / rho_face
            rows.append(i)
            cols.append(j)
            values.append(val)
            diag_val -= val
        rows.append(i)
        cols.append(i)
        values.append(diag_val)

    n = n_target
    if _sparse is not None:
        a_coo = _sparse.coo_matrix((values, (rows, cols)), shape=(n, n), dtype=np.float32)
        a_coo.eliminate_zeros()
        a_csr = a_coo.tocsr()
        a_coo2 = a_csr.tocoo()
        r = np.asarray(a_coo2.row, dtype=np.uint32)
        c = np.asarray(a_coo2.col, dtype=np.uint32)
        v = np.asarray(a_coo2.data, dtype=np.float32)
    else:
        r = np.asarray(rows, dtype=np.uint32)
        c = np.asarray(cols, dtype=np.uint32)
        v = np.asarray(values, dtype=np.float32)

    node_array = np.zeros(n_target, dtype=NODE_DTYPE)
    for i, (x, y, morton, rho) in enumerate(nodes):
        node_array[i]["position"] = (float(x), float(y), 0.0)
        node_array[i]["mass"] = np.float32(rho)
        node_array[i]["morton"] = np.uint32(morton)
        node_array[i]["active"] = np.uint32(1)

    node_array.tofile(str(frame_dir / "nodes.bin"))
    r.tofile(str(frame_dir / "edge_index_rows.bin"))
    c.tofile(str(frame_dir / "edge_index_cols.bin"))
    v.tofile(str(frame_dir / "A_values.bin"))

    with open(frame_dir / "meta.txt", "w", encoding="utf-8") as f:
        f.write(f"numNodes: {n_target}\n")


def generate_frame(
    frame_idx: int,
    out_dir: Path,
    n_target: int,
    frame_rng: Any,
    *,
    dataset_type: str,
    rho_light: float,
    rho_heavy: float,
    blob_threshold: float = 0.3,
    shell_thickness: float = 0.02,
    sequence_length: int = 15,
    motion_span: float = 0.35,
    episode_rng: Optional[Any] = None,
    sequence_index: int = 0,
) -> None:
    _ = blob_threshold
    grid_w, grid_h = _grid_dims(n_target)

    if dataset_type == "hard-multiphase":
        nodes_full = _rho_grid_hard_multiphase(grid_w, grid_h, frame_rng, rho_light, rho_heavy)
    elif dataset_type == "thin-shells":
        nodes_full = _rho_grid_thin_shells(
            grid_w, grid_h, frame_rng, rho_light, rho_heavy, shell_thickness
        )
    elif dataset_type == "moving-barrier":
        if episode_rng is None:
            episode_rng = frame_rng
        wall_thickness = float(episode_rng.uniform(0.10, 0.25))
        gap_type = str(episode_rng.choice(["top", "bottom", "middle_hole"]))
        start_c = float(episode_rng.uniform(0.25, 0.75))
        delta = float(episode_rng.uniform(-abs(motion_span), abs(motion_span)))
        sl = max(sequence_length, 2)
        t = sequence_index / (sl - 1)
        wall_x_center = float(np.clip(start_c + t * delta, 0.08, 0.92))
        nodes_full = _rho_grid_hard_multiphase(
            grid_w,
            grid_h,
            frame_rng,
            rho_light,
            rho_heavy,
            wall_x_center=wall_x_center,
            wall_thickness=wall_thickness,
            gap_type=gap_type,
        )
    else:
        raise ValueError(f"unknown dataset_type: {dataset_type}")

    _write_frame(frame_idx, out_dir, n_target, nodes_full)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate LeafOnly dataset frames (hard-multiphase barrier, thin shells, moving barrier)."
    )
    p.add_argument(
        "--dataset-type",
        choices=tuple(TYPE_OUTPUT_SUBDIR.keys()),
        default="hard-multiphase",
        help="Pattern family; default output folder under --output-root uses this name.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Root directory; defaults train/test are <root>/<type>/{train,test}.",
    )
    p.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Override train output directory (default: <output-root>/<type>/train).",
    )
    p.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Override test output directory (default: <output-root>/<type>/test).",
    )
    p.add_argument(
        "--n-target",
        type=int,
        default=_DEFAULT_N_TARGET,
        help=f"Nodes per frame (default {_DEFAULT_N_TARGET}).",
    )
    p.add_argument("--num-train", type=int, default=100)
    p.add_argument("--num-test", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rho-light", type=float, default=1.0, help="Light phase density.")
    p.add_argument(
        "--rho-heavy",
        type=float,
        default=10.0,
        help="Heavy phase density (10:1 is typical for interactive contrast; use 100 for stiffer systems).",
    )
    p.add_argument(
        "--blob-threshold",
        type=float,
        default=0.3,
        help="Unused; kept for compatibility.",
    )
    p.add_argument(
        "--shell-thickness",
        type=float,
        default=0.02,
        help="Normalized cup wall thickness for --dataset-type thin-shells (~1–3 cells depending on N).",
    )
    p.add_argument(
        "--sequence-length",
        type=int,
        default=15,
        help="Frames per translating episode for --dataset-type moving-barrier.",
    )
    p.add_argument(
        "--motion-span",
        type=float,
        default=0.35,
        help="Scale for how far the barrier slides (normalized) within an episode.",
    )
    p.add_argument(
        "--pilot",
        action="store_true",
        help="Small dry run: 8 train / 4 test into <type>/pilot_train and pilot_test if dirs not overridden.",
    )
    args = p.parse_args()

    sub = TYPE_OUTPUT_SUBDIR[args.dataset_type]
    base = Path.cwd()
    out_root = (base / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()

    train_dir = args.train_dir
    test_dir = args.test_dir
    if train_dir is None:
        train_dir = out_root / sub / "train"
    else:
        train_dir = (base / train_dir).resolve() if not train_dir.is_absolute() else train_dir.resolve()
    if test_dir is None:
        test_dir = out_root / sub / "test"
    else:
        test_dir = (base / test_dir).resolve() if not test_dir.is_absolute() else test_dir.resolve()

    if args.pilot:
        args.num_train = 8
        args.num_test = 4
        if args.train_dir is None and args.test_dir is None:
            train_dir = out_root / sub / "pilot_train"
            test_dir = out_root / sub / "pilot_test"

    n_target = int(args.n_target)
    if n_target < 1:
        raise SystemExit("--n-target must be positive")
    rho_light = float(args.rho_light)
    rho_heavy = float(args.rho_heavy)
    if rho_light <= 0 or rho_heavy <= 0:
        raise SystemExit("--rho-light and --rho-heavy must be positive")

    seq_len = max(2, int(args.sequence_length))

    for split_name, out_dir, count in (
        ("train", train_dir, args.num_train),
        ("test", test_dir, args.num_test),
    ):
        print(f"[{args.dataset_type}] Generating {count} frames -> {out_dir} ({split_name})")
        for i in range(count):
            seed_i = int(args.seed) + i + (0 if split_name == "train" else 10_000)
            frame_rng = _rng(seed_i)

            episode_rng: Optional[Any] = None
            sequence_index = 0
            if args.dataset_type == "moving-barrier":
                ep = i // seq_len
                sequence_index = i % seq_len
                episode_rng = _rng(int(args.seed) + ep * 1_000_003 + (0 if split_name == "train" else 500_000))

            generate_frame(
                i,
                out_dir,
                n_target,
                frame_rng,
                dataset_type=args.dataset_type,
                rho_light=rho_light,
                rho_heavy=rho_heavy,
                blob_threshold=float(args.blob_threshold),
                shell_thickness=float(args.shell_thickness),
                sequence_length=seq_len,
                motion_span=float(args.motion_span),
                episode_rng=episode_rng,
                sequence_index=sequence_index,
            )

    print("Done.")


if __name__ == "__main__":
    main()
