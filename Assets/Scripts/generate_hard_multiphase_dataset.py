#!/usr/bin/env python3
"""
Generate Morton-sorted, high-contrast coefficient frames for LeafOnly training / inspection.

Writes per-frame binaries compatible with ``leafonly.data.FluidGraphDataset``:
``nodes.bin``, ``edge_index_rows.bin``, ``edge_index_cols.bin``, ``A_values.bin``, ``meta.txt``.

Grid is sized so width * height >= N_TARGET; nodes are sorted by Morton (Z-order) code,
then truncated to exactly N_TARGET. The Laplacian uses a 5-point stencil with harmonic-mean
density on each face. Larger ``rho_heavy / rho_light`` increases coefficient jumps and usually
worsens conditioning (harder at large N / float32); decrease the ratio for easier systems.

If SciPy is installed, COO edges are normalized via CSR round-trip (same as historical path);
otherwise raw COO triplets are written (equivalent for this acyclic stencil build).
"""
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    from scipy import sparse as _sparse
except ModuleNotFoundError:  # pragma: no cover
    _sparse = None

# Must match ``leafonly.data.NODE_DTYPE`` / ``leafonly.config.MAX_MIXED_SIZE`` (no torch import here).
_DEFAULT_N_TARGET = 2048
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
    """Interleave bits of x and y for Z-order curve spatial locality (up to 16 bits per coord)."""
    res = 0
    for i in range(16):
        res |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1))
    return int(res)


def _grid_dims(n_target: int) -> Tuple[int, int]:
    """Rectangle with w*h >= n_target (fixes int(sqrt(N))^2 < N, e.g. 90*90 vs 8192)."""
    w = int(np.ceil(np.sqrt(n_target)))
    h = int(np.ceil(n_target / w))
    return w, h


def generate_hard_frame(
    frame_idx: int,
    out_dir: Path,
    n_target: int,
    rng,
    rho_light: float = 1.0,
    rho_heavy: float = 100.0,
    blob_threshold: float = 0.3,
) -> None:
    frame_dir = Path(out_dir) / f"frame_{frame_idx:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    grid_w, grid_h = _grid_dims(n_target)

    nodes: list[tuple[int, int, int, float]] = []
    for y in range(grid_h):
        for x in range(grid_w):
            morton = get_morton_code(x, y)
            blob = np.sin(x / 5.0) * np.cos(y / 5.0) + float(rng.normal(0, 0.2))
            rho = rho_heavy if blob > blob_threshold else rho_light
            nodes.append((x, y, morton, rho))

    nodes.sort(key=lambda n: n[2])
    nodes = nodes[:n_target]

    coord_to_idx = {(n[0], n[1]): i for i, n in enumerate(nodes)}

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []

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


def main() -> None:
    p = argparse.ArgumentParser(description="Generate hard multiphase Morton-sorted LeafOnly frames.")
    p.add_argument(
        "--n-target",
        type=int,
        default=_DEFAULT_N_TARGET,
        help=f"Nodes per frame (default {_DEFAULT_N_TARGET}, align with leafonly.config.MAX_MIXED_SIZE).",
    )
    p.add_argument("--train-dir", type=Path, default=Path("data/hard_multiphase_train"))
    p.add_argument("--test-dir", type=Path, default=Path("data/hard_multiphase_test"))
    p.add_argument("--num-train", type=int, default=100)
    p.add_argument("--num-test", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rho-light", type=float, default=1.0, help="Light phase density (default 1).")
    p.add_argument(
        "--rho-heavy",
        type=float,
        default=10.0,
        help="Heavy phase density; lower vs rho-light for milder jumps / better conditioning (default 10).",
    )
    p.add_argument(
        "--blob-threshold",
        type=float,
        default=0.3,
        help="Blob field threshold for heavy vs light (default 0.3); higher → less heavy phase.",
    )
    p.add_argument(
        "--pilot",
        action="store_true",
        help="Small dry run: --num-train 8 --num-test 4 under data/hard_multiphase_pilot_{train,test} (overrides train/test dirs unless you pass them explicitly after this flag).",
    )
    args = p.parse_args()

    if args.pilot:
        args.num_train = 8
        args.num_test = 4
        if str(args.train_dir) == "data/hard_multiphase_train" and str(args.test_dir) == "data/hard_multiphase_test":
            args.train_dir = Path("data/hard_multiphase_pilot_train")
            args.test_dir = Path("data/hard_multiphase_pilot_test")

    n_target = int(args.n_target)
    if n_target < 1:
        raise SystemExit("--n-target must be positive")
    rho_light = float(args.rho_light)
    rho_heavy = float(args.rho_heavy)
    if rho_light <= 0 or rho_heavy <= 0:
        raise SystemExit("--rho-light and --rho-heavy must be positive")
    blob_threshold = float(args.blob_threshold)

    base = Path.cwd()
    train_dir = (base / args.train_dir).resolve() if not args.train_dir.is_absolute() else args.train_dir.resolve()
    test_dir = (base / args.test_dir).resolve() if not args.test_dir.is_absolute() else args.test_dir.resolve()

    for split_name, out_dir, count in (
        ("train", train_dir, args.num_train),
        ("test", test_dir, args.num_test),
    ):
        print(f"Generating {count} frames -> {out_dir} ({split_name})")
        for i in range(count):
            seed_i = int(args.seed) + i + (0 if split_name == "train" else 10_000)
            if hasattr(np.random, "default_rng"):
                frame_rng = np.random.default_rng(seed_i)
            else:
                frame_rng = np.random.RandomState(seed_i)
            generate_hard_frame(
                i,
                out_dir,
                n_target,
                frame_rng,
                rho_light=rho_light,
                rho_heavy=rho_heavy,
                blob_threshold=blob_threshold,
            )

    print("Done.")


if __name__ == "__main__":
    main()
