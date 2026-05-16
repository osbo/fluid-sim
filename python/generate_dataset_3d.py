#!/usr/bin/env python3
"""Generate the 3D extension of the multiphase_v2 dataset.

Cubic grid (default 20^3 = 8000 nodes, no truncation), 7-point Laplacian with
harmonic-mean density on faces. Each frame has 1..max_barriers axis-aligned
rectangular-prism barriers; each prism is a slab perpendicular to one of x/y/z
with optional rectangular-prism gap-window carved out of it. The gap window is
the intersection of two independent 2D gap_types (top/bottom/middle_hole), so
gaps are 3D-localized (corner block / edge strip / center hole), not extrusions
of a 2D pattern. With 25% probability the slab is fully closed.

Output matches leafonly.data.FluidGraphDataset:
``nodes.bin``, ``edge_index_rows.bin``, ``edge_index_cols.bin``, ``A_values.bin``, ``meta.txt``.
"""
import argparse
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

try:
    from scipy import sparse as _sparse
except ModuleNotFoundError:
    _sparse = None


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


def _rng(seed: int) -> Any:
    if hasattr(np.random, "default_rng"):
        return np.random.default_rng(int(seed))
    return np.random.RandomState(int(seed))


def get_morton_code_3d(x: int, y: int, z: int) -> int:
    res = 0
    for i in range(10):
        res |= (
            ((x & (1 << i)) << (2 * i))
            | ((y & (1 << i)) << (2 * i + 1))
            | ((z & (1 << i)) << (2 * i + 2))
        )
    return int(res)


def _wall_along_axis(coord_frac: float, gap_type: str) -> bool:
    """Returns True at coords where ``gap_type`` keeps the wall (= blocks flow)."""
    if gap_type == "top" and coord_frac > 0.2:
        return True
    if gap_type == "bottom" and coord_frac < 0.8:
        return True
    if gap_type == "middle_hole" and (coord_frac < 0.4 or coord_frac > 0.6):
        return True
    if gap_type == "closed":
        return True
    return False


Barrier = Tuple[str, float, float, str, str]


def _in_prism_barrier(
    x: int,
    y: int,
    z: int,
    dims: Tuple[int, int, int],
    barrier: Barrier,
) -> bool:
    """3D slab perpendicular to one axis; gap is the AND of two 2D gap types
    along the free axes (so gap is 3D-localized, never an extruded 2D pattern)."""
    normal_axis, center, thickness, gap_a, gap_b = barrier
    gw, gh, gd = dims
    if normal_axis == "x":
        n_lo = gw * (center - thickness / 2.0)
        n_hi = gw * (center + thickness / 2.0)
        if not (n_lo < x < n_hi):
            return False
        a_frac = (y + 0.5) / gh
        b_frac = (z + 0.5) / gd
    elif normal_axis == "y":
        n_lo = gh * (center - thickness / 2.0)
        n_hi = gh * (center + thickness / 2.0)
        if not (n_lo < y < n_hi):
            return False
        a_frac = (x + 0.5) / gw
        b_frac = (z + 0.5) / gd
    else:  # z
        n_lo = gd * (center - thickness / 2.0)
        n_hi = gd * (center + thickness / 2.0)
        if not (n_lo < z < n_hi):
            return False
        a_frac = (x + 0.5) / gw
        b_frac = (y + 0.5) / gh
    return _wall_along_axis(a_frac, gap_a) and _wall_along_axis(b_frac, gap_b)


def _sample_barrier(rng: Any) -> Barrier:
    axis = str(rng.choice(["x", "y", "z"]))
    center = float(rng.uniform(0.2, 0.8))
    thickness = float(rng.uniform(0.05, 0.20))
    if float(rng.uniform()) < 0.25:
        return (axis, center, thickness, "closed", "closed")
    open_types = ["top", "bottom", "middle_hole"]
    gap_a = str(rng.choice(open_types))
    gap_b = str(rng.choice(open_types))
    return (axis, center, thickness, gap_a, gap_b)


def _rho_grid(
    gw: int,
    gh: int,
    gd: int,
    frame_rng: Any,
    rho_light: float,
    rho_heavy_min: float,
    rho_heavy_max: float,
    min_barriers: int,
    max_barriers: int,
) -> Tuple[np.ndarray, np.ndarray, float, List[Barrier]]:
    log_lo = np.log(max(rho_light * 1.01, float(rho_heavy_min)))
    log_hi = np.log(max(float(rho_heavy_min) * 1.01, float(rho_heavy_max)))
    rho_heavy = float(np.exp(frame_rng.uniform(log_lo, log_hi)))

    n_b = int(frame_rng.choice(list(range(min_barriers, max_barriers + 1))))
    barriers = [_sample_barrier(frame_rng) for _ in range(n_b)]

    n_total = gw * gh * gd
    coords = np.zeros((n_total, 3), dtype=np.int32)
    morton = np.zeros(n_total, dtype=np.uint32)
    rho = np.empty(n_total, dtype=np.float64)
    dims = (gw, gh, gd)
    idx = 0
    for z in range(gd):
        for y in range(gh):
            for x in range(gw):
                coords[idx] = (x, y, z)
                morton[idx] = get_morton_code_3d(x, y, z)
                in_wall = any(_in_prism_barrier(x, y, z, dims, b) for b in barriers)
                noise = float(frame_rng.normal(0, 0.05))
                rho[idx] = (rho_heavy if in_wall else rho_light) * (1.0 + noise)
                idx += 1
    return coords, morton, rho, barriers


def _build_and_write_frame(
    frame_idx: int,
    out_dir: Path,
    coords: np.ndarray,
    morton: np.ndarray,
    rho: np.ndarray,
    dims: Tuple[int, int, int],
) -> None:
    order = np.argsort(morton, kind="stable")
    coords = coords[order]
    morton = morton[order]
    rho = rho[order]
    n = coords.shape[0]

    gw, gh, gd = dims
    # Build coord -> sorted index lookup table.
    lookup = -np.ones(gw * gh * gd, dtype=np.int64)
    flat = coords[:, 0] + gw * (coords[:, 1] + gh * coords[:, 2])
    lookup[flat] = np.arange(n, dtype=np.int64)

    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []
    neighbor_offsets = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))

    for i in range(n):
        x, y, z = int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])
        rho_i = float(rho[i])
        diag_val = 0.0
        for dx, dy, dz in neighbor_offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            if nx < 0 or nx >= gw or ny < 0 or ny >= gh or nz < 0 or nz >= gd:
                continue
            j = int(lookup[nx + gw * (ny + gh * nz)])
            if j < 0:
                continue
            rho_j = float(rho[j])
            rho_face = 2.0 * rho_i * rho_j / (rho_i + rho_j)
            val = -1.0 / rho_face
            rows.append(i)
            cols.append(j)
            values.append(val)
            diag_val -= val
        rows.append(i)
        cols.append(i)
        values.append(diag_val)

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

    node_array = np.zeros(n, dtype=NODE_DTYPE)
    node_array["position"][:, 0] = coords[:, 0].astype(np.float32)
    node_array["position"][:, 1] = coords[:, 1].astype(np.float32)
    node_array["position"][:, 2] = coords[:, 2].astype(np.float32)
    node_array["mass"] = rho.astype(np.float32)
    node_array["morton"] = morton.astype(np.uint32)
    node_array["active"] = np.uint32(1)

    frame_dir = Path(out_dir) / f"frame_{frame_idx:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    node_array.tofile(str(frame_dir / "nodes.bin"))
    r.tofile(str(frame_dir / "edge_index_rows.bin"))
    c.tofile(str(frame_dir / "edge_index_cols.bin"))
    v.tofile(str(frame_dir / "A_values.bin"))
    with open(frame_dir / "meta.txt", "w", encoding="utf-8") as f:
        f.write(f"numNodes: {n}\n")
        f.write(f"gridDims: {gw} {gh} {gd}\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate 3D multiphase_v2 dataset (cubic grid, prism barriers, 7-pt Laplacian)."
    )
    p.add_argument("--output-root", type=Path, default=Path("data"))
    p.add_argument(
        "--bundle-name",
        type=str,
        default="multiphase_v2_3d_8000",
        help="Subdirectory under output-root for the bundle.",
    )
    p.add_argument("--grid", type=int, default=20, help="Cube edge length; n = grid^3.")
    p.add_argument("--num-train", type=int, default=100)
    p.add_argument("--num-test", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rho-light", type=float, default=1.0)
    p.add_argument("--rho-heavy", type=float, default=100.0)
    p.add_argument("--rho-heavy-min", type=float, default=5.0)
    p.add_argument("--min-barriers", type=int, default=1)
    p.add_argument("--max-barriers", type=int, default=3)
    args = p.parse_args()

    g = int(args.grid)
    if g < 2:
        raise SystemExit("--grid must be at least 2")
    if args.rho_heavy_min >= args.rho_heavy:
        raise SystemExit("--rho-heavy-min must be less than --rho-heavy")
    if args.min_barriers < 1 or args.max_barriers < args.min_barriers:
        raise SystemExit("require 1 <= --min-barriers <= --max-barriers")

    base = Path.cwd()
    out_root = (
        args.output_root.resolve() if args.output_root.is_absolute() else (base / args.output_root).resolve()
    )
    bundle_root = out_root / args.bundle_name
    train_dir = bundle_root / "train"
    test_dir = bundle_root / "test"

    for split_name, out_dir, count in (
        ("train", train_dir, args.num_train),
        ("test", test_dir, args.num_test),
    ):
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[multiphase-v2-3d] Generating {count} frames -> {out_dir} ({split_name})")
        for i in range(count):
            seed_i = int(args.seed) + i + (0 if split_name == "train" else 10_000)
            frame_rng = _rng(seed_i)
            coords, morton, rho, _barriers = _rho_grid(
                g,
                g,
                g,
                frame_rng,
                float(args.rho_light),
                float(args.rho_heavy_min),
                float(args.rho_heavy),
                int(args.min_barriers),
                int(args.max_barriers),
            )
            _build_and_write_frame(i, out_dir, coords, morton, rho, (g, g, g))
            if (i + 1) % 10 == 0 or i + 1 == count:
                print(f"  {split_name}: {i + 1}/{count}")

    print("Done.")


if __name__ == "__main__":
    main()
