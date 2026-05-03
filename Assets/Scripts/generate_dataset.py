#!/usr/bin/env python3
"""
Generate Morton-sorted coefficient frames for LeafOnly training / inspection.

Writes per-frame binaries compatible with ``leafonly.data.FluidGraphDataset``:
``nodes.bin``, ``edge_index_rows.bin``, ``edge_index_cols.bin``, ``A_values.bin``, ``meta.txt``.

Choose a ``--dataset-type``. With no ``--train-dir`` / ``--test-dir``, frames go to
``data/unstructured_multiphase_8192/{train,test}/`` (see module constants). If you pass either
directory flag, missing sides fall back to ``<output-root>/<type_folder>/…``.

- **hard-multiphase**: randomized vertical barrier (wall position, thickness, gap topology).
- **moving-barrier**: same barrier family as hard-multiphase but wall center slides along an episode
  (``--sequence-length`` frames per slide) to mimic moving geometry without rebuilding AMG every frame.
- **unstructured-multiphase**: continuous ``[0,1]^2`` particles (jittered lattice), quantized Morton order,
  radius graph via ``KDTree`` (irregular degrees), synthetic vortex ``velocity``, same barrier family as
  hard-multiphase in normalized coordinates.
- **fractured-media**: same point cloud + radius graph as unstructured, but density is ``rho_heavy`` (rock matrix)
  everywhere except thin bands along random fracture lines where ``rho_light`` applies (veins). Use e.g.
  ``--rho-light 1 --rho-heavy 10000`` for extreme contrast.

Grid (non-unstructured): width * height >= ``n-target``; nodes sorted by Morton code then truncated to ``n-target``.
5-point Laplacian with harmonic-mean density on faces.

``unstructured-multiphase`` and ``fractured-media`` require SciPy (``KDTree`` + optional CSR round-trip). Other types use SciPy only if installed.
"""
import argparse
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    from scipy import sparse as _sparse
    from scipy.spatial import KDTree as _KDTree
except ModuleNotFoundError:  # pragma: no cover
    _sparse = None
    _KDTree = None  # type: ignore


def _rng(seed: int) -> Any:
    if hasattr(np.random, "default_rng"):
        return np.random.default_rng(int(seed))
    return np.random.RandomState(int(seed))


# Defaults (override with CLI).
_DEFAULT_N_TARGET = 2048
_DEFAULT_NUM_TRAIN = 100
_DEFAULT_NUM_TEST = 20
_DEFAULT_DATASET_TYPE = "hard-multiphase"
_DEFAULT_DATASET_BUNDLE = "hard_multiphase_2048"

# Default output folder name per --dataset-type (under --output-root).
TYPE_OUTPUT_SUBDIR = {
    "hard-multiphase": "hard_multiphase",
    "moving-barrier": "moving_barrier",
    "unstructured-multiphase": "unstructured_multiphase",
    "fractured-media": "fractured_media",
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


def _in_wall_barrier_continuous(
    px: float,
    py: float,
    wall_x_center: float,
    wall_thickness: float,
    gap_type: str,
) -> bool:
    """Same vertical barrier semantics as ``_in_wall_barrier``, in normalized ``[0,1]^2`` coords."""
    x_lo = wall_x_center - wall_thickness / 2.0
    x_hi = wall_x_center + wall_thickness / 2.0
    if not (x_lo < px < x_hi):
        return False
    if gap_type == "top" and py > 0.2:
        return True
    if gap_type == "bottom" and py < 0.8:
        return True
    if gap_type == "middle_hole" and (py < 0.4 or py > 0.6):
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


# Unstructured point cloud: (px, py, morton, rho, (vx, vy)).
UnstructuredNode = Tuple[float, float, int, float, Tuple[float, float]]


def _rho_point_cloud_multiphase(
    n_target: int,
    frame_rng: Any,
    rho_light: float,
    rho_heavy: float,
    morton_bins: int,
    wall_x_center: Optional[float] = None,
    wall_thickness: Optional[float] = None,
    gap_type: Optional[str] = None,
) -> List[UnstructuredNode]:
    """Jittered regular lattice in ``[0,1]^2``, barrier phasing, quantized Morton, vortex velocity."""
    if wall_x_center is None:
        wall_x_center = float(frame_rng.uniform(0.25, 0.75))
    if wall_thickness is None:
        wall_thickness = float(frame_rng.uniform(0.10, 0.25))
    if gap_type is None:
        gap_type = str(frame_rng.choice(["top", "bottom", "middle_hole"]))

    w = int(np.ceil(np.sqrt(n_target)))
    h = int(np.ceil(n_target / w))
    qmax = max(1, int(morton_bins) - 1)

    nodes: List[UnstructuredNode] = []
    for i in range(n_target):
        grid_x = (i % w) / max(w, 1)
        grid_y = (i // w) / max(h, 1)
        jx = grid_x + float(frame_rng.uniform(-0.4 / max(w, 1), 0.4 / max(w, 1)))
        jy = grid_y + float(frame_rng.uniform(-0.4 / max(h, 1), 0.4 / max(h, 1)))
        px = float(np.clip(jx, 0.0, 1.0))
        py = float(np.clip(jy, 0.0, 1.0))

        qx = int(np.clip(px * qmax, 0, qmax))
        qy = int(np.clip(py * qmax, 0, qmax))
        morton = get_morton_code(qx, qy)

        in_wall = _in_wall_barrier_continuous(px, py, wall_x_center, wall_thickness, gap_type)
        noise = float(frame_rng.normal(0, 0.05))
        rho = (rho_heavy if in_wall else rho_light) * (1.0 + noise)

        vx = -(py - 0.5) * 2.0
        vy = (px - 0.5) * 2.0
        nodes.append((px, py, morton, rho, (vx, vy)))
    return nodes


def _rho_point_cloud_fractured(
    n_target: int,
    frame_rng: Any,
    rho_matrix: float,
    rho_fracture: float,
    morton_bins: int,
    num_fractures: int = 8,
    fracture_thickness: float = 0.015,
) -> List[UnstructuredNode]:
    """Jittered lattice; heavy matrix except thin strips along random lines (fractures = light rho)."""
    w = int(np.ceil(np.sqrt(n_target)))
    h = int(np.ceil(n_target / w))
    qmax = max(1, int(morton_bins) - 1)

    lines: List[Tuple[float, float, float]] = []
    for _ in range(max(1, int(num_fractures))):
        x0 = float(frame_rng.uniform(0.1, 0.9))
        y0 = float(frame_rng.uniform(0.1, 0.9))
        angle = float(frame_rng.uniform(0.0, float(np.pi)))
        nx = float(np.sin(angle))
        ny = float(-np.cos(angle))
        lines.append((nx, ny, -(nx * x0 + ny * y0)))

    half_w = float(fracture_thickness) / 2.0
    nodes: List[UnstructuredNode] = []
    for i in range(n_target):
        grid_x = (i % w) / max(w, 1)
        grid_y = (i // w) / max(h, 1)
        jx = grid_x + float(frame_rng.uniform(-0.4 / max(w, 1), 0.4 / max(w, 1)))
        jy = grid_y + float(frame_rng.uniform(-0.4 / max(h, 1), 0.4 / max(h, 1)))
        px = float(np.clip(jx, 0.0, 1.0))
        py = float(np.clip(jy, 0.0, 1.0))

        qx = int(np.clip(px * qmax, 0, qmax))
        qy = int(np.clip(py * qmax, 0, qmax))
        morton = get_morton_code(qx, qy)

        in_fracture = False
        for ln_x, ln_y, d in lines:
            dist = abs(ln_x * px + ln_y * py + d)
            if dist < half_w:
                in_fracture = True
                break

        rho = float(rho_fracture if in_fracture else rho_matrix)
        vx = -(py - 0.5) * 2.0
        vy = (px - 0.5) * 2.0
        nodes.append((px, py, morton, rho, (vx, vy)))
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


def _write_frame_unstructured(
    frame_idx: int,
    out_dir: Path,
    n_target: int,
    nodes_full: List[UnstructuredNode],
    search_radius: float,
) -> None:
    """Radius graph Laplacian (harmonic-mean density weights); expects SciPy KDTree."""
    if _KDTree is None:
        raise RuntimeError("unstructured-multiphase requires scipy (KDTree)")

    nodes_full.sort(key=lambda n: n[2])
    nodes = nodes_full[:n_target]

    frame_dir = Path(out_dir) / f"frame_{frame_idx:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    positions = np.array([[n[0], n[1]] for n in nodes], dtype=np.float64)
    tree = _KDTree(positions)
    rad = float(search_radius)
    if rad <= 0:
        raise ValueError("search_radius must be positive")

    pairs = tree.query_pairs(rad, p=2, eps=0.0)
    adj: List[List[int]] = [[] for _ in range(n_target)]
    for i, j in pairs:
        adj[i].append(j)
        adj[j].append(i)

    # Guarantee each node has at least one edge (avoids empty rows).
    k_nn = min(8, n_target)
    for i in range(n_target):
        if adj[i]:
            continue
        _, idx = tree.query(positions[i], k=k_nn)
        idx = np.atleast_1d(idx)
        for j in idx:
            j = int(j)
            if j != i and j not in adj[i]:
                adj[i].append(j)
                adj[j].append(i)

    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []

    for i in range(n_target):
        px, py, _m, rho_i, _vel = nodes[i]
        diag_val = 0.0
        for j in adj[i]:
            nx, ny, _mj, rho_j, _ = nodes[j]
            dist = float(np.hypot(px - nx, py - ny))
            if dist <= 1e-12:
                continue
            weight = max(0.0, (rad - dist) / rad)
            rho_face = 2.0 * rho_i * rho_j / (rho_i + rho_j)
            val = -weight / rho_face
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
    for i, (px, py, morton, rho, (vx, vy)) in enumerate(nodes):
        node_array[i]["position"] = (px, py, 0.0)
        node_array[i]["velocity"] = (vx, vy, 0.0)
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
    sequence_length: int = 15,
    motion_span: float = 0.35,
    episode_rng: Optional[Any] = None,
    sequence_index: int = 0,
    smoothing_radius_scale: float = 1.5,
    morton_bins: int = 1024,
    num_fractures: int = 8,
    fracture_thickness: float = 0.015,
) -> None:
    _ = blob_threshold
    grid_w, grid_h = _grid_dims(n_target)

    if dataset_type == "unstructured-multiphase":
        if _KDTree is None:
            raise RuntimeError("dataset-type unstructured-multiphase requires scipy (install scipy).")
        nodes_cloud = _rho_point_cloud_multiphase(
            n_target, frame_rng, rho_light, rho_heavy, morton_bins
        )
        search_radius = float(smoothing_radius_scale) / float(np.sqrt(n_target))
        _write_frame_unstructured(frame_idx, out_dir, n_target, nodes_cloud, search_radius)
        return

    if dataset_type == "fractured-media":
        if _KDTree is None:
            raise RuntimeError("dataset-type fractured-media requires scipy (install scipy).")
        nodes_cloud = _rho_point_cloud_fractured(
            n_target,
            frame_rng,
            rho_matrix=rho_heavy,
            rho_fracture=rho_light,
            morton_bins=morton_bins,
            num_fractures=num_fractures,
            fracture_thickness=fracture_thickness,
        )
        search_radius = float(smoothing_radius_scale) / float(np.sqrt(n_target))
        _write_frame_unstructured(frame_idx, out_dir, n_target, nodes_cloud, search_radius)
        return

    if dataset_type == "hard-multiphase":
        nodes_full = _rho_grid_hard_multiphase(grid_w, grid_h, frame_rng, rho_light, rho_heavy)
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
        description="Generate LeafOnly multiphase dataset frames (grid, moving barrier, unstructured cloud, fractured media)."
    )
    p.add_argument(
        "--dataset-type",
        choices=tuple(TYPE_OUTPUT_SUBDIR.keys()),
        default=_DEFAULT_DATASET_TYPE,
        help="Pattern family; with explicit dirs, train/test default under --output-root uses TYPE_OUTPUT_SUBDIR.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Root when resolving partial paths; default train/test use data/<bundle>/ unless dirs set.",
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
    p.add_argument("--num-train", type=int, default=_DEFAULT_NUM_TRAIN)
    p.add_argument("--num-test", type=int, default=_DEFAULT_NUM_TEST)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rho-light", type=float, default=1.0, help="Light phase density.")
    p.add_argument(
        "--rho-heavy",
        type=float,
        default=10.0,
        help=(
            "Heavy/matrix density. Grid multiphase: often 10. Fractured-media: rock (e.g. 10000); "
            "fractures use --rho-light."
        ),
    )
    p.add_argument(
        "--blob-threshold",
        type=float,
        default=0.3,
        help="Unused; kept for compatibility.",
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
        "--smoothing-radius-scale",
        type=float,
        default=1.5,
        help="SPH-style connectivity: search radius = scale / sqrt(n-target) for point-cloud dataset types.",
    )
    p.add_argument(
        "--morton-quant-bins",
        type=int,
        default=1024,
        help="Virtual grid resolution for Morton quantization (point-cloud types); indices use bins-1.",
    )
    p.add_argument(
        "--num-fractures",
        type=int,
        default=8,
        help="Number of random line fractures for --dataset-type fractured-media.",
    )
    p.add_argument(
        "--fracture-thickness",
        type=float,
        default=0.015,
        help="Strip half-width in normalized coords for fractured-media (distance to line < thickness/2).",
    )
    p.add_argument(
        "--pilot",
        action="store_true",
        help="Small dry run: 8 train / 4 test into bundle pilot_* dirs if train/test dirs not set.",
    )
    args = p.parse_args()

    sub = TYPE_OUTPUT_SUBDIR[args.dataset_type]
    base = Path.cwd()
    out_root = (base / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()

    train_dir = args.train_dir
    test_dir = args.test_dir
    if train_dir is None and test_dir is None:
        bundle_root = (base / Path("data") / _DEFAULT_DATASET_BUNDLE).resolve()
        train_dir = bundle_root / "train"
        test_dir = bundle_root / "test"
    else:
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
            br = (base / Path("data") / _DEFAULT_DATASET_BUNDLE).resolve()
            train_dir = br / "pilot_train"
            test_dir = br / "pilot_test"

    n_target = int(args.n_target)
    if n_target < 1:
        raise SystemExit("--n-target must be positive")
    rho_light = float(args.rho_light)
    rho_heavy = float(args.rho_heavy)
    if rho_light <= 0 or rho_heavy <= 0:
        raise SystemExit("--rho-light and --rho-heavy must be positive")
    if args.dataset_type in ("unstructured-multiphase", "fractured-media") and _KDTree is None:
        raise SystemExit(f"{args.dataset_type} requires scipy; install scipy for KDTree support.")
    if int(args.morton_quant_bins) < 2:
        raise SystemExit("--morton-quant-bins must be at least 2")
    if int(args.num_fractures) < 1:
        raise SystemExit("--num-fractures must be at least 1")
    if float(args.fracture_thickness) <= 0:
        raise SystemExit("--fracture-thickness must be positive")

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
                sequence_length=seq_len,
                motion_span=float(args.motion_span),
                episode_rng=episode_rng,
                sequence_index=sequence_index,
                smoothing_radius_scale=float(args.smoothing_radius_scale),
                morton_bins=int(args.morton_quant_bins),
                num_fractures=int(args.num_fractures),
                fracture_thickness=float(args.fracture_thickness),
            )

    print("Done.")


if __name__ == "__main__":
    main()
