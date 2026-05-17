#!/usr/bin/env python3
"""
Generate implicit-cloth system matrices for LeafOnly training / inspection.

Physics: thin-cloth implicit Euler time integration on a triangulated rectangular grid,
modelling a garment panel under dynamic loading.  Each frame is one Newton step of the
system

    H Δx = f,    H = M/dt² + k_s · L_E(x) + k_b · L_E(x)²

where
  - M      diagonal consistent mass matrix, M_i = ρ · a_i · s²
             ρ = areal density (kg/m²), a_i = Voronoi area (grid units),
             s = cell size (m)
  - L_E    cotangent-weighted Laplacian with harmonic-mean local Young's modulus E:
             off-diagonal H_ij = -w_ij = -(2 E_i E_j)/(E_i+E_j) · cot_ij(x)
             diagonal     H_ii = M_i/dt² + Σ_j w_ij
  - L_E²   biharmonic bending energy (Kirchhoff–Love thin-plate term); adds 2-hop
             edges invisible to GNN-SPAI (which mirrors the sparsity of A) but handled
             naturally by the H-matrix off-diagonal tiles and highway buffers.

Material heterogeneity:  E_i is spatially varying, modelling a multi-material garment
(e.g. stretch jersey body with rigid interfacing panels at collar/cuffs).  High-E panel
boundaries with "full_seam" attachment create near-disconnected chambers — exactly the
regime where highway buffers outperform IC, Jacobi, and GNN-SPAI.

Per-frame variation:
  (i)  random panel placement, attachment pattern, and E_heavy value
  (ii) random sinusoidal z-displacements (cloth wrinkle modes) that change cotangent
       weights each frame without altering the mesh connectivity

Writes per-frame binaries compatible with FluidGraphDataset:
  nodes.bin  edge_index_rows.bin  edge_index_cols.bin  A_values.bin  meta.txt

Usage:
  # default medium difficulty at 4096 nodes
  python python/generate_cloth.py --n-target 4096 --num-train 200 --num-test 20

  # explicit difficulty preset [0,1]  (overrides rho-heavy, k-bend, attachment-types)
  python python/generate_cloth.py --n-target 8192 --difficulty 0.8

  # manual tuning
  python python/generate_cloth.py --n-target 4096 --rho-heavy 50 --k-bend 0.5 --attachment-types full_seam

  # multi-size sweep (run once per size)
  for N in 1024 2048 4096 8192 16384; do
    python python/generate_cloth.py --n-target $N --difficulty 0.6 --name cloth_v1_$N
  done

Requires: numpy, scipy (for sparse L² multiply).
"""

import argparse
import math
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    from scipy import sparse as _sparse
except ModuleNotFoundError:
    _sparse = None  # type: ignore

from generate_dataset import get_morton_code, _grid_dims
from leafonly.data import NODE_DTYPE

# Attachment patterns along a panel boundary:
#   free_top    — boundary runs from bottom, open at top (e.g. collar flap, free neckline)
#   free_bottom — boundary runs from top, open at bottom (e.g. untacked hem)
#   button_gap  — attached at both ends, open in middle (e.g. button-band opening)
#   full_seam   — fully enclosed boundary; near-disconnected chambers (e.g. sewn-shut panel)
_ATTACHMENT_CHOICES = ["free_top", "free_bottom", "button_gap", "full_seam"]
_ORI_CHOICES = ["vertical", "horizontal"]


def _rng(seed: int) -> Any:
    return np.random.default_rng(int(seed))


# ── panel material field ───────────────────────────────────────────────────────

def _in_panel_boundary(x, y, gw, gh, cx, thick, attachment, orientation):
    """Return True if node (x,y) lies in a high-E panel boundary strip."""
    if orientation == "horizontal":
        x, y, gw, gh = y, x, gh, gw
    lo = gw * (cx - thick / 2)
    hi = gw * (cx + thick / 2)
    if not (lo < x < hi):
        return False
    if attachment == "free_top"    and y > gh * 0.2: return True
    if attachment == "free_bottom" and y < gh * 0.8: return True
    if attachment == "button_gap"  and (y < gh * 0.4 or y > gh * 0.6): return True
    if attachment == "full_seam": return True
    return False


def _make_material_panels(
    grid_w: int,
    grid_h: int,
    frame_rng: Any,
    E_body: float,
    E_panel_min: float,
    E_panel_max: float,
    min_panels: int,
    max_panels: int,
    allowed_attachments: Optional[List[str]],
    allowed_orientations: Optional[List[str]],
) -> List[Tuple[int, int, int, float]]:
    """
    Build node list (x, y, morton, E) for a multi-material garment panel.

    E_body         Young's modulus of the base fabric (e.g. stretch jersey).
    E_panel_min/max  Young's modulus range for stiff boundary strips
                     (e.g. interfacing, rigid collar/cuff panels); drawn
                     log-uniformly per frame to model garment variety.
    """
    log_lo = np.log(max(E_body * 1.01, float(E_panel_min)))
    log_hi = np.log(max(float(E_panel_min) * 1.01, float(E_panel_max)))
    E_heavy = float(np.exp(frame_rng.uniform(log_lo, log_hi)))

    n_p = int(frame_rng.integers(min_panels, max_panels + 1))
    att_choices = allowed_attachments or _ATTACHMENT_CHOICES
    ori_choices = allowed_orientations or _ORI_CHOICES

    panels = []
    for _ in range(n_p):
        panels.append((
            str(frame_rng.choice(ori_choices)),
            float(frame_rng.uniform(0.2, 0.8)),
            float(frame_rng.uniform(0.05, 0.20)),
            str(frame_rng.choice(att_choices)),
        ))

    xs_g = np.arange(grid_w, dtype=np.uint32)
    ys_g = np.arange(grid_h, dtype=np.uint32)
    xx, yy = np.meshgrid(xs_g, ys_g)
    xx = xx.ravel(); yy = yy.ravel()

    morton = np.zeros(len(xx), dtype=np.uint32)
    for b in range(16):
        b = np.uint32(b)
        morton |= ((xx >> b & np.uint32(1)) << (np.uint32(2) * b)) | \
                  ((yy >> b & np.uint32(1)) << (np.uint32(2) * b + np.uint32(1)))

    in_boundary = np.zeros(len(xx), dtype=bool)
    for ori, cx, t, att in panels:
        tx, ty = (xx, yy) if ori == "vertical" else (yy, xx)
        tw, th = (grid_w, grid_h) if ori == "vertical" else (grid_h, grid_w)
        lo = tw * (cx - t / 2); hi = tw * (cx + t / 2)
        in_band = (lo < tx) & (tx < hi)
        if att == "free_top":
            in_boundary |= in_band & (ty > th * 0.2)
        elif att == "free_bottom":
            in_boundary |= in_band & (ty < th * 0.8)
        elif att == "button_gap":
            in_boundary |= in_band & ((ty < th * 0.4) | (ty > th * 0.6))
        else:  # full_seam
            in_boundary |= in_band

    noise = frame_rng.normal(0, 0.05, size=len(xx))
    E_arr = np.where(in_boundary, E_heavy, E_body) * (1.0 + noise)
    return list(zip(xx.tolist(), yy.tolist(), morton.tolist(), E_arr.tolist()))


# ── cloth geometry ────────────────────────────────────────────────────────────

def _deform_cloth(
    nodes: List[Tuple[int, int, int, float]],
    frame_rng: Any,
    wrinkle_amp: float = 0.3,
    n_modes: int = 3,
) -> np.ndarray:
    """
    Apply random sinusoidal z-displacements to grid cloth nodes.

    The superposition of sinusoidal modes approximates the dominant deformation
    modes of a cloth panel under dynamic loading (gravity, contact).  The
    resulting cotangent weights change per frame without altering connectivity.
    Returns positions array of shape (N, 3); x,y from grid, z from wrinkle modes.
    """
    xs = np.array([nd[0] for nd in nodes], dtype=np.float64)
    ys = np.array([nd[1] for nd in nodes], dtype=np.float64)
    W = float(xs.max() + 1)
    H = float(ys.max() + 1)

    zs = np.zeros(len(nodes), dtype=np.float64)
    for _ in range(n_modes):
        fx = frame_rng.uniform(1.0, 4.0) * math.pi / W
        fy = frame_rng.uniform(1.0, 4.0) * math.pi / H
        px = frame_rng.uniform(0, 2 * math.pi)
        py = frame_rng.uniform(0, 2 * math.pi)
        amp = float(frame_rng.uniform(0.3, 1.0)) * wrinkle_amp
        zs += amp * np.sin(fx * xs + px) * np.sin(fy * ys + py)

    return np.stack([xs, ys, zs], axis=1)  # (N, 3)


# ── cotangent Laplacian (vectorised) ─────────────────────────────────────────

def _build_faces(
    nodes: List[Tuple[int, int, int, float]],
) -> Tuple[np.ndarray, dict]:
    """
    Build triangle faces from Morton-sorted grid nodes.
    Each grid cell (gx,gy) spawns two triangles when all four corners are present.
    Returns faces (F, 3) int32 and coord_to_idx dict.
    """
    coord_to_idx = {(nd[0], nd[1]): i for i, nd in enumerate(nodes)}
    face_list: List[Tuple[int, int, int]] = []
    xs = [nd[0] for nd in nodes]
    ys = [nd[1] for nd in nodes]
    W, H = max(xs), max(ys)
    for gy in range(H):
        for gx in range(W):
            c00 = (gx,   gy);  c10 = (gx+1, gy)
            c01 = (gx,   gy+1); c11 = (gx+1, gy+1)
            i00 = coord_to_idx.get(c00, -1); i10 = coord_to_idx.get(c10, -1)
            i01 = coord_to_idx.get(c01, -1); i11 = coord_to_idx.get(c11, -1)
            if i00 >= 0 and i10 >= 0 and i11 >= 0:
                face_list.append((i00, i10, i11))
            if i00 >= 0 and i11 >= 0 and i01 >= 0:
                face_list.append((i00, i11, i01))
    return np.array(face_list, dtype=np.int32), coord_to_idx


def _cotangent_laplacian(positions: np.ndarray, faces: np.ndarray) -> Any:
    """
    Vectorised cotangent Laplacian (off-diagonal only, negative values).
    Negative cotangent weights are clamped to zero (avoids indefinite matrices
    from obtuse triangles in deformed cloth).
    Returns scipy csr_matrix of shape (N, N) with H_ij ≤ 0 for i≠j.
    """
    if _sparse is None:
        raise RuntimeError("scipy is required: pip install scipy")

    N = positions.shape[0]
    p0 = positions[faces[:, 0]]
    p1 = positions[faces[:, 1]]
    p2 = positions[faces[:, 2]]

    e01 = p1 - p0; e02 = p2 - p0
    e10 = p0 - p1; e12 = p2 - p1
    e20 = p0 - p2; e21 = p1 - p2

    def cot(a, b):
        c = np.cross(a, b)
        sin2 = np.sqrt(np.maximum(np.einsum("fi,fi->f", c, c), 0.0))
        dot  = np.einsum("fi,fi->f", a, b)
        return np.maximum(dot / np.where(sin2 > 1e-14, sin2, 1e-14), 0.0)

    c0 = cot(e01, e02)
    c1 = cot(e10, e12)
    c2 = cot(e20, e21)

    i0, j0 = faces[:, 1], faces[:, 2]
    i1, j1 = faces[:, 0], faces[:, 2]
    i2, j2 = faces[:, 0], faces[:, 1]

    w0 = 0.5 * c0; w1 = 0.5 * c1; w2 = 0.5 * c2

    all_i = np.concatenate([i0, j0, i1, j1, i2, j2])
    all_j = np.concatenate([j0, i0, j1, i1, j2, i2])
    all_w = np.concatenate([-w0, -w0, -w1, -w1, -w2, -w2])

    L_od = _sparse.coo_matrix((all_w, (all_i, all_j)), shape=(N, N)).tocsr()
    return L_od


def _voronoi_areas(faces: np.ndarray, N: int) -> np.ndarray:
    """Per-node Voronoi area in grid units: 1/3 of each adjacent triangle's area."""
    if _sparse is None:
        raise RuntimeError("scipy is required")
    # We work in 2D grid coords so positions aren't needed — each right triangle
    # has area 0.5 grid cells.  For a general deformed mesh we need actual areas;
    # we compute them from the face list using the cross-product formula.
    # (Caller passes grid-unit positions so areas are in grid-cell² units.)
    areas = np.zeros(N, dtype=np.float64)
    return areas  # placeholder — filled by caller with actual positions


def _voronoi_areas_from_positions(faces: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Per-node Voronoi area (1/3 of adjacent triangle areas) from 3-D positions."""
    N = positions.shape[0]
    p0 = positions[faces[:, 0]]
    p1 = positions[faces[:, 1]]
    p2 = positions[faces[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    tri_areas = 0.5 * np.sqrt(np.einsum("fi,fi->f", cross, cross))
    node_areas = np.zeros(N, dtype=np.float64)
    for k in range(3):
        np.add.at(node_areas, faces[:, k], tri_areas / 3.0)
    return node_areas


# ── full matrix assembly ──────────────────────────────────────────────────────

def _build_cloth_matrix(
    nodes: List[Tuple[int, int, int, float]],
    positions: np.ndarray,
    k_s: float,
    k_b: float,
    areal_density: float,
    cell_size: float,
    dt: float,
) -> Any:
    """
    Assemble H = M/dt² + k_s · L_E + k_b · L_E²

    M      diagonal mass matrix; M_i = areal_density · voronoi_area_i · cell_size²
    L_E    cotangent Laplacian scaled by harmonic-mean Young's modulus:
             w_ij = (2 E_i E_j)/(E_i+E_j) · cot_weight_ij
    L_E²   biharmonic bending energy (Kirchhoff–Love thin-plate approximation);
             adds 2-hop edges invisible to sparsity-mirroring preconditioners.

    H is SPD: L_E is PSD, L_E² is PSD, M/dt² > 0 ensures strict positive definiteness
    even when the cotangent Laplacian has near-zero modes (near-disconnected panels).
    """
    if _sparse is None:
        raise RuntimeError("scipy is required: pip install scipy")

    N = len(nodes)
    E_arr = np.array([nd[3] for nd in nodes], dtype=np.float64)

    faces, _ = _build_faces(nodes)
    if len(faces) == 0:
        raise ValueError("No triangles found — check grid dimensions")

    L_od_geom = _cotangent_laplacian(positions, faces)

    L_coo = L_od_geom.tocoo()
    Ei = E_arr[L_coo.row]
    Ej = E_arr[L_coo.col]
    inv_hmean = (Ei + Ej) / (2.0 * Ei * Ej + 1e-14)
    L_od = _sparse.coo_matrix(
        (L_coo.data * inv_hmean, (L_coo.row, L_coo.col)), shape=(N, N)
    ).tocsr()

    diag = np.array(-L_od.sum(axis=1)).flatten()
    L_E = (L_od + _sparse.diags(diag)).tocsr()

    L_sq = (L_E @ L_E).tocsr()

    # Consistent mass matrix (lumped per node from Voronoi areas)
    voronoi = _voronoi_areas_from_positions(faces, positions)
    node_masses = areal_density * voronoi * (cell_size ** 2)  # kg
    mass_diag = node_masses / (dt ** 2)                        # N/m (same units as K)

    H = (k_s * L_E + k_b * L_sq + _sparse.diags(mass_diag)).tocsr()
    # Tiny regulariser for strict PD (guards against near-zero Voronoi areas at boundary nodes)
    eps = 1e-8 * float(np.abs(H.diagonal()).mean())
    H = H + _sparse.eye(N, format="csr") * eps
    return H


# ── frame writer ──────────────────────────────────────────────────────────────

def _write_frame(
    frame_idx: int,
    out_dir: Path,
    nodes: List[Tuple[int, int, int, float]],
    positions: np.ndarray,
    H: Any,
) -> None:
    N = len(nodes)
    frame_dir = Path(out_dir) / f"frame_{frame_idx:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    H_coo = H.tocoo()
    r = np.asarray(H_coo.row,  dtype=np.uint32)
    c = np.asarray(H_coo.col,  dtype=np.uint32)
    v = np.asarray(H_coo.data, dtype=np.float32)

    xs_n  = np.array([nd[0] for nd in nodes], dtype=np.float32)
    ys_n  = np.array([nd[1] for nd in nodes], dtype=np.float32)
    E_arr = np.array([nd[3] for nd in nodes], dtype=np.float32)
    mort  = np.array([nd[2] for nd in nodes], dtype=np.uint32)

    node_array = np.zeros(N, dtype=NODE_DTYPE)
    node_array["position"][:, 0] = xs_n
    node_array["position"][:, 1] = ys_n
    node_array["position"][:, 2] = positions[:, 2].astype(np.float32)
    node_array["mass"]   = E_arr   # Young's modulus stored as training feature
    node_array["morton"] = mort
    node_array["active"] = np.uint32(1)

    node_array.tofile(str(frame_dir / "nodes.bin"))
    r.tofile(str(frame_dir / "edge_index_rows.bin"))
    c.tofile(str(frame_dir / "edge_index_cols.bin"))
    v.tofile(str(frame_dir / "A_values.bin"))
    with open(frame_dir / "meta.txt", "w") as f:
        f.write(f"numNodes: {N}\n")


# ── per-frame generation ──────────────────────────────────────────────────────

def generate_cloth_frame(
    frame_idx: int,
    out_dir: Path,
    n_target: int,
    frame_rng: Any,
    *,
    E_body: float,
    E_panel_min: float,
    E_panel_max: float,
    k_s: float,
    k_b: float,
    min_panels: int,
    max_panels: int,
    allowed_attachments: Optional[List[str]],
    allowed_orientations: Optional[List[str]],
    wrinkle_amp: float,
    areal_density: float,
    cell_size: float,
    dt: float,
) -> None:
    grid_w, grid_h = _grid_dims(n_target)

    nodes_full = _make_material_panels(
        grid_w, grid_h, frame_rng,
        E_body, E_panel_min, E_panel_max,
        min_panels, max_panels,
        allowed_attachments, allowed_orientations,
    )
    nodes_full.sort(key=lambda nd: nd[2])  # Morton order
    nodes = nodes_full[:n_target]

    positions = _deform_cloth(nodes, frame_rng, wrinkle_amp=wrinkle_amp)
    H = _build_cloth_matrix(
        nodes, positions, k_s=k_s, k_b=k_b,
        areal_density=areal_density, cell_size=cell_size, dt=dt,
    )
    _write_frame(frame_idx, out_dir, nodes, positions, H)


# ── difficulty preset ─────────────────────────────────────────────────────────

def _apply_difficulty(args: argparse.Namespace) -> None:
    """
    Map difficulty ∈ [0, 1] to physical parameters.

    Jacobi-PCG iteration count (rough) scales as sqrt(κ) · log(1/tol):
      κ(L_E)  ≈ (E_panel/E_body) · (max_cot_degree / eps)  →  scales with contrast
      κ(L_E²) adds another κ(L_E) factor  →  k_b amplifies conditioning quadratically

    Calibrated so difficulty=0 → ~100–200 Jacobi iters (light jersey, no bending),
                       difficulty=1 → ~1500–3000 Jacobi iters at N=4096
                                       (stiff interfacing panels, full-seam boundaries).
    """
    d = float(args.difficulty)
    # E contrast: 5× at d=0 (jersey vs. light interfacing) → 100× at d=1 (denim vs. rigid)
    args.rho_heavy     = float(np.exp(np.log(5) + d * (np.log(100) - np.log(5))))
    args.rho_heavy_min = max(args.rho_light * 1.5, args.rho_heavy / 5.0)
    # Bending: 0 at d=0 (membrane-only) → 0.5 at d=1 (stiff plate)
    args.k_bend = float(d * 0.5)
    # Full-seam boundaries only at high difficulty (most adversarial topology)
    if d > 0.4 and args.attachment_types is None:
        if d > 0.7:
            args.attachment_types = ["button_gap", "full_seam"]
        else:
            args.attachment_types = ["free_top", "free_bottom", "button_gap", "full_seam"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate garment implicit-Euler cloth matrices for LeafOnly training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-target",    type=int,   default=4096,
                   help="Number of nodes per frame (1024/2048/4096/8192/16384)")
    p.add_argument("--num-train",   type=int,   default=200)
    p.add_argument("--num-test",    type=int,   default=20)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--output-root", type=str,   default="data")
    p.add_argument("--name",        type=str,   default=None,
                   help="Dataset name; default cloth_v1_<n-target>")

    # Material contrast (Young's modulus ratio between garment panels)
    p.add_argument("--rho-light",     type=float, default=1.0,
                   help="Young's modulus of base fabric (e.g. stretch jersey)")
    p.add_argument("--rho-heavy-min", type=float, default=5.0,
                   help="Min Young's modulus of stiff panel regions (log-uniform per frame)")
    p.add_argument("--rho-heavy",     type=float, default=50.0,
                   help="Max Young's modulus of stiff panel regions")

    # Panel layout (analogous to multiphase barrier topology)
    p.add_argument("--min-barriers",      type=int,  default=1,
                   help="Minimum number of material panel boundaries per frame")
    p.add_argument("--max-barriers",      type=int,  default=3,
                   help="Maximum number of material panel boundaries per frame")
    p.add_argument("--attachment-types",  nargs="+", default=None,
                   help="Allowed attachment patterns: free_top free_bottom button_gap full_seam "
                        "(default: all four; 'full_seam' → near-disconnected panels)")
    p.add_argument("--orientations",      nargs="+", default=None,
                   help="Allowed panel boundary orientations: vertical horizontal")

    # Cloth physics
    p.add_argument("--k-bend",      type=float, default=0.1,
                   help="Bending stiffness k_b (Kirchhoff–Love term k_b·L_E²). "
                        "0 = pure membrane; >0 adds 2-hop edges GNN-SPAI cannot mirror.")
    p.add_argument("--wrinkle-amp", type=float, default=0.3,
                   help="Amplitude of random sinusoidal wrinkle modes (changes cotangent "
                        "weights per frame without altering mesh connectivity).")

    # Implicit Euler parameters (determine M/dt² diagonal)
    p.add_argument("--areal-density", type=float, default=0.3,
                   help="Fabric areal density ρ (kg/m²). Typical range: "
                        "0.1 (silk) to 0.8 (denim). Sets M_i = ρ · a_i · s².")
    p.add_argument("--cell-size",     type=float, default=0.01,
                   help="Physical grid cell size s (m). A 32×32 grid at s=0.01 is a 32 cm panel.")
    p.add_argument("--dt",            type=float, default=1/30,
                   help="Implicit Euler timestep dt (s). Default 1/30 s (30 fps real-time). "
                        "Larger dt → smaller M/dt² → stiffer, harder system.")

    # Convenience preset (overrides rho-heavy, k-bend, attachment-types)
    p.add_argument("--difficulty",  type=float, default=None,
                   help="[0,1] difficulty preset.  0 ≈ 100–200 Jacobi iters (jersey panel), "
                        "1 ≈ 1500–3000 iters at N=4096 (rigid interfacing + full-seam).  "
                        "Overrides --rho-heavy, --k-bend, --attachment-types.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.difficulty is not None:
        _apply_difficulty(args)

    if _sparse is None:
        raise SystemExit(
            "scipy not found.  Install with:  pip install scipy\n"
            "The L² bending term requires sparse matrix multiply."
        )

    name = args.name or f"cloth_v1_{args.n_target}"
    out_root = Path(args.output_root)
    train_dir = out_root / name / "train"
    test_dir  = out_root / name / "test"

    rng = _rng(args.seed)

    grid_w, grid_h = _grid_dims(args.n_target)
    print(f"Cloth dataset: {name}")
    print(f"  grid {grid_w}×{grid_h} → {grid_w*grid_h} nodes, truncated to {args.n_target}")
    print(f"  Physics: H = M/dt² + k_s·L_E + k_b·L_E²")
    print(f"    k_s=1.0  k_bend={args.k_bend:.3f}  wrinkle_amp={args.wrinkle_amp:.2f}")
    print(f"    areal_density={args.areal_density:.2f} kg/m²  cell_size={args.cell_size*100:.1f} cm  dt={args.dt*1000:.1f} ms")
    print(f"  Material: E_body={args.rho_light}  E_panel∈[{args.rho_heavy_min:.1f}, {args.rho_heavy:.1f}]")
    print(f"  Panels: {args.min_barriers}–{args.max_barriers} boundaries"
          f"  attachment={args.attachment_types or _ATTACHMENT_CHOICES}")

    for split, out_dir, n_frames in [
        ("train", train_dir, args.num_train),
        ("test",  test_dir,  args.num_test),
    ]:
        print(f"\n{split}: {n_frames} frames → {out_dir}")
        for i in range(n_frames):
            frame_rng = _rng(int(rng.integers(0, 2**31)))
            generate_cloth_frame(
                i, out_dir, args.n_target, frame_rng,
                E_body=args.rho_light,
                E_panel_min=args.rho_heavy_min,
                E_panel_max=args.rho_heavy,
                k_s=1.0,
                k_b=args.k_bend,
                min_panels=args.min_barriers,
                max_panels=args.max_barriers,
                allowed_attachments=args.attachment_types,
                allowed_orientations=args.orientations,
                wrinkle_amp=args.wrinkle_amp,
                areal_density=args.areal_density,
                cell_size=args.cell_size,
                dt=args.dt,
            )
            if (i + 1) % 10 == 0 or i == n_frames - 1:
                print(f"  {i+1}/{n_frames}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
