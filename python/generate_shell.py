#!/usr/bin/env python3
"""
Generate 3D thin-shell implicit-Euler system matrices for LeafOnly training.

Physics: one implicit Euler Newton step on a triangulated 3D surface mesh,
modelling a thin structural shell or garment panel under dynamic loading:

    H = M/dt² + k_s · L_E(x) + k_b · L_E(x)²

where
  M    diagonal lumped mass matrix, M_i = ρ · a_i · s²
         ρ = areal density (kg/m²), a_i = Voronoi area on the 3D surface,
         s = reference cell size (m)
  L_E  cotangent-weighted Laplacian computed from actual 3D positions,
         incorporating Gaussian curvature; harmonic-mean Young's modulus E:
         off-diagonal H_ij = -(2 E_i E_j)/(E_i+E_j) · cot_ij(x)
  L_E² biharmonic bending energy (Kirchhoff–Love thin-plate term); adds
         2-hop edges invisible to GNN-SPAI and local preconditioners,
         handled naturally by the H-matrix off-diagonal tiles.

Surface shapes (one sampled per frame when --shape random):
  cylinder      periodic in u  (sleeve, pressure vessel, tube)
  torus         periodic in u and v  (tire, toroidal membrane)
  spherical_cap periodic in u, open at poles  (dome, cap, hood)
  saddle        hyperbolic paraboloid, no periodicity  (shell roof, panel)

Periodic connectivity (cylinder / torus) creates wrap-around edges between
nodes that are Morton-distant — a naturally adversarial structure for local
preconditioners that requires the highway buffers to route correctly.

Material heterogeneity: panel-boundary logic identical to cloth and multiphase
v2. High-E strips in (u, v) parameter space create near-disconnected regions on
the 3D surface; "full_seam" boundaries isolate chambers geodesically.

Per-frame variation:
  (i)   random shape parameters (radii, aspect ratio, curvature)
  (ii)  random sinusoidal normal-direction deformations (vibration modes)
  (iii) random panel placement, attachment pattern, and E_heavy

Writes per-frame binaries compatible with FluidGraphDataset:
  nodes.bin  edge_index_rows.bin  edge_index_cols.bin  A_values.bin  meta.txt

Usage:
  python python/generate_shell.py --n-target 1024 --num-train 200 --num-test 20
  python python/generate_shell.py --n-target 4096 --shape cylinder --difficulty 0.6
  python python/generate_shell.py --n-target 1024 --shape random --name shell_v1_1024

Requires: numpy, scipy (sparse L² multiply).
"""

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import sparse as _sparse
except ModuleNotFoundError:
    _sparse = None  # type: ignore

from generate_dataset import _grid_dims
from leafonly.data import NODE_DTYPE

# Attachment patterns along a panel boundary (same semantics as generate_cloth.py).
_ATTACHMENT_CHOICES = ["free_top", "free_bottom", "button_gap", "full_seam"]
_ORI_CHOICES        = ["vertical", "horizontal"]
_SHAPE_CHOICES      = ["cylinder", "torus", "spherical_cap", "saddle"]


def _rng(seed: int) -> Any:
    return np.random.default_rng(int(seed))


# ── surface parameterization ───────────────────────────────────────────────────

def _sample_shape_params(shape: str, frame_rng: Any) -> Dict:
    """Sample random geometric parameters for one surface instance."""
    if shape == "cylinder":
        return dict(
            R=float(frame_rng.uniform(0.8, 1.2)),
            H=float(frame_rng.uniform(1.5, 3.0)),
            wrap_u=True, wrap_v=False,
        )
    if shape == "torus":
        R = float(frame_rng.uniform(1.5, 2.5))
        r = R * float(frame_rng.uniform(0.15, 0.35))
        return dict(R=R, r=r, wrap_u=True, wrap_v=True)
    if shape == "spherical_cap":
        return dict(
            R=float(frame_rng.uniform(0.8, 1.5)),
            theta_max=float(frame_rng.uniform(math.pi / 3, 4 * math.pi / 5)),
            wrap_u=True, wrap_v=False,
        )
    if shape == "saddle":
        return dict(
            a=float(frame_rng.uniform(0.8, 1.2)),
            b=float(frame_rng.uniform(0.8, 1.2)),
            c=float(frame_rng.uniform(0.2, 0.6)),
            wrap_u=False, wrap_v=False,
        )
    raise ValueError(f"Unknown shape: {shape!r}")


def _base_positions(
    u_idx: np.ndarray,
    v_idx: np.ndarray,
    grid_w: int,
    grid_h: int,
    shape: str,
    params: Dict,
) -> np.ndarray:
    """Map integer (u_idx, v_idx) grid coordinates to 3D surface positions."""
    wrap_v = params["wrap_v"]
    u = u_idx.astype(np.float64) / grid_w
    v = v_idx.astype(np.float64) / grid_h if wrap_v else \
        v_idx.astype(np.float64) / max(grid_h - 1, 1)

    if shape == "cylinder":
        R, H = params["R"], params["H"]
        return np.stack([
            R * np.cos(2 * math.pi * u),
            R * np.sin(2 * math.pi * u),
            H * v,
        ], axis=1)

    if shape == "torus":
        R, r = params["R"], params["r"]
        return np.stack([
            (R + r * np.cos(2 * math.pi * v)) * np.cos(2 * math.pi * u),
            (R + r * np.cos(2 * math.pi * v)) * np.sin(2 * math.pi * u),
            r * np.sin(2 * math.pi * v),
        ], axis=1)

    if shape == "spherical_cap":
        R, theta_max = params["R"], params["theta_max"]
        # Start at theta_min > 0 to avoid the pole singularity (degenerate triangles).
        theta_min = theta_max / grid_h
        theta = theta_min + (theta_max - theta_min) * v
        return np.stack([
            R * np.sin(theta) * np.cos(2 * math.pi * u),
            R * np.sin(theta) * np.sin(2 * math.pi * u),
            R * np.cos(theta),
        ], axis=1)

    if shape == "saddle":
        a, b, c = params["a"], params["b"], params["c"]
        xu = a * (u - 0.5)
        yv = b * (v - 0.5)
        # Normalise so z ∈ [−c, c] at the grid corners.
        z = c * 4.0 * (xu ** 2 / max(a ** 2, 1e-8) - yv ** 2 / max(b ** 2, 1e-8))
        return np.stack([xu, yv, z], axis=1)

    raise ValueError(f"Unknown shape: {shape!r}")


# ── mesh topology ──────────────────────────────────────────────────────────────

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
    Build node list (u_idx, v_idx, morton, E) with material panel field.
    Panels are defined in (u, v) parameter space; same logic as cloth and
    multiphase v2.  On a cylinder the "vertical" panels run along the axis,
    on a torus along the toroidal direction, etc.
    """
    log_lo  = np.log(max(E_body * 1.01, float(E_panel_min)))
    log_hi  = np.log(max(float(E_panel_min) * 1.01, float(E_panel_max)))
    E_heavy = float(np.exp(frame_rng.uniform(log_lo, log_hi)))

    n_p         = int(frame_rng.integers(min_panels, max_panels + 1))
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

    us = np.arange(grid_w, dtype=np.uint32)
    vs = np.arange(grid_h, dtype=np.uint32)
    uu, vv = np.meshgrid(us, vs)
    uu = uu.ravel(); vv = vv.ravel()

    # Morton encode from (u_idx, v_idx) — parameter-space spatial ordering.
    morton = np.zeros(len(uu), dtype=np.uint32)
    for b in range(16):
        b_np = np.uint32(b)
        morton |= (
            ((uu >> b_np & np.uint32(1)) << (np.uint32(2) * b_np)) |
            ((vv >> b_np & np.uint32(1)) << (np.uint32(2) * b_np + np.uint32(1)))
        )

    in_boundary = np.zeros(len(uu), dtype=bool)
    for ori, cx, t, att in panels:
        tx, ty = (uu, vv) if ori == "vertical" else (vv, uu)
        tw, th = (grid_w, grid_h) if ori == "vertical" else (grid_h, grid_w)
        lo = tw * (cx - t / 2); hi = tw * (cx + t / 2)
        in_band = (lo < tx) & (tx < hi)
        if   att == "free_top":    in_boundary |= in_band & (ty > th * 0.2)
        elif att == "free_bottom": in_boundary |= in_band & (ty < th * 0.8)
        elif att == "button_gap":  in_boundary |= in_band & ((ty < th * 0.4) | (ty > th * 0.6))
        else:                      in_boundary |= in_band   # full_seam

    noise = frame_rng.normal(0, 0.05, size=len(uu))
    E_arr = np.where(in_boundary, E_heavy, E_body) * (1.0 + noise)
    return list(zip(uu.tolist(), vv.tolist(), morton.tolist(), E_arr.tolist()))


def _build_faces_periodic(
    coord_to_idx: dict,
    grid_w: int,
    grid_h: int,
    wrap_u: bool,
    wrap_v: bool,
) -> np.ndarray:
    """
    Build triangle faces with optional periodic boundaries.

    For cylinder / torus, the wrap-around edges connect Morton-distant nodes,
    creating long-range connections in the matrix that expose the limitation of
    local preconditioners (IC fills only within a local band; highways cross it).
    Faces are skipped when any corner node was truncated out of the node set.
    """
    n_u = grid_w if wrap_u else grid_w - 1
    n_v = grid_h if wrap_v else grid_h - 1
    faces = []
    for gv in range(n_v):
        for gu in range(n_u):
            gu1 = (gu + 1) % grid_w
            gv1 = (gv + 1) % grid_h
            i00 = coord_to_idx.get((gu,  gv),  -1)
            i10 = coord_to_idx.get((gu1, gv),  -1)
            i01 = coord_to_idx.get((gu,  gv1), -1)
            i11 = coord_to_idx.get((gu1, gv1), -1)
            if i00 >= 0 and i10 >= 0 and i11 >= 0:
                faces.append((i00, i10, i11))
            if i00 >= 0 and i11 >= 0 and i01 >= 0:
                faces.append((i00, i11, i01))
    return np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)


# ── geometry helpers ───────────────────────────────────────────────────────────

def _vertex_normals(faces: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals for normal-direction deformation."""
    N   = positions.shape[0]
    p0  = positions[faces[:, 0]]
    p1  = positions[faces[:, 1]]
    p2  = positions[faces[:, 2]]
    fn  = np.cross(p1 - p0, p2 - p0)   # area-weighted face normals
    vn  = np.zeros((N, 3), dtype=np.float64)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    nrm = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(nrm, 1e-14)


def _deform_shell(
    base_pos: np.ndarray,
    faces: np.ndarray,
    u_norm: np.ndarray,
    v_norm: np.ndarray,
    frame_rng: Any,
    deform_amp: float,
    n_modes: int = 3,
) -> np.ndarray:
    """
    Apply random sinusoidal normal-direction deformations to the base surface.

    Displacing along the surface normal keeps nodes on a deformed version of
    the same surface — physically, this models dynamic loading (pressure,
    impact, vibration modes) that causes the shell to deflect out-of-plane,
    changing cotangent weights per frame without altering mesh connectivity.
    Amplitude is scaled by the bounding-box diagonal so deform_amp=0.05 is
    consistently 5% of the surface extent regardless of shape or size.
    """
    bbox_diag = float(np.linalg.norm(base_pos.max(axis=0) - base_pos.min(axis=0)))
    normals   = _vertex_normals(faces, base_pos)
    disp      = np.zeros(len(base_pos), dtype=np.float64)
    for _ in range(n_modes):
        fu  = frame_rng.uniform(1.0, 4.0) * 2 * math.pi
        fv  = frame_rng.uniform(1.0, 4.0) * 2 * math.pi
        pu  = frame_rng.uniform(0, 2 * math.pi)
        pv  = frame_rng.uniform(0, 2 * math.pi)
        amp = float(frame_rng.uniform(0.3, 1.0)) * deform_amp * bbox_diag
        disp += amp * np.sin(fu * u_norm + pu) * np.sin(fv * v_norm + pv)
    return base_pos + normals * disp[:, np.newaxis]


def _cotangent_laplacian(positions: np.ndarray, faces: np.ndarray) -> Any:
    """
    Vectorised cotangent Laplacian from 3D positions (off-diagonal, negative).
    Clamped to zero for obtuse angles — preserves positive semi-definiteness.
    Identical to generate_cloth.py; works for any 3D embedding.
    """
    if _sparse is None:
        raise RuntimeError("scipy required: pip install scipy")
    N  = positions.shape[0]
    p0 = positions[faces[:, 0]]
    p1 = positions[faces[:, 1]]
    p2 = positions[faces[:, 2]]

    def cot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        c    = np.cross(a, b)
        sin2 = np.sqrt(np.maximum(np.einsum("fi,fi->f", c, c), 0.0))
        dot  = np.einsum("fi,fi->f", a, b)
        return np.maximum(dot / np.where(sin2 > 1e-14, sin2, 1e-14), 0.0)

    c0 = cot(p1 - p0, p2 - p0)
    c1 = cot(p0 - p1, p2 - p1)
    c2 = cot(p0 - p2, p1 - p2)

    i0, j0 = faces[:, 1], faces[:, 2]
    i1, j1 = faces[:, 0], faces[:, 2]
    i2, j2 = faces[:, 0], faces[:, 1]
    all_i  = np.concatenate([i0, j0, i1, j1, i2, j2])
    all_j  = np.concatenate([j0, i0, j1, i1, j2, i2])
    all_w  = np.concatenate([-0.5*c0, -0.5*c0, -0.5*c1, -0.5*c1, -0.5*c2, -0.5*c2])
    return _sparse.coo_matrix((all_w, (all_i, all_j)), shape=(N, N)).tocsr()


def _voronoi_areas(faces: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Per-node Voronoi area on the 3D surface (1/3 of adjacent triangle areas)."""
    N   = positions.shape[0]
    p0  = positions[faces[:, 0]]
    p1  = positions[faces[:, 1]]
    p2  = positions[faces[:, 2]]
    tri = 0.5 * np.sqrt(np.einsum("fi,fi->f",
                                   np.cross(p1 - p0, p2 - p0),
                                   np.cross(p1 - p0, p2 - p0)))
    areas = np.zeros(N, dtype=np.float64)
    for k in range(3):
        np.add.at(areas, faces[:, k], tri / 3.0)
    return areas


# ── matrix assembly ────────────────────────────────────────────────────────────

def _build_shell_matrix(
    nodes: List[Tuple[int, int, int, float]],
    positions: np.ndarray,
    faces: np.ndarray,
    k_s: float,
    k_b: float,
    areal_density: float,
    cell_size: float,
    dt: float,
) -> Any:
    """
    Assemble H = M/dt² + k_s · L_E + k_b · L_E²

    L_E  cotangent Laplacian from actual 3D positions with harmonic-mean E.
    L_E² biharmonic bending (Kirchhoff–Love): 2-hop edges invisible to
         sparsity-mirroring preconditioners (GNN-SPAI, IC).
    M    lumped mass from Voronoi areas on the 3D surface:
         M_i = areal_density · a_i · cell_size²

    H is SPD: L_E PSD, L_E² PSD, M/dt² > 0 lifts the minimum eigenvalue
    away from zero even for near-disconnected panel boundaries (full_seam),
    and for the wrap-around near-singularity of periodic meshes.
    """
    if _sparse is None:
        raise RuntimeError("scipy required: pip install scipy")
    N     = len(nodes)
    E_arr = np.array([nd[3] for nd in nodes], dtype=np.float64)

    # Cotangent Laplacian with harmonic-mean Young's modulus
    L_od_geom = _cotangent_laplacian(positions, faces)
    coo       = L_od_geom.tocoo()
    Ei, Ej    = E_arr[coo.row], E_arr[coo.col]
    inv_hm    = (Ei + Ej) / (2.0 * Ei * Ej + 1e-14)
    L_od      = _sparse.coo_matrix(
        (coo.data * inv_hm, (coo.row, coo.col)), shape=(N, N)
    ).tocsr()
    diag = np.array(-L_od.sum(axis=1)).ravel()
    L_E  = (L_od + _sparse.diags(diag)).tocsr()
    L_sq = (L_E @ L_E).tocsr()

    # Diagonal mass matrix from 3D Voronoi areas
    areas      = _voronoi_areas(faces, positions)
    mass_diag  = areal_density * areas * (cell_size ** 2) / (dt ** 2)

    H   = (k_s * L_E + k_b * L_sq + _sparse.diags(mass_diag)).tocsr()
    eps = 1e-8 * float(np.abs(H.diagonal()).mean())
    return (H + _sparse.eye(N, format="csr") * eps).tocsr()


# ── frame writer ───────────────────────────────────────────────────────────────

def _write_frame(
    frame_idx: int,
    out_dir: Path,
    nodes: List[Tuple[int, int, int, float]],
    positions: np.ndarray,
    H: Any,
) -> None:
    N       = len(nodes)
    fdir    = Path(out_dir) / f"frame_{frame_idx:04d}"
    fdir.mkdir(parents=True, exist_ok=True)

    H_coo   = H.tocoo()
    np.asarray(H_coo.row,  dtype=np.uint32).tofile(str(fdir / "edge_index_rows.bin"))
    np.asarray(H_coo.col,  dtype=np.uint32).tofile(str(fdir / "edge_index_cols.bin"))
    np.asarray(H_coo.data, dtype=np.float32).tofile(str(fdir / "A_values.bin"))

    node_array             = np.zeros(N, dtype=NODE_DTYPE)
    node_array["position"] = positions.astype(np.float32)
    node_array["mass"]     = np.array([nd[3] for nd in nodes], dtype=np.float32)
    node_array["morton"]   = np.array([nd[2] for nd in nodes], dtype=np.uint32)
    node_array["active"]   = np.uint32(1)
    node_array.tofile(str(fdir / "nodes.bin"))

    with open(fdir / "meta.txt", "w") as f:
        f.write(f"numNodes: {N}\n")


# ── per-frame generation ───────────────────────────────────────────────────────

def generate_shell_frame(
    frame_idx: int,
    out_dir: Path,
    n_target: int,
    frame_rng: Any,
    *,
    shape: str,
    E_body: float,
    E_panel_min: float,
    E_panel_max: float,
    k_s: float,
    k_b: float,
    min_panels: int,
    max_panels: int,
    allowed_attachments: Optional[List[str]],
    allowed_orientations: Optional[List[str]],
    deform_amp: float,
    areal_density: float,
    cell_size: float,
    dt: float,
) -> None:
    grid_w, grid_h = _grid_dims(n_target)

    frame_shape = str(frame_rng.choice(_SHAPE_CHOICES)) if shape == "random" else shape
    params  = _sample_shape_params(frame_shape, frame_rng)
    wrap_u  = params["wrap_u"]
    wrap_v  = params["wrap_v"]

    # Material field in parameter space (same logic as multiphase v2 / cloth)
    nodes_full = _make_material_panels(
        grid_w, grid_h, frame_rng,
        E_body, E_panel_min, E_panel_max,
        min_panels, max_panels,
        allowed_attachments, allowed_orientations,
    )
    nodes_full.sort(key=lambda nd: nd[2])   # Morton order
    nodes = nodes_full[:n_target]

    coord_to_idx = {(nd[0], nd[1]): i for i, nd in enumerate(nodes)}
    u_arr = np.array([nd[0] for nd in nodes], dtype=np.float64)
    v_arr = np.array([nd[1] for nd in nodes], dtype=np.float64)

    base_pos = _base_positions(u_arr, v_arr, grid_w, grid_h, frame_shape, params)
    faces    = _build_faces_periodic(coord_to_idx, grid_w, grid_h, wrap_u, wrap_v)
    if len(faces) == 0:
        raise ValueError(f"No faces generated for frame {frame_idx} ({frame_shape})")

    positions = _deform_shell(
        base_pos, faces,
        u_norm=u_arr / grid_w, v_norm=v_arr / grid_h,
        frame_rng=frame_rng, deform_amp=deform_amp,
    )
    H = _build_shell_matrix(
        nodes, positions, faces,
        k_s=k_s, k_b=k_b,
        areal_density=areal_density, cell_size=cell_size, dt=dt,
    )
    _write_frame(frame_idx, out_dir, nodes, positions, H)


# ── difficulty preset ──────────────────────────────────────────────────────────

def _apply_difficulty(args: argparse.Namespace) -> None:
    """
    [0, 1] → physical parameters; same calibration as generate_cloth.py.
    difficulty=0: ~100–200 Jacobi iters (body fabric, no bending).
    difficulty=1: ~1500–3000 iters at N=4096 (rigid panel + full-seam).
    """
    d = float(args.difficulty)
    args.rho_heavy     = float(np.exp(np.log(5) + d * (np.log(100) - np.log(5))))
    args.rho_heavy_min = max(args.rho_light * 1.5, args.rho_heavy / 5.0)
    args.k_bend        = float(d * 0.5)
    if d > 0.4 and args.attachment_types is None:
        args.attachment_types = (
            ["button_gap", "full_seam"] if d > 0.7
            else ["free_top", "free_bottom", "button_gap", "full_seam"]
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate 3D thin-shell implicit-Euler matrices for LeafOnly training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-target",    type=int, default=4096)
    p.add_argument("--num-train",   type=int, default=200)
    p.add_argument("--num-test",    type=int, default=20)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--output-root", type=str, default="data")
    p.add_argument("--name",        type=str, default=None,
                   help="Dataset name; default shell_v1_<n-target>")

    p.add_argument("--shape",
                   choices=_SHAPE_CHOICES + ["random"], default="random",
                   help=(
                       "Surface shape per frame. 'random' samples uniformly from all four. "
                       "cylinder/torus: periodic edges create Morton-distant long-range coupling. "
                       "spherical_cap: polar concentration creates natural stiffness gradient. "
                       "saddle: negative Gaussian curvature, no periodicity."
                   ))

    # Material contrast (same axes as multiphase v2 and cloth)
    p.add_argument("--rho-light",     type=float, default=1.0,
                   help="Young's modulus of base shell material")
    p.add_argument("--rho-heavy-min", type=float, default=5.0,
                   help="Min Young's modulus of stiff panel regions (log-uniform per frame)")
    p.add_argument("--rho-heavy",     type=float, default=50.0,
                   help="Max Young's modulus of stiff panel regions")

    p.add_argument("--min-barriers",     type=int,  default=1)
    p.add_argument("--max-barriers",     type=int,  default=3)
    p.add_argument("--attachment-types", nargs="+", default=None,
                   help="Allowed attachment patterns (default: all four)")
    p.add_argument("--orientations",     nargs="+", default=None)

    # Shell physics
    p.add_argument("--k-bend",     type=float, default=0.1,
                   help="Bending stiffness k_b (Kirchhoff–Love k_b·L_E²).")
    p.add_argument("--deform-amp", type=float, default=0.05,
                   help="Normal-deformation amplitude as fraction of bounding-box diagonal. "
                        "Modulates cotangent weights per frame without changing connectivity.")

    # Implicit Euler (determine M/dt² diagonal)
    p.add_argument("--areal-density", type=float, default=0.3,
                   help="Shell areal density ρ (kg/m²).")
    p.add_argument("--cell-size",     type=float, default=0.01,
                   help="Reference grid cell size s (m) for mass scaling.")
    p.add_argument("--dt",            type=float, default=1/30,
                   help="Implicit Euler timestep dt (s). Default 1/30 s (30 fps).")

    p.add_argument("--difficulty", type=float, default=None,
                   help="[0,1] preset overriding --rho-heavy, --k-bend, --attachment-types.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.difficulty is not None:
        _apply_difficulty(args)

    if _sparse is None:
        raise SystemExit("scipy not found. Install with: pip install scipy")

    name      = args.name or f"shell_v1_{args.n_target}"
    out_root  = Path(args.output_root)
    train_dir = out_root / name / "train"
    test_dir  = out_root / name / "test"
    rng       = _rng(args.seed)

    grid_w, grid_h = _grid_dims(args.n_target)
    print(f"3D thin-shell dataset: {name}")
    print(f"  grid {grid_w}×{grid_h}  n_target={args.n_target}  shape={args.shape}")
    print(f"  Physics: H = M/dt² + k_s·L_E + k_b·L_E²")
    print(f"    k_s=1.0  k_bend={args.k_bend:.3f}  deform_amp={args.deform_amp:.3f}")
    print(f"    ρ={args.areal_density:.2f} kg/m²  cell={args.cell_size*100:.1f} cm  dt={args.dt*1000:.1f} ms")
    print(f"  Material: E_body={args.rho_light}  E_panel∈[{args.rho_heavy_min:.1f},{args.rho_heavy:.1f}]")
    print(f"  Panels: {args.min_barriers}–{args.max_barriers}  "
          f"attachment={args.attachment_types or _ATTACHMENT_CHOICES}")

    for split, out_dir, n_frames in [
        ("train", train_dir, args.num_train),
        ("test",  test_dir,  args.num_test),
    ]:
        print(f"\n{split}: {n_frames} frames → {out_dir}")
        for i in range(n_frames):
            frame_rng = _rng(int(rng.integers(0, 2**31)))
            generate_shell_frame(
                i, out_dir, args.n_target, frame_rng,
                shape=args.shape,
                E_body=args.rho_light,
                E_panel_min=args.rho_heavy_min,
                E_panel_max=args.rho_heavy,
                k_s=1.0,
                k_b=args.k_bend,
                min_panels=args.min_barriers,
                max_panels=args.max_barriers,
                allowed_attachments=args.attachment_types,
                allowed_orientations=args.orientations,
                deform_amp=args.deform_amp,
                areal_density=args.areal_density,
                cell_size=args.cell_size,
                dt=args.dt,
            )
            if (i + 1) % 10 == 0 or i == n_frames - 1:
                print(f"  {i+1}/{n_frames}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
