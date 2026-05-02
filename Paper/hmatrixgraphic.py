"""
Schematic figure for weak-admissibility H-matrix tiling (not HODLR).

Uses the same dyadic admissibility rule as ``Assets/Scripts/leafonly/hmatrix.py``
(``standard_admissible_unique_blocks``). Constants below are only for the drawing.

Toggle ``SHOW_HIGHWAYS`` (or run with ``--bare``) to draw the partition without the red bands.
Highway band (red): mimics ``LeafOnlyNet._accumulate_highway_from_blocks`` / prolongation
(``HM_PROLONG_ROW_LEAF_IDX``, ``HM_PROLONG_COL_LEAF_IDX`` in ``hmatrix.py``): one chosen
off-diagonal tile ``(r0, c0, S)`` scatters into all row leaves ``r0..r0+S-1`` and column
leaves ``c0..c0+S-1``. We show the union of those **row-strip** and **column-strip** regions
restricted to the **upper triangle including the diagonal** (leaf row index ``<=`` leaf column
index), i.e. on and above the main diagonal in the leaf grid.
"""

from __future__ import annotations

import argparse

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- Illustration parameters (any power-of-two leaf count; plot scale is arbitrary) ---
NUM_LEAVES = 64
PIX_PER_LEAF = 20
ETA = 1.0

N_DRAW = NUM_LEAVES * PIX_PER_LEAF

# --- Highway demo (row/col prolong strips, upper triangle including diagonal) ---
# Set to ``False`` for partition-only figure (no red bands). CLI ``--bare`` also turns highways off.
SHOW_HIGHWAYS = True
# Off-diagonal block size ``S``: ``0`` = largest, ``1`` = second-largest, ``2`` = one dyadic
# step smaller (half the linear extent), etc.
HIGHWAY_S_RANK = 2
# Red overlay on top of tile blue (R, G, B, alpha); a bit punchier than the very muted version.
HIGHWAY_FACE_RGBA = (0.92, 0.22, 0.24, 0.48)
HIGHWAY_EDGE = (0.58, 0.14, 0.16, 0.62)

# --- Palette ---
BG_COLOR = "#F4F5F7"
LEAF_SLATE = "#3D4754"
DIAG_FILL = LEAF_SLATE
DIAG_EDGE = "#2A323C"
OFF_OUTLINE = "#7A8494"
OFF_OUTLINE_WIDTH = 0.9
OFF_BG_HUE = 0.58
OFF_BG_VALUE = 0.99
OFF_BG_SAT_MAX = 1.0 / 3.0
OFF_BG_SAT_LARGE = 0.07
STRIP_THICKNESS = PIX_PER_LEAF * 0.5
DISPLAY_INSET_PX = 5.0
STRIP_FACE = (0.24, 0.28, 0.33)
STRIP_ALPHA = 0.55
STRIP_EDGE = (0.18, 0.22, 0.26, 0.4)
STRIP_EDGE_WIDTH = 0.2
STRIP_SHADOW_OFFSET = (0.9, -0.9)
STRIP_SHADOW_ALPHA = 0.32

FIG_SIZE = (10, 10)
SPACING = 0.002


def standard_admissible_unique_blocks(num_units: int, eta: float) -> np.ndarray:
    nu = int(num_units)
    if nu <= 0:
        return np.zeros((0, 3), dtype=np.int64)
    r = np.arange(nu, dtype=np.int64)[:, None]
    c = np.arange(nu, dtype=np.int64)[None, :]
    s_max = np.ones((nu, nu), dtype=np.int64)
    max_level = (nu.bit_length() - 1) if nu > 1 else 0
    eta_f = float(eta)
    for lv in range(1, max_level + 1):
        s = 1 << lv
        r0 = (r // s) * s
        c0 = (c // s) * s
        d = np.abs(r0 - c0).astype(np.float64) - float(s)
        admissible = (d > 0) & (float(s) <= eta_f * d)
        s_max = np.where(admissible, s, s_max)
    final_r0 = (r // s_max) * s_max
    final_c0 = (c // s_max) * s_max
    flat = np.stack([final_r0.ravel(), final_c0.ravel(), s_max.ravel()], axis=-1)
    return np.unique(flat, axis=0)


def _is_dense_on_diag_leaf(r0: int, c0: int, s: int) -> bool:
    return r0 == c0 and s == 1


def pick_highway_seed_block(tiles: np.ndarray, num_leaves: int) -> tuple[int, int, int] | None:
    """
    Pick one off-diagonal tile ``(r0, c0, S)`` for the highway illustration.

    ``HIGHWAY_S_RANK`` picks the distinct off-diagonal size ``S`` (largest first).  Among
    upper-triangular tiles ``(r0 < c0)`` with that ``S``, choose the one whose block center
    ``(r0+S/2, c0+S/2)`` is closest to the middle of the ``K``-by-``K`` leaf grid (``K/2``),
    so the row/column bands sit nearer the figure center than the top-left chain.
    """
    off: list[tuple[int, int, int]] = []
    for row in tiles:
        r0, c0, s = int(row[0]), int(row[1]), int(row[2])
        if _is_dense_on_diag_leaf(r0, c0, s):
            continue
        off.append((r0, c0, s))
    if not off:
        return None
    sizes = sorted({s for _, _, s in off}, reverse=True)
    rank = min(HIGHWAY_S_RANK, len(sizes) - 1)
    s_pick = sizes[rank]
    cand = [(r0, c0, s) for (r0, c0, s) in off if s == s_pick and r0 < c0]
    if not cand:
        return None
    k = float(num_leaves)
    ctr = 0.5 * k

    def center_dist2(t: tuple[int, int, int]) -> float:
        r0, c0, s = t
        mr = r0 + 0.5 * float(s)
        mc = c0 + 0.5 * float(s)
        return (mr - ctr) ** 2 + (mc - ctr) ** 2

    cand.sort(key=lambda t: (center_dist2(t), t[0], t[1]))
    return cand[3]


def highway_upper_tri_leaf_cells(r0b: int, c0b: int, sb: int, num_leaves: int) -> set[tuple[int, int]]:
    """
    Leaf pairs ``(R, C)`` with ``R <= C`` (upper triangle including diagonal) touched by the
    row- and column-strip prolongation index sets for tile ``(r0b, c0b, sb)``.
    """
    k = int(num_leaves)
    cells: set[tuple[int, int]] = set()
    for r in range(r0b, r0b + sb):
        for c in range(r, k):
            cells.add((r, c))
    for c in range(c0b, c0b + sb):
        for r in range(0, c + 1):
            cells.add((r, c))
    return cells


def _off_diag_s_bounds(tiles: np.ndarray) -> tuple[int, int]:
    s_vals: list[int] = []
    for r0, c0, s in tiles:
        r0, c0, s = int(r0), int(c0), int(s)
        if not _is_dense_on_diag_leaf(r0, c0, s):
            s_vals.append(s)
    if not s_vals:
        return 1, 1
    return min(s_vals), max(s_vals)


def off_block_fill_rgba(s: int, s_min: int, s_max: int) -> tuple[float, float, float, float]:
    a = float(s * s)
    a0 = float(s_min * s_min)
    a1 = float(s_max * s_max)
    if a1 <= a0:
        t = 0.0
    else:
        t = (np.log(a) - np.log(a0)) / (np.log(a1) - np.log(a0))
    t = float(np.clip(t, 0.0, 1.0))
    sat = OFF_BG_SAT_MAX + (OFF_BG_SAT_LARGE - OFF_BG_SAT_MAX) * t
    rgb = mcolors.hsv_to_rgb([OFF_BG_HUE, sat, OFF_BG_VALUE])
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)


def _display_px_to_data_d(ax, n_px: float) -> float:
    x0, x1 = ax.get_xlim()
    px0 = ax.transData.transform((x0, 0.0))
    px1 = ax.transData.transform((x1, 0.0))
    disp_w = abs(px1[0] - px0[0])
    dx = abs(n_px * (x1 - x0) / disp_w) if disp_w > 1e-9 else 0.0

    y0, y1 = ax.get_ylim()
    py0 = ax.transData.transform((0.0, y0))
    py1 = ax.transData.transform((0.0, y1))
    disp_h = abs(py1[1] - py0[1])
    dy = abs(n_px * (y1 - y0) / disp_h) if disp_h > 1e-9 else 0.0

    return float(max(dx, dy))


def _tile_inner_bounds(r0: int, c0: int, s: int, pix_per_leaf: float) -> tuple[float, float, float, float]:
    px = float(c0 * pix_per_leaf)
    py = float(r0 * pix_per_leaf)
    w = h = float(s * pix_per_leaf)
    return px + SPACING, py + SPACING, w - 2 * SPACING, h - 2 * SPACING


def _leaf_cell_inner_bounds(r: int, c: int, pix_per_leaf: float) -> tuple[float, float, float, float]:
    """Same spacing convention as partition tiles, for a single leaf cell."""
    return _tile_inner_bounds(r, c, 1, pix_per_leaf)


def _add_strip_shadow(patch: patches.Patch) -> None:
    patch.set_path_effects(
        [
            pe.SimplePatchShadow(
                offset=STRIP_SHADOW_OFFSET,
                shadow_rgbFace="#1e2329",
                alpha=STRIP_SHADOW_ALPHA,
            ),
            pe.Normal(),
        ]
    )


def _draw_leaf_cell_background(
    ax,
    r: int,
    c: int,
    pix_per_leaf: float,
    *,
    facecolor,
    linewidth: float = 0.0,
    edgecolor: str | tuple | None = None,
    zorder: float = 2.0,
) -> None:
    lx, ly, lw, lh = _leaf_cell_inner_bounds(r, c, pix_per_leaf)
    ax.add_patch(
        patches.Rectangle(
            (lx, ly),
            lw,
            lh,
            linewidth=linewidth,
            edgecolor=edgecolor if linewidth > 0 else "none",
            facecolor=facecolor,
            zorder=zorder,
        )
    )


def _draw_admissible_tile_leaf_backgrounds(
    ax,
    r0: int,
    c0: int,
    s: int,
    pix_per_leaf: float,
    fill_rgba: tuple[float, float, float, float],
    highway_cells: set[tuple[int, int]],
) -> None:
    """Per-leaf fill: highway cells get tile blue then a translucent red overlay (softer hue)."""
    for r in range(r0, r0 + s):
        for c in range(c0, c0 + s):
            _draw_leaf_cell_background(ax, r, c, pix_per_leaf, facecolor=fill_rgba, zorder=2.0)
            if (r, c) in highway_cells:
                _draw_leaf_cell_background(
                    ax, r, c, pix_per_leaf, facecolor=HIGHWAY_FACE_RGBA, zorder=2.25
                )


def draw_dense_leaf(ax, r0: int, c0: int, s: int, pix_per_leaf: float, highway_cells: set[tuple[int, int]]) -> None:
    x0, y0, bw, bh = _tile_inner_bounds(r0, c0, s, pix_per_leaf)
    on_hw = (r0, c0) in highway_cells
    rect = patches.Rectangle(
        (x0, y0),
        bw,
        bh,
        linewidth=0.5,
        edgecolor=HIGHWAY_EDGE if on_hw else DIAG_EDGE,
        facecolor=DIAG_FILL,
        zorder=6,
    )
    ax.add_patch(rect)
    if on_hw:
        ax.add_patch(
            patches.Rectangle(
                (x0, y0),
                bw,
                bh,
                linewidth=0,
                edgecolor="none",
                facecolor=HIGHWAY_FACE_RGBA,
                zorder=6.2,
            )
        )


def draw_admissible_tile(
    ax,
    r0: int,
    c0: int,
    s: int,
    pix_per_leaf: float,
    s_min: int,
    s_max: int,
    highway_cells: set[tuple[int, int]],
) -> None:
    x0, y0, bw, bh = _tile_inner_bounds(r0, c0, s, pix_per_leaf)
    d_raw = _display_px_to_data_d(ax, DISPLAY_INSET_PX)
    t = min(STRIP_THICKNESS, 0.42 * min(bw, bh))
    d_max = max(0.0, min((bw - t), (bh - t)) / 2.0 - 1e-6)
    d = min(d_raw, d_max) if d_max > 0.0 else 0.0

    fill_rgba = off_block_fill_rgba(s, s_min, s_max)
    _draw_admissible_tile_leaf_backgrounds(ax, r0, c0, s, pix_per_leaf, fill_rgba, highway_cells)

    ix, iy = x0 + d, y0 + d
    vh = max(0.0, bh - 2.0 * d)
    hw = max(0.0, bw - 2.0 * d)
    vert = patches.Rectangle(
        (ix, iy),
        t,
        vh,
        linewidth=STRIP_EDGE_WIDTH,
        edgecolor=STRIP_EDGE,
        facecolor=(*STRIP_FACE, STRIP_ALPHA),
        zorder=3,
    )
    ax.add_patch(vert)
    _add_strip_shadow(vert)

    horiz = patches.Rectangle(
        (ix, iy),
        hw,
        t,
        linewidth=STRIP_EDGE_WIDTH,
        edgecolor=STRIP_EDGE,
        facecolor=(*STRIP_FACE, STRIP_ALPHA),
        zorder=4,
    )
    ax.add_patch(horiz)
    _add_strip_shadow(horiz)

    outline = patches.Rectangle(
        (x0, y0),
        bw,
        bh,
        linewidth=OFF_OUTLINE_WIDTH,
        edgecolor=OFF_OUTLINE,
        facecolor="none",
        zorder=5,
    )
    ax.add_patch(outline)


def visualize_hmatrix(ax, *, show_highways: bool | None = None) -> None:
    if show_highways is None:
        show_highways = SHOW_HIGHWAYS
    tiles = standard_admissible_unique_blocks(NUM_LEAVES, ETA)
    s_min, s_max = _off_diag_s_bounds(tiles)
    highway_cells: set[tuple[int, int]] = set()
    if show_highways:
        seed = pick_highway_seed_block(tiles, NUM_LEAVES)
        if seed is not None:
            rr, cc, ss = seed
            highway_cells = highway_upper_tri_leaf_cells(rr, cc, ss, NUM_LEAVES)

    order = np.argsort(-tiles[:, 2].astype(np.int64))
    tiles = tiles[order]
    for r0, c0, s in tiles:
        r0, c0, s = int(r0), int(c0), int(s)
        if _is_dense_on_diag_leaf(r0, c0, s):
            draw_dense_leaf(ax, r0, c0, s, PIX_PER_LEAF, highway_cells)
        else:
            draw_admissible_tile(ax, r0, c0, s, PIX_PER_LEAF, s_min, s_max, highway_cells)


def main() -> None:
    parser = argparse.ArgumentParser(description="Schematic weak-admissibility H-matrix partition.")
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="Write PNG to PATH instead of opening an interactive window.",
    )
    parser.add_argument(
        "--bare",
        action="store_true",
        help="Partition only: omit highway overlay (overrides SHOW_HIGHWAYS).",
    )
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.set_xlim(0, N_DRAW)
    ax.set_ylim(N_DRAW, 0)
    ax.axis("off")

    visualize_hmatrix(ax, show_highways=SHOW_HIGHWAYS and (not args.bare))

    plt.subplots_adjust(top=0.95, bottom=0.05)

    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor=BG_COLOR)
        plt.close(fig)
        return

    plt.show()


if __name__ == "__main__":
    main()
