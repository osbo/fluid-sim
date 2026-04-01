"""
Interactive H-Matrix Inverse Sensitivity Visualizer
Run with: python3 InspectInverse.py
"""
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from leafonly.data import FluidGraphDataset
from leafonly.config import HMATRIX_ETA, LEAF_SIZE, problem_padded_num_nodes
from leafonly.hmatrix import standard_admissible_unique_blocks
from leafonly.checkpoint import read_leaf_only_header


def main():
    print("Loading data and computing A^-1...")

    leaf_only_weights_path = _script_dir / "leaf_only_weights.bytes"
    if leaf_only_weights_path.exists():
        header_info = read_leaf_only_header(leaf_only_weights_path)
        leaf_L = int(header_info[1])
    else:
        leaf_L = LEAF_SIZE

    data_folder = _script_dir.parent / "StreamingAssets" / "TestData"
    dataset = FluidGraphDataset([Path(data_folder)])
    if len(dataset) == 0:
        raise SystemExit("No frames found in TestData.")
        
    frame_idx = min(600, len(dataset) - 1)
    batch = dataset[frame_idx]

    num_nodes_real = int(batch['num_nodes'])
    viz_n = problem_padded_num_nodes(num_nodes_real)
    print(f"Physical system {viz_n}x{viz_n} (leaf_size={leaf_L})")

    ei, ev = batch["edge_index"], batch["edge_values"]
    em = (ei[0] < viz_n) & (ei[1] < viz_n)

    A_small = torch.sparse_coo_tensor(ei[:, em], ev[em], (viz_n, viz_n)).coalesce().to_dense().numpy()
    A_phys_f64 = A_small.astype(np.float64, copy=False)
    A_inv = np.linalg.inv(A_phys_f64)

    print("Computing eigendecomposition of A...")
    vals, vecs = np.linalg.eigh(A_phys_f64)

    num_units = viz_n // leaf_L
    u = standard_admissible_unique_blocks(num_units, float(HMATRIX_ETA), torch.device("cpu"), dtype=torch.float32)
    blocks = [row for row in u.numpy() if row[0] <= row[1]]

    print(f"Profiling {len(blocks)} unique parameter blocks...")
    block_info = []
    
    for row in blocks:
        r_leaf, c_leaf, size_leaves = int(row[0]), int(row[1]), int(row[2])
        r0, c0 = r_leaf * leaf_L, c_leaf * leaf_L
        size_px = size_leaves * leaf_L
        r1, c1 = r0 + size_px, c0 + size_px
        on_diag_unit = (r_leaf == c_leaf) and (size_leaves == 1)
        
        norm_R = np.linalg.norm(vecs[r0:r1, :], axis=0)
        norm_C = np.linalg.norm(vecs[c0:c1, :], axis=0)
        W = norm_R * norm_C
        avg_lam = np.sum(vals * W) / (np.sum(W) + 1e-12)
        
        if r0 == c0:
            B = A_inv[r0:r1, c0:c1]
            A_sub = A_phys_f64[r0:r1, c0:c1]
            EA = B @ A_sub
        else:
            B = A_inv[r0:r1, c0:c1]
            Lr, Lc = r1 - r0, c1 - c0
            E_UU = np.zeros((Lr + Lc, Lr + Lc), dtype=np.float64)
            E_UU[0:Lr, Lr:Lr+Lc] = B
            E_UU[Lr:Lr+Lc, 0:Lr] = B.T
            A_UU = np.zeros((Lr + Lc, Lr + Lc), dtype=np.float64)
            A_UU[0:Lr, 0:Lr] = A_phys_f64[r0:r1, r0:r1]
            A_UU[0:Lr, Lr:Lr+Lc] = A_phys_f64[r0:r1, c0:c1]
            A_UU[Lr:Lr+Lc, 0:Lr] = A_phys_f64[c0:c1, r0:r1]
            A_UU[Lr:Lr+Lc, Lr:Lr+Lc] = A_phys_f64[c0:c1, c0:c1]
            EA = E_UU @ A_UU
            
        evs = np.linalg.eigvals(EA)
        impact_below = max(0.0, float(np.max(evs.real)))
        impact_above = max(0.0, float(-np.min(evs.real)))
        
        block_info.append({
            'r0': r0, 'r1': r1,
            'c0': c0, 'c1': c1,
            'size_px': size_px,
            'size_leaves': size_leaves,
            'on_diag_unit': on_diag_unit,
            'avg_lam': avg_lam,
            'impact_above': impact_above,
            'impact_below': impact_below
        })

    lambdas = [b['avg_lam'] for b in block_info]
    impacts_above = [b['impact_above'] for b in block_info]
    impacts_below = [b['impact_below'] for b in block_info]
    max_above = np.percentile(impacts_above, 98) + 1e-9
    max_below = np.percentile(impacts_below, 98) + 1e-9
    median_above, median_below = float(np.median(impacts_above)), float(np.median(impacts_below))

    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    fig.canvas.manager.set_window_title("H-Matrix Architecture & Sensitivity Profiler")
    gs = fig.add_gridspec(2, 3)

    ax_inv = fig.add_subplot(gs[0, 0])
    ax_sens = fig.add_subplot(gs[0, 1])
    ax_hw_sens = fig.add_subplot(gs[0, 2])

    ax_eig = fig.add_subplot(gs[1, 0])
    ax_cond = fig.add_subplot(gs[1, 1])
    ax_bar = fig.add_subplot(gs[1, 2])

    # --- Row 0: Matrices ---
    log_ainv = np.log10(np.abs(A_inv) + 1e-9)
    im1 = ax_inv.imshow(log_ainv, cmap='magma', aspect='equal')
    ax_inv.set_title(f"True Inverse A^{{-1}} (log10)\n<-- Click any block!")
    fig.colorbar(im1, ax=ax_inv, fraction=0.046, pad=0.04)

    im_sens = ax_sens.imshow(np.zeros_like(A_inv), cmap='magma', aspect='equal')
    ax_sens.set_title("Sensitivity (S) of Clicked Block\n(Symmetric Application)")
    fig.colorbar(im_sens, ax=ax_sens, fraction=0.046, pad=0.04)

    im_hw = ax_hw_sens.imshow(np.zeros_like(A_inv), cmap='magma', aspect='equal')
    ax_hw_sens.set_title("1-Step Highway Receptive Field\n(Sensitivity available after 1 smear)")
    fig.colorbar(im_hw, ax=ax_hw_sens, fraction=0.046, pad=0.04)

    # --- Row 1: Metrics & Coverage ---
    norm_lam = plt.Normalize(vmin=min(lambdas), vmax=max(lambdas))
    cmap_lam = plt.get_cmap("turbo")
    ax_eig.set_title("Blocks by Targeted Eigenvalue\n(Spatially weighted average $\\lambda$ of A)")
    sm_lam = plt.cm.ScalarMappable(cmap=cmap_lam, norm=norm_lam)
    sm_lam.set_array([])
    fig.colorbar(sm_lam, ax=ax_eig, fraction=0.046, pad=0.04, label="Average $\\lambda$ (higher = high frequency)")

    cmap_above, cmap_below = plt.get_cmap("Reds"), plt.get_cmap("Blues")
    ax_cond.set_title("Block Impact on Condition Number\n(Red: fixes small $\\lambda_A$, Blue: fixes large $\\lambda_A$)")
    sm_above = plt.cm.ScalarMappable(cmap=cmap_above, norm=plt.Normalize(vmin=0, vmax=max_above))
    sm_above.set_array([])
    cbar_above = fig.colorbar(sm_above, ax=ax_cond, orientation='horizontal', fraction=0.04, pad=0.04)
    sm_below = plt.cm.ScalarMappable(cmap=cmap_below, norm=plt.Normalize(vmin=0, vmax=max_below))
    sm_below.set_array([])
    cbar_below = fig.colorbar(sm_below, ax=ax_cond, orientation='horizontal', fraction=0.04, pad=0.08)

    for cb, med in ((cbar_above, median_above), (cbar_below, median_below)):
        cb.ax.axvline(med, color="#555", linestyle="--", linewidth=1.25, zorder=5, clip_on=False)

    ax_bar.set_title("Global Sensitivity Energy (S²) Captured")
    ax_bar.set_ylabel("% of Clicked Block's Total Reach")
    ax_bar.set_ylim(0, 105)
    bars = ax_bar.bar(["Base Block\n(symmetric)", "1-Step Highway", "2-Step Highway", "Global\n(Dense)"], [0, 0, 0, 100], color=['#e8a0c8', 'cyan', 'royalblue', 'gray'])
    bar_texts = [ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, "", ha='center', va='bottom', fontsize=11, fontweight='bold') for bar in bars]

    rects_inv, rects_eig, rects_cond = [], [], []
    green_cmap = plt.get_cmap("Greens")
    max_log = int(np.log2(num_units)) if num_units > 1 else 0

    for i, b in enumerate(block_info):
        x, y, size_px = b['c0'], b['r0'], b['size_px']
        if b['on_diag_unit']:
            face1, edge1, lw1 = "#e8a0c8", "white", 1.0
        else:
            k = int(np.log2(b['size_leaves'])) if b['size_leaves'] > 1 else 0
            face1 = green_cmap(0.35 + 0.55 * (k / max(max_log, 1)))
            edge1, lw1 = "#1a3d1a", 1.2
            
        b['orig_edge'], b['orig_lw'] = edge1, lw1
        rect1 = Rectangle((x - 0.5, y - 0.5), size_px, size_px, linewidth=lw1, edgecolor=edge1, facecolor=face1, alpha=0.45, picker=False)
        ax_inv.add_patch(rect1)
        rects_inv.append(rect1)
        
        face3 = cmap_lam(norm_lam(b['avg_lam']))
        rect3 = Rectangle((x - 0.5, y - 0.5), size_px, size_px, linewidth=1.0, edgecolor="#111", facecolor=face3, alpha=0.9, picker=False)
        ax_eig.add_patch(rect3)
        rects_eig.append(rect3)
        
        val_above, val_below = min(b['impact_above'], max_above), min(b['impact_below'], max_below)
        face4 = cmap_above(val_above / max_above) if b['impact_above'] >= b['impact_below'] else cmap_below(val_below / max_below)
        rect4 = Rectangle((x - 0.5, y - 0.5), size_px, size_px, linewidth=1.0, edgecolor="#111", facecolor=face4, alpha=0.9, zorder=1, picker=False)
        ax_cond.add_patch(rect4)
        rects_cond.append(rect4)

    for ax in (ax_inv, ax_eig, ax_cond, ax_sens, ax_hw_sens):
        ax.set_xlim(-0.5, viz_n - 0.5)
        ax.set_ylim(viz_n - 0.5, -0.5)
        ax.set_aspect('equal')

    state = {'selected_idx': None, 'cbar_marker_above': None, 'cbar_marker_below': None}

    def onclick(event):
        if event.inaxes not in (ax_inv, ax_eig, ax_cond): return
        x, y = event.xdata, event.ydata
        if x is None or y is None: return

        clicked_idx = None
        for i, b in enumerate(block_info):
            if b['c0'] - 0.5 <= x <= b['c1'] - 0.5 and b['r0'] - 0.5 <= y <= b['r1'] - 0.5:
                clicked_idx = i
                break

        if clicked_idx is None: return

        for key in ("cbar_marker_above", "cbar_marker_below"):
            m = state.get(key)
            if m is not None:
                m.remove()
                state[key] = None

        if state['selected_idx'] is not None:
            old_idx = state['selected_idx']
            old_info = block_info[old_idx]
            for r_list in (rects_inv, rects_eig, rects_cond):
                r = r_list[old_idx]
                r.set_edgecolor(old_info['orig_edge'] if r_list == rects_inv else '#111')
                r.set_linewidth(old_info['orig_lw'] if r_list == rects_inv else 1.0)
                r.set_zorder(1)

        for r_list in (rects_inv, rects_eig, rects_cond):
            r_list[clicked_idx].set_edgecolor('cyan')
            r_list[clicked_idx].set_linewidth(3.5)
            r_list[clicked_idx].set_zorder(10)
            
        state['selected_idx'] = clicked_idx
        b = block_info[clicked_idx]

        state["cbar_marker_above"] = cbar_above.ax.axvline(b["impact_above"], color="yellow", linewidth=2.8, zorder=6, clip_on=False)
        state["cbar_marker_below"] = cbar_below.ax.axvline(b["impact_below"], color="yellow", linewidth=2.8, zorder=6, clip_on=False)
        
        R, C = slice(b['r0'], b['r1']), slice(b['c0'], b['c1'])
        u = np.sqrt(np.sum(A_inv[R, :]**2, axis=0))
        v = np.sqrt(np.sum(A_inv[:, C]**2, axis=1))
        S = np.outer(u, v)

        # Compute symmetric sensitivity for display
        S_sym = np.zeros_like(S)
        S_sym[R, C] = S[R, C]
        S_sym[C, R] = S[R, C].T

        log_S = np.log10(S_sym + 1e-12)
        im_sens.set_data(log_S)
        im_sens.set_clim(vmin=np.min(log_S), vmax=np.max(log_S))
        ax_sens.set_title(f"Sensitivity (S) of Clicked Block (Symmetric)\nTargeting $\\lambda \\approx {b['avg_lam']:.2f}$")

        R0_set = set(range(b['r0'] // leaf_L, b['r1'] // leaf_L))
        C0_set = set(range(b['c0'] // leaf_L, b['c1'] // leaf_L))
        H0_set = R0_set.union(C0_set)

        step1_blocks = []
        H1_set = set(H0_set)
        for bi in block_info:
            Ri_set = set(range(bi['r0'] // leaf_L, bi['r1'] // leaf_L))
            Ci_set = set(range(bi['c0'] // leaf_L, bi['c1'] // leaf_L))
            if Ri_set.intersection(H0_set) or Ci_set.intersection(H0_set):
                step1_blocks.append(bi)
                H1_set.update(Ri_set)
                H1_set.update(Ci_set)
                
        step2_blocks = []
        for bi in block_info:
            Ri_set = set(range(bi['r0'] // leaf_L, bi['r1'] // leaf_L))
            Ci_set = set(range(bi['c0'] // leaf_L, bi['c1'] // leaf_L))
            if Ri_set.intersection(H1_set) or Ci_set.intersection(H1_set):
                step2_blocks.append(bi)

        e_total = np.sum(S**2)

        # Symmetric masks: only tiles in the architecture reach (no blanket diagonal)
        mask_blk = np.zeros((viz_n, viz_n), dtype=bool)
        mask_blk[b["r0"]:b["r1"], b["c0"]:b["c1"]] = True
        mask_blk[b["c0"]:b["c1"], b["r0"]:b["r1"]] = True
        e_base = float(np.sum(S[mask_blk] ** 2))

        mask_hw1 = np.zeros((viz_n, viz_n), dtype=bool)
        for bi in step1_blocks:
            mask_hw1[bi["r0"] : bi["r1"], bi["c0"] : bi["c1"]] = True
            mask_hw1[bi["c0"] : bi["c1"], bi["r0"] : bi["r1"]] = True
        e_step1 = float(np.sum(S[mask_hw1] ** 2))

        mask_hw2 = np.zeros((viz_n, viz_n), dtype=bool)
        for bi in step2_blocks:
            mask_hw2[bi["r0"] : bi["r1"], bi["c0"] : bi["c1"]] = True
            mask_hw2[bi["c0"] : bi["c1"], bi["r0"] : bi["r1"]] = True
        e_step2 = float(np.sum(S[mask_hw2] ** 2))

        S_hw = np.where(mask_hw1, S, 1e-12)
        log_hw = np.log10(S_hw)
        im_hw.set_data(log_hw)
        if np.any(mask_hw1):
            im_hw.set_clim(
                float(np.percentile(log_hw[mask_hw1], 1)),
                float(np.percentile(log_hw[mask_hw1], 99)),
            )
        else:
            im_hw.set_clim(-12.0, 0.0)

        pcts = [e_base / e_total * 100, e_step1 / e_total * 100, e_step2 / e_total * 100, 100.0]
        for bar, txt, pct in zip(bars, bar_texts, pcts):
            bar.set_height(pct)
            txt.set_y(pct + 2)
            txt.set_text(f"{pct:.1f}%")

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', onclick)
    print("Interactive window ready.")
    plt.show()

if __name__ == "__main__":
    main()