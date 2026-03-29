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
from leafonly.config import HMATRIX_ETA, LEAF_SIZE, MAX_MIXED_SIZE
from leafonly.hmatrix import standard_admissible_unique_blocks
from leafonly.checkpoint import read_leaf_only_header

def main():
    print("Loading data and computing A^-1...")
    
    # 1. Resolve parameters matching InspectModel
    leaf_only_weights_path = _script_dir / "leaf_only_weights.bytes"
    if leaf_only_weights_path.exists():
        header_info = read_leaf_only_header(leaf_only_weights_path)
        leaf_L = int(header_info[1])
    else:
        leaf_L = LEAF_SIZE

    # 2. Load the dataset
    data_folder = _script_dir.parent / "StreamingAssets" / "TestData"
    dataset = FluidGraphDataset([Path(data_folder)])
    if len(dataset) == 0:
        raise SystemExit("No frames found in TestData.")
        
    frame_idx = min(600, len(dataset) - 1)
    batch = dataset[frame_idx]

    num_nodes_real = int(batch['num_nodes'])
    viz_n = (min(MAX_MIXED_SIZE, num_nodes_real) // leaf_L) * leaf_L
    print(f"Physical system {viz_n}x{viz_n} (leaf_size={leaf_L})")

    ei, ev = batch["edge_index"], batch["edge_values"]
    em = (ei[0] < viz_n) & (ei[1] < viz_n)

    A_small = torch.sparse_coo_tensor(ei[:, em], ev[em], (viz_n, viz_n)).coalesce().to_dense().numpy()
    A_phys_f64 = A_small.astype(np.float64, copy=False)

    A_inv = np.linalg.inv(A_phys_f64)

    print("Computing eigendecomposition of A to profile block frequency targeting...")
    vals, vecs = np.linalg.eigh(A_phys_f64)

    # 3. Generate the exact weak-admissibility H-matrix blocks
    num_units = viz_n // leaf_L
    u = standard_admissible_unique_blocks(num_units, float(HMATRIX_ETA), torch.device("cpu"), dtype=torch.float32)
    blocks = u.numpy()

    print("Computing condition number impact per block (Eigenvalues of EA)...")
    block_info = []
    
    for row in blocks:
        r_leaf, c_leaf, size_leaves = int(row[0]), int(row[1]), int(row[2])
        r0, c0 = r_leaf * leaf_L, c_leaf * leaf_L
        size_px = size_leaves * leaf_L
        r1, c1 = r0 + size_px, c0 + size_px
        on_diag_unit = (r_leaf == c_leaf) and (size_leaves == 1)
        
        # Metric 1: Targeted Eigenvalues (Spatial weighting)
        norm_R = np.linalg.norm(vecs[r0:r1, :], axis=0)
        norm_C = np.linalg.norm(vecs[c0:c1, :], axis=0)
        W = norm_R * norm_C
        avg_lam = np.sum(vals * W) / (np.sum(W) + 1e-12)
        
        # Metric 2: Exact Condition Number Impact (Matrix Perturbation Theory)
        # If we omit block B, the preconditioned system becomes (A^-1 - E)A = I - EA.
        # The eigenvalues shift exactly by the eigenvalues of EA.
        if r0 == c0:
            # Diagonal block
            B = A_inv[r0:r1, c0:c1]
            A_sub = A_phys_f64[r0:r1, c0:c1]
            EA = B @ A_sub
        else:
            # Off-diagonal block (symmetric perturbation involves R union C)
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
        
        # Max positive eval of EA pulls the preconditioned cluster BELOW 1
        impact_below = max(0.0, float(np.max(evs.real)))
        # Min negative eval of EA pushes the preconditioned cluster ABOVE 1
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

    # Normalize metrics (using 98th percentile to avoid single-block washouts)
    lambdas = [b['avg_lam'] for b in block_info]
    impacts_above = [b['impact_above'] for b in block_info]
    impacts_below = [b['impact_below'] for b in block_info]
    
    max_above = np.percentile(impacts_above, 98) + 1e-9
    max_below = np.percentile(impacts_below, 98) + 1e-9

    # 4. Setup Interactive Matplotlib UI (2x2 Grid)
    fig, ((ax_inv, ax_sens), (ax_eig, ax_cond)) = plt.subplots(2, 2, figsize=(18, 16), constrained_layout=True)
    fig.canvas.manager.set_window_title("H-Matrix Inverse Sensitivity Profiler")

    # --- TOP LEFT: True Inverse ---
    log_ainv = np.log10(np.abs(A_inv) + 1e-9)
    im1 = ax_inv.imshow(log_ainv, cmap='magma', aspect='equal')
    ax_inv.set_title(f"True Inverse A^{{-1}} (log10)\n<-- Click any block!")
    fig.colorbar(im1, ax=ax_inv, fraction=0.046, pad=0.04)

    # --- BOTTOM LEFT: Targeted Eigenvalues ---
    norm_lam = plt.Normalize(vmin=min(lambdas), vmax=max(lambdas))
    cmap_lam = plt.get_cmap("turbo")
    
    ax_eig.set_title("Blocks by Targeted Eigenvalue\n(Spatially weighted average $\\lambda$ of A)")
    sm_lam = plt.cm.ScalarMappable(cmap=cmap_lam, norm=norm_lam)
    sm_lam.set_array([])
    fig.colorbar(sm_lam, ax=ax_eig, fraction=0.046, pad=0.04, label="Average $\\lambda$ (higher = high frequency)")

    # --- BOTTOM RIGHT: Impact on Condition Number ---
    cmap_above = plt.get_cmap("Reds")
    cmap_below = plt.get_cmap("Blues")
    
    ax_cond.set_title(
        "Block Impact on Condition Number (Cluster Spread)\n"
        "(Whole block: Reds if above-cluster impact wins, Blues if below-cluster wins)"
    )

    sm_above = plt.cm.ScalarMappable(cmap=cmap_above, norm=plt.Normalize(vmin=0, vmax=max_above))
    sm_above.set_array([])
    cbar_above = fig.colorbar(sm_above, ax=ax_cond, orientation='horizontal', fraction=0.04, pad=0.04)
    cbar_above.set_label("Impact Above Cluster (fixes small $\\lambda_A$) — Red scale")

    sm_below = plt.cm.ScalarMappable(cmap=cmap_below, norm=plt.Normalize(vmin=0, vmax=max_below))
    sm_below.set_array([])
    cbar_below = fig.colorbar(sm_below, ax=ax_cond, orientation='horizontal', fraction=0.04, pad=0.08)
    cbar_below.set_label("Impact Below Cluster (fixes large $\\lambda_A$) — Blue scale")

    rects_inv, rects_eig, rects_cond = [], [], []
    green_cmap = plt.get_cmap("Greens")
    max_log = int(np.log2(num_units)) if num_units > 1 else 0

    # Draw the partition grid on the axes
    for i, b in enumerate(block_info):
        x, y, size_px = b['c0'], b['r0'], b['size_px']

        # 1. Styling for Inverse (Structure)
        if b['on_diag_unit']:
            face1, edge1, lw1 = "#e8a0c8", "white", 1.0
        else:
            k = int(np.log2(b['size_leaves'])) if b['size_leaves'] > 1 else 0
            face1 = green_cmap(0.35 + 0.55 * (k / max(max_log, 1)))
            edge1, lw1 = "#1a3d1a", 1.2
            
        b['orig_edge'] = edge1
        b['orig_lw'] = lw1

        rect1 = Rectangle((x - 0.5, y - 0.5), size_px, size_px,
                          linewidth=lw1, edgecolor=edge1, facecolor=face1, alpha=0.45, picker=False)
        ax_inv.add_patch(rect1)
        rects_inv.append(rect1)
        
        # 2. Styling for Eigenvalues
        face3 = cmap_lam(norm_lam(b['avg_lam']))
        rect3 = Rectangle((x - 0.5, y - 0.5), size_px, size_px,
                          linewidth=1.0, edgecolor="#111", facecolor=face3, alpha=0.9, picker=False)
        ax_eig.add_patch(rect3)
        rects_eig.append(rect3)
        
        # 3. Condition number: full square uses the dominant impact (same units as raw metrics).
        val_above = min(b['impact_above'], max_above)
        val_below = min(b['impact_below'], max_below)
        if b['impact_above'] >= b['impact_below']:
            face4 = cmap_above(val_above / max_above)
        else:
            face4 = cmap_below(val_below / max_below)

        rect4 = Rectangle(
            (x - 0.5, y - 0.5),
            size_px,
            size_px,
            linewidth=1.0,
            edgecolor="#111",
            facecolor=face4,
            alpha=0.9,
            zorder=1,
            picker=False,
        )
        ax_cond.add_patch(rect4)
        rects_cond.append(rect4)

    # Standardize axes limits natively matching imshow (removing inverted y-axis mapping)
    for ax in (ax_inv, ax_eig, ax_cond):
        ax.set_xlim(-0.5, viz_n - 0.5)
        ax.set_ylim(viz_n - 0.5, -0.5)
        ax.set_aspect('equal')

    # --- TOP RIGHT: Empty Sensitivity Map ---
    im2 = ax_sens.imshow(np.zeros_like(A_inv), cmap='magma', aspect='equal')
    ax_sens.set_title("Derivative Sensitivity vs Original A\n(Select a block on the left)")
    fig.colorbar(im2, ax=ax_sens, fraction=0.046, pad=0.04)

    # State container for click event
    state = {'selected_idx': None}
    energy_label = ax_sens.text(
        0.99, 0.01, "",
        transform=ax_sens.transAxes, ha="right", va="bottom",
        fontsize=9, linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="wheat", edgecolor="#666", alpha=0.92),
    )

    def onclick(event):
        if event.inaxes not in (ax_inv, ax_eig, ax_cond):
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        clicked_idx = None
        for i, b in enumerate(block_info):
            if b['c0'] - 0.5 <= x <= b['c1'] - 0.5 and b['r0'] - 0.5 <= y <= b['r1'] - 0.5:
                clicked_idx = i
                break

        if clicked_idx is None:
            return

        # Restore previous selection
        if state['selected_idx'] is not None:
            old_idx = state['selected_idx']
            old_info = block_info[old_idx]
            
            r1, r3, r4 = rects_inv[old_idx], rects_eig[old_idx], rects_cond[old_idx]
            
            r1.set_edgecolor(old_info['orig_edge'])
            r1.set_linewidth(old_info['orig_lw'])
            r1.set_zorder(1)
            
            r3.set_edgecolor('#111')
            r3.set_linewidth(1.0)
            r3.set_zorder(1)
            
            r4.set_edgecolor('#111')
            r4.set_linewidth(1.0)
            r4.set_zorder(1)

        # Highlight new selection on ALL three input axes
        for r_list in (rects_inv, rects_eig, rects_cond):
            r_list[clicked_idx].set_edgecolor('cyan')
            r_list[clicked_idx].set_linewidth(3.5)
            r_list[clicked_idx].set_zorder(10)
            
        state['selected_idx'] = clicked_idx

        b = block_info[clicked_idx]
        R = slice(b['r0'], b['r1'])
        C = slice(b['c0'], b['c1'])

        # Compute u_x = || A^-1[R, x] ||_2 
        u = np.sqrt(np.sum(A_inv[R, :]**2, axis=0))

        # Compute v_y = || A^-1[y, C] ||_2 
        v = np.sqrt(np.sum(A_inv[:, C]**2, axis=1))

        # S_xy = u_x * v_y (Frobenius norm of partial derivative matrix)
        S = np.outer(u, v)

        log_S = np.log10(S + 1e-12)

        im2.set_data(log_S)
        im2.set_clim(vmin=np.min(log_S), vmax=np.max(log_S))

        # Share of aggregate sensitivity "energy"
        S_sq = S * S
        e_in = float(np.sum(S_sq[R, C]))
        e_total = float(np.sum(S_sq))
        if e_total > 0:
            pct_in = 100.0 * e_in / e_total
            pct_out = 100.0 - pct_in
            energy_label.set_text(
                "Sensitivity energy vs A (Σ S²)\n"
                f"Inside block R×C: {pct_in:.1f}%\n"
                f"Outside block:     {pct_out:.1f}%"
            )

        ax_sens.set_title(
            f"Sensitivity of Block R=[{b['r0']}:{b['r1']}], C=[{b['c0']}:{b['c1']}]\n"
            f"Targeting $\\lambda \\approx {b['avg_lam']:.2f}$ | Impact: +{b['impact_above']:.2f}, -{b['impact_below']:.2f}"
        )

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', onclick)
    
    print("Interactive window ready.")
    plt.show()

if __name__ == "__main__":
    main()