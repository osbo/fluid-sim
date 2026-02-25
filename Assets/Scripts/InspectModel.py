"""Inspect LeafOnly preconditioner: load weights, build M, compare A·M with AMG. Run with python3 InspectModel.py."""
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

try:
    import pyamg
    HAS_AMG = True
except ImportError:
    HAS_AMG = False
    print("Warning: 'pyamg' not installed. AMG baseline will be skipped.")

import torch

from LeafOnly import (
    LeafOnlyNet,
    read_leaf_only_header,
    load_leaf_only_weights,
    LEAF_SIZE,
    VIEW_SIZE,
    FluidGraphDataset,
)


def get_dense_amg(A_sparse_scipy, viz_limit=200, maxiter=1, tol=1e-6, progress_interval=200):
    """Build dense M by columns via AMG solves."""
    if not HAS_AMG:
        return np.eye(min(A_sparse_scipy.shape[0], viz_limit))
    print("\nComputing AMG baseline...")
    dtype = np.float64
    if A_sparse_scipy.dtype != dtype:
        A_sparse_scipy = A_sparse_scipy.astype(dtype)
    ml = pyamg.smoothed_aggregation_solver(A_sparse_scipy)
    limit = min(A_sparse_scipy.shape[0], viz_limit)
    M_dense = np.zeros((limit, limit), dtype=dtype)
    for i in range(limit):
        e_i = np.zeros(A_sparse_scipy.shape[0], dtype=dtype)
        e_i[i] = 1.0
        x0 = np.zeros(A_sparse_scipy.shape[0], dtype=dtype)
        M_dense[:, i] = ml.solve(e_i, x0=x0, maxiter=maxiter, cycle='V', tol=tol)[:limit]
        if progress_interval and (i + 1) % progress_interval == 0:
            print(f"  AMG column {i + 1}/{limit}")
    return M_dense


def main():
    script_dir = _script_dir
    data_folder = script_dir.parent / "StreamingAssets" / "TestData"
    leaf_only_weights_path = script_dir / "leaf_only_weights.bytes"
    out_path = script_dir / "inspect_model_plot.png"
    n_leaf = VIEW_SIZE
    frame_idx = 600

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')

    print(f"Loading data from {data_folder}")
    dataset = FluidGraphDataset([Path(data_folder)])
    if len(dataset) == 0:
        raise SystemExit("No frames found (need nodes.bin, edge_index_*.bin, A_values.bin).")
    frame_idx = min(frame_idx, len(dataset) - 1)
    batch = dataset[frame_idx]

    x = batch['x'].unsqueeze(0).to(device)
    num_nodes_real = int(batch['num_nodes'])
    print(f"System N={num_nodes_real}")

    A_sparse_cpu = torch.sparse_coo_tensor(
        batch['edge_index'], batch['edge_values'],
        (num_nodes_real, num_nodes_real)
    ).coalesce()
    A_viz = A_sparse_cpu.to_dense().numpy()
    n = num_nodes_real
    viz_n = min(n_leaf, n)

    # LeafOnly
    print("\nLoading and running LeafOnly...")
    if not leaf_only_weights_path.exists():
        raise SystemExit(f"Leaf-only weights not found: {leaf_only_weights_path}. Run LeafOnly.py first.")
    d_model_lo, leaf_size_lo, input_dim_lo, num_layers_lo = read_leaf_only_header(leaf_only_weights_path)
    model_leaf = LeafOnlyNet(
        input_dim=input_dim_lo, d_model=d_model_lo, leaf_size=LEAF_SIZE, num_layers=num_layers_lo,
        n_nodes=n_leaf,
    ).to(device)
    load_leaf_only_weights(model_leaf, leaf_only_weights_path)

    x_leaf = x[:, :n_leaf, :].clone()
    ei = batch['edge_index']
    em = (ei[0] < n_leaf) & (ei[1] < n_leaf)
    edge_index_leaf = ei[:, em].to(device)
    edge_values_leaf = batch['edge_values'][em].to(device)
    scale_A = batch.get('scale_A')
    if scale_A is not None and not isinstance(scale_A, torch.Tensor):
        scale_A = torch.tensor(scale_A, device=device, dtype=x_leaf.dtype)

    with torch.no_grad():
        diag_blocks, off_diag_list = model_leaf(
            x_leaf, edge_index=edge_index_leaf, edge_values=edge_values_leaf, scale_A=scale_A
        )
    num_leaves = n_leaf // LEAF_SIZE
    M_neural = np.zeros((n_leaf, n_leaf), dtype=np.float64)
    for b in range(num_leaves):
        r0, r1 = b * LEAF_SIZE, (b + 1) * LEAF_SIZE
        M_neural[r0:r1, r0:r1] = diag_blocks[0, b].cpu().numpy()
    if off_diag_list and getattr(model_leaf, 'off_diag_struct', None):
        for idx, spec in enumerate(model_leaf.off_diag_struct):
            U = off_diag_list[idx][0][0].cpu().numpy()
            V = off_diag_list[idx][1][0].cpu().numpy()
            rs, re = spec["row_start"], spec["row_end"]
            cs, ce = spec["col_start"], spec["col_end"]
            M_neural[rs:re, cs:ce] = U @ V.T
            M_neural[cs:ce, rs:re] = V @ U.T
    print(f"  LeafOnly: {num_leaves} leaves, {n_leaf}x{n_leaf} M")

    # AMG
    row = batch['edge_index'][0].numpy()
    col = batch['edge_index'][1].numpy()
    data = batch['edge_values'].numpy().astype(np.float64)
    A_scipy = csr_matrix((data, (row, col)), shape=(num_nodes_real, num_nodes_real))
    M_amg = get_dense_amg(A_scipy, viz_limit=viz_n, tol=1e-6, progress_interval=200)

    # Viz
    A_viz_n = A_viz[:viz_n, :viz_n]
    M_neural_n = M_neural[:viz_n, :viz_n]
    M_amg_n = M_amg[:viz_n, :viz_n]

    A_inv_viz = np.linalg.inv(A_viz_n)
    diag_ainv = np.diag(A_inv_viz)
    print(f"\nTrue inverse A^{{-1}} diagonal (viz {viz_n}x{viz_n}): min={diag_ainv.min():.6f}, max={diag_ainv.max():.6f}, mean={diag_ainv.mean():.6f}")

    cond_A = np.linalg.cond(A_viz_n)
    print(f"Condition number (block A): {cond_A:.2e}")
    print(f"Leaf boundaries: every {LEAF_SIZE}")

    methods = [("LeafOnly", M_neural_n), ("AMG", M_amg_n)]
    n_cols = 4
    n_rows = 1 + len(methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 + 3 * n_rows))

    # Row 0: A, A^{-1}
    axes[0, 0].imshow(np.log10(np.abs(A_viz_n) + 1e-9), cmap='magma', aspect='auto')
    axes[0, 0].set_title(f"A (input) log10 [leaf {LEAF_SIZE}x{LEAF_SIZE}]")
    plt.colorbar(axes[0, 0].images[0], ax=axes[0, 0])
    axes[0, 1].imshow(np.log10(np.abs(A_inv_viz) + 1e-9), cmap='magma', aspect='auto')
    axes[0, 1].set_title(f"A^{{-1}} (viz {viz_n}x{viz_n}) log10")
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1])
    for j in range(2, n_cols):
        axes[0, j].axis('off')

    for idx, (name, M) in enumerate(methods):
        row = 1 + idx
        axes[row, 0].imshow(np.log10(np.abs(M) + 1e-9), cmap='magma', aspect='auto')
        axes[row, 0].set_title(f"{name} M (log10)")
        plt.colorbar(axes[row, 0].images[0], ax=axes[row, 0])

        AM = A_viz_n @ M
        am_min, am_max = np.percentile(AM, [2, 98])
        if am_max - am_min < 1e-8:
            am_min, am_max = 0.0, 1.0
        axes[row, 1].imshow(AM, cmap='RdBu_r', vmin=am_min, vmax=am_max, aspect='auto')
        axes[row, 1].set_title(f"{name} A·M")
        plt.colorbar(axes[row, 1].images[0], ax=axes[row, 1])

        ax_d = axes[row, 2]
        try:
            evals = np.linalg.eigvals(AM)
            ax_d.scatter(evals.real, evals.imag, alpha=0.7, s=12, c='C0', edgecolors='none')
            ax_d.axvline(1.0, color='r', linestyle='--', alpha=0.8)
            ax_d.axhline(0.0, color='k', linestyle='-', alpha=0.3)
            ax_d.set_xlabel('Re(λ)')
            ax_d.set_ylabel('Im(λ)')
            ax_d.set_title(f"{name} Eigenvalues of A·M (n={viz_n})")
            ax_d.set_aspect('equal', adjustable='box')
            r_min, r_max = evals.real.min(), evals.real.max()
            i_max = np.abs(evals.imag).max()
            margin = 0.1 * max(r_max - r_min, 2 * i_max, 1.0) or 0.2
            ax_d.set_xlim(min(r_min, 1.0) - margin, max(r_max, 1.0) + margin)
            ax_d.set_ylim(-max(i_max, margin), max(i_max, margin))
        except Exception as e:
            ax_d.text(0.5, 0.5, f"eig failed:\n{e}", transform=ax_d.transAxes, ha='center', va='center', fontsize=9)
            ax_d.set_title(f"{name} Eigenvalues (failed)")

        ax_t = axes[row, 3]
        ax_t.axis('off')
        cond_AM = np.linalg.cond(AM)
        err_fro = np.linalg.norm(AM - np.eye(viz_n)) / np.linalg.norm(np.eye(viz_n))
        msg = [
            f"Method: {name}",
            f"Cond(AM): {cond_AM:.2e}",
            f"Improvement: {cond_A/cond_AM:.2f}x",
            f"Frobenius Err: {err_fro:.4f}",
        ]
        color = 'green' if cond_A / cond_AM > 1.0 else 'red'
        for i, line in enumerate(msg):
            ax_t.text(0.1, 0.8 - i * 0.15, line, fontsize=12, color='black' if i != 2 else color, fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    main()
