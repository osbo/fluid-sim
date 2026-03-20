"""
RankDecay: SVD of A chunked into HODLR blocks.
Uses the same data loading and frame as InspectModel; chunks the system matrix A
into HODLR blocks (leaf-aligned sizes), computes SVD on each block, and plots
singular value decay to show how rank required varies with block size and
distance from the diagonal.
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
from pathlib import Path

from NeuralPreconditioner import FluidGraphDataset
from leafonly.config import LEAF_SIZE


def get_hodlr_block_sizes(n, leaf_size=None):
    """Return list of (level_idx, block_size) for HODLR off-diagonal levels.
    Level 0 = coarsest (largest block_size, n/2); level num_levels-1 = finest (block_size=leaf_size).
    Requires n = leaf_size * 2^depth for some depth.
    """
    if leaf_size is None:
        leaf_size = LEAF_SIZE
    if n % leaf_size != 0:
        return []
    num_blocks = n // leaf_size
    if num_blocks & (num_blocks - 1) != 0:
        return []
    num_levels = int(round(math.log2(num_blocks)))
    return [
        (level_idx, leaf_size * (2 ** (num_levels - 1 - level_idx)))
        for level_idx in range(num_levels)
    ]


def enumerate_off_diagonal_blocks(A, leaf_size=None):
    """
    Yield (level_idx, block_size, block_matrix, label) for each off-diagonal
    HODLR block. block_matrix is the dense 2D slice of A.
    """
    if leaf_size is None:
        leaf_size = LEAF_SIZE
    n = A.shape[0]
    levels = get_hodlr_block_sizes(n, leaf_size)
    for level_idx, block_size in levels:
        num_blocks_at_level = n // block_size
        num_pairs = num_blocks_at_level // 2
        for p in range(num_pairs):
            i_start = p * 2 * block_size
            j_start = (p * 2 + 1) * block_size
            # Upper block: rows [i_start : i_start+block_size], cols [j_start : j_start+block_size]
            block_upper = A[i_start : i_start + block_size, j_start : j_start + block_size]
            label = f"L{level_idx} bs={block_size} pair={p}"
            yield level_idx, block_size, block_upper.copy(), label


def enumerate_diagonal_blocks(A, leaf_size=None):
    """Yield (block_idx, block_matrix) for each diagonal leaf block."""
    if leaf_size is None:
        leaf_size = LEAF_SIZE
    n = A.shape[0]
    num_blocks = n // leaf_size
    for b in range(num_blocks):
        start = b * leaf_size
        block = A[start : start + leaf_size, start : start + leaf_size]
        yield b, block.copy()


def compute_svd_decay(block):
    """Return array of singular values (descending)."""
    U, s, Vh = np.linalg.svd(block, full_matrices=False)
    return np.asarray(s, dtype=float)


def main():
    parser = argparse.ArgumentParser(
        description="SVD rank decay of A in HODLR blocks (same data/frame as InspectModel)."
    )
    script_dir = Path(__file__).parent
    default_data = script_dir.parent / "StreamingAssets" / "TestData"

    parser.add_argument("--data_folder", type=str, default=str(default_data))
    parser.add_argument("--frame", type=int, default=600,
                        help="Dataset frame index (same as InspectModel default).")
    parser.add_argument("--viz_limit", type=int, default=0,
                        help="Max size of A to use. Default 0 = full N.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to save plot (default: script_dir/rank_decay_plot.png)")
    args = parser.parse_args()

    # 1. Load data (same as InspectModel)
    print(f"Loading data from {args.data_folder}")
    dataset = FluidGraphDataset([Path(args.data_folder)])
    batch = dataset[args.frame]
    num_nodes_real = int(batch["num_nodes"])
    A_sparse = np.zeros((num_nodes_real, num_nodes_real))
    row, col = batch["edge_index"].numpy()
    val = batch["edge_values"].numpy()
    A_sparse[row, col] = val
    n = num_nodes_real
    print(f"Frame {args.frame}: N={n}")

    # Full N: pad to power-of-2 multiple of leaf_size so all HODLR levels exist
    viz_n = (args.viz_limit if args.viz_limit > 0 else n)
    viz_n = min(viz_n, n)
    num_blocks_min = (viz_n + LEAF_SIZE - 1) // LEAF_SIZE
    depth = int(math.ceil(math.log2(num_blocks_min)))
    padded_n = LEAF_SIZE * (2 ** depth)  # >= viz_n, all HODLR levels 32, 64, 128, 256, 512, ...
    A = np.zeros((padded_n, padded_n), dtype=np.float64)
    A[:viz_n, :viz_n] = A_sparse[:viz_n, :viz_n]
    for i in range(viz_n, padded_n):
        A[i, i] = 1.0
    n_work = padded_n
    print(f"Working matrix size: {n_work} (full N={n}, padded for {depth} levels)")

    levels = get_hodlr_block_sizes(n_work, LEAF_SIZE)
    if not levels:
        print("N is not a power-of-2 multiple of leaf_size; cannot run HODLR block enumeration.")
        return

    # Off-diagonal: group by (level_idx, block_size), collect singular values
    from collections import defaultdict
    off_diag_by_level = defaultdict(list)
    for level_idx, block_size, block_mat, label in enumerate_off_diagonal_blocks(A, LEAF_SIZE):
        s = compute_svd_decay(block_mat)
        off_diag_by_level[(level_idx, block_size)].append((label, s))

    # Fit exponential decay with a=1 (y = exp(-x/tau))
    THRESH = 1e-3
    layer_indices = []
    tau_exp = []
    block_sizes = []
    print("\n--- Exponential fit (a=1) to singular values > 1e-3 ---")
    for level_idx, block_size in levels:
        key = (level_idx, block_size)
        if key not in off_diag_by_level:
            continue
        entries = off_diag_by_level[key]
        max_len = max(len(s) for _, s in entries)
        avg_s = np.zeros(max_len)
        cnt = np.zeros(max_len)
        for _, s in entries:
            avg_s[: len(s)] += s
            cnt[: len(s)] += 1
        avg_s = np.where(cnt > 0, avg_s / np.maximum(cnt, 1), 0.0)
        x = np.arange(1, len(avg_s) + 1, dtype=float)
        mask = avg_s > THRESH
        if mask.sum() < 2:
            print(f"  L{level_idx} bs={block_size}: too few points above {THRESH}, skipping")
            continue
        x_fit = x[mask]
        y_fit = avg_s[mask]
        log_y = np.log(y_fit + 1e-300)
        # Exponential with a=1: log(y) = -x/tau  ->  tau = -1/b1
        b1_exp = np.sum(x_fit * log_y) / (np.sum(x_fit**2) + 1e-300)
        tau = -1.0 / b1_exp if b1_exp < 0 else np.nan
        layer_indices.append(level_idx)
        tau_exp.append(tau)
        block_sizes.append(block_size)
        print(f"  L{level_idx} bs={block_size}: tau={tau:.4e}")

    # Global fit: tau = bs^p  ->  log(tau) = p*log(bs)  (no intercept)
    valid = np.array([np.isfinite(t) and t > 0 for t in tau_exp], dtype=bool)
    if np.sum(valid) >= 2:
        bs_arr = np.array(block_sizes, dtype=float)[valid]
        tau_arr = np.array(tau_exp, dtype=float)[valid]
        log_bs = np.log(bs_arr)
        log_tau = np.log(tau_arr)
        p = np.sum(log_bs * log_tau) / (np.sum(log_bs**2) + 1e-300)
        print(f"\n  Global fit tau = bs^p  ->  p = {p:.4e}")
    else:
        print("\n  Global fit tau = bs^p: skipped (need tau>0 for at least 2 layers)")

    # Plot: x = layer, y = tau (exponential characteristic rank)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(layer_indices, tau_exp, "o-", color="black", linewidth=1.5, markersize=8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("tau (exponential decay scale)")
    ax.set_title("Singular value decay: y=exp(-x/tau), s > 1e-3")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = args.output or (script_dir / "rank_decay_plot.png")
    out_path = Path(out_path)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.show()

    # Rank decay by layer: all layers on one graph, normalized to max per layer, full rainbow colormap
    print("\n--- Normalization max (per layer, for rank decay plot) ---")
    cmap = plt.get_cmap("viridis")  # full rainbow
    n_layers = len(levels)
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    for i, (level_idx, block_size) in enumerate(levels):
        key = (level_idx, block_size)
        if key not in off_diag_by_level:
            continue
        entries = off_diag_by_level[key]
        max_len = max(len(s) for _, s in entries)
        avg_s = np.zeros(max_len)
        cnt = np.zeros(max_len)
        for _, s in entries:
            avg_s[: len(s)] += s
            cnt[: len(s)] += 1
        avg_s = np.where(cnt > 0, avg_s / np.maximum(cnt, 1), 0.0)
        s_max = avg_s.max()
        print(f"  L{level_idx} bs={block_size}: max = {s_max:.4e}")
        if s_max > 0:
            avg_s = avg_s / s_max
        rank_idx = np.arange(1, len(avg_s) + 1, dtype=float)
        color = cmap(i / max(1, n_layers - 1)) if n_layers > 1 else cmap(0)
        ax2.plot(rank_idx, avg_s, "o-", color=color, linewidth=1, markersize=3, label=f"L{level_idx} bs={block_size}")
    ax2.set_xlabel("Singular value index (rank)")
    ax2.set_ylabel("Singular value (normalized to max per layer)")
    ax2.set_title("Rank decay by layer")
    ax2.set_xlim(0, 256)
    ax2.set_ylim(0, 1)
    ax2.legend(ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path_layers = script_dir / "rank_decay_layer8.png"
    plt.savefig(out_path_layers, dpi=150, bbox_inches="tight")
    print(f"Rank decay by layer saved to {out_path_layers}")
    plt.show()


if __name__ == "__main__":
    main()
