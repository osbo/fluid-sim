import sys
from pathlib import Path

# Ensure script directory is on path so NeuralPreconditioner imports when run from any cwd
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.sparse import csr_matrix
import time
import math

# Optional: Try importing pyamg
try:
    import pyamg
    HAS_AMG = True
except ImportError:
    HAS_AMG = False
    print("Warning: 'pyamg' not installed. AMG baseline will be skipped.")

# Import model, apply, data, and weight loading from NeuralPreconditioner
from NeuralPreconditioner import (
    HGT_OL,
    apply_neural_hodlr,
    apply_hodlr_matrix,
    FluidGraphDataset,
    _pad_to_hodlr_size,
    read_weights_header,
    load_hgt_ol_weights_from_bytes,
)

# --- 1. Overfit Baseline (Dense 32x32 leaf blocks, like Neural) ---

def _hodlr_rank_schedule_n23(depth, leaf_size, num_nodes, max_rank, min_rank=4, scale=0.5):
    """Rank per level using N^(2/3) scaling: r(level) = scale * block_size^(2/3), clamped to [min_rank, max_rank].
    Levels are ordered Coarse (index 0) -> Fine, matching apply_hodlr_matrix (level_idx 0 = coarsest)."""
    ranks = []
    for level_idx in range(depth):
        # Same block_size formula as NeuralPreconditioner.apply_hodlr_matrix
        block_size = leaf_size * (2 ** (depth - 1 - level_idx))
        r = max(min_rank, min(max_rank, int(round(scale * (block_size ** (2 / 3))))))
        if r % 2 != 0:
            r = min(max_rank, r + 1)  # keep even for stability
        ranks.append(r)
    return ranks


class DirectHODLRBlocks(nn.Module):
    """Overfit baseline with dense 32x32 leaf blocks (like Neural), so M has dense chunks on the diagonal.
    Uses N^(2/3) rank scaling per level and a learnable scale for the low-rank part."""
    def __init__(self, num_nodes, depth, max_rank, device, leaf_size=32, rank_scale=2.0):
        super().__init__()
        assert num_nodes % leaf_size == 0, "num_nodes must be divisible by leaf_size"
        self.depth = depth
        self.leaf_size = leaf_size
        num_blocks = num_nodes // leaf_size

        # Learnable dense blocks [1, NumBlocks, leaf_size, leaf_size]; init near identity (L so L@L.T ~ I)
        eye = torch.eye(leaf_size, device=device).unsqueeze(0).unsqueeze(0).expand(1, num_blocks, -1, -1).clone()
        self.leaf_blocks = nn.Parameter(eye + 0.01 * torch.randn(1, num_blocks, leaf_size, leaf_size, device=device))

        # N^(2/3) rank schedule: Coarse -> Fine (index 0 = coarsest, matches apply_hodlr_matrix)
        self.ranks_coarse_to_fine = _hodlr_rank_schedule_n23(depth, leaf_size, num_nodes, max_rank, scale=rank_scale)
        if depth > 0:
            print(f"  Rank Schedule (N^2/3, Coarse->Fine): {self.ranks_coarse_to_fine}")

        self.u_factors = nn.ParameterList()
        for r in self.ranks_coarse_to_fine:
            u = nn.Parameter(torch.randn(1, num_nodes, r, device=device) * 0.05)
            self.u_factors.append(u)

        # Learnable scale for the entire HODLR off-diagonal sum so optimizer can balance dense vs low-rank
        self.hodlr_scale = nn.Parameter(torch.tensor(1.0, device=device))

    def forward(self, x):
        s = self.hodlr_scale
        factors = [(u * s, u * s) for u in self.u_factors]
        return apply_hodlr_matrix(self.leaf_blocks, factors, x, leaf_size=self.leaf_size)

    def get_params(self):
        s = self.hodlr_scale
        factors = [(u * s, u * s) for u in self.u_factors]
        return self.leaf_blocks, factors


def train_overfit_baseline_blocks(A_indices, A_values, num_nodes, depth, max_rank, device, leaf_size=32, steps=100):
    """Overfit with dense leaf blocks (same structure as Neural); M will show dense chunks on the diagonal."""
    print(f"\n--- Training Overfit HODLR Baseline (block diagonal, {steps} steps) ---")
    model = DirectHODLRBlocks(num_nodes, depth, max_rank, device, leaf_size=leaf_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    batch_size = 32

    if device.type == 'mps':
        A_sparse = torch.sparse_coo_tensor(A_indices.cpu(), A_values.cpu(), (num_nodes, num_nodes)).to('cpu')
    else:
        A_sparse = torch.sparse_coo_tensor(A_indices, A_values, (num_nodes, num_nodes))

    t0 = time.time()
    for i in range(steps):
        optimizer.zero_grad()
        leaf_blocks, factors = model.get_params()

        z = torch.randn(batch_size, num_nodes, 1, device=device)
        if device.type == 'mps':
            z_cpu = z.squeeze().t().cpu()
            Az_cpu = torch.sparse.mm(A_sparse, z_cpu).t().unsqueeze(-1)
            Az = Az_cpu.to(device)
        else:
            z_flat = z.squeeze(-1).t()
            Az_flat = torch.sparse.mm(A_sparse, z_flat)
            Az = Az_flat.t().unsqueeze(-1)

        leaf_exp = leaf_blocks.expand(batch_size, -1, -1, -1)
        factors_exp = [(u.expand(batch_size, -1, -1), v.expand(batch_size, -1, -1)) for u, v in factors]
        res = apply_hodlr_matrix(leaf_exp, factors_exp, Az, leaf_size=leaf_size)

        loss = F.mse_loss(res, z)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"  [Step {i}] Loss: {loss.item():.6f}")

    print(f"  Overfit Training finished in {time.time()-t0:.2f}s. Final Loss: {loss.item():.6f}")
    return model.get_params()


# --- 3. Dense Reconstruction ---

def get_dense_matrix_from_neural(leaf_blocks, factors, padded_size, num_nodes_real, device, leaf_size=32, viz_limit=200, use_neural_apply=False, off_diag_scale=None):
    """
    Reconstructs dense M from HODLR output (full n x n).
    Operates on padded_size; we probe in padded space, then crop to num_nodes_real.
    use_neural_apply: True for HGT_OL output (token-space factors -> apply_neural_hodlr),
                      False for overfit baseline (full-node factors -> apply_hodlr_matrix).
    off_diag_scale: when use_neural_apply, pass model.hodlr_scale for correct scaling.
    """
    viz_limit = min(num_nodes_real, viz_limit)
    M = torch.zeros(viz_limit, viz_limit, device='cpu')
    batch_size = 32
    apply_fn = apply_neural_hodlr if use_neural_apply else apply_hodlr_matrix

    with torch.no_grad():
        for i in range(0, viz_limit, batch_size):
            end = min(i + batch_size, viz_limit)
            current_batch = end - i

            x_pad = torch.zeros(current_batch, padded_size, 1, device=device)
            batch_indices = torch.arange(current_batch, device=device)
            row_indices = torch.arange(i, end, device=device)
            x_pad[batch_indices, row_indices, 0] = 1.0

            leaf_exp = leaf_blocks.expand(current_batch, -1, -1, -1)
            if use_neural_apply:
                factors_exp = [u.expand(current_batch, -1, -1) for u in factors]
            else:
                factors_exp = [(u.expand(current_batch, -1, -1), v.expand(current_batch, -1, -1)) for u, v in factors]

            if use_neural_apply:
                y_pad = apply_fn(leaf_exp, factors_exp, x_pad, leaf_size=leaf_size, off_diag_scale=off_diag_scale)
            else:
                y_pad = apply_fn(leaf_exp, factors_exp, x_pad, leaf_size=leaf_size)

            y_real = y_pad[:, :viz_limit, 0].cpu()
            M[:, i:end] = y_real.T

    return M.numpy()

def get_dense_amg(A_sparse_scipy, viz_limit=200, maxiter=1, tol=1e-6, progress_interval=200):
    """Build dense M by columns via AMG solves. Use smaller viz_limit to reduce work."""
    if not HAS_AMG: return np.eye(min(A_sparse_scipy.shape[0], viz_limit))
    print("\n--- Computing AMG Baseline ---")
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

# --- 4. Main ---

def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).parent
    default_data = script_dir.parent / "StreamingAssets" / "TestData"
    default_model = script_dir / "model_weights.bytes"
    
    parser.add_argument('--data_folder', type=str, default=str(default_data))
    parser.add_argument('--weights', type=str, default=str(default_model))
    parser.add_argument('--d_model', type=int, default=None,
                        help='Override d_model when loading (default: use value from weights file header). Use e.g. 64 to load old/smaller checkpoints.')
    parser.add_argument('--rank', type=int, default=128)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save plot image (default: script_dir/inspect_model_plot.png)')
    parser.add_argument('--viz_limit', type=int, default=300,
                        help="Max size for dense M reconstruction and viz (reduces AMG work). Use 0 for full N.")
    parser.add_argument('--overfit', action='store_true',
                        help='Enable Overfit HODLR baseline training and its graphs (disabled by default).')
    args = parser.parse_args()
    SKIP_OVERFIT = not args.overfit

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')

    # 1. Load Data (Raw)
    print(f"Loading Data from {args.data_folder}")
    dataset = FluidGraphDataset([Path(args.data_folder)])
    if len(dataset) == 0:
        raise SystemExit("No frames found in data folder (need nodes.bin, edge_index_*.bin, A_values.bin).")
    frame_idx = min(600, len(dataset) - 1)
    batch = dataset[frame_idx]
    
    # Raw Data (Not Padded yet)
    x = batch['x'].unsqueeze(0).to(device)
    num_nodes_real = int(batch['num_nodes'])
    A_indices = batch['edge_index'].to(device)
    A_values = batch['edge_values'].to(device)
    
    print(f"System N={num_nodes_real}")

    # Ground Truth A (full n x n for viz and matmul)
    A_sparse_cpu = torch.sparse_coo_tensor(batch['edge_index'], batch['edge_values'], (num_nodes_real, num_nodes_real)).coalesce()
    n = num_nodes_real
    A_viz = A_sparse_cpu.to_dense().numpy()

    # --- MODEL 1: HGT_OL (Neural Preconditioner) ---
    # HGT_OL uses SymmetricNeighborEmbed and expects 124-dim node features (pos, diag, 24 neighbor slots).
    print("\n1. Loading and running HGT_OL (neural preconditioner)...")
    leaf_size = 32
    d_model_h, nhead, depth_file, input_dim_file, max_rank_file = read_weights_header(Path(args.weights))
    d_model = args.d_model if args.d_model is not None else d_model_h
    if args.d_model is not None:
        print(f"  Using d_model={d_model} (override; header had {d_model_h})")
    depth = depth_file
    if x.shape[-1] != 124:
        raise SystemExit(f"Neural model expects 124-dim input (SymmetricNeighborEmbed); got {x.shape[-1]}. Use FluidGraphDataset with the same feature layout.")
    # Use N_pad from the saved model (leaf_size * 2^depth) so sequence length matches training
    N_pad = leaf_size * (2 ** depth)
    padded_size = N_pad
    if num_nodes_real > N_pad:
        raise SystemExit(f"Data has num_nodes={num_nodes_real} but model (depth={depth}) expects N_pad={N_pad}. Use weights trained for this size.")
    rank_scale = 2.0  # must match training; not stored in header
    model = HGT_OL(
        input_dim=input_dim_file,
        d_model=d_model,
        depth=depth,
        leaf_size=leaf_size,
        max_rank=max_rank_file,
        rank_scale=rank_scale,
    ).to(device)
    load_hgt_ol_weights_from_bytes(model, Path(args.weights))

    # Pad node features to N_pad
    x_padded = F.pad(x, (0, 0, 0, N_pad - num_nodes_real), value=0.0)
    with torch.no_grad():
        leaf_blocks_neural, factors_neural = model(x_padded)

    print(f"  HGT_OL inference: padded_size={padded_size}, depth={depth}")

    viz_n = (args.viz_limit if args.viz_limit > 0 else n)
    viz_n = min(viz_n, n)

    M_neural = get_dense_matrix_from_neural(
        leaf_blocks_neural, factors_neural, padded_size, num_nodes_real, device, leaf_size=leaf_size, viz_limit=viz_n, use_neural_apply=True, off_diag_scale=model.hodlr_scale
    )

    # --- MODEL 2: OVERFIT HODLR (Block Diagonal, same structure as Neural) ---
    if SKIP_OVERFIT:
        print("\n2. Overfit HODLR skipped (disabled).")
        M_overfit = np.eye(viz_n)  # placeholder, not plotted
    else:
        print("\n2. Training Overfit HODLR (block diagonal)...")
        leaf_size = 32
        # Mandate power-of-2 * leaf_size so HODLR levels align: finest block_size=leaf_size, then double each level
        num_blocks_min = (num_nodes_real + leaf_size - 1) // leaf_size
        active_depth = int(math.ceil(math.log2(num_blocks_min)))
        padded_size_overfit = leaf_size * (2 ** active_depth)
        leaf_blocks_overfit, factors_overfit = train_overfit_baseline_blocks(
            A_indices, A_values, padded_size_overfit, active_depth, args.rank, device, leaf_size=leaf_size
        )
        print(f"  Viz limit: {viz_n} (full N={n})")

        M_overfit = get_dense_matrix_from_neural(
            leaf_blocks_overfit, factors_overfit, padded_size_overfit, num_nodes_real, device, leaf_size=leaf_size, viz_limit=viz_n
        )

    # --- MODEL 3: AMG (temporarily disabled) ---
    SKIP_AMG = False
    if SKIP_AMG:
        print("\n3. AMG skipped (disabled).")
        M_amg = np.eye(viz_n)
    else:
        print("\n3. Computing AMG...")
        row = batch['edge_index'][0].numpy()
        col = batch['edge_index'][1].numpy()
        data = batch['edge_values'].numpy().astype(np.float64)
        A_scipy = csr_matrix((data, (row, col)), shape=(num_nodes_real, num_nodes_real))
        M_amg = get_dense_amg(A_scipy, viz_limit=viz_n, tol=1e-6, progress_interval=200)

    # --- Visualization (use viz_n so all matrices are consistent) ---
    A_viz_n = A_viz[:viz_n, :viz_n]
    M_neural_n = M_neural[:viz_n, :viz_n]
    M_overfit_n = M_overfit[:viz_n, :viz_n]
    M_amg_n = M_amg[:viz_n, :viz_n]

    # Approximate true inverse of the viz block (for comparison; only this block is inverted)
    A_inv_viz = np.linalg.inv(A_viz_n)
    diag_ainv = np.diag(A_inv_viz)
    print(f"\nTrue inverse A^{{-1}} diagonal (viz {viz_n}x{viz_n}): min={diag_ainv.min():.6f}, max={diag_ainv.max():.6f}, mean={diag_ainv.mean():.6f}, std={diag_ainv.std():.6f}")

    matrices = {
        "A (Input)": A_viz_n,
        "Neural M": M_neural_n,
        "Overfit M": M_overfit_n,
        "AMG M": M_amg_n
    }
    
    leaf_size = 32

    fig, axes = plt.subplots(4, 4, figsize=(20, 14))
    # Row 0: A (input) and A^{-1} (viz block)
    ax_a = axes[0, 0]
    im_a = ax_a.imshow(np.log10(np.abs(A_viz_n) + 1e-9), cmap='magma', aspect='auto')
    ax_a.set_title(f"A (input) log10 [grid=leaf {leaf_size}x{leaf_size}]")
    plt.colorbar(im_a, ax=ax_a)
    ax_ainv = axes[0, 1]
    im_ainv = ax_ainv.imshow(np.log10(np.abs(A_inv_viz) + 1e-9), cmap='magma', aspect='auto')
    ax_ainv.set_title(f"A^{{-1}} (viz {viz_n}x{viz_n} block) log10")
    plt.colorbar(im_ainv, ax=ax_ainv)
    for j in range(2, 4):
        axes[0, j].axis('off')
    # Row 1–3: Neural, Overfit (if enabled), AMG (or placeholder when disabled)
    amg_label = "AMG (disabled)" if SKIP_AMG else "AMG"
    methods = [("Neural", M_neural_n)]
    if not SKIP_OVERFIT:
        methods.append(("Overfit", M_overfit_n))
    methods.append((amg_label, M_amg_n))
    cond_A = np.linalg.cond(A_viz_n)
    print(f"\nCondition Number (Block A): {cond_A:.2e}")
    print(f"Leaf boundaries: every {leaf_size} (cyan grid on M plots; nodes are Morton-ordered)")

    for idx, (name, M) in enumerate(methods):
        row = idx + 1
        ax_m = axes[row, 0]
        im = ax_m.imshow(np.log10(np.abs(M) + 1e-9), cmap='magma', aspect='auto')
        ax_m.set_title(f"{name} M (log10) [grid=leaf {leaf_size}x{leaf_size}]")
        plt.colorbar(im, ax=ax_m)

        AM = A_viz_n @ M
        ax_am = axes[row, 1]
        am_min, am_max = np.percentile(AM, [2, 98])
        if am_max - am_min < 1e-8: am_min, am_max = 0.0, 1.0
        im2 = ax_am.imshow(AM, cmap='RdBu_r', vmin=am_min, vmax=am_max, aspect='auto')
        plt.colorbar(im2, ax=ax_am)
        ax_am.set_title(f"{name} A·M")
        
        ax_d = axes[row, 2]
        ax_d.plot(np.diag(AM), alpha=0.8)
        ax_d.axhline(1.0, color='r', linestyle='--')
        ax_d.set_ylim(-0.5, 2.0)
        ax_d.set_title(f"{name} Diag(A·M)")
        
        ax_t = axes[row, 3]
        ax_t.axis('off')
        
        cond_AM = np.linalg.cond(AM)
        err_fro = np.linalg.norm(AM - np.eye(viz_n)) / np.linalg.norm(np.eye(viz_n))
        
        msg = [
            f"Method: {name}",
            f"Cond(AM): {cond_AM:.2e}",
            f"Improvement: {cond_A/cond_AM:.2f}x",
            f"Frobenius Err: {err_fro:.4f}"
        ]
        color = 'green' if cond_A/cond_AM > 1.0 else 'red'
        for i, line in enumerate(msg):
            ax_t.text(0.1, 0.8 - i*0.15, line, fontsize=12, color='black' if i!=2 else color, fontfamily='monospace')

    plt.tight_layout()

    out_path = args.output
    if out_path is None:
        out_path = script_dir / "inspect_model_plot.png"
    else:
        out_path = Path(out_path)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")

    # plt.show()

if __name__ == "__main__":
    main()