import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import struct
import argparse
from pathlib import Path
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu
import time
import math

# Optional: Try importing pyamg
try:
    import pyamg
    HAS_AMG = True
except ImportError:
    HAS_AMG = False
    print("Warning: 'pyamg' not installed. AMG baseline will be skipped.")

# Import components from your updated NeuralPreconditioner
from NeuralPreconditioner import NeuralHODLR, apply_hodlr_matrix, FluidGraphDataset, HODLRLoss

# --- 1. Weight Loading ---

def read_weights_header(path):
    """Read the 28-byte header only. Returns (d_model, nhead, max_depth, input_dim, max_rank)."""
    with open(path, 'rb') as f:
        header = f.read(28)
    _, _, d_model, nhead, max_depth, input_dim, max_rank = struct.unpack('<ffiiiii', header)
    return d_model, nhead, max_depth, input_dim, max_rank

def load_weights_from_bytes(model, path):
    """Load NeuralHODLR weights from .bytes format (Dynamic Depth + Block Leaves)."""
    print(f"Loading weights from {path}...")
    with open(path, 'rb') as f:
        # Header: 2 floats + 5 ints = 28 bytes
        # Format: <ffiiiii 
        # (0, 0, d_model, nhead, max_depth, input_dim, max_rank)
        header = f.read(28)
        _, _, d_model, nhead, max_depth, input_dim, max_rank = struct.unpack('<ffiiiii', header)
        
        print(f"  Header Info -> Capacity (Depth): {max_depth}, Max Rank: {max_rank}, d_model: {d_model}")

        # Helper to read tensors
        def read_packed_tensor(target_param, transpose_check=False):
            num_elements = target_param.numel()
            # Padding: Aligned to 4 bytes (float32) but stored as float16
            read_len = num_elements + (1 if num_elements % 2 else 0)
            bytes_to_read = read_len * 2 
            
            buffer = f.read(bytes_to_read)
            if len(buffer) != bytes_to_read:
                raise ValueError(f"Unexpected EOF: wanted {bytes_to_read} bytes, got {len(buffer)}")
            
            packed = np.frombuffer(buffer, dtype=np.uint32)
            data_fp16 = packed.view(np.float16)
            
            if num_elements % 2 != 0:
                data_fp16 = data_fp16[:-1]
                
            data_fp32 = torch.from_numpy(data_fp16.astype(np.float32)).to(target_param.device)
            
            if transpose_check and len(target_param.shape) == 2:
                reshaped = data_fp32.view(target_param.shape[1], target_param.shape[0]).t()
            else:
                reshaped = data_fp32.view(target_param.shape)
                
            with torch.no_grad():
                target_param.copy_(reshaped)

        pack = lambda p, trans: read_packed_tensor(p, trans)
        
        # 1. Feature Projector
        pack(model.feature_proj.weight, True); pack(model.feature_proj.bias, False)
        
        # 2. Down Samplers
        for d in model.down_samplers:
            pack(d.local_mixer[0].weight, True); pack(d.local_mixer[0].bias, False)
            pack(d.local_mixer[2].weight, True); pack(d.local_mixer[2].bias, False)
            pack(d.norm.weight, False); pack(d.norm.bias, False)
            
        # 3. Bottleneck
        bn = model.bottleneck
        pack(bn.q_proj.weight, True); pack(bn.q_proj.bias, False); pack(bn.k_proj.weight, True); pack(bn.k_proj.bias, False)
        pack(bn.v_proj.weight, True); pack(bn.v_proj.bias, False); pack(bn.out_proj.weight, True); pack(bn.out_proj.bias, False)
        pack(bn.norm.weight, False); pack(bn.norm.bias, False)
        
        # 4. Up Samplers & HODLR Heads
        # Iterates 0 (Fine) -> MaxDepth (Coarse) matches export order
        for up, hodlr_head in zip(model.up_samplers, model.hodlr_heads):
            pack(up.up_proj.weight, True); pack(up.up_proj.bias, False)
            pack(up.fusion[0].weight, True); pack(up.fusion[0].bias, False)
            pack(up.fusion[2].weight, True); pack(up.fusion[2].bias, False)
            pack(up.norm.weight, False); pack(up.norm.bias, False)
            
            pack(hodlr_head.proj_u.weight, True); pack(hodlr_head.proj_u.bias, False)
            pack(hodlr_head.proj_u.weight, True); pack(hodlr_head.proj_u.bias, False) # U written twice
            
        # 5. Output Heads (Leaf Head)
        pack(model.norm_out.weight, False); pack(model.norm_out.bias, False)
        pack(model.leaf_head.net[0].weight, True); pack(model.leaf_head.net[0].bias, False)
        pack(model.leaf_head.net[2].weight, True); pack(model.leaf_head.net[2].bias, False)
        
    print("Weights loaded successfully.")

# --- 2. Overfit Baseline ---
# Option A: Scalar diagonal (legacy). Option B: Dense 32x32 leaf blocks (like Neural).

def apply_hodlr_matrix_diag(diag, factors, x):
    """Legacy scalar apply for Overfit Baseline."""
    B, N, _ = x.shape
    y = diag * x
    for level_idx, (u_coarse, v_coarse) in enumerate(factors):
        num_splits = 2 ** (level_idx + 1)
        block_size = N // num_splits
        if block_size == 0: continue
        
        if u_coarse.shape[1] != N:
            u_full = F.interpolate(u_coarse.transpose(1, 2), size=N, mode='linear', align_corners=False).transpose(1, 2)
            v_full = F.interpolate(v_coarse.transpose(1, 2), size=N, mode='linear', align_corners=False).transpose(1, 2)
        else:
            u_full, v_full = u_coarse, v_coarse
            
        num_pairs = num_splits // 2
        valid_len = num_pairs * 2 * block_size
        
        x_view = x[:, :valid_len].view(B, num_pairs, 2, block_size, 1)
        u_view = u_full[:, :valid_len].view(B, num_pairs, 2, block_size, -1)
        v_view = v_full[:, :valid_len].view(B, num_pairs, 2, block_size, -1)
        
        x_L, x_R = x_view[:, :, 0], x_view[:, :, 1]
        u_L, u_R = u_view[:, :, 0], u_view[:, :, 1]
        v_L, v_R = v_view[:, :, 0], v_view[:, :, 1]
        
        x_R_proj = torch.matmul(v_R.transpose(-2, -1), x_R)
        x_L_proj = torch.matmul(v_L.transpose(-2, -1), x_L)
        
        y_L_update = torch.matmul(u_L, x_R_proj)
        y_R_update = torch.matmul(u_R, x_L_proj)
        
        updates = torch.stack([y_L_update, y_R_update], dim=2).view(B, valid_len, 1)
        if valid_len < N:
            y = y + F.pad(updates, (0, 0, 0, N - valid_len))
        else:
            y = y + updates
    return y


class DirectHODLR(nn.Module):
    """Overfit Baseline (Scalar Diagonal + HODLR)."""
    def __init__(self, num_nodes, depth, max_rank, device):
        super().__init__()
        self.depth = depth
        # Learnable Scalar Diagonal
        self.diag = nn.Parameter(torch.ones(1, num_nodes, 1, device=device))
        
        self.u_factors = nn.ParameterList()
        
        # Rank Schedule: Match the Bottom-Up approach
        # Fine -> Coarse
        rank_schedule = []
        for i in range(depth):
            # i=0 (Level 0, finest).
            r = min(max_rank, 4 * (2 ** i))
            rank_schedule.append(r)
        
        # We need factors to be Coarse -> Fine for the apply function
        # But our schedule is Fine -> Coarse. 
        # The apply function (legacy) expects Coarse First.
        # So we reverse the schedule list for creation.
        # Wait, the neural model stores them Fine->Coarse in list, but applies them... how?
        # New apply_hodlr_matrix iterates factors. First factor = Root (Coarse).
        # We should stick to Coarse->Fine for storage here to match simple logic.
        
        self.ranks_coarse_to_fine = rank_schedule[::-1] # Reverse

        for r in self.ranks_coarse_to_fine:
            u = nn.Parameter(torch.randn(1, num_nodes, r, device=device) * 0.001)
            self.u_factors.append(u)

    def forward(self, x):
        factors = [(u, u) for u in self.u_factors]
        return apply_hodlr_matrix_diag(self.diag, factors, x)
    
    def get_params(self):
        factors = [(u, u) for u in self.u_factors]
        return self.diag, factors


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


def train_overfit_baseline_blocks(A_indices, A_values, num_nodes, depth, max_rank, device, leaf_size=32, steps=4000):
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

def get_dense_matrix_from_neural(leaf_blocks, factors, padded_size, num_nodes_real, device, leaf_size=32, viz_limit=200):
    """
    Reconstructs dense M from Neural Output (full n x n).
    The Neural Output operates on 'padded_size'; we probe in Padded Space, then crop to num_nodes_real.
    """
    viz_limit = min(num_nodes_real, viz_limit)
    M = torch.zeros(viz_limit, viz_limit, device='cpu')
    batch_size = 32

    with torch.no_grad():
        for i in range(0, viz_limit, batch_size):
            end = min(i + batch_size, viz_limit)
            current_batch = end - i

            # 1. Create Identity Probes in Padded Space [Batch, PaddedSize, 1]
            x_pad = torch.zeros(current_batch, padded_size, 1, device=device)
            batch_indices = torch.arange(current_batch, device=device)
            row_indices = torch.arange(i, end, device=device)
            x_pad[batch_indices, row_indices, 0] = 1.0

            # 2. Expand Params
            leaf_exp = leaf_blocks.expand(current_batch, -1, -1, -1)
            factors_exp = [(u.expand(current_batch, -1, -1), v.expand(current_batch, -1, -1)) for u, v in factors]

            # 3. Apply Neural M
            y_pad = apply_hodlr_matrix(leaf_exp, factors_exp, x_pad, leaf_size=leaf_size)

            # 4. Crop to real size and store
            y_real = y_pad[:, :viz_limit, 0].cpu()
            M[:, i:end] = y_real.T

    return M.numpy()

def get_dense_matrix_overfit(diag, factors, num_nodes, device, viz_limit=200):
    limit = min(num_nodes, viz_limit)
    M = torch.zeros(limit, limit, device='cpu')
    batch_size = 32
    with torch.no_grad():
        for i in range(0, limit, batch_size):
            end = min(i + batch_size, limit)
            current_batch = end - i
            
            x = torch.zeros(current_batch, num_nodes, 1, device=device)
            indices = torch.arange(current_batch, device=device)
            rows = torch.arange(i, end, device=device)
            x[indices, rows, 0] = 1.0
            
            diag_exp = diag.expand(current_batch, -1, -1)
            factors_exp = [(u.expand(current_batch, -1, -1), v.expand(current_batch, -1, -1)) for u, v in factors]
            
            y = apply_hodlr_matrix_diag(diag_exp, factors_exp, x)
            M[:, i:end] = y[:, :limit, 0].T.cpu()
            
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
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--rank', type=int, default=128)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save plot image (default: script_dir/inspect_model_plot.png)')
    parser.add_argument('--viz_limit', type=int, default=4000,
                        help="Max size for dense M reconstruction and viz (reduces AMG work). Use 0 for full N.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')

    # 1. Load Data (Raw)
    print(f"Loading Data from {args.data_folder}")
    dataset = FluidGraphDataset([Path(args.data_folder)])
    batch = dataset[600]
    
    # Raw Data (Not Padded yet)
    x = torch.from_numpy(batch['x']).unsqueeze(0).to(device)
    num_nodes_real = int(batch['num_nodes'])
    A_indices = batch['edge_index'].to(device)
    A_values = batch['edge_values'].to(device)
    
    print(f"System N={num_nodes_real}")

    # Ground Truth A (full n x n for viz and matmul)
    A_sparse_cpu = torch.sparse_coo_tensor(batch['edge_index'], batch['edge_values'], (num_nodes_real, num_nodes_real)).coalesce()
    n = num_nodes_real
    A_viz = A_sparse_cpu.to_dense().numpy()

    # --- MODEL 1: NEURAL HODLR ---
    print("\n1. Running Neural HODLR...")
    
    # Initialize Model with Capacity
    # Build model from weights file header so parameter shapes match (avoids Unexpected EOF)
    d_model, nhead, max_depth_file, _, max_rank_file = read_weights_header(Path(args.weights))
    model = NeuralHODLR(d_model=d_model, max_rank=max_rank_file, max_depth=max_depth_file, leaf_size=32).to(device)
    load_weights_from_bytes(model, Path(args.weights))
    
    with torch.no_grad():
        # Forward returns PADDED results and the size used
        leaf_blocks_neural, factors_neural, padded_size = model(x)
        
    print(f"  Neural Inference used Padded Size: {padded_size} (Active Depth: {int(math.log2(padded_size/32))})")
    
    viz_n = (args.viz_limit if args.viz_limit > 0 else n)
    viz_n = min(viz_n, n)
    
    M_neural = get_dense_matrix_from_neural(
        leaf_blocks_neural, factors_neural, padded_size, num_nodes_real, device, leaf_size=32, viz_limit=viz_n
    )

    # --- MODEL 2: OVERFIT HODLR (Block Diagonal, same structure as Neural) ---
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
    # Row 1–3: Neural, Overfit, AMG (or placeholder when disabled)
    amg_label = "AMG (disabled)" if SKIP_AMG else "AMG"
    methods = [("Neural", M_neural_n), ("Overfit", M_overfit_n), (amg_label, M_amg_n)]
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

    plt.show()

if __name__ == "__main__":
    main()