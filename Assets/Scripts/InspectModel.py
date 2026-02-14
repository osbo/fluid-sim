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

# Optional: Try importing pyamg
try:
    import pyamg
    HAS_AMG = True
except ImportError:
    HAS_AMG = False
    print("Warning: 'pyamg' not installed. AMG baseline will be skipped.")

# Import components from your updated NeuralPreconditioner
from NeuralPreconditioner import NeuralHODLR, apply_hodlr_matrix, FluidGraphDataset, HODLRLoss

# --- 1. Weight Loading with Variable Rank Support ---

def load_weights_from_bytes(model, path):
    """Load NeuralHODLR weights from .bytes format with Variable Rank support."""
    print(f"Loading weights from {path}...")
    with open(path, 'rb') as f:
        # Header: 2 floats (8 bytes) + 5 ints (20 bytes) = 28 bytes
        header = f.read(28)
        # Unpack header: The last integer is now MAX_RANK
        _, _, d_model, nhead, depth, input_dim, max_rank = struct.unpack('<ffiiiii', header)
        
        print(f"  Header Info -> Depth: {depth}, Max Rank: {max_rank}, d_model: {d_model}")

        # Helper to read tensors based on the TARGET shape in the initialized model
        def read_packed_tensor(target_param, transpose_check=False):
            num_elements = target_param.numel()
            # Padding logic: File format aligns to 4 bytes (float32), but weights are float16
            # If num_elements is odd, 1 dummy float16 was added.
            read_len = num_elements + (1 if num_elements % 2 else 0)
            bytes_to_read = read_len * 2 # 2 bytes per float16
            
            buffer = f.read(bytes_to_read)
            if len(buffer) != bytes_to_read:
                raise ValueError(f"Unexpected EOF: wanted {bytes_to_read} bytes, got {len(buffer)}")
            
            packed = np.frombuffer(buffer, dtype=np.uint32)
            data_fp16 = packed.view(np.float16)
            
            # Remove padding if necessary
            if num_elements % 2 != 0:
                data_fp16 = data_fp16[:-1]
                
            data_fp32 = torch.from_numpy(data_fp16.astype(np.float32)).to(target_param.device)
            
            # Reshape logic
            if transpose_check and len(target_param.shape) == 2:
                # Weights were flattened from (Out, In), but typically stored as (In, Out) or similar in Unity
                # We adhere to the target shape. 
                # If target is (Out, In), and data came in row-major, we verify:
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
        
        # 4. Up Samplers & HODLR Heads (Variable Rank happens here)
        # Because we iterate over model.hodlr_heads, and the model was init'd with the schedule,
        # target_param.numel() will be correct for each specific level!
        for up, hodlr_head in zip(model.up_samplers, model.hodlr_heads):
            # UpSampler
            pack(up.up_proj.weight, True); pack(up.up_proj.bias, False)
            pack(up.fusion[0].weight, True); pack(up.fusion[0].bias, False)
            pack(up.fusion[2].weight, True); pack(up.fusion[2].bias, False)
            pack(up.norm.weight, False); pack(up.norm.bias, False)
            
            # HODLR Head (Rank varies per loop)
            pack(hodlr_head.proj_u.weight, True); pack(hodlr_head.proj_u.bias, False)
            pack(hodlr_head.proj_u.weight, True); pack(hodlr_head.proj_u.bias, False) # U written twice in file
            
        # 5. Output Heads (BlockDiagonalHead / LeafHead)
        pack(model.norm_out.weight, False); pack(model.norm_out.bias, False)
        pack(model.leaf_head.net[0].weight, True); pack(model.leaf_head.net[0].bias, False)
        pack(model.leaf_head.net[2].weight, True); pack(model.leaf_head.net[2].bias, False)
        
    print("Weights loaded successfully.")

# --- 2. Overfit Baseline: Diag-based HODLR (legacy apply) ---

def apply_hodlr_matrix_diag(diag, factors, x):
    """Legacy apply: diag is [B, N, 1], element-wise then HODLR off-diagonals. For overfit baseline only."""
    B, N, _ = x.shape
    y = diag * x
    for level_idx, (u_coarse, v_coarse) in enumerate(factors):
        num_splits = 2 ** (level_idx + 1)
        block_size = N // num_splits
        if block_size == 0:
            continue
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
    """
    Overfit Baseline.
    Updated to use Telescoping Rank (Variable Rank) to match Neural Architecture.
    """
    def __init__(self, num_nodes, depth, max_rank, device, min_rank=4):
        super().__init__()
        self.depth = depth
        # Learnable Diagonal (Initialize near 1.0)
        self.diag = nn.Parameter(torch.ones(1, num_nodes, 1, device=device))
        
        # Learnable Low-Rank Factors (Variable Rank)
        self.u_factors = nn.ParameterList()
        
        # Recreate the schedule logic
        rank_schedule = []
        for i in range(depth):
            dist_from_top = (depth - 1) - i
            calc_rank = max(min_rank, max_rank // (2 ** dist_from_top))
            rank_schedule.append(calc_rank)
            
        print(f"Overfit Baseline Schedule: {rank_schedule}")

        for r in rank_schedule:
            # Shape: [1, N, CurrentRank]
            u = nn.Parameter(torch.randn(1, num_nodes, r, device=device) * 0.001)
            self.u_factors.append(u)

    def forward(self, x):
        factors = [(u, u) for u in self.u_factors]
        return apply_hodlr_matrix_diag(self.diag, factors, x)
    
    def get_params(self):
        factors = [(u, u) for u in self.u_factors]
        return self.diag, factors

def train_overfit_baseline(A_indices, A_values, num_nodes, depth, max_rank, device, steps=400):
    print(f"\n--- Training Overfit HODLR Baseline ({steps} steps) ---")
    
    # 1. Setup Model (Now uses max_rank to build schedule)
    model = DirectHODLR(num_nodes, depth, max_rank, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005) 
    
    # 2. CREATE FIXED PROBES
    num_probes = 64
    z_fixed = torch.randn(num_probes, num_nodes, 1, device=device)
    
    # Handle Sparse Matrix logic for Pre-computation
    if device.type == 'mps':
        A_sparse = torch.sparse_coo_tensor(A_indices.cpu(), A_values.cpu(), (num_nodes, num_nodes)).to('cpu')
        z_cpu = z_fixed.squeeze().t() 
        Az_cpu = torch.sparse.mm(A_sparse, z_cpu).t().unsqueeze(-1)
        Az_fixed = Az_cpu.to(device)
    else:
        A_sparse = torch.sparse_coo_tensor(A_indices, A_values, (num_nodes, num_nodes))
        z_flat = z_fixed.squeeze(-1).t() 
        Az_flat = torch.sparse.mm(A_sparse, z_flat) 
        Az_fixed = Az_flat.t().unsqueeze(-1) 

    t0 = time.time()
    for i in range(steps):
        optimizer.zero_grad()
        diag, factors = model.get_params()
        
        # Expand for batch
        B = Az_fixed.shape[0]
        diag_exp = diag.expand(B, -1, -1)
        factors_exp = [(u.expand(B, -1, -1), v.expand(B, -1, -1)) for u, v in factors]

        res = apply_hodlr_matrix_diag(diag_exp, factors_exp, Az_fixed)
        loss = torch.nn.functional.mse_loss(res, z_fixed)
        
        loss.backward()
        optimizer.step()
        
        if i % 200 == 0:
            print(f"  [Step {i}] Loss: {loss.item():.6f}")
            
    print(f"  Overfit Training finished in {time.time()-t0:.2f}s. Final Loss: {loss.item():.6f}")
    return model.get_params()

# --- 3. Dense Reconstruction Helpers ---

LEAF_SIZE = 32

def get_dense_matrix_from_hodlr_blocks(leaf_blocks, factors, num_nodes, device, viz_limit=1000, leaf_size=32):
    """Reconstruct dense M from neural model output (leaf_blocks + factors). N must be divisible by leaf_size."""
    limit = min(num_nodes, viz_limit)
    # Align limit to leaf_size so block apply is valid
    limit = (limit // leaf_size) * leaf_size
    if limit == 0:
        limit = min(leaf_size, num_nodes)
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
            leaf_exp = leaf_blocks.expand(current_batch, -1, -1, -1)
            factors_exp = [(u.expand(current_batch, -1, -1), v.expand(current_batch, -1, -1))
                           for u, v in factors]
            y = apply_hodlr_matrix(leaf_exp, factors_exp, x, leaf_size=leaf_size)
            M[:, i:end] = y[:, :limit, 0].T.cpu()
    return M.numpy()


def get_dense_matrix_from_hodlr(diag, factors, num_nodes, device, viz_limit=1000):
    """Reconstruct dense M from overfit baseline (diag + factors)."""
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
            factors_exp = [(u.expand(current_batch, -1, -1), v.expand(current_batch, -1, -1))
                           for u, v in factors]
            y = apply_hodlr_matrix_diag(diag_exp, factors_exp, x)
            M[:, i:end] = y[:, :limit, 0].T.cpu()
    return M.numpy()

def get_dense_amg(A_sparse_scipy, viz_limit=1000):
    if not HAS_AMG: return np.eye(viz_limit)
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
        M_dense[:, i] = ml.solve(e_i, x0=x0, maxiter=1, cycle='V', tol=1e-10)[:limit]
        
    return M_dense

# --- 4. Main Logic ---

def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).parent
    default_data = script_dir.parent / "StreamingAssets" / "TestData"
    default_model = script_dir / "model_weights.bytes"
    
    parser.add_argument('--data_folder', type=str, default=str(default_data))
    parser.add_argument('--weights', type=str, default=str(default_model))
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--depth', type=int, default=5)
    # Note: --rank now means MAX_RANK
    parser.add_argument('--rank', type=int, default=32) 
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')

    # 1. Load Data
    print(f"Loading Data from {args.data_folder}")
    dataset = FluidGraphDataset([Path(args.data_folder)])
    batch = dataset[600]
    
    x = torch.from_numpy(batch['x']).unsqueeze(0).to(device)
    num_nodes = int(batch['num_nodes']) if hasattr(batch['num_nodes'], '__int__') else batch['num_nodes']
    A_indices = batch['edge_index'].to(device)
    A_values = batch['edge_values'].to(device)

    # Ensure N is padded to leaf_size (32) for block-diagonal apply
    leaf_size = LEAF_SIZE
    pad = (leaf_size - (num_nodes % leaf_size)) % leaf_size
    if pad > 0:
        num_nodes += pad
        x = F.pad(x, (0, 0, 0, pad))

    print(f"System N={num_nodes}")

    # Build Sparse A for Ground Truth; viz_limit aligned to leaf_size so A_viz and M_neural shapes match
    A_sparse_cpu = torch.sparse_coo_tensor(
        batch['edge_index'], batch['edge_values'], (num_nodes, num_nodes)
    ).coalesce()
    viz_limit = min(1000, (num_nodes // leaf_size) * leaf_size) or leaf_size
    viz_limit = (viz_limit // leaf_size) * leaf_size  # e.g. 1000 -> 992
    A_dense_full = A_sparse_cpu.to_dense().numpy()
    A_viz = A_dense_full[:viz_limit, :viz_limit]

    # --- MODEL 1: NEURAL HODLR (BlockDiagonalHead + factors) ---
    print("\n1. Running Neural HODLR...")
    model = NeuralHODLR(d_model=args.d_model, depth=args.depth, rank=args.rank, leaf_size=leaf_size).to(device)
    load_weights_from_bytes(model, Path(args.weights))
    
    with torch.no_grad():
        leaf_blocks_neural, factors_neural = model(x)
    
    M_neural = get_dense_matrix_from_hodlr_blocks(
        leaf_blocks_neural, factors_neural, num_nodes, device, viz_limit, leaf_size=leaf_size
    )

    # --- MODEL 2: OVERFIT HODLR ---
    print("\n2. Training Overfit HODLR...")
    # Pass args.rank as MAX_RANK
    diag_overfit, factors_overfit = train_overfit_baseline(
        A_indices, A_values, num_nodes, args.depth, args.rank, device
    )
    M_overfit = get_dense_matrix_from_hodlr(diag_overfit, factors_overfit, num_nodes, device, viz_limit)

    # --- MODEL 3: AMG ---
    print("\n3. Computing AMG...")
    row = batch['edge_index'][0].numpy()
    col = batch['edge_index'][1].numpy()
    data = batch['edge_values'].numpy().astype(np.float64)
    A_scipy = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    M_amg = get_dense_amg(A_scipy, viz_limit)

    # --- Visualization ---
    matrices = {
        "A (Input)": A_viz,
        "Neural M": M_neural,
        "Overfit M": M_overfit,
        "AMG M": M_amg
    }
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    methods = [("Neural", M_neural), ("Overfit", M_overfit), ("AMG", M_amg)]
    cond_A = np.linalg.cond(A_viz)
    print(f"\nCondition Number (Block A): {cond_A:.2e}")
    
    for idx, (name, M) in enumerate(methods):
        ax_m = axes[idx, 0]
        im = ax_m.imshow(np.log10(np.abs(M) + 1e-9), cmap='magma', aspect='auto')
        ax_m.set_title(f"{name} M (log10)")
        plt.colorbar(im, ax=ax_m)

        AM = A_viz @ M
        ax_am = axes[idx, 1]
        am_min, am_max = np.percentile(AM, [2, 98])
        if am_max - am_min < 1e-8: am_min, am_max = 0.0, 1.0
        im2 = ax_am.imshow(AM, cmap='RdBu_r', vmin=am_min, vmax=am_max, aspect='auto')
        plt.colorbar(im2, ax=ax_am)
        ax_am.set_title(f"{name} A·M")
        
        ax_d = axes[idx, 2]
        ax_d.plot(np.diag(AM), alpha=0.8)
        ax_d.axhline(1.0, color='r', linestyle='--')
        ax_d.set_ylim(-0.5, 2.0)
        ax_d.set_title(f"{name} Diag(A·M)")
        
        ax_t = axes[idx, 3]
        ax_t.axis('off')
        
        cond_AM = np.linalg.cond(AM)
        err_fro = np.linalg.norm(AM - np.eye(viz_limit)) / np.linalg.norm(np.eye(viz_limit))
        
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
    plt.show()

if __name__ == "__main__":
    main()