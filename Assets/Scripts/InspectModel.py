import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# Use non-interactive backend when no display (e.g. SSH) so plt.show() doesn't hang
if not os.environ.get('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import struct
import argparse
from pathlib import Path
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu
import time
import pyamg

# Import your existing components
from NeuralPreconditioner import NeuralHODLR, apply_hodlr_matrix, FluidGraphDataset, HODLRLoss

def load_weights_from_bytes(model, path):
    """Load NeuralHODLR weights from .bytes format (matches export_weights in NeuralPreconditioner)."""
    with open(path, 'rb') as f:
        header = f.read(28)
        _, _, d_model, nhead, depth, input_dim, rank = struct.unpack('<ffiiiii', header)

        def read_packed_tensor(target_param, transpose_check=False):
            num_elements = target_param.numel()
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
            reshaped = data_fp32.view(target_param.shape[1], target_param.shape[0]).t() if (transpose_check and len(target_param.shape) == 2) else data_fp32.view(target_param.shape)
            with torch.no_grad():
                target_param.copy_(reshaped)

        pack = lambda p, trans: read_packed_tensor(p, trans)
        pack(model.feature_proj.weight, True); pack(model.feature_proj.bias, False)
        for d in model.down_samplers:
            pack(d.local_mixer[0].weight, True); pack(d.local_mixer[0].bias, False)
            pack(d.local_mixer[2].weight, True); pack(d.local_mixer[2].bias, False)
            pack(d.norm.weight, False); pack(d.norm.bias, False)
        bn = model.bottleneck
        pack(bn.q_proj.weight, True); pack(bn.q_proj.bias, False); pack(bn.k_proj.weight, True); pack(bn.k_proj.bias, False)
        pack(bn.v_proj.weight, True); pack(bn.v_proj.bias, False); pack(bn.out_proj.weight, True); pack(bn.out_proj.bias, False)
        pack(bn.norm.weight, False); pack(bn.norm.bias, False)
        for up, hodlr_head in zip(model.up_samplers, model.hodlr_heads):
            pack(up.up_proj.weight, True); pack(up.up_proj.bias, False)
            pack(up.fusion[0].weight, True); pack(up.fusion[0].bias, False)
            pack(up.fusion[2].weight, True); pack(up.fusion[2].bias, False)
            pack(up.norm.weight, False); pack(up.norm.bias, False)
            pack(hodlr_head.proj_u.weight, True); pack(hodlr_head.proj_u.bias, False)
            pack(hodlr_head.proj_u.weight, True); pack(hodlr_head.proj_u.bias, False)  # U written twice
        pack(model.norm_out.weight, False); pack(model.norm_out.bias, False)
        pack(model.diag_head.net[0].weight, True); pack(model.diag_head.net[0].bias, False)
        pack(model.diag_head.net[2].weight, True); pack(model.diag_head.net[2].bias, False)

# --- 1. The Overfit Baseline Module ---
class DirectHODLR(nn.Module):
    """
    A standalone container for HODLR parameters (Diagonal + Factors)
    that we can optimize directly via Gradient Descent, bypassing the Neural Net.
    """
    def __init__(self, num_nodes, depth, rank, device):
        super().__init__()
        self.depth = depth
        # Learnable Diagonal (Initialize near 1.0)
        self.diag = nn.Parameter(torch.ones(1, num_nodes, 1, device=device))
        
        # Learnable Low-Rank Factors (Initialize small random)
        self.u_factors = nn.ParameterList()
        for i in range(depth):
            # Shape matches what the neural net outputs: [1, N, Rank]
            # We enforce symmetry by only learning U and setting V=U
            u = nn.Parameter(torch.randn(1, num_nodes, rank, device=device) * 0.001)
            self.u_factors.append(u)

    def forward(self, x):
        # Package factors as (U, U) tuples for the symmetric apply function
        factors = [(u, u) for u in self.u_factors]
        return apply_hodlr_matrix(self.diag, factors, x)
    
    def get_params(self):
        factors = [(u, u) for u in self.u_factors]
        return self.diag, factors

def train_overfit_baseline(A_indices, A_values, num_nodes, depth, rank, device, steps=400):
    print(f"\n--- Training Overfit HODLR Baseline ({steps} steps) ---")
    
    # 1. Setup Model
    model = DirectHODLR(num_nodes, depth, rank, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005) # Lower LR slightly
    
    # 2. CREATE FIXED PROBES (The Fix)
    # We use a larger batch (e.g., 64) to cover more of the vector space
    num_probes = 64
    z_fixed = torch.randn(num_probes, num_nodes, 1, device=device)
    
    # Pre-compute A * z_fixed since A is constant for this test.
    # Sparse matmul only runs on CPU (MPS/CUDA sparse mm can be unsupported).
    A_sparse = torch.sparse_coo_tensor(
        A_indices.cpu(), A_values.cpu(), (num_nodes, num_nodes), device='cpu'
    )
    z_flat_cpu = z_fixed.squeeze(-1).t().cpu()  # [N, B] on CPU
    Az_flat_cpu = torch.sparse.mm(A_sparse, z_flat_cpu)  # [N, B]
    Az_fixed = Az_flat_cpu.t().unsqueeze(-1).to(device)  # [B, N, 1]

    t0 = time.time()
    for i in range(steps):
        optimizer.zero_grad()
        
        diag, factors = model.get_params()
        # Expand to batch size (apply_hodlr_matrix expects diag/factors to match x batch dim)
        B = Az_fixed.shape[0]
        diag_exp = diag.expand(B, -1, -1)
        factors_exp = [(u.expand(B, -1, -1), v.expand(B, -1, -1)) for u, v in factors]

        # Apply HODLR to the PRE-COMPUTED (A*z)
        # We are solving M * (A*z) = z  => M = A^-1
        res = apply_hodlr_matrix(diag_exp, factors_exp, Az_fixed)
        
        loss = torch.nn.functional.mse_loss(res, z_fixed)
        
        loss.backward()
        optimizer.step()
        
        if i % 200 == 0:
            print(f"  [Step {i}] Loss: {loss.item():.6f}")
            
    print(f"  Overfit Training finished in {time.time()-t0:.2f}s. Final Loss: {loss.item():.6f}")
    return model.get_params()

# --- 2. Dense Reconstruction Helpers ---

def get_dense_matrix_from_hodlr(diag, factors, num_nodes, device, viz_limit=1000):
    """Reconstructs the dense matrix (or top-left block) from HODLR parameters."""
    limit = min(num_nodes, viz_limit)
    M = torch.zeros(limit, limit, device='cpu')
    
    # We only reconstruct the top-left 'limit' columns
    batch_size = 32
    with torch.no_grad():
        for i in range(0, limit, batch_size):
            end = min(i + batch_size, limit)
            current_batch = end - i
            
            # Create standard basis vectors e_i
            x = torch.zeros(current_batch, num_nodes, 1, device=device)
            indices = torch.arange(current_batch, device=device)
            rows = torch.arange(i, end, device=device)
            x[indices, rows, 0] = 1.0
            
            # Expand HODLR params
            diag_exp = diag.expand(current_batch, -1, -1)
            factors_exp = [(u.expand(current_batch, -1, -1), v.expand(current_batch, -1, -1)) 
                           for u, v in factors]
            
            # Apply
            y = apply_hodlr_matrix(diag_exp, factors_exp, x)
            
            # Store
            M[:, i:end] = y[:, :limit, 0].T.cpu()
            
    return M.numpy()

def get_dense_amg(A_sparse_scipy, viz_limit=1000):
    """Generates the dense AMG preconditioner matrix by applying the solver to Identity."""
    
    print("\n--- Computing AMG Baseline ---")
    # pyamg requires A, x, b to have the same dtype
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

# --- 3. Main Logic ---

def main():
    parser = argparse.ArgumentParser()
    # Hardcode paths or use arguments
    script_dir = Path(__file__).parent
    default_data = script_dir.parent / "StreamingAssets" / "TestData"
    default_model = script_dir / "model_weights.bytes"
    
    parser.add_argument('--data_folder', type=str, default=str(default_data))
    parser.add_argument('--weights', type=str, default=str(default_model))
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save plot image (default: script_dir/inspect_model_plot.png)')
    args = parser.parse_args()

    # Prefer GPU: CUDA (NVIDIA) then Metal (Apple MPS), else CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # 1. Load Data
    print(f"Loading Data from {args.data_folder}")
    dataset = FluidGraphDataset([Path(args.data_folder)])
    batch = dataset[600] # Use frame 600
    
    x = torch.from_numpy(batch['x']).unsqueeze(0).to(device)
    num_nodes = batch['num_nodes']
    A_indices = batch['edge_index'].to(device)
    A_values = batch['edge_values'].to(device)
    
    print(f"System N={num_nodes}")

    # Build Sparse A for Ground Truth
    A_sparse_cpu = torch.sparse_coo_tensor(batch['edge_index'], batch['edge_values'], (num_nodes, num_nodes)).coalesce()
    viz_limit = 1000
    A_dense_full = A_sparse_cpu.to_dense().numpy()
    A_viz = A_dense_full[:viz_limit, :viz_limit]

    # --- MODEL 1: NEURAL HODLR ---
    print("\n1. Running Neural HODLR...")
    model = NeuralHODLR(d_model=args.d_model, depth=args.depth, rank=args.rank).to(device)
    load_weights_from_bytes(model, Path(args.weights))
    
    with torch.no_grad():
        diag_neural, factors_neural = model(x)
    
    M_neural = get_dense_matrix_from_hodlr(diag_neural, factors_neural, num_nodes, device, viz_limit)

    # --- MODEL 2: OVERFIT HODLR (Theoretical Max) ---
    print("\n2. Training Overfit HODLR...")
    diag_overfit, factors_overfit = train_overfit_baseline(
        A_indices, A_values, num_nodes, args.depth, args.rank, device
    )
    M_overfit = get_dense_matrix_from_hodlr(diag_overfit, factors_overfit, num_nodes, device, viz_limit)

    # --- MODEL 3: AMG (Classical Standard) ---
    print("\n3. Computing AMG...")
    row = batch['edge_index'][0].numpy()
    col = batch['edge_index'][1].numpy()
    data = batch['edge_values'].numpy().astype(np.float64)
    A_scipy = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    M_amg = get_dense_amg(A_scipy, viz_limit)

    # --- Visualization & Metrics ---
    
    matrices = {
        "A (Input)": A_viz,
        "Neural M": M_neural,
        "Overfit M": M_overfit,
        "AMG M": M_amg
    }
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    
    # Loop through methods to compute A*M and Metrics
    methods = [("Neural", M_neural), ("Overfit", M_overfit), ("AMG", M_amg)]
    
    # Compute Condition Numbers (Approximate on viz block)
    cond_A = np.linalg.cond(A_viz)
    print(f"\nCondition Number (Block A): {cond_A:.2e}")
    
    for idx, (name, M) in enumerate(methods):
        # 1. Plot M
        ax_m = axes[idx, 0]
        im = ax_m.imshow(np.log10(np.abs(M) + 1e-9), cmap='magma', aspect='auto')
        ax_m.set_title(f"{name} M (log10)")
        plt.colorbar(im, ax=ax_m)

        # 2. Plot A*M (Identity check)
        AM = A_viz @ M  # Matrix product
        ax_am = axes[idx, 1]
        # Data-driven scale so structure is visible (ideal: diag=1, off-diag=0)
        am_min, am_max = np.percentile(AM, [2, 98])
        if am_max - am_min < 1e-8:
            am_min, am_max = AM.min(), AM.max()
            if am_max - am_min < 1e-8:
                am_min, am_max = 0.0, 1.0
        im2 = ax_am.imshow(AM, cmap='RdBu_r', vmin=am_min, vmax=am_max, aspect='auto')
        plt.colorbar(im2, ax=ax_am, label='A·M value')
        ax_am.set_title(f"{name} A·M")
        
        # 3. Plot Diagonal of A*M (Should be 1.0)
        ax_d = axes[idx, 2]
        ax_d.plot(np.diag(AM), alpha=0.8)
        ax_d.axhline(1.0, color='r', linestyle='--')
        ax_d.set_ylim(-0.5, 2.0)
        ax_d.set_title(f"{name} Diag(A·M)")
        
        # 4. Metrics Text
        ax_t = axes[idx, 3]
        ax_t.axis('off')
        
        # Metrics
        cond_AM = np.linalg.cond(AM)
        err_fro = np.linalg.norm(AM - np.eye(viz_limit)) / np.linalg.norm(np.eye(viz_limit))
        
        msg = [
            f"Method: {name}",
            f"Cond(AM): {cond_AM:.2e}",
            f"Improvement: {cond_A/cond_AM:.2f}x",
            f"Frobenius Err: {err_fro:.4f}"
        ]
        
        # Color code improvement
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