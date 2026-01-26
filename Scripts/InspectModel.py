#!/usr/bin/env python3
"""
inspectModel.py
Loads a trained NeuralPreconditioner model (from .bytes format), runs a single frame,
and visualizes the learned matrix G and preconditioner M = G*G^T.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import struct
import argparse
from pathlib import Path
from NeuralPreconditioner import HierarchicalVCycleTransformer, FluidGraphDataset

def load_weights_from_bytes(model, path):
    """
    Reverse engineers the export_weights function to load .bytes files.
    """
    print(f"Loading weights from {path}...")
    with open(path, 'rb') as f:
        # 1. Read Header
        # Format: [reserved1, reserved2, d_model, nhead, num_levels, input_dim]
        header_data = f.read(24) # 2 floats (8) + 4 ints (16) = 24 bytes
        _, _, d_model, nhead, num_levels, input_dim = struct.unpack('<ffiiii', header_data)
        
        # Verify architecture matches
        if d_model != model.d_model:
            print(f"Warning: File d_model ({d_model}) != Model d_model ({model.d_model})")
        
        # 2. Helper to read packed tensor
        def read_packed_tensor(target_param, transpose_check=False):
            # Calculate number of elements
            num_elements = target_param.numel()
            
            # The export padded odd-length arrays to be even
            read_len = num_elements
            if num_elements % 2 != 0:
                read_len += 1
                
            # Calculate bytes: 2 elements per 4-byte uint32 (packed fp16)
            # effectively 2 bytes per element
            bytes_to_read = read_len * 2 
            
            buffer = f.read(bytes_to_read)
            if len(buffer) != bytes_to_read:
                raise ValueError(f"Unexpected EOF. Wanted {bytes_to_read} bytes, got {len(buffer)}")
            
            # Read as uint32 (packed halfs)
            packed = np.frombuffer(buffer, dtype=np.uint32)
            
            # Unpack into float16
            data_fp16 = packed.view(np.float16)
            
            # Remove padding if it existed
            if num_elements % 2 != 0:
                data_fp16 = data_fp16[:-1]
                
            # Convert to float32
            data_fp32 = torch.from_numpy(data_fp16.astype(np.float32)).to(target_param.device)
            
            # Reshape
            # Note: Export transposed 2D weights. We must transpose back.
            if len(target_param.shape) == 2 and transpose_check:
                # Target is [Out, In], Export was [In, Out]
                # We read flat data, reshape to [In, Out], then transpose to [Out, In]
                reshaped = data_fp32.view(target_param.shape[1], target_param.shape[0]).t()
            else:
                reshaped = data_fp32.view(target_param.shape)
                
            # Copy into parameter
            with torch.no_grad():
                target_param.copy_(reshaped)

        # --- Load Weights in Exact Order of Export ---
        
        # 1. Feature Projection
        read_packed_tensor(model.feature_proj.weight, transpose_check=True)
        read_packed_tensor(model.feature_proj.bias)
        read_packed_tensor(model.layer_embed.weight, transpose_check=True)

        # 2. Down-Samplers
        for down in model.down_samplers:
            read_packed_tensor(down.local_mixer[0].weight, transpose_check=True)
            read_packed_tensor(down.local_mixer[0].bias)
            read_packed_tensor(down.local_mixer[2].weight, transpose_check=True)
            read_packed_tensor(down.local_mixer[2].bias)
            read_packed_tensor(down.norm.weight)
            read_packed_tensor(down.norm.bias)

        # 3. Bottleneck
        bn = model.bottleneck
        read_packed_tensor(bn.q_proj.weight, transpose_check=True)
        read_packed_tensor(bn.q_proj.bias)
        read_packed_tensor(bn.k_proj.weight, transpose_check=True)
        read_packed_tensor(bn.k_proj.bias)
        read_packed_tensor(bn.v_proj.weight, transpose_check=True)
        read_packed_tensor(bn.v_proj.bias)
        read_packed_tensor(bn.out_proj.weight, transpose_check=True)
        read_packed_tensor(bn.out_proj.bias)
        read_packed_tensor(bn.norm.weight)
        read_packed_tensor(bn.norm.bias)

        # 4. Up-Samplers
        for up in model.up_samplers:
            read_packed_tensor(up.up_proj.weight, transpose_check=True)
            read_packed_tensor(up.up_proj.bias)
            read_packed_tensor(up.fusion[0].weight, transpose_check=True)
            read_packed_tensor(up.fusion[0].bias)
            read_packed_tensor(up.fusion[2].weight, transpose_check=True)
            read_packed_tensor(up.fusion[2].bias)
            read_packed_tensor(up.norm.weight)
            read_packed_tensor(up.norm.bias)

        # 5. Output Head
        read_packed_tensor(model.norm_out.weight)
        read_packed_tensor(model.norm_out.bias)
        read_packed_tensor(model.head.weight, transpose_check=True)
        read_packed_tensor(model.head.bias)

    print("Weights loaded successfully.")

def reconstruct_dense_matrices(G_coeffs, A_diag, A_off, neighbors, valid_mask):
    """
    Reconstructs dense numpy matrices from sparse tensors for visualization.
    Only considers 'valid' nodes (removes padding).
    """
    # Identify valid nodes
    num_nodes = int(valid_mask.sum().item())
    
    # Extract valid data (CPU)
    coeffs = G_coeffs[0, :num_nodes].detach().cpu().numpy() # [N, 25]
    nbrs = neighbors[0, :num_nodes].cpu().numpy().astype(int) # [N, 24]
    
    A_d = A_diag[0, :num_nodes].cpu().numpy() # [N, 1]
    A_o = A_off[0, :num_nodes].cpu().numpy() # [N, 24]
    
    # 1. Build Dense G (Lower Triangular)
    G = np.zeros((num_nodes, num_nodes))
    
    # Fill Diagonal
    np.fill_diagonal(G, coeffs[:, 0])
    
    # Fill Off-Diagonals
    # Row i, Column nbrs[i, k]
    for i in range(num_nodes):
        for k in range(24):
            col = nbrs[i, k]
            # Valid neighbor check (valid index and not Air/Wall)
            if 0 <= col < num_nodes:
                val = coeffs[i, k + 1] # +1 because 0 is diag
                G[i, col] = val
                
    # 2. Build Dense A (Symmetric Laplacian)
    A = np.zeros((num_nodes, num_nodes))
    np.fill_diagonal(A, A_d[:, 0])
    
    for i in range(num_nodes):
        for k in range(24):
            col = nbrs[i, k]
            if 0 <= col < num_nodes:
                val = A_o[i, k]
                A[i, col] = val
                # A is symmetric, ensure we fill both if the neighbor relation isn't perfectly symmetric in data
                # (Ideally the data is symmetric, but let's trust the row-wise definition)
    
    return G, A

def main():
    parser = argparse.ArgumentParser()
    # Get script directory (Assets/Scripts/)
    script_dir = Path(__file__).parent
    
    # Default paths relative to script directory
    default_data_path = script_dir.parent / "StreamingAssets" / "TestData"
    default_weights_path = script_dir / "model_weights.bytes"
    
    parser.add_argument('--data_folder', type=str, default=str(default_data_path))
    parser.add_argument('--weights', type=str, default=str(default_weights_path), help='Path to model weights file')
    args = parser.parse_args()

    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')
    print(f"Using device: {device}")

    # 2. Load Model Structure
    # Note: Params must match training. If you changed d_model/k, update here.
    model = HierarchicalVCycleTransformer(input_dim=58, d_model=32, nhead=4, k=32).to(device)
    
    # 3. Load Weights - resolve path relative to script if not absolute
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = script_dir / weights_path
    
    if not weights_path.exists():
        print(f"Error: Weights file not found at {weights_path}")
        print(f"  (resolved from: {args.weights})")
        return
    load_weights_from_bytes(model, weights_path)
    model.eval()

    # 4. Load ONE sample frame - resolve path relative to script if not absolute
    data_path = Path(args.data_folder)
    if not data_path.is_absolute():
        data_path = script_dir.parent / "StreamingAssets" / "TestData"
    
    dataset = FluidGraphDataset([data_path])
    if len(dataset) == 0:
        print(f"No data found at {data_path}")
        return
        
    # Get first frame
    print("Loading frame 0...")
    batch = dataset[0]
    
    # Prepare inputs (add batch dim)
    x = torch.from_numpy(batch['x']).unsqueeze(0).to(device)
    layers = torch.from_numpy(batch['layers']).unsqueeze(0).to(device)
    nbrs = torch.from_numpy(batch['neighbors']).unsqueeze(0).to(device)
    A_diag = torch.from_numpy(batch['A_diag']).unsqueeze(0).to(device)
    A_off = torch.from_numpy(batch['A_off']).unsqueeze(0).to(device)
    mask = torch.from_numpy(batch['mask']).unsqueeze(0).to(device)

    # 5. Run Model
    with torch.no_grad():
        G_coeffs = model(x, layers)

    # 6. Reconstruct Dense Matrices
    print("Reconstructing dense matrices (this may take a moment for large N)...")
    G, A = reconstruct_dense_matrices(G_coeffs, A_diag, A_off, nbrs, mask)
    
    # 7. Compute Derived Matrices
    M = G @ G.T
    AM = A @ M # We want this to be close to Identity * scale
    
    # Normalize A and M for visualization scale comparison
    # (Optional, but helps to see structure)
    
    # 8. Statistics
    G_diag = np.diag(G)
    G_off = G.copy()
    np.fill_diagonal(G_off, 0)
    
    avg_diag = np.mean(np.abs(G_diag))
    avg_off = np.mean(np.abs(G_off[G_off != 0])) # Mean of non-zero off-diagonals
    max_off = np.max(np.abs(G_off))
    
    print("-" * 30)
    print("MATRIX STATISTICS")
    print("-" * 30)
    print(f"Matrix Size (N): {G.shape[0]}")
    print(f"G Diagonal Mean: {avg_diag:.6f}")
    print(f"G Off-Diag Mean: {avg_off:.6f} (non-zeros)")
    print(f"G Max Off-Diag:  {max_off:.6f}")
    print(f"Ratio (Off/Diag): {avg_off / avg_diag * 100:.2f}%")
    
    # Check if off-diagonals are effectively zero
    if avg_off < 1e-5:
        print("\n[ALERT] The model is effectively learning a DIAGONAL matrix.")
    else:
        print("\n[OK] The model is learning OFF-DIAGONAL structure.")
    
    # 9. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Helper to plot subsample if matrix is too huge
    plot_N = min(G.shape[0], 200) # Only plot top-left 200x200 for clarity
    
    # A (Physics)
    ax = axes[0, 0]
    im = ax.imshow(A[:plot_N, :plot_N], cmap='seismic', vmin=-100, vmax=100) # Clamp for visibility
    ax.set_title(f"Physics Matrix A (Top-Left {plot_N}x{plot_N})")
    plt.colorbar(im, ax=ax)

    # G (Learned Factor)
    ax = axes[0, 1]
    im = ax.imshow(G[:plot_N, :plot_N], cmap='viridis')
    ax.set_title("Learned Factor G (Lower Triangular)")
    plt.colorbar(im, ax=ax)

    # M (Preconditioner)
    ax = axes[1, 0]
    im = ax.imshow(M[:plot_N, :plot_N], cmap='viridis')
    ax.set_title("Preconditioner M = G * G^T")
    plt.colorbar(im, ax=ax)

    # A * M (Target = Identity)
    ax = axes[1, 1]
    prod = AM[:plot_N, :plot_N]
    # Normalize product for visualization so Identity is visible
    # We expect roughly scaled identity
    im = ax.imshow(prod, cmap='magma') 
    ax.set_title("A * M (Should look like Diagonal/Identity)")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

    # 10. Histogram of Coeffs
    plt.figure(figsize=(10, 5))
    coeffs_flat = G_coeffs[0, :int(mask.sum()), 1:].cpu().numpy().flatten() # Off-diagonals
    coeffs_diag = G_coeffs[0, :int(mask.sum()), 0].cpu().numpy().flatten()  # Diagonals
    
    plt.hist(coeffs_diag, bins=50, alpha=0.7, label='Diagonal', log=True)
    plt.hist(coeffs_flat, bins=50, alpha=0.7, label='Off-Diagonal', log=True)
    plt.title("Distribution of Learned Coefficients")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()