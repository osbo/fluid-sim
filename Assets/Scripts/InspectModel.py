#!/usr/bin/env python3
"""
InspectModel.py
Plots the true inverse of A, the reconstructed M from the model, and their difference.
All normalized by their respective means.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from pathlib import Path
import scipy.sparse as sp
import struct

# Import from NeuralPreconditioner
from NeuralPreconditioner import (
    SPAIGenerator, 
    FluidGraphDataset, 
    compute_laplacian_stencil
)

def load_weights_from_bytes(model, path):
    """
    Loads model weights from the packed .bytes file format.
    
    Format:
    - Header: 6 values (2 floats, 4 ints): [0.0, 0.0, d_model, num_heads, num_layers, input_dim]
    - Weights: Packed as float16 pairs in uint32 (two float16s per uint32)
    - 2D weights are transposed (Input-Major for HLSL), need to transpose back
    - Odd-length tensors are padded with 0.0
    """
    device = next(model.parameters()).device  # Get device from model
    
    with open(path, 'rb') as f:
        # Read header
        header = struct.unpack('<ffiiii', f.read(24))
        d_model, num_heads, num_layers, input_dim = header[2], header[3], header[4], header[5]
        
        print(f"Loading model: d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}, input_dim={input_dim}")
        
        def read_packed_tensor(shape, is_2d=False):
            """Read a tensor from packed float16 format."""
            total_elements = np.prod(shape)
            # Account for padding if odd
            packed_elements = (total_elements + 1) // 2
            packed_bytes = packed_elements * 4  # uint32 = 4 bytes
            buffer_data = f.read(packed_bytes)
            packed_data = np.frombuffer(buffer_data, dtype=np.uint32)
            
            # Unpack: each uint32 contains two float16s
            # Low 16 bits = first float, High 16 bits = second float
            float16_pairs = packed_data.view(np.float16)
            # Take only the elements we need (in case of padding)
            tensor_data = float16_pairs[:total_elements].astype(np.float32)
            
            # Reshape
            # For 2D weights: data was stored as [In, Out] (transposed from [Out, In])
            # So we need to reshape to [In, Out] first, then transpose back to [Out, In]
            if is_2d and len(shape) == 2:
                # Reshape to the transposed shape [In, Out]
                tensor = tensor_data.reshape(shape[1], shape[0])
                # Transpose back to original shape [Out, In]
                tensor = tensor.T
            else:
                tensor = tensor_data.reshape(shape)
            
            # Move to device and return
            return torch.from_numpy(tensor).to(device)
        
        # 1. Feature Projection
        expected_shape = model.feature_proj.weight.shape
        loaded_weight = read_packed_tensor(expected_shape, is_2d=True)
        print(f"Feature proj weight - expected: {expected_shape}, loaded: {loaded_weight.shape}")
        model.feature_proj.weight.data = loaded_weight
        model.feature_proj.bias.data = read_packed_tensor(
            model.feature_proj.bias.shape
        )
        model.layer_embed.weight.data = read_packed_tensor(
            model.layer_embed.weight.shape, is_2d=True
        )
        model.window_pos_embed.data = read_packed_tensor(
            model.window_pos_embed.shape
        )
        
        # 2. Transformer Layers
        for layer in model.encoder.layers:
            layer.self_attn.in_proj_weight.data = read_packed_tensor(
                layer.self_attn.in_proj_weight.shape, is_2d=True
            )
            layer.self_attn.in_proj_bias.data = read_packed_tensor(
                layer.self_attn.in_proj_bias.shape
            )
            layer.self_attn.out_proj.weight.data = read_packed_tensor(
                layer.self_attn.out_proj.weight.shape, is_2d=True
            )
            layer.self_attn.out_proj.bias.data = read_packed_tensor(
                layer.self_attn.out_proj.bias.shape
            )
            layer.norm1.weight.data = read_packed_tensor(
                layer.norm1.weight.shape
            )
            layer.norm1.bias.data = read_packed_tensor(
                layer.norm1.bias.shape
            )
            layer.linear1.weight.data = read_packed_tensor(
                layer.linear1.weight.shape, is_2d=True
            )
            layer.linear1.bias.data = read_packed_tensor(
                layer.linear1.bias.shape
            )
            layer.linear2.weight.data = read_packed_tensor(
                layer.linear2.weight.shape, is_2d=True
            )
            layer.linear2.bias.data = read_packed_tensor(
                layer.linear2.bias.shape
            )
            layer.norm2.weight.data = read_packed_tensor(
                layer.norm2.weight.shape
            )
            layer.norm2.bias.data = read_packed_tensor(
                layer.norm2.bias.shape
            )
        
        # 3. Output Head
        model.norm_out.weight.data = read_packed_tensor(
            model.norm_out.weight.shape
        )
        model.norm_out.bias.data = read_packed_tensor(
            model.norm_out.bias.shape
        )
        model.head.weight.data = read_packed_tensor(
            model.head.weight.shape, is_2d=True
        )
        model.head.bias.data = read_packed_tensor(
            model.head.bias.shape
        )
    
    print("Model weights loaded successfully from .bytes file")

def build_matrix_A_from_stencil(A_diag, A_off, neighbors, num_nodes):
    """
    Builds a dense matrix A from the stencil representation.
    
    Args:
        A_diag: [N, 1] diagonal entries
        A_off: [N, 24] off-diagonal entries
        neighbors: [N, 24] neighbor indices
        num_nodes: number of nodes
    
    Returns:
        A_dense: [N, N] dense matrix
    """
    N = num_nodes
    A_dense = np.zeros((N, N), dtype=np.float32)
    
    # Set diagonal
    A_dense[np.arange(N), np.arange(N)] = A_diag.squeeze()
    
    # Set off-diagonals
    for i in range(N):
        for k in range(24):
            j = neighbors[i, k]
            if j < N:  # Valid fluid neighbor
                A_dense[i, j] = A_off[i, k]
    
    return A_dense

def compute_M_from_model(model, x, layers, neighbors, num_nodes, device, epsilon=1e-4):
    """
    Computes M = G * G^T + epsilon * I from the model output.
    
    Args:
        model: SPAIGenerator model
        x: [1, N, 58] input features
        layers: [1, N] layer indices
        neighbors: [N, 24] neighbor indices
        num_nodes: number of nodes
        device: torch device
        epsilon: regularization parameter
    
    Returns:
        M_dense: [N, N] dense matrix M
    """
    model.eval()
    with torch.no_grad():
        # Verify input shapes
        print(f"Model input shapes - x: {x.shape}, layers: {layers.shape}")
        # Forward pass
        G_coeffs = model(x, layers)  # [1, N, 25]
        G_coeffs = G_coeffs[0]  # [N, 25]
    
    N = num_nodes
    device = G_coeffs.device
    
    # Build sparse G matrix
    # G_coeffs[:, 0] is diagonal, G_coeffs[:, 1:] are off-diagonals
    G_diag = G_coeffs[:, 0]  # [N]
    G_off = G_coeffs[:, 1:]  # [N, 24]
    
    # Build G as a dense matrix
    G_dense = torch.zeros(N, N, device=device, dtype=torch.float32)
    
    # Set diagonal
    G_dense[np.arange(N), np.arange(N)] = G_diag
    
    # Set off-diagonals
    neighbors_t = torch.from_numpy(neighbors).long().to(device)
    for i in range(N):
        for k in range(24):
            j = neighbors_t[i, k].item()
            if j < N:
                G_dense[i, j] = G_off[i, k]
    
    # Compute M = G * G^T + epsilon * I
    M_dense = torch.mm(G_dense, G_dense.t()) + epsilon * torch.eye(N, device=device)
    
    return M_dense.cpu().numpy()

def main():
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Load dataset
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / "StreamingAssets" / "TestData"
    
    if not data_path.exists():
        print(f"Error: Data path not found at {data_path}")
        return
    
    dataset = FluidGraphDataset([data_path])
    if len(dataset) == 0:
        print("No data found!")
        return
    
    # Get frame at 1/4 of the way through
    frame_idx = len(dataset) // 4
    print(f"Loading frame {frame_idx} / {len(dataset)} ({frame_idx/len(dataset)*100:.1f}%)")
    
    # Load frame data
    frame_data = dataset[frame_idx]
    
    # Extract unpadded data
    x = frame_data['x']  # [max_nodes, 58]
    layers = frame_data['layers']  # [max_nodes]
    neighbors = frame_data['neighbors']  # [max_nodes, 24]
    A_diag = frame_data['A_diag']  # [max_nodes, 1]
    A_off = frame_data['A_off']  # [max_nodes, 24]
    mask = frame_data['mask']  # [max_nodes, 1]
    
    # Find actual number of nodes
    num_nodes = int(mask.sum())
    print(f"Number of nodes: {num_nodes}")
    
    # Extract unpadded slices
    x_unpadded = x[:num_nodes]
    layers_unpadded = layers[:num_nodes]
    neighbors_unpadded = neighbors[:num_nodes]
    A_diag_unpadded = A_diag[:num_nodes]
    A_off_unpadded = A_off[:num_nodes]
    
    # Build full matrix A
    print("Building matrix A from stencil...")
    A_dense = build_matrix_A_from_stencil(
        A_diag_unpadded, 
        A_off_unpadded, 
        neighbors_unpadded, 
        num_nodes
    )
    
    # Compute true inverse
    print("Computing true inverse of A...")
    A_inv_true = la.inv(A_dense.astype(np.float64)).astype(np.float32)
    
    # Load model
    model_path = script_dir / "model_weights.bytes"
    if not model_path.exists():
        print(f"Error: Model weights not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}...")
    model = SPAIGenerator(input_dim=58, d_model=32, num_layers=4, nhead=4, window_size=256).to(device)
    
    # Load weights from .bytes file
    try:
        load_weights_from_bytes(model, model_path)
    except Exception as e:
        print(f"Error: Could not load model weights from .bytes file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Prepare model input
    x_tensor = torch.from_numpy(x_unpadded).unsqueeze(0).to(device)  # [1, N, 58]
    layers_tensor = torch.from_numpy(layers_unpadded).long().unsqueeze(0).to(device)  # [1, N]
    
    # Verify shapes
    print(f"Input shapes - x_tensor: {x_tensor.shape}, layers_tensor: {layers_tensor.shape}")
    
    # Compute M from model
    print("Computing M from model...")
    M_dense = compute_M_from_model(
        model, x_tensor, layers_tensor, 
        neighbors_unpadded, num_nodes, device
    )
    
    # Normalize by mean (only over non-zero elements for sparse matrices)
    A_inv_nonzero = np.abs(A_inv_true[A_inv_true != 0])
    M_nonzero = np.abs(M_dense[M_dense != 0])
    
    mean_A_inv = np.mean(A_inv_nonzero) if len(A_inv_nonzero) > 0 else 1.0
    mean_M = np.mean(M_nonzero) if len(M_nonzero) > 0 else 1.0
    
    A_inv_normalized = A_inv_true / (mean_A_inv + 1e-10)
    M_normalized = M_dense / (mean_M + 1e-10)
    
    # Compute difference (each normalized by their own mean)
    diff_normalized = A_inv_normalized - M_normalized
    
    # Plot
    print("Plotting...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: True Inverse (normalized)
    im1 = axes[0].imshow(
        np.abs(A_inv_normalized), 
        cmap='inferno', 
        interpolation='none',
        norm=plt.matplotlib.colors.LogNorm(vmin=1e-3, vmax=1e3)
    )
    axes[0].set_title(f"True Inverse $A^{{-1}}$ (Normalized by Non-Zero Mean={mean_A_inv:.2e})")
    plt.colorbar(im1, ax=axes[0])
    
    # Plot 2: Reconstructed M (normalized)
    im2 = axes[1].imshow(
        np.abs(M_normalized), 
        cmap='inferno', 
        interpolation='none',
        norm=plt.matplotlib.colors.LogNorm(vmin=1e-3, vmax=1e3)
    )
    axes[1].set_title(f"Reconstructed M (Normalized by Non-Zero Mean={mean_M:.2e})")
    plt.colorbar(im2, ax=axes[1])
    
    # Plot 3: Difference (normalized)
    # Use symmetric colormap for difference
    vmax_diff = np.max(np.abs(diff_normalized))
    im3 = axes[2].imshow(
        diff_normalized, 
        cmap='RdBu_r', 
        interpolation='none',
        vmin=-vmax_diff, 
        vmax=vmax_diff
    )
    axes[2].set_title(f"Difference (Each Normalized by Own Mean)")
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    nnz_A_inv = np.count_nonzero(A_inv_true)
    nnz_M = np.count_nonzero(M_dense)
    sparsity_A_inv = 1.0 - (nnz_A_inv / (num_nodes * num_nodes))
    sparsity_M = 1.0 - (nnz_M / (num_nodes * num_nodes))
    
    print(f"\n{'='*60}")
    print("Statistics")
    print(f"{'='*60}")
    print(f"Matrix size: {num_nodes} x {num_nodes}")
    print(f"\nTrue Inverse A^-1:")
    print(f"  Non-zero elements: {nnz_A_inv} ({sparsity_A_inv*100:.2f}% sparse)")
    print(f"  Mean absolute value (non-zero only): {mean_A_inv:.6e}")
    print(f"  Min (non-zero): {np.min(A_inv_nonzero) if len(A_inv_nonzero) > 0 else 0.0:.6e}")
    print(f"  Max: {np.max(np.abs(A_inv_true)):.6e}")
    print(f"\nReconstructed M:")
    print(f"  Non-zero elements: {nnz_M} ({sparsity_M*100:.2f}% sparse)")
    print(f"  Mean absolute value (non-zero only): {mean_M:.6e}")
    print(f"  Min (non-zero): {np.min(M_nonzero) if len(M_nonzero) > 0 else 0.0:.6e}")
    print(f"  Max: {np.max(np.abs(M_dense)):.6e}")
    print(f"\nDifference (normalized):")
    print(f"  Mean absolute error: {np.mean(np.abs(diff_normalized)):.6e}")
    print(f"  Max absolute error: {np.max(np.abs(diff_normalized)):.6e}")
    print(f"  Relative error: {np.linalg.norm(diff_normalized) / np.linalg.norm(A_inv_normalized):.6e}")

if __name__ == "__main__":
    main()
