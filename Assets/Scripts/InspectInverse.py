#!/usr/bin/env python3
"""
InspectInverse.py

1. Constructs the exact Matrix A from a simulation frame.
2. Computes the dense Inverse A^-1.
3. Analyzes the structure of A^-1 (Decay vs Distance).
4. Finds the 'Theoretical Ceiling' of the fixed sparsity pattern by 
   overfitting a G matrix to this specific frame.
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import scipy.linalg
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time

# --- 0. UTILITY FUNCTIONS ---

def smooth_losses(losses, window_size=4):
    """
    Smooth loss array by averaging over a moving window.
    window_size: Number of points to average (default 4).
    """
    if len(losses) < window_size:
        return losses
    
    smoothed = []
    for i in range(len(losses)):
        start = max(0, i - window_size // 2)
        end = min(len(losses), i + window_size // 2 + 1)
        smoothed.append(np.mean(losses[start:end]))
    
    return np.array(smoothed)

def compute_nnz_per_row_cdf(indices, num_nodes):
    """
    Compute the CDF of number of nonzeros per row.
    Returns: (nnz_values, cdf_percentages)
    where cdf_percentages[i] is the percentage of rows with <= nnz_values[i] nonzeros.
    """
    # Count nonzeros per row
    row_counts = torch.zeros(num_nodes, dtype=torch.long, device=indices.device)
    row_indices = indices[0]  # First row of indices contains row numbers
    row_counts.scatter_add_(0, row_indices, torch.ones_like(row_indices))
    
    # Convert to numpy for easier processing
    counts_np = row_counts.cpu().numpy()
    
    # Compute CDF
    max_nnz = int(counts_np.max())
    nnz_values = np.arange(0, max_nnz + 1)
    cdf = np.array([np.sum(counts_np <= nnz) / num_nodes * 100 for nnz in nnz_values])
    
    return nnz_values, cdf

# --- 1. LOADER UTILITIES (Adapted from your existing scripts) ---

def compute_laplacian_stencil(neighbors, layers, num_nodes):
    """Recomputes the physics stencil values exactly as the C# solver does."""
    device = neighbors.device
    N = num_nodes
    IDX_AIR = N
    IDX_WALL = N + 1
    
    dx = torch.pow(2.0, layers.float()).view(N, 1, 1)
    safe_nbrs = neighbors.long().clone()
    safe_nbrs[safe_nbrs >= N] = 0
    layers_flat = layers.view(-1)
    nbr_layers = layers_flat[safe_nbrs]
    dx_nbrs = torch.pow(2.0, nbr_layers.float()).view(N, 6, 4)
    
    raw_nbrs = neighbors.view(N, 6, 4)
    is_wall = (raw_nbrs == IDX_WALL)
    is_air  = (raw_nbrs == IDX_AIR)
    is_fluid = (raw_nbrs < N)
    
    self_layer_exp = layers.view(N, 1, 1).expand(N, 6, 4)
    is_coarse_or_same = is_fluid & (nbr_layers.view(N, 6, 4) >= self_layer_exp)
    slot0_active = is_coarse_or_same[:, :, 0]
    
    dist_c = torch.maximum(0.5 * (dx + dx_nbrs), torch.tensor(1e-6, device=device))
    weight_c = (dx**2) / dist_c
    
    dx_target = dx_nbrs.clone()
    dx_target[is_air] = dx.expand(N, 6, 4)[is_air] * 0.5 
    dist_s = torch.maximum(0.5 * (dx + dx_target), torch.tensor(1e-6, device=device))
    weight_s = (dx**2 / 4.0) / dist_s
    
    A_off = torch.zeros(N, 6, 4, device=device)
    mask_c = slot0_active.unsqueeze(2).expand(N, 6, 4).clone()
    mask_c[:, :, 1:] = False
    A_off[mask_c] = -weight_c[mask_c]
    mask_sub = (~slot0_active.unsqueeze(2).expand(N, 6, 4)) & (~is_wall)
    A_off[mask_sub] = -weight_s[mask_sub]
    
    A_off_flat = A_off.view(N, 24)
    A_diag = torch.sum(torch.abs(A_off_flat), dim=1, keepdim=True)
    A_off_flat[neighbors == IDX_AIR] = 0.0
    
    return A_diag, A_off_flat

def load_frame_data(frame_path):
    path = Path(frame_path)
    num_nodes = 0
    with open(path / "meta.txt", 'r') as f:
        for line in f:
            if 'numNodes' in line: num_nodes = int(line.split(':')[1].strip())
            
    def read_bin(name, dtype):
        return np.fromfile(path / name, dtype=dtype)
        
    node_dtype = np.dtype([('position', '3<f4'), ('velocity', '3<f4'), ('face_vels', '6<f4'), ('mass', '<f4'), ('layer', '<u4'), ('morton', '<u4'), ('active', '<u4')])
    
    raw_nodes = read_bin("nodes.bin", node_dtype)
    raw_nbrs = read_bin("neighbors.bin", np.uint32).reshape(num_nodes, 24)
    
    pos = torch.from_numpy(raw_nodes['position'][:num_nodes] / 1024.0)
    layers = torch.from_numpy(raw_nodes['layer'][:num_nodes].astype(np.int64)).unsqueeze(1)
    neighbors = torch.from_numpy(raw_nbrs.astype(np.int64))
    
    # Calculate A
    A_diag, A_off = compute_laplacian_stencil(neighbors, layers, num_nodes)
    
    return pos, layers, neighbors, A_diag, A_off, num_nodes

def build_scipy_A(A_diag, A_off, neighbors, num_nodes):
    """Constructs the exact sparse Matrix A (scipy.sparse.csr_matrix)."""
    rows = []
    cols = []
    data = []
    
    # Diagonal
    for i in range(num_nodes):
        rows.append(i)
        cols.append(i)
        data.append(A_diag[i].item())
        
    # Off-diagonal
    A_off_np = A_off.numpy()
    neighbors_np = neighbors.numpy()
    
    for i in range(num_nodes):
        for k in range(24):
            nbr = neighbors_np[i, k]
            val = A_off_np[i, k]
            if nbr < num_nodes and abs(val) > 1e-9:
                rows.append(i)
                cols.append(nbr)
                data.append(val)
                
    return sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

# --- 2. SPARSITY PATTERN HELPER ---

def get_sparsity_pattern(neighbors, num_nodes, degree=1):
    """
    Computes the indices [2, E] for a K-hop sparsity pattern.
    degree=1: Immediate neighbors (matches current script).
    degree=2: Neighbors + Neighbors of Neighbors.
    """
    device = neighbors.device
    N = num_nodes
    
    # 1. Convert fixed [N, 24] neighbors to sparse indices [2, E]
    # Filter out invalid neighbors (>= N)
    mask = neighbors < N
    row_idx = torch.arange(N, device=device).view(N, 1).expand(N, 24)
    
    src = row_idx[mask]
    dst = neighbors[mask]
    
    # Create the Degree 1 adjacency matrix (Structure only)
    # Value 1.0 indicates a connection
    indices = torch.stack([src, dst], dim=0)
    values = torch.ones(indices.shape[1], device=device)
    
    adj = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()
    
    # 2. Compute Higher Degrees via Matrix Power
    # (A + I)^k captures all nodes reachable in k steps
    
    # Start with Identity (Degree 0)
    eye_idx = torch.arange(N, device=device).unsqueeze(0).repeat(2, 1)
    eye_val = torch.ones(N, device=device)
    reached = torch.sparse_coo_tensor(eye_idx, eye_val, (N, N)).coalesce()
    
    # Add Degree 1
    reached = (reached + adj).coalesce()
    
    # Iterate for higher degrees
    curr_adj = adj
    for _ in range(degree - 1):
        # Symbolic matmul: adj_new = adj @ curr_adj
        curr_adj = torch.sparse.mm(adj, curr_adj).coalesce()
        reached = (reached + curr_adj).coalesce()
        
    # 3. Extract Indices and Remove Diagonal
    # (Diagonal is handled separately by G_diag parameter)
    final_indices = reached.indices()
    
    # Mask out self-loops (row == col)
    row, col = final_indices[0], final_indices[1]
    mask_off_diag = row != col
    
    off_diag_indices = final_indices[:, mask_off_diag]
    
    print(f"Sparsity Pattern (Degree {degree}): {off_diag_indices.shape[1]} edges "
          f"(~{off_diag_indices.shape[1]/N:.1f} per node)")
            
    return off_diag_indices

# --- 3. OPTIMIZATION ROUTINE ---

def find_optimal_sparse_G_generic(A_diag, A_off, neighbors, num_nodes, indices, device='cuda', steps=2000):
    """
    Generic optimization for ANY sparsity pattern defined by `indices`.
    indices: [2, E] LongTensor of (row, col) coordinates for learnable weights.
    """
    print(f"\n--- Optimizing generic G (Edges: {indices.shape[1]}) ---")
    
    # 1. Setup Fixed Physics (Matrix A)
    # We still use the custom kernel for A because it's already built and fast
    t_neighbors = neighbors.to(device)
    t_A_diag = A_diag.to(device)
    t_A_off = A_off.to(device)
    norm_A = (torch.abs(t_A_diag).sum() + torch.abs(t_A_off).sum()) / (num_nodes * 25)
    
    def apply_A(x):
        # Use existing efficient kernel for A
        coeffs_d = t_A_diag.unsqueeze(0)
        coeffs_o = t_A_off.unsqueeze(0)
        
        # Inline the gather logic briefly or reuse the one from script
        # (Assuming reuse of simple gather logic from original script for A)
        res = x * coeffs_d
        B, N, _ = x.shape
        safe_nbrs = t_neighbors.clone(); safe_nbrs[safe_nbrs >= N] = 0
        batch_idx = torch.arange(B, device=device).view(B, 1, 1) * N
        flat_idx = (safe_nbrs + batch_idx).view(-1)
        x_flat = x.view(-1, 1)
        x_nbrs = x_flat[flat_idx].view(B, N, 24)
        mask = (t_neighbors < N).float().unsqueeze(0)
        res += (x_nbrs * coeffs_o * mask).sum(dim=2, keepdim=True)
        return res

    # 2. Setup Learnable G (Sparse)
    # G_diag: [N, 1]
    # G_off_val: [E] (The values corresponding to 'indices')
    G_diag = torch.ones(num_nodes, 1, device=device, requires_grad=True)
    G_off_val = torch.zeros(indices.shape[1], device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([G_diag, G_off_val], lr=0.01)
    
    losses = []
    
    for i in range(steps):
        optimizer.zero_grad()
        
        # Construct G as a sparse tensor on the fly
        # Note: torch.sparse doesn't support autograd for 'indices', only 'values', which is what we want.
        G_sparse_off = torch.sparse_coo_tensor(indices, G_off_val, (num_nodes, num_nodes)).coalesce()
        
        # Probe Vector
        w = torch.randn(num_nodes, 1, device=device)
        
        # Forward Pass: v = (G G^T + eps I) w
        # v = G * (G^T * w)
        
        # 1. u = G^T * w
        # u = D * w + G_off^T * w
        # Sparse Transpose MM: (M^T x) is equivalent to (x^T M)^T or specialized ops.
        # PyTorch sparse: sparse.mm(mat.t(), dense) works.
        u = w * G_diag
        u += torch.sparse.mm(G_sparse_off.t(), w)
        
        # 2. v = G * u
        v = u * G_diag
        v += torch.sparse.mm(G_sparse_off, u)
        
        # 3. Loss = || 1/|A| * A * v - w ||^2
        y = apply_A(v.unsqueeze(0)).squeeze(0) # Add/remove batch dim for apply_A
        z = y / norm_A
        
        loss = ((z - w)**2).mean()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if i % 500 == 0:
            print(f"  Step {i}: Loss = {loss.item():.6f}")

    return losses

def find_optimal_sparse_G(A_diag, A_off, neighbors, num_nodes, device='cuda', steps=2000):
    """
    Trains a G matrix *from scratch* for THIS specific A to find the theoretical best fit.
    Constraints: G has same sparsity as A (1 Diag + 24 Neighbors).
    Objective: Minimize || I - A * (G G^T) ||_SAI
    """
    print(f"\n--- Finding 'Theoretical Ceiling' for Sparsity Pattern (Optimization) ---")
    
    # Initialize trainable params (Diag + Off-Diag)
    # Start with Identity-like (Diag=1, Off=0)
    G_diag = torch.ones(num_nodes, 1, device=device, requires_grad=True)
    G_off = torch.zeros(num_nodes, 24, device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([G_diag, G_off], lr=0.01)
    
    # Move fixed physics data to GPU
    t_neighbors = neighbors.to(device)
    t_A_diag = A_diag.to(device)
    t_A_off = A_off.to(device)
    
    # Scale invariance factor
    norm_A = (torch.abs(t_A_diag).sum() + torch.abs(t_A_off).sum()) / (num_nodes * 25)
    
    losses = []
    
    # helper for sparse multiplication
    def apply_sparse_G(x, g_d, g_o):
        # x: [B, N, 1]
        # g_d: [N, 1] or [1, N, 1]
        # g_o: [N, 24] or [1, N, 24]
        
        # Handle both cases: if g_d already has batch dimension, use it; otherwise unsqueeze
        if g_d.dim() == 2:
            g_d = g_d.unsqueeze(0)  # [N, 1] -> [1, N, 1]
        if g_o.dim() == 2:
            g_o = g_o.unsqueeze(0)  # [N, 24] -> [1, N, 24]
        
        res = x * g_d
        
        # Gather neighbors
        B, N, _ = x.shape
        safe_nbrs = t_neighbors.clone()
        safe_nbrs[safe_nbrs >= N] = 0
        batch_idx = torch.arange(B, device=device).view(B, 1, 1) * N
        flat_idx = (safe_nbrs + batch_idx).view(-1)
        
        x_flat = x.view(-1, 1)
        x_nbrs = x_flat[flat_idx].view(B, N, 24)
        mask = (t_neighbors < N).float().unsqueeze(0)
        
        res += (x_nbrs * g_o * mask).sum(dim=2, keepdim=True)
        return res

    def apply_sparse_GT(x, g_d, g_o):
        # Scatter: y_j += G_ij * x_i
        # g_d: [N, 1] or [1, N, 1]
        # g_o: [N, 24] or [1, N, 24]
        
        # Handle both cases: if g_d already has batch dimension, use it; otherwise unsqueeze
        if g_d.dim() == 2:
            g_d = g_d.unsqueeze(0)  # [N, 1] -> [1, N, 1]
        if g_o.dim() == 2:
            g_o = g_o.unsqueeze(0)  # [N, 24] -> [1, N, 24]
        
        res = x * g_d
        B, N, _ = x.shape
        
        vals = x * g_o # Values to scatter
        target_idx = t_neighbors # [N, 24]
        
        batch_idx = torch.arange(B, device=device).view(B, 1, 1) * N
        flat_target = (target_idx.unsqueeze(0) + batch_idx).view(-1)
        flat_vals = vals.view(-1)
        
        mask = (target_idx < N).view(-1).repeat(B)
        
        res_flat = res.view(-1)
        res_flat.index_add_(0, flat_target[mask], flat_vals[mask])
        return res_flat.view(B, N, 1)
        
    def apply_A(x):
        coeffs_d = t_A_diag.unsqueeze(0)
        coeffs_o = t_A_off.unsqueeze(0)
        return apply_sparse_G(x, coeffs_d, coeffs_o)

    start_t = time.time()
    for i in range(steps):
        optimizer.zero_grad()
        
        # Stochastic Trace Estimation
        # Loss = || (1/||A||) * A * G * G^T * w - w ||^2
        w = torch.randn(1, num_nodes, 1, device=device)
        
        # u = G^T * w
        u = apply_sparse_GT(w, G_diag, G_off)
        # v = G * u
        v = apply_sparse_G(u, G_diag, G_off)
        # y = A * v
        y = apply_A(v)
        
        z = y / norm_A
        
        loss = ((z - w)**2).mean()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if i % 500 == 0:
            print(f"  Step {i}: Loss = {loss.item():.6f}")

    print(f"  Final Optimization Loss: {losses[-1]:.6f} (Time: {time.time()-start_t:.1f}s)")
    return losses, torch.cat([G_diag.detach().cpu(), G_off.detach().cpu()], dim=1)

# --- 3. MAIN ANALYSIS ---

def inspect_matrix(frame_path):
    print(f"Loading {frame_path}...")
    pos, layers, neighbors, A_diag, A_off, N = load_frame_data(frame_path)
    print(f"Matrix Size: {N}x{N}")
    
    # 1. Build Dense A
    print("Constructing sparse A...")
    A_sp = build_scipy_A(A_diag, A_off, neighbors, N)
    
    # 2. Compute Inverse (Dense)
    print("Computing Dense Inverse A^-1 (this may take a moment)...")
    t0 = time.time()
    # Use splu if N is huge, but dense inv is better for visualization if RAM allows
    # For N=10k, dense is ~800MB. Fine.
    try:
        A_dense = A_sp.todense()
        A_inv = scipy.linalg.inv(A_dense)
        print(f"Inversion complete ({time.time()-t0:.2f}s).")
    except Exception as e:
        print(f"Dense inversion failed (probably OOM): {e}")
        return

    # 3. Analyze Decay (Structure)
    print("Analyzing Inverse Structure...")
    # Sample center node (middle of fluid)
    center_idx = N // 2
    row_inv = np.abs(A_inv[center_idx, :])
    row_pos = pos.numpy()
    center_pos = row_pos[center_idx]
    
    # Calculate geometric distances
    dists = np.linalg.norm(row_pos - center_pos, axis=1)
    
    # Filter only significant values
    mask = row_inv > 1e-6
    dists = dists[mask]
    vals = row_inv[mask]
    
    # 4. Find Optimal Sparse G (The Ceiling) - Compare Degree 1 vs Degree 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- 1. Degree 1 Optimization (Baseline) ---
    print("Generating Degree 1 Sparsity Pattern...")
    indices_d1 = get_sparsity_pattern(neighbors.to(device), N, degree=1)
    losses_d1 = find_optimal_sparse_G_generic(
        A_diag, A_off, neighbors, N, indices_d1, device, steps=3000
    )
    
    # --- 2. Degree 2 Optimization (Expanded) ---
    print("Generating Degree 2 Sparsity Pattern...")
    indices_d2 = get_sparsity_pattern(neighbors.to(device), N, degree=2)
    losses_d2 = find_optimal_sparse_G_generic(
        A_diag, A_off, neighbors, N, indices_d2, device, steps=3000
    )
    
    # --- 3. Degree 3 Optimization (Further Expanded) ---
    print("Generating Degree 3 Sparsity Pattern...")
    indices_d3 = get_sparsity_pattern(neighbors.to(device), N, degree=3)
    losses_d3 = find_optimal_sparse_G_generic(
        A_diag, A_off, neighbors, N, indices_d3, device, steps=3000
    )
    
    opt_losses = losses_d1  # Keep for backward compatibility with plotting
    
    # 5. Load Trained Model (if available) to compare
    model_loss = "N/A"
    try:
        # Assuming we are in Scripts folder
        model_path = Path(__file__).parent / "model_weights.bytes"
        if model_path.exists():
            from NeuralPreconditioner import SPAIGenerator, load_model_from_bytes, SAILoss
            model = SPAIGenerator(input_dim=58, d_model=32).to(device)
            # You might need the load_model_from_bytes helper from your other script here
            # For now, let's just note we can't easily load without copy-pasting that helper
            # But the optimization baseline is the key comparison.
            pass
    except:
        pass

    # --- PLOTTING ---
    plt.figure(figsize=(18, 10))
    
    # Plot 1: Inverse Decay
    plt.subplot(2, 3, 1)
    plt.scatter(dists, vals, alpha=0.5, s=2)
    plt.yscale('log')
    plt.title(f"A^-1 Row Decay (Node {center_idx})")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Magnitude |(A^-1)ij|")
    plt.grid(True, which="both", alpha=0.3)
    
    # Plot 2: Spy Plot of Inverse (Sparsity Pattern)
    plt.subplot(2, 3, 2)
    # Threshold to see "effective" sparsity
    limit = 1e-3 * np.max(np.abs(A_inv))
    plt.spy(A_inv > limit, markersize=0.1)
    plt.title(f"Structure of A^-1 (Values > {limit:.1e})")
    
    # Plot 3: Cholesky of Inverse (Ideal G structure)
    try:
        # Cholesky of A^-1 (Must be SPD)
        # Add tiny jitter to ensure stability
        L_true = scipy.linalg.cholesky(A_inv + np.eye(N)*1e-6, lower=True)
        
        plt.subplot(2, 3, 3)
        plt.spy(L_true > limit, markersize=0.1)
        plt.title(f"Structure of True Cholesky(A^-1)")
        
        # Calculate how much energy is captured by the fixed pattern
        # Mask of allowed neighbors
        nbrs_np = neighbors.numpy()
        captured_energy = 0.0
        total_energy = np.sum(L_true**2)
        
        # Slow python loop check (approximate via sampling if needed)
        # Actually, let's just use the optimization result for the "Fit" metric
        
    except Exception as e:
        print(f"Cholesky failed: {e}")

    # Plot 4: Optimization Curve (The Ceiling) - Degree 1 vs Degree 2 vs Degree 3
    plt.subplot(2, 3, 4)
    # Smooth the loss arrays to reduce noise
    losses_d1_smooth = smooth_losses(losses_d1, window_size=4)
    losses_d2_smooth = smooth_losses(losses_d2, window_size=4)
    losses_d3_smooth = smooth_losses(losses_d3, window_size=4)
    
    plt.plot(losses_d1_smooth, label=f'Degree 1 (Immediate) - Final: {losses_d1[-1]:.5f}')
    plt.plot(losses_d2_smooth, label=f'Degree 2 (Expanded) - Final: {losses_d2[-1]:.5f}')
    plt.plot(losses_d3_smooth, label=f'Degree 3 (Further Expanded) - Final: {losses_d3[-1]:.5f}')
    plt.yscale('log')
    plt.title("Theoretical Ceiling: Degree 1 vs Degree 2 vs Degree 3 Stencil")
    plt.xlabel("Optimization Steps")
    plt.ylabel("SAI Loss")
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    
    # Plot 5: CDF of Nonzeros per Row
    plt.subplot(2, 3, 5)
    nnz_d1, cdf_d1 = compute_nnz_per_row_cdf(indices_d1, N)
    nnz_d2, cdf_d2 = compute_nnz_per_row_cdf(indices_d2, N)
    nnz_d3, cdf_d3 = compute_nnz_per_row_cdf(indices_d3, N)
    
    plt.plot(nnz_d1, cdf_d1, label='Degree 1', marker='o', markersize=3, linewidth=2)
    plt.plot(nnz_d2, cdf_d2, label='Degree 2', marker='s', markersize=3, linewidth=2)
    plt.plot(nnz_d3, cdf_d3, label='Degree 3', marker='^', markersize=3, linewidth=2)
    plt.xlabel("Number of Nonzeros per Row")
    plt.ylabel("Percent of Rows (≤ nnz)")
    plt.title("CDF: Nonzeros per Row by Degree")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.show()
    
    print("\n--- Summary ---")
    print(f"1. Decay Analysis: See Plot 1. If line is flat, inverse is global (hard for sparse G).")
    print(f"2. Structure Analysis: See Plot 2/3. If Cholesky is dense, your G pattern is insufficient.")
    print(f"3. Theoretical Limit Comparison:")
    print(f"   - Degree 1 (Immediate neighbors): {losses_d1[-1]:.5f}")
    print(f"   - Degree 2 (2-hop neighbors): {losses_d2[-1]:.5f}")
    print(f"   - Degree 3 (3-hop neighbors): {losses_d3[-1]:.5f}")
    print(f"   (If your Neural Network loss is >> this, the Network is underfitting.)")
    print(f"   (If this loss is high (>0.1), the Sparsity Pattern itself is the bottleneck.)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=str, help="Path to frame folder")
    args = parser.parse_args()
    
    target_frame = args.frame
    
    # Auto-find if not provided
    if not target_frame:
        root = Path(__file__).parent.parent / "StreamingAssets" / "TestData"
        if root.exists():
            runs = sorted([d for d in root.iterdir() if d.name.startswith("Run_")])
            if runs:
                frames = sorted([f for f in runs[0].iterdir() if f.name.startswith("frame_")])
                if frames:
                    target_frame = frames[min(len(frames)//2, len(frames)-1)]
    
    if target_frame:
        inspect_matrix(target_frame)
    else:
        print("Could not auto-locate data. Please provide --frame path.")