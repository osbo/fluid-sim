#!/usr/bin/env python3
"""
TestNeuralPreconditioner.py
Benchmarks the Neural SPAI Preconditioner vs Standard CG vs Jacobi CG.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import time
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os

# ==========================================
# 1. MODEL DEFINITION
# ==========================================

class SPAIGenerator(nn.Module):
    # CHANGE 1: Default d_model=32 (matches paper's ~24k parameters)
    def __init__(self, input_dim=58, d_model=32, num_layers=4, nhead=4, window_size=256, max_octree_depth=12):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        
        # 1. Stem
        self.feature_proj = nn.Linear(input_dim, d_model)
        self.layer_embed = nn.Embedding(max_octree_depth, d_model)
        self.window_pos_embed = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)
        
        # 2. Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, 
            dropout=0.0, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Head
        self.norm_out = nn.LayerNorm(d_model)
        # CHANGE 2: Output 25 coefficients directly (1 Diag + 24 Neighbors)
        self.head = nn.Linear(d_model, 25) 

    def forward(self, x, layers):
        # x: [B, N, C]
        # layers: [B, N]
        B, N, C = x.shape
        
        # A. Embedding
        h = self.feature_proj(x) + self.layer_embed(layers)
        
        # B. Windowing
        pad_len = (self.window_size - (N % self.window_size)) % self.window_size
        if pad_len > 0:
            h = F.pad(h, (0, 0, 0, pad_len))
            
        N_padded = h.shape[1]
        num_windows = N_padded // self.window_size
        
        # Reshape to [Batch * NumWindows, WindowSize, d_model]
        h_windows = h.view(B * num_windows, self.window_size, self.d_model)
        h_windows = h_windows + self.window_pos_embed
        
        # C. Transformer
        h_encoded = self.encoder(h_windows)
        
        # D. Un-Window
        h_flat = h_encoded.view(B, N_padded, self.d_model)
        
        if pad_len > 0:
            h_flat = h_flat[:, :N, :]
            
        # E. Output
        return self.head(self.norm_out(h_flat))

# ==========================================
# 2. MODEL LOADER
# ==========================================

def load_model_from_bytes(model, bytes_path, device):
    """Load model weights from .bytes file (Unity format)"""
    file_size = os.path.getsize(bytes_path)
    print(f"Reading weights from {bytes_path} (Size: {file_size} bytes)")
    
    with open(bytes_path, 'rb') as f:
        header_bytes = f.read(24)
        header = struct.unpack('<ffiiii', header_bytes)
        p_mean, p_std, file_d_model, heads, num_layers, input_dim = header
        
        print(f"  Header Info: d_model={file_d_model}, heads={heads}, layers={num_layers}, input_dim={input_dim}")
        
        if file_d_model != model.d_model:
            raise ValueError(f"Model mismatch! File d_model={file_d_model}, Code d_model={model.d_model}")
        
        # Verify num_heads matches
        model_num_heads = model.encoder.layers[0].self_attn.num_heads
        if heads != model_num_heads:
            raise ValueError(f"Model mismatch! File heads={heads}, Model num_heads={model_num_heads}. "
                           f"Please recreate model with nhead={heads}")
            
        def read_tensor(shape, target_tensor, tensor_name="tensor"):
            size = int(np.prod(shape))
            data_bytes = f.read(size * 4)
            data = np.frombuffer(data_bytes, dtype=np.float32)
            
            try:
                t = torch.from_numpy(data.reshape(shape)).to(device)
            except Exception as e:
                raise RuntimeError(f"Reshape failed for {tensor_name}: {e}")

            if t.shape != target_tensor.shape:
                if t.T.shape == target_tensor.shape:
                    t = t.T
                else:
                    raise ValueError(f"Shape mismatch {tensor_name}: File {t.shape} vs Model {target_tensor.shape}")
            
            target_tensor.data.copy_(t)

        # 1. Stem
        read_tensor([input_dim, file_d_model], model.feature_proj.weight, "feature_proj.weight") 
        read_tensor([file_d_model], model.feature_proj.bias, "feature_proj.bias")
        
        read_tensor([12, file_d_model], model.layer_embed.weight, "layer_embed")
        read_tensor([1, model.window_size, file_d_model], model.window_pos_embed, "window_pos_embed")
        
        # 2. Layers
        for i, layer in enumerate(model.encoder.layers):
            prefix = f"layer_{i}"
            read_tensor([file_d_model, 3 * file_d_model], layer.self_attn.in_proj_weight, f"{prefix}.in_proj_w")
            read_tensor([3 * file_d_model], layer.self_attn.in_proj_bias, f"{prefix}.in_proj_b")
            
            read_tensor([file_d_model, file_d_model], layer.self_attn.out_proj.weight, f"{prefix}.out_proj_w")
            read_tensor([file_d_model], layer.self_attn.out_proj.bias, f"{prefix}.out_proj_b")
            
            read_tensor([file_d_model], layer.norm1.weight, f"{prefix}.norm1_w")
            read_tensor([file_d_model], layer.norm1.bias, f"{prefix}.norm1_b")
            
            read_tensor([file_d_model, file_d_model * 2], layer.linear1.weight, f"{prefix}.linear1_w")
            read_tensor([file_d_model * 2], layer.linear1.bias, f"{prefix}.linear1_b")
            
            read_tensor([file_d_model * 2, file_d_model], layer.linear2.weight, f"{prefix}.linear2_w")
            read_tensor([file_d_model], layer.linear2.bias, f"{prefix}.linear2_b")
            
            read_tensor([file_d_model], layer.norm2.weight, f"{prefix}.norm2_w")
            read_tensor([file_d_model], layer.norm2.bias, f"{prefix}.norm2_b")
        
        # 3. Head 
        read_tensor([file_d_model], model.norm_out.weight, "norm_out.weight")
        read_tensor([file_d_model], model.norm_out.bias, "norm_out.bias")
        
        # CHANGE 3: Read 25 output weights instead of 7
        read_tensor([file_d_model, 25], model.head.weight, "head.weight")
        read_tensor([25], model.head.bias, "head.bias")

# ==========================================
# 3. PHYSICS & DATA
# ==========================================

def compute_laplacian_stencil(neighbors, layers, num_nodes):
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
    
    is_wall = (neighbors == IDX_WALL).view(N, 6, 4)
    is_air  = (neighbors == IDX_AIR).view(N, 6, 4)
    is_fluid = (neighbors < N).view(N, 6, 4)
    
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
    # Diagonal is sum of absolute off-diagonals (Laplacian property)
    A_diag = torch.sum(torch.abs(A_off_flat), dim=1, keepdim=True)
    A_off_flat[neighbors == IDX_AIR] = 0.0
    
    return A_diag, A_off_flat

def load_frame(frame_path, device):
    path = Path(frame_path)
    num_nodes = 0
    with open(path / "meta.txt", 'r') as f:
        for line in f:
            if 'numNodes' in line: num_nodes = int(line.split(':')[1].strip())
            
    def read_bin(name, dtype):
        return np.fromfile(path / name, dtype=dtype)
        
    node_dtype = np.dtype([
        ('position', '3<f4'), ('velocity', '3<f4'), ('face_vels', '6<f4'),
        ('mass', '<f4'), ('layer', '<u4'), ('morton', '<u4'), ('active', '<u4')
    ])
    
    raw_nodes = read_bin("nodes.bin", node_dtype)
    raw_nbrs = read_bin("neighbors.bin", np.uint32).reshape(num_nodes, 24)
    raw_div = read_bin("divergence.bin", np.float32)[:num_nodes]
    
    pos = torch.from_numpy(raw_nodes['position'][:num_nodes] / 1024.0).to(device)
    layers = torch.from_numpy(raw_nodes['layer'][:num_nodes].astype(np.int64)).unsqueeze(1).to(device)
    neighbors = torch.from_numpy(raw_nbrs.astype(np.int64)).to(device)
    b = torch.from_numpy(raw_div).unsqueeze(1).to(device)
    
    IDX_WALL = num_nodes + 1
    is_wall_full = (neighbors == IDX_WALL)
    wall_indices = [0, 4, 8, 12, 16, 20]
    wall_flags = is_wall_full[:, wall_indices].float()
    
    IDX_AIR = num_nodes
    air_flags = (neighbors == IDX_AIR).float()
    
    A_diag, A_off = compute_laplacian_stencil(neighbors, layers, num_nodes)
    
    x = torch.cat([pos, wall_flags, air_flags, A_diag, A_off], dim=1)
    
    return x.unsqueeze(0), layers.squeeze(1).unsqueeze(0), neighbors, b, A_diag, A_off, num_nodes

# ==========================================
# 4. SOLVER (Fixed dims)
# ==========================================

# ==========================================
# OPTIMIZED SOLVER HELPERS (Pre-computed)
# ==========================================

def precompute_indices(neighbors, num_nodes, batch_size=1):
    """
    Call this ONCE before the CG loop to generate static index buffers.
    """
    device = neighbors.device
    N = num_nodes
    
    # Reshape neighbors to [B, N, 24] if needed
    if neighbors.dim() == 2:
        neighbors = neighbors.unsqueeze(0)  # [1, N, 24]
    
    # --- For G (Gather) ---
    safe_nbrs = neighbors.clone()
    safe_nbrs[safe_nbrs >= N] = 0
    
    batch_idx = torch.arange(batch_size, device=device).view(batch_size, 1, 1) * N
    gather_indices = (safe_nbrs + batch_idx).view(-1)
    
    # Mask for gather (to zero out invalid neighbors)
    gather_mask = (neighbors < N).float().unsqueeze(-1) # [B, N, 24, 1]

    # --- For GT (Scatter) ---
    # We need to flatten the target indices for index_add_
    target_idx = neighbors.view(-1) # [B*N*24]
    
    # Create batch offsets for the flat targets
    # neighbors is [B, N, 24], so we need batch offsets repeated 24 times per node
    batch_offsets_scatter = torch.arange(batch_size, device=device).view(batch_size, 1, 1) * N
    flat_targets_raw = (neighbors + batch_offsets_scatter).view(-1)
    
    # Create a boolean mask for valid scatters
    valid_scatter_mask = (target_idx < N)
    
    # Filter now to avoid boolean indexing inside the loop
    # We will use these to index into the 'values' tensor we want to scatter
    scatter_active_indices = torch.nonzero(valid_scatter_mask, as_tuple=True)[0]
    scatter_target_indices = flat_targets_raw[scatter_active_indices]

    return {
        'gather_idx': gather_indices,
        'gather_mask': gather_mask,
        'scatter_src_idx': scatter_active_indices,
        'scatter_dst_idx': scatter_target_indices
    }

def apply_sparse_G_optimized(x, coeffs, precomputed):
    """ Optimized Gather: y = G * x """
    # x: [N, 1] -> add batch dim -> [1, N, 1]
    if x.dim() == 2:
        x = x.unsqueeze(0)  # [1, N, 1]
    if coeffs.dim() == 2:
        coeffs = coeffs.unsqueeze(0)  # [1, N, 25]
    
    B, N, _ = x.shape
    
    # 1. Diagonal
    y = x * coeffs[:, :, 0:1]
    
    # 2. Off-Diagonal (Gather)
    # Use pre-computed flat indices to pick neighbors from x
    x_flat = x.view(-1, 1)
    x_nbrs = x_flat.index_select(0, precomputed['gather_idx']) # Faster than x[idx]
    x_nbrs = x_nbrs.view(B, N, 24)
    
    # Apply Mask & Coeffs
    # Note: mask is already expanded
    val = (x_nbrs * coeffs[:, :, 1:] * precomputed['gather_mask'].squeeze(-1)).sum(dim=2, keepdim=True)
    
    result = y + val
    # Remove batch dimension if input didn't have it
    return result.squeeze(0) if result.shape[0] == 1 else result

def apply_sparse_GT_optimized(x, coeffs, precomputed, N):
    """ Optimized Scatter: y = G.T * x """
    # x: [N, 1] -> add batch dim -> [1, N, 1]
    if x.dim() == 2:
        x = x.unsqueeze(0)  # [1, N, 1]
    if coeffs.dim() == 2:
        coeffs = coeffs.unsqueeze(0)  # [1, N, 25]
    
    # 1. Diagonal
    y = x * coeffs[:, :, 0:1]
    
    # 2. Off-Diagonal (Scatter)
    # Prepare values to scatter: val = x_i * coeff_ij
    vals = x * coeffs[:, :, 1:] # [B, N, 24]
    vals_flat = vals.view(-1)
    
    # Select only the valid values using pre-computed indices
    valid_vals = vals_flat.index_select(0, precomputed['scatter_src_idx'])
    
    # Add to destination
    # We initialize y_scatter as zero
    y_scatter = torch.zeros_like(x.view(-1))
    
    # index_add_ is still necessary, but we removed all overhead around it
    y_scatter.index_add_(0, precomputed['scatter_dst_idx'], valid_vals)
    
    result = y + y_scatter.view(*x.shape)
    # Remove batch dimension if input didn't have it
    return result.squeeze(0) if result.shape[0] == 1 else result

def apply_sparse_G(x, coeffs, neighbors, num_nodes):
    N = num_nodes
    y = x * coeffs[:, 0:1]
    safe_nbrs = neighbors.clone()
    safe_nbrs[safe_nbrs >= N] = 0
    x_nbrs = x[safe_nbrs]
    mask = (neighbors < N).float().unsqueeze(-1)
    y += (x_nbrs * coeffs[:, 1:].unsqueeze(-1) * mask).sum(dim=1)
    return y

def apply_sparse_GT(x, coeffs, neighbors, num_nodes):
    N = num_nodes
    y = x * coeffs[:, 0:1] # Diagonal
    
    vals = x.unsqueeze(1) * coeffs[:, 1:].unsqueeze(-1) # [N, 24, 1]
    target_idx = neighbors.view(-1)
    vals_flat = vals.view(-1, 1)
    
    mask = target_idx < N
    valid_idx = target_idx[mask]
    valid_vals = vals_flat[mask]
    
    if valid_idx.numel() > 0:
        y_flat = torch.zeros(N, 1, device=x.device)
        y_flat.index_add_(0, valid_idx, valid_vals) 
        y += y_flat
        
    return y

def apply_neural_preconditioner(r, coeffs, neighbors, num_nodes, precomputed=None):
    # Standard SPAI: z = G * G^T * r
    if precomputed is not None:
        u = apply_sparse_GT_optimized(r, coeffs, precomputed, num_nodes)
        z = apply_sparse_G_optimized(u, coeffs, precomputed)
    else:
        u = apply_sparse_GT(r, coeffs, neighbors, num_nodes)
        z = apply_sparse_G(u, coeffs, neighbors, num_nodes)
    
    # Tiny regularization just for numerical safety
    z += 1e-6 * r 
    return z

def apply_physics_A(x, A_diag, A_off, neighbors, num_nodes, precomputed=None):
    coeffs = torch.cat([A_diag, A_off], dim=1)
    if precomputed is not None:
        return apply_sparse_G_optimized(x, coeffs, precomputed)
    else:
        return apply_sparse_G(x, coeffs, neighbors, num_nodes)

def run_pcg(b, A_diag, A_off, neighbors, num_nodes, coeffs=None, precon_type='identity', max_iter=200, tol=1e-5, precomputed=None):
    """
    PCG Solver.
    precon_type: 'identity', 'jacobi', or 'neural'
    precomputed: Optional precomputed indices for optimization
    """
    x = torch.zeros_like(b)
    r = b.clone()
    r_norm = torch.norm(r).item()
    if r_norm < 1e-6: return x, 0, [r_norm]
    
    hist = [r_norm]
    
    def apply_precond(res):
        if precon_type == 'neural' and coeffs is not None:
            return apply_neural_preconditioner(res, coeffs, neighbors, num_nodes, precomputed)
        elif precon_type == 'jacobi':
            # Jacobi: z = D^-1 * r.  (A_diag is D)
            return res / (A_diag + 1e-10) 
        else:
            return res.clone()
            
    z = apply_precond(r)
    p = z.clone()
    rho = torch.dot(r.view(-1), z.view(-1))
    
    for k in range(max_iter):
        Ap = apply_physics_A(p, A_diag, A_off, neighbors, num_nodes, precomputed)
        pAp = torch.dot(p.view(-1), Ap.view(-1))
        
        if abs(pAp) < 1e-12: break
        
        alpha = rho / pAp
        x += alpha * p
        r -= alpha * Ap
        
        r_n = torch.norm(r).item()
        hist.append(r_n)
        
        if r_n < tol:
            return x, k+1, hist
            
        z = apply_precond(r)
            
        rho_new = torch.dot(r.view(-1), z.view(-1))
        beta = rho_new / rho
        p = z + beta * p
        rho = rho_new
    
    return x, max_iter, hist

# ==========================================
# 5. MAIN
# ==========================================

def benchmark_frame(frame_path, model, device):
    """Benchmark a single frame. Returns (it_base, t_base, it_jacobi, t_jacobi, it_net, t_net, t_inf, hist_base, hist_jacobi, hist_net)"""
    x, layers, neighbors, b, A_diag, A_off, N = load_frame(frame_path, device)
    
    # Pre-compute indices once for optimization
    precomputed = precompute_indices(neighbors, N, batch_size=1)
    
    # 1. Generate Neural Preconditioner
    G_coeffs = None
    t_inf = 0.0
    with torch.no_grad():
        t0 = time.time()
        
        # CHANGE 4: No expansion needed. Get [N, 25] directly.
        G_coeffs = model(x, layers).squeeze(0)[:N]
        
        if device.type == 'cuda': torch.cuda.synchronize()
        t_inf = (time.time() - t0) * 1000
    
    # 2. Standard CG (Identity)
    t0 = time.time()
    _, it_base, hist_base = run_pcg(b, A_diag, A_off, neighbors, N, precon_type='identity', precomputed=precomputed)
    if device.type == 'cuda': torch.cuda.synchronize()
    t_base = (time.time() - t0) * 1000
    
    # 3. Jacobi CG
    t0 = time.time()
    _, it_jacobi, hist_jacobi = run_pcg(b, A_diag, A_off, neighbors, N, precon_type='jacobi', precomputed=precomputed)
    if device.type == 'cuda': torch.cuda.synchronize()
    t_jacobi = (time.time() - t0) * 1000
    
    # 4. Neural SPAI CG
    t0 = time.time()
    _, it_net, hist_net = run_pcg(b, A_diag, A_off, neighbors, N, coeffs=G_coeffs, precon_type='neural', precomputed=precomputed)
    if device.type == 'cuda': torch.cuda.synchronize()
    t_net = (time.time() - t0) * 1000
    
    return it_base, t_base, it_jacobi, t_jacobi, it_net, t_net, t_inf, hist_base, hist_jacobi, hist_net

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).parent
    default_data_path = script_dir.parent / "StreamingAssets" / "TestData"
    
    default_frame = None
    all_frames = []
    if default_data_path.exists():
        run_dirs = sorted([d for d in default_data_path.iterdir() if d.is_dir() and d.name.startswith('Run_')])
        if run_dirs:
            all_frames = sorted([f for f in run_dirs[0].iterdir() if f.is_dir() and f.name.startswith('frame_')])
            if all_frames: default_frame = str(all_frames[min(600, len(all_frames)-1)])
    
    parser.add_argument('--frame', type=str, default=default_frame)
    parser.add_argument('--model', type=str, default=str(script_dir / "model_weights.bytes"))
    parser.add_argument('--single', action='store_true', help='Test a single frame instead of range.')
    args = parser.parse_args()
        
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")
    
    # CHANGE 5: Set d_model=32 (matches paper's ~24k parameters)
    # Note: nhead=4 and window_size=256
    model = SPAIGenerator(input_dim=58, d_model=32, num_layers=4, nhead=4, window_size=256).to(device)
    
    # Load Weights
    try:
        load_model_from_bytes(model, args.model, device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Model Load Failed: {e}")
        import traceback
        traceback.print_exc()
        print("Using Identity Preconditioner.")
    
    if not args.single:
        # Range mode: test frames 150-200
        if not all_frames:
            print("Error: No frames found in TestData")
            exit(1)
        
        # Find frames 150-200
        frame_range = []
        for f in all_frames:
            try:
                frame_num = int(f.name.split('_')[1])
                if 150 <= frame_num <= 200:
                    frame_range.append(f)
            except (ValueError, IndexError):
                continue
        
        if not frame_range:
            print("Error: No frames found in range 150-200")
            exit(1)
        
        print(f"\nTesting {len(frame_range)} frames (150-200)...")
        
        stats = {
            'base_iter': [], 'base_time': [],
            'jacobi_iter': [], 'jacobi_time': [],
            'net_iter': [], 'net_time': [], 'inf_time': []
        }
        
        for i, frame_path in enumerate(frame_range):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(frame_range)} frames...")
            
            # Run a dummy pass to warm up caches/JIT
            if i == 0:
                try:
                    benchmark_frame(str(frame_path), model, device)
                except Exception as e:
                    print(f"  Warning: Warmup failed for {frame_path.name}: {e}")
                    continue
            
            try:
                ib, tb, ij, tj, i_net, t_net, t_inf, _, _, _ = benchmark_frame(str(frame_path), model, device)
                stats['base_iter'].append(ib)
                stats['base_time'].append(tb)
                stats['jacobi_iter'].append(ij)
                stats['jacobi_time'].append(tj)
                stats['net_iter'].append(i_net)
                stats['net_time'].append(t_net)
                stats['inf_time'].append(t_inf)
            except Exception as e:
                print(f"  Warning: Failed to process {frame_path.name}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Average Results ({len(frame_range)} frames):")
        print(f"{'='*60}")
        print(f"Standard CG:")
        print(f"  Avg Iterations: {np.mean(stats['base_iter']):.2f}")
        print(f"  Avg Time:       {np.mean(stats['base_time']):.2f} ms")
        print(f"\nJacobi CG:")
        print(f"  Avg Iterations: {np.mean(stats['jacobi_iter']):.2f}")
        print(f"  Avg Time:       {np.mean(stats['jacobi_time']):.2f} ms")
        print(f"  Speedup vs Std: {np.mean(stats['base_time'])/np.mean(stats['jacobi_time']):.2f}x")
        print(f"\nNeural SPAI CG:")
        print(f"  Avg Inference:  {np.mean(stats['inf_time']):.2f} ms")
        print(f"  Avg Iterations: {np.mean(stats['net_iter']):.2f}")
        print(f"  Avg Solver Time: {np.mean(stats['net_time']):.2f} ms")
        print(f"  Avg Total Time:  {np.mean(stats['inf_time']) + np.mean(stats['net_time']):.2f} ms")
        
        total_net = np.mean(stats['inf_time']) + np.mean(stats['net_time'])
        print(f"\nSpeedup (Total Time):")
        print(f"  Neural vs Std:    {np.mean(stats['base_time']) / total_net:.2f}x")
        print(f"  Neural vs Jacobi: {np.mean(stats['jacobi_time']) / total_net:.2f}x")
        print(f"{'='*60}")
        
    else:
        # Single frame mode
        if not args.frame:
            print("Error: No frame found. Provide --frame path or use --single.")
            exit(1)
        
        print(f"Loading frame: {args.frame}...")
        x, layers, neighbors, b, A_diag, A_off, N = load_frame(args.frame, device)
        print(f"Num Nodes: {N}")
        
        # Pre-compute indices once for optimization
        precomputed = precompute_indices(neighbors, N, batch_size=1)
        
        # Benchmarking
        G_coeffs = None
        t_inf = 0.0
        with torch.no_grad():
            t0 = time.time()
            # CHANGE 4: No expansion needed. Get [N, 25] directly.
            G_coeffs = model(x, layers).squeeze(0)[:N]
            
            if device.type == 'cuda': torch.cuda.synchronize()
            t_inf = (time.time() - t0) * 1000
            print(f"Neural Inference Time: {t_inf:.2f} ms")
        
        print("\n--- Standard CG ---")
        t0 = time.time()
        _, it_base, hist_base = run_pcg(b, A_diag, A_off, neighbors, N, precon_type='identity', precomputed=precomputed)
        if device.type == 'cuda': torch.cuda.synchronize()
        t_base = (time.time() - t0) * 1000
        print(f"Iterations: {it_base} | Time: {t_base:.2f} ms")
        
        print("\n--- Jacobi CG ---")
        t0 = time.time()
        _, it_jacobi, hist_jacobi = run_pcg(b, A_diag, A_off, neighbors, N, precon_type='jacobi', precomputed=precomputed)
        if device.type == 'cuda': torch.cuda.synchronize()
        t_jacobi = (time.time() - t0) * 1000
        print(f"Iterations: {it_jacobi} | Time: {t_jacobi:.2f} ms")

        print("\n--- Neural SPAI CG ---")
        t0 = time.time()
        _, it_net, hist_net = run_pcg(b, A_diag, A_off, neighbors, N, coeffs=G_coeffs, precon_type='neural', precomputed=precomputed)
        if device.type == 'cuda': torch.cuda.synchronize()
        t_net = (time.time() - t0) * 1000
        t_total = t_inf + t_net
        print(f"Solver:           {t_net:.2f} ms ({it_net} iters)")
        print(f"Total:            {t_total:.2f} ms")
        
        # Plot
        plt.figure(figsize=(10,6))
        plt.semilogy(hist_base, label='Standard CG', linestyle=':', linewidth=2, color='gray')
        plt.semilogy(hist_jacobi, label='Jacobi CG', linestyle='--', linewidth=2, color='orange')
        plt.semilogy(hist_net, label='Neural SPAI CG', linewidth=2, color='blue')
        plt.title(f"Convergence: Neural SPAI vs Jacobi vs Standard\nFrame: {Path(args.frame).name}")
        plt.xlabel("Iteration")
        plt.ylabel("Residual Norm ||r||")
        plt.grid(True, which='both', alpha=0.3)
        plt.legend()
        plt.show()