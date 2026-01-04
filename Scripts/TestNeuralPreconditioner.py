#!/usr/bin/env python3
"""
TestNeuralPreconditioner.py
Benchmarks the Neural SPAI Preconditioner vs Standard CG.
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
    def __init__(self, input_dim=58, d_model=128, num_layers=4, nhead=4, window_size=512, max_octree_depth=12):
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
        self.head = nn.Linear(d_model, 25) 

    def forward(self, x, layers):
        # x: [B, N, C]
        # layers: [B, N] (Must be 2D, not 3D!)
        B, N, C = x.shape
        
        # A. Embedding
        # layer_embed(layers) -> [B, N, D]. feature_proj(x) -> [B, N, D]
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
        # Read header: p_mean(f), p_std(f), d_model(i), heads(i), num_layers(i), input_dim(i)
        header_bytes = f.read(24)
        header = struct.unpack('<ffiiii', header_bytes)
        p_mean, p_std, file_d_model, heads, num_layers, input_dim = header
        
        print(f"  Header Info: d_model={file_d_model}, heads={heads}, layers={num_layers}, input_dim={input_dim}")
        
        if file_d_model != model.d_model:
            raise ValueError(f"Model mismatch! File d_model={file_d_model}, Code d_model={model.d_model}")
            
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
        read_tensor([1, 512, file_d_model], model.window_pos_embed, "window_pos_embed")
        
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
    # Corrected: Create layers as [N, 1] for physics calc
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
    
    # Return: x [1, N, 58], layers [1, N] (Fixed from [1, N, 1])
    return x.unsqueeze(0), layers.squeeze(1).unsqueeze(0), neighbors, b, A_diag, A_off, num_nodes

# ==========================================
# 4. SOLVER (Fixed dims)
# ==========================================

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
        # Fix: valid_vals is [M, 1], y_flat is [N, 1]. No squeeze needed.
        y_flat.index_add_(0, valid_idx, valid_vals) 
        y += y_flat
        
    return y

def apply_preconditioner(r, coeffs, neighbors, num_nodes):
    u = apply_sparse_GT(r, coeffs, neighbors, num_nodes)
    z = apply_sparse_G(u, coeffs, neighbors, num_nodes)
    z += 1e-4 * r
    return z

def apply_physics_A(x, A_diag, A_off, neighbors, num_nodes):
    coeffs = torch.cat([A_diag, A_off], dim=1)
    return apply_sparse_G(x, coeffs, neighbors, num_nodes)

def run_pcg(b, A_diag, A_off, neighbors, num_nodes, coeffs=None, max_iter=200, tol=1e-5):
    x = torch.zeros_like(b)
    r = b.clone()
    r_norm = torch.norm(r).item()
    if r_norm < 1e-6: return x, 0, [r_norm]
    
    hist = [r_norm]
    
    if coeffs is not None:
        z = apply_preconditioner(r, coeffs, neighbors, num_nodes)
    else:
        z = r.clone()
        
    p = z.clone()
    rho = torch.dot(r.view(-1), z.view(-1))
    
    for k in range(max_iter):
        Ap = apply_physics_A(p, A_diag, A_off, neighbors, num_nodes)
        pAp = torch.dot(p.view(-1), Ap.view(-1))
        
        if abs(pAp) < 1e-12: break
        
        alpha = rho / pAp
        x += alpha * p
        r -= alpha * Ap
        
        r_n = torch.norm(r).item()
        hist.append(r_n)
        
        if r_n < tol:
            return x, k+1, hist
            
        if coeffs is not None:
            z = apply_preconditioner(r, coeffs, neighbors, num_nodes)
        else:
            z = r.clone()
            
        rho_new = torch.dot(r.view(-1), z.view(-1))
        beta = rho_new / rho
        p = z + beta * p
        rho = rho_new
        
    return x, max_iter, hist

# ==========================================
# 5. MAIN
# ==========================================

def benchmark_frame(frame_path, model, device):
    """Benchmark a single frame. Returns (it_base, t_base, it_net, t_net, t_inf, hist_base, hist_net)"""
    x, layers, neighbors, b, A_diag, A_off, N = load_frame(frame_path, device)
    
    # Generate Preconditioner
    G_coeffs = None
    t_inf = 0.0
    with torch.no_grad():
        t0 = time.time()
        G_coeffs = model(x, layers).squeeze(0)[:N]
        if device.type == 'cuda': torch.cuda.synchronize()
        t_inf = (time.time() - t0) * 1000
    
    # Standard CG
    t0 = time.time()
    _, it_base, hist_base = run_pcg(b, A_diag, A_off, neighbors, N, coeffs=None)
    if device.type == 'cuda': torch.cuda.synchronize()
    t_base = (time.time() - t0) * 1000
    
    # Neural SPAI CG
    t0 = time.time()
    _, it_net, hist_net = run_pcg(b, A_diag, A_off, neighbors, N, coeffs=G_coeffs)
    if device.type == 'cuda': torch.cuda.synchronize()
    t_net = (time.time() - t0) * 1000
    
    return it_base, t_base, it_net, t_net, t_inf, hist_base, hist_net

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
    parser.add_argument('--single', action='store_true', help='Test a single frame instead of range (with plot). Default is range mode (frames 150-200).')
    args = parser.parse_args()
        
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")
    
    # Initialize Model
    model = SPAIGenerator(input_dim=58, d_model=128).to(device)
    
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
        
        # Collect statistics
        stats_base_iters = []
        stats_base_times = []
        stats_net_iters = []
        stats_net_times = []
        stats_inf_times = []
        
        for i, frame_path in enumerate(frame_range):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(frame_range)} frames...")
            
            try:
                it_base, t_base, it_net, t_net, t_inf, _, _ = benchmark_frame(str(frame_path), model, device)
                stats_base_iters.append(it_base)
                stats_base_times.append(t_base)
                stats_net_iters.append(it_net)
                stats_net_times.append(t_net)
                stats_inf_times.append(t_inf)
            except Exception as e:
                print(f"  Warning: Failed to process {frame_path.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate and print averages
        print(f"\n{'='*60}")
        print(f"Average Results ({len(stats_base_iters)} frames):")
        print(f"{'='*60}")
        print(f"Standard CG:")
        print(f"  Avg Iterations: {np.mean(stats_base_iters):.2f}")
        print(f"  Avg Time:       {np.mean(stats_base_times):.2f} ms")
        print(f"\nNeural SPAI CG:")
        print(f"  Avg Inference:  {np.mean(stats_inf_times):.2f} ms")
        print(f"  Avg Iterations: {np.mean(stats_net_iters):.2f}")
        print(f"  Avg Solver Time: {np.mean(stats_net_times):.2f} ms")
        print(f"  Avg Total Time:  {np.mean(stats_inf_times) + np.mean(stats_net_times):.2f} ms")
        print(f"\nSpeedup:")
        print(f"  Iterations: {np.mean(stats_base_iters) / np.mean(stats_net_iters):.2f}x")
        print(f"  Time:       {np.mean(stats_base_times) / (np.mean(stats_inf_times) + np.mean(stats_net_times)):.2f}x")
        print(f"{'='*60}")
        
    else:
        # Single frame mode
        if not args.frame:
            print("Error: No frame found. Provide --frame path or use --single for single frame mode.")
            exit(1)
        
        print(f"Loading frame: {args.frame}...")
        x, layers, neighbors, b, A_diag, A_off, N = load_frame(args.frame, device)
        print(f"Num Nodes: {N}")
        
        # Generate preconditioner for this frame
        G_coeffs = None
        t_inf = 0.0
        with torch.no_grad():
            t0 = time.time()
            G_coeffs = model(x, layers).squeeze(0)[:N]
            if device.type == 'cuda': torch.cuda.synchronize()
            t_inf = (time.time() - t0) * 1000
            print(f"Inference Time: {t_inf:.2f} ms")
        
        # Benchmark
        print("\n--- Standard CG ---")
        t0 = time.time()
        _, it_base, hist_base = run_pcg(b, A_diag, A_off, neighbors, N, coeffs=None)
        if device.type == 'cuda': torch.cuda.synchronize()
        t_base = (time.time() - t0) * 1000
        print(f"Iterations: {it_base} | Time: {t_base:.2f} ms")
        
        print("\n--- Neural SPAI CG ---")
        t0 = time.time()
        _, it_net, hist_net = run_pcg(b, A_diag, A_off, neighbors, N, coeffs=G_coeffs)
        if device.type == 'cuda': torch.cuda.synchronize()
        t_net = (time.time() - t0) * 1000
        t_total = t_inf + t_net
        print(f"Neural Inference: {t_inf:.2f} ms")
        print(f"Solver:           {t_net:.2f} ms ({it_net} iters)")
        print(f"Total:            {t_total:.2f} ms")
        
        # Plot
        plt.figure(figsize=(10,6))
        plt.semilogy(hist_base, label='Standard CG', linestyle='--', linewidth=2)
        plt.semilogy(hist_net, label='Neural SPAI PCG', linewidth=2)
        plt.title(f"Convergence: Neural SPAI vs Standard CG\nFrame: {Path(args.frame).name}")
        plt.xlabel("Iteration")
        plt.ylabel("Residual Norm ||r||")
        plt.grid(True, which='both', alpha=0.3)
        plt.legend()
        plt.show()