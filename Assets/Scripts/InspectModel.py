#!/usr/bin/env python3
"""
InspectModel.py
Introspects the Neural SPAI model to determine if it is effectively
learning useful attention patterns or collapsing to a simple heuristic (Jacobi).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import matplotlib.pyplot as plt
from pathlib import Path
import os
import copy

# ==========================================
# 1. INSTRUMENTED MODEL ARCHITECTURE
# ==========================================

class FourierPositionEmbed(nn.Module):
    def __init__(self, d_model, scale=1024.0):
        super().__init__()
        self.d_model = d_model
        
        # We need even dimensions to split between sin and cos
        assert d_model % 2 == 0, "d_model must be even for Fourier Embeddings"
        
        # B matrix: [3, d_model/2]
        # Projects 3D coordinates -> Frequency spectrum
        self.register_buffer('B', torch.randn(3, d_model // 2) * scale)

    def forward(self, pos):
        """
        pos: [Batch, N, 3] in range [0, 1]
        Returns: [Batch, N, d_model]
        """
        # 1. Project positions: (B, N, 3) @ (3, D/2) -> (B, N, D/2)
        proj = (2 * np.pi * pos) @ self.B
        
        # 2. Apply Sin/Cos components
        # Result is [sin(proj), cos(proj)] -> (B, N, D)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class InstrumentedTransformerLayer(nn.TransformerEncoderLayer):
    """
    A Transformer layer that saves its attention weights for inspection.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Standard implementation but capturing weights
        # src shape: [Batch, SeqLen, Dim]
        
        # 1. Self Attention
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False # We want individual heads [Batch, Heads, Seq, Seq]
        )
        self.last_attn_weights = attn_weights.detach().cpu()
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 2. Feed Forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class InstrumentedSPAIGenerator(nn.Module):
    def __init__(self, input_dim=58, d_model=32, num_layers=4, nhead=4, window_size=256, max_octree_depth=12):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        
        # 1. Stem
        self.feature_proj = nn.Linear(input_dim, d_model)
        self.layer_embed = nn.Embedding(max_octree_depth, d_model)
        # 3D Fourier Position Embedding
        # Scale heuristic: For depth 12 (resolution 4096), try 512.0 to 1024.0
        # Start with 128.0 (good for resolving ~1/200th of the domain)
        self.pos_embed = FourierPositionEmbed(d_model, scale=1024.0)
        
        # 2. Backbone (Using Instrumented Layers manually)
        # We use ModuleList so we can access individual layers easily
        self.layers = nn.ModuleList([
            InstrumentedTransformerLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, 
                dropout=0.0, activation='gelu', batch_first=True, norm_first=True
            ) for _ in range(num_layers)
        ])
        
        # 3. Head
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 25) 

    def forward(self, x, layers):
        """
        x: [Batch, N, 58]
        # Note: x[:, :, 0:3] contains the normalized position (x, y, z)
        layers: [Batch, N]
        """
        B, N, C = x.shape
        
        # A. Extract 3D Position (Indices 0, 1, 2)
        pos = x[:, :, :3]
        
        # B. Compute 3D Fourier Embedding
        pos_embedding = self.pos_embed(pos)  # [B, N, d_model]
        
        # C. Combine Features + Layer + 3D Position
        # Note: We apply this BEFORE windowing/padding so the geometry is preserved globally
        h = self.feature_proj(x) + self.layer_embed(layers) + pos_embedding
        
        # D. Windowing
        pad_len = (self.window_size - (N % self.window_size)) % self.window_size
        if pad_len > 0:
            h = F.pad(h, (0, 0, 0, pad_len))
            
        N_padded = h.shape[1]
        num_windows = N_padded // self.window_size
        
        h_windows = h.view(B * num_windows, self.window_size, self.d_model)
        
        # E. Transformer Loop
        for layer in self.layers:
            h_windows = layer(h_windows)
        
        # F. Un-Window
        h_flat = h_windows.view(B, N_padded, self.d_model)
        
        if pad_len > 0:
            h_flat = h_flat[:, :N, :]
            
        # G. Output
        return self.head(self.norm_out(h_flat))

# ==========================================
# 2. HELPERS (COPIED FROM TEST SCRIPT)
# ==========================================

def load_model_from_bytes(model, bytes_path, device):
    """Modified loader to handle the change from model.encoder to model.layers"""
    if not os.path.exists(bytes_path):
        raise FileNotFoundError(f"Model file not found: {bytes_path}")
        
    file_size = os.path.getsize(bytes_path)
    print(f"Reading weights from {bytes_path} (Size: {file_size} bytes)")
    
    with open(bytes_path, 'rb') as f:
        header_bytes = f.read(24)
        header = struct.unpack('<ffiiii', header_bytes)
        p_mean, p_std, file_d_model, heads, num_layers, input_dim = header
        
        def read_packed_tensor(shape, target_tensor, tensor_name="tensor"):
            num_elements = int(np.prod(shape))
            padded_count = num_elements if num_elements % 2 == 0 else num_elements + 1
            uint_count = padded_count // 2
            data_bytes = f.read(uint_count * 4)
            data_uint = np.frombuffer(data_bytes, dtype=np.uint32)
            data_fp16 = data_uint.view(np.float16)
            if num_elements % 2 != 0: data_fp16 = data_fp16[:num_elements]
            data_fp32 = data_fp16.astype(np.float32)
            
            try:
                t = torch.from_numpy(data_fp32.reshape(shape)).to(device)
            except Exception as e:
                # Fallback for shape mismatches (transpose fix)
                if len(shape) == 2 and shape[0] != shape[1]:
                     t = torch.from_numpy(data_fp32.reshape(shape[::-1])).to(device).T
                else:
                    raise e
                    
            if t.shape != target_tensor.shape:
                if t.T.shape == target_tensor.shape: t = t.T
                else: raise ValueError(f"Shape mismatch {tensor_name}")
            target_tensor.data.copy_(t)

        read_packed_tensor([input_dim, file_d_model], model.feature_proj.weight, "feature_proj.weight") 
        read_packed_tensor([file_d_model], model.feature_proj.bias, "feature_proj.bias")
        read_packed_tensor([12, file_d_model], model.layer_embed.weight, "layer_embed")
        # B matrix for Fourier Position Embedding (fixed buffer, needed for GPU reconstruction)
        read_packed_tensor([3, file_d_model // 2], model.pos_embed.B, "pos_embed.B")
        
        # Load layers (iterate our ModuleList instead of TransformerEncoder)
        for i, layer in enumerate(model.layers):
            prefix = f"layer_{i}"
            read_packed_tensor([file_d_model, 3 * file_d_model], layer.self_attn.in_proj_weight, f"{prefix}.in_proj_w")
            read_packed_tensor([3 * file_d_model], layer.self_attn.in_proj_bias, f"{prefix}.in_proj_b")
            read_packed_tensor([file_d_model, file_d_model], layer.self_attn.out_proj.weight, f"{prefix}.out_proj_w")
            read_packed_tensor([file_d_model], layer.self_attn.out_proj.bias, f"{prefix}.out_proj_b")
            read_packed_tensor([file_d_model], layer.norm1.weight, f"{prefix}.norm1_w")
            read_packed_tensor([file_d_model], layer.norm1.bias, f"{prefix}.norm1_b")
            read_packed_tensor([file_d_model, file_d_model * 2], layer.linear1.weight, f"{prefix}.linear1_w")
            read_packed_tensor([file_d_model * 2], layer.linear1.bias, f"{prefix}.linear1_b")
            read_packed_tensor([file_d_model * 2, file_d_model], layer.linear2.weight, f"{prefix}.linear2_w")
            read_packed_tensor([file_d_model], layer.linear2.bias, f"{prefix}.linear2_b")
            read_packed_tensor([file_d_model], layer.norm2.weight, f"{prefix}.norm2_w")
            read_packed_tensor([file_d_model], layer.norm2.bias, f"{prefix}.norm2_b")
        
        read_packed_tensor([file_d_model], model.norm_out.weight, "norm_out.weight")
        read_packed_tensor([file_d_model], model.norm_out.bias, "norm_out.bias")
        read_packed_tensor([file_d_model, 25], model.head.weight, "head.weight")
        read_packed_tensor([25], model.head.bias, "head.bias")

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
    
    pos = torch.from_numpy(raw_nodes['position'][:num_nodes] / 1024.0).to(device)
    layers = torch.from_numpy(raw_nodes['layer'][:num_nodes].astype(np.int64)).unsqueeze(1).to(device)
    neighbors = torch.from_numpy(raw_nbrs.astype(np.int64)).to(device)
    
    IDX_WALL = num_nodes + 1
    is_wall_full = (neighbors == IDX_WALL)
    wall_indices = [0, 4, 8, 12, 16, 20]
    wall_flags = is_wall_full[:, wall_indices].float()
    
    IDX_AIR = num_nodes
    air_flags = (neighbors == IDX_AIR).float()
    
    A_diag, A_off = compute_laplacian_stencil(neighbors, layers, num_nodes)
    
    # Matches features_np in NeuralPreconditioner.py
    x = torch.cat([pos, wall_flags, air_flags, A_diag, A_off], dim=1)
    
    return x.unsqueeze(0), layers.squeeze(1).unsqueeze(0), A_diag, A_off, num_nodes

# ==========================================
# 3. ANALYSIS LOGIC
# ==========================================

def entropy(probs, dim=-1):
    """Calculates entropy of a probability distribution."""
    # Add epsilon to avoid log(0)
    p = probs + 1e-9
    return -(p * torch.log(p)).sum(dim=dim)

def analyze_model_behavior(model_path, frame_path):
    device = torch.device('cpu') # Use CPU for easier inspection
    print(f"\n--- INSPECTING MODEL ---")
    print(f"Model: {model_path}")
    print(f"Frame: {frame_path}")

    # 1. Load Model
    model = InstrumentedSPAIGenerator(input_dim=58, d_model=32, num_layers=4, nhead=4, window_size=256).to(device)
    try:
        load_model_from_bytes(model, model_path, device)
        model.eval()
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return

    # 2. Load Data
    x, layers, A_diag, A_off, N = load_frame(frame_path, device)
    
    # 3. Run Inference
    with torch.no_grad():
        G_coeffs = model(x, layers).squeeze(0)[:N]

    # --- METRIC 1: Is it just Jacobi? ---
    # Jacobi means G = diag(1/sqrt(A_ii)).
    # Neural model outputs G coefficients. G_coeffs[:, 0] is diagonal.
    
    # Calculate True Jacobi Diagonal
    jacobi_diag = 1.0 / torch.sqrt(A_diag.squeeze() + 1e-6)
    neural_diag = G_coeffs[:, 0]
    
    # Normalize for scale comparison (sometimes models learn scaled versions)
    j_norm = jacobi_diag / jacobi_diag.mean()
    n_norm = neural_diag / neural_diag.mean()
    
    correlation = torch.corrcoef(torch.stack([j_norm, n_norm]))[0, 1].item()
    
    # Check off-diagonals magnitude
    neural_off_diag = G_coeffs[:, 1:]
    avg_off_mag = torch.mean(torch.abs(neural_off_diag)).item()
    avg_diag_mag = torch.mean(torch.abs(neural_diag)).item()
    off_diag_ratio = avg_off_mag / avg_diag_mag

    print(f"\n[1] VS JACOBI HEURISTIC")
    print(f"    Diagonal Correlation: {correlation:.4f} (1.0 = Perfectly Linear w/ Jacobi)")
    print(f"    Off-Diagonal Ratio:   {off_diag_ratio:.4f} (Near 0.0 = Diagonal Matrix)")
    
    if correlation > 0.98 and off_diag_ratio < 0.05:
        print("    >> CONCLUSION: MODEL HAS COLLAPSED TO JACOBI (Diagonal Preconditioner).")
    elif off_diag_ratio > 0.1:
        print("    >> CONCLUSION: Model is predicting significant OFF-DIAGONAL terms.")
    else:
        print("    >> CONCLUSION: Model is mostly diagonal but strictly follows Jacobi scaling.")

    # --- METRIC 2: Attention Analysis ---
    print(f"\n[2] ATTENTION ANALYSIS")
    
    # Gather attention weights from the last layer
    # Shape: [Batch*Windows, Heads, WindowSize, WindowSize]
    attn = model.layers[-1].last_attn_weights
    num_heads = attn.shape[1]
    
    # 2a. Diagonal Dominance (Self-Attention)
    # How much does a token attend to itself vs neighbors?
    diag_attn = torch.diagonal(attn, dim1=-2, dim2=-1) # [B*W, Heads, WindowSize]
    avg_self_attn = diag_attn.mean().item()
    
    print(f"    Avg Self-Attention:   {avg_self_attn:.4f} (If ~1.0, tokens ignore neighbors)")
    
    # 2b. Attention Entropy (Focus)
    # Max entropy for window 256 is ln(256) ≈ 5.54
    attn_entropy = entropy(attn, dim=-1).mean().item()
    print(f"    Avg Attention Entropy: {attn_entropy:.4f} / 5.54 (Lower = More Focused)")
    
    # 2c. Head Diversity
    # Do heads look at different things?
    # We compare Head 0 vs Head 1 on the first window
    h0 = attn[0, 0, 50, :].numpy() # Window 0, Head 0, Token 50
    h1 = attn[0, 1, 50, :].numpy() # Window 0, Head 1, Token 50
    
    head_diff = np.abs(h0 - h1).sum()
    print(f"    Head Divergence:      {head_diff:.4f} (0.0 = Heads are identical)")
    
    if avg_self_attn > 0.9:
        print("    >> WARNING: Model is ignoring context (Pure Self-Attention).")
    elif attn_entropy > 5.0:
        print("    >> WARNING: Model is averaging everything (Global Pooling).")
    else:
        print("    >> POSITIVE: Attention seems structured and focused.")

    # --- METRIC 3: Context Sensitivity (Perturbation Test) ---
    print(f"\n[3] CONTEXT SENSITIVITY TEST")
    
    # Pick a test node that is NOT a boundary (to be safe)
    test_idx = N // 2
    
    # 1. Get Baseline Output
    base_out = G_coeffs[test_idx].clone()
    
    # 2. Perturb input: Pretend there is a wall to the 'Right' (Index 4 in wall flags)
    # x shape: [1, N, 58]
    # Indices: Pos(3) + Wall(6) ...
    # Wall start at index 3. Right is 3 + 1 = 4.
    x_perturbed = x.clone()
    x_perturbed[0, test_idx, 3+1] = 1.0 # Set Right Wall Flag
    
    with torch.no_grad():
        out_perturbed = model(x_perturbed, layers).squeeze(0)[test_idx]
        
    diff = torch.norm(base_out - out_perturbed).item()
    print(f"    Output Change on Wall: {diff:.6f}")
    
    if diff < 1e-5:
        print("    >> FAIL: Model output ignored the added wall.")
    else:
        print("    >> PASS: Model reacted to context change.")

    print(f"{'-'*40}")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    
    # 1. Find Data
    data_path = script_dir.parent / "StreamingAssets" / "TestData"
    frame_path = None
    if data_path.exists():
        # Try to find a frame inside a Run folder
        runs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('Run_')])
        if runs:
            frames = sorted([f for f in runs[0].iterdir() if f.is_dir() and f.name.startswith('frame_')])
            # Pick a middle frame (more interesting physics)
            if frames: frame_path = frames[min(100, len(frames)-1)]
    
    # 2. Find Model
    model_path = script_dir / "model_weights.bytes"
    
    if frame_path and model_path.exists():
        analyze_model_behavior(str(model_path), str(frame_path))
    else:
        print("Error: Could not find model_weights.bytes or TestData.")
        print(f"Looking in: {script_dir}")