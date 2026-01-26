#!/usr/bin/env python3
"""
Neural SPAI Generator
Learns to generate a sparse matrix G such that A^-1 ≈ G * G^T + εI.
Architecture: 1D Windowed Transformer.
Loss: Unsupervised Scale-Invariant Aligned Identity (SAI) Loss via Stochastic Trace Estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from scipy.sparse import coo_matrix
import argparse
import struct
import time
import os

# --- 1. Physics Helper: Stencil Computation ---

def compute_laplacian_stencil(neighbors, layers, num_nodes):
    """
    Computes the explicit stencil weights (Matrix A) for the negative Laplacian.
    Returns: A_diag [N, 1], A_off [N, 24]
    """
    device = neighbors.device
    N = num_nodes
    IDX_AIR = N
    IDX_WALL = N + 1
    
    # Expand inputs for broadcasting
    # layers: [N, 1] -> 2^layer gives cell size dx
    dx = torch.pow(2.0, layers.float()).view(N, 1, 1)
    
    # Safe neighbor lookup (clamp invalid indices to 0 for gathering)
    safe_nbrs = neighbors.long().clone()
    safe_nbrs[safe_nbrs >= N] = 0
    
    # Get neighbor layers
    layers_flat = layers.view(-1)
    nbr_layers = layers_flat[safe_nbrs] # [N, 24]
    dx_nbrs = torch.pow(2.0, nbr_layers.float()).view(N, 6, 4)
    
    # Reshape neighbors to [N, 6 faces, 4 sub-neighbors]
    raw_nbrs = neighbors.view(N, 6, 4)
    is_wall = (raw_nbrs == IDX_WALL)
    is_air  = (raw_nbrs == IDX_AIR)
    is_fluid = (raw_nbrs < N)
    
    # --- Logic for Mixed-Resolution Interfaces ---
    # We need to distinguish between 'Coarse/Same' neighbors (Slot 0) and 'Fine' neighbors (Slots 0-3)
    
    # Slot 0 Logic: Active if neighbor is fluid AND (Same Layer OR Coarser Layer)
    self_layer_exp = layers.view(N, 1, 1).expand(N, 6, 4)
    is_coarse_or_same = is_fluid & (nbr_layers.view(N, 6, 4) >= self_layer_exp)
    slot0_active = is_coarse_or_same[:, :, 0] # Boolean [N, 6]
    
    # Calculate Distances & Weights
    # Case 1: Coarse/Same (Distance between centers)
    dist_c = torch.maximum(0.5 * (dx + dx_nbrs), torch.tensor(1e-6, device=device))
    weight_c = (dx**2) / dist_c
    
    # Case 2: Fine/Air (Distance to interface or fine neighbor center)
    # For Dirichlet (Air), distance is effectively dx/2
    dx_target = dx_nbrs.clone()
    dx_target[is_air] = dx.expand(N, 6, 4)[is_air] * 0.5 
    
    dist_s = torch.maximum(0.5 * (dx + dx_target), torch.tensor(1e-6, device=device))
    weight_s = (dx**2 / 4.0) / dist_s # Area is 1/4th
    
    # Accumulate A (Off-Diagonals are negative)
    A_off = torch.zeros(N, 6, 4, device=device)
    
    # Apply Coarse/Same Weights (Only to Slot 0)
    mask_c = slot0_active.unsqueeze(2).expand(N, 6, 4).clone()
    mask_c[:, :, 1:] = False # Only valid for slot 0
    A_off[mask_c] = -weight_c[mask_c]
    
    # Apply Fine/Air Weights (Valid for all slots if not blocked by wall or coarse neighbor)
    mask_sub = (~slot0_active.unsqueeze(2).expand(N, 6, 4)) & (~is_wall)
    A_off[mask_sub] = -weight_s[mask_sub]
    
    # Flatten
    A_off_flat = A_off.view(N, 24)
    
    # Diagonal is sum of absolute off-diagonals (Laplacian property)
    # Includes weights to Air (Dirichlet)
    A_diag = torch.sum(torch.abs(A_off_flat), dim=1, keepdim=True)
    
    # Zero out flux to Air in the sparse matrix
    # (Because p_air is fixed at 0, it doesn't appear as a column in the system matrix)
    is_air_flat = (neighbors == IDX_AIR)
    A_off_flat[is_air_flat] = 0.0
    
    return A_diag, A_off_flat

# --- 2. Architecture: Hierarchical V-Cycle Transformer ---

def apply_rope(x, freqs_cis):
    B, nhead, N, head_dim = x.shape
    x_reshaped = x.view(B, nhead, N, head_dim // 2, 2)
    if len(freqs_cis.shape) == 3:
        cos_part = freqs_cis[:, :, 0]
        sin_part = freqs_cis[:, :, 1]
    else:
        cos_part = freqs_cis.real if hasattr(freqs_cis, 'real') else freqs_cis
        sin_part = freqs_cis.imag if hasattr(freqs_cis, 'imag') else torch.zeros_like(freqs_cis)
    cos_part = cos_part.unsqueeze(0).unsqueeze(0)
    sin_part = sin_part.unsqueeze(0).unsqueeze(0)
    x0, x1 = x_reshaped[..., 0], x_reshaped[..., 1]
    x_rotated = torch.stack([x0 * cos_part - x1 * sin_part, x0 * sin_part + x1 * cos_part], dim=-1)
    return x_rotated.view(B, nhead, N, head_dim)

def precompute_freqs_cis(seq_len, head_dim, base=10000.0, device='cpu'):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)
    return torch.stack([cos_freqs, sin_freqs], dim=-1)

class HierarchicalScheduler:
    def __init__(self, k=32, min_leaf_size=128):
        self.k = k
        self.min_leaf_size = min_leaf_size
    
    def compute_schedule(self, seq_len):
        depth = 0
        padding_schedule = []
        current_len = seq_len
        while current_len > self.min_leaf_size:
            remainder = current_len % self.k
            padding = (self.k - remainder) % self.k
            padding_schedule.append(padding)
            current_len = (current_len + padding) // self.k
            depth += 1
        if depth == 0: return 0, []
        return depth, padding_schedule

class DownSampler(nn.Module):
    def __init__(self, d_model, k, expansion=2):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.local_mixer = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.ReLU(),
            nn.Linear(d_model * expansion, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        B, N, D = x.shape
        assert N % self.k == 0, f"N ({N}) must be divisible by k ({self.k})"
        x_grouped = x.view(B, N // self.k, self.k, D)
        x_mixed = self.local_mixer(x_grouped)
        x_pooled = x_mixed.mean(dim=2)
        x_pooled = self.norm(x_pooled)
        return x_pooled

class BottleneckAttention(nn.Module):
    def __init__(self, d_model, nhead, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.max_seq_len = max_seq_len
        self.register_buffer('freqs_cis', precompute_freqs_cis(max_seq_len, self.head_dim))
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        B, N, D = x.shape
        if N > self.max_seq_len:
            freqs_cis = precompute_freqs_cis(N, self.head_dim, device=x.device)
        else:
            freqs_cis = self.freqs_cis[:N]
        x_norm = self.norm(x)
        q = self.q_proj(x_norm).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        q_rope = apply_rope(q, freqs_cis)
        k_rope = apply_rope(k, freqs_cis)
        scores = torch.matmul(q_rope, k_rope.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        output = self.out_proj(attn_output)
        return x + output

class UpSampler(nn.Module):
    def __init__(self, d_model, k, expansion=2):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.up_proj = nn.Linear(d_model, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * expansion),
            nn.ReLU(),
            nn.Linear(d_model * expansion, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x_coarse, x_fine):
        B, N_coarse, D = x_coarse.shape
        _, N_fine, _ = x_fine.shape
        assert N_fine == N_coarse * self.k, f"Size mismatch: {N_fine} != {N_coarse} * {self.k}"
        x_coarse_proj = self.up_proj(x_coarse)
        x_coarse_expanded = x_coarse_proj.repeat_interleave(self.k, dim=1)
        x_concat = torch.cat([x_fine, x_coarse_expanded], dim=-1)
        x_fused = self.fusion(x_concat)
        x_fused = self.norm(x_fused)
        return x_fused

class HierarchicalVCycleTransformer(nn.Module):
    def __init__(self, input_dim=58, d_model=32, nhead=4, k=32, min_leaf_size=128, max_octree_depth=12):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.min_leaf_size = min_leaf_size
        self.scheduler = HierarchicalScheduler(k=k, min_leaf_size=min_leaf_size)
        self.feature_proj = nn.Linear(input_dim, d_model)
        self.layer_embed = nn.Embedding(max_octree_depth, d_model)
        self.down_samplers = nn.ModuleList()
        self.bottleneck = BottleneckAttention(d_model, nhead)
        self.up_samplers = nn.ModuleList()
        max_depth = 5
        for _ in range(max_depth):
            self.down_samplers.append(DownSampler(d_model, k))
            self.up_samplers.append(UpSampler(d_model, k))
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 25)
        with torch.no_grad():
            self.head.weight.normal_(0.0, 0.001)
            self.head.bias.uniform_(-0.01, 0.01)
            self.head.bias[0].fill_(1.0)
    
    def forward(self, x, layers):
        B, N, C = x.shape
        h = self.feature_proj(x) + self.layer_embed(layers)
        depth, padding_schedule = self.scheduler.compute_schedule(N)
        if depth == 0:
            h = self.bottleneck(h)
            coeffs = self.head(self.norm_out(h))
            return coeffs
        skip_stack = []
        residual_stack = []
        current = h
        for level in range(depth):
            residual_stack.append(current.clone())
            padding = padding_schedule[level]
            if padding > 0: current = F.pad(current, (0, 0, 0, padding))
            skip_stack.append(current.clone())
            down_sampler = self.down_samplers[level] if level < len(self.down_samplers) else self.down_samplers[-1]
            current = down_sampler(current)
        current = self.bottleneck(current)
        for level in range(depth - 1, -1, -1):
            x_fine = skip_stack.pop()
            x_residual = residual_stack.pop()
            up_sampler = self.up_samplers[level] if level < len(self.up_samplers) else self.up_samplers[-1]
            current = up_sampler(current, x_fine)
            padding = padding_schedule[level]
            if padding > 0: current = current[:, :current.shape[1] - padding, :]
            current = current + x_residual
        coeffs = self.head(self.norm_out(current))
        return coeffs

# --- 3. Unsupervised SAI Loss ---

class SAILoss(nn.Module):
    def __init__(self, epsilon=1e-4, num_probe_vectors=1):
        super().__init__()
        self.epsilon = epsilon
        self.num_probe_vectors = num_probe_vectors

    def forward(self, G_coeffs, A_diag, A_off, neighbors, valid_mask):
        B, N, _ = G_coeffs.shape
        device = G_coeffs.device
        active_count = valid_mask.sum()
        norm_A = (torch.abs(A_diag).sum() + torch.abs(A_off).sum()) / (active_count * 25 + 1e-6)
        total_loss = 0.0
        for _ in range(self.num_probe_vectors):
            w = torch.randn(B, N, 1, device=device) * valid_mask
            u = self.apply_GT(w, G_coeffs, neighbors, N)
            v_part = self.apply_G(u, G_coeffs, neighbors, N)
            v = v_part + self.epsilon * w
            y = self.apply_A(v, A_diag, A_off, neighbors, N)
            z = y / (norm_A + 1e-8)
            diff = (z - w) * valid_mask
            loss = (diff ** 2).sum() / (valid_mask.sum() + 1e-6)
            total_loss += loss
        return total_loss / self.num_probe_vectors

    def apply_G(self, x, coeffs, neighbors, N):
        B = x.shape[0]
        res = x * coeffs[:, :, 0:1]
        safe_nbrs = neighbors.clone()
        safe_nbrs[safe_nbrs >= N] = 0
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1) * N
        flat_idx = (safe_nbrs + batch_idx).view(-1)
        x_flat = x.view(-1, 1)
        x_nbrs = x_flat[flat_idx].view(B, N, 24)
        mask = (neighbors < N).float()
        x_nbrs = x_nbrs * mask
        res += (x_nbrs * coeffs[:, :, 1:]).sum(dim=2, keepdim=True)
        return res

    def apply_GT(self, x, coeffs, neighbors, N):
        B = x.shape[0]
        res = x * coeffs[:, :, 0:1]
        vals = x * coeffs[:, :, 1:]
        target_idx = neighbors.clone()
        mask = target_idx < N
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1) * N
        flat_target = (target_idx + batch_idx).view(-1)
        flat_vals = vals.view(-1)
        flat_mask = mask.view(-1)
        res_flat = res.view(-1)
        valid_indices = flat_target[flat_mask]
        valid_values = flat_vals[flat_mask]
        if valid_indices.numel() > 0:
            res_flat.index_add_(0, valid_indices, valid_values)
        return res_flat.view(B, N, 1)

    def apply_A(self, x, A_diag, A_off, neighbors, N):
        coeffs = torch.cat([A_diag, A_off], dim=2)
        return self.apply_G(x, coeffs, neighbors, N)

# --- 4. Dataset ---

class FluidGraphDataset(Dataset):
    def __init__(self, root_dirs):
        self.frame_paths = []
        for d in root_dirs:
            d = Path(d)
            if not d.exists(): continue
            all_items = list(d.iterdir())
            run_dirs = sorted([f for f in all_items if f.is_dir() and f.name.startswith('Run_')])
            if run_dirs:
                for run_dir in run_dirs:
                    frames = sorted([f for f in run_dir.iterdir() if f.is_dir() and f.name.startswith('frame_')])
                    self.frame_paths.extend(frames)
            else:
                frames = sorted([f for f in all_items if f.is_dir() and f.name.startswith('frame_')])
                self.frame_paths.extend(frames)

        max_n = 0
        for p in self.frame_paths:
            try:
                with open(p / "meta.txt", 'r') as f:
                    for line in f:
                        if 'numNodes' in line: max_n = max(max_n, int(line.split(':')[1].strip()))
            except: continue
        
        window_size = 256
        self.max_nodes = ((max_n + window_size - 1) // window_size) * window_size
        print(f"Dataset: {len(self.frame_paths)} frames. Max Nodes padded to: {self.max_nodes}")
        
        self.node_dtype = np.dtype([
            ('position', '3<f4'), ('velocity', '3<f4'), ('face_vels', '6<f4'),
            ('mass', '<f4'), ('layer', '<u4'), ('morton', '<u4'), ('active', '<u4')
        ])

    def __len__(self): return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        
        num_nodes = 0
        with open(frame_path / "meta.txt", 'r') as f:
            for line in f:
                if 'numNodes' in line: num_nodes = int(line.split(':')[1].strip())
        if num_nodes == 0: return self.__getitem__((idx+1)%len(self))

        def read_bin(name, dtype):
            return np.fromfile(frame_path / name, dtype=dtype)
            
        raw_nodes = read_bin("nodes.bin", self.node_dtype)
        # Correct SoA read: (24, N) -> Transpose to (N, 24)
        raw_nbrs = read_bin("neighbors.bin", np.uint32).reshape(24, num_nodes).T
        
        # 3. Compute Physics (Matrix A) on CPU for Input Features
        t_nbrs = torch.from_numpy(raw_nbrs.astype(np.int64))
        t_layers = torch.from_numpy(raw_nodes['layer'][:num_nodes].astype(np.int64)).unsqueeze(1)
        A_diag, A_off = compute_laplacian_stencil(t_nbrs, t_layers, num_nodes)
        
        # 3b. FAST Symmetrization: A_sym = (A + A.T) / 2
        # Replaces the O(NNZ) Python loop with vectorized CSR extraction
        
        # Build Sparse COO from current tensors
        rows = []
        cols = []
        data = []
        
        A_off_np = A_off.numpy()
        nbrs_np = raw_nbrs.astype(np.int64)
        
        # Flatten and filter for COO construction
        # This is O(N * 24) using numpy vectorization
        row_indices = np.repeat(np.arange(num_nodes), 24)
        col_indices = nbrs_np.flatten()
        vals_flat = A_off_np.flatten()
        
        valid_mask = col_indices < num_nodes
        # Also filter extremely small values to keep sparsity clean
        value_mask = np.abs(vals_flat) > 1e-9
        final_mask = valid_mask & value_mask
        
        A_coo = coo_matrix(
            (vals_flat[final_mask], (row_indices[final_mask], col_indices[final_mask])), 
            shape=(num_nodes, num_nodes)
        )
        
        # Symmetrize
        A_sym = (A_coo + A_coo.T) / 2.0
        
        # Write back to A_off tensor using vectorized CSR lookup
        A_off.fill_(0.0)
        A_off_np_out = A_off.numpy()
        
        # Convert symmetric matrix to CSR for fast indexing
        A_sym_csr = A_sym.tocsr()
        
        # Iterate over the 24 fixed slots (constant time loop)
        for k in range(24):
            # Target neighbors for all nodes in slot k
            target_cols = nbrs_np[:, k]
            
            # Mask of where we have a valid neighbor
            valid_locs = target_cols < num_nodes
            
            # Identify (row, col) pairs
            valid_rows = np.where(valid_locs)[0]
            valid_targets = target_cols[valid_locs]
            
            if len(valid_rows) > 0:
                # Fast extract: A_sym[row, col]
                # Scipy fancy indexing A[r, c] returns a matrix, we extract array
                # Note: reshaping to -1 ensures we get a 1D array
                vals = np.array(A_sym_csr[valid_rows, valid_targets]).reshape(-1)
                A_off_np_out[valid_rows, k] = vals
                
        # Recalculate diagonal
        A_diag_np = np.sum(np.abs(A_off_np_out), axis=1, keepdims=True)
        A_diag = torch.from_numpy(A_diag_np.astype(np.float32))
        
        # 4. Build Input Features (Dim 58)
        pos = raw_nodes['position'][:num_nodes] / 1024.0
        IDX_WALL = num_nodes + 1
        is_wall_full = (raw_nbrs == IDX_WALL)
        wall_indices = [0, 4, 8, 12, 16, 20]
        wall_flags = is_wall_full[:, wall_indices].astype(np.float32)
        IDX_AIR = num_nodes
        air_flags = (raw_nbrs == IDX_AIR).astype(np.float32)
        
        features_np = np.column_stack([
            pos,                  # 3
            wall_flags,           # 6
            air_flags,            # 24
            A_diag.numpy(),       # 1
            A_off.numpy()         # 24
        ])
        
        # 5. Padding
        padded_features = np.zeros((self.max_nodes, 58), dtype=np.float32)
        padded_features[:num_nodes] = features_np
        padded_layers = np.zeros((self.max_nodes,), dtype=np.int64)
        padded_layers[:num_nodes] = t_layers.squeeze().numpy()
        padded_nbrs = np.full((self.max_nodes, 24), self.max_nodes, dtype=np.int64)
        padded_nbrs[:num_nodes] = raw_nbrs
        padded_A_diag = np.zeros((self.max_nodes, 1), dtype=np.float32)
        padded_A_diag[:num_nodes] = A_diag.numpy()
        padded_A_off = np.zeros((self.max_nodes, 24), dtype=np.float32)
        padded_A_off[:num_nodes] = A_off.numpy()
        valid_mask = np.zeros((self.max_nodes, 1), dtype=np.float32)
        valid_mask[:num_nodes] = 1.0
        
        return {
            'x': padded_features,
            'layers': padded_layers,
            'neighbors': padded_nbrs,
            'A_diag': padded_A_diag,
            'A_off': padded_A_off,
            'mask': valid_mask
        }

# --- 5. Export Utility ---

def export_weights(model, path):
    print(f"Exporting packed 16-bit weights to {path}...")
    with open(path, 'wb') as f:
        num_levels = len(model.down_samplers)
        nhead = model.bottleneck.nhead
        header = struct.pack('<ffiiii', 0.0, 0.0, model.d_model, nhead, num_levels, 58)
        f.write(header)
        def write_packed_tensor(tensor):
            if len(tensor.shape) == 2:
                t_data = tensor.t().detach().cpu().numpy().flatten()
            else:
                t_data = tensor.detach().cpu().numpy().flatten()
            if len(t_data) % 2 != 0: t_data = np.append(t_data, 0.0)
            data_packed = t_data.astype(np.float16).view(np.uint32)
            f.write(data_packed.tobytes())
        write_packed_tensor(model.feature_proj.weight)
        write_packed_tensor(model.feature_proj.bias)
        write_packed_tensor(model.layer_embed.weight)
        for down_sampler in model.down_samplers:
            write_packed_tensor(down_sampler.local_mixer[0].weight)
            write_packed_tensor(down_sampler.local_mixer[0].bias)
            write_packed_tensor(down_sampler.local_mixer[2].weight)
            write_packed_tensor(down_sampler.local_mixer[2].bias)
            write_packed_tensor(down_sampler.norm.weight)
            write_packed_tensor(down_sampler.norm.bias)
        bottleneck = model.bottleneck
        write_packed_tensor(bottleneck.q_proj.weight)
        write_packed_tensor(bottleneck.q_proj.bias)
        write_packed_tensor(bottleneck.k_proj.weight)
        write_packed_tensor(bottleneck.k_proj.bias)
        write_packed_tensor(bottleneck.v_proj.weight)
        write_packed_tensor(bottleneck.v_proj.bias)
        write_packed_tensor(bottleneck.out_proj.weight)
        write_packed_tensor(bottleneck.out_proj.bias)
        write_packed_tensor(bottleneck.norm.weight)
        write_packed_tensor(bottleneck.norm.bias)
        for up_sampler in model.up_samplers:
            write_packed_tensor(up_sampler.up_proj.weight)
            write_packed_tensor(up_sampler.up_proj.bias)
            write_packed_tensor(up_sampler.fusion[0].weight)
            write_packed_tensor(up_sampler.fusion[0].bias)
            write_packed_tensor(up_sampler.fusion[2].weight)
            write_packed_tensor(up_sampler.fusion[2].bias)
            write_packed_tensor(up_sampler.norm.weight)
            write_packed_tensor(up_sampler.norm.bias)
        write_packed_tensor(model.norm_out.weight)
        write_packed_tensor(model.norm_out.bias)
        write_packed_tensor(model.head.weight)
        write_packed_tensor(model.head.bias)
    print("Export complete.")

# --- 6. Training Loop ---

def train(args):
    if torch.backends.mps.is_available(): device = torch.device('mps')
    elif torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')
    print(f"Device: {device}")
    
    data_path = Path(args.data_folder)
    if not data_path.is_absolute():
        script_dir = Path(__file__).parent
        data_path = script_dir.parent / "StreamingAssets" / "TestData"
    
    dataset = FluidGraphDataset([data_path])
    if len(dataset) == 0:
        print("No data found! Check path.")
        return
    
    use_random_sampling = args.frames_per_epoch > 0
    # Enable multiprocessing for data loading
    # Use 0 for Windows/debugging if issues arise, otherwise 4-8 is good
    num_workers = min(os.cpu_count(), 8) if os.name != 'nt' else 0 
    print(f"Data Loader workers: {num_workers}")
    
    if use_random_sampling:
        loader = None
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                          num_workers=num_workers, pin_memory=True)
    
    model = HierarchicalVCycleTransformer(
        input_dim=58, 
        d_model=args.d_model, 
        nhead=args.nhead if hasattr(args, 'nhead') else 4,
        k=args.k if hasattr(args, 'k') else 32,
        min_leaf_size=args.min_leaf_size if hasattr(args, 'min_leaf_size') else 128
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = SAILoss(num_probe_vectors=args.num_probe_vectors)
    
    use_amp = device.type in ['cuda', 'mps']
    scaler = None
    if device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("AMP enabled with GradScaler (CUDA)")
    elif device.type == 'mps':
        print("AMP enabled (MPS)")
    else:
        print("AMP disabled (CPU)")
    
    try:
        model = torch.compile(model)
        criterion = torch.compile(criterion)
        print("Model and loss function compiled with torch.compile")
    except Exception as e:
        print(f"Warning: torch.compile failed ({e}), continuing without compilation")
    
    script_dir = Path(__file__).parent
    weights_path = script_dir / "model_weights.bytes"
    print("Starting Training...")
    
    use_cuda_events = device.type == 'cuda'
    if use_cuda_events:
        start_event = torch.cuda.Event(enable_timing=True)
        fwd_end_event = torch.cuda.Event(enable_timing=True)
        loss_end_event = torch.cuda.Event(enable_timing=True)
    
    epoch_times = []
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        
        if use_random_sampling:
            num_frames = min(args.frames_per_epoch, len(dataset))
            indices = torch.randperm(len(dataset))[:num_frames].tolist()
            sampler = SubsetRandomSampler(indices)
            loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
            print(f"Epoch {epoch+1}: Using {num_frames} randomly selected frames")
        
        total_fwd_time = 0.0
        total_loss_time = 0.0
        
        for batch in loader:
            x = batch['x'].to(device, non_blocking=True)
            layers = batch['layers'].to(device, non_blocking=True)
            nbrs = batch['neighbors'].to(device, non_blocking=True)
            A_diag = batch['A_diag'].to(device, non_blocking=True)
            A_off = batch['A_off'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if use_cuda_events: start_event.record()
            else: fwd_start = time.time()
            
            if use_amp:
                device_type = 'cuda' if device.type == 'cuda' else 'mps'
                with torch.amp.autocast(device_type=device_type):
                    G_coeffs = model(x, layers)
            else:
                G_coeffs = model(x, layers)
            
            if use_cuda_events: fwd_end_event.record()
            else: fwd_end = time.time()
            
            if use_amp:
                device_type = 'cuda' if device.type == 'cuda' else 'mps'
                with torch.amp.autocast(device_type=device_type):
                    loss = criterion(G_coeffs, A_diag, A_off, nbrs, mask)
            else:
                loss = criterion(G_coeffs, A_diag, A_off, nbrs, mask)
            
            if use_cuda_events: loss_end_event.record()
            else: loss_end = time.time()
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss.item()
            
            if use_cuda_events:
                torch.cuda.synchronize()
                fwd = start_event.elapsed_time(fwd_end_event)
                loss = fwd_end_event.elapsed_time(loss_end_event)
            else:
                fwd = (fwd_end - fwd_start) * 1000.0
                loss = (loss_end - fwd_end) * 1000.0
            
            total_fwd_time += fwd
            total_loss_time += loss
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_time = sum(epoch_times) / len(epoch_times)
        avg_fwd = total_fwd_time / len(loader)
        avg_loss_time = total_loss_time / len(loader)
        avg_loss = total_loss / len(loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s (forward: {avg_fwd:.1f}ms, loss: {avg_loss_time:.1f}ms) | Avg Time: {avg_time:.2f}s")
        
        if (epoch + 1) % 5 == 0:
            export_weights(model, weights_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_data_path = Path(__file__).parent.parent / "StreamingAssets" / "TestData"
    parser.add_argument('--data_folder', type=str, default=str(default_data_path))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--min_leaf_size', type=int, default=128)
    parser.add_argument('--num_probe_vectors', type=int, default=1)
    parser.add_argument('--frames_per_epoch', type=int, default=0)
    args = parser.parse_args()
    
    # Enable Windows multiprocessing fix
    if os.name == 'nt':
        import multiprocessing
        multiprocessing.freeze_support()
        
    train(args)