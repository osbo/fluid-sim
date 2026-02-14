#!/usr/bin/env python3
"""
Neural HODLR Preconditioner (Hierarchical Off-Diagonal Low-Rank)
Learns to generate a hierarchical matrix inverse A^-1 ≈ HODLR.
FIXED: Enforces Symmetry, Smooth Upsampling, and Morton Sorting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import argparse
import struct
import time
import os

# --- 1. Core Components ---

def apply_rope(x, freqs_cis):
    B, nhead, N, head_dim = x.shape
    x_reshaped = x.view(B, nhead, N, head_dim // 2, 2)
    if freqs_cis.dim() == 3: 
        cos_part = freqs_cis[..., 0]
        sin_part = freqs_cis[..., 1]
    else:
        cos_part = freqs_cis.real
        sin_part = freqs_cis.imag
        
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

# --- 2. U-Net Components ---

class DownSampler(nn.Module):
    def __init__(self, d_model, k, expansion=2):
        super().__init__()
        self.k = k
        self.local_mixer = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Linear(d_model * expansion, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        B, N, D = x.shape
        pad = (self.k - (N % self.k)) % self.k
        if pad > 0: x = F.pad(x, (0, 0, 0, pad))
        
        B, N_pad, _ = x.shape
        x_grouped = x.view(B, N_pad // self.k, self.k, D)
        x_mixed = self.local_mixer(x_grouped)
        x_pooled = x_mixed.mean(dim=2)
        x_pooled = self.norm(x_pooled)
        return x_pooled

class UpSampler(nn.Module):
    def __init__(self, d_model, k, expansion=2):
        super().__init__()
        self.k = k
        self.up_proj = nn.Linear(d_model, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * expansion),
            nn.GELU(),
            nn.Linear(d_model * expansion, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x_coarse, x_fine):
        x_coarse_proj = self.up_proj(x_coarse)
        x_coarse_expanded = x_coarse_proj.repeat_interleave(self.k, dim=1)
        
        if x_coarse_expanded.shape[1] > x_fine.shape[1]:
            x_coarse_expanded = x_coarse_expanded[:, :x_fine.shape[1], :]
        elif x_coarse_expanded.shape[1] < x_fine.shape[1]:
            diff = x_fine.shape[1] - x_coarse_expanded.shape[1]
            x_coarse_expanded = F.pad(x_coarse_expanded, (0,0,0,diff))
            
        x_concat = torch.cat([x_fine, x_coarse_expanded], dim=-1)
        x_fused = self.fusion(x_concat)
        x_fused = self.norm(x_fused)
        return x_fused

class BottleneckAttention(nn.Module):
    def __init__(self, d_model, nhead, max_seq_len=4096):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.register_buffer('freqs_cis', precompute_freqs_cis(max_seq_len, self.head_dim))
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        B, N, D = x.shape
        if N > self.freqs_cis.shape[0]:
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
        return x + self.out_proj(attn_output)

# --- 3. HODLR Heads ---

class HODLRLevelHead(nn.Module):
    """
    Predicts Low-Rank factor U for symmetric approximation U*U^T.
    """
    def __init__(self, d_model, rank=4):
        super().__init__()
        self.rank = rank
        # Only U projection for symmetry
        self.proj_u = nn.Linear(d_model, rank)
        
    def forward(self, x):
        u = self.proj_u(x)
        return u, u # Return U, U for V=U symmetry

class DiagonalHead(nn.Module):
    """Predicts the sparse diagonal component"""
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Softplus()
        )
        # Initialize bias to +1.0 so we start near Identity
        nn.init.constant_(self.net[-2].bias, 1.0)

    def forward(self, x):
        return self.net(x) + 1e-4

# --- 4. Main Architecture ---

class NeuralHODLR(nn.Module):
    def __init__(self, input_dim=4, d_model=64, nhead=4, k=2, rank=4, depth=4):
        super().__init__()
        self.d_model = d_model
        self.k = k 
        self.depth = depth
        
        self.feature_proj = nn.Linear(input_dim, d_model)
        
        self.down_samplers = nn.ModuleList()
        self.up_samplers = nn.ModuleList()
        self.hodlr_heads = nn.ModuleList()
        
        for _ in range(depth):
            self.down_samplers.append(DownSampler(d_model, k))
            self.up_samplers.append(UpSampler(d_model, k))
            self.hodlr_heads.append(HODLRLevelHead(d_model, rank=rank))
            
        self.bottleneck = BottleneckAttention(d_model, nhead)
        self.diag_head = DiagonalHead(d_model)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.feature_proj(x)
        skip_stack = []
        
        # Downward
        for layer in self.down_samplers:
            skip_stack.append(h)
            h = layer(h)
            
        h = self.bottleneck(h)
        
        # Upward
        hodlr_factors = []
        for i in range(self.depth - 1, -1, -1):
            # Symmetry: predict U, set V=U
            u, v = self.hodlr_heads[i](h)
            hodlr_factors.append((u, v))
            
            skip = skip_stack.pop()
            h = self.up_samplers[i](h, skip)
            
        h_final = self.norm_out(h)
        diag = self.diag_head(h_final)
        
        return diag, hodlr_factors

# --- 5. HODLR Operations ---

def apply_hodlr_matrix(diag, factors, x, k=2):
    """
    Applies symmetric HODLR.
    FIX: Uses LINEAR interpolation to avoid blocky artifacts.
    """
    B, N, _ = x.shape
    y = diag * x
    
    # Iterate Coarse -> Fine (Deepest level first)
    for level_idx, (u_coarse, v_coarse) in enumerate(factors):
        num_splits = 2 ** (level_idx + 1)
        block_size = N // num_splits
        if block_size == 0: continue
        
        # 1. Smooth Upsampling (Linear instead of Nearest)
        # This fixes the "Checkerboard" artifacts
        if u_coarse.shape[1] != N:
            u_full = F.interpolate(u_coarse.transpose(1,2), size=N, mode='linear', align_corners=False).transpose(1,2)
            v_full = F.interpolate(v_coarse.transpose(1,2), size=N, mode='linear', align_corners=False).transpose(1,2)
        else:
            u_full, v_full = u_coarse, v_coarse
            
        # 2. Block Logic
        num_pairs = num_splits // 2
        valid_len = num_pairs * 2 * block_size
        
        x_view = x[:, :valid_len].view(B, num_pairs, 2, block_size, 1)
        u_view = u_full[:, :valid_len].view(B, num_pairs, 2, block_size, -1)
        v_view = v_full[:, :valid_len].view(B, num_pairs, 2, block_size, -1)
        
        x_L = x_view[:, :, 0]
        x_R = x_view[:, :, 1]
        u_L = u_view[:, :, 0]
        u_R = u_view[:, :, 1]
        v_L = v_view[:, :, 0] # v_L is u_L if symmetric
        v_R = v_view[:, :, 1]
        
        # Cross-Term Interactions
        # If Symmetric: Off-Diagonal = U_L * U_R^T
        # Left gets signal from Right: y_L += U_L * (V_R^T * x_R)
        
        x_R_proj = torch.matmul(v_R.transpose(-2, -1), x_R) # [B, P, R, 1]
        x_L_proj = torch.matmul(v_L.transpose(-2, -1), x_L)
        
        y_L_update = torch.matmul(u_L, x_R_proj)
        y_R_update = torch.matmul(u_R, x_L_proj)
        
        updates = torch.stack([y_L_update, y_R_update], dim=2).view(B, valid_len, 1)
        
        if valid_len < N:
            y = y + F.pad(updates, (0,0,0,N-valid_len))
        else:
            y = y + updates
            
    return y

class HODLRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, diag, factors, A_sparse_pack, num_nodes):
        A_indices, A_values = A_sparse_pack
        B = 1
        N = num_nodes
        
        # Probing
        z = torch.randn(B, N, 1, device=diag.device)
        w = apply_hodlr_matrix(diag, factors, z)
        
        # MPS/CPU device handling for Sparse MM
        target_device = diag.device
        mm_device = torch.device('cpu') if target_device.type == 'mps' else target_device
            
        w_flat = w.squeeze().unsqueeze(1).to(mm_device)
        A_indices_mm = A_indices.to(mm_device)
        A_values_mm = A_values.to(mm_device)
        
        A_sparse = torch.sparse_coo_tensor(A_indices_mm, A_values_mm, (N, N), device=mm_device)
        y_flat = torch.sparse.mm(A_sparse, w_flat)
        y = y_flat.view(B, N, 1).to(target_device)
        
        loss = F.mse_loss(y, z)
        return loss

# --- 6. Dataset (Sorting Added) ---

class FluidGraphDataset(Dataset):
    def __init__(self, root_dirs):
        self.frame_paths = []
        for d in root_dirs:
            d = Path(d)
            if d.exists():
                frames = sorted([f for f in d.rglob('nodes.bin')])
                self.frame_paths.extend([f.parent for f in frames])

        print(f"Dataset: Found {len(self.frame_paths)} frames.")
        self.node_dtype = np.dtype([
            ('position', '3<f4'), ('velocity', '3<f4'), ('face_vels', '6<f4'),
            ('mass', '<f4'), ('layer', '<u4'), ('morton', '<u4'), ('active', '<u4')
        ])

    def __len__(self): return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        try:
            raw_nodes = np.fromfile(frame_path / "nodes.bin", dtype=self.node_dtype)
            rows = np.fromfile(frame_path / "edge_index_rows.bin", dtype=np.uint32)
            cols = np.fromfile(frame_path / "edge_index_cols.bin", dtype=np.uint32)
            vals = np.fromfile(frame_path / "A_values.bin", dtype=np.float32)
            
            num_nodes = len(raw_nodes)
            
            # --- SORT BY MORTON CODE ---
            # HODLR requires 1D locality to match 3D locality
            morton = raw_nodes['morton']
            perm = np.argsort(morton)
            inv_perm = np.empty_like(perm)
            inv_perm[perm] = np.arange(len(perm))
            
            # Permute Features
            pos = raw_nodes['position'][perm]
            
            # Permute Sparse Matrix A -> P * A * P^T
            # Remap indices: new_idx = inv_perm[old_idx]
            new_rows = inv_perm[rows]
            new_cols = inv_perm[cols]
            
            # Re-sort the edge list for cleanliness (optional but good for sparse efficiency)
            # We just need correct indices for now
            
            # Extract Diagonal (after permuting, diagonal is still r==c)
            # But we need to find it in the new indices
            mask_diag = new_rows == new_cols
            diag_map = np.zeros(num_nodes, dtype=np.float32)
            
            # This is tricky: A_values are associated with (rows, cols)
            # We moved rows/cols. The values move with them.
            r_diag = new_rows[mask_diag]
            v_diag = vals[mask_diag]
            diag_map[r_diag] = v_diag
            
            x = np.column_stack([pos, diag_map]).astype(np.float32)
            
            return {
                'x': x,
                'edge_index': torch.stack([torch.from_numpy(new_rows).long(), torch.from_numpy(new_cols).long()]),
                'edge_values': torch.from_numpy(vals),
                'num_nodes': num_nodes
            }
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self))

# --- 7. Training Loop ---

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')
    print(f"Device: {device}")
    
    data_path = Path(args.data_folder)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / "StreamingAssets" / "TestData"
        
    dataset = FluidGraphDataset([data_path])
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    model = NeuralHODLR(d_model=args.d_model, depth=args.depth, rank=args.rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = HODLRLoss()
    
    print("Starting HODLR Training (Symmetric + Linear Upsampling)...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for i, batch in enumerate(loader):
            x = batch['x'].to(device)
            num_nodes = batch['num_nodes'].item()
            A_indices = batch['edge_index'][0].to(device) # Keep on device, move in Loss
            A_values = batch['edge_values'][0].to(device)
            
            optimizer.zero_grad()
            diag, factors = model(x)
            loss = criterion(diag, factors, (A_indices, A_values), num_nodes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            if i % 10 == 0:
                print(f"Ep {epoch} Step {i}: {loss.item():.5f}")
                export_weights(model, Path(__file__).parent / "model_weights.bytes")
                
        print(f"Epoch {epoch} Done. Avg Loss: {total_loss / len(loader):.5f}")

def export_weights(model, path):
    with open(path, 'wb') as f:
        # Header matching C# struct layout
        header = struct.pack('<ffiiiii', 0, 0, model.d_model, model.bottleneck.nhead, model.depth, 4, model.hodlr_heads[0].rank)
        f.write(header)
        
        def pack(t):
            d = t.detach().cpu().numpy().flatten()
            if len(t.shape)==2: d = t.t().detach().cpu().numpy().flatten()
            if len(d)%2!=0: d=np.append(d, 0.0)
            f.write(d.astype(np.float16).view(np.uint32).tobytes())
            
        pack(model.feature_proj.weight); pack(model.feature_proj.bias)
        for d in model.down_samplers:
            pack(d.local_mixer[0].weight); pack(d.local_mixer[0].bias)
            pack(d.local_mixer[2].weight); pack(d.local_mixer[2].bias)
            pack(d.norm.weight); pack(d.norm.bias)
        bn = model.bottleneck
        pack(bn.q_proj.weight); pack(bn.q_proj.bias); pack(bn.k_proj.weight); pack(bn.k_proj.bias)
        pack(bn.v_proj.weight); pack(bn.v_proj.bias); pack(bn.out_proj.weight); pack(bn.out_proj.bias)
        pack(bn.norm.weight); pack(bn.norm.bias)
        for u, h in zip(model.up_samplers, model.hodlr_heads):
            pack(u.up_proj.weight); pack(u.up_proj.bias)
            pack(u.fusion[0].weight); pack(u.fusion[0].bias)
            pack(u.fusion[2].weight); pack(u.fusion[2].bias)
            pack(u.norm.weight); pack(u.norm.bias)
            pack(h.proj_u.weight); pack(h.proj_u.bias)
            pack(h.proj_u.weight); pack(h.proj_u.bias) # Write U twice for V slot to maintain file format
        pack(model.norm_out.weight); pack(model.norm_out.bias)
        pack(model.diag_head.net[0].weight); pack(model.diag_head.net[0].bias)
        pack(model.diag_head.net[2].weight); pack(model.diag_head.net[2].bias)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default="")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--rank', type=int, default=4)
    args = parser.parse_args()
    train(args)