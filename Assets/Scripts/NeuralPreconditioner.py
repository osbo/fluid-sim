#!/usr/bin/env python3
"""
Neural HODLR Preconditioner (Hierarchical Off-Diagonal Low-Rank)
Updated: Dense Leaf Blocks (32x32) for High Performance.
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

# --- 1. Core Components (Rotary Embeddings) ---

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
        # Pad if N is not divisible by k
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
        
        # Handle shape mismatches due to padding in downsample
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
    def __init__(self, d_model, nhead, max_seq_len=8192):
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
            # Dynamic resize if sequence is longer than expected
            freqs_cis = precompute_freqs_cis(N, self.head_dim, device=x.device)
        else:
            freqs_cis = self.freqs_cis[:N]
            
        x_norm = self.norm(x)
        q = self.q_proj(x_norm).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        
        q_rope = apply_rope(q, freqs_cis)
        k_rope = apply_rope(k, freqs_cis)
        
        # Scaled Dot Product Attention
        scores = torch.matmul(q_rope, k_rope.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        return x + self.out_proj(attn_output)

# --- 3. Heads ---

class HODLRLevelHead(nn.Module):
    def __init__(self, d_model, rank):
        super().__init__()
        self.rank = rank
        self.proj_u = nn.Linear(d_model, rank)
        
    def forward(self, x):
        u = self.proj_u(x)
        return u, u 

class BlockDiagonalHead(nn.Module):
    """
    Predicts a Dense Block L (Lower Triangular) for each leaf block.
    Leaf Size: M (e.g. 32)
    Output: M x M block per leaf.
    """
    def __init__(self, d_model, leaf_size=32):
        super().__init__()
        self.leaf_size = leaf_size
        # Number of elements in a triangular matrix (including diagonal) = M*(M+1)/2
        # However, for simplicity and GPU batching, we predict the full MxM and mask it.
        # This allows us to predict a full dense block if we wanted, but Cholesky is safer.
        self.out_dim = leaf_size * leaf_size
        
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, self.out_dim)
        )
        
        # Initialize near Identity
        # We want L * L^T ≈ I  => L ≈ I
        with torch.no_grad():
            self.net[-1].weight.data *= 0.01
            self.net[-1].bias.data.fill_(0)
            # Set diagonal elements to 1.0 (approx)
            # This is complex in a flattened array, so we do it in forward or just let it learn.
            # A simple bias init helps convergence:
            eye = torch.eye(leaf_size).flatten()
            self.net[-1].bias.data += eye

    def forward(self, x):
        # x is at the FINEST level (N nodes)
        # We need to process it in chunks of 'leaf_size'
        # But wait: x has shape [B, N, D].
        # We want one L matrix for every 'leaf_size' block of nodes.
        # So we pool x by 'leaf_size' then predict?
        # BETTER: Predict per-node, then reshape?
        # No, a block is a property of a group.
        
        B, N, D = x.shape
        num_blocks = N // self.leaf_size
        
        if N % self.leaf_size != 0:
            # Handle edge case: pad or truncate?
            # For HODLR, N is usually power of 2 aligned. 
            pass

        # Reshape to [B, NumBlocks, LeafSize, D]
        x_blocked = x.view(B, num_blocks, self.leaf_size, D)
        
        # Average features across the block to get a "Block Embedding"
        x_pooled = x_blocked.mean(dim=2) # [B, NumBlocks, D]
        
        # Predict L
        L_flat = self.net(x_pooled) # [B, NumBlocks, LeafSize*LeafSize]
        L = L_flat.view(B, num_blocks, self.leaf_size, self.leaf_size)
        
        # Enforce Lower Triangular (for stability) or Full Symmetric
        # Let's do Full Symmetric via L * L^T + epsilon * I
        # This guarantees PSD.
        return L 

# --- 4. Main Architecture ---

class NeuralHODLR(nn.Module):
    def __init__(self, input_dim=4, d_model=64, nhead=4, k=2, rank=32, depth=5, leaf_size=32):
        super().__init__()
        self.d_model = d_model
        self.k = k 
        self.depth = depth
        self.leaf_size = leaf_size
        self.max_rank = rank
        
        # --- Rank Schedule ---
        self.rank_schedule = []
        for i in range(depth):
            dist_from_top = (depth - 1) - i
            calc_rank = max(4, rank // (2 ** dist_from_top))
            self.rank_schedule.append(calc_rank)
            
        print("\n=== HODLR Architecture (Dense Leaves) ===")
        print(f"Depth: {depth}, Max Rank: {rank}, Leaf Size: {leaf_size}")
        print(f"Hierarchy:")
        for i, r in enumerate(self.rank_schedule):
            print(f"  L{i}: Rank {r}")
        print("=========================================\n")

        self.feature_proj = nn.Linear(input_dim, d_model)
        
        self.down_samplers = nn.ModuleList()
        self.up_samplers = nn.ModuleList()
        self.hodlr_heads = nn.ModuleList()
        
        for i in range(depth):
            self.down_samplers.append(DownSampler(d_model, k))
            self.up_samplers.append(UpSampler(d_model, k))
            self.hodlr_heads.append(HODLRLevelHead(d_model, rank=self.rank_schedule[i]))
            
        self.bottleneck = BottleneckAttention(d_model, nhead)
        
        # Dense Block Head instead of Scalar Diagonal
        self.leaf_head = BlockDiagonalHead(d_model, leaf_size)
        
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.feature_proj(x)
        skip_stack = []
        
        for layer in self.down_samplers:
            skip_stack.append(h)
            h = layer(h)
            
        h = self.bottleneck(h)
        
        hodlr_factors = []
        for i in range(self.depth - 1, -1, -1):
            u, v = self.hodlr_heads[i](h)
            hodlr_factors.append((u, v))
            
            skip = skip_stack.pop()
            h = self.up_samplers[i](h, skip)
            
        h_final = self.norm_out(h)
        
        # Predict Dense Blocks
        # h_final is at the finest level (N nodes)
        leaf_blocks = self.leaf_head(h_final)
        
        return leaf_blocks, hodlr_factors

# --- 5. HODLR Operations (Updated for Dense Blocks) ---

def apply_hodlr_matrix(leaf_blocks, factors, x, leaf_size=32):
    """
    x: [B, N, 1]
    leaf_blocks: [B, NumBlocks, M, M] where M=leaf_size
    """
    B, N, _ = x.shape
    
    # 1. Apply Dense Leaf Blocks
    # Reshape x to match blocks: [B, NumBlocks, M, 1]
    num_blocks = N // leaf_size
    x_blocked = x.view(B, num_blocks, leaf_size, 1)
    
    # Block Matrix Vector Multiply
    # y = L * L^T * x (Symmetric PSD)
    # Or just B * x if we predict full B.
    # Let's use B * B^T + eps * I for stability
    
    L = leaf_blocks
    # L is [B, NumBlocks, M, M]
    
    # y_block = L @ (L.transpose(-2, -1) @ x_blocked)
    temp = torch.matmul(L.transpose(-2, -1), x_blocked)
    y_blocked = torch.matmul(L, temp)
    
    # Add small identity for numerical stability if needed
    y_blocked = y_blocked + x_blocked * 1e-4
    
    y = y_blocked.view(B, N, 1)
    
    # 2. Apply HODLR Off-Diagonals (Same as before)
    # Iterate Coarse -> Fine
    for level_idx, (u_coarse, v_coarse) in enumerate(factors):
        num_splits = 2 ** (level_idx + 1)
        block_size = N // num_splits
        if block_size == 0: continue
        
        if u_coarse.shape[1] != N:
            u_full = F.interpolate(u_coarse.transpose(1,2), size=N, mode='linear', align_corners=False).transpose(1,2)
            v_full = F.interpolate(v_coarse.transpose(1,2), size=N, mode='linear', align_corners=False).transpose(1,2)
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
            y = y + F.pad(updates, (0,0,0,N-valid_len))
        else:
            y = y + updates
            
    return y

class HODLRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, leaf_blocks, factors, A_sparse_pack, num_nodes):
        A_indices, A_values = A_sparse_pack
        B = 1
        N = num_nodes
        
        # Stochastic Probing
        z = torch.randn(B, N, 1, device=leaf_blocks.device)
        
        # M * z
        w = apply_hodlr_matrix(leaf_blocks, factors, z, leaf_size=32)
        
        # A * (M * z)
        target_device = leaf_blocks.device
        mm_device = torch.device('cpu') if target_device.type == 'mps' else target_device
        
        w_flat = w.squeeze().unsqueeze(1).to(mm_device)
        A_indices_mm = A_indices.to(mm_device)
        A_values_mm = A_values.to(mm_device)
        A_sparse = torch.sparse_coo_tensor(A_indices_mm, A_values_mm, (N, N), device=mm_device)
        
        y_flat = torch.sparse.mm(A_sparse, w_flat)
        y = y_flat.view(B, N, 1).to(target_device)
        
        # Loss: || A*M*z - z ||
        loss = F.mse_loss(y, z)
        return loss

# --- 6. Dataset ---
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
            morton = raw_nodes['morton']
            perm = np.argsort(morton)
            inv_perm = np.empty_like(perm)
            inv_perm[perm] = np.arange(len(perm))
            pos = raw_nodes['position'][perm]
            new_rows = inv_perm[rows]
            new_cols = inv_perm[cols]
            
            # --- PAD TO 32 ---
            # Our dense blocks require N % 32 == 0
            # We will pad the graph with dummy isolated nodes if needed
            leaf_size = 32
            pad = (leaf_size - (num_nodes % leaf_size)) % leaf_size
            
            if pad > 0:
                # Pad positions
                pos = np.pad(pos, ((0, pad), (0, 0)), mode='constant')
                # Pad diagonal with 1.0 (Identity for dummy nodes)
                # Pad rows/cols indices? No, just let sparse matrix handle it implicitly
                # Wait, sparse mm requires size match.
                num_nodes += pad
            
            mask_diag = new_rows == new_cols
            diag_map = np.zeros(num_nodes, dtype=np.float32)
            # Fill diag_map only for original nodes
            # Dummy nodes will be 0.0 here, but we can fix in feature engineering
            
            r_diag = new_rows[mask_diag]
            v_diag = vals[mask_diag]
            diag_map[r_diag] = v_diag
            
            # Pad diag map for dummy nodes with 1.0?
            if pad > 0:
                diag_map[-pad:] = 1.0
            
            x = np.column_stack([pos, diag_map]).astype(np.float32)
            
            return {
                'x': x,
                'edge_index': torch.stack([torch.from_numpy(new_rows).long(), torch.from_numpy(new_cols).long()]),
                'edge_values': torch.from_numpy(vals),
                'num_nodes': num_nodes
            }
        except Exception as e:
            print(f"Error loading {frame_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

# --- 7. Training & Export ---

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')
    print(f"Device: {device}")
    
    data_path = Path(args.data_folder)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / "StreamingAssets" / "TestData"
        
    dataset = FluidGraphDataset([data_path])
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    model = NeuralHODLR(
        d_model=args.d_model, 
        depth=args.depth, 
        rank=args.rank,     
        leaf_size=32 # Hardcoded for now
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = HODLRLoss()
    
    print("Starting HODLR Training (Dense Leaf Blocks 32x32)...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for i, batch in enumerate(loader):
            x = batch['x'].to(device)
            num_nodes = batch['num_nodes'].item()
            A_indices = batch['edge_index'][0].to(device)
            A_values = batch['edge_values'][0].to(device)
            
            # Safety Check: Matrix sizes must match padded N
            # Our Dataset pads x, but sparse indices are original.
            # Sparse MM allows shape mismatch if we specify size (N, N)
            
            optimizer.zero_grad()
            leaf_blocks, factors = model(x)
            loss = criterion(leaf_blocks, factors, (A_indices, A_values), num_nodes)
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
        # Header: rank is MAX_RANK. leaf_size added as an implicit assumption or we can hijack a field.
        # Let's hijack 'k' (upsample factor) or just rely on convention.
        # Format: <ffiiiii (7 values)
        # 0,0, d_model, nhead, depth, input_dim, max_rank
        header = struct.pack('<ffiiiii', 0, 0, model.d_model, model.bottleneck.nhead, model.depth, 4, model.max_rank)
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
            pack(h.proj_u.weight); pack(h.proj_u.bias) 
            
        pack(model.norm_out.weight); pack(model.norm_out.bias)
        
        # Dense Leaf Head Export
        # It's an MLP: Linear -> Linear
        pack(model.leaf_head.net[0].weight); pack(model.leaf_head.net[0].bias)
        pack(model.leaf_head.net[2].weight); pack(model.leaf_head.net[2].bias)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default="")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--rank', type=int, default=32) 
    args = parser.parse_args()
    train(args)