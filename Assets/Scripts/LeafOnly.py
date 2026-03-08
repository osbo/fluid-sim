"""
LeafOnly: train HODLR-style block preconditioner (32x32 diagonal leaves + off-diagonal low-rank blocks).
Attention is within each 32-node block. Run with python3 LeafOnly.py; view_size defaults to 512.
Add --mixed_sizes to train on dynamically varying block sizes for scale invariance.
"""
import warnings
# Suppress deprecation from torch._inductor (torch.compile path); fixed in newer PyTorch
warnings.filterwarnings("ignore", category=FutureWarning, module="torch._inductor")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import argparse
import struct
import random
from collections import deque
from pathlib import Path
import time

# --- Dataset (same format as InspectModel: x, edge_index, edge_values, num_nodes) ---

NODE_DTYPE = np.dtype([
    ('position', '3<f4'), ('velocity', '3<f4'), ('face_vels', '6<f4'),
    ('mass', '<f4'), ('layer', '<u4'), ('morton', '<u4'), ('active', '<u4')
])

def _read_num_nodes(frame_path):
    p = Path(frame_path)
    meta = p / "meta.txt"
    if meta.exists():
        with open(meta, 'r') as f:
            for line in f:
                if 'numNodes' in line or 'num_nodes' in line:
                    return int(line.split(':')[1].strip())
    nodes_bin = p / "nodes.bin"
    if nodes_bin.exists():
        return nodes_bin.stat().st_size // 64
    return 0

class FluidGraphDataset:
    def __init__(self, data_folders):
        self.frame_paths = []
        for folder in data_folders:
            folder = Path(folder)
            if not folder.exists():
                continue
            for nodes_file in folder.rglob("nodes.bin"):
                frame_dir = nodes_file.parent
                if (frame_dir / "edge_index_rows.bin").exists() and (frame_dir / "A_values.bin").exists():
                    self.frame_paths.append(frame_dir)
        self.frame_paths = sorted(self.frame_paths)

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_path = Path(self.frame_paths[idx])
        num_nodes = _read_num_nodes(frame_path)
        if num_nodes <= 0:
            raise ValueError(f"Invalid num_nodes at {frame_path}")

        raw_nodes = np.fromfile(frame_path / "nodes.bin", dtype=NODE_DTYPE)[:num_nodes]
        rows = np.fromfile(frame_path / "edge_index_rows.bin", dtype=np.uint32)
        cols = np.fromfile(frame_path / "edge_index_cols.bin", dtype=np.uint32)
        vals = np.fromfile(frame_path / "A_values.bin", dtype=np.float32)

        positions = np.asarray(raw_nodes['position'], dtype=np.float32)
        layer = np.asarray(raw_nodes['layer'], dtype=np.float32)
        mass = np.asarray(raw_nodes['mass'], dtype=np.float32)
        diag_map = np.zeros(num_nodes, dtype=np.float32)
        for r, c, v in zip(rows, cols, vals):
            if r == c:
                diag_map[r] = v

        # Fixed absolute scale for positions
        positions_n = positions / 1024.0
        # Layer: keep it linear/ordinal for the MLP instead of exponential
        layer_n = layer / 4.0

        scale_A = float(np.abs(vals).max())
        if scale_A <= 0.0:
            scale_A = 1.0

        dg_path = frame_path / "diffusion_gradient.bin"
        if dg_path.exists():
            diffusion_grad = np.fromfile(dg_path, dtype=np.float32).reshape(num_nodes, 3)
        else:
            diffusion_grad = np.zeros((num_nodes, 3), dtype=np.float32)

        # --- PHYSICS-PRESERVING NORMALIZATION ---
        # 1. Diag Map: Must be scaled IDENTICALLY to the edge values to preserve the condition number
        diag_map_n = diag_map / scale_A

        # 2. SymLog Normalization for heavy-tailed features: sign(x) * log(1 + |x|)
        # This crushes outliers but leaves 0.0 as exactly 0.0.
        mass_n = np.sign(mass) * np.log1p(np.abs(mass))
        diffusion_grad_n = np.sign(diffusion_grad) * np.log1p(np.abs(diffusion_grad))

        # 3. Global Features for the Scale Predictor (keeping length 12 to match your MLP)
        diff_mag = np.linalg.norm(diffusion_grad_n, axis=1)
        global_features = np.array([
            math.log2(max(1, num_nodes)),
            math.log10(max(1e-6, scale_A)),
            float(np.mean(mass_n)), float(np.std(mass_n)),
            float(np.mean(diag_map_n)), float(np.std(diag_map_n)),
            float(np.mean(diff_mag)), float(np.std(diff_mag)),
            0.0, 0.0, 0.0, 0.0,  # Padding
        ], dtype=np.float32)

        n_float = 9
        x = np.zeros((num_nodes, n_float), dtype=np.float32)
        x[:, :3] = positions_n
        x[:, 3] = layer_n
        x[:, 4] = mass_n
        x[:, 5:8] = diffusion_grad_n
        x[:, 8] = diag_map_n

        return {
            'x': torch.from_numpy(x).float(),
            'edge_index': torch.stack([torch.from_numpy(rows.astype(np.int64)), torch.from_numpy(cols.astype(np.int64))]),
            'edge_values': torch.from_numpy(vals.copy()),
            'num_nodes': int(num_nodes),
            'scale_A': scale_A,
            'frame_path': str(frame_path),
            'global_features': torch.from_numpy(global_features).float(),
        }

# --- Embedding and graph layers ---

class SparsePhysicsGCN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear_self = nn.Linear(d_model, d_model)
        self.linear_neighbor = nn.Linear(d_model, d_model)
        self.update_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x, edge_index, edge_values, scale_A=None):
        B, N, C = x.shape
        x_flat = x.squeeze(0) if B == 1 else x.view(B * N, C)

        row, col = edge_index[0], edge_index[1]
        w = edge_values.clone().to(x.dtype)
        if scale_A is not None and scale_A != 1.0:
            s = scale_A.to(x.dtype).squeeze() if isinstance(scale_A, torch.Tensor) else torch.tensor(scale_A, device=w.device, dtype=x.dtype)
            w = w / s

        neighbor_features = self.linear_neighbor(x_flat)
        messages = neighbor_features[col] * w.unsqueeze(-1)
        aggr = torch.zeros_like(x_flat)
        aggr.index_add_(0, row, messages)

        self_features = self.linear_self(x_flat)
        out = self.update_gate(torch.cat([self_features, aggr], dim=-1))
        out = x_flat + out

        return out.unsqueeze(0) if B == 1 else out.view(B, N, C)

class PhysicsAwareEmbedding(nn.Module):
    """Lift uses only x[:, 3:] (no absolute coords) for translation/rotation invariance."""
    NUM_GCN_LAYERS = 2
    LIFT_FEATURES = 6  # layer, mass, diffusion_grad(3), diag — excludes x[:, :3]

    def __init__(self, input_dim, d_model, use_gcn=True):
        super().__init__()
        self.use_gcn = use_gcn
        lift_in = self.LIFT_FEATURES  # drop absolute coords from lift
        self.lift = nn.Sequential(
            nn.Linear(lift_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.gcn = nn.ModuleList([SparsePhysicsGCN(d_model) for _ in range(self.NUM_GCN_LAYERS)]) if use_gcn else None
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, edge_index=None, edge_values=None, scale_A=None):
        h = self.lift(x[..., 3:])  # no absolute coords: layer, mass, diffusion_grad, diag only
        if self.use_gcn and self.gcn is not None and edge_index is not None and edge_values is not None:
            for gcn_layer in self.gcn:
                h = gcn_layer(h, edge_index, edge_values, scale_A)
        return self.norm(h)

# --- Leaf block connectivity and attention ---
# Masked attention within each leaf block: 1 = self + 1-hop only; 2 = up to 2-hop; etc.
ATTENTION_HOPS = 1

def build_leaf_block_physics_bias(edge_index, edge_values, scale_A, leaf_size, device, dtype=torch.float32, num_nodes=None):
    rows, cols = edge_index[0], edge_index[1]
    if num_nodes is not None:
        N = num_nodes
    else:
        N = int(max(rows.max().item(), cols.max().item()) + 1) if rows.numel() else 0
    num_blocks = N // leaf_size
    if num_blocks == 0:
        return None
    block_r = rows // leaf_size
    block_c = cols // leaf_size
    in_block = (block_r == block_c) & (block_r < num_blocks)
    r_l = rows[in_block] % leaf_size
    c_l = cols[in_block] % leaf_size
    b_l = block_r[in_block]
    w = edge_values[in_block].to(device=device, dtype=dtype)
    if scale_A is not None and scale_A != 1.0:
        s = scale_A.to(device=device, dtype=dtype).squeeze() if isinstance(scale_A, torch.Tensor) else torch.tensor(scale_A, device=device, dtype=dtype)
        w = w / s
    bias = torch.zeros(num_blocks, leaf_size, leaf_size, device=device, dtype=dtype)
    bias[:, torch.arange(leaf_size, device=device), torch.arange(leaf_size, device=device)] = 1.0
    bias[b_l, r_l, c_l] = w
    return bias

def build_leaf_block_connectivity(edge_index, edge_values, positions, scale_A, leaf_size, device, dtype=torch.float32, num_hops=ATTENTION_HOPS):
    N = positions.shape[0]
    num_blocks = N // leaf_size
    if num_blocks == 0:
        return None, None
    rows, cols = edge_index[0], edge_index[1]
    block_r = rows // leaf_size
    block_c = cols // leaf_size
    in_block = (block_r == block_c) & (block_r < num_blocks)
    r_l, c_l = rows[in_block] % leaf_size, cols[in_block] % leaf_size
    b_l = block_r[in_block]
    pos_r = positions[rows[in_block]]
    pos_c = positions[cols[in_block]]
    dx = (pos_c - pos_r).to(device=device, dtype=dtype)
    w = edge_values[in_block].to(device=device, dtype=dtype)
    if scale_A is not None and scale_A != 1.0:
        s = scale_A.to(device=device, dtype=dtype).squeeze() if isinstance(scale_A, torch.Tensor) else torch.tensor(scale_A, device=device, dtype=dtype)
        w = w / s
    edge_feats_flat = torch.cat([dx, w.unsqueeze(1)], dim=1)

    # Batched adjacency (num_blocks, leaf_size, leaf_size) from in-block edges
    adj = torch.zeros(num_blocks, leaf_size, leaf_size, device=device, dtype=dtype)
    if b_l.numel() > 0:
        adj[b_l, r_l, c_l] = 1.0
    # n-hop mask = I + A + A^2 + ... + A^num_hops (powers of adjacency within leaf)
    reachable = torch.eye(leaf_size, device=device, dtype=dtype).unsqueeze(0).expand(num_blocks, -1, -1).clone()
    cur = adj.clone()
    for _ in range(num_hops):
        reachable = (reachable + cur).clamp(0.0, 1.0)
        cur = torch.bmm(cur, adj)
    attn_mask = torch.zeros(num_blocks, leaf_size, leaf_size + 1, device=device, dtype=dtype)
    attn_mask[:, :, :leaf_size] = reachable
    attn_mask[:, :, leaf_size] = 1.0
    edge_feats = torch.zeros(num_blocks, leaf_size, leaf_size + 1, 4, device=device, dtype=dtype)
    edge_feats[:, torch.arange(leaf_size, device=device), torch.arange(leaf_size, device=device), :] = 0.0
    if b_l.numel() > 0:
        edge_feats[b_l, r_l, c_l, :] = edge_feats_flat.to(device)
    return attn_mask, edge_feats

class LeafBlockAttention(nn.Module):
    def __init__(self, dim, block_size, num_heads=2, mask_attention=True, use_global_node=True):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.mask_attention = mask_attention
        self.use_global_node = use_global_node
        self.num_heads = num_heads if dim % num_heads == 0 else 1
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.edge_gate = nn.Linear(4, self.num_heads)  # dx, dy, dz, w
        nn.init.normal_(self.edge_gate.weight, std=0.01)
        nn.init.zeros_(self.edge_gate.bias)
        self.last_attn_self = 0.0
        self.last_attn_neighbor = 0.0
        self.last_attn_block = 0.0
        self.last_attn_matrix = None
        self.last_scores_matrix = None
        self.last_bias_physics_matrix = None

    def forward(self, x, edge_index=None, edge_values=None, positions=None, scale_A=None, save_attention=False, attn_mask=None, edge_feats=None, physics_bias=None):
        B, N, C = x.shape
        if edge_index is None or (self.mask_attention and positions is None):
            return self._forward_dense_fallback(x, B, N, C, save_attention)

        device = x.device
        dtype = x.dtype
        pad = 0
        if N % self.block_size != 0:
            pad = self.block_size - (N % self.block_size)
            x = F.pad(x, (0, 0, 0, pad))
        N_pad = x.shape[1]
        num_blocks = N_pad // self.block_size
        x_blk = x.view(B, num_blocks, self.block_size, C)

        if not self.mask_attention:
            return self._forward_unmasked_32(x_blk, B, N_pad, N, pad, num_blocks, device, dtype, edge_index, edge_values, scale_A, save_attention, physics_bias, N_pad)

        if attn_mask is None or edge_feats is None:
            attn_mask, edge_feats = build_leaf_block_connectivity(
                edge_index, edge_values, positions, scale_A, self.block_size, device, dtype
            )
        if attn_mask is None:
            return x[:, :N, :] if pad > 0 else x

        if not self.use_global_node:
            return self._forward_masked_32_only(x_blk, B, N_pad, N, pad, num_blocks, device, dtype, attn_mask, edge_feats, save_attention)

        block_node = x_blk.mean(dim=2, keepdim=True)
        kv = torch.cat([x_blk, block_node], dim=2)
        qkv_q = self.qkv(x_blk)
        qkv_kv = self.qkv(kv)
        q = qkv_q[..., :C]
        k = qkv_kv[..., C:2*C]
        v = qkv_kv[..., 2*C:3*C]
        q = q.view(B, num_blocks, self.block_size, self.num_heads, self.head_dim)
        k = k.view(B, num_blocks, self.block_size + 1, self.num_heads, self.head_dim)
        v = v.view(B, num_blocks, self.block_size + 1, self.num_heads, self.head_dim)

        scores = torch.einsum('bnqhd,bnkhd->bnqkh', q, k) * self.scale

        bias_physics = edge_feats[..., :4].clone()
        bias_physics[:, :, self.block_size, :] = 0.0
        bias_physics[:, :, self.block_size, 3] = 1.0
        arange = torch.arange(self.block_size, device=device)
        bias_physics[:, arange, arange, :] = 0.0
        bias_physics[:, arange, arange, 3] = 1.0
        scores = scores + bias_physics[..., 3:4].unsqueeze(0)
        mask_expanded = attn_mask.unsqueeze(0).unsqueeze(-1)
        scores = scores.masked_fill(mask_expanded == 0, float('-inf'))

        if save_attention:
            with torch.no_grad():
                self.last_scores_matrix = scores.mean(dim=-1)[:, :, :, :self.block_size].cpu().float()
                self.last_bias_physics_matrix = bias_physics[:, :, :self.block_size].cpu().float()

        attn_probs = F.softmax(scores, dim=3)

        linear_edge_weights = self.edge_gate(bias_physics)
        linear_edge_weights = linear_edge_weights.unsqueeze(0).masked_fill(mask_expanded == 0, 0.0)

        combined_weights = attn_probs + linear_edge_weights

        if not torch.compiler.is_compiling():
            with torch.no_grad():
                attn_viz = combined_weights.mean(dim=-1)
                arange = torch.arange(self.block_size, device=attn_viz.device)
                self.last_attn_self = attn_viz[:, :, arange, arange].mean().item()
                self.last_attn_block = attn_viz[:, :, :, self.block_size].mean().item()
                to_nodes = attn_viz[:, :, :, :self.block_size].sum(dim=3)
                self.last_attn_neighbor = (to_nodes - attn_viz[:, :, arange, arange]).mean().item()
                if save_attention:
                    self.last_attn_matrix = attn_viz[:, :, :, :self.block_size].cpu().float()

        x_out = torch.einsum('bnqkh,bnkhd->bnqhd', combined_weights, v)
        x_out = x_out.reshape(B, num_blocks, self.block_size, C)
        x_out = self.proj(x_out.view(B, N_pad, C))
        if pad > 0:
            x_out = x_out[:, :N, :]
        return x_out

    def _forward_masked_32_only(self, x_blk, B, N_pad, N_orig, pad, num_blocks, device, dtype, attn_mask, edge_feats, save_attention):
        C = x_blk.shape[-1]
        attn_mask = attn_mask[:, :, :self.block_size]
        edge_feats = edge_feats[:, :, :self.block_size, :]
        qkv = self.qkv(x_blk)
        q = qkv[..., :C].view(B, num_blocks, self.block_size, self.num_heads, self.head_dim)
        k = qkv[..., C:2*C].view(B, num_blocks, self.block_size, self.num_heads, self.head_dim)
        v = qkv[..., 2*C:3*C].view(B, num_blocks, self.block_size, self.num_heads, self.head_dim)
        scores = torch.einsum('bnqhd,bnkhd->bnqkh', q, k) * self.scale
        bias_physics = edge_feats[..., :4].clone()
        arange = torch.arange(self.block_size, device=device)
        bias_physics[:, arange, arange, :] = 0.0
        bias_physics[:, arange, arange, 3] = 1.0
        scores = scores + bias_physics[..., 3:4].unsqueeze(0)
        mask_expanded = attn_mask.unsqueeze(0).unsqueeze(-1)
        scores = scores.masked_fill(mask_expanded == 0, float('-inf'))
        if save_attention:
            with torch.no_grad():
                self.last_scores_matrix = scores.mean(dim=-1).cpu().float()
                self.last_bias_physics_matrix = bias_physics.cpu().float()
        attn_probs = F.softmax(scores, dim=3)
        linear_edge_weights = self.edge_gate(bias_physics).unsqueeze(0).masked_fill(mask_expanded == 0, 0.0)
        combined_weights = attn_probs + linear_edge_weights
        if not torch.compiler.is_compiling():
            with torch.no_grad():
                attn_viz = combined_weights.mean(dim=-1)
                arange = torch.arange(self.block_size, device=attn_viz.device)
                self.last_attn_self = attn_viz[:, :, arange, arange].mean().item()
                self.last_attn_block = 0.0
                to_nodes = attn_viz.sum(dim=3)
                self.last_attn_neighbor = (to_nodes - attn_viz[:, :, arange, arange]).mean().item()
                if save_attention:
                    self.last_attn_matrix = attn_viz.cpu().float()
        x_out = torch.einsum('bnqkh,bnkhd->bnqhd', combined_weights, v)
        x_out = x_out.reshape(B, num_blocks, self.block_size, C)
        x_out = self.proj(x_out.view(B, N_pad, C))
        return x_out[:, :N_orig, :] if pad > 0 else x_out

    def _forward_unmasked_32(self, x_blk, B, N_pad, N_orig, pad, num_blocks, device, dtype, edge_index, edge_values, scale_A, save_attention, physics_bias=None, num_nodes=None):
        C = x_blk.shape[-1]
        if physics_bias is None:
            physics_bias = build_leaf_block_physics_bias(edge_index, edge_values, scale_A, self.block_size, device, dtype, num_nodes=num_nodes or N_pad)
        if physics_bias is None:
            physics_bias = torch.eye(self.block_size, device=device, dtype=dtype).unsqueeze(0).expand(num_blocks, -1, -1)
        qkv = self.qkv(x_blk)
        q, k, v = qkv[..., :C], qkv[..., C:2*C], qkv[..., 2*C:3*C]
        q = q.view(B, num_blocks, self.block_size, self.num_heads, self.head_dim)
        k = k.view(B, num_blocks, self.block_size, self.num_heads, self.head_dim)
        v = v.view(B, num_blocks, self.block_size, self.num_heads, self.head_dim)
        scores = torch.einsum('bnqhd,bnkhd->bnqkh', q, k) * self.scale
        physics_bias_4d = torch.zeros(num_blocks, self.block_size, self.block_size, 4, device=device, dtype=dtype)
        physics_bias_4d[..., 3] = physics_bias
        arange = torch.arange(self.block_size, device=device)
        physics_bias_4d[:, arange, arange, :] = 0.0
        physics_bias_4d[:, arange, arange, 3] = 1.0
        scores = scores + physics_bias_4d[..., 3:4].unsqueeze(0)
        attn_probs = F.softmax(scores, dim=3)
        linear_edge_weights = self.edge_gate(physics_bias_4d).unsqueeze(0)
        combined_weights = attn_probs + linear_edge_weights
        if not torch.compiler.is_compiling():
            with torch.no_grad():
                attn_viz = combined_weights.mean(dim=-1)
                arange = torch.arange(self.block_size, device=attn_viz.device)
                self.last_attn_self = attn_viz[:, :, arange, arange].mean().item()
                self.last_attn_block = 0.0
                to_nodes = attn_viz.sum(dim=3)
                self.last_attn_neighbor = (to_nodes - attn_viz[:, :, arange, arange]).mean().item()
                self.last_scores_matrix = None
                self.last_bias_physics_matrix = physics_bias.cpu().float() if save_attention else None
                if save_attention:
                    self.last_attn_matrix = attn_viz[:, :, :, :].cpu().float()
        x_out = torch.einsum('bnqkh,bnkhd->bnqhd', combined_weights, v)
        x_out = x_out.reshape(B, num_blocks, self.block_size, C)
        x_out = self.proj(x_out.view(B, N_pad, C))
        if pad > 0:
            x_out = x_out[:, :N_orig, :]
        return x_out

    def _forward_dense_fallback(self, x, B, N, C, save_attention):
        pad = 0
        if N % self.block_size != 0:
            pad = self.block_size - (N % self.block_size)
            x = F.pad(x, (0, 0, 0, pad))
        N_pad = x.shape[1]
        num_blocks = N_pad // self.block_size
        x_blk = x.view(B * num_blocks, self.block_size, C)
        qkv = self.qkv(x_blk).reshape(B * num_blocks, self.block_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if not torch.compiler.is_compiling():
            with torch.no_grad():
                arange = torch.arange(self.block_size, device=attn.device)
                self.last_attn_self = attn[:, arange, arange].mean().item()
                self.last_attn_block = 0.0
                self.last_attn_neighbor = (attn.sum(dim=-1) - attn[:, arange, arange]).mean().item()
                self.last_scores_matrix = None
                self.last_bias_physics_matrix = None
                if save_attention:
                    self.last_attn_matrix = attn.cpu().float()
        x_out = (attn @ v).transpose(1, 2).reshape(B * num_blocks, self.block_size, C)
        x_out = self.proj(x_out)
        x_out = x_out.view(B, N_pad, C)
        if pad > 0:
            x_out = x_out[:, :N, :]
        return x_out

class TransformerBlock(nn.Module):
    def __init__(self, dim, block_size, attn_module, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attn_module

    def forward(self, x, edge_index=None, edge_values=None, positions=None, scale_A=None, save_attention=False, attn_mask=None, edge_feats=None, physics_bias=None):
        x = x + self.attn(
            self.norm1(x), edge_index=edge_index, edge_values=edge_values, positions=positions,
            scale_A=scale_A, save_attention=save_attention, attn_mask=attn_mask, edge_feats=edge_feats, physics_bias=physics_bias
        )
        return x

def _most_recent_run_folder(base_path):
    base = Path(base_path)
    if not base.exists(): return base
    runs = sorted([p for p in base.glob("Run_*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs: return base
    return runs[0]

LEAF_SIZE = 32
VIEW_SIZE = 0

def next_valid_size(n, leaf_size=LEAF_SIZE):
    if n <= 0:
        return leaf_size * 2
    num_blocks = (n + leaf_size - 1) // leaf_size
    if num_blocks <= 1:
        return leaf_size * 2
    p = 1
    while p < num_blocks:
        p *= 2
    return p * leaf_size

RANK_BASE_LEVEL1 = 16
OFF_DIAG_SUPER = 32
GLOBAL_FEATURES_DIM = 12  # dataset global_features vector for scale predictor  

def _rank_for_side(side, rank_base=RANK_BASE_LEVEL1, min_rank=4, max_rank=512):
    r = rank_base * ((side / 32.0) ** (2.0 / 3.0))
    r = max(min_rank, min(max_rank, int(round(r))))
    return r if r % 2 == 0 else r + 1  

@torch._dynamo.disable
def build_hodlr_off_diag_structure(n_nodes, leaf_size=LEAF_SIZE, rank_base=RANK_BASE_LEVEL1):
    n_nodes = int(n_nodes)
    leaf_size = int(leaf_size)
    num_blocks = n_nodes // leaf_size
    if num_blocks & (num_blocks - 1) != 0:
        raise ValueError(f"n_nodes must be power-of-2 * leaf_size; got n_nodes={n_nodes}, leaf_size={leaf_size}")
    depth = int(round(math.log2(num_blocks)))
    specs = []
    for level_idx, tree_level in enumerate(range(depth, 0, -1), start=1):
        segment_size = n_nodes // (2 ** tree_level)
        n_blocks_at_level = 2 ** (tree_level - 1)
        rank = _rank_for_side(segment_size, rank_base=rank_base)
        for k in range(n_blocks_at_level):
            row_start = 2 * k * segment_size
            row_end = (2 * k + 1) * segment_size
            col_start = (2 * k + 1) * segment_size
            col_end = (2 * k + 2) * segment_size
            specs.append({
                "row_start": row_start, "row_end": row_end,
                "col_start": col_start, "col_end": col_end,
                "side": segment_size, "rank": rank, "level": level_idx,
            })
    return specs


def build_off_diag_super_connectivity(edge_index, rs, re, cs, ce, device, leaf_size=LEAF_SIZE):
    side = re - rs
    assert side == (ce - cs) and side % leaf_size == 0, f"Block must be square and multiple of {leaf_size}"
    n_super = OFF_DIAG_SUPER
    group_size = side // n_super
    row, col = edge_index[0], edge_index[1]
    in_block = (row >= rs) & (row < re) & (col >= cs) & (col < ce)
    r_super = ((row[in_block] - rs) // group_size).clamp(0, n_super - 1)
    c_super = ((col[in_block] - cs) // group_size).clamp(0, n_super - 1)
    mask = torch.zeros(n_super, n_super, device=device, dtype=torch.bool)
    mask[r_super, c_super] = True
    for i in range(n_super):
        if not mask[i].any():
            mask[i, 0] = True
    return mask

def print_hodlr_structure(n_nodes, leaf_size=LEAF_SIZE, rank_base=RANK_BASE_LEVEL1):
    num_blocks = n_nodes // leaf_size
    depth = int(round(math.log2(num_blocks)))
    specs = build_hodlr_off_diag_structure(n_nodes, leaf_size, rank_base)
    lines = [
        "HODLR off-diagonal structure:",
        f"  N = {n_nodes}, leaf_size = {leaf_size}, num_diagonal_blocks = {num_blocks}",
        f"  Levels: 0 (diagonal {leaf_size}x{leaf_size}) + off-diag levels 1..{depth}",
        f"  Downsampling: on (32x32 attn per block)",
    ]
    by_level = {}
    for s in specs:
        L = s["level"]
        by_level.setdefault(L, []).append(s)
    for L in sorted(by_level.keys()):
        bl = by_level[L]
        side = bl[0]["side"]
        rank = bl[0]["rank"]
        lines.append(f"  Level {L}: {len(bl)} block(s), side = {side}x{side}, rank = {rank}, attn {OFF_DIAG_SUPER}x{OFF_DIAG_SUPER}")
    lines.append(f"  Total off-diag blocks: {len(specs)}")
    print("\n".join(lines) + "\n")


class LeafCore(nn.Module):
    def __init__(self, input_dim=9, d_model=128, leaf_size=32, num_layers=3, num_heads=4, mask_attention=True, use_global_node=True, use_gcn=True, use_jacobi=True):
        super().__init__()
        self.leaf_size = leaf_size
        self.mask_attention = mask_attention
        self.use_global_node = use_global_node
        self.use_jacobi = use_jacobi
        self.embed = PhysicsAwareEmbedding(input_dim, d_model, use_gcn=use_gcn)
        self.enc_input_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, block_size=leaf_size, attn_module=LeafBlockAttention(d_model, leaf_size, num_heads=num_heads, mask_attention=mask_attention, use_global_node=use_global_node))
            for _ in range(num_layers)
        ])
        self.leaf_head = nn.Linear(d_model, leaf_size)
        nn.init.normal_(self.leaf_head.weight, std=0.01)
        nn.init.constant_(self.leaf_head.bias, 0.0)
        # Local gate for the Jacobi diagonal (Sigmoid(0) = 0.5 → neutral multiplier 1.0 with * 2.0)
        self.jacobi_gate = nn.Linear(d_model, 1)
        nn.init.normal_(self.jacobi_gate.weight, std=0.01)
        nn.init.constant_(self.jacobi_gate.bias, 0.0)
        # Inits from observed settling (e.g. sLeaf~7e-3, sJ~1.5)
        self.log_hodlr_scale_leaf = nn.Parameter(torch.ones(1) * math.log(7e-3))
        self.log_jacobi_scale_leaf = nn.Parameter(torch.ones(1) * math.log(1.5))  # only used when use_jacobi=True

    def get_leaf_blocks(self, h, x, scale_A=None, log_hodlr_scale=None, log_jacobi_scale=None):
        B, N, _ = h.shape
        num_leaves = N // self.leaf_size
        h_leaves = h.view(B, num_leaves, self.leaf_size, -1)
        # 1. Base SPSD matrix
        u_leaf = self.leaf_head(h_leaves)
        if log_hodlr_scale is not None:
            leaf_scale = torch.exp(log_hodlr_scale).view(B, 1, 1, 1)
        else:
            leaf_scale = torch.exp(self.log_hodlr_scale_leaf)
        leaf_base = torch.matmul(u_leaf, u_leaf.transpose(-1, -2)) * leaf_scale
        if not self.use_jacobi:
            return leaf_base
        # 2. Extract and format the true Jacobi diagonal
        diag_A_n = x[..., 8]
        s = scale_A if scale_A is not None else 1.0
        if isinstance(s, torch.Tensor):
            diag_A_real = diag_A_n * s
        else:
            diag_A_real = diag_A_n * s
        jacobi_diag = torch.zeros_like(diag_A_real)
        mask = diag_A_real.abs() > 1e-6
        jacobi_diag[mask] = 1.0 / diag_A_real[mask]
        jacobi_diag_blocks = jacobi_diag.view(B, num_leaves, self.leaf_size)
        mask_blocks = mask.view(B, num_leaves, self.leaf_size)
        new_diag = torch.where(mask_blocks, jacobi_diag_blocks, torch.ones_like(jacobi_diag_blocks))
        # 3. Apply the local gating (Sigmoid(0)=0.5 * 2.0 = 1.0 neutral)
        if log_jacobi_scale is not None:
            jacobi_scale = torch.exp(log_jacobi_scale).view(B, 1, 1)
        else:
            jacobi_scale = torch.exp(self.log_jacobi_scale_leaf)
        j_gate = torch.sigmoid(self.jacobi_gate(h_leaves)).squeeze(-1)
        self._last_j_gate = j_gate.detach()  # for logging open/closed %
        local_jacobi_diag = new_diag * j_gate * jacobi_scale * 2.0
        return leaf_base + torch.diag_embed(local_jacobi_diag)

    def forward_features(self, x, edge_index=None, edge_values=None, scale_A=None, save_attention=False, precomputed_leaf_connectivity=None):
        B, N, _ = x.shape
        positions = x[0, :, :3] if x.dim() == 3 else x[:, :3]
        h = self.embed(x, edge_index, edge_values, scale_A)
        h = self.enc_input_proj(h)
        attn_mask, edge_feats, physics_bias = None, None, None
        if edge_index is not None and edge_values is not None:
            device, dtype = x.device, x.dtype
            if self.mask_attention and positions is not None:
                if precomputed_leaf_connectivity is not None:
                    attn_mask, edge_feats = precomputed_leaf_connectivity
                else:
                    attn_mask, edge_feats = build_leaf_block_connectivity(
                        edge_index, edge_values, positions, scale_A, self.leaf_size, device, dtype, num_hops=ATTENTION_HOPS
                    )
            else:
                physics_bias = build_leaf_block_physics_bias(edge_index, edge_values, scale_A, self.leaf_size, device, dtype, num_nodes=x.shape[1])
        for i, block in enumerate(self.blocks):
            h = block(
                h, edge_index=edge_index, edge_values=edge_values, positions=positions,
                scale_A=scale_A, save_attention=save_attention, attn_mask=attn_mask, edge_feats=edge_feats, physics_bias=physics_bias
            )
        return h

    def forward(self, x, edge_index=None, edge_values=None, scale_A=None, save_attention=False):
        h = self.forward_features(x, edge_index, edge_values, scale_A, save_attention)
        return self.get_leaf_blocks(h, x, scale_A)


class UniversalOffDiagBlock(nn.Module):
    def __init__(self, d_model, max_rank=512):
        super().__init__()
        self.n_super = OFF_DIAG_SUPER
        self.max_rank = max_rank

        # d_model + 1 (scale) + 3 (dx) + 1 (dist) = d_model + 5
        self.attn_q = nn.Linear(d_model + 5, d_model)
        self.attn_k = nn.Linear(d_model + 5, d_model)
        self.attn_v = nn.Linear(d_model + 5, d_model)
        
        self.W_U = nn.Parameter(torch.randn(max_rank, 2 * d_model))
        self.b_U = nn.Parameter(torch.zeros(max_rank))
        self.W_V = nn.Parameter(torch.randn(max_rank, 2 * d_model))
        self.b_V = nn.Parameter(torch.zeros(max_rank))
        
        for m in (self.attn_q, self.attn_k, self.attn_v):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
            
        nn.init.normal_(self.W_U, std=0.01)
        nn.init.normal_(self.W_V, std=0.01)
        self._scale_attn = d_model ** -0.5

    def forward(self, h_row, h_col, pos_row, pos_col, mask, scale, rank):
        B = h_row.shape[0]
        side = h_row.shape[1]
        g = side // self.n_super

        down_row = h_row.view(B, self.n_super, g, -1).mean(dim=2)
        down_col = h_col.view(B, self.n_super, g, -1).mean(dim=2)

        # Downsample positions (average physical location of the block)
        down_pos_row = pos_row.view(B, self.n_super, g, -1).mean(dim=2)
        down_pos_col = pos_col.view(B, self.n_super, g, -1).mean(dim=2)
        delta_x = down_pos_col - down_pos_row
        dist = torch.norm(delta_x, dim=-1, keepdim=True).clamp(min=1e-8)

        scale_val = math.log2(max(1.0, side / float(self.n_super)))
        scale_tensor = torch.full((B, self.n_super, 1), scale_val, device=h_row.device, dtype=h_row.dtype)

        down_row_cond = torch.cat([down_row, scale_tensor, delta_x, dist], dim=-1)
        down_col_cond = torch.cat([down_col, scale_tensor, -delta_x, dist], dim=-1)

        Q = self.attn_q(down_row_cond)
        K = self.attn_k(down_col_cond)
        V = self.attn_v(down_col_cond)
        
        scores = (Q @ K.transpose(-2, -1)) * self._scale_attn
        # Handle both 2D (single block) and 3D (batched blocks) masks
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn_probs = F.softmax(scores, dim=-1)
        row_out = (attn_probs @ V).reshape(B, self.n_super, -1)
        
        row_out_expanded = row_out.repeat_interleave(g, dim=1)
        down_col_expanded = down_col.repeat_interleave(g, dim=1)
        
        h_row_3 = h_row.reshape(B, -1, h_row.shape[-1])
        row_exp_3 = row_out_expanded.reshape(B, -1, row_out_expanded.shape[-1])
        h_col_3 = h_col.reshape(B, -1, h_col.shape[-1])
        col_exp_3 = down_col_expanded.reshape(B, -1, down_col_expanded.shape[-1])
        
        features_U = torch.cat([h_row_3, row_exp_3], dim=-1)
        features_V = torch.cat([h_col_3, col_exp_3], dim=-1)
        
        safe_rank = min(rank, self.max_rank)
        W_U_sliced = self.W_U[:safe_rank, :]
        b_U_sliced = self.b_U[:safe_rank]
        W_V_sliced = self.W_V[:safe_rank, :]
        b_V_sliced = self.b_V[:safe_rank]
        
        U_raw = F.linear(features_U, W_U_sliced, b_U_sliced)
        V_raw = F.linear(features_V, W_V_sliced, b_V_sliced)
        
        # Counteract rank inflation (Scaled Dot-Product style)
        rank_scale = math.sqrt(safe_rank)
        # Combine with the learned physical scale; [B,1] or [B,1,1] -> broadcast to (B, side, rank)
        unified_scalar = torch.sqrt(scale / rank_scale)
        if unified_scalar.dim() == 2:
            unified_scalar = unified_scalar.unsqueeze(-1)
        return U_raw * unified_scalar, V_raw * unified_scalar


class LeafOnlyNet(nn.Module):
    def __init__(self, input_dim=9, d_model=128, leaf_size=32, num_layers=3, num_heads=4, rank_base=RANK_BASE_LEVEL1, mask_attention=True, use_global_node=True, use_gcn=True, use_jacobi=True):
        super().__init__()
        self.leaf_size = leaf_size
        self.rank_base = rank_base
        self.core = LeafCore(input_dim=input_dim, d_model=d_model, leaf_size=leaf_size, num_layers=num_layers, num_heads=num_heads, mask_attention=mask_attention, use_global_node=use_global_node, use_gcn=use_gcn, use_jacobi=use_jacobi)
        # level_scale_params = [log(base_scale), decay_per_level]; init from observed settling (lvl ~ (-4.88, -0.77))
        self.level_scale_params = nn.Parameter(
            torch.tensor([-4.88, -0.77])
        )
        # Scale predictor: input = [global_features (12), h.mean(1) (d_model)] -> [log_hodlr, log_jacobi, level_p0, level_p1]
        scale_input_dim = GLOBAL_FEATURES_DIM + d_model
        self.scale_head = nn.Linear(scale_input_dim, 4)
        with torch.no_grad():
            self.scale_head.bias.copy_(torch.tensor([
                math.log(7e-3), math.log(1.5), -4.88, -0.77
            ], dtype=self.scale_head.bias.dtype))
            self.scale_head.weight.zero_()

        self.universal_off_diag = UniversalOffDiagBlock(d_model, max_rank=512)
        self.off_diag_struct = []

    @property
    def embed(self):
        return self.core.embed

    @property
    def enc_input_proj(self):
        return self.core.enc_input_proj

    @property
    def blocks(self):
        return self.core.blocks

    @property
    def leaf_head(self):
        return self.core.leaf_head

    @property
    def log_hodlr_scale_leaf(self):
        return self.core.log_hodlr_scale_leaf

    def _off_diag(self, h, x, edge_index, current_struct, precomputed_masks=None, level_p0=None, level_p1=None):
        B, N, C = h.shape
        if level_p0 is not None and level_p1 is not None:
            level_p0 = level_p0.view(B, 1)
            level_p1 = level_p1.view(B, 1)
        else:
            level_p0 = self.level_scale_params[0].view(1, 1).expand(B, 1)
            level_p1 = self.level_scale_params[1].view(1, 1).expand(B, 1)

        # 1. Fetch or compute masks
        if precomputed_masks is not None:
            mask_list = precomputed_masks
        else:
            cache_id = id(edge_index)
            if getattr(self, "_off_diag_mask_cache_id", None) != cache_id:
                self._off_diag_mask_cache = [
                    build_off_diag_super_connectivity(
                        edge_index, spec["row_start"], spec["row_end"], spec["col_start"], spec["col_end"],
                        h.device, self.leaf_size
                    )
                    for spec in current_struct
                ]
                self._off_diag_mask_cache_id = cache_id
            mask_list = self._off_diag_mask_cache

        result = [None] * len(current_struct)

        # 2. Group blocks by their HODLR level
        level_to_blocks = {}
        for idx, spec in enumerate(current_struct):
            lvl = spec["level"]
            if lvl not in level_to_blocks:
                level_to_blocks[lvl] = []
            level_to_blocks[lvl].append((idx, spec))

        # 3. Execute batched blocks per level
        for lvl, blocks in level_to_blocks.items():
            K = len(blocks)
            rank = blocks[0][1]["rank"]

            # Compute dynamic scale for this level
            L_tensor = torch.tensor(lvl, dtype=h.dtype, device=h.device)
            log_scale = level_p0.view(B, 1) + (L_tensor - 1.0) * level_p1.view(B, 1)
            scale = torch.exp(log_scale)  # [B, 1]
            scale_batched = scale.repeat(K, 1)  # [K*B, 1]

            h_rows, h_cols = [], []
            pos_rows, pos_cols = [], []
            masks = []

            for idx, spec in blocks:
                rs, re = spec["row_start"], spec["row_end"]
                cs, ce = spec["col_start"], spec["col_end"]
                h_rows.append(h[:, rs:re])
                h_cols.append(h[:, cs:ce])
                pos_rows.append(x[:, rs:re, :3])
                pos_cols.append(x[:, cs:ce, :3])
                masks.append(mask_list[idx])

            h_row_batched = torch.cat(h_rows, dim=0)
            h_col_batched = torch.cat(h_cols, dim=0)
            pos_row_batched = torch.cat(pos_rows, dim=0)
            pos_col_batched = torch.cat(pos_cols, dim=0)

            stacked_masks = torch.stack(masks, dim=0)
            batched_masks = stacked_masks.repeat_interleave(B, dim=0)

            U_batched, V_batched = self.universal_off_diag(
                h_row_batched, h_col_batched, pos_row_batched, pos_col_batched,
                batched_masks, scale_batched, rank=rank
            )

            U_splits = torch.split(U_batched, B, dim=0)
            V_splits = torch.split(V_batched, B, dim=0)

            for i, (idx, _) in enumerate(blocks):
                result[idx] = (U_splits[i], V_splits[i])

        return result

    def forward(self, x, edge_index=None, edge_values=None, scale_A=None, save_attention=False, precomputed_masks=None, precomputed_leaf_connectivity=None, global_features=None):
        B, N, _ = x.shape
        assert N % self.leaf_size == 0, f"LeafOnly expects N divisible by leaf_size {self.leaf_size}, got {N}"

        current_struct = build_hodlr_off_diag_structure(N, self.leaf_size, self.rank_base)
        self.off_diag_struct = current_struct

        h = self.core.forward_features(x, edge_index=edge_index, edge_values=edge_values, scale_A=scale_A, save_attention=save_attention, precomputed_leaf_connectivity=precomputed_leaf_connectivity)
        if global_features is not None:
            gf = global_features if global_features.dim() == 2 else global_features.unsqueeze(0)
        else:
            gf = torch.zeros(B, GLOBAL_FEATURES_DIM, device=h.device, dtype=h.dtype)
        scale_input = torch.cat([gf, h.mean(dim=1)], dim=-1)
        scale_params = self.scale_head(scale_input)
        self._last_scale_params = scale_params.detach()
        log_hodlr = scale_params[:, 0:1]
        log_jacobi = scale_params[:, 1:2]
        level_p0 = scale_params[:, 2:3]
        level_p1 = scale_params[:, 3:4]

        diag_blocks = self.core.get_leaf_blocks(h, x, scale_A, log_hodlr_scale=log_hodlr, log_jacobi_scale=log_jacobi)

        if current_struct:
            off_diag_list = self._off_diag(h, x, edge_index, current_struct, precomputed_masks=precomputed_masks, level_p0=level_p0, level_p1=level_p1)
            return (diag_blocks, off_diag_list)
        return (diag_blocks, [])


def apply_block_structured_M(diag_blocks, off_diag_list, x, off_diag_struct, leaf_size=LEAF_SIZE):
    B, N, K = x.shape
    num_leaves = N // leaf_size
    x_blk = x.view(B, num_leaves, leaf_size, K)
    out = torch.einsum('blij,bljk->blik', diag_blocks, x_blk).reshape(B, N, K)
    for idx, spec in enumerate(off_diag_struct):
        U, V = off_diag_list[idx]   
        rs, re = spec["row_start"], spec["row_end"]
        cs, ce = spec["col_start"], spec["col_end"]
        x_col = x[:, cs:ce]   
        x_row = x[:, rs:re]   
        out[:, rs:re] = out[:, rs:re] + torch.matmul(U, torch.matmul(V.transpose(1, 2), x_col))
        out[:, cs:ce] = out[:, cs:ce] + torch.matmul(V, torch.matmul(U.transpose(1, 2), x_row))
    return out


def apply_leaf_only(leaf_blocks, x, off_diag_list=None, off_diag_struct=None):
    B, N, K = x.shape
    num_leaves = leaf_blocks.shape[1]
    leaf_size = leaf_blocks.shape[2]
    if not off_diag_list or not off_diag_struct:
        x_leaves = x.view(B, num_leaves, leaf_size, K)
        y_leaves = torch.matmul(leaf_blocks, x_leaves)
        return y_leaves.view(B, N, K)
    return apply_block_structured_M(leaf_blocks, off_diag_list, x, off_diag_struct, leaf_size=leaf_size)


# --- Save / Load ---

LEAF_ONLY_HEADER_BYTES_OLD = 20
LEAF_ONLY_HEADER_BYTES_V2 = 24
LEAF_ONLY_HEADER_BYTES = 28
LEAF_ONLY_HEADER_BYTES_V3 = 32   # + format_version (1=no jacobi in scale section, 2=jacobi present)

def read_leaf_only_header(path):
    path = Path(path)
    with open(path, 'rb') as f:
        header = f.read(LEAF_ONLY_HEADER_BYTES_V3)
    if len(header) < LEAF_ONLY_HEADER_BYTES_OLD:
        raise ValueError("LeafOnly weights file too short")
    if len(header) >= LEAF_ONLY_HEADER_BYTES:
        d_model, leaf_size, input_dim, num_layers, num_heads, use_gcn, num_gcn_layers = struct.unpack('<iiiiiii', header[:LEAF_ONLY_HEADER_BYTES])
        format_version = struct.unpack('<i', header[LEAF_ONLY_HEADER_BYTES:LEAF_ONLY_HEADER_BYTES_V3])[0] if len(header) >= LEAF_ONLY_HEADER_BYTES_V3 else 1
        if format_version not in (1, 2):
            format_version = 1
            header_bytes = LEAF_ONLY_HEADER_BYTES
        else:
            header_bytes = LEAF_ONLY_HEADER_BYTES_V3
        return d_model, leaf_size, input_dim, num_layers, num_heads, use_gcn, num_gcn_layers, format_version, header_bytes
    if len(header) >= LEAF_ONLY_HEADER_BYTES_V2:
        d_model, leaf_size, input_dim, num_layers, num_heads, use_gcn = struct.unpack('<iiiiii', header[:LEAF_ONLY_HEADER_BYTES_V2])
        return d_model, leaf_size, input_dim, num_layers, num_heads, use_gcn, 1, 1, LEAF_ONLY_HEADER_BYTES_V2
    d_model, leaf_size, input_dim, num_layers, num_heads = struct.unpack('<iiiii', header[:LEAF_ONLY_HEADER_BYTES_OLD])
    return d_model, leaf_size, input_dim, num_layers, num_heads, 1, 1, 1, LEAF_ONLY_HEADER_BYTES_OLD  


def _write_gcn_layer(f, gcn_layer):
    _write_packed_tensor(f, gcn_layer.linear_self.weight.detach().cpu().float(), transpose=True)
    _write_packed_tensor(f, gcn_layer.linear_self.bias.detach().cpu().float(), transpose=False)
    _write_packed_tensor(f, gcn_layer.linear_neighbor.weight.detach().cpu().float(), transpose=True)
    _write_packed_tensor(f, gcn_layer.linear_neighbor.bias.detach().cpu().float(), transpose=False)
    _write_packed_tensor(f, gcn_layer.update_gate[0].weight.detach().cpu().float(), transpose=True)
    _write_packed_tensor(f, gcn_layer.update_gate[0].bias.detach().cpu().float(), transpose=False)
    _write_packed_tensor(f, gcn_layer.update_gate[2].weight.detach().cpu().float(), transpose=True)
    _write_packed_tensor(f, gcn_layer.update_gate[2].bias.detach().cpu().float(), transpose=False)

def save_leaf_only_weights(model, path, input_dim=9):
    path = Path(path)
    d_model = model.embed.lift[0].weight.shape[0]
    num_heads = model.blocks[0].attn.num_heads if model.blocks else 4
    use_gcn = model.embed.gcn is not None
    num_gcn_layers = len(model.embed.gcn) if use_gcn else 0
    use_jacobi = getattr(model.core, 'use_jacobi', True)
    format_version = 2 if use_jacobi else 1
    with open(path, 'wb') as f:
        f.write(struct.pack('<iiiiiiii', d_model, model.leaf_size, input_dim, len(model.blocks), num_heads, int(use_gcn), num_gcn_layers, format_version))
        _write_packed_tensor(f, model.embed.lift[0].weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.lift[0].bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.lift[2].weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.lift[2].bias.detach().cpu().float(), transpose=False)
        if use_gcn:
            for gcn_layer in model.embed.gcn:
                _write_gcn_layer(f, gcn_layer)
        _write_packed_tensor(f, model.embed.norm.weight.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.norm.bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.enc_input_proj.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.enc_input_proj.bias.detach().cpu().float(), transpose=False)
        for block in model.blocks:
            _write_packed_tensor(f, block.norm1.weight.detach().cpu().float(), False)
            _write_packed_tensor(f, block.norm1.bias.detach().cpu().float(), False)
            _write_packed_tensor(f, block.attn.qkv.weight.detach().cpu().float(), True)
            _write_packed_tensor(f, block.attn.qkv.bias.detach().cpu().float(), False)
            _write_packed_tensor(f, block.attn.proj.weight.detach().cpu().float(), True)
            _write_packed_tensor(f, block.attn.proj.bias.detach().cpu().float(), False)
            _write_packed_tensor(f, block.attn.edge_gate.weight.detach().cpu().float(), True)
            _write_packed_tensor(f, block.attn.edge_gate.bias.detach().cpu().float(), False)
        _write_packed_tensor(f, model.leaf_head.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.leaf_head.bias.detach().cpu().float(), transpose=False)
        if hasattr(model, 'scale_head'):
            _write_packed_tensor(f, torch.exp(model.scale_head.bias[0:1]).detach().cpu().float(), transpose=False)
        else:
            _write_packed_tensor(f, torch.exp(model.log_hodlr_scale_leaf).detach().cpu().float(), transpose=False)
        if use_jacobi:
            _write_packed_tensor(f, model.core.jacobi_gate.weight.detach().cpu().float(), transpose=True)
            _write_packed_tensor(f, model.core.jacobi_gate.bias.detach().cpu().float(), transpose=False)
            if hasattr(model, 'scale_head'):
                _write_packed_tensor(f, torch.exp(model.scale_head.bias[1:2]).detach().cpu().float(), transpose=False)
            else:
                _write_packed_tensor(f, torch.exp(model.core.log_jacobi_scale_leaf).detach().cpu().float(), transpose=False)
        
        if getattr(model, 'universal_off_diag', None) is not None:
            f.write(struct.pack('<II', 0xFFFFFFFC, 0)) 
            blk = model.universal_off_diag
            _write_packed_tensor(f, blk.attn_q.weight.detach().cpu().float(), transpose=True)
            _write_packed_tensor(f, blk.attn_q.bias.detach().cpu().float(), transpose=False)
            _write_packed_tensor(f, blk.attn_k.weight.detach().cpu().float(), transpose=True)
            _write_packed_tensor(f, blk.attn_k.bias.detach().cpu().float(), transpose=False)
            _write_packed_tensor(f, blk.attn_v.weight.detach().cpu().float(), transpose=True)
            _write_packed_tensor(f, blk.attn_v.bias.detach().cpu().float(), transpose=False)
            
            _write_packed_tensor(f, blk.W_U.detach().cpu().float(), transpose=True)
            _write_packed_tensor(f, blk.b_U.detach().cpu().float(), transpose=False)
            _write_packed_tensor(f, blk.W_V.detach().cpu().float(), transpose=True)
            _write_packed_tensor(f, blk.b_V.detach().cpu().float(), transpose=False)
            if hasattr(model, 'scale_head'):
                _write_packed_tensor(f, model.scale_head.bias[2:4].detach().cpu().float(), transpose=False)
            else:
                _write_packed_tensor(f, model.level_scale_params.detach().cpu().float(), transpose=False)
        if hasattr(model, 'scale_head'):
            f.write(struct.pack('<II', 0xFFFFFFFB, 1))  # version 1 = scale_head input 12+d_model
            _write_packed_tensor(f, model.scale_head.weight.detach().cpu().float(), transpose=True)
            _write_packed_tensor(f, model.scale_head.bias.detach().cpu().float(), transpose=False)

def _write_packed_tensor(f, param, transpose=False):
    import numpy as np
    t = param.float() if isinstance(param, torch.Tensor) else torch.tensor(param)
    if transpose and t.dim() == 2:
        t = t.t()
    arr = t.numpy().astype(np.float16)
    n = arr.size
    pad = (1 if n % 2 else 0)
    f.write(arr.tobytes())
    if pad:
        f.write(np.zeros(1, dtype=np.float16).tobytes())

def _read_gcn_layer_into(f, gcn_layer, read_tensor):
    _read_into(f, gcn_layer.linear_self.weight, read_tensor, transpose=True)
    _read_into(f, gcn_layer.linear_self.bias, read_tensor, transpose=False)
    _read_into(f, gcn_layer.linear_neighbor.weight, read_tensor, transpose=True)
    _read_into(f, gcn_layer.linear_neighbor.bias, read_tensor, transpose=False)
    _read_into(f, gcn_layer.update_gate[0].weight, read_tensor, transpose=True)
    _read_into(f, gcn_layer.update_gate[0].bias, read_tensor, transpose=False)
    _read_into(f, gcn_layer.update_gate[2].weight, read_tensor, transpose=True)
    _read_into(f, gcn_layer.update_gate[2].bias, read_tensor, transpose=False)

def load_leaf_only_weights(model, path):
    import numpy as np
    path = Path(path)
    result = read_leaf_only_header(path)
    d_model_lo, leaf_size_lo, input_dim_lo, num_layers_lo, _num_heads_lo, use_gcn_file, num_gcn_layers_file, format_version, header_bytes = result
    if model.leaf_size != leaf_size_lo or len(model.blocks) != num_layers_lo:
        raise ValueError(f"Checkpoint leaf_size={leaf_size_lo} num_layers={num_layers_lo} != model {model.leaf_size} {len(model.blocks)}")
    has_gcn = model.embed.gcn is not None
    if has_gcn and not use_gcn_file:
        raise ValueError("Checkpoint was saved without GCN (use_gcn=False); cannot load into model with use_gcn=True.")

    def read_tensor(f, shape, transpose=False):
        num_elements = int(torch.Size(shape).numel())
        read_len = num_elements + (1 if num_elements % 2 else 0)
        buf = f.read(read_len * 2)
        packed = np.frombuffer(buf, dtype=np.uint32)
        data_fp16 = packed.view(np.float16)
        if num_elements % 2 != 0:
            data_fp16 = data_fp16[:-1]
        data_fp32 = torch.from_numpy(data_fp16.astype(np.float32))
        if transpose and len(shape) == 2:
            data_fp32 = data_fp32.view(shape[1], shape[0]).t()
        else:
            data_fp32 = data_fp32.view(shape)
        return data_fp32

    def skip_tensor(f, shape, transpose=False):
        read_tensor(f, shape, transpose=transpose)

    def skip_gcn_layer(f, d_model):
        skip_tensor(f, (d_model, d_model), transpose=True)
        skip_tensor(f, (d_model,), transpose=False)
        skip_tensor(f, (d_model, d_model), transpose=True)
        skip_tensor(f, (d_model,), transpose=False)
        skip_tensor(f, (d_model, d_model * 2), transpose=True)
        skip_tensor(f, (d_model,), transpose=False)
        skip_tensor(f, (d_model, d_model), transpose=True)
        skip_tensor(f, (d_model,), transpose=False)

    with open(path, 'rb') as f:
        f.seek(header_bytes)
        _read_into(f, model.embed.lift[0].weight, read_tensor, transpose=True)
        _read_into(f, model.embed.lift[0].bias, read_tensor, transpose=False)
        _read_into(f, model.embed.lift[2].weight, read_tensor, transpose=True)
        _read_into(f, model.embed.lift[2].bias, read_tensor, transpose=False)
        if has_gcn and use_gcn_file:
            for i in range(min(num_gcn_layers_file, len(model.embed.gcn))):
                _read_gcn_layer_into(f, model.embed.gcn[i], read_tensor)
        elif not has_gcn and use_gcn_file:
            for _ in range(num_gcn_layers_file):
                skip_gcn_layer(f, d_model_lo)
        _read_into(f, model.embed.norm.weight, read_tensor, transpose=False)
        _read_into(f, model.embed.norm.bias, read_tensor, transpose=False)
        _read_into(f, model.enc_input_proj.weight, read_tensor, transpose=True)
        _read_into(f, model.enc_input_proj.bias, read_tensor, transpose=False)
        for block in model.blocks:
            _read_into(f, block.norm1.weight, read_tensor, transpose=False)
            _read_into(f, block.norm1.bias, read_tensor, transpose=False)
            _read_into(f, block.attn.qkv.weight, read_tensor, transpose=True)
            _read_into(f, block.attn.qkv.bias, read_tensor, transpose=False)
            _read_into(f, block.attn.proj.weight, read_tensor, transpose=True)
            _read_into(f, block.attn.proj.bias, read_tensor, transpose=False)
            _read_into(f, block.attn.edge_gate.weight, read_tensor, transpose=True)
            _read_into(f, block.attn.edge_gate.bias, read_tensor, transpose=False)
        _read_into(f, model.leaf_head.weight, read_tensor, transpose=True)
        _read_into(f, model.leaf_head.bias, read_tensor, transpose=False)
        scale = read_tensor(f, (1,), transpose=False).to(model.log_hodlr_scale_leaf.device)
        with torch.no_grad():
            model.log_hodlr_scale_leaf.copy_(torch.log(scale.clamp(min=1e-12)))
        if format_version >= 2 and getattr(model.core, 'use_jacobi', True):
            _read_into(f, model.core.jacobi_gate.weight, read_tensor, transpose=True)
            _read_into(f, model.core.jacobi_gate.bias, read_tensor, transpose=False)
            jacobi_scale = read_tensor(f, (1,), transpose=False).to(model.core.log_jacobi_scale_leaf.device)
            with torch.no_grad():
                model.core.log_jacobi_scale_leaf.copy_(torch.log(jacobi_scale.clamp(min=1e-12)))
        
        scale_head_loaded = False
        extra = f.read(8)
        if len(extra) == 8:
            sentinel, _ = struct.unpack('<II', extra)
            if sentinel == 0xFFFFFFFC and getattr(model, 'universal_off_diag', None) is not None:
                blk = model.universal_off_diag
                _read_into(f, blk.attn_q.weight, read_tensor, transpose=True)
                _read_into(f, blk.attn_q.bias, read_tensor, transpose=False)
                _read_into(f, blk.attn_k.weight, read_tensor, transpose=True)
                _read_into(f, blk.attn_k.bias, read_tensor, transpose=False)
                _read_into(f, blk.attn_v.weight, read_tensor, transpose=True)
                _read_into(f, blk.attn_v.bias, read_tensor, transpose=False)
                
                _read_into(f, blk.W_U, read_tensor, transpose=True)
                _read_into(f, blk.b_U, read_tensor, transpose=False)
                _read_into(f, blk.W_V, read_tensor, transpose=True)
                _read_into(f, blk.b_V, read_tensor, transpose=False)
                
                scale_params = read_tensor(f, (2,), transpose=False).to(model.level_scale_params.device)
                with torch.no_grad():
                    model.level_scale_params.copy_(scale_params)
            elif sentinel == 0xFFFFFFFF or sentinel == 0xFFFFFFFD:
                raise ValueError("Attempted to load an older discrete/hypernetwork checkpoint into Matryoshka version. Clear checkpoint file.")
            extra2 = f.read(8)
            if len(extra2) == 8:
                sentinel2, scale_head_version = struct.unpack('<II', extra2)
                if sentinel2 == 0xFFFFFFFB and hasattr(model, 'scale_head'):
                    if scale_head_version >= 1:
                        _read_into(f, model.scale_head.weight, read_tensor, transpose=True)
                        _read_into(f, model.scale_head.bias, read_tensor, transpose=False)
                    else:
                        w_old = read_tensor(f, (4, d_model_lo), transpose=True).to(model.scale_head.weight.device)
                        b_old = read_tensor(f, (4,), transpose=False).to(model.scale_head.bias.device)
                        with torch.no_grad():
                            model.scale_head.weight.data[:, :GLOBAL_FEATURES_DIM].zero_()
                            model.scale_head.weight.data[:, GLOBAL_FEATURES_DIM:].copy_(w_old)
                            model.scale_head.bias.copy_(b_old)
                    scale_head_loaded = True

    if hasattr(model, 'scale_head') and not scale_head_loaded:
        with torch.no_grad():
            model.scale_head.bias.data[0].copy_(model.log_hodlr_scale_leaf.squeeze())
            if getattr(model.core, 'use_jacobi', True):
                model.scale_head.bias.data[1].copy_(model.core.log_jacobi_scale_leaf.squeeze())
            else:
                model.scale_head.bias.data[1].copy_(torch.tensor(math.log(1.5), device=model.scale_head.bias.device, dtype=model.scale_head.bias.dtype))
            model.scale_head.bias.data[2].copy_(model.level_scale_params[0])
            model.scale_head.bias.data[3].copy_(model.level_scale_params[1])

def _read_into(f, param, read_fn, transpose=False):
    t = read_fn(f, param.shape, transpose=transpose).to(param.device)
    with torch.no_grad():
        param.copy_(t)

RAM_MASK_CACHE = {}

def get_or_compute_masks(frame_path_str, edge_index, off_diag_struct, device, leaf_size=LEAF_SIZE):
    if not off_diag_struct:
        return []
    frame_path = Path(frame_path_str)
    num_blocks = len(off_diag_struct)
    n_super = OFF_DIAG_SUPER
    cache_key = f"{frame_path.name}_b{num_blocks}"
    bin_path = frame_path / f"off_diag_masks_b{num_blocks}.bin"
    if cache_key in RAM_MASK_CACHE:
        return RAM_MASK_CACHE[cache_key]
    if bin_path.exists():
        flat_data = np.fromfile(bin_path, dtype=np.uint8).reshape(num_blocks, n_super, n_super)
        masks = [torch.from_numpy(flat_data[i].copy()).to(device=device, dtype=torch.bool) for i in range(num_blocks)]
        RAM_MASK_CACHE[cache_key] = masks
        return masks
    masks = [
        build_off_diag_super_connectivity(
            edge_index, spec["row_start"], spec["row_end"], spec["col_start"], spec["col_end"],
            device, leaf_size
        )
        for spec in off_diag_struct
    ]
    masks_np = np.stack([m.cpu().numpy().astype(np.uint8) for m in masks])
    masks_np.tofile(bin_path)
    RAM_MASK_CACHE[cache_key] = masks
    return masks


def train_leaf_only():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    default_data = script_dir.parent / "StreamingAssets" / "TestData"
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--data_folder', type=str, default=str(default_data))
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=2, help='LeafBlockAttention heads; must divide d_model')
    parser.add_argument('--frame', type=int, default=600, help='Frame index to use when --use_single_frame is True. Default: 600.')
    parser.add_argument('--use_single_frame', type=bool, default=True, help='If True, train on a single frame (--frame). If False, use --num_frames (random sample or all).')
    parser.add_argument('--num_frames', type=int, default=30, help='When --use_single_frame False: number of frames to randomly sample; 0 = use all frames.')
    parser.add_argument('--save', type=str, default=str(script_dir / "leaf_only_weights.bytes"))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--view_size', type=int, default=VIEW_SIZE, help=f"Number of nodes (padded to power-of-2*{LEAF_SIZE} if needed); 0 = use all nodes in frame. Default {VIEW_SIZE}")
    parser.add_argument('--use_global_node', type=bool, default=True, help='When True, use 33rd global node per block. When False, masked attention on 32x32 only.')
    parser.add_argument('--use_gcn', type=bool, default=True, help='When True, run SparsePhysicsGCN before transformer blocks (graph message passing). When False, raw inputs pass through lift only.')
    parser.add_argument('--print_timing', type=bool, default=True, help='On step 200, print detailed timing of each training substep')
    parser.add_argument('--mixed_sizes', type=bool, default=True, help='Train on randomly sampled sub-graph sizes to force scale invariance. Default False = train on whole matrix per frame.')
    parser.add_argument('--contexts_per_step', type=int, default=1, help='Gradient accumulation: number of random cached contexts per optimizer step.')
    parser.add_argument('--continue_training', action='store_true', help='Load initial weights from the saved .bytes file (--save) and continue training from that state.')
    parser.add_argument('--evaluate_gradients', action='store_true', help='Run gradient interference analysis then exit (uses same data config as training).')
    args = parser.parse_args()
    use_global_node = args.use_global_node
    use_gcn = args.use_gcn

    if args.evaluate_gradients:
        evaluate_gradient_interference(args)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
    print(f"Using device: {device}")

    data_path = Path(args.data_folder)
    if not data_path.exists():
        raise SystemExit(f"Data folder not found: {data_path}")
    run_folder = _most_recent_run_folder(data_path)
    if run_folder != data_path:
        print(f"  [startup] Using most recent run: {run_folder.name}")
    dataset = FluidGraphDataset([run_folder])
    if len(dataset) == 0:
        raise SystemExit(f"No frames found under {run_folder}")

    # Which frames to use: single frame (default) or random sample / all
    if args.use_single_frame:
        frame_idx = min(args.frame, len(dataset) - 1)
        frame_indices = [frame_idx]
        print(f"  [startup] Using single frame index {frame_idx} (--use_single_frame True, --frame {args.frame})")
    else:
        rng = random.Random(args.seed)
        if args.num_frames <= 0:
            frame_indices = list(range(len(dataset)))
            print(f"  [startup] Using all {len(frame_indices)} frames (--num_frames 0)")
        else:
            n_sample = min(args.num_frames, len(dataset))
            frame_indices = sorted(rng.sample(range(len(dataset)), n_sample))
            print(f"  [startup] Random sample of {n_sample} frames (--num_frames {args.num_frames})")

    MIN_MIXED_SIZE = 256
    MAX_MIXED_SIZE = 256
    base_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]

    # Pre-extract and cache sub-graphs from every frame (sizes unchanged)
    training_contexts = []
    for frame_idx in frame_indices:
        batch = dataset[frame_idx]
        num_nodes_real = int(batch['num_nodes'])

        if args.mixed_sizes:
            target_sizes = []
            for s in base_sizes:
                if MIN_MIXED_SIZE <= s <= MAX_MIXED_SIZE and s <= num_nodes_real:
                    target_sizes.append(s)
            if MIN_MIXED_SIZE <= num_nodes_real <= MAX_MIXED_SIZE:
                target_sizes.append(num_nodes_real)
            target_sizes = sorted(set(target_sizes))
        else:
            target_sizes = [num_nodes_real if args.view_size == 0 else max(LEAF_SIZE * 2, args.view_size)]
            if target_sizes[0] > num_nodes_real:
                continue

        for n in target_sizes:
            if n > num_nodes_real:
                continue
            n_pad = next_valid_size(n, LEAF_SIZE)

            x_full = batch['x']
            x_input = x_full[:n].unsqueeze(0).to(device)
            if n_pad > n:
                x_input = F.pad(x_input, (0, 0, 0, n_pad - n), value=0.0)
            # Mean-center the active nodes (ignoring padded zeros)
            active_pos = x_input[0, :n, :3]
            centroid = active_pos.mean(dim=0, keepdim=True)
            x_input[0, :n, :3] = active_pos - centroid

            rows, cols = batch['edge_index'][0], batch['edge_index'][1]
            mask = (rows < n) & (cols < n)
            edge_index = batch['edge_index'][:, mask].to(device)
            edge_values = batch['edge_values'][mask].to(device)

            scale_A = batch.get('scale_A')
            if scale_A is not None and not isinstance(scale_A, torch.Tensor):
                scale_A = torch.tensor(scale_A, device=device, dtype=x_input.dtype)

            A_indices = batch['edge_index'][:, mask]
            A_vals = batch['edge_values'][mask]
            A_sparse = torch.sparse_coo_tensor(A_indices, A_vals, (n, n)).coalesce()

            if device.type == 'mps':
                A_small = A_sparse.to_dense().to(device)
            else:
                A_small = A_sparse.to(device).to_dense()

            A_dense = torch.zeros(n_pad, n_pad, device=device, dtype=A_small.dtype)
            A_dense[:n, :n] = A_small
            A_dense[n:, n:] = torch.eye(n_pad - n, device=device, dtype=A_small.dtype)

            dummy_struct = build_hodlr_off_diag_structure(n_pad, LEAF_SIZE, RANK_BASE_LEVEL1)
            precomputed_masks = get_or_compute_masks(
                batch['frame_path'], edge_index, dummy_struct, device, LEAF_SIZE
            )
            # Use n_pad positions so num_blocks = n_pad//leaf_size matches forward (x.shape[1] == n_pad)
            positions_ctx = x_input[0, :n_pad, :3]
            leaf_attn_mask, leaf_edge_feats = build_leaf_block_connectivity(
                edge_index, edge_values, positions_ctx, scale_A, LEAF_SIZE, device, x_input.dtype, num_hops=ATTENTION_HOPS
            )
            precomputed_leaf_connectivity = (leaf_attn_mask, leaf_edge_feats)

            batch_vectors = max(128, int(round(n_pad ** 0.5)))
            global_feat = batch.get('global_features')
            if global_feat is not None:
                global_feat = global_feat.to(device)

            training_contexts.append({
                'n_pad': n_pad, 'n_orig': n, 'num_leaves': n_pad // LEAF_SIZE,
                'x_input': x_input, 'edge_index': edge_index, 'edge_values': edge_values,
                'scale_A': scale_A, 'A_dense': A_dense, 'precomputed_masks': precomputed_masks,
                'precomputed_leaf_connectivity': precomputed_leaf_connectivity,
                'batch_vectors': batch_vectors,
                'global_features': global_feat,
            })

    if len(training_contexts) == 0:
        raise SystemExit("No valid (frame, size) pairs: ensure frames have at least MIN_MIXED_SIZE nodes.")
    print(f"  [startup] Cached {len(training_contexts)} training contexts")
    contexts_per_step = max(1, int(args.contexts_per_step))
    print(f"  [startup] contexts_per_step = {contexts_per_step} (gradient accumulation)")

    torch.manual_seed(args.seed)
    
    # Initialize model using a dummy maximum padding to ensure parameters are sized correctly
    max_n_pad = max(ctx['n_pad'] for ctx in training_contexts)
    model = LeafOnlyNet(
        input_dim=9, d_model=args.d_model, leaf_size=LEAF_SIZE, num_layers=args.num_layers,
        num_heads=args.num_heads, rank_base=RANK_BASE_LEVEL1, mask_attention=True, use_global_node=use_global_node, use_gcn=use_gcn,
    ).to(device)
    
    if device.type == 'cuda':
        # dynamic=True causes dynamo to use symbolic dims (s0//32 vs s7) that don't unify in attention
        # (scores + bias_physics), leading to shape errors. Use dynamic=False so we trace with concrete
        # shapes; each distinct (n_pad, edge_index size, etc.) recompiles once.
        # Allow more cached compilations so multi-frame / mixed-size training doesn't hit the default limit (8).
        if hasattr(torch._dynamo.config, 'cache_size_limit'):
            torch._dynamo.config.cache_size_limit = 128
        # NOTE: First step(s) per unique size will be slow while compiling, then it will fly.
        model = torch.compile(model, dynamic=False)

    if args.continue_training:
        save_path = Path(args.save)
        if save_path.exists():
            load_leaf_only_weights(model, args.save)
            print(f"  [startup] continue_training: Loaded initial state from {args.save}")
        else:
            raise SystemExit(f"--continue_training given but save file not found: {args.save}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print_hodlr_structure(max_n_pad, LEAF_SIZE, RANK_BASE_LEVEL1)
    
    model.train()
    print_interval = 100
    loss_history = deque(maxlen=print_interval)
    t_start = time.perf_counter()
    t_start_train = t_start
    t_start_avg = None

    def _sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    TIMING_STEP = 200
    for step in range(args.steps):
        do_timing = args.print_timing and (step == TIMING_STEP - 1)
        do_log = (step % print_interval == 0)
        step_loss_sum = 0.0
        ratio_off_sum = 0.0
        ratio_off_count = 0

        if do_timing:
            _sync()
            t0 = time.perf_counter()
            print(f"\n--- Timing triggered on step {TIMING_STEP} (contexts_per_step={contexts_per_step}) ---")
            
        optimizer.zero_grad()
        if do_timing:
            _sync()
            t_zero = time.perf_counter() - t0
            t_forward = 0.0
            t_sample_z = 0.0
            t_az = 0.0
            t_apply_m = 0.0
            t_loss = 0.0
            t_backward = 0.0
            n_orig_t, n_pad_t = None, None

        for micro in range(contexts_per_step):
            # Randomly select a cached graph size for this micro-batch
            ctx = random.choice(training_contexts)
            x_input = ctx['x_input']
            edge_index = ctx['edge_index']
            edge_values = ctx['edge_values']
            scale_A = ctx['scale_A']
            A_dense = ctx['A_dense']
            pre_masks = ctx['precomputed_masks']
            pre_leaf = ctx['precomputed_leaf_connectivity']
            n_pad = ctx['n_pad']
            n_orig = ctx['n_orig']
            batch_vectors = ctx['batch_vectors']
            if do_timing and micro == 0:
                n_orig_t, n_pad_t = n_orig, n_pad

            if do_timing:
                _sync()
                t0 = time.perf_counter()
            global_feat = ctx.get('global_features')
            diag_blocks, off_diag_list = model(
                x_input, edge_index=edge_index, edge_values=edge_values,
                scale_A=scale_A, precomputed_masks=pre_masks, precomputed_leaf_connectivity=pre_leaf,
                global_features=global_feat
            )
            if do_timing:
                _sync()
                t_forward += time.perf_counter() - t0

            if do_timing:
                _sync()
                t0 = time.perf_counter()
            # SAI loss: E_z || M A z - z ||^2
            Z = torch.randn(1, n_pad, batch_vectors, device=device, dtype=x_input.dtype)
            Z[:, n_orig:, :] = 0.0
            if do_timing:
                _sync()
                t_sample_z += time.perf_counter() - t0

            if do_timing:
                _sync()
                t0 = time.perf_counter()
            AZ = (A_dense @ Z.squeeze(0)).unsqueeze(0)
            if do_timing:
                _sync()
                t_az += time.perf_counter() - t0

            if do_timing:
                _sync()
                t0 = time.perf_counter()
            MAZ = apply_block_structured_M(diag_blocks, off_diag_list, AZ, model.off_diag_struct, leaf_size=LEAF_SIZE)
            if do_log:
                MAZ_diag_only = apply_block_structured_M(diag_blocks, [], AZ, [], leaf_size=LEAF_SIZE)
                _norm_diag = MAZ_diag_only.norm().item()
                _norm_off = (MAZ - MAZ_diag_only).norm().item()
                ratio_off_sum += _norm_off / (_norm_diag + 1e-12)
                ratio_off_count += 1
            if do_timing:
                _sync()
                t_apply_m += time.perf_counter() - t0

            if do_timing:
                _sync()
                t0 = time.perf_counter()
            residual = MAZ - Z
            raw_loss = (residual ** 2).mean()
            step_loss_sum += raw_loss.item()
            loss = raw_loss / contexts_per_step
            if do_timing:
                _sync()
                t_loss += time.perf_counter() - t0

            if do_timing:
                _sync()
                t0 = time.perf_counter()
            loss.backward()
            if do_timing:
                _sync()
                t_backward += time.perf_counter() - t0

        step_loss = step_loss_sum / contexts_per_step
        loss_history.append(step_loss)
        _ratio_off = (ratio_off_sum / ratio_off_count) if ratio_off_count > 0 else 0.0

        if step % print_interval == 0:
            def _grad_norm(params):
                t = 0.0
                for p in params:
                    if p.grad is not None:
                        t += p.grad.data.pow(2).sum().item()
                return t ** 0.5
            def _grad_stats(params, thresh=1e-10):
                n, has_grad, nz, mx = len(params), 0, 0, 0.0
                for p in params:
                    if p.grad is not None:
                        has_grad += 1
                        g = p.grad.data
                        nrm = g.pow(2).sum().item() ** 0.5
                        if nrm > thresh:
                            nz += 1
                        mx = max(mx, g.abs().max().item())
                return n, has_grad, nz, mx
            def _get_param_lists():
                all_names = list(model.named_parameters())
                gcn = [p for n, p in all_names if 'embed' in n and 'blocks' not in n and 'leaf_head' not in n]
                attn = [p for n, p in all_names if 'blocks' in n]
                leaf = [p for n, p in all_names if 'leaf_head' in n and 'leaf_scale' not in n]
                off = [p for n, p in all_names if 'universal_off_diag' in n]
                scale = [p for n, p in all_names if 'scale_head' in n]
                if not scale:
                    scale = [p for n, p in all_names if 'log_hodlr_scale_leaf' in n or ('log_jacobi_scale_leaf' in n and getattr(model.core, 'use_jacobi', True)) or 'level_scale_params' in n]
                return gcn, attn, leaf, off, scale
            _gcn, _attn, _leaf, _off, _scale = _get_param_lists()
            log_gcn = _grad_norm(_gcn)
            log_attn = _grad_norm(_attn)
            log_leaf = _grad_norm(_leaf)
            log_off = _grad_norm(_off)
            log_scale = _grad_norm(_scale)
            log_tot = _grad_norm(list(model.parameters()))
            if hasattr(model, '_last_scale_params') and model._last_scale_params is not None:
                sp = model._last_scale_params
                s_leaf = torch.exp(sp[:, 0].mean()).item()
                s_jacobi = torch.exp(sp[:, 1].mean()).item() if getattr(model.core, 'use_jacobi', True) else 0.0
                l0, l1 = sp[:, 2].mean().item(), sp[:, 3].mean().item()
            else:
                s_leaf = torch.exp(model.core.log_hodlr_scale_leaf).item()
                s_jacobi = torch.exp(model.core.log_jacobi_scale_leaf).item() if getattr(model.core, 'use_jacobi', True) else 0.0
                l0, l1 = model.level_scale_params.detach().cpu().tolist()
            ng, hg, nzg, mg = _grad_stats(_gcn)
            na, ha, nza, ma = _grad_stats(_attn)
            nl, hl, nzl, ml = _grad_stats(_leaf)
            no, ho, nzo, mo = _grad_stats(_off)
            nsc, hsc, nzsc, msc = _grad_stats(_scale)

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if do_timing:
            _sync()
            t_clip = time.perf_counter() - t0
            t0 = time.perf_counter()
            
        optimizer.step()
        if do_timing:
            _sync()
            t_optim = time.perf_counter() - t0
            total = t_zero + t_forward + t_sample_z + t_az + t_apply_m + t_loss + t_backward + t_clip + t_optim
            if n_orig_t is not None and n_pad_t is not None:
                print(f"  first micro-batch size: n={n_orig_t} (padded to {n_pad_t})")
            print(f"--- Step {TIMING_STEP} detailed timing (ms) ---")
            print(f"  zero_grad:        {t_zero*1000:8.2f}")
            print(f"  model forward:    {t_forward*1000:8.2f}")
            print(f"  sample Z:         {t_sample_z*1000:8.2f}")
            print(f"  A @ Z:            {t_az*1000:8.2f}")
            print(f"  apply M (MAZ):    {t_apply_m*1000:8.2f}")
            print(f"  residual + loss:  {t_loss*1000:8.2f}")
            print(f"  backward:         {t_backward*1000:8.2f}")
            print(f"  clip_grad_norm:   {t_clip*1000:8.2f}")
            print(f"  optimizer.step:   {t_optim*1000:8.2f}")
            print(f"  total:            {total*1000:8.2f}")
            print("----------------------------------------\n")

        if step % print_interval == 0:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_start
            steps_in_period = step + 1 if step == 0 else print_interval
            jstr = f" sJ={s_jacobi:.3f}" if getattr(model.core, 'use_jacobi', True) else ""
            extra = f" gGCN={log_gcn:.2e} gAttn={log_attn:.2e} gLeaf={log_leaf:.2e} gOff={log_off:.2e} gScl={log_scale:.2e} gTot={log_tot:.2e} sLeaf={s_leaf:.3e}{jstr} lvl=({l0:.2f},{l1:.2f})"
            nz_str = f" nzG={nzg}/{ng} nzA={nza}/{na} nzL={nzl}/{nl} nzO={nzo}/{no} nzS={nzsc}/{nsc}"
            max_str = f" mxG={mg:.1e} mxA={ma:.1e} mxL={ml:.1e} mxO={mo:.1e} mxS={msc:.1e}"
            gate_str = ""
            if getattr(model.core, "use_jacobi", False) and getattr(model.core, "_last_j_gate", None) is not None:
                jg = model.core._last_j_gate
                g_mean = jg.mean().item()
                g_var = jg.var().item()
                gate_str = f" gAvg={g_mean:.3f} gVar={g_var:.3f}"
            n = len(loss_history)
            if n > 0:
                arr = np.array(loss_history)
                loss_avg, loss_std = float(arr.mean()), float(arr.std())
                loss_str = f"Loss avg={loss_avg:.4f} ± {loss_std:.4f}"
            else:
                loss_str = f"Loss {step_loss:.6f}"
            if step >= 200:
                now = time.perf_counter()
                if step == 200:
                    t_start_avg = now
                    avg_per_100 = elapsed
                else:
                    avg_per_100 = (now - t_start_avg) * 100 / (step - 200)
                print(f"{step:05d}: {loss_str}  ({elapsed:.3f}s, avg: {avg_per_100:.3f}s){extra}")
                print(f"       rOff={_ratio_off:.3f}{nz_str}{max_str}{gate_str}")
            else:
                print(f"{step:05d}: {loss_str}  ({elapsed:.3f}s){extra}")
                print(f"       rOff={_ratio_off:.3f}{nz_str}{max_str}{gate_str}")
            
            if step > 0 and step % (print_interval * 10) == 0:
                save_leaf_only_weights(model, args.save, input_dim=9)
            
            t_start = time.perf_counter()

    save_leaf_only_weights(model, args.save, input_dim=9)
    print(f"Saved to {args.save}")


def evaluate_gradient_interference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')

    print(f"\n--- Starting Gradient Interference Analysis on {device} ---")

    # 1. Setup Model and Load Weights
    model = LeafOnlyNet(
        input_dim=9, d_model=args.d_model, leaf_size=LEAF_SIZE, num_layers=args.num_layers,
        num_heads=args.num_heads, rank_base=RANK_BASE_LEVEL1, mask_attention=True,
        use_global_node=args.use_global_node, use_gcn=args.use_gcn,
    ).to(device)

    save_path = Path(args.save)
    if save_path.exists():
        load_leaf_only_weights(model, args.save)
        print(f"Loaded weights from {args.save}")
    else:
        print("WARNING: No saved weights found. Analyzing randomly initialized gradients.")

    model.train()  # Need train mode for gradients

    # 2. Setup Data Contexts: same config as training (single vs multi) but only 10 passes/frames by default
    num_eval = 10
    data_path = Path(args.data_folder)
    run_folder = _most_recent_run_folder(data_path)
    dataset = FluidGraphDataset([run_folder])

    if args.use_single_frame:
        frame_idx = min(args.frame, len(dataset) - 1)
        frame_indices = [frame_idx] * num_eval
        print(f"Evaluating {num_eval} passes on SINGLE frame index {frame_idx}")
    else:
        rng = random.Random(args.seed)
        n_sample = min(num_eval, len(dataset))
        frame_indices = sorted(rng.sample(range(len(dataset)), n_sample))
        print(f"Evaluating across {n_sample} random frames")

    # 3. Define Parameter Groups to Analyze
    param_groups = {
        'Input Embeddings (Lift)': lambda m: list(m.embed.lift.parameters()),
        'Neighbor Impact (GCN)': lambda m: list(m.embed.gcn.parameters()) if m.embed.gcn else [],
        'Transformer Attention': lambda m: [p for b in m.blocks for p in b.parameters()],
        'Off-Diagonal Routing': lambda m: list(m.universal_off_diag.parameters()),
        'Diagonal Leaf Head': lambda m: list(m.core.leaf_head.parameters()),
        'Scale head (predicted)': lambda m: list(m.scale_head.parameters()) if getattr(m, 'scale_head', None) is not None else [],
    }

    group_gradients = {name: [] for name in param_groups.keys()}
    global_features_list = []

    # 4. Extract Gradients per Frame
    for step, frame_idx in enumerate(frame_indices):
        batch = dataset[frame_idx]
        if batch.get('global_features') is not None:
            global_features_list.append(batch['global_features'].detach().cpu().numpy())
        n_orig = int(batch['num_nodes'])
        n_pad = next_valid_size(n_orig, LEAF_SIZE)

        x_input = batch['x'][:n_orig].unsqueeze(0).to(device)
        if n_pad > n_orig:
            x_input = F.pad(x_input, (0, 0, 0, n_pad - n_orig), value=0.0)

        active_pos = x_input[0, :n_orig, :3]
        x_input[0, :n_orig, :3] = active_pos - active_pos.mean(dim=0, keepdim=True)

        rows, cols = batch['edge_index'][0], batch['edge_index'][1]
        mask = (rows < n_orig) & (cols < n_orig)
        edge_index = batch['edge_index'][:, mask].to(device)
        edge_values = batch['edge_values'][mask].to(device)
        scale_A = batch.get('scale_A')
        if scale_A is not None and not isinstance(scale_A, torch.Tensor):
            scale_A = torch.tensor(scale_A, device=device, dtype=x_input.dtype)

        A_sparse = torch.sparse_coo_tensor(edge_index, edge_values, (n_orig, n_orig)).coalesce()
        A_small = A_sparse.to_dense().to(device) if device.type == 'mps' else A_sparse.to(device).to_dense()
        A_dense = torch.zeros(n_pad, n_pad, device=device, dtype=A_small.dtype)
        A_dense[:n_orig, :n_orig] = A_small
        A_dense[n_orig:, n_orig:] = torch.eye(n_pad - n_orig, device=device, dtype=A_small.dtype)

        dummy_struct = build_hodlr_off_diag_structure(n_pad, LEAF_SIZE, RANK_BASE_LEVEL1)
        pre_masks = get_or_compute_masks(batch['frame_path'], edge_index, dummy_struct, device, LEAF_SIZE)
        pre_leaf = build_leaf_block_connectivity(
            edge_index, edge_values, x_input[0, :n_pad, :3], scale_A, LEAF_SIZE, device, x_input.dtype, num_hops=ATTENTION_HOPS
        )

        # Forward Pass
        model.zero_grad()
        global_feat = batch.get('global_features')
        if global_feat is not None:
            global_feat = global_feat.to(device)
            if global_feat.dim() == 1:
                global_feat = global_feat.unsqueeze(0)
        diag_blocks, off_diag_list = model(
            x_input, edge_index=edge_index, edge_values=edge_values,
            scale_A=scale_A, precomputed_masks=pre_masks, precomputed_leaf_connectivity=pre_leaf,
            global_features=global_feat
        )

        # Compute Loss
        batch_vectors = max(128, int(round(n_pad ** 0.5)))
        Z = torch.randn(1, n_pad, batch_vectors, device=device, dtype=x_input.dtype)
        Z[:, n_orig:, :] = 0.0
        AZ = (A_dense @ Z.squeeze(0)).unsqueeze(0)
        MAZ = apply_block_structured_M(diag_blocks, off_diag_list, AZ, model.off_diag_struct, leaf_size=LEAF_SIZE)

        loss = ((MAZ - Z) ** 2).mean()
        loss.backward()

        # Extract and flatten gradients for this step
        for group_name, get_params in param_groups.items():
            params = get_params(model)
            if not params:
                continue

            grads = []
            for p in params:
                if p.grad is not None:
                    grads.append(p.grad.detach().clone().view(-1))
                else:
                    grads.append(torch.zeros_like(p).view(-1))

            flat_grad = torch.cat(grads)
            group_gradients[group_name].append(flat_grad)

        print(f"  Processed frame {frame_idx} (Loss: {loss.item():.4f})")

    # 5. Calculate Metrics
    print("\n=== Gradient Interference Report ===")

    for group_name, grads in group_gradients.items():
        if not grads:
            continue

        grads_stack = torch.stack(grads)  # Shape: (num_frames, num_params)
        num_frames = grads_stack.shape[0]

        # Mean gradient and variance across frames
        mean_grad = grads_stack.mean(dim=0)
        var_grad = grads_stack.var(dim=0, unbiased=False)

        # SNR: ||mu|| / sqrt(mean(variance) + eps)
        signal_norm = mean_grad.norm().item()
        noise_norm = torch.sqrt(var_grad.mean() + 1e-12).item()
        snr = signal_norm / noise_norm if noise_norm > 0 else float('inf')

        # Pairwise Cosine Similarity
        cos_sims = []
        for i in range(num_frames):
            for j in range(i + 1, num_frames):
                sim = F.cosine_similarity(grads_stack[i].unsqueeze(0), grads_stack[j].unsqueeze(0), dim=1).item()
                cos_sims.append(sim)

        avg_cos_sim = sum(cos_sims) / len(cos_sims) if cos_sims else 1.0

        print(f"{group_name.upper()}:")
        print(f"  Avg Cosine Similarity : {avg_cos_sim:8.4f}  (1.0 = aligned, 0.0 = orthogonal, -1.0 = fighting)")
        print(f"  Signal-to-Noise (SNR) : {snr:8.4f}")
        print(f"  Gradient Magnitude     : {signal_norm:8.4e}")
        print("-" * 40)

    if global_features_list:
        gf_arr = np.array(global_features_list)
        gf_mean = gf_arr.mean(axis=0)
        gf_std = gf_arr.std(axis=0)
        print("FRAME STATS (global_features, mean ± std across evaluated frames):")
        print(f"  log2(N), log(scale_A), mass_mu, mass_std, diag_mu, diag_std, diff_mu/std(3): {gf_mean.tolist()}")
        print(f"  stds: {gf_std.tolist()}")
        print("-" * 40)


if __name__ == "__main__":
    train_leaf_only()