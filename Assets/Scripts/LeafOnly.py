"""
LeafOnly: train HODLR-style block preconditioner (32x32 diagonal leaves + off-diagonal low-rank blocks).
Attention is within each 32-node block. Run with python3 LeafOnly.py; view_size defaults to 512 (16 leaves).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import argparse
import struct
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

        pos_scale = float(np.abs(positions).max())
        if pos_scale <= 0.0: pos_scale = 1.0
        positions_n = positions / pos_scale

        # 2^layer (no normalization); matches ApplyPressureGradient scale exp2((float)node.layer)
        layer_val = np.exp2(layer)

        mass_max = float(np.max(mass))
        if mass_max <= 1e-9: mass_max = 1.0
        mass_n = mass / mass_max

        scale_A = float(np.abs(vals).max())
        if scale_A <= 0.0: scale_A = 1.0
        diag_map_n = diag_map / scale_A

        # Diffusion gradient: float3 per node (points toward fluid bulk, away from free surface)
        dg_path = frame_path / "diffusion_gradient.bin"
        if dg_path.exists():
            diffusion_grad = np.fromfile(dg_path, dtype=np.float32).reshape(num_nodes, 3)
        else:
            diffusion_grad = np.zeros((num_nodes, 3), dtype=np.float32)

        # Input: position (3) + layer (1) + mass (1) + diffusion_gradient (3) + diagonal (1) = 9
        n_float = 9
        x = np.zeros((num_nodes, n_float), dtype=np.float32)
        x[:, :3] = positions_n
        x[:, 3] = layer_val
        x[:, 4] = mass_n
        x[:, 5:8] = diffusion_grad
        x[:, 8] = diag_map_n

        return {
            'x': torch.from_numpy(x).float(),
            'edge_index': torch.stack([torch.from_numpy(rows.astype(np.int64)), torch.from_numpy(cols.astype(np.int64))]),
            'edge_values': torch.from_numpy(vals.copy()),
            'num_nodes': int(num_nodes),
            'scale_A': scale_A,
            'frame_path': str(frame_path),
        }

# --- Embedding and graph layers ---

class SparsePhysicsGCN(nn.Module):
    """Message passing layer: neighbor features scaled by PDE matrix values (edge_values)."""
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
    """Lift 4D input to d_model; when use_gcn=True run exactly 2 physics-weighted message passes (GCN) with the graph."""
    NUM_GCN_LAYERS = 2

    def __init__(self, input_dim, d_model, use_gcn=True):
        super().__init__()
        self.use_gcn = use_gcn
        self.lift = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.gcn = nn.ModuleList([SparsePhysicsGCN(d_model) for _ in range(self.NUM_GCN_LAYERS)]) if use_gcn else None
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, edge_index=None, edge_values=None, scale_A=None):
        h = self.lift(x)
        if self.use_gcn and self.gcn is not None and edge_index is not None and edge_values is not None:
            for gcn_layer in self.gcn:
                h = gcn_layer(h, edge_index, edge_values, scale_A)
        return self.norm(h)

# --- Leaf block connectivity and attention ---

def build_leaf_block_physics_bias(edge_index, edge_values, scale_A, leaf_size, device, dtype=torch.float32, num_nodes=None):
    """Build per-block (num_blocks, leaf_size, leaf_size) physics bias from A-matrix edges. 1 on diagonal, 0 off-edges, edge value on edges. No mask, no global node. num_nodes: if provided (e.g. N_pad), num_blocks = num_nodes // leaf_size."""
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


def build_leaf_block_connectivity(edge_index, edge_values, positions, scale_A, leaf_size, device, dtype=torch.float32):
    """Build per-block attention mask and edge features for leaf blocks."""
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

    attn_mask = torch.zeros(num_blocks, leaf_size, leaf_size + 1, device=device, dtype=dtype)
    edge_feats = torch.zeros(num_blocks, leaf_size, leaf_size + 1, 4, device=device, dtype=dtype)
    attn_mask[:, torch.arange(leaf_size, device=device), torch.arange(leaf_size, device=device)] = 1.0
    edge_feats[:, torch.arange(leaf_size, device=device), torch.arange(leaf_size, device=device), :] = 0.0
    attn_mask[:, :, leaf_size] = 1.0
    attn_mask[b_l, r_l, c_l] = 1.0
    edge_feats[b_l, r_l, c_l, :] = edge_feats_flat.to(device)
    return attn_mask, edge_feats

class LeafBlockAttention(nn.Module):
    """Attention within each leaf block. mask_attention=True: -inf mask; use_global_node=True adds 33rd node. mask_attention=False: dense 32x32, physics bias only."""
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
        self.edge_gate = nn.Linear(1, self.num_heads)
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

        w = edge_feats[..., 3].clone()
        w[:, :, self.block_size] = 1.0
        w[:, torch.arange(self.block_size, device=device), torch.arange(self.block_size, device=device)] = 1.0
        bias_physics = w
        scores = scores + bias_physics.unsqueeze(0).unsqueeze(-1)
        mask_expanded = attn_mask.unsqueeze(0).unsqueeze(-1)
        scores = scores.masked_fill(mask_expanded == 0, float('-inf'))

        if save_attention:
            with torch.no_grad():
                self.last_scores_matrix = scores.mean(dim=-1)[:, :, :, :self.block_size].cpu().float()
                self.last_bias_physics_matrix = bias_physics[:, :, :self.block_size].cpu().float()

        attn_probs = F.softmax(scores, dim=3)

        linear_edge_weights = self.edge_gate(bias_physics.unsqueeze(-1))
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
        """Masked 32x32 attention only; no 33rd global node. Uses -inf mask and edge_feats on the 32x32 block."""
        C = x_blk.shape[-1]
        attn_mask = attn_mask[:, :, :self.block_size]
        edge_feats = edge_feats[:, :, :self.block_size, :]
        qkv = self.qkv(x_blk)
        q = qkv[..., :C].view(B, num_blocks, self.block_size, self.num_heads, self.head_dim)
        k = qkv[..., C:2*C].view(B, num_blocks, self.block_size, self.num_heads, self.head_dim)
        v = qkv[..., 2*C:3*C].view(B, num_blocks, self.block_size, self.num_heads, self.head_dim)
        scores = torch.einsum('bnqhd,bnkhd->bnqkh', q, k) * self.scale
        w = edge_feats[..., 3].clone()
        w[:, torch.arange(self.block_size, device=device), torch.arange(self.block_size, device=device)] = 1.0
        bias_physics = w
        scores = scores + bias_physics.unsqueeze(0).unsqueeze(-1)
        mask_expanded = attn_mask.unsqueeze(0).unsqueeze(-1)
        scores = scores.masked_fill(mask_expanded == 0, float('-inf'))
        if save_attention:
            with torch.no_grad():
                self.last_scores_matrix = scores.mean(dim=-1).cpu().float()
                self.last_bias_physics_matrix = bias_physics.cpu().float()
        attn_probs = F.softmax(scores, dim=3)
        linear_edge_weights = self.edge_gate(bias_physics.unsqueeze(-1)).unsqueeze(0).masked_fill(mask_expanded == 0, 0.0)
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
        """Dense 32x32 attention; physics bias only, no -inf mask. No global node."""
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
        scores = scores + physics_bias.unsqueeze(0).unsqueeze(-1)
        attn_probs = F.softmax(scores, dim=3)
        linear_edge_weights = self.edge_gate(physics_bias.unsqueeze(-1)).unsqueeze(0)
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
    """Pre-norm attention with residual: x = x + attn(norm1(x)). No MLP."""
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
# Number of nodes (view size); padded to next power-of-2 * LEAF_SIZE if needed (e.g. 500 -> 512).
VIEW_SIZE = 512


def next_valid_size(n, leaf_size=LEAF_SIZE):
    """Smallest value >= n that is power-of-2 * leaf_size (for HODLR structure)."""
    if n <= 0:
        return leaf_size * 2
    num_blocks = (n + leaf_size - 1) // leaf_size
    if num_blocks <= 1:
        return leaf_size * 2
    p = 1
    while p < num_blocks:
        p *= 2
    return p * leaf_size
# Rank for smallest off-diagonal (side=32); larger blocks scale as rank_base * (side/32)^(2/3).
RANK_BASE_LEVEL1 = 16
OFF_DIAG_SUPER = 32  # Off-diagonal attention is always over 32 super-nodes per side (each = group of side/32 tokens).


def _rank_for_side(side, rank_base=RANK_BASE_LEVEL1, min_rank=4, max_rank=128):
    """Rank for off-diagonal block of side length `side`; scale n^(2/3) with base 20 at side 32."""
    r = rank_base * ((side / 32.0) ** (2.0 / 3.0))
    r = max(min_rank, min(max_rank, int(round(r))))
    return r if r % 2 == 0 else r + 1  # keep even for stability


def build_hodlr_off_diag_structure(n_nodes, leaf_size=LEAF_SIZE, rank_base=RANK_BASE_LEVEL1):
    """
    Build HODLR off-diagonal block structure for n_nodes (must be power-of-2 * leaf_size).
    Returns list of specs, each: row_start, row_end, col_start, col_end, side, rank, level.
    Levels are 1 (smallest blocks) to depth (one big block). Order: smallest first.
    """
    num_blocks = n_nodes // leaf_size
    if num_blocks & (num_blocks - 1) != 0:
        raise ValueError(f"n_nodes must be power-of-2 * leaf_size; got n_nodes={n_nodes}, leaf_size={leaf_size}")
    depth = int(round(math.log2(num_blocks)))
    specs = []
    # Level 1 = smallest blocks (side 32), level depth = one big block. Order: smallest first.
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
    """
    Build 32x32 connectivity mask for off-diagonal block [rs:re] x [cs:ce].
    Super-node i (row) is connected to super-node j (col) iff there exists an original edge
    from any token in row group i to any token in col group j (logical OR).
    """
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
    """Print HODLR block structure (layers, # blocks per layer, rank per block) before training."""
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
    """
    Shared leaf path: embed -> enc_input_proj -> transformer blocks (leaf attention) -> leaf_head -> U@U.T + Jacobi diag.
    Input x: (B, N, 9) = position (3), layer (1), mass (1), diffusion_gradient (3), diagonal (1). Supports any N divisible by leaf_size.
    mask_attention=True: -inf masked attention; use_global_node=True adds 33rd node. mask_attention=False: dense 32x32, no mask.
    Output shape: (B, num_leaves, leaf_size, leaf_size).
    """
    def __init__(self, input_dim=9, d_model=128, leaf_size=32, num_layers=3, num_heads=4, mask_attention=True, use_global_node=True, use_gcn=True):
        super().__init__()
        self.leaf_size = leaf_size
        self.mask_attention = mask_attention
        self.use_global_node = use_global_node
        self.embed = PhysicsAwareEmbedding(input_dim, d_model, use_gcn=use_gcn)
        self.enc_input_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, block_size=leaf_size, attn_module=LeafBlockAttention(d_model, leaf_size, num_heads=num_heads, mask_attention=mask_attention, use_global_node=use_global_node))
            for _ in range(num_layers)
        ])
        self.leaf_head = nn.Linear(d_model, leaf_size)
        nn.init.normal_(self.leaf_head.weight, std=0.01)
        nn.init.constant_(self.leaf_head.bias, 0.0)
        self.log_hodlr_scale_leaf = nn.Parameter(torch.ones(1) * math.log(1e-2))

    def get_leaf_blocks(self, h, x, scale_A=None):
        """From encoder output h (B, N, d_model) and input x (B, N, input_dim), compute dense leaf blocks (B, num_leaves, leaf_size, leaf_size). Diag from x[..., 8]."""
        B, N, _ = h.shape
        num_leaves = N // self.leaf_size
        h_leaves = h.view(B, num_leaves, self.leaf_size, -1)
        u_leaf = self.leaf_head(h_leaves)
        leaf_scale = torch.exp(self.log_hodlr_scale_leaf)
        leaf_base = torch.matmul(u_leaf, u_leaf.transpose(-1, -2)) * leaf_scale
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
        return leaf_base + torch.diag_embed(new_diag)

    def forward_features(self, x, edge_index=None, edge_values=None, scale_A=None, save_attention=False):
        """Embed -> enc_input_proj -> blocks; returns h (B, N, d_model) for use by off-diag heads."""
        B, N, _ = x.shape
        positions = x[0, :, :3] if x.dim() == 3 else x[:, :3]
        h = self.embed(x, edge_index, edge_values, scale_A)
        h = self.enc_input_proj(h)
        attn_mask, edge_feats, physics_bias = None, None, None
        if edge_index is not None and edge_values is not None:
            device, dtype = x.device, x.dtype
            if self.mask_attention and positions is not None:
                attn_mask, edge_feats = build_leaf_block_connectivity(
                    edge_index, edge_values, positions, scale_A, self.leaf_size, device, dtype
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
        """Full forward: embed -> enc_input_proj -> blocks -> get_leaf_blocks. x: (B, N, input_dim), N divisible by leaf_size."""
        h = self.forward_features(x, edge_index, edge_values, scale_A, save_attention)
        return self.get_leaf_blocks(h, x, scale_A)


class OffDiagBlock(nn.Module):
    """
    Off-diagonal block with 32x32 masked attention. Super-nodes are downsampled for attention;
    connectivity is logical-OR over original edges. Rows of U/V use original per-token h so we
    get (B, side, rank) without upsampling: U[i] = proj_U(concat(h_row[i], row_super[i//g])).
    """
    def __init__(self, d_model, side, rank):
        super().__init__()
        self.side = side
        self.rank = rank
        self.n_super = OFF_DIAG_SUPER
        self.group_size = side // self.n_super  # 1, 2, or 4
        self.attn_q = nn.Linear(d_model, d_model)
        self.attn_k = nn.Linear(d_model, d_model)
        self.attn_v = nn.Linear(d_model, d_model)
        # Per-row from original token + super-node summary (no upsampling)
        self.proj_U = nn.Linear(2 * d_model, rank)
        self.proj_V = nn.Linear(2 * d_model, rank)
        for m in (self.attn_q, self.attn_k, self.attn_v, self.proj_U, self.proj_V):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
        self._scale_attn = d_model ** -0.5

    def forward(self, h_row, h_col, mask, scale):
        # h_row (B, side, d_model), h_col (B, side, d_model); mask (32, 32) bool
        B = h_row.shape[0]
        g = self.group_size
        down_row = h_row.view(B, self.n_super, g, -1).mean(dim=2)   # (B, 32, d_model)
        down_col = h_col.view(B, self.n_super, g, -1).mean(dim=2)   # (B, 32, d_model)
        Q = self.attn_q(down_row)
        K = self.attn_k(down_col)
        V = self.attn_v(down_col)
        scores = (Q @ K.transpose(-2, -1)) * self._scale_attn
        scores = scores.masked_fill(~mask.unsqueeze(0), float('-inf'))  # (1,32,32) mask on (B,32,32)
        attn_probs = F.softmax(scores, dim=-1)
        row_out = (attn_probs @ V).reshape(B, self.n_super, -1)   # (B, 32, d_model)
        # Expand super-node output to side: row_out_expanded[i] = row_out[i//g]
        row_out_expanded = row_out.repeat_interleave(g, dim=1)   # (B, side, d_model)
        down_col_expanded = down_col.repeat_interleave(g, dim=1)   # (B, side, d_model)
        # Ensure 3D for cat (avoid 3/4 dim mismatch from any broadcast)
        h_row_3 = h_row.reshape(B, -1, h_row.shape[-1])
        row_exp_3 = row_out_expanded.reshape(B, -1, row_out_expanded.shape[-1])
        h_col_3 = h_col.reshape(B, -1, h_col.shape[-1])
        col_exp_3 = down_col_expanded.reshape(B, -1, down_col_expanded.shape[-1])
        U = self.proj_U(torch.cat([h_row_3, row_exp_3], dim=-1))   # (B, side, rank)
        V = self.proj_V(torch.cat([h_col_3, col_exp_3], dim=-1))   # (B, side, rank)
        return U * scale, V * scale


class LeafOnlyNet(nn.Module):
    """
    Wrapper around LeafCore with HODLR off-diagonals. N must equal n_nodes (divisible by leaf_size).
    Outputs (diag_blocks, off_diag_list); off_diag_list is empty list when n_nodes has only one block.
    """
    def __init__(self, input_dim=9, d_model=128, leaf_size=32, num_layers=3, num_heads=4, n_nodes=512, rank_base=RANK_BASE_LEVEL1, mask_attention=True, use_global_node=True, use_gcn=True):
        super().__init__()
        self.leaf_size = leaf_size
        self.n_nodes = n_nodes
        self.core = LeafCore(input_dim=input_dim, d_model=d_model, leaf_size=leaf_size, num_layers=num_layers, num_heads=num_heads, mask_attention=mask_attention, use_global_node=use_global_node, use_gcn=use_gcn)
        self.off_diag_struct = build_hodlr_off_diag_structure(n_nodes, leaf_size, rank_base)
        if self.off_diag_struct:
            self.log_off_diag_scale = nn.Parameter(torch.ones(1) * math.log(1e-2))
            # One OffDiagBlock per HODLR level (all blocks at same scale share QKV)
            by_level = {s["level"]: (s["side"], s["rank"]) for s in self.off_diag_struct}
            self.off_diag_levels = nn.ModuleList([
                OffDiagBlock(d_model, side, rank)
                for _level in sorted(by_level.keys())
                for (side, rank) in [by_level[_level]]
            ])
        else:
            self.off_diag_levels = []
            self.log_off_diag_scale = None

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

    def _off_diag(self, h, edge_index, precomputed_masks=None):
        """h (B, N, d_model), edge_index (2, E). Return list of (U, V) per off_diag_struct; U,V shape (B, side, rank). Use precomputed_masks if provided, else compute dynamically (inference)."""
        B, N, C = h.shape
        scale = torch.exp(self.log_off_diag_scale)
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
                    for spec in self.off_diag_struct
                ]
                self._off_diag_mask_cache_id = cache_id
            mask_list = self._off_diag_mask_cache
        result = []
        for idx, spec in enumerate(self.off_diag_struct):
            rs, re = spec["row_start"], spec["row_end"]
            cs, ce = spec["col_start"], spec["col_end"]
            h_row = h[:, rs:re]
            h_col = h[:, cs:ce]
            mask = mask_list[idx]
            level_module = self.off_diag_levels[spec["level"] - 1]
            U, V = level_module(h_row, h_col, mask, scale)
            result.append((U, V))
        return result

    def forward(self, x, edge_index=None, edge_values=None, scale_A=None, save_attention=False, precomputed_masks=None):
        B, N, _ = x.shape
        assert N % self.leaf_size == 0, f"LeafOnly expects N divisible by leaf_size {self.leaf_size}, got {N}"
        if self.off_diag_struct and N == self.n_nodes:
            h = self.core.forward_features(x, edge_index=edge_index, edge_values=edge_values, scale_A=scale_A, save_attention=save_attention)
            diag_blocks = self.core.get_leaf_blocks(h, x, scale_A)
            off_diag_list = self._off_diag(h, edge_index, precomputed_masks=precomputed_masks)
            return (diag_blocks, off_diag_list)
        diag_blocks = self.core(x, edge_index=edge_index, edge_values=edge_values, scale_A=scale_A, save_attention=save_attention)
        return (diag_blocks, [])


def apply_block_structured_M(diag_blocks, off_diag_list, x, off_diag_struct, leaf_size=LEAF_SIZE):
    """
    Apply SPD block-structured M to x. Diagonal from diag_blocks; off-diagonals from off_diag_list
    using off_diag_struct (list of row_start, row_end, col_start, col_end, side, rank).
    Diagonal applied in one batched einsum (no Python loop over leaves).
    """
    B, N, K = x.shape
    num_leaves = N // leaf_size
    x_blk = x.view(B, num_leaves, leaf_size, K)
    out = torch.einsum('blij,bljk->blik', diag_blocks, x_blk).reshape(B, N, K)
    for idx, spec in enumerate(off_diag_struct):
        U, V = off_diag_list[idx]   # (B, side, rank)
        rs, re = spec["row_start"], spec["row_end"]
        cs, ce = spec["col_start"], spec["col_end"]
        x_col = x[:, cs:ce]   # (B, side, K)
        x_row = x[:, rs:re]   # (B, side, K)
        out[:, rs:re] = out[:, rs:re] + torch.matmul(U, torch.matmul(V.transpose(1, 2), x_col))
        out[:, cs:ce] = out[:, cs:ce] + torch.matmul(V, torch.matmul(U.transpose(1, 2), x_row))
    return out


def apply_leaf_only(leaf_blocks, x, off_diag_list=None, off_diag_struct=None):
    """Apply M to x. If off_diag_list/off_diag_struct missing or empty: block-diagonal only."""
    B, N, K = x.shape
    num_leaves = leaf_blocks.shape[1]
    leaf_size = leaf_blocks.shape[2]
    if not off_diag_list or not off_diag_struct:
        x_leaves = x.view(B, num_leaves, leaf_size, K)
        y_leaves = torch.matmul(leaf_blocks, x_leaves)
        return y_leaves.view(B, N, K)
    return apply_block_structured_M(leaf_blocks, off_diag_list, x, off_diag_struct, leaf_size=leaf_size)


# --- Save / Load (for InspectModel) ---

LEAF_ONLY_HEADER_BYTES_OLD = 20   # d_model, leaf_size, input_dim, num_layers, num_heads
LEAF_ONLY_HEADER_BYTES_V2 = 24    # + use_gcn (int)
LEAF_ONLY_HEADER_BYTES = 28       # + num_gcn_layers (int); when use_gcn, 1 or 2

def read_leaf_only_header(path):
    path = Path(path)
    with open(path, 'rb') as f:
        header = f.read(LEAF_ONLY_HEADER_BYTES)
    if len(header) < LEAF_ONLY_HEADER_BYTES_OLD:
        raise ValueError("LeafOnly weights file too short")
    if len(header) >= LEAF_ONLY_HEADER_BYTES:
        d_model, leaf_size, input_dim, num_layers, num_heads, use_gcn, num_gcn_layers = struct.unpack('<iiiiiii', header)
        return d_model, leaf_size, input_dim, num_layers, num_heads, use_gcn, num_gcn_layers, LEAF_ONLY_HEADER_BYTES
    if len(header) >= LEAF_ONLY_HEADER_BYTES_V2:
        d_model, leaf_size, input_dim, num_layers, num_heads, use_gcn = struct.unpack('<iiiiii', header)
        return d_model, leaf_size, input_dim, num_layers, num_heads, use_gcn, 1, LEAF_ONLY_HEADER_BYTES_V2  # 1 GCN layer in old files
    d_model, leaf_size, input_dim, num_layers, num_heads = struct.unpack('<iiiii', header)
    return d_model, leaf_size, input_dim, num_layers, num_heads, 1, 1, LEAF_ONLY_HEADER_BYTES_OLD  # use_gcn=1, 1 layer for oldest files


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
    with open(path, 'wb') as f:
        f.write(struct.pack('<iiiiiii', d_model, model.leaf_size, input_dim, len(model.blocks), num_heads, int(use_gcn), num_gcn_layers))
        # Embed: lift, [gcn layers if use_gcn], norm
        _write_packed_tensor(f, model.embed.lift[0].weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.lift[0].bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.lift[2].weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.lift[2].bias.detach().cpu().float(), transpose=False)
        if use_gcn:
            for gcn_layer in model.embed.gcn:
                _write_gcn_layer(f, gcn_layer)
        _write_packed_tensor(f, model.embed.norm.weight.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.norm.bias.detach().cpu().float(), transpose=False)
        # enc_input_proj
        _write_packed_tensor(f, model.enc_input_proj.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.enc_input_proj.bias.detach().cpu().float(), transpose=False)
        # Blocks
        for block in model.blocks:
            _write_packed_tensor(f, block.norm1.weight.detach().cpu().float(), False)
            _write_packed_tensor(f, block.norm1.bias.detach().cpu().float(), False)
            _write_packed_tensor(f, block.attn.qkv.weight.detach().cpu().float(), True)
            _write_packed_tensor(f, block.attn.qkv.bias.detach().cpu().float(), False)
            _write_packed_tensor(f, block.attn.proj.weight.detach().cpu().float(), True)
            _write_packed_tensor(f, block.attn.proj.bias.detach().cpu().float(), False)
            _write_packed_tensor(f, block.attn.edge_gate.weight.detach().cpu().float(), True)
            _write_packed_tensor(f, block.attn.edge_gate.bias.detach().cpu().float(), False)
        # leaf_head + scale
        _write_packed_tensor(f, model.leaf_head.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.leaf_head.bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, torch.exp(model.log_hodlr_scale_leaf).detach().cpu().float(), transpose=False)
        # Off-diagonal: sentinel then OffDiagBlocks (downsampled or full-res) + scale
        if getattr(model, 'off_diag_levels', None) and len(model.off_diag_levels) > 0:
            f.write(struct.pack('<II', 0xFFFFFFFF, 0))
            for blk in model.off_diag_levels:
                _write_packed_tensor(f, blk.attn_q.weight.detach().cpu().float(), transpose=True)
                _write_packed_tensor(f, blk.attn_q.bias.detach().cpu().float(), transpose=False)
                _write_packed_tensor(f, blk.attn_k.weight.detach().cpu().float(), transpose=True)
                _write_packed_tensor(f, blk.attn_k.bias.detach().cpu().float(), transpose=False)
                _write_packed_tensor(f, blk.attn_v.weight.detach().cpu().float(), transpose=True)
                _write_packed_tensor(f, blk.attn_v.bias.detach().cpu().float(), transpose=False)
                _write_packed_tensor(f, blk.proj_U.weight.detach().cpu().float(), transpose=True)
                _write_packed_tensor(f, blk.proj_U.bias.detach().cpu().float(), transpose=False)
                _write_packed_tensor(f, blk.proj_V.weight.detach().cpu().float(), transpose=True)
                _write_packed_tensor(f, blk.proj_V.bias.detach().cpu().float(), transpose=False)
            _write_packed_tensor(f, torch.exp(model.log_off_diag_scale).detach().cpu().float(), transpose=False)


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
    d_model_lo, leaf_size_lo, input_dim_lo, num_layers_lo, _num_heads_lo, use_gcn_file, num_gcn_layers_file, header_bytes = result
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
        # Embed: lift, [gcn layers if use_gcn_file], norm
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
        # Optional off-diagonal section: sentinel (8 bytes) then blocks + scale
        extra = f.read(8)
        if len(extra) == 8:
            sentinel, _ = struct.unpack('<II', extra)
            if sentinel == 0xFFFFFFFF and getattr(model, 'off_diag_levels', None) and len(model.off_diag_levels) > 0:
                for blk in model.off_diag_levels:
                    _read_into(f, blk.attn_q.weight, read_tensor, transpose=True)
                    _read_into(f, blk.attn_q.bias, read_tensor, transpose=False)
                    _read_into(f, blk.attn_k.weight, read_tensor, transpose=True)
                    _read_into(f, blk.attn_k.bias, read_tensor, transpose=False)
                    _read_into(f, blk.attn_v.weight, read_tensor, transpose=True)
                    _read_into(f, blk.attn_v.bias, read_tensor, transpose=False)
                    _read_into(f, blk.proj_U.weight, read_tensor, transpose=True)
                    _read_into(f, blk.proj_U.bias, read_tensor, transpose=False)
                    _read_into(f, blk.proj_V.weight, read_tensor, transpose=True)
                    _read_into(f, blk.proj_V.bias, read_tensor, transpose=False)
                scale_off = read_tensor(f, (1,), transpose=False).to(model.log_off_diag_scale.device)
                with torch.no_grad():
                    model.log_off_diag_scale.copy_(torch.log(scale_off.clamp(min=1e-12)))


def _read_into(f, param, read_fn, transpose=False):
    t = read_fn(f, param.shape, transpose=transpose).to(param.device)
    with torch.no_grad():
        param.copy_(t)


# Global RAM cache for off-diag masks (frame_path key -> list of bool tensors)
RAM_MASK_CACHE = {}


def get_or_compute_masks(frame_path_str, edge_index, off_diag_struct, device, leaf_size=LEAF_SIZE):
    """Fetches masks from RAM, then disk, or computes and saves them. Returns list of (n_super, n_super) bool tensors on device."""
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
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--data_folder', type=str, default=str(default_data))
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=2, help='LeafBlockAttention heads; must divide d_model')
    parser.add_argument('--frame', type=int, default=600)
    parser.add_argument('--save', type=str, default=str(script_dir / "leaf_only_weights.bytes"))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--view_size', type=int, default=VIEW_SIZE, help=f"Number of nodes (padded to power-of-2*{LEAF_SIZE} if needed); 0 = use all nodes in frame. Default {VIEW_SIZE}")
    parser.add_argument('--use_global_node', type=bool, default=True, help='When True, use 33rd global node per block. When False, masked attention on 32x32 only.')
    parser.add_argument('--use_gcn', type=bool, default=True, help='When True, run SparsePhysicsGCN before transformer blocks (graph message passing). When False, raw inputs pass through lift only.')
    parser.add_argument('--print_timing', type=bool, default=True, help='On step 200, print detailed timing of each training substep')
    args = parser.parse_args()
    use_global_node = args.use_global_node
    use_gcn = args.use_gcn

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
    frame_idx = min(args.frame, len(dataset) - 1)
    batch = dataset[frame_idx]

    if args.view_size == 0:
        n = int(batch['num_nodes'])
        print(f"  view_size=0: using all nodes in frame: N={n}")
    else:
        n = max(LEAF_SIZE * 2, args.view_size)  # at least 2 leaves
    n_pad = next_valid_size(n, LEAF_SIZE)
    if n_pad != n:
        print(f"  Padding view: {n} nodes -> {n_pad} (power-of-2 * {LEAF_SIZE})")
    num_leaves = n_pad // LEAF_SIZE

    x_full = batch['x']  # (num_nodes, 4)
    x_input = x_full[:n].unsqueeze(0).to(device)  # (1, n, 4)
    if n_pad > n:
        x_input = F.pad(x_input, (0, 0, 0, n_pad - n), value=0.0)  # (1, n_pad, 4)
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
    # A_pad: first n rows/cols = A, padded block = identity so padded dofs pass through
    A_dense = torch.zeros(n_pad, n_pad, device=device, dtype=A_small.dtype)
    A_dense[:n, :n] = A_small
    A_dense[n:, n:] = torch.eye(n_pad - n, device=device, dtype=A_small.dtype)

    torch.manual_seed(args.seed)
    model = LeafOnlyNet(
        input_dim=9, d_model=args.d_model, leaf_size=LEAF_SIZE, num_layers=args.num_layers,
        num_heads=args.num_heads, n_nodes=n_pad, mask_attention=True, use_global_node=use_global_node, use_gcn=use_gcn,
    ).to(device)
    precomputed_masks = get_or_compute_masks(
        batch['frame_path'], edge_index, model.off_diag_struct, device, LEAF_SIZE
    )
    # MPS has a 31 constant-buffer limit; Inductor backward can exceed it. Compile only on CUDA.
    if device.type == 'cuda':
        model = torch.compile(model)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print_hodlr_structure(n_pad, LEAF_SIZE, RANK_BASE_LEVEL1)
    print(f"Ready: {run_folder.name} frame_{frame_idx:04d}, first {n} nodes (padded to {n_pad}, {num_leaves} leaves), {edge_index.shape[1]} edges, {args.num_layers} layers, d_model={args.d_model}, num_heads={args.num_heads}, use_global_node={use_global_node}, use_gcn={use_gcn}, seed={args.seed}")

    model.train()
    batch_vectors = max(1, int(round(n_pad ** 0.5)))  # SAI loss: batch size = sqrt(view_size) for stochastic trace estimation
    print_interval = 100
    t_start = time.perf_counter()

    def _sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    TIMING_STEP = 200  # 1-based: we time the 200th step (step index 199)
    for step in range(args.steps):
        do_timing = args.print_timing and (step == TIMING_STEP - 1)

        if do_timing:
            _sync()
            t0 = time.perf_counter()
        optimizer.zero_grad()
        if do_timing:
            _sync()
            t_zero = time.perf_counter() - t0
            t0 = time.perf_counter()
        diag_blocks, off_diag_list = model(
            x_input, edge_index=edge_index, edge_values=edge_values,
            scale_A=scale_A, precomputed_masks=precomputed_masks
        )
        if do_timing:
            _sync()
            t_forward = time.perf_counter() - t0
            t0 = time.perf_counter()
        # SAI loss: E_z || M A z - z ||^2 (padded dofs: z[n:]=0, so AZ[n:]=0)
        Z = torch.randn(1, n_pad, batch_vectors, device=device, dtype=x_input.dtype)
        Z[:, n:, :] = 0.0
        if do_timing:
            _sync()
            t_sample_z = time.perf_counter() - t0
            t0 = time.perf_counter()
        AZ = (A_dense @ Z.squeeze(0)).unsqueeze(0)  # (1, n_pad, batch_vectors)
        if do_timing:
            _sync()
            t_az = time.perf_counter() - t0
            t0 = time.perf_counter()
        MAZ = apply_block_structured_M(diag_blocks, off_diag_list, AZ, model.off_diag_struct, leaf_size=LEAF_SIZE)
        if do_timing:
            _sync()
            t_apply_m = time.perf_counter() - t0
            t0 = time.perf_counter()
        residual = MAZ - Z
        loss = (residual ** 2).mean()
        if do_timing:
            _sync()
            t_loss = time.perf_counter() - t0
            t0 = time.perf_counter()
        loss.backward()
        if do_timing:
            _sync()
            t_backward = time.perf_counter() - t0
            t0 = time.perf_counter()
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
            print(f"\n--- Step {TIMING_STEP} detailed timing (ms) ---")
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
            print(f"Step {step}: SAI loss (E||MAz-z||²) {loss.item():.6f}  (last {steps_in_period} steps: {elapsed:.3f}s)")
            save_leaf_only_weights(model, args.save, input_dim=9)
            t_start = time.perf_counter()

    save_leaf_only_weights(model, args.save, input_dim=9)
    print(f"Saved to {args.save}")


if __name__ == "__main__":
    train_leaf_only()
