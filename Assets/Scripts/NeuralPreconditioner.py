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
from collections import defaultdict
from torch.utils.checkpoint import checkpoint

# --- 0. Dataset (same format as InspectModel: x, edge_index, edge_values, num_nodes) ---

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
    # Fallback: nodes.bin size
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
        diag_map = np.zeros(num_nodes, dtype=np.float32)
        for r, c, v in zip(rows, cols, vals):
            if r == c:
                diag_map[r] = v

        pos_scale = float(np.abs(positions).max())
        if pos_scale <= 0.0: pos_scale = 1.0
        positions_n = positions / pos_scale

        scale_A = float(np.abs(vals).max())
        if scale_A <= 0.0: scale_A = 1.0
        diag_map_n = diag_map / scale_A

        # Input: only per-node info (position 3 + diagonal 1), no neighbor slots
        n_float = 4
        x = np.zeros((num_nodes, n_float), dtype=np.float32)
        x[:, :3] = positions_n
        x[:, 3] = diag_map_n

        return {
            'x': torch.from_numpy(x).float(),
            'edge_index': torch.stack([torch.from_numpy(rows.astype(np.int64)), torch.from_numpy(cols.astype(np.int64))]),
            'edge_values': torch.from_numpy(vals.copy()),
            'num_nodes': int(num_nodes),
            'scale_A': scale_A,
        }

# --- 1. Hierarchical Transformer Components ---

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=8192):
        super().__init__()
        self.rope_dim = 2 * (dim // 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len is None: seq_len = x.shape[-2]
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    d = cos.shape[-1]
    q_rope = (q[..., :d] * cos) + (rotate_half(q[..., :d]) * sin)
    k_rope = (k[..., :d] * cos) + (rotate_half(k[..., :d]) * sin)
    q_embed = torch.cat([q_rope, q[..., d:]], dim=-1) if d < q.shape[-1] else q_rope
    k_embed = torch.cat([k_rope, k[..., d:]], dim=-1) if d < k.shape[-1] else k_rope
    return q_embed, k_embed

class HODLRBlockAttention(nn.Module):
    def __init__(self, dim, block_size, num_heads=4):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.num_heads = num_heads if dim % num_heads == 0 else 1
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, N, C = x.shape
        pad = 0
        if N % self.block_size != 0:
            pad = self.block_size - (N % self.block_size)
            x = F.pad(x, (0, 0, 0, pad))

        B, N_pad, _ = x.shape
        num_blocks = N_pad // self.block_size

        x_blk = x.view(B * num_blocks, self.block_size, C)
        qkv = self.qkv(x_blk).reshape(B * num_blocks, self.block_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rope(q, seq_len=self.block_size)
        q, k = apply_rope(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x_out = (attn @ v).transpose(1, 2).reshape(B * num_blocks, self.block_size, C)
        x_out = self.proj(x_out)
        x_out = x_out.view(B, N_pad, C)
        if pad > 0:
            x_out = x_out[:, :N, :]
        return x_out


def build_merged_connectivity(edge_index, edge_values, positions, level, device, dtype=torch.float32, scale_A=None):
    """
    At `level` (0 = original), merge edges: (r, c) -> (r // 2^level, c // 2^level).
    Additively blend edge values for same (i, j). Positions = centroid of original positions per super-node.
    Returns: merged_edge_index (2, E'), merged_edge_values (E'), merged_positions (N', 3).
    """
    if level == 0:
        return edge_index.to(device), edge_values.to(device), positions.to(device)
    rows, cols = edge_index[0], edge_index[1]
    scale = 2 ** level
    i = (rows // scale).long()
    j = (cols // scale).long()
    w = edge_values.to(dtype=dtype)
    if scale_A is not None and scale_A != 1.0:
        s = scale_A.item() if isinstance(scale_A, torch.Tensor) and scale_A.numel() == 1 else float(scale_A)
        w = w / s
    agg = defaultdict(lambda: 0.0)
    for a, b, v in zip(i.tolist(), j.tolist(), w.tolist()):
        agg[(a, b)] += v
    if not agg:
        n_merged = max(positions.shape[0] // scale, 1)
        pos_flat = positions.to(device=device, dtype=dtype)
        merged_positions = torch.zeros(n_merged, 3, device=device, dtype=dtype)
        for idx in range(n_merged):
            start, end = idx * scale, min((idx + 1) * scale, positions.shape[0])
            if start < end:
                merged_positions[idx] = pos_flat[start:end].mean(dim=0)
        return (
            torch.zeros(2, 0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=dtype, device=device),
            merged_positions,
        )
    pairs = list(agg.keys())
    merged_i = torch.tensor([p[0] for p in pairs], device=device, dtype=torch.long)
    merged_j = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.long)
    merged_w = torch.tensor([agg[p] for p in pairs], device=device, dtype=dtype)
    merged_edge_index = torch.stack([merged_i, merged_j], dim=0)
    n_merged = positions.shape[0] // scale
    pos_flat = positions.to(device=device, dtype=dtype)
    merged_positions = torch.zeros(n_merged, 3, device=device, dtype=dtype)
    for idx in range(n_merged):
        start, end = idx * scale, min((idx + 1) * scale, positions.shape[0])
        if start < end:
            merged_positions[idx] = pos_flat[start:end].mean(dim=0)
    return merged_edge_index, merged_w, merged_positions


def build_leaf_block_connectivity(edge_index, edge_values, positions, scale_A, leaf_size, device, dtype=torch.float32):
    """
    Build per-block attention mask and edge features for leaf blocks.
    - positions: (N, 3) normalized; edge_values normalized by scale_A at call site.
    - Returns: attn_mask (num_blocks, 32, 33), edge_feats (num_blocks, 32, 33, 4).
      attn_mask[b,i,j]=1 if node i in block b can attend to j (j in 0..31 = node, 32 = block node).
      edge_feats[b,i,j,:] = (dx, dy, dz, w_norm) for edges; 0 elsewhere. Self and block node get weight 1 in the MLP step.
    """
    N = positions.shape[0]
    num_blocks = N // leaf_size
    if num_blocks == 0:
        return None, None
    rows, cols = edge_index[0], edge_index[1]
    # Only in-block edges: same block for r and c
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
        s = scale_A.item() if isinstance(scale_A, torch.Tensor) and scale_A.numel() == 1 else float(scale_A)
        w = w / s
    edge_feats_flat = torch.cat([dx, w.unsqueeze(1)], dim=1)

    attn_mask = torch.zeros(num_blocks, leaf_size, leaf_size + 1, device=device, dtype=dtype)
    edge_feats = torch.zeros(num_blocks, leaf_size, leaf_size + 1, 4, device=device, dtype=dtype)
    # Self-attend
    attn_mask[:, torch.arange(leaf_size, device=device), torch.arange(leaf_size, device=device)] = 1.0
    edge_feats[:, torch.arange(leaf_size, device=device), torch.arange(leaf_size, device=device), :] = 0.0  # MLP will not be used; we set weight=1
    # Block node: all 32 attend to column 32
    attn_mask[:, :, leaf_size] = 1.0
    # In-block edges
    attn_mask[b_l, r_l, c_l] = 1.0
    edge_feats[b_l, r_l, c_l, :] = edge_feats_flat.to(device)
    return attn_mask, edge_feats


class LeafBlockAttention(nn.Module):
    """
    Masked attention within each leaf block of 32 nodes.

    FIXES APPLIED:
    1. Decoupled Heads: Scores keep head dimension (bnqkh) so heads learn distinct patterns;
       softmax over keys (dim=3). Previously einsum collapsed heads.
    2. Linear Edge Bypass: edge_gate projects normalized A_ij to per-head signed weights,
       added AFTER softmax. Allows negative weights and magnitudes > 1 for inverse/stencil-like ops.
    """
    def __init__(self, dim, block_size, num_heads=4):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.num_heads = num_heads if dim % num_heads == 0 else 1
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        # Learnable projection for Linear Edge Bypass: scalar edge weight -> per-head signed weight
        self.edge_gate = nn.Linear(1, self.num_heads)
        nn.init.normal_(self.edge_gate.weight, std=0.01)
        nn.init.zeros_(self.edge_gate.bias)
        self.last_attn_self = 0.0
        self.last_attn_neighbor = 0.0
        self.last_attn_block = 0.0
        self.last_attn_matrix = None
        self.last_scores_matrix = None
        self.last_bias_physics_matrix = None

    def forward(self, x, edge_index=None, edge_values=None, positions=None, scale_A=None, save_attention=False):
        B, N, C = x.shape
        if edge_index is None or positions is None:
            return self._forward_dense_fallback(x, B, N, C, save_attention)

        device = x.device
        dtype = x.dtype
        pad = 0
        if N % self.block_size != 0:
            pad = self.block_size - (N % self.block_size)
            x = F.pad(x, (0, 0, 0, pad))
        N_pad = x.shape[1]
        num_blocks = N_pad // self.block_size

        attn_mask, edge_feats = build_leaf_block_connectivity(
            edge_index, edge_values, positions, scale_A, self.block_size, device, dtype
        )
        if attn_mask is None:
            return x[:, :N, :] if pad > 0 else x

        x_blk = x.view(B, num_blocks, self.block_size, C)
        block_node = x_blk.mean(dim=2, keepdim=True)
        kv = torch.cat([x_blk, block_node], dim=2)
        qkv_q = self.qkv(x_blk)
        qkv_kv = self.qkv(kv)
        q = qkv_q[..., :C]
        k = qkv_kv[..., C:2*C]
        v = qkv_kv[..., 2*C:3*C]
        q = q.view(B, num_blocks, self.block_size, self.num_heads, self.head_dim)
        k = k.view(B, num_blocks, 33, self.num_heads, self.head_dim)
        v = v.view(B, num_blocks, 33, self.num_heads, self.head_dim)

        # Scores with head dimension preserved: (B, n, 32, 33, H)
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

        # Linear Edge Bypass: signed, unbounded per-head weights from edge values
        linear_edge_weights = self.edge_gate(bias_physics.unsqueeze(-1))  # (num_blocks, 32, 33, H)
        linear_edge_weights = linear_edge_weights.unsqueeze(0).masked_fill(mask_expanded == 0, 0.0)

        combined_weights = attn_probs + linear_edge_weights

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
    def __init__(self, dim, block_size, heads=4, mlp_ratio=4.0, attn_module=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attn_module if attn_module is not None else HODLRBlockAttention(dim, block_size, heads)

    def forward(self, x, edge_index=None, edge_values=None, positions=None, scale_A=None, save_attention=False):
        x = x + self.attn(self.norm1(x), edge_index=edge_index, edge_values=edge_values, positions=positions, scale_A=scale_A, save_attention=save_attention)
        return x

class SparsePhysicsGCN(nn.Module):
    """
    Lightweight message passing layer: neighbor features scaled by PDE matrix values (edge_values).
    Uses index_add_ for aggregation; no PyTorch Geometric dependency.
    """
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
        w = edge_values.clone()
        if scale_A is not None and scale_A != 1.0:
            s = scale_A.item() if isinstance(scale_A, torch.Tensor) and scale_A.numel() == 1 else float(scale_A)
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
    """Lift 4D input to d_model and run one physics-weighted message pass (GCN) with the graph."""
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.lift = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.gcn = SparsePhysicsGCN(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, edge_index=None, edge_values=None, scale_A=None):
        h = self.lift(x)
        if edge_index is not None and edge_values is not None:
            h = self.gcn(h, edge_index, edge_values, scale_A)
        return self.norm(h)


class HODLRHead(nn.Module):
    """Generates U and V directly from the pooled token."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj_u = nn.Linear(in_dim, out_dim)
        self.proj_v = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj_u(x), self.proj_v(x)

def _print_hodlr_architecture(leaf_size, depth, ranks, num_layers_per_scale=1):
    lines = [
        "HODLR architecture (Down-Only pass):",
        f"  Leaf:           dense {leaf_size}x{leaf_size} diagonal blocks",
        f"  Layers/scale:   {num_layers_per_scale} (residual attention blocks)",
    ]
    for j in range(depth):
        level_idx_from_bottom = depth - 1 - j
        rank_j = ranks[level_idx_from_bottom]
        lines.append(f"  Scale {j}: Length N/{2**j} -> {num_layers_per_scale} layers -> Level {level_idx_from_bottom} factors (Rank {rank_j})")
    print("\n" + "\n".join(lines) + "\n")

# --- 3. The Main Architecture (Down-Only) ---

class HGT_OL(nn.Module):
    """
    HGT-OL (Down-Only): 
    - Full-resolution embedding extracts leaf dense blocks.
    - Top-Down path: At each level (N/2^k), extracts HODLR factors for that resolution, 
      then downsamples tokens 2->1.
    - No upsampling, no skip connections. 
    - Full `d_model` is maintained down the pipeline to preserve neighbor info.
    """
    def __init__(self,
                 input_dim=1,
                 d_model=128,
                 depth=4,
                 leaf_size=32,
                 max_rank=128,
                 rank_scale=2.0,
                 num_layers_per_scale=3):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.leaf_size = leaf_size
        self.max_rank = max_rank
        self.rank_scale = rank_scale
        self.num_layers_per_scale = num_layers_per_scale

        # 1. Embedding: lift + physics GCN (graph at full resolution)
        self.embed = PhysicsAwareEmbedding(input_dim, d_model)
        self.enc_input_proj = nn.Linear(d_model, d_model)

        # HODLR Ranks (Ordered Coarse -> Fine)
        self.ranks = []
        for i in range(depth):
            level_idx_from_bottom = depth - 1 - i
            block_real_size = leaf_size * (2 ** level_idx_from_bottom)
            r = min(max_rank, int(round(rank_scale * (block_real_size ** (2.0 / 3.0)))))
            r = r if r % 2 == 0 else r + 1
            self.ranks.append(r)

        # 2. Down-Path: per scale, level GCN then multiple residual-attention layers; leaf/HODLR/downsample after last layer
        self.enc_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.hodlr_heads = nn.ModuleList()
        self.level_gcns = nn.ModuleList()

        for j in range(depth):
            layers = nn.ModuleList([
                TransformerBlock(d_model, block_size=leaf_size, attn_module=LeafBlockAttention(d_model, leaf_size, num_heads=4))
                for _ in range(num_layers_per_scale)
            ])
            self.enc_blocks.append(layers)
            self.down_samples.append(nn.Linear(d_model * 2, d_model))
            self.level_gcns.append(SparsePhysicsGCN(d_model))
            rank_j = self.ranks[depth - 1 - j]
            expansion = 2 ** j
            self.hodlr_heads.append(HODLRHead(d_model, rank_j * expansion))

        # 3. Leaf Generation (d_model -> leaf block; applied at j=0 after last layer)
        self.leaf_head = nn.Linear(d_model, leaf_size)
        nn.init.normal_(self.leaf_head.weight, std=0.01)
        nn.init.constant_(self.leaf_head.bias, 0.0)
        
        self.log_hodlr_scale_leaf = nn.Parameter(torch.ones(1) * math.log(1e-2))
        self.log_hodlr_scales = nn.Parameter(torch.ones(depth) * math.log(1e-2))

        for head in self.hodlr_heads:
            nn.init.normal_(head.proj_u.weight, std=0.01)
            nn.init.constant_(head.proj_u.bias, 0.0)
            nn.init.normal_(head.proj_v.weight, std=0.01)
            nn.init.constant_(head.proj_v.bias, 0.0)

        _print_hodlr_architecture(leaf_size, depth, self.ranks, num_layers_per_scale)

    def forward(self, x, pos=None, scale_A=None, edge_index=None, edge_values=None, save_attention=False, _timing=None):
        B, N, _ = x.shape
        if _timing is not None: t0 = time.time()
        positions = x[..., :3].squeeze(0) if x.dim() == 3 else x[..., :3]

        h = self.embed(x, edge_index, edge_values, scale_A)
        h = self.enc_input_proj(h)
        if _timing is not None: _timing["embed+pos"] = time.time() - t0

        # --- 2. Down-Only HODLR Hierarchy ---
        factors_fine_to_coarse = []
        dense_blocks_out = None  # Initialize before the loop

        # Precompute per-level connectivity for masked attention and level GCNs
        graph_per_level = None
        if edge_index is not None and edge_values is not None:
            device = x.device
            dtype = h.dtype
            graph_per_level = []
            for level in range(self.depth):
                ei_j, ev_j, pos_j = build_merged_connectivity(edge_index, edge_values, positions, level, device, dtype, scale_A=scale_A)
                graph_per_level.append((ei_j, ev_j, pos_j))

        for j in range(self.depth):
            if _timing is not None: t0 = time.time()

            # Hierarchical physics: GCN at this scale (j=0 already done in embed)
            if graph_per_level is not None and j > 0:
                ei_j, ev_j, pos_j = graph_per_level[j]
                h = self.level_gcns[j](h, ei_j, ev_j, scale_A)

            # Multiple residual-attention layers at this scale; leaf/HODLR/downsample only after last
            nlay = self.num_layers_per_scale
            for layer_idx in range(nlay):
                block = self.enc_blocks[j][layer_idx]
                save_attn = save_attention and (layer_idx == nlay - 1)
                if graph_per_level is not None:
                    ei_j, ev_j, pos_j = graph_per_level[j]
                    if self.training:
                        # checkpoint does not forward kwargs in reentrant mode; pass args positionally
                        h = checkpoint(
                            block, h, ei_j, ev_j, pos_j, scale_A, save_attn, use_reentrant=True
                        )
                    else:
                        h = block(h, edge_index=ei_j, edge_values=ev_j, positions=pos_j, scale_A=scale_A, save_attention=save_attn)
                else:
                    if self.training:
                        h = checkpoint(block, h, None, None, None, None, save_attn, use_reentrant=True)
                    else:
                        h = block(h, save_attention=save_attn)

            # --- After last layer at this scale: extract leaf (j==0), HODLR factors, then downsample ---
            if j == 0:
                if _timing is not None: t_leaf = time.time()
                num_leaves = N // self.leaf_size
                h_leaves = h.view(B, num_leaves, self.leaf_size, -1)
                u_leaf = self.leaf_head(h_leaves)

                leaf_base = torch.matmul(u_leaf, u_leaf.transpose(-1, -2)) * torch.exp(self.log_hodlr_scale_leaf)

                diag_A_n = x[..., 3]
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
                dense_blocks_out = leaf_base + torch.diag_embed(new_diag)
                if _timing is not None: _timing["leaf_head"] = time.time() - t_leaf
            # ----------------------------------------------------------

            # Extract factors for this scale
            u_expanded, v_expanded = self.hodlr_heads[j](h)
            rank_j = self.ranks[self.depth - 1 - j]
            
            # View mathematically unfolds the (R * 2^j) dimension out over the sequence 
            # turning shape (B, N/2^j, R*2^j) into the perfect (B, N, R) needed for HODLR
            u = u_expanded.view(B, N, rank_j)
            v = v_expanded.view(B, N, rank_j)
            factors_fine_to_coarse.append((u, v))

            # Downsample for the next, coarser scale
            B_h, n_curr, c_curr = h.shape
            h_reshaped = h.view(B_h, n_curr // 2, 2 * c_curr)
            h = self.down_samples[j](h_reshaped)
            
            if _timing is not None: _timing[f"enc_{j}"] = time.time() - t0

        # HODLR requires ordered from Coarse -> Fine. Reverse the extracted list!
        factors_levels = factors_fine_to_coarse[::-1]

        return dense_blocks_out, factors_levels

# --- 4. Fast HODLR MatMul ---

def apply_neural_hodlr(leaf_blocks, factors_levels, x, leaf_size=32, off_diag_scale=None):
    B, N, K = x.shape

    num_leaves = N // leaf_size
    x_leaves = x.view(B, num_leaves, leaf_size, K)
    y_leaves = torch.matmul(leaf_blocks, x_leaves)
    y = y_leaves.view(B, N, K)

    for i, factor in enumerate(factors_levels):
        if isinstance(factor, tuple):
            u_full, v_full = factor[0], factor[1]
        else:
            u_full = v_full = factor
            
        scale_i = off_diag_scale[i] if off_diag_scale is not None else 1.0
        B, n_tokens, rank = u_full.shape
        num_splits = 2 ** (i + 1)
        block_size_node = N // num_splits
        if block_size_node < 1: continue
        
        num_pairs = num_splits // 2
        u_view = u_full.view(B, num_pairs, 2, block_size_node, rank)
        v_view = v_full.view(B, num_pairs, 2, block_size_node, rank)
        u_L, u_R = u_view[:, :, 0], u_view[:, :, 1]
        v_L, v_R = v_view[:, :, 0], v_view[:, :, 1]

        x_view = x.view(B, num_pairs, 2, block_size_node, K)
        x_L, x_R = x_view[:, :, 0], x_view[:, :, 1]

        vr_xr = torch.matmul(v_R.transpose(-2, -1), x_R)
        y_L_update = torch.matmul(u_L, vr_xr)
        vl_xl = torch.matmul(v_L.transpose(-2, -1), x_L)
        y_R_update = torch.matmul(u_R, vl_xl)
        
        updates = torch.cat([y_L_update, y_R_update], dim=2)
        y = y + scale_i * updates.view(B, N, K)
        
    return y

# --- 4b. Save/Load HGT_OL ---

def read_weights_header(path):
    path = Path(path)
    with open(path, 'rb') as f:
        header = f.read(32)
    if len(header) < 32:
        _, _, d_model, nhead, depth, input_dim, max_rank = struct.unpack('<ffiiiii', header)
        num_layers_per_scale = 1
    else:
        _, _, d_model, nhead, depth, input_dim, max_rank, num_layers_per_scale = struct.unpack('<ffiiiiii', header)
    return d_model, nhead, depth, input_dim, max_rank, num_layers_per_scale

def _write_packed_tensor(f, param, transpose=False):
    t = param.detach().cpu().float()
    if transpose and t.dim() == 2:
        t = t.t()
    arr = t.numpy().astype(np.float16)
    n = arr.size
    pad = (1 if n % 2 else 0)
    f.write(arr.tobytes())
    if pad:
        f.write(np.zeros(1, dtype=np.float16).tobytes())

def save_weights_to_bytes(model, path, input_dim=None):
    path = Path(path)
    d_model = model.d_model
    depth = model.depth
    max_rank = model.max_rank
    num_layers_per_scale = getattr(model, 'num_layers_per_scale', 1)
    if input_dim is None: input_dim = 4  
    nhead = 4
    
    with open(path, 'wb') as f:
        f.write(struct.pack('<ffiiiiii', 0.0, 0.0, d_model, nhead, depth, input_dim, max_rank, num_layers_per_scale))
        # 1. Embed (PhysicsAwareEmbedding: lift + gcn + norm)
        _write_packed_tensor(f, model.embed.lift[0].weight, transpose=True)
        _write_packed_tensor(f, model.embed.lift[0].bias, transpose=False)
        _write_packed_tensor(f, model.embed.lift[2].weight, transpose=True)
        _write_packed_tensor(f, model.embed.lift[2].bias, transpose=False)
        _write_packed_tensor(f, model.embed.gcn.linear_self.weight, transpose=True)
        _write_packed_tensor(f, model.embed.gcn.linear_self.bias, transpose=False)
        _write_packed_tensor(f, model.embed.gcn.linear_neighbor.weight, transpose=True)
        _write_packed_tensor(f, model.embed.gcn.linear_neighbor.bias, transpose=False)
        _write_packed_tensor(f, model.embed.gcn.update_gate[0].weight, transpose=True)
        _write_packed_tensor(f, model.embed.gcn.update_gate[0].bias, transpose=False)
        _write_packed_tensor(f, model.embed.gcn.update_gate[2].weight, transpose=True)
        _write_packed_tensor(f, model.embed.gcn.update_gate[2].bias, transpose=False)
        _write_packed_tensor(f, model.embed.norm.weight, transpose=False)
        _write_packed_tensor(f, model.embed.norm.bias, transpose=False)
        
        # 2. Encoder input proj 
        _write_packed_tensor(f, model.enc_input_proj.weight, transpose=True)
        _write_packed_tensor(f, model.enc_input_proj.bias, transpose=False)
        
        # 3. Down Path: per scale, level_gcn then num_layers_per_scale blocks then down + head
        for i in range(model.depth):
            gcn = model.level_gcns[i]
            _write_packed_tensor(f, gcn.linear_self.weight, transpose=True)
            _write_packed_tensor(f, gcn.linear_self.bias, transpose=False)
            _write_packed_tensor(f, gcn.linear_neighbor.weight, transpose=True)
            _write_packed_tensor(f, gcn.linear_neighbor.bias, transpose=False)
            _write_packed_tensor(f, gcn.update_gate[0].weight, transpose=True)
            _write_packed_tensor(f, gcn.update_gate[0].bias, transpose=False)
            _write_packed_tensor(f, gcn.update_gate[2].weight, transpose=True)
            _write_packed_tensor(f, gcn.update_gate[2].bias, transpose=False)
            for layer_idx in range(num_layers_per_scale):
                block = model.enc_blocks[i][layer_idx]
                _write_packed_tensor(f, block.norm1.weight, False)
                _write_packed_tensor(f, block.norm1.bias, False)
                _write_packed_tensor(f, block.attn.qkv.weight, True)
                _write_packed_tensor(f, block.attn.qkv.bias, False)
                _write_packed_tensor(f, block.attn.proj.weight, True)
                _write_packed_tensor(f, block.attn.proj.bias, False)
                _write_packed_tensor(f, block.attn.edge_gate.weight, True)
                _write_packed_tensor(f, block.attn.edge_gate.bias, False)
            
            down = model.down_samples[i]
            _write_packed_tensor(f, down.weight, True)
            _write_packed_tensor(f, down.bias, False)
            
            head = model.hodlr_heads[i]
            _write_packed_tensor(f, head.proj_u.weight, True)
            _write_packed_tensor(f, head.proj_u.bias, False)
            _write_packed_tensor(f, head.proj_v.weight, True)
            _write_packed_tensor(f, head.proj_v.bias, False)
            
        # 4. Leaf (leaf_head: d_model -> leaf_size)
        _write_packed_tensor(f, model.leaf_head.weight, True)
        _write_packed_tensor(f, model.leaf_head.bias, False)
        _write_packed_tensor(f, torch.exp(model.log_hodlr_scale_leaf).detach(), False)
        _write_packed_tensor(f, torch.exp(model.log_hodlr_scales).detach(), False)

def _read_packed_tensor(f, target_param, transpose=False):
    num_elements = target_param.numel()
    read_len = num_elements + (1 if num_elements % 2 else 0)
    bytes_to_read = read_len * 2
    buffer = f.read(bytes_to_read)
    if len(buffer) != bytes_to_read:
        raise ValueError(f"Unexpected EOF: wanted {bytes_to_read} bytes, got {len(buffer)}")
    packed = np.frombuffer(buffer, dtype=np.uint32)
    data_fp16 = packed.view(np.float16)
    if num_elements % 2 != 0:
        data_fp16 = data_fp16[:-1]
    data_fp32 = torch.from_numpy(data_fp16.astype(np.float32)).to(target_param.device)
    if transpose and data_fp32.dim() == 2:
        reshaped = data_fp32.view(target_param.shape[1], target_param.shape[0]).t()
    else:
        reshaped = data_fp32.view(target_param.shape)
    with torch.no_grad():
        target_param.copy_(reshaped)

def load_hgt_ol_weights_from_bytes(model, path):
    path = Path(path)
    with open(path, 'rb') as f:
        header = f.read(32)
        if len(header) < 32:
            # Old 28-byte format: one layer per scale
            _, _, d_model, nhead, depth, input_dim, max_rank = struct.unpack('<ffiiiii', header)
            num_layers_per_scale = 1
        else:
            _, _, d_model, nhead, depth, input_dim, max_rank, num_layers_per_scale = struct.unpack('<ffiiiiii', header)
        model_nlay = getattr(model, 'num_layers_per_scale', 1)
        if model_nlay < num_layers_per_scale:
            raise ValueError(
                f"Checkpoint has num_layers_per_scale={num_layers_per_scale}, model has {model_nlay}; "
                "model must have >= checkpoint layers per scale."
            )
        
        # 1. Embed (PhysicsAwareEmbedding)
        _read_packed_tensor(f, model.embed.lift[0].weight, transpose=True)
        _read_packed_tensor(f, model.embed.lift[0].bias, False)
        _read_packed_tensor(f, model.embed.lift[2].weight, transpose=True)
        _read_packed_tensor(f, model.embed.lift[2].bias, False)
        _read_packed_tensor(f, model.embed.gcn.linear_self.weight, transpose=True)
        _read_packed_tensor(f, model.embed.gcn.linear_self.bias, False)
        _read_packed_tensor(f, model.embed.gcn.linear_neighbor.weight, transpose=True)
        _read_packed_tensor(f, model.embed.gcn.linear_neighbor.bias, False)
        _read_packed_tensor(f, model.embed.gcn.update_gate[0].weight, transpose=True)
        _read_packed_tensor(f, model.embed.gcn.update_gate[0].bias, False)
        _read_packed_tensor(f, model.embed.gcn.update_gate[2].weight, transpose=True)
        _read_packed_tensor(f, model.embed.gcn.update_gate[2].bias, False)
        _read_packed_tensor(f, model.embed.norm.weight, False)
        _read_packed_tensor(f, model.embed.norm.bias, False)
        
        # 2. Enc Input Proj
        _read_packed_tensor(f, model.enc_input_proj.weight, transpose=True)
        _read_packed_tensor(f, model.enc_input_proj.bias, False)
        
        # 3. Down Path: level_gcn then num_layers_per_scale blocks per scale, then down + head
        for i in range(model.depth):
            gcn = model.level_gcns[i]
            _read_packed_tensor(f, gcn.linear_self.weight, transpose=True)
            _read_packed_tensor(f, gcn.linear_self.bias, False)
            _read_packed_tensor(f, gcn.linear_neighbor.weight, transpose=True)
            _read_packed_tensor(f, gcn.linear_neighbor.bias, False)
            _read_packed_tensor(f, gcn.update_gate[0].weight, transpose=True)
            _read_packed_tensor(f, gcn.update_gate[0].bias, False)
            _read_packed_tensor(f, gcn.update_gate[2].weight, transpose=True)
            _read_packed_tensor(f, gcn.update_gate[2].bias, False)
            for layer_idx in range(num_layers_per_scale):
                block = model.enc_blocks[i][layer_idx]
                _read_packed_tensor(f, block.norm1.weight, False)
                _read_packed_tensor(f, block.norm1.bias, False)
                _read_packed_tensor(f, block.attn.qkv.weight, True)
                _read_packed_tensor(f, block.attn.qkv.bias, False)
                _read_packed_tensor(f, block.attn.proj.weight, True)
                _read_packed_tensor(f, block.attn.proj.bias, False)
                _read_packed_tensor(f, block.attn.edge_gate.weight, True)
                _read_packed_tensor(f, block.attn.edge_gate.bias, False)
            
            down = model.down_samples[i]
            _read_packed_tensor(f, down.weight, True)
            _read_packed_tensor(f, down.bias, False)
            
            head = model.hodlr_heads[i]
            _read_packed_tensor(f, head.proj_u.weight, True)
            _read_packed_tensor(f, head.proj_u.bias, False)
            _read_packed_tensor(f, head.proj_v.weight, True)
            _read_packed_tensor(f, head.proj_v.bias, False)

        # 4. Leaf (leaf_head: d_model -> leaf_size)
        _read_packed_tensor(f, model.leaf_head.weight, True)
        _read_packed_tensor(f, model.leaf_head.bias, False)
        
        _hodlr_leaf = torch.empty(1)
        _read_packed_tensor(f, _hodlr_leaf, False)
        with torch.no_grad():
            model.log_hodlr_scale_leaf.copy_(torch.log(_hodlr_leaf.clamp(min=1e-10)).to(model.log_hodlr_scale_leaf.device))
            
        depth = model.depth
        level_bytes = 2 * (depth + (1 if depth % 2 else 0))
        remainder = f.read(level_bytes)
        if len(remainder) >= 2 * depth:
            level_scales = np.frombuffer(remainder[: 2 * depth], dtype=np.float16).astype(np.float32)
            with torch.no_grad():
                model.log_hodlr_scales.copy_(torch.log(torch.from_numpy(level_scales).clamp(min=1e-10).to(model.log_hodlr_scales.device)))
        else:
            with torch.no_grad():
                model.log_hodlr_scales.copy_(torch.ones(depth, device=model.log_hodlr_scales.device) * model.log_hodlr_scale_leaf.item())
    print(f"Loaded HGT_OL weights from {path}")

# --- 5. Training Loop ---

def _pad_to_hodlr_size(n_real, leaf_size=32):
    num_blocks_min = (n_real + leaf_size - 1) // leaf_size
    depth = max(1, int(math.ceil(math.log2(num_blocks_min))))
    return leaf_size * (2 ** depth)

def _most_recent_run_folder(base_path):
    base = Path(base_path)
    if not base.exists(): return base
    runs = sorted(base.glob("Run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs: return base
    return runs[0]

def print_diagnostics(model, leaf_blocks, factors, step, optimizer=None):
    print(f"\n--- Diagnostics Step {step} ---")

    # --- 1. Leaf attention: self / neighbors / block node (from last layer at scale 0) ---
    if hasattr(model, 'enc_blocks') and len(model.enc_blocks) > 0:
        attn_mod = model.enc_blocks[0][-1].attn
        if hasattr(attn_mod, 'last_attn_self'):
            s, n, b = attn_mod.last_attn_self, attn_mod.last_attn_neighbor, attn_mod.last_attn_block
            total = s + n + b
            if total > 0:
                print(f"  Attention (leaf) | Self: {s:.4f} | Neighbors: {n:.4f} | Block node: {b:.4f}  (sum={total:.4f})")
            else:
                print(f"  Attention (leaf) | Self: {s:.4f} | Neighbors: {n:.4f} | Block node: {b:.4f}")

    if hasattr(model, 'log_hodlr_scale_leaf'):
        leaf_s = torch.exp(model.log_hodlr_scale_leaf).item()
        print(f"  Leaf off-diag scale: {leaf_s:.6f}")
    if hasattr(model, 'log_hodlr_scales'):
        scales = torch.exp(model.log_hodlr_scales).cpu().numpy()
        print(f"  HODLR level scales: {[f'{s:.6f}' for s in scales]}")

    leaf_diag = torch.diagonal(leaf_blocks, dim1=-2, dim2=-1)
    print(f"  Leaf Blocks (Diag): Mean={leaf_diag.mean().item():.4f}, Min={leaf_diag.min().item():.4f}, Max={leaf_diag.max().item():.4f}")

    leaf_off = leaf_blocks - torch.diag_embed(leaf_diag)
    print(f"  Leaf Blocks (Off):  MeanAbs={leaf_off.abs().mean().item():.4f}, Max={leaf_off.abs().max().item():.4f}")

    print("  HODLR Levels (Coarse -> Fine):")
    for i, factor_tuple in enumerate(factors):
        if isinstance(factor_tuple, tuple):
            U, V = factor_tuple[0], factor_tuple[1]
        else:
            U = V = factor_tuple
        u_norm = U.norm(dim=-1).mean().item()
        u_max = U.abs().max().item()
        v_norm = V.norm(dim=-1).mean().item()
        v_max = V.abs().max().item()
        print(f"    Lvl {i}: U MeanNorm={u_norm:.5f} MaxVal={u_max:.5f}  V MeanNorm={v_norm:.5f} MaxVal={v_max:.5f}")
    print("----------------------------\n")

def train_hgt_ol():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    default_data = script_dir.parent / "StreamingAssets" / "TestData"
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--data_folder', type=str, default=str(default_data))
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--num_probes', type=int, default=32)
    parser.add_argument('--leaf_size', type=int, default=32)
    parser.add_argument('--frame', type=int, default=600)
    parser.add_argument('--rank_scale', type=float, default=2.0)
    parser.add_argument('--max_rank', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--viz_limit', type=int, default=150)
    parser.add_argument('--save_attention', action='store_true', default=True, help='Save leaf attention weights every diagnostic step to debug_attention.pt for InspectModel --debug_attention')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')
    print(f"Using device: {device}")

    t0 = time.time()
    data_path = Path(args.data_folder)
    if not data_path.exists():
        raise SystemExit(f"Data folder not found: {data_path}")
    run_folder = _most_recent_run_folder(data_path)
    if run_folder != data_path:
        print(f"  [startup] Using most recent run: {run_folder.name}")
    dataset = FluidGraphDataset([run_folder])
    if len(dataset) == 0:
        raise SystemExit(f"No frames found under {run_folder}")
    print(f"  [startup] FluidGraphDataset: {time.time()-t0:.2f}s ({len(dataset)} frames in {run_folder.name})")

    t0 = time.time()
    leaf_size = args.leaf_size
    frame_idx = min(args.frame, len(dataset) - 1) if dataset else 0
    sample = dataset[frame_idx]
    num_nodes_full = sample['num_nodes']
    n_real = min(num_nodes_full, args.viz_limit) if args.viz_limit > 0 else num_nodes_full
    N_pad = _pad_to_hodlr_size(n_real, leaf_size)
    depth = int(round(math.log2(N_pad // leaf_size)))
    input_dim = sample['x'].shape[1]
    d_model = args.d_model
    if args.viz_limit > 0 and n_real < num_nodes_full:
        print(f"  [startup] viz_limit={args.viz_limit}: training on first {n_real} nodes (full frame has {num_nodes_full})")
    print(f"  [startup] load frame {frame_idx} + compute N_pad/depth: {time.time()-t0:.2f}s")

    t0 = time.time()
    model = HGT_OL(
        input_dim=input_dim, d_model=d_model, depth=depth, leaf_size=leaf_size,
        max_rank=args.max_rank, rank_scale=args.rank_scale,
    )
    print(f"  [startup] HGT_OL() build: {time.time()-t0:.2f}s")

    t0 = time.time()
    model = model.to(device)
    print(f"  [startup] model.to(device): {time.time()-t0:.2f}s")

    print(f"Ready: {run_folder.name} frame_{frame_idx:04d}, num_nodes={n_real}, N_pad={N_pad}, depth={depth}, d_model={d_model}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    batch = dataset[frame_idx]
    x_input = batch['x'].unsqueeze(0)[:, :n_real, :].to(device)
    pad_len = N_pad - n_real
    if pad_len > 0:
        x_input = F.pad(x_input, (0, 0, 0, pad_len), value=0.0)
        
    rows, cols = batch['edge_index'][0], batch['edge_index'][1]
    mask = (rows < n_real) & (cols < n_real)
    edge_index_viz = batch['edge_index'][:, mask]
    edge_values_viz = batch['edge_values'][mask]
    mm_device = torch.device('cpu') if device.type == 'mps' else device
    A_sparse = torch.sparse_coo_tensor(
        edge_index_viz.to(mm_device),
        edge_values_viz.to(mm_device),
        (n_real, n_real),
    ).coalesce()
    N_cur = N_pad

    model.train()
    step_start = time.time()

    for step in range(args.steps):
        t0 = time.time()
        optimizer.zero_grad()

        scale_A = batch.get('scale_A')
        if scale_A is not None and not isinstance(scale_A, torch.Tensor):
            scale_A = torch.tensor(scale_A, device=device, dtype=x_input.dtype)
            
        edge_index_dev = edge_index_viz.to(device)
        edge_values_dev = edge_values_viz.to(device)
        leaf_blocks, factors = model(x_input, scale_A=scale_A, edge_index=edge_index_dev, edge_values=edge_values_dev)

        num_probes = args.num_probes
        z = torch.randn(1, n_real, num_probes, device=device)
        z_mm = z.squeeze(0).to(mm_device)
        y_flat = torch.sparse.mm(A_sparse, z_mm)
        y = y_flat.unsqueeze(0).to(device)
        
        if N_cur > n_real:
            y = F.pad(y, (0, 0, 0, N_cur - n_real), value=0.0)

        z_hat = apply_neural_hodlr(leaf_blocks, factors, y, leaf_size=leaf_size, off_diag_scale=torch.exp(model.log_hodlr_scales))
        z_hat_real = z_hat[:, :n_real, :]
        
        loss = F.mse_loss(z_hat_real, z)
        loss.backward()

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
        optimizer.step()

        if step % 100 == 0:
            t_interval_start = time.time()
            print(f"Step {step}: Loss {loss.item():.6f} ({time.time() - step_start:.1f}s elapsed for last 100 steps)")
            model.eval()
            with torch.no_grad():
                lb_debug, fact_debug = model(x_input, scale_A=scale_A, edge_index=edge_index_dev, edge_values=edge_values_dev, save_attention=args.save_attention)
                print_diagnostics(model, lb_debug, fact_debug, step, optimizer)
            if args.save_attention:
                blocks_data = []
                for j in range(model.depth):
                    attn_mod = model.enc_blocks[j][-1].attn
                    if not hasattr(attn_mod, 'last_attn_matrix'):
                        continue
                    m = attn_mod.last_attn_matrix
                    if m is None:
                        continue
                    attn_blocks = m[0] if m.dim() == 4 else m
                    attn_np = attn_blocks.numpy() if isinstance(attn_blocks, torch.Tensor) else attn_blocks
                    scores_np = None
                    bias_np = None
                    if hasattr(attn_mod, 'last_scores_matrix') and attn_mod.last_scores_matrix is not None:
                        s = attn_mod.last_scores_matrix
                        scores_blocks = s[0] if s.dim() == 4 else s
                        scores_np = scores_blocks.numpy() if isinstance(scores_blocks, torch.Tensor) else scores_blocks
                    if hasattr(attn_mod, 'last_bias_physics_matrix') and attn_mod.last_bias_physics_matrix is not None:
                        b = attn_mod.last_bias_physics_matrix
                        bias_np = b.numpy() if isinstance(b, torch.Tensor) else b
                    blocks_data.append({'attn': attn_np, 'scores': scores_np, 'bias_physics': bias_np})
                if blocks_data:
                    torch.save({
                        'blocks': blocks_data,
                        'depth': model.depth,
                        'n_real': n_real,
                        'viz_limit': args.viz_limit,
                        'step': step,
                        'leaf_size': model.leaf_size,
                    }, script_dir / "debug_attention.pt")
            model.train()
            
            out_path = script_dir / "model_weights.bytes"
            save_weights_to_bytes(model, out_path, input_dim=input_dim)
            step_start = time.time()

    print("Training complete.")

if __name__ == "__main__":
    train_hgt_ol()