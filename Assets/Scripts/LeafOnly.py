"""
LeafOnly: train a single 32x32 leaf block (U@U.T output) with no HODLR/hierarchy.
Uses first 32 nodes only; for dialing in leaf architecture and comparing with full HGT_OL.
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

# --- Leaf block connectivity and attention ---

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
        s = scale_A.item() if isinstance(scale_A, torch.Tensor) and scale_A.numel() == 1 else float(scale_A)
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
    """Masked attention within each leaf block of 32 nodes."""
    def __init__(self, dim, block_size, num_heads=4):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
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
    def __init__(self, dim, block_size, attn_module, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attn_module

    def forward(self, x, edge_index=None, edge_values=None, positions=None, scale_A=None, save_attention=False):
        x = x + self.attn(self.norm1(x), edge_index=edge_index, edge_values=edge_values, positions=positions, scale_A=scale_A, save_attention=save_attention)
        return x

def _most_recent_run_folder(base_path):
    base = Path(base_path)
    if not base.exists(): return base
    runs = sorted(base.glob("Run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs: return base
    return runs[0]

LEAF_SIZE = 32

# Set True to print forward-pass diagnostics (same format as NeuralPreconditioner)
class LeafCore(nn.Module):
    """
    Shared leaf path: embed -> enc_input_proj -> transformer blocks (leaf attention) -> leaf_head -> U@U.T + Jacobi diag.
    Supports any N divisible by leaf_size (used by both LeafOnlyNet and HGT_OL scale 0).
    Output shape: (B, num_leaves, leaf_size, leaf_size).
    """
    def __init__(self, input_dim=4, d_model=128, leaf_size=32, num_layers=3, num_heads=4):
        super().__init__()
        self.leaf_size = leaf_size
        self.embed = PhysicsAwareEmbedding(input_dim, d_model)
        self.enc_input_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, block_size=leaf_size, attn_module=LeafBlockAttention(d_model, leaf_size, num_heads=num_heads))
            for _ in range(num_layers)
        ])
        self.leaf_head = nn.Linear(d_model, leaf_size)
        nn.init.normal_(self.leaf_head.weight, std=0.01)
        nn.init.constant_(self.leaf_head.bias, 0.0)
        self.log_hodlr_scale_leaf = nn.Parameter(torch.ones(1) * math.log(1e-2))

    def get_leaf_blocks(self, h, x, scale_A=None):
        """From encoder output h (B, N, d_model) and input x (B, N, 4), compute dense leaf blocks (B, num_leaves, leaf_size, leaf_size)."""
        B, N, _ = h.shape
        num_leaves = N // self.leaf_size
        h_leaves = h.view(B, num_leaves, self.leaf_size, -1)
        u_leaf = self.leaf_head(h_leaves)
        leaf_scale = torch.exp(self.log_hodlr_scale_leaf)
        leaf_base = torch.matmul(u_leaf, u_leaf.transpose(-1, -2)) * leaf_scale
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
        return leaf_base + torch.diag_embed(new_diag)

    def forward(self, x, edge_index=None, edge_values=None, scale_A=None, save_attention=False):
        """Full forward: embed -> enc_input_proj -> blocks -> get_leaf_blocks. x: (B, N, input_dim), N divisible by leaf_size."""
        B, N, _ = x.shape
        positions = x[0, :, :3] if x.dim() == 3 else x[:, :3]

        h = self.embed(x, edge_index, edge_values, scale_A)
        h = self.enc_input_proj(h)

        for i, block in enumerate(self.blocks):
            h = block(h, edge_index=edge_index, edge_values=edge_values, positions=positions, scale_A=scale_A, save_attention=save_attention)

        dense_blocks = self.get_leaf_blocks(h, x, scale_A)
        return dense_blocks


class LeafOnlyNet(nn.Module):
    """
    Single leaf (32 nodes): thin wrapper around LeafCore with assert N==leaf_size.
    Output shape: (B, 1, 32, 32).
    """
    def __init__(self, input_dim=4, d_model=128, leaf_size=32, num_layers=3, num_heads=4):
        super().__init__()
        self.leaf_size = leaf_size
        self.core = LeafCore(input_dim=input_dim, d_model=d_model, leaf_size=leaf_size, num_layers=num_layers, num_heads=num_heads)

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

    def forward(self, x, edge_index=None, edge_values=None, scale_A=None, save_attention=False):
        B, N, _ = x.shape
        assert N == self.leaf_size, f"LeafOnly expects exactly {self.leaf_size} nodes, got {N}"
        return self.core(x, edge_index=edge_index, edge_values=edge_values, scale_A=scale_A, save_attention=save_attention)


def apply_leaf_only(leaf_blocks, x):
    """leaf_blocks (B, 1, 32, 32), x (B, 32, K). Returns (B, 32, K)."""
    B, N, K = x.shape
    x_leaves = x.view(B, 1, N, K)
    y_leaves = torch.matmul(leaf_blocks, x_leaves)
    return y_leaves.view(B, N, K)


# --- Save / Load (for InspectModel) ---

def read_leaf_only_header(path):
    path = Path(path)
    with open(path, 'rb') as f:
        header = f.read(16)
    if len(header) < 16:
        raise ValueError("LeafOnly weights file too short")
    # d_model, leaf_size, input_dim, num_layers
    d_model, leaf_size, input_dim, num_layers = struct.unpack('<iiii', header)
    return d_model, leaf_size, input_dim, num_layers


def save_leaf_only_weights(model, path, input_dim=4):
    path = Path(path)
    with open(path, 'wb') as f:
        f.write(struct.pack('<iiii', model.embed.lift[0].weight.shape[0], model.leaf_size, input_dim, len(model.blocks)))
        # Embed: lift, gcn, norm
        _write_packed_tensor(f, model.embed.lift[0].weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.lift[0].bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.lift[2].weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.lift[2].bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.gcn.linear_self.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.gcn.linear_self.bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.gcn.linear_neighbor.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.gcn.linear_neighbor.bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.gcn.update_gate[0].weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.gcn.update_gate[0].bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.gcn.update_gate[2].weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.gcn.update_gate[2].bias.detach().cpu().float(), transpose=False)
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


def load_leaf_only_weights(model, path):
    import numpy as np
    path = Path(path)
    with open(path, 'rb') as f:
        header = f.read(16)
        d_model, leaf_size, input_dim, num_layers = struct.unpack('<iiii', header)
    if model.leaf_size != leaf_size or len(model.blocks) != num_layers:
        raise ValueError(f"Checkpoint leaf_size={leaf_size} num_layers={num_layers} != model {model.leaf_size} {len(model.blocks)}")

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

    with open(path, 'rb') as f:
        f.seek(16)
        # Embed
        _read_into(f, model.embed.lift[0].weight, read_tensor, transpose=True)
        _read_into(f, model.embed.lift[0].bias, read_tensor, transpose=False)
        _read_into(f, model.embed.lift[2].weight, read_tensor, transpose=True)
        _read_into(f, model.embed.lift[2].bias, read_tensor, transpose=False)
        _read_into(f, model.embed.gcn.linear_self.weight, read_tensor, transpose=True)
        _read_into(f, model.embed.gcn.linear_self.bias, read_tensor, transpose=False)
        _read_into(f, model.embed.gcn.linear_neighbor.weight, read_tensor, transpose=True)
        _read_into(f, model.embed.gcn.linear_neighbor.bias, read_tensor, transpose=False)
        _read_into(f, model.embed.gcn.update_gate[0].weight, read_tensor, transpose=True)
        _read_into(f, model.embed.gcn.update_gate[0].bias, read_tensor, transpose=False)
        _read_into(f, model.embed.gcn.update_gate[2].weight, read_tensor, transpose=True)
        _read_into(f, model.embed.gcn.update_gate[2].bias, read_tensor, transpose=False)
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


def _read_into(f, param, read_fn, transpose=False):
    t = read_fn(f, param.shape, transpose=transpose).to(param.device)
    with torch.no_grad():
        param.copy_(t)


def train_leaf_only():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    default_data = script_dir.parent / "StreamingAssets" / "TestData"
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--data_folder', type=str, default=str(default_data))
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--frame', type=int, default=600)
    parser.add_argument('--save', type=str, default=str(script_dir / "leaf_only_weights.bytes"))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
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

    n = LEAF_SIZE
    x_full = batch['x']  # (num_nodes, 4)
    x_input = x_full[:n].unsqueeze(0).to(device)  # (1, 32, 4)
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
        A_dense = A_sparse.to_dense().to(device)
    else:
        A_dense = A_sparse.to(device).to_dense()

    torch.manual_seed(args.seed)
    model = LeafOnlyNet(input_dim=4, d_model=args.d_model, leaf_size=LEAF_SIZE, num_layers=args.num_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"Ready: {run_folder.name} frame_{frame_idx:04d}, first {n} nodes, {edge_index.shape[1]} edges, {args.num_layers} layers, d_model={args.d_model}, seed={args.seed}")

    model.train()
    batch_vectors = 32  # SAI loss: batch of 32 random vectors for stochastic trace estimation
    print_interval = 100
    t_start = time.perf_counter()
    for step in range(args.steps):
        optimizer.zero_grad()
        leaf_blocks = model(x_input, edge_index=edge_index, edge_values=edge_values, scale_A=scale_A)
        M_block = leaf_blocks[0, 0]  # (32, 32)
        # SAI loss: E_z || M A z - z ||^2 with batch of 32 vectors (all ops MPS-friendly, no SVD)
        Z = torch.randn(n, batch_vectors, device=device, dtype=x_input.dtype)
        AZ = A_dense @ Z  # (32, 32)
        MAZ = M_block @ AZ  # (32, 32)
        residual = MAZ - Z
        loss = (residual ** 2).mean()
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if step % print_interval == 0:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_start
            steps_in_period = step + 1 if step == 0 else print_interval
            print(f"Step {step}: SAI loss (E||MAz-z||²) {loss.item():.6f}  (last {steps_in_period} steps: {elapsed:.3f}s)")
            save_leaf_only_weights(model, args.save, input_dim=4)
            t_start = time.perf_counter()

    save_leaf_only_weights(model, args.save, input_dim=4)
    print(f"Saved to {args.save}")


if __name__ == "__main__":
    train_leaf_only()
