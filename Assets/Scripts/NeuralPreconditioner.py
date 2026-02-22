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

# Per-node input: position (3), diag (1), then for each off-diagonal slot: (neighbor_index/N, neighbor_pos 3, weight 1) so the NN sees matrix structure.
MAX_OFFDIAG_SLOTS = 24  # match sim neighbor slots; pad if degree < this

class FluidGraphDataset:
    """
    Loads fluid graph frames: nodes (position), edge_index, A values.
    Returns 'x' with: position (3), diag (1), then MAX_OFFDIAG_SLOTS × (index, pos_j 3, weight) so the model sees off-diagonal structure.
    Nodes are assumed already in Morton (Z-order). Also returns 'edge_index', 'edge_values', 'num_nodes'.
    """
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
        if rows.shape[0] != vals.shape[0] or cols.shape[0] != vals.shape[0]:
            raise ValueError(f"Mismatch lengths at {frame_path}")

        positions = np.asarray(raw_nodes['position'], dtype=np.float32)
        diag_map = np.zeros(num_nodes, dtype=np.float32)
        # Group off-diagonals by row: row i -> list of (col, weight)
        row_edges = defaultdict(list)
        for r, c, v in zip(rows, cols, vals):
            if r == c:
                diag_map[r] = v
            else:
                row_edges[r].append((c, v))

        # Normalize so inputs are mostly in [-1, 1] for scale-invariant generalization
        pos_scale = float(np.abs(positions).max())
        if pos_scale <= 0.0:
            pos_scale = 1.0
        positions_n = positions / pos_scale

        scale_A = float(np.abs(vals).max())
        if scale_A <= 0.0:
            scale_A = 1.0
        diag_map_n = diag_map / scale_A

        # Build per-node features: [pos(3), diag(1), slot0(5), slot1(5), ...] with slot = (j/N, pos_j[0], pos_j[1], pos_j[2], A_ij)
        n_float = 3 + 1 + MAX_OFFDIAG_SLOTS * 5
        x = np.zeros((num_nodes, n_float), dtype=np.float32)
        x[:, :3] = positions_n
        x[:, 3] = diag_map_n
        for i in range(num_nodes):
            slots = row_edges.get(i, [])
            for k, (j, v) in enumerate(slots):
                if k >= MAX_OFFDIAG_SLOTS:
                    break
                base = 4 + k * 5
                # Compute relative index and apply a sign-preserving log scale (int() avoids uint32 overflow)
                delta_idx = float(int(j) - int(i))
                x[i, base + 0] = math.copysign(math.log1p(abs(delta_idx)), delta_idx)

                x[i, base + 1:base + 4] = positions_n[j]
                x[i, base + 4] = v / scale_A

        return {
            'x': torch.from_numpy(x).float(),
            'edge_index': torch.stack([torch.from_numpy(rows.astype(np.int64)), torch.from_numpy(cols.astype(np.int64))]),
            'edge_values': torch.from_numpy(vals.copy()),
            'num_nodes': int(num_nodes),
            'scale_A': scale_A,
        }

# --- 1. Hierarchical Transformer Components ---
# Inputs are in Morton order. Attention is restricted to HODLR blocks (see below).

class RotaryEmbedding(nn.Module):
    """RoPE for head_dim; uses even rope_dim = 2*(dim//2) so odd head_dims (e.g. 5) work."""
    def __init__(self, dim, max_seq_len=8192):
        super().__init__()
        self.rope_dim = 2 * (dim // 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            # Shape: (1, 1, seq_len, rope_dim) to broadcast with (B*num_blocks, heads, seq_len, head_dim)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # RoPE may use only first rope_dim dims when head_dim is odd
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

class TransformerBlock(nn.Module):
    def __init__(self, dim, block_size, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = HODLRBlockAttention(dim, block_size, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SymmetricNeighborEmbed(nn.Module):
    """ Encodes center node and sum-pools neighbor geometry, weights, and sequence index. """
    def __init__(self, embed_dim, max_offdiag_slots=24):
        super().__init__()
        self.max_offdiag_slots = max_offdiag_slots

        # Center feature: [pos_x, pos_y, pos_z, diag]
        self.center_proj = nn.Linear(4, embed_dim)

        # Neighbor feature: [rel_idx, delta_x, delta_y, delta_z, weight]
        self.neighbor_proj = nn.Linear(5, embed_dim)

    def forward(self, x):
        # x shape: (B, N, 124)
        pos_i = x[..., 0:3]
        diag_i = x[..., 3:4]

        # 1. Project center node
        center_feat = torch.cat([pos_i, diag_i], dim=-1)
        h_center = self.center_proj(center_feat)

        # 2. Process and pool neighbors
        h_neighbors = torch.zeros_like(h_center)

        for k in range(self.max_offdiag_slots):
            base = 4 + k * 5

            # Grab the log-scaled relative index, position, and weight
            rel_idx = x[..., base+0 : base+1]
            pos_j = x[..., base+1 : base+4]
            weight_ij = x[..., base+4 : base+5]

            # Identify active neighbors vs zero-padded slots
            is_active = (weight_ij != 0.0).float()

            # Relative position and masked index
            delta_pos = (pos_j - pos_i) * is_active
            rel_idx = rel_idx * is_active

            # Concatenate [rel_idx, delta_p, weight] and project
            neighbor_feat = torch.cat([rel_idx, delta_pos, weight_ij], dim=-1)
            e_ij = self.neighbor_proj(neighbor_feat)

            # Mask the output to ensure padded slots (and their biases) add exactly 0
            e_ij = e_ij * is_active

            # Symmetric Sum Pool
            h_neighbors = h_neighbors + e_ij

        return h_center + h_neighbors


class HODLRHead(nn.Module):
    """
    Generates U and V for a specific HODLR level. Separate proj_u and proj_v allow
    distinct row and column bases for off-diagonal blocks (e.g. A_12 vs A_21).
    """
    def __init__(self, in_dim, rank):
        super().__init__()
        self.rank = rank
        self.proj_u = nn.Linear(in_dim, rank)
        self.proj_v = nn.Linear(in_dim, rank)

    def forward(self, x):
        return self.proj_u(x), self.proj_v(x)


def _print_hodlr_architecture(leaf_size, depth, ranks, decoder_dims, bottleneck_dim, root_block_size):
    """Print HODLR architecture once, from leaves (finest) to bottleneck (coarsest)."""
    lines = [
        "HODLR architecture (leaves -> bottleneck):",
        f"  Leaf:           dense {leaf_size}x{leaf_size} diagonal blocks",
    ]
    for level_idx_from_bottom in range(depth):
        block_real_size = leaf_size * (2 ** level_idx_from_bottom)
        rank_i = ranks[depth - 1 - level_idx_from_bottom]
        dim_i = decoder_dims[depth - 1 - level_idx_from_bottom]
        num_attn_blocks = 2 ** (level_idx_from_bottom + 1)
        lines.append(
            f"  Level {level_idx_from_bottom}:  BlockSize {block_real_size:4d}, Rank {rank_i:3d}, Dim {dim_i:3d}, AttnBlocks {num_attn_blocks:3d} (each {leaf_size}x{leaf_size})"
        )
    lines.append(
        f"  Bottleneck:     BlockSize {root_block_size:4d}, Dim {bottleneck_dim:3d}, AttnBlocks 1 (each {leaf_size}x{leaf_size})"
    )
    print("\n" + "\n".join(lines) + "\n")


# --- 3. The Main Architecture ---

class HGT_OL(nn.Module):
    """
    HGT-OL: Up-down U-Net over the HODLR tree. Attention is restricted to HODLR blocks at every level.
    - Encoder: merge 2->1 (pairs of tokens). At each level we have N/2^k tokens; we do leaf_size x leaf_size
      attention within each of the (N/2^k)/leaf_size blocks (each block = one HODLR node at that level).
    - Bottleneck: one root block (leaf_size tokens).
    - Decoder: split 1->2, skip connections, then same block attention; per-level HODLR heads output U,V
      with rank following N^(2/3) schedule (coarse->fine, e.g. 20, 32, 52, 82, 128, 128 for scale 2 max_rank 128).
    - Leaf head: dense 32x32 blocks on the diagonal. So we never do global attention; only within-block.
    """
    def __init__(self,
                 input_dim=1,
                 d_model=64,
                 depth=4,
                 leaf_size=32,
                 max_rank=64,
                 rank_scale=0.5):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.leaf_size = leaf_size
        self.max_rank = max_rank
        self.rank_scale = rank_scale

        # 1. Embedding
        self.embed = SymmetricNeighborEmbed(d_model)

        # Block size = HODLR block at current resolution (leaf_size tokens per block at every level)
        enc_dec_block = leaf_size

        # Rank schedule: same factor for both encoder and decoder (coarse -> fine order)
        RANK_DIM_CONSTANT = 1
        ranks = []
        for i in range(depth):
            level_idx_from_bottom = depth - 1 - i
            block_real_size = leaf_size * (2 ** level_idx_from_bottom)
            r = min(max_rank, int(round(rank_scale * (block_real_size ** (2.0 / 3.0)))))
            r = r if r % 2 == 0 else r + 1
            ranks.append(r)
        # Decoder dims (coarse -> fine): dim_i = factor * rank_i
        decoder_dims = [RANK_DIM_CONSTANT * r for r in ranks]
        # Encoder dims (fine -> coarse): enc level j uses same rank as decoder level (depth-1-j), so same dim
        encoder_dims = [RANK_DIM_CONSTANT * ranks[depth - 1 - j] for j in range(depth)]

        # Project embed output (d_model) into rank-sized encoder dim at finest level
        self.enc_input_proj = nn.Linear(d_model, encoder_dims[0])

        # 2. Encoder (Bottom-Up): each level uses rank-sized dim (same factor as decoder)
        self.enc_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for j in range(depth):
            self.enc_blocks.append(TransformerBlock(encoder_dims[j], block_size=enc_dec_block))
            next_dim = encoder_dims[j + 1] if j + 1 < depth else encoder_dims[-1]
            self.down_samples.append(nn.Linear(encoder_dims[j] * 2, next_dim))
        bottleneck_dim = encoder_dims[-1]

        # Bottleneck = global/root layer: same rank-sized dim as coarsest encoder
        self.bottleneck = TransformerBlock(bottleneck_dim, block_size=enc_dec_block)
        root_block_size = leaf_size * (2 ** depth)

        # 3. Decoder (Top-Down): each level has dim = same factor * rank; skip dim matches (encoder_dims[depth-1-i] == decoder_dims[i])
        self.dec_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        self.skip_fusions = nn.ModuleList()
        self.hodlr_heads = nn.ModuleList()

        prev_dim = bottleneck_dim
        for i in range(depth):
            dim_i = decoder_dims[i]
            rank_i = ranks[i]
            level_idx_from_bottom = depth - 1 - i
            block_real_size = leaf_size * (2 ** level_idx_from_bottom)
            enc_skip_dim = encoder_dims[depth - 1 - i]  # same as dim_i when same factor

            self.up_samples.append(nn.Linear(prev_dim, dim_i * 2))
            self.skip_projs.append(nn.Linear(enc_skip_dim, dim_i))
            self.skip_fusions.append(nn.Linear(dim_i * 2, dim_i))
            self.dec_blocks.append(TransformerBlock(dim_i, block_size=enc_dec_block))
            prev_dim = dim_i

        # Attach all HODLR heads to the final, N-length, full-resolution feature map (leaf_dim)
        leaf_dim = decoder_dims[-1]
        self.hodlr_heads = nn.ModuleList([
            HODLRHead(leaf_dim, ranks[i]) for i in range(depth)
        ])

        # 4. Leaf: project decoder dim (leaf_dim) up to leaf_size, then full-rank U (leaf_size x leaf_size)
        self.leaf_proj = nn.Linear(leaf_dim, leaf_size)
        # default init (no override) so leaf_proj(h) has normal scale
        self.leaf_head = nn.Linear(leaf_size, leaf_size)
        nn.init.normal_(self.leaf_head.weight, std=0.01)
        nn.init.constant_(self.leaf_head.bias, 0.0)
        # Per-level log-scales so fine levels can grow independently (avoids fighting a single decaying scale)
        self.log_hodlr_scale_leaf = nn.Parameter(torch.ones(1) * math.log(1e-2))
        self.log_hodlr_scales = nn.Parameter(torch.ones(depth) * math.log(1e-2))

        # --- Same small init as HODLR heads (off-diagonal only; leaf init done above)
        for head in self.hodlr_heads:
            nn.init.normal_(head.proj_u.weight, std=0.01)
            nn.init.constant_(head.proj_u.bias, 0.0)
            nn.init.normal_(head.proj_v.weight, std=0.01)
            nn.init.constant_(head.proj_v.bias, 0.0)

        # Print architecture once: leaves -> Level 0 .. Level (depth-1) -> bottleneck
        _print_hodlr_architecture(leaf_size, depth, ranks, decoder_dims, bottleneck_dim, root_block_size)

    def forward(self, x, pos=None, scale_A=None, _timing=None):
        # Input x is already in Morton order; no reordering. scale_A: when x has normalized diag, use for Jacobi.
        B, N, _ = x.shape
        if _timing is not None:
            t0 = time.time()
        h = self.embed(x)
        h = self.enc_input_proj(h)
        if _timing is not None:
            _timing["embed+pos"] = time.time() - t0

        # 2. Encoder (checkpoint to save activation memory; only one frame so recompute is cheap)
        skips = []
        for i, (block, down) in enumerate(zip(self.enc_blocks, self.down_samples)):
            if _timing is not None:
                t0 = time.time()
            h = checkpoint(block, h, use_reentrant=True) if self.training else block(h)
            skips.append(h)
            B, n_curr, c_curr = h.shape
            h_reshaped = h.view(B, n_curr // 2, 2 * c_curr)
            h = down(h_reshaped)
            if _timing is not None:
                _timing[f"enc_{i}"] = time.time() - t0

        if _timing is not None:
            t0 = time.time()
        h = checkpoint(self.bottleneck, h, use_reentrant=True) if self.training else self.bottleneck(h)
        if _timing is not None:
            _timing["bottleneck"] = time.time() - t0

        # 3. Decoder (no factor generation here; h is built up to full N tokens)
        for i in range(len(self.dec_blocks)):
            if _timing is not None:
                t0 = time.time()
            B, n_coarse, _ = h.shape
            h = self.up_samples[i](h)
            h = h.view(B, n_coarse * 2, -1)
            skip = skips.pop()
            skip = self.skip_projs[i](skip)
            h = torch.cat([h, skip], dim=-1)
            h = self.skip_fusions[i](h)
            h = checkpoint(self.dec_blocks[i], h, use_reentrant=True) if self.training else self.dec_blocks[i](h)
            if _timing is not None:
                _timing[f"dec_{i}"] = time.time() - t0

        # Generate ALL factors from the final, N-length, full-resolution feature map h
        factors_levels = []
        for i in range(self.depth):
            u, v = self.hodlr_heads[i](h)
            factors_levels.append((u, v))

        # 4. Leaf: off-diagonal same as HODLR (normalized U@U.T * scale); diagonal from data (Jacobi 1/diag(A))
        if _timing is not None:
            t0 = time.time()
        B, N, _ = h.shape
        num_leaves = N // self.leaf_size
        h_leaves = h.view(B, num_leaves, self.leaf_size, -1)
        h_leaves = self.leaf_proj(h_leaves)
        u_leaf = self.leaf_head(h_leaves)  # (B, num_leaves, leaf_size, leaf_size)

        # Off-diagonal only: scale * U@U.T (no identity), then zero the diagonal of that
        dense_blocks_out = torch.matmul(u_leaf, u_leaf.transpose(-1, -2)) * torch.exp(self.log_hodlr_scale_leaf)
        leaf_diag_off = torch.diagonal(dense_blocks_out, dim1=-2, dim2=-1)
        dense_blocks_out = dense_blocks_out - torch.diag_embed(leaf_diag_off)

        # Diagonal: from data (Jacobi 1/diag(A)); use 1.0 where A has no diagonal entry
        diag_A_n = x[..., 3]
        if scale_A is not None:
            s = scale_A if isinstance(scale_A, torch.Tensor) else torch.tensor(scale_A, device=x.device, dtype=x.dtype)
            diag_A_real = diag_A_n * s
        else:
            diag_A_real = diag_A_n
        jacobi_diag = torch.zeros_like(diag_A_real)
        mask = diag_A_real.abs() > 1e-6
        jacobi_diag[mask] = 1.0 / diag_A_real[mask]
        jacobi_diag_blocks = jacobi_diag.view(B, num_leaves, self.leaf_size)
        mask_blocks = mask.view(B, num_leaves, self.leaf_size)
        new_diag = torch.where(mask_blocks, jacobi_diag_blocks, torch.ones_like(jacobi_diag_blocks))
        dense_blocks_out = dense_blocks_out + torch.diag_embed(new_diag)

        if _timing is not None:
            _timing["leaf_head"] = time.time() - t0
        return dense_blocks_out, factors_levels


# --- 4. Fast HODLR MatMul (The "Forward" Pass of the Operator) ---

def apply_neural_hodlr(leaf_blocks, factors_levels, x, leaf_size=32, off_diag_scale=None):
    """
    Applies the HODLR matrix defined by (leaf_blocks, factors) to vector(s) x.
    x: (B, N, 1) or (B, N, K) for K probe vectors (processed in parallel).
    off_diag_scale: optional tensor (depth,) for per-level scales, or scalar/(1,) for single scale (legacy).
    Returns same shape as x.
    """
    B, N, K = x.shape

    # 1. Leaf Level (Diagonal Blocks) — leaf scale is already baked into leaf_blocks
    num_leaves = N // leaf_size
    x_leaves = x.view(B, num_leaves, leaf_size, K)
    y_leaves = torch.matmul(leaf_blocks, x_leaves)
    y = y_leaves.view(B, N, K)

    # 2. Off-Diagonal Levels: per-level scale (u_full, v_full are natively (B, N, rank))
    for i, factor in enumerate(factors_levels):
        if isinstance(factor, tuple):
            u_full, v_full = factor[0], factor[1]
        else:
            u_full = v_full = factor
        if off_diag_scale is not None:
            scale_i = off_diag_scale[i] if off_diag_scale.numel() > 1 else off_diag_scale.squeeze()
        else:
            scale_i = 1.0
        B, n_tokens, rank = u_full.shape
        num_splits = 2 ** (i + 1)
        block_size_node = N // num_splits
        if block_size_node < 1:
            continue
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


# --- 4b. Save/Load HGT_OL to .bytes (28-byte header + float16 tensors) ---

def read_weights_header(path):
    """Read the 28-byte header only. Returns (d_model, nhead, depth, input_dim, max_rank)."""
    path = Path(path)
    with open(path, 'rb') as f:
        header = f.read(28)
    _, _, d_model, nhead, depth, input_dim, max_rank = struct.unpack('<ffiiiii', header)
    return d_model, nhead, depth, input_dim, max_rank


def _write_packed_tensor(f, param, transpose=False):
    """Write one tensor as float16; 2D weights transposed so loader can view(shape[1], shape[0]).t()."""
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
    """Save HGT_OL to model_weights.bytes: 28-byte header then float16 tensors in fixed order."""
    path = Path(path)
    d_model = model.d_model
    depth = model.depth
    max_rank = model.max_rank
    if input_dim is None:
        input_dim = 124  # SymmetricNeighborEmbed expects 124-dim input
    nhead = 4
    with open(path, 'wb') as f:
        f.write(struct.pack('<ffiiiii', 0.0, 0.0, d_model, nhead, depth, input_dim, max_rank))
        # 1. Embed
        _write_packed_tensor(f, model.embed.center_proj.weight, transpose=True)
        _write_packed_tensor(f, model.embed.center_proj.bias, transpose=False)
        _write_packed_tensor(f, model.embed.neighbor_proj.weight, transpose=True)
        _write_packed_tensor(f, model.embed.neighbor_proj.bias, transpose=False)
        # 2. Encoder input proj (d_model -> rank-sized finest)
        _write_packed_tensor(f, model.enc_input_proj.weight, transpose=True)
        _write_packed_tensor(f, model.enc_input_proj.bias, transpose=False)
        # 3. Encoder blocks + down_samples
        for block, down in zip(model.enc_blocks, model.down_samples):
            _write_packed_tensor(f, block.norm1.weight, False)
            _write_packed_tensor(f, block.norm1.bias, False)
            _write_packed_tensor(f, block.attn.qkv.weight, True)
            _write_packed_tensor(f, block.attn.qkv.bias, False)
            _write_packed_tensor(f, block.attn.proj.weight, True)
            _write_packed_tensor(f, block.attn.proj.bias, False)
            _write_packed_tensor(f, block.norm2.weight, False)
            _write_packed_tensor(f, block.norm2.bias, False)
            _write_packed_tensor(f, block.mlp[0].weight, True)
            _write_packed_tensor(f, block.mlp[0].bias, False)
            _write_packed_tensor(f, block.mlp[2].weight, True)
            _write_packed_tensor(f, block.mlp[2].bias, False)
            _write_packed_tensor(f, down.weight, True)
            _write_packed_tensor(f, down.bias, False)
        # 4. Bottleneck
        bn = model.bottleneck
        _write_packed_tensor(f, bn.norm1.weight, False)
        _write_packed_tensor(f, bn.norm1.bias, False)
        _write_packed_tensor(f, bn.attn.qkv.weight, True)
        _write_packed_tensor(f, bn.attn.qkv.bias, False)
        _write_packed_tensor(f, bn.attn.proj.weight, True)
        _write_packed_tensor(f, bn.attn.proj.bias, False)
        _write_packed_tensor(f, bn.norm2.weight, False)
        _write_packed_tensor(f, bn.norm2.bias, False)
        _write_packed_tensor(f, bn.mlp[0].weight, True)
        _write_packed_tensor(f, bn.mlp[0].bias, False)
        _write_packed_tensor(f, bn.mlp[2].weight, True)
        _write_packed_tensor(f, bn.mlp[2].bias, False)
        # 5. Decoder: up_samples, skip_projs, skip_fusions, dec_blocks, hodlr_heads
        for i in range(model.depth):
            _write_packed_tensor(f, model.up_samples[i].weight, True)
            _write_packed_tensor(f, model.up_samples[i].bias, False)
            _write_packed_tensor(f, model.skip_projs[i].weight, True)
            _write_packed_tensor(f, model.skip_projs[i].bias, False)
            _write_packed_tensor(f, model.skip_fusions[i].weight, True)
            _write_packed_tensor(f, model.skip_fusions[i].bias, False)
            blk = model.dec_blocks[i]
            _write_packed_tensor(f, blk.norm1.weight, False)
            _write_packed_tensor(f, blk.norm1.bias, False)
            _write_packed_tensor(f, blk.attn.qkv.weight, True)
            _write_packed_tensor(f, blk.attn.qkv.bias, False)
            _write_packed_tensor(f, blk.attn.proj.weight, True)
            _write_packed_tensor(f, blk.attn.proj.bias, False)
            _write_packed_tensor(f, blk.norm2.weight, False)
            _write_packed_tensor(f, blk.norm2.bias, False)
            _write_packed_tensor(f, blk.mlp[0].weight, True)
            _write_packed_tensor(f, blk.mlp[0].bias, False)
            _write_packed_tensor(f, blk.mlp[2].weight, True)
            _write_packed_tensor(f, blk.mlp[2].bias, False)
            _write_packed_tensor(f, model.hodlr_heads[i].proj_u.weight, True)
            _write_packed_tensor(f, model.hodlr_heads[i].proj_u.bias, False)
            _write_packed_tensor(f, model.hodlr_heads[i].proj_v.weight, True)
            _write_packed_tensor(f, model.hodlr_heads[i].proj_v.bias, False)
        # 6. Leaf: leaf_proj, then leaf_head + hodlr_scale
        _write_packed_tensor(f, model.leaf_proj.weight, True)
        _write_packed_tensor(f, model.leaf_proj.bias, False)
        _write_packed_tensor(f, model.leaf_head.weight, True)
        _write_packed_tensor(f, model.leaf_head.bias, False)
        _write_packed_tensor(f, torch.exp(model.log_hodlr_scale_leaf).detach(), False)
        _write_packed_tensor(f, torch.exp(model.log_hodlr_scales).detach(), False)
    # print(f"Saved to {path}")


def _read_packed_tensor(f, target_param, transpose=False):
    """Read one tensor as float16; 2D weights are stored transposed so we view(shape[1], shape[0]).t()."""
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
    """Load HGT_OL weights from .bytes (same order as save_weights_to_bytes)."""
    path = Path(path)
    with open(path, 'rb') as f:
        header = f.read(28)
        _, _, d_model, nhead, depth, input_dim, max_rank = struct.unpack('<ffiiiii', header)
        # 1. Embed
        _read_packed_tensor(f, model.embed.center_proj.weight, transpose=True)
        _read_packed_tensor(f, model.embed.center_proj.bias, False)
        _read_packed_tensor(f, model.embed.neighbor_proj.weight, transpose=True)
        _read_packed_tensor(f, model.embed.neighbor_proj.bias, False)
        # 2. Encoder input proj
        _read_packed_tensor(f, model.enc_input_proj.weight, transpose=True)
        _read_packed_tensor(f, model.enc_input_proj.bias, False)
        # 3. Encoder blocks + down_samples
        for block, down in zip(model.enc_blocks, model.down_samples):
            _read_packed_tensor(f, block.norm1.weight, False)
            _read_packed_tensor(f, block.norm1.bias, False)
            _read_packed_tensor(f, block.attn.qkv.weight, True)
            _read_packed_tensor(f, block.attn.qkv.bias, False)
            _read_packed_tensor(f, block.attn.proj.weight, True)
            _read_packed_tensor(f, block.attn.proj.bias, False)
            _read_packed_tensor(f, block.norm2.weight, False)
            _read_packed_tensor(f, block.norm2.bias, False)
            _read_packed_tensor(f, block.mlp[0].weight, True)
            _read_packed_tensor(f, block.mlp[0].bias, False)
            _read_packed_tensor(f, block.mlp[2].weight, True)
            _read_packed_tensor(f, block.mlp[2].bias, False)
            _read_packed_tensor(f, down.weight, True)
            _read_packed_tensor(f, down.bias, False)
        # 4. Bottleneck
        bn = model.bottleneck
        _read_packed_tensor(f, bn.norm1.weight, False)
        _read_packed_tensor(f, bn.norm1.bias, False)
        _read_packed_tensor(f, bn.attn.qkv.weight, True)
        _read_packed_tensor(f, bn.attn.qkv.bias, False)
        _read_packed_tensor(f, bn.attn.proj.weight, True)
        _read_packed_tensor(f, bn.attn.proj.bias, False)
        _read_packed_tensor(f, bn.norm2.weight, False)
        _read_packed_tensor(f, bn.norm2.bias, False)
        _read_packed_tensor(f, bn.mlp[0].weight, True)
        _read_packed_tensor(f, bn.mlp[0].bias, False)
        _read_packed_tensor(f, bn.mlp[2].weight, True)
        _read_packed_tensor(f, bn.mlp[2].bias, False)
        # 5. Decoder
        for i in range(model.depth):
            _read_packed_tensor(f, model.up_samples[i].weight, True)
            _read_packed_tensor(f, model.up_samples[i].bias, False)
            _read_packed_tensor(f, model.skip_projs[i].weight, True)
            _read_packed_tensor(f, model.skip_projs[i].bias, False)
            _read_packed_tensor(f, model.skip_fusions[i].weight, True)
            _read_packed_tensor(f, model.skip_fusions[i].bias, False)
            blk = model.dec_blocks[i]
            _read_packed_tensor(f, blk.norm1.weight, False)
            _read_packed_tensor(f, blk.norm1.bias, False)
            _read_packed_tensor(f, blk.attn.qkv.weight, True)
            _read_packed_tensor(f, blk.attn.qkv.bias, False)
            _read_packed_tensor(f, blk.attn.proj.weight, True)
            _read_packed_tensor(f, blk.attn.proj.bias, False)
            _read_packed_tensor(f, blk.norm2.weight, False)
            _read_packed_tensor(f, blk.norm2.bias, False)
            _read_packed_tensor(f, blk.mlp[0].weight, True)
            _read_packed_tensor(f, blk.mlp[0].bias, False)
            _read_packed_tensor(f, blk.mlp[2].weight, True)
            _read_packed_tensor(f, blk.mlp[2].bias, False)
            _read_packed_tensor(f, model.hodlr_heads[i].proj_u.weight, True)
            _read_packed_tensor(f, model.hodlr_heads[i].proj_u.bias, False)
            try:
                _read_packed_tensor(f, model.hodlr_heads[i].proj_v.weight, True)
                _read_packed_tensor(f, model.hodlr_heads[i].proj_v.bias, False)
            except (ValueError, OSError):
                with torch.no_grad():
                    model.hodlr_heads[i].proj_v.weight.copy_(model.hodlr_heads[i].proj_u.weight)
                    model.hodlr_heads[i].proj_v.bias.copy_(model.hodlr_heads[i].proj_u.bias)
                for j in range(i + 1, model.depth):
                    _read_packed_tensor(f, model.hodlr_heads[j].proj_u.weight, True)
                    _read_packed_tensor(f, model.hodlr_heads[j].proj_u.bias, False)
                    with torch.no_grad():
                        model.hodlr_heads[j].proj_v.weight.copy_(model.hodlr_heads[j].proj_u.weight)
                        model.hodlr_heads[j].proj_v.bias.copy_(model.hodlr_heads[j].proj_u.bias)
                break
        # 6. Leaf: leaf_proj, then leaf_head + hodlr_scale
        _read_packed_tensor(f, model.leaf_proj.weight, True)
        _read_packed_tensor(f, model.leaf_proj.bias, False)
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


def apply_hodlr_matrix(leaf_blocks, factors, x, leaf_size=32):
    """
    Applies HODLR matrix (leaf blocks + full-node U,V factors) to x.
    factors: list of (u, v) per level; each u, v shape (B, N, R) with N = padded size.
    Used by overfit baseline (DirectHODLRBlocks). For HGT_OL output use apply_neural_hodlr instead.
    """
    B, N, _ = x.shape
    num_leaves = N // leaf_size
    x_leaves = x.view(B, num_leaves, leaf_size, 1)
    y = torch.matmul(leaf_blocks, x_leaves).view(B, N, 1)
    for i, (u_full, v_full) in enumerate(factors):
        num_splits = 2 ** (i + 1)
        block_size_node = N // num_splits
        if block_size_node < 1:
            continue
        num_pairs = num_splits // 2
        u_view = u_full.view(B, num_pairs, 2, block_size_node, -1)
        v_view = v_full.view(B, num_pairs, 2, block_size_node, -1)
        x_view = x.view(B, num_pairs, 2, block_size_node, 1)
        x_L, x_R = x_view[:, :, 0], x_view[:, :, 1]
        u_L, u_R = u_view[:, :, 0], u_view[:, :, 1]
        v_L, v_R = v_view[:, :, 0], v_view[:, :, 1]
        vr_xr = torch.matmul(v_R.transpose(-2, -1), x_R)
        y_L_update = torch.matmul(u_L, vr_xr)
        vl_xl = torch.matmul(v_L.transpose(-2, -1), x_L)
        y_R_update = torch.matmul(u_R, vl_xl)
        y = y + torch.cat([y_L_update, y_R_update], dim=2).view(B, N, 1)
    return y


# --- 5. Training Loop ---

def _pad_to_hodlr_size(n_real, leaf_size=32):
    """Pad n_real to next power-of-2 multiple of leaf_size for HODLR hierarchy."""
    num_blocks_min = (n_real + leaf_size - 1) // leaf_size
    depth = max(1, int(math.ceil(math.log2(num_blocks_min))))
    return leaf_size * (2 ** depth)

def _most_recent_run_folder(base_path):
    """Return the most recent Run_* folder under base_path (by mtime). If none, return base_path."""
    base = Path(base_path)
    if not base.exists():
        return base
    runs = sorted(base.glob("Run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        return base
    return runs[0]


def print_diagnostics(model, leaf_blocks, factors, step):
    """
    Prints internal stats of the HODLR matrix components to debug scaling and physics.
    """
    print(f"\n--- Diagnostics Step {step} ---")

    # 1. Per-level scales (leaf + HODLR levels)
    if hasattr(model, 'log_hodlr_scale_leaf'):
        leaf_s = torch.exp(model.log_hodlr_scale_leaf).item()
        print(f"  Leaf off-diag scale: {leaf_s:.6f}")
    if hasattr(model, 'log_hodlr_scales'):
        scales = torch.exp(model.log_hodlr_scales).cpu().numpy()
        print(f"  HODLR level scales: {[f'{s:.6f}' for s in scales]}")

    # 2. Leaf Block Stats (The Diagonal)
    # leaf_blocks shape: (B, NumLeaves, LeafSize, LeafSize)
    leaf_diag = torch.diagonal(leaf_blocks, dim1=-2, dim2=-1)
    print(f"  Leaf Blocks (Diag): Mean={leaf_diag.mean().item():.4f}, Min={leaf_diag.min().item():.4f}, Max={leaf_diag.max().item():.4f}")

    # Check off-diagonal elements of leaf blocks (should be smaller than diagonal)
    leaf_off = leaf_blocks - torch.diag_embed(leaf_diag)
    print(f"  Leaf Blocks (Off):  MeanAbs={leaf_off.abs().mean().item():.4f}, Max={leaf_off.abs().max().item():.4f}")

    # 3. HODLR Factor Stats (U and V per level)
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
    # Known location: Assets/StreamingAssets/TestData (under repo root = script_dir.parent.parent for Assets)
    default_data = script_dir.parent / "StreamingAssets" / "TestData"
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--data_folder', type=str, default=str(default_data),
                        help="Path to TestData (e.g. .../StreamingAssets/TestData). Uses most recent Run_* subfolder.")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Clip gradient norm to avoid explosion (0 = no clip).')
    parser.add_argument('--num_probes', type=int, default=32, help='Number of random probe vectors per step (reduces gradient variance).')
    parser.add_argument('--leaf_size', type=int, default=32)
    parser.add_argument('--frame', type=int, default=600, help="Single frame index to use (e.g. 600, same as InspectModel).")
    parser.add_argument('--rank_scale', type=float, default=2.0, help="Rank = min(max_rank, scale * block^(2/3)).")
    parser.add_argument('--max_rank', type=int, default=128, help="Cap on HODLR rank per level.")
    parser.add_argument('--d_model', type=int, default=128, help="Base channel dim (64 = larger model, needs ~44GB+ GPU; 32 fits single-frame on 44GB).")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
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
        raise SystemExit(f"No frames found under {run_folder} (need nodes.bin, edge_index_*.bin, A_values.bin)")
    print(f"  [startup] FluidGraphDataset: {time.time()-t0:.2f}s ({len(dataset)} frames in {run_folder.name})")

    t0 = time.time()
    leaf_size = args.leaf_size
    frame_idx = min(args.frame, len(dataset) - 1) if dataset else 0
    # dataset[600] = frame_0600 (paths sorted: frame_0000, frame_0001, ...)
    sample = dataset[frame_idx]
    num_nodes_real = sample['num_nodes']
    N_pad = _pad_to_hodlr_size(num_nodes_real, leaf_size)
    depth = int(round(math.log2(N_pad // leaf_size)))
    input_dim = sample['x'].shape[1]
    d_model = args.d_model
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

    print(f"Data: single frame {frame_idx} only (~few MB). Memory is model+optimizer, not data.")
    print(f"Ready: {run_folder.name} frame_{frame_idx:04d}, num_nodes={num_nodes_real}, N_pad={N_pad}, depth={depth}, d_model={d_model}, num_probes={args.num_probes}, rank_scale={args.rank_scale}, max_rank={args.max_rank}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Single frame only: batch is identical every step, so load once and reuse.
    batch = dataset[frame_idx]
    n_real = batch['num_nodes']
    N_cur = _pad_to_hodlr_size(n_real, leaf_size)
    x_input = batch['x'].unsqueeze(0).to(device)
    pad_len = N_cur - n_real
    if pad_len > 0:
        x_input = F.pad(x_input, (0, 0, 0, pad_len), value=0.0)
    A_sparse = torch.sparse_coo_tensor(
        batch['edge_index'].to(device),
        batch['edge_values'].to(device),
        (n_real, n_real),
    ).coalesce()

    model.train()
    step_start = time.time()

    for step in range(args.steps):
        if step == 100:
            t_step_start = time.time()
            print("Step 100 (timing breakdown below; total step time at end):")
        t0 = time.time()
        optimizer.zero_grad()
        if step == 100:
            print(f"  [timing] zero_grad: {time.time()-t0:.3f}s")
            print(f"  [timing] get_batch: 0.000s (cached; n_real={n_real}, N_cur={N_cur})")
            t0 = time.time()

        t0 = time.time()
        timing_dict = {} if step == 100 else None
        scale_A = batch.get('scale_A')
        if scale_A is not None and not isinstance(scale_A, torch.Tensor):
            scale_A = torch.tensor(scale_A, device=device, dtype=x_input.dtype)
        leaf_blocks, factors = model(x_input, scale_A=scale_A, _timing=timing_dict)
        if step == 100:
            print(f"  [timing] model(x): {time.time()-t0:.3f}s (leaf_blocks {tuple(leaf_blocks.shape)}, {len(factors)} levels)")
            for k, v in sorted(timing_dict.items(), key=lambda x: x[0]):
                print(f"    model.{k}: {v:.3f}s")

        t0 = time.time()
        num_probes = args.num_probes
        z = torch.randn(1, n_real, num_probes, device=device)
        y_flat = torch.sparse.mm(A_sparse, z.squeeze(0))
        y = y_flat.unsqueeze(0)
        if step == 100:
            print(f"  [timing] z + A@z (sparse mm): {time.time()-t0:.3f}s (z {tuple(z.shape)})")

        t0 = time.time()
        if N_cur > n_real:
            y = F.pad(y, (0, 0, 0, N_cur - n_real), value=0.0)
        if step == 100:
            print(f"  [timing] pad y to N_cur: {time.time()-t0:.3f}s (y {tuple(y.shape)})")

        t0 = time.time()
        z_hat = apply_neural_hodlr(leaf_blocks, factors, y, leaf_size=leaf_size, off_diag_scale=torch.exp(model.log_hodlr_scales))
        if step == 100:
            print(f"  [timing] apply_neural_hodlr: {time.time()-t0:.3f}s")

        t0 = time.time()
        z_hat_real = z_hat[:, :n_real, :]
        loss = F.mse_loss(z_hat_real, z)
        if step == 100:
            print(f"  [timing] loss: {time.time()-t0:.3f}s")

        t0 = time.time()
        loss.backward()
        if step == 100:
            print(f"  [timing] loss.backward(): {time.time()-t0:.3f}s")

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        t0 = time.time()
        optimizer.step()
        if step == 100:
            print(f"  [timing] optimizer.step(): {time.time()-t0:.3f}s")
            print(f"  [timing] >>> step 100 total (train only): {time.time() - t_step_start:.3f}s")

        if step % 100 == 0:
            t_interval_start = time.time()
            print(f"Step {step}: Loss {loss.item():.6f} ({time.time() - step_start:.1f}s elapsed for last {step + 1 if step == 0 else 100} steps)")
            t0 = time.time()
            model.eval()
            with torch.no_grad():
                lb_debug, fact_debug = model(x_input, scale_A=scale_A, _timing=None)
                print_diagnostics(model, lb_debug, fact_debug, step)
            model.train()
            t_diag = time.time() - t0
            t0 = time.time()
            out_path = script_dir / "model_weights.bytes"
            save_weights_to_bytes(model, out_path, input_dim=input_dim)
            t_save = time.time() - t0
            if step == 100:
                print(f"  [timing] diagnostics (eval forward + print): {t_diag:.3f}s")
                print(f"  [timing] save_weights_to_bytes: {t_save:.3f}s")
                print(f"  [timing] >>> step 100 checkpoint total (diagnostics + save): {time.time() - t_interval_start:.3f}s")
            step_start = time.time()

    print("Training complete.")

if __name__ == "__main__":
    train_hgt_ol()