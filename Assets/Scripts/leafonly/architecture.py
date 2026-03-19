import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    ATTENTION_HOPS,
    GLOBAL_FEATURES_DIM,
    LEAF_SIZE,
)
from .data import build_leaf_block_connectivity


class SparsePhysicsGCN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear_self = nn.Linear(d_model, d_model)
        self.linear_neighbor = nn.Linear(d_model, d_model)
        self.update_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x, edge_index, edge_values):
        B, N, C = x.shape
        x_flat = x.squeeze(0) if B == 1 else x.view(B * N, C)

        row, col = edge_index[0], edge_index[1]
        w = edge_values.clone().to(x.dtype)

        neighbor_features = self.linear_neighbor(x_flat)
        messages = neighbor_features[col] * w.unsqueeze(-1)
        aggr = torch.zeros_like(x_flat)
        aggr.index_add_(0, row, messages)

        self_features = self.linear_self(x_flat)
        out = self.update_gate(torch.cat([self_features, aggr], dim=-1))
        out = x_flat + out

        return out.unsqueeze(0) if B == 1 else out.view(B, N, C)


class PhysicsAwareEmbedding(nn.Module):
    DEFAULT_NUM_GCN_LAYERS = 2
    LIFT_FEATURES = 6

    def __init__(self, input_dim, d_model, use_gcn=True, global_features_dim=0, num_gcn_layers=DEFAULT_NUM_GCN_LAYERS):
        super().__init__()
        self.use_gcn = use_gcn
        self.global_features_dim = global_features_dim
        lift_in = self.LIFT_FEATURES + (global_features_dim or 0)
        self.lift = nn.Sequential(
            nn.Linear(lift_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        n_gcn = max(0, int(num_gcn_layers))
        if not use_gcn:
            n_gcn = 0
        self.gcn = nn.ModuleList([SparsePhysicsGCN(d_model) for _ in range(n_gcn)]) if n_gcn > 0 else None
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, edge_index=None, edge_values=None, global_features=None):
        node_feats = x[..., 3:]
        if self.global_features_dim > 0:
            if global_features is None:
                raise ValueError("global_features is required for PhysicsAwareEmbedding.")
            gf = global_features.unsqueeze(0) if global_features.dim() == 1 else global_features
            gf = gf.unsqueeze(1).expand(-1, node_feats.size(1), -1)
            node_feats = torch.cat([node_feats, gf], dim=-1)
        h = self.lift(node_feats)
        if self.use_gcn and self.gcn is not None and edge_index is not None and edge_values is not None:
            for gcn_layer in self.gcn:
                h = gcn_layer(h, edge_index, edge_values)
        h = self.norm(h)
        return h


class LeafBlockAttention(nn.Module):
    VALID_LAYOUTS = ("32x32", "32x33", "32x34")

    def __init__(self, dim, block_size, num_heads=2, attention_layout="32x33"):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        if attention_layout not in self.VALID_LAYOUTS:
            raise ValueError(f"Unsupported attention_layout='{attention_layout}'. Expected one of {self.VALID_LAYOUTS}.")
        self.attention_layout = attention_layout
        self.num_heads = num_heads if dim % num_heads == 0 else 1
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.edge_gate = nn.Linear(4, self.num_heads)
        nn.init.normal_(self.edge_gate.weight, std=0.01)
        nn.init.zeros_(self.edge_gate.bias)
        self.last_attn_self = 0.0
        self.last_attn_neighbor = 0.0
        self.last_attn_block = 0.0
        self.last_attn_matrix = None
        self.last_scores_matrix = None
        self.last_bias_physics_matrix = None

    def forward(self, x, edge_index=None, edge_values=None, positions=None, save_attention=False, attn_mask=None, edge_feats=None):
        B, N, C = x.shape
        if edge_index is None or edge_values is None or positions is None:
            raise ValueError("LeafBlockAttention requires edge_index, edge_values, and positions.")

        device = x.device
        dtype = x.dtype
        node_pad = 0
        if N % self.block_size != 0:
            node_pad = self.block_size - (N % self.block_size)
            x = F.pad(x, (0, 0, 0, node_pad))
        N_pad = x.shape[1]
        num_blocks = N_pad // self.block_size
        x_blk = x.view(B, num_blocks, self.block_size, C)

        if attn_mask is None or edge_feats is None:
            attn_mask, edge_feats = build_leaf_block_connectivity(
                edge_index, edge_values, positions, self.block_size, device, dtype
            )
        if attn_mask is None:
            return x[:, :N, :] if node_pad > 0 else x

        use_block_node = self.attention_layout in ("32x33", "32x34")
        use_matrix_node = self.attention_layout == "32x34"
        kv_parts = [x_blk]
        if use_block_node:
            block_node = x_blk.mean(dim=2, keepdim=True)
            kv_parts.append(block_node)
        if use_matrix_node:
            matrix_node = x.mean(dim=1, keepdim=True).unsqueeze(1).expand(-1, num_blocks, -1, -1)
            kv_parts.append(matrix_node)
        kv = torch.cat(kv_parts, dim=2)
        qkv_q = self.qkv(x_blk)
        qkv_kv = self.qkv(kv)
        q = qkv_q[..., :C]
        k = qkv_kv[..., C:2 * C]
        v = qkv_kv[..., 2 * C:3 * C]
        key_count = kv.shape[2]
        q = q.view(B, num_blocks, self.block_size, self.num_heads, self.head_dim)
        k = k.view(B, num_blocks, key_count, self.num_heads, self.head_dim)
        v = v.view(B, num_blocks, key_count, self.num_heads, self.head_dim)

        scores = torch.einsum("bnqhd,bnkhd->bnqkh", q, k) * self.scale

        if edge_feats.dim() == 4:
            bias_physics = edge_feats[..., :4].unsqueeze(0).expand(B, -1, -1, -1, -1).clone()
        else:
            bias_physics = edge_feats[..., :4].clone()
        if bias_physics.shape[3] != key_count:
            if bias_physics.shape[3] > key_count:
                bias_physics = bias_physics[:, :, :, :key_count, :]
            else:
                key_pad = key_count - bias_physics.shape[3]
                pad_tensor = torch.zeros(*bias_physics.shape[:3], key_pad, bias_physics.shape[4], device=bias_physics.device, dtype=bias_physics.dtype)
                bias_physics = torch.cat([bias_physics, pad_tensor], dim=3)
        special_start = self.block_size
        if use_block_node:
            bias_physics[:, :, :, special_start, :] = 0.0
            bias_physics[:, :, :, special_start, 3] = 1.0
            special_start += 1
        if use_matrix_node:
            bias_physics[:, :, :, special_start, :] = 0.0
            bias_physics[:, :, :, special_start, 3] = 1.0
        arange = torch.arange(self.block_size, device=device)
        bias_physics[:, :, arange, arange, :] = 0.0
        bias_physics[:, :, arange, arange, 3] = 1.0
        scores = scores + bias_physics[..., 3:4]
        if attn_mask.dim() == 3:
            mask_base = attn_mask.unsqueeze(0)
        else:
            mask_base = attn_mask
        if mask_base.shape[3] != key_count:
            if mask_base.shape[3] > key_count:
                mask_base = mask_base[:, :, :, :key_count]
            else:
                key_pad = key_count - mask_base.shape[3]
                pad_tensor = torch.ones(*mask_base.shape[:3], key_pad, device=mask_base.device, dtype=mask_base.dtype)
                mask_base = torch.cat([mask_base, pad_tensor], dim=3)
        mask_expanded = mask_base.unsqueeze(-1)
        scores = scores.masked_fill(mask_expanded == 0, float("-inf"))

        if save_attention:
            with torch.no_grad():
                self.last_scores_matrix = scores.mean(dim=-1)[:, :, :, :self.block_size].cpu().float()
                self.last_bias_physics_matrix = bias_physics[:, :, :self.block_size].cpu().float()

        attn_probs = F.softmax(scores, dim=3)
        linear_edge_weights = self.edge_gate(bias_physics)
        linear_edge_weights = linear_edge_weights.masked_fill(mask_expanded == 0, 0.0)
        combined_weights = attn_probs + linear_edge_weights

        if not torch.compiler.is_compiling():
            with torch.no_grad():
                attn_viz = combined_weights.mean(dim=-1)
                arange = torch.arange(self.block_size, device=attn_viz.device)
                self.last_attn_self = attn_viz[:, :, arange, arange].mean().item()
                if use_block_node and attn_viz.shape[3] > self.block_size:
                    self.last_attn_block = attn_viz[:, :, :, self.block_size].mean().item()
                else:
                    self.last_attn_block = 0.0
                to_nodes = attn_viz[:, :, :, :self.block_size].sum(dim=3)
                self.last_attn_neighbor = (to_nodes - attn_viz[:, :, arange, arange]).mean().item()
                if save_attention:
                    self.last_attn_matrix = attn_viz[:, :, :, :self.block_size].cpu().float()

        x_out = torch.einsum("bnqkh,bnkhd->bnqhd", combined_weights, v)
        x_out = x_out.reshape(B, num_blocks, self.block_size, C)
        x_out = self.proj(x_out.view(B, N_pad, C))
        if node_pad > 0:
            x_out = x_out[:, :N, :]
        return x_out


class TransformerBlock(nn.Module):
    def __init__(self, dim, block_size, attn_module, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attn_module

    def forward(self, x, edge_index=None, edge_values=None, positions=None, save_attention=False, attn_mask=None, edge_feats=None):
        x = x + self.attn(
            self.norm1(x),
            edge_index=edge_index,
            edge_values=edge_values,
            positions=positions,
            save_attention=save_attention,
            attn_mask=attn_mask,
            edge_feats=edge_feats,
        )
        return x


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


class LeafOnlyNet(nn.Module):
    def __init__(
        self,
        input_dim=9,
        d_model=128,
        leaf_size=32,
        num_layers=2,
        num_heads=4,
        attention_layout="32x33",
        use_gcn=True,
        num_gcn_layers=PhysicsAwareEmbedding.DEFAULT_NUM_GCN_LAYERS,
        use_jacobi=True,
    ):
        super().__init__()
        self.leaf_size = leaf_size
        self.embed = PhysicsAwareEmbedding(
            input_dim,
            d_model,
            use_gcn=use_gcn,
            global_features_dim=GLOBAL_FEATURES_DIM,
            num_gcn_layers=num_gcn_layers,
        )
        self.enc_input_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    block_size=leaf_size,
                    attn_module=LeafBlockAttention(
                        d_model,
                        leaf_size,
                        num_heads=num_heads,
                        attention_layout=attention_layout,
                    ),
                )
                for _ in range(num_layers)
            ]
        )
        self.leaf_head = nn.Linear(d_model, leaf_size)
        nn.init.normal_(self.leaf_head.weight, std=0.001)
        nn.init.constant_(self.leaf_head.bias, 0.0)
        self.jacobi_gate = nn.Linear(d_model, 1)
        nn.init.normal_(self.jacobi_gate.weight, std=0.01)
        nn.init.constant_(self.jacobi_gate.bias, 0.0)
        self.use_jacobi = use_jacobi

    def _get_leaf_blocks(self, h):
        B, N, _ = h.shape
        num_leaves = N // self.leaf_size
        h_leaves = h.view(B, num_leaves, self.leaf_size, -1)
        u_leaf = self.leaf_head(h_leaves)
        return torch.matmul(u_leaf, u_leaf.transpose(-1, -2))

    def _get_jacobi_scale(self, h):
        if not self.use_jacobi:
            return None
        B, N, C = h.shape
        num_leaves = N // self.leaf_size
        h_leaves = h.view(B, num_leaves, self.leaf_size, C)
        j_gate = torch.sigmoid(self.jacobi_gate(h_leaves)).squeeze(-1) * 2.0
        self._last_j_gate = j_gate.detach()
        return j_gate.view(B, N)

    def forward(self, x, edge_index, edge_values, save_attention=False, precomputed_leaf_connectivity=None, global_features=None):
        B, N, _ = x.shape
        assert N % self.leaf_size == 0, f"LeafOnly expects N divisible by leaf_size {self.leaf_size}, got {N}"
        if global_features is None:
            raise ValueError("global_features is required for LeafOnlyNet.forward.")
        positions = x[0, :, :3] if x.dim() == 3 else x[:, :3]
        h = self.embed(x, edge_index, edge_values, global_features=global_features)
        h = self.enc_input_proj(h)
        attn_mask, edge_feats = None, None
        if edge_index is not None and edge_values is not None:
            device, dtype = x.device, x.dtype
            if precomputed_leaf_connectivity is not None:
                attn_mask, edge_feats = precomputed_leaf_connectivity
            else:
                attn_mask, edge_feats = build_leaf_block_connectivity(
                    edge_index, edge_values, positions, self.leaf_size, device, dtype, num_hops=ATTENTION_HOPS
                )
        for block in self.blocks:
            h = block(
                h,
                edge_index=edge_index,
                edge_values=edge_values,
                positions=positions,
                save_attention=save_attention,
                attn_mask=attn_mask,
                edge_feats=edge_feats,
            )
        diag_blocks = self._get_leaf_blocks(h)
        jacobi_scale = self._get_jacobi_scale(h)
        return diag_blocks, jacobi_scale


def apply_block_diagonal_M(precond_out, x, leaf_size=LEAF_SIZE, jacobi_inv_diag=None):
    diag_blocks, jacobi_scale = precond_out
    B, N, K = x.shape
    num_leaves = diag_blocks.shape[1]
    x_leaves = x.view(B, num_leaves, leaf_size, K)
    y = torch.matmul(diag_blocks, x_leaves).view(B, N, K)
    if jacobi_scale is not None:
        if jacobi_inv_diag is None:
            raise ValueError("jacobi_inv_diag is required when jacobi_scale is present.")
        if jacobi_inv_diag.shape[:2] != (B, N):
            raise ValueError(f"jacobi_inv_diag shape {jacobi_inv_diag.shape} does not match (B,N)=({B},{N}).")
        y = y + (jacobi_scale * jacobi_inv_diag).unsqueeze(-1) * x
    return y
