import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    ATTENTION_HOPS,
    GLOBAL_FEATURES_DIM,
    LEAF_SIZE,
)
from .data import build_leaf_block_connectivity


def _normalize_attention_layout_string(layout: str) -> str:
    return layout.strip().lower().replace("×", "x")


def parse_attention_layout(layout: str, block_size: int) -> tuple[bool, bool]:
    """
    Returns (use_block_node, use_matrix_node) for LeafBlockAttention.
    Layout must be '{L}x{L}', '{L}x{L+1}', or '{L}x{L+2}' with L == block_size (query rows = leaf nodes).
    """
    parts = _normalize_attention_layout_string(layout).split("x")
    if len(parts) != 2:
        raise ValueError(
            f"attention_layout {layout!r} must look like '{{L}}x{{L}}', '{{L}}x{{L+1}}', or '{{L}}x{{L+2}}'"
        )
    r, c = int(parts[0]), int(parts[1])
    L = int(block_size)
    if r != L:
        raise ValueError(f"attention_layout first dimension ({r}) must equal leaf_size ({L})")
    if c == L:
        return False, False
    if c == L + 1:
        return True, False
    if c == L + 2:
        return True, True
    raise ValueError(
        f"attention_layout second dimension ({c}) must be L, L+1, or L+2 for leaf_size={L}; got {layout!r}"
    )


def attention_layout_choices(leaf_size: int) -> tuple[str, str, str]:
    L = int(leaf_size)
    return (f"{L}x{L}", f"{L}x{L + 1}", f"{L}x{L + 2}")


def default_attention_layout(leaf_size: int) -> str:
    L = int(leaf_size)
    return f"{L}x{L + 1}"


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
        w = edge_values.to(x.dtype)

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
    def __init__(self, dim, block_size, num_heads=8, attention_layout=None):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        if attention_layout is None:
            attention_layout = default_attention_layout(block_size)
        self.use_block_node, self.use_matrix_node = parse_attention_layout(attention_layout, block_size)
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
        device = x.device
        dtype = x.dtype
        node_pad = 0
        if N % self.block_size != 0:
            node_pad = self.block_size - (N % self.block_size)
            x = F.pad(x, (0, 0, 0, node_pad))
        N_pad = x.shape[1]
        num_blocks = N_pad // self.block_size
        x_blk = x.view(B, num_blocks, self.block_size, C)

        if attn_mask is None:
            if edge_feats is not None:
                raise ValueError("LeafBlockAttention: edge_feats set but attn_mask is None.")
            return x[:, :N, :] if node_pad > 0 else x
        if edge_feats is None:
            raise ValueError("LeafBlockAttention requires edge_feats when attn_mask is provided.")

        use_block_node = self.use_block_node
        use_matrix_node = self.use_matrix_node
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

        if not self.training and not torch.compiler.is_compiling():
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
        leaf_size=LEAF_SIZE,
        num_layers=2,
        num_heads=8,
        attention_layout=None,
        use_gcn=True,
        num_gcn_layers=PhysicsAwareEmbedding.DEFAULT_NUM_GCN_LAYERS,
        use_jacobi=True,
    ):
        super().__init__()
        self.leaf_size = int(leaf_size)
        if attention_layout is None:
            attention_layout = default_attention_layout(self.leaf_size)
        self.embed = PhysicsAwareEmbedding(
            input_dim,
            d_model,
            use_gcn=use_gcn,
            global_features_dim=GLOBAL_FEATURES_DIM,
            num_gcn_layers=num_gcn_layers,
        )
        self.enc_input_proj = nn.Linear(d_model, d_model)
        L = self.leaf_size
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    block_size=L,
                    attn_module=LeafBlockAttention(
                        d_model,
                        L,
                        num_heads=num_heads,
                        attention_layout=attention_layout,
                    ),
                )
                for _ in range(num_layers)
            ]
        )
        self.off_diag_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    block_size=L,
                    attn_module=LeafBlockAttention(
                        d_model,
                        L,
                        num_heads=num_heads,
                        attention_layout=attention_layout,
                    ),
                )
                for _ in range(num_layers)
            ]
        )
        self.off_diag_head_U = nn.Linear(d_model, L)
        self.off_diag_head_V = nn.Linear(d_model, L)
        nn.init.normal_(self.off_diag_head_U.weight, std=0.001)
        nn.init.constant_(self.off_diag_head_U.bias, 0.0)
        nn.init.normal_(self.off_diag_head_V.weight, std=0.001)
        nn.init.constant_(self.off_diag_head_V.bias, 0.0)
        self.leaf_head = nn.Linear(d_model, L)
        nn.init.normal_(self.leaf_head.weight, std=0.001)
        nn.init.constant_(self.leaf_head.bias, 0.0)
        self.jacobi_gate = nn.Linear(d_model, 1)
        nn.init.normal_(self.jacobi_gate.weight, std=0.01)
        nn.init.constant_(self.jacobi_gate.bias, 0.0)
        self.use_jacobi = use_jacobi

    def _get_leaf_blocks(self, h, mode="diagonal"):
        B, N_or_P_len, _ = h.shape
        num_leaves = N_or_P_len // self.leaf_size
        h_leaves = h.view(B, num_leaves, self.leaf_size, -1)

        if mode == "diagonal":
            u_leaf = self.leaf_head(h_leaves)
            return torch.matmul(u_leaf, u_leaf.transpose(-1, -2))
        u_leaf = self.off_diag_head_U(h_leaves)
        v_leaf = self.off_diag_head_V(h_leaves)
        return torch.matmul(u_leaf, v_leaf.transpose(-1, -2))

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
        attn_mask, edge_feats, off_attn_mask, off_edge_feats = None, None, None, None

        if edge_index is not None and edge_values is not None:
            device, dtype = x.device, x.dtype
            if precomputed_leaf_connectivity is not None:
                attn_mask, edge_feats, off_attn_mask, off_edge_feats = precomputed_leaf_connectivity
            else:
                attn_mask, edge_feats, off_attn_mask, off_edge_feats = build_leaf_block_connectivity(
                    edge_index,
                    edge_values,
                    positions,
                    self.leaf_size,
                    device,
                    dtype,
                    num_hops=ATTENTION_HOPS,
                )

        h_diag = h
        for block in self.blocks:
            h_diag = block(
                h_diag,
                edge_index=edge_index,
                edge_values=edge_values,
                positions=positions,
                save_attention=save_attention,
                attn_mask=attn_mask,
                edge_feats=edge_feats,
            )
        diag_blocks = self._get_leaf_blocks(h_diag, mode="diagonal")

        B_h, N_h, C_h = h.shape
        K = N_h // self.leaf_size
        r_idx, c_idx = torch.triu_indices(K, K, offset=1, device=h.device)
        P = r_idx.shape[0]

        if P > 0:
            h_k = h.view(B_h, K, self.leaf_size, C_h)
            h_pairs = (h_k[:, r_idx] + h_k[:, c_idx]).view(B_h, P * self.leaf_size, C_h)

            h_off = h_pairs
            for block in self.off_diag_blocks:
                h_off = block(
                    h_off,
                    edge_index=edge_index,
                    edge_values=edge_values,
                    positions=positions,
                    save_attention=False,
                    attn_mask=off_attn_mask,
                    edge_feats=off_edge_feats,
                )
            off_diag_blocks = self._get_leaf_blocks(h_off, mode="off-diagonal")
        else:
            off_diag_blocks = torch.empty((B_h, 0, self.leaf_size, self.leaf_size), device=h.device, dtype=h.dtype)

        jacobi_scale = self._get_jacobi_scale(h_diag)

        B = diag_blocks.shape[0]
        packed_diag = diag_blocks.reshape(B, -1)
        packed_off = off_diag_blocks.reshape(B, -1)

        packed = torch.cat([packed_diag, packed_off], dim=1)
        if jacobi_scale is not None:
            packed = torch.cat([packed, jacobi_scale], dim=1)

        return packed


def unpack_precond(precond_packed, N, leaf_size=LEAF_SIZE):
    B = precond_packed.shape[0]
    num_leaves = N // leaf_size
    diag_size = num_leaves * leaf_size * leaf_size
    P = (num_leaves * (num_leaves - 1)) // 2
    off_size = P * leaf_size * leaf_size

    diag_blocks = precond_packed[:, :diag_size].view(B, num_leaves, leaf_size, leaf_size)
    off_diag_blocks = None
    if P > 0:
        off_diag_blocks = precond_packed[:, diag_size : diag_size + off_size].view(B, P, leaf_size, leaf_size)

    jacobi_scale = precond_packed[:, diag_size + off_size :] if precond_packed.shape[1] > diag_size + off_size else None
    return diag_blocks, off_diag_blocks, jacobi_scale


def apply_block_diagonal_M(precond_packed, x, leaf_size=LEAF_SIZE, jacobi_inv_diag=None):
    B, N, K_dim = x.shape
    num_leaves = N // leaf_size
    diag_size = num_leaves * leaf_size * leaf_size
    P = (num_leaves * (num_leaves - 1)) // 2
    off_size = P * leaf_size * leaf_size

    diag_blocks = precond_packed[:, :diag_size].view(B, num_leaves, leaf_size, leaf_size)
    x_leaves = x.view(B, num_leaves, leaf_size, K_dim)

    y_leaves = torch.matmul(diag_blocks, x_leaves)

    if P > 0:
        off_diag_blocks = precond_packed[:, diag_size : diag_size + off_size].view(B, P, leaf_size, leaf_size)
        r_idx, c_idx = torch.triu_indices(num_leaves, num_leaves, offset=1, device=x.device)

        y_r_add = torch.matmul(off_diag_blocks, x_leaves[:, c_idx])
        y_c_add = torch.matmul(off_diag_blocks.transpose(-1, -2), x_leaves[:, r_idx])

        y_leaves.index_add_(1, r_idx, y_r_add)
        y_leaves.index_add_(1, c_idx, y_c_add)

    y = y_leaves.view(B, N, K_dim)

    if precond_packed.shape[1] > diag_size + off_size:
        if jacobi_inv_diag is None:
            raise ValueError("jacobi_inv_diag is required when jacobi_scale is present.")
        if jacobi_inv_diag.shape[:2] != (B, N):
            raise ValueError(f"jacobi_inv_diag shape {jacobi_inv_diag.shape} does not match (B,N)=({B},{N}).")
        jacobi_scale = precond_packed[:, diag_size + off_size :]
        y = y + (jacobi_scale * jacobi_inv_diag).unsqueeze(-1) * x
    return y
