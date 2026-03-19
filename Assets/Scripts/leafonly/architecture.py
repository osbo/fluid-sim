import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    ATTENTION_HOPS,
    GLOBAL_FEATURES_DIM,
    LEAF_SIZE,
    OFF_DIAG_SUPER,
    RANK_BASE_LEVEL1,
)
from .data import build_leaf_block_connectivity, build_off_diag_super_connectivity_features


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
    NUM_GCN_LAYERS = 2
    LIFT_FEATURES = 6

    def __init__(self, input_dim, d_model, use_gcn=True, global_features_dim=0):
        super().__init__()
        self.use_gcn = use_gcn
        self.global_features_dim = global_features_dim
        lift_in = self.LIFT_FEATURES + (global_features_dim or 0)
        self.lift = nn.Sequential(
            nn.Linear(lift_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.gcn = nn.ModuleList([SparsePhysicsGCN(d_model) for _ in range(self.NUM_GCN_LAYERS)]) if use_gcn else None
        self.norm = nn.LayerNorm(d_model)
        if global_features_dim > 0:
            self.lift_film = nn.Sequential(
                nn.Linear(global_features_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2 * d_model),
            )
            with torch.no_grad():
                self.lift_film[2].weight.zero_()
                self.lift_film[2].bias.zero_()
                self.lift_film[2].bias[:d_model].fill_(1.0)
        else:
            self.lift_film = None

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
        if self.lift_film is not None:
            gf = global_features.unsqueeze(0) if global_features.dim() == 1 else global_features
            film_out = self.lift_film(gf)
            gamma, beta = film_out.chunk(2, dim=-1)
            h = (gamma.unsqueeze(1) * h) + beta.unsqueeze(1)
        return h


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
        if not self.mask_attention:
            raise ValueError("mask_attention=False path has been removed.")
        if not self.use_global_node:
            raise ValueError("use_global_node=False path has been removed.")

        device = x.device
        dtype = x.dtype
        pad = 0
        if N % self.block_size != 0:
            pad = self.block_size - (N % self.block_size)
            x = F.pad(x, (0, 0, 0, pad))
        N_pad = x.shape[1]
        num_blocks = N_pad // self.block_size
        x_blk = x.view(B, num_blocks, self.block_size, C)

        if attn_mask is None or edge_feats is None:
            attn_mask, edge_feats = build_leaf_block_connectivity(
                edge_index, edge_values, positions, self.block_size, device, dtype
            )
        if attn_mask is None:
            return x[:, :N, :] if pad > 0 else x

        block_node = x_blk.mean(dim=2, keepdim=True)
        kv = torch.cat([x_blk, block_node], dim=2)
        qkv_q = self.qkv(x_blk)
        qkv_kv = self.qkv(kv)
        q = qkv_q[..., :C]
        k = qkv_kv[..., C:2 * C]
        v = qkv_kv[..., 2 * C:3 * C]
        q = q.view(B, num_blocks, self.block_size, self.num_heads, self.head_dim)
        k = k.view(B, num_blocks, self.block_size + 1, self.num_heads, self.head_dim)
        v = v.view(B, num_blocks, self.block_size + 1, self.num_heads, self.head_dim)

        scores = torch.einsum("bnqhd,bnkhd->bnqkh", q, k) * self.scale

        if edge_feats.dim() == 4:
            bias_physics = edge_feats[..., :4].unsqueeze(0).expand(B, -1, -1, -1, -1).clone()
        else:
            bias_physics = edge_feats[..., :4].clone()
        bias_physics[:, :, :, self.block_size, :] = 0.0
        bias_physics[:, :, :, self.block_size, 3] = 1.0
        arange = torch.arange(self.block_size, device=device)
        bias_physics[:, :, arange, arange, :] = 0.0
        bias_physics[:, :, arange, arange, 3] = 1.0
        scores = scores + bias_physics[..., 3:4]
        if attn_mask.dim() == 3:
            mask_expanded = attn_mask.unsqueeze(0).unsqueeze(-1)
        else:
            mask_expanded = attn_mask.unsqueeze(-1)
        scores = scores.masked_fill(mask_expanded == 0, float("-inf"))

        if save_attention:
            with torch.no_grad():
                self.last_scores_matrix = scores.mean(dim=-1)[:, :, :, :self.block_size].cpu().float()
                self.last_bias_physics_matrix = bias_physics[:, :, :self.block_size].cpu().float()

        attn_probs = F.softmax(scores, dim=3)
        linear_edge_weights = self.edge_gate(bias_physics).masked_fill(mask_expanded == 0, 0.0)
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

        x_out = torch.einsum("bnqkh,bnkhd->bnqhd", combined_weights, v)
        x_out = x_out.reshape(B, num_blocks, self.block_size, C)
        x_out = self.proj(x_out.view(B, N_pad, C))
        if pad > 0:
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
            specs.append(
                {
                    "row_start": row_start,
                    "row_end": row_end,
                    "col_start": col_start,
                    "col_end": col_end,
                    "side": segment_size,
                    "rank": rank,
                    "level": level_idx,
                }
            )
    return specs


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
        self.embed = PhysicsAwareEmbedding(input_dim, d_model, use_gcn=use_gcn, global_features_dim=GLOBAL_FEATURES_DIM)
        self.enc_input_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    block_size=leaf_size,
                    attn_module=LeafBlockAttention(d_model, leaf_size, num_heads=num_heads, mask_attention=mask_attention, use_global_node=use_global_node),
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

    def get_leaf_blocks(self, h, x):
        B, N, _ = h.shape
        num_leaves = N // self.leaf_size
        h_leaves = h.view(B, num_leaves, self.leaf_size, -1)
        u_leaf = self.leaf_head(h_leaves)
        leaf_base = torch.matmul(u_leaf, u_leaf.transpose(-1, -2))
        if not self.use_jacobi:
            return leaf_base
        diag_A_n = x[..., 8]
        jacobi_diag = torch.zeros_like(diag_A_n)
        mask = diag_A_n.abs() > 1e-6
        jacobi_diag[mask] = 1.0 / diag_A_n[mask]
        jacobi_diag_blocks = jacobi_diag.view(B, num_leaves, self.leaf_size)
        mask_blocks = mask.view(B, num_leaves, self.leaf_size)
        new_diag = torch.where(mask_blocks, jacobi_diag_blocks, torch.ones_like(jacobi_diag_blocks))
        j_gate = torch.sigmoid(self.jacobi_gate(h_leaves)).squeeze(-1) * 2.0
        self._last_j_gate = j_gate.detach()
        local_jacobi_diag = new_diag * j_gate
        return leaf_base + torch.diag_embed(local_jacobi_diag)

    def forward_features(self, x, edge_index=None, edge_values=None, save_attention=False, precomputed_leaf_connectivity=None, global_features=None):
        B, N, _ = x.shape
        positions = x[0, :, :3] if x.dim() == 3 else x[:, :3]
        if global_features is None:
            raise ValueError("global_features is required for LeafCore.forward_features.")
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
        return h

    def forward(self, x, edge_index=None, edge_values=None, save_attention=False):
        h = self.forward_features(x, edge_index, edge_values, save_attention=save_attention)
        return self.get_leaf_blocks(h, x)


class LevelSpecificOffDiagBlock(nn.Module):
    NUM_GLOBAL_TOKENS = 1

    def __init__(self, d_model, exact_rank, global_features_dim=0):
        super().__init__()
        self.n_super = OFF_DIAG_SUPER
        self.n_attn = self.n_super + self.NUM_GLOBAL_TOKENS
        self.exact_rank = max(1, int(exact_rank))
        self.global_features_dim = int(global_features_dim or 0)
        self._context_dim = 4 + self.global_features_dim
        self._proj_in_dim = 3 * d_model + self._context_dim
        qkv_in_dim = d_model + self._context_dim
        self.attn_q = nn.Linear(qkv_in_dim, d_model)
        self.attn_k = nn.Linear(qkv_in_dim, d_model)
        self.attn_v = nn.Linear(qkv_in_dim, d_model)
        self.super_edge_gate = nn.Linear(1, 1)
        self.feature_ln = nn.LayerNorm(self._proj_in_dim)
        self.proj_U = nn.Linear(self._proj_in_dim, self.exact_rank)
        self.proj_V = nn.Linear(self._proj_in_dim, self.exact_rank)
        with torch.no_grad():
            nn.init.zeros_(self.proj_U.weight)
            nn.init.zeros_(self.proj_U.bias)
            nn.init.normal_(self.proj_V.weight, std=0.01)
            nn.init.zeros_(self.proj_V.bias)
        for m in (self.attn_q, self.attn_k, self.attn_v):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.super_edge_gate.weight.fill_(1.0)
            self.super_edge_gate.bias.zero_()
        self._scale_attn = d_model ** -0.5

    def forward(self, h_row, h_col, pos_row, pos_col, mask, super_edge_strength=None, return_features=False, global_features=None, return_attn_debug=False):
        B = h_row.shape[0]
        side = h_row.shape[1]
        g = side // self.n_super
        down_row = h_row.view(B, self.n_super, g, -1).amax(dim=2)
        down_col = h_col.view(B, self.n_super, g, -1).amax(dim=2)
        down_pos_row = pos_row.view(B, self.n_super, g, -1).amax(dim=2)
        down_pos_col = pos_col.view(B, self.n_super, g, -1).amax(dim=2)
        delta_x = down_pos_col - down_pos_row
        dist = torch.norm(delta_x, dim=-1, keepdim=True).clamp(min=1e-8)
        if self.global_features_dim > 0:
            if global_features is None:
                raise ValueError("global_features is required for LevelSpecificOffDiagBlock.")
            gf = global_features if global_features.dim() == 2 else global_features.unsqueeze(0)
            gf_exp = gf.unsqueeze(1).expand(-1, self.n_super, -1)
            context_U = torch.cat([delta_x, dist, gf_exp], dim=-1)
            context_V = torch.cat([-delta_x, dist, gf_exp], dim=-1)
        else:
            context_U = torch.cat([delta_x, dist], dim=-1)
            context_V = torch.cat([-delta_x, dist], dim=-1)
        down_row_cond = torch.cat([down_row, context_U], dim=-1)
        down_col_cond = torch.cat([down_col, context_V], dim=-1)
        block_avg_row = torch.cat([down_row.amax(dim=1, keepdim=True), context_U.amax(dim=1, keepdim=True)], dim=-1)
        block_avg_col = torch.cat([down_col.amax(dim=1, keepdim=True), context_V.amax(dim=1, keepdim=True)], dim=-1)
        row_tokens = torch.cat([down_row_cond, block_avg_row], dim=1)
        col_tokens = torch.cat([down_col_cond, block_avg_col], dim=1)
        Q = self.attn_q(row_tokens)
        K = self.attn_k(col_tokens)
        V = self.attn_v(col_tokens)
        scores = (Q @ K.transpose(-2, -1)) * self._scale_attn
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        edge_scale_33 = None
        scores_pre_mask = scores
        if super_edge_strength is not None:
            edge_strength = super_edge_strength
            if edge_strength.dim() == 2:
                edge_strength = edge_strength.unsqueeze(0)
            edge_bias = self.super_edge_gate(edge_strength.to(dtype=scores.dtype).unsqueeze(-1)).squeeze(-1)
            edge_bias_33 = F.pad(edge_bias, (0, self.NUM_GLOBAL_TOKENS, 0, self.NUM_GLOBAL_TOKENS), value=0.0)
            edge_scale_33 = torch.exp(edge_bias_33.clamp(min=-10.0, max=10.0))
        mask_33 = F.pad(mask, (0, self.NUM_GLOBAL_TOKENS, 0, self.NUM_GLOBAL_TOKENS), value=True)
        scores_masked = scores_pre_mask.masked_fill(~mask_33, float("-inf"))
        attn_probs = F.softmax(scores_masked, dim=-1)
        if edge_scale_33 is not None:
            attn_probs = attn_probs * edge_scale_33
            attn_probs = attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + 1e-12)

        attn_debug = None
        if return_attn_debug:
            with torch.no_grad():
                scores_sel = scores_pre_mask.masked_select(mask_33)
                scores_mean = scores_sel.mean().item() if scores_sel.numel() > 0 else float("nan")
                scores_std = scores_sel.std(unbiased=False).item() if scores_sel.numel() > 1 else float("nan")
                scores_max = scores_sel.max().item() if scores_sel.numel() > 0 else float("nan")
                p = attn_probs.clamp_min(1e-12)
                entropy = (-(p * p.log()).sum(dim=-1))
                entropy_mean = entropy.mean().item()
                entropy_std = entropy.std(unbiased=False).item()
                if edge_scale_33 is not None:
                    es_sel = edge_scale_33.masked_select(mask_33)
                    es_mean = es_sel.mean().item() if es_sel.numel() > 0 else float("nan")
                    es_std = es_sel.std(unbiased=False).item() if es_sel.numel() > 1 else float("nan")
                else:
                    es_mean, es_std = float("nan"), float("nan")
                attn_debug = {
                    "scores_mean_on_mask": scores_mean,
                    "scores_std_on_mask": scores_std,
                    "scores_max_on_mask": scores_max,
                    "attn_entropy_mean": entropy_mean,
                    "attn_entropy_std": entropy_std,
                    "edge_scale_mean_on_mask": es_mean,
                    "edge_scale_std_on_mask": es_std,
                    "attn_probs_mean": attn_probs.mean().item(),
                    "attn_probs_std": attn_probs.std(unbiased=False).item(),
                }
        row_out_full = attn_probs @ V
        row_out = row_out_full[:, : self.n_super].reshape(B, self.n_super, -1)
        global_ctx = row_out_full[:, self.n_super :, :].mean(dim=1)
        row_out_expanded = row_out.repeat_interleave(g, dim=1)
        down_col_expanded = down_col.repeat_interleave(g, dim=1)
        context_U_expanded = context_U.repeat_interleave(g, dim=1)
        context_V_expanded = context_V.repeat_interleave(g, dim=1)
        global_ctx_expanded = global_ctx.unsqueeze(1).expand(-1, side, -1)
        h_row_3 = h_row.reshape(B, -1, h_row.shape[-1])
        row_exp_3 = row_out_expanded.reshape(B, -1, row_out_expanded.shape[-1])
        h_col_3 = h_col.reshape(B, -1, h_col.shape[-1])
        col_exp_3 = down_col_expanded.reshape(B, -1, down_col_expanded.shape[-1])
        features_U = torch.cat([h_row_3, row_exp_3, global_ctx_expanded, context_U_expanded], dim=-1)
        features_V = torch.cat([h_col_3, col_exp_3, global_ctx_expanded, context_V_expanded], dim=-1)
        features_U = self.feature_ln(features_U)
        features_V = self.feature_ln(features_V)
        U_raw = self.proj_U(features_U)
        V_raw = self.proj_V(features_V)
        rank_scale = math.sqrt(float(self.exact_rank))
        out_U = U_raw / rank_scale
        out_V = V_raw / rank_scale
        if return_features:
            if return_attn_debug:
                return out_U, out_V, features_U.detach(), features_V.detach(), attn_debug
            return out_U, out_V, features_U.detach(), features_V.detach()
        if return_attn_debug:
            return out_U, out_V, attn_debug
        return out_U, out_V


class LeafOnlyNet(nn.Module):
    def __init__(self, input_dim=9, d_model=128, leaf_size=32, num_layers=3, num_heads=4, rank_base=RANK_BASE_LEVEL1, mask_attention=True, use_global_node=True, use_gcn=True, use_jacobi=True, max_levels=3):
        super().__init__()
        self.leaf_size = leaf_size
        self.rank_base = rank_base
        self.max_levels = max(1, int(max_levels))
        self.core = LeafCore(input_dim=input_dim, d_model=d_model, leaf_size=leaf_size, num_layers=num_layers, num_heads=num_heads, mask_attention=mask_attention, use_global_node=use_global_node, use_gcn=use_gcn, use_jacobi=use_jacobi)
        self.level_ranks = [_rank_for_side(self.leaf_size * (2 ** lvl_idx), rank_base=self.rank_base) for lvl_idx in range(self.max_levels)]
        self.level_off_diag = nn.ModuleList(
            [LevelSpecificOffDiagBlock(d_model, exact_rank=self.level_ranks[lvl_idx], global_features_dim=GLOBAL_FEATURES_DIM) for lvl_idx in range(self.max_levels)]
        )
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

    def _off_diag(self, h, x, edge_index, edge_values, current_struct, precomputed_masks, global_features):
        B, N, C = h.shape
        if precomputed_masks is None or len(precomputed_masks) == 0 or not isinstance(precomputed_masks[0], tuple):
            raise ValueError("precomputed_masks must be provided as a non-empty list of (mask, strength) tuples.")
        if global_features is None:
            raise ValueError("global_features is required in LeafOnlyNet._off_diag.")
        mask_list = [m for (m, _) in precomputed_masks]
        edge_strength_list = [w for (_, w) in precomputed_masks]
        result = [None] * len(current_struct)

        level_to_blocks = {}
        for idx, spec in enumerate(current_struct):
            lvl = spec["level"]
            if lvl not in level_to_blocks:
                level_to_blocks[lvl] = []
            level_to_blocks[lvl].append((idx, spec))

        for lvl in sorted(level_to_blocks.keys()):
            module_idx = lvl - 1
            if module_idx >= self.max_levels:
                raise ValueError(f"Graph requires {lvl} off-diagonal levels, but model only has {self.max_levels} level blocks.")
            blocks = level_to_blocks[lvl]
            rank = blocks[0][1]["rank"]
            expected_rank = self.level_off_diag[module_idx].exact_rank
            if rank != expected_rank:
                raise ValueError(f"HODLR rank mismatch at level {lvl}: struct rank={rank}, level block rank={expected_rank}.")
            h_rows, h_cols, pos_rows, pos_cols, masks, edge_strengths = [], [], [], [], [], []
            for idx, spec in blocks:
                rs, re = spec["row_start"], spec["row_end"]
                cs, ce = spec["col_start"], spec["col_end"]
                h_rows.append(h[:, rs:re])
                h_cols.append(h[:, cs:ce])
                pos_rows.append(x[:, rs:re, :3])
                pos_cols.append(x[:, cs:ce, :3])
                masks.append(mask_list[idx])
                edge_strengths.append(edge_strength_list[idx])

            h_row_batched = torch.cat(h_rows, dim=0)
            h_col_batched = torch.cat(h_cols, dim=0)
            pos_row_batched = torch.cat(pos_rows, dim=0)
            pos_col_batched = torch.cat(pos_cols, dim=0)
            K = len(blocks)
            stacked_masks = torch.stack(masks, dim=0)
            stacked_edge_strengths = torch.stack(edge_strengths, dim=0)
            if stacked_masks.dim() == 4:
                batched_masks = stacked_masks.permute(1, 0, 2, 3).reshape(B * K, self.level_off_diag[module_idx].n_super, self.level_off_diag[module_idx].n_super)
                batched_edge_strengths = stacked_edge_strengths.permute(1, 0, 2, 3).reshape(B * K, self.level_off_diag[module_idx].n_super, self.level_off_diag[module_idx].n_super)
            else:
                batched_masks = stacked_masks.repeat_interleave(B, dim=0)
                batched_edge_strengths = stacked_edge_strengths.repeat_interleave(B, dim=0)
            gf = global_features.unsqueeze(0).expand(B, -1) if global_features.dim() == 1 else global_features
            gf_batched = gf.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
            U_batched, V_batched = self.level_off_diag[module_idx](
                h_row_batched,
                h_col_batched,
                pos_row_batched,
                pos_col_batched,
                batched_masks,
                super_edge_strength=batched_edge_strengths,
                global_features=gf_batched,
            )
            U_splits = torch.split(U_batched, B, dim=0)
            V_splits = torch.split(V_batched, B, dim=0)
            for i, (idx, _) in enumerate(blocks):
                result[idx] = (U_splits[i], V_splits[i])

        return result

    def forward(self, x, edge_index, edge_values, scale_A=None, save_attention=False, precomputed_masks=None, precomputed_leaf_connectivity=None, global_features=None):
        B, N, _ = x.shape
        assert N % self.leaf_size == 0, f"LeafOnly expects N divisible by leaf_size {self.leaf_size}, got {N}"
        if global_features is None:
            raise ValueError("global_features is required for LeafOnlyNet.forward.")
        current_struct = build_hodlr_off_diag_structure(N, self.leaf_size, self.rank_base)
        self.off_diag_struct = current_struct
        if precomputed_masks is None:
            # Compatibility path for inference callers (e.g. InspectModel) that do not precompute.
            precomputed_masks = [
                build_off_diag_super_connectivity_features(
                    edge_index,
                    edge_values,
                    spec["row_start"],
                    spec["row_end"],
                    spec["col_start"],
                    spec["col_end"],
                    x.device,
                    x.dtype,
                    self.leaf_size,
                )
                for spec in current_struct
            ]
        if precomputed_leaf_connectivity is None:
            positions = x[0, :N, :3]
            precomputed_leaf_connectivity = build_leaf_block_connectivity(
                edge_index,
                edge_values,
                positions,
                self.leaf_size,
                x.device,
                x.dtype,
                num_hops=ATTENTION_HOPS,
            )
        h = self.core.forward_features(
            x,
            edge_index=edge_index,
            edge_values=edge_values,
            save_attention=save_attention,
            precomputed_leaf_connectivity=precomputed_leaf_connectivity,
            global_features=global_features,
        )
        diag_blocks = self.core.get_leaf_blocks(h, x)
        if current_struct:
            off_diag_list = self._off_diag(h, x, edge_index, edge_values, current_struct, precomputed_masks=precomputed_masks, global_features=global_features)
            return (diag_blocks, off_diag_list)
        return (diag_blocks, [])


def apply_block_structured_M(diag_blocks, off_diag_list, x, off_diag_struct, leaf_size=LEAF_SIZE, scale_A=1.0):
    return apply_block_structured_M_with_levels(
        diag_blocks, off_diag_list, x, off_diag_struct, leaf_size=leaf_size, levels_to_include=None, scale_A=scale_A
    )


def _apply_scale_equivariance(out, scale_A):
    eps = 1e-6
    if scale_A is None:
        return out
    if isinstance(scale_A, torch.Tensor):
        if scale_A.numel() == 1:
            s = scale_A.to(device=out.device, dtype=out.dtype).clamp(min=eps)
            return out / s
        s = scale_A.to(device=out.device, dtype=out.dtype).view(-1, 1, 1).clamp(min=eps)
        return out / s
    s = max(float(scale_A), eps)
    return out / s


class HODLROperator:
    def __init__(self, diag_blocks, off_diag_list, off_diag_struct, leaf_size=LEAF_SIZE, scale_A=1.0):
        self.diag_blocks = diag_blocks
        self.off_diag_list = off_diag_list
        self.off_diag_struct = off_diag_struct
        self.leaf_size = leaf_size
        self.scale_A = scale_A

    def apply(self, x, levels_to_include=None):
        B, N, K = x.shape
        num_leaves = self.diag_blocks.shape[1]
        x_leaves = x.view(B, num_leaves, self.leaf_size, K)
        y = torch.matmul(self.diag_blocks, x_leaves).view(B, N, K)
        if self.off_diag_list and self.off_diag_struct:
            for (U, V), spec in zip(self.off_diag_list, self.off_diag_struct):
                lvl = spec.get("level", None)
                if levels_to_include is not None and lvl not in levels_to_include:
                    continue
                rs, re = spec["row_start"], spec["row_end"]
                cs, ce = spec["col_start"], spec["col_end"]
                y[:, rs:re, :] += U @ (V.transpose(1, 2) @ x[:, cs:ce, :])
                y[:, cs:ce, :] += V @ (U.transpose(1, 2) @ x[:, rs:re, :])
        return _apply_scale_equivariance(y, self.scale_A)


def build_hodlr_operator(diag_blocks, off_diag_list, off_diag_struct, leaf_size=LEAF_SIZE, scale_A=1.0):
    return HODLROperator(diag_blocks, off_diag_list, off_diag_struct, leaf_size=leaf_size, scale_A=scale_A)


def apply_block_structured_M_with_levels(diag_blocks, off_diag_list, x, off_diag_struct, leaf_size=LEAF_SIZE, levels_to_include=None, scale_A=1.0):
    if not off_diag_list or not off_diag_struct:
        B, N, K = x.shape
        num_leaves = diag_blocks.shape[1]
        x_leaves = x.view(B, num_leaves, leaf_size, K)
        out = torch.matmul(diag_blocks, x_leaves).view(B, N, K)
        return _apply_scale_equivariance(out, scale_A)
    op = HODLROperator(diag_blocks, off_diag_list, off_diag_struct, leaf_size=leaf_size, scale_A=scale_A)
    return op.apply(x, levels_to_include=levels_to_include)


def apply_leaf_only(leaf_blocks, x, off_diag_list=None, off_diag_struct=None, scale_A=1.0):
    B, N, K = x.shape
    num_leaves = leaf_blocks.shape[1]
    leaf_size = leaf_blocks.shape[2]
    if not off_diag_list or not off_diag_struct:
        x_leaves = x.view(B, num_leaves, leaf_size, K)
        y_leaves = torch.matmul(leaf_blocks, x_leaves)
        return _apply_scale_equivariance(y_leaves.view(B, N, K), scale_A)
    return apply_block_structured_M(leaf_blocks, off_diag_list, x, off_diag_struct, leaf_size=leaf_size, scale_A=scale_A)

