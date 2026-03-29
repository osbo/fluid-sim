import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    ATTENTION_HOPS,
    GLOBAL_FEATURES_DIM,
    LEAF_SIZE,
    MAX_NUM_LEAVES,
    OFF_DIAG_TOKEN_POOL,
)
from .data import build_leaf_block_connectivity
from .hmatrix import (
    HM_ADJ_CPU,
    HM_C0_CPU,
    HM_PROLONG_COL_LEAF_IDX,
    HM_PROLONG_GATHER_IDX,
    HM_PROLONG_ROW_LEAF_IDX,
    HM_R0_CPU,
    HM_S_CPU,
    HM_SUM_W_COL_CPU,
    HM_SUM_W_ROW_CPU,
    NUM_HMATRIX_OFF_BLOCKS,
    hm_leaf_mean_pool_weights,
)

# Prolongation indices live on CPU in hmatrix; copy once per device. CUDAGraph capture must not call .to().
_HM_PROLONG_GPU: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

# BSR crow/col + value permutation per (N, leaf, apply sizes, device); built once from a dummy COO→BSR.
_BSR_TOPOLOGY_CACHE: dict = {}


def _hm_prolong_scatter_indices(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = str(device)
    got = _HM_PROLONG_GPU.get(key)
    if got is None:
        dt = torch.long
        _HM_PROLONG_GPU[key] = got = (
            HM_PROLONG_ROW_LEAF_IDX.to(device=device, dtype=dt),
            HM_PROLONG_COL_LEAF_IDX.to(device=device, dtype=dt),
            HM_PROLONG_GATHER_IDX.to(device=device, dtype=dt),
        )
    return got


def _merge_batch_blocks(t: torch.Tensor, merge: bool) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
    if not merge or t.dim() < 4:
        return t, None
    B, nb = t.shape[0], t.shape[1]
    return t.reshape(B * nb, *t.shape[2:]), (B, nb)


def _unmerge_batch_blocks(t: torch.Tensor, batch_shape: Optional[Tuple[int, int]], tail_shape: Tuple[int, ...]) -> torch.Tensor:
    if batch_shape is None:
        return t
    B, nb = batch_shape
    return t.view(B, nb, *tail_shape)


def pool_leaf_attn_mask(attn_mask: torch.Tensor, leaf_size: int, pool: int) -> torch.Tensor:
    """Or-pool query×key within each leaf; supports (M, L, K) or (B, M, L, K)."""
    if pool <= 1:
        return attn_mask
    L = int(leaf_size)
    assert L % pool == 0, f"leaf_size {L} not divisible by pool {pool}"
    Ls = L // pool
    m, batch_shape = _merge_batch_blocks(attn_mask, attn_mask.dim() == 4)
    k_old = m.shape[-1]
    special = k_old - L
    if special not in (1, 2):
        raise ValueError(f"expected L+1 or L+2 key layout, got key_dim={k_old} for L={L}")
    nb = m.shape[0]
    sub = m[:, :L, :L].reshape(nb, Ls, pool, Ls, pool).amax(dim=(2, 4))
    parts = [sub]
    for j in range(special):
        c = m[:, :L, L + j].reshape(nb, Ls, pool).amax(dim=2)
        parts.append(c.unsqueeze(-1))
    out = torch.cat(parts, dim=-1)
    return _unmerge_batch_blocks(out, batch_shape, tuple(out.shape[1:]))


def pool_leaf_edge_feats(edge_feats: torch.Tensor, leaf_size: int, pool: int) -> torch.Tensor:
    """Mean-pool edge feat grid to match ``pool_leaf_attn_mask``."""
    if pool <= 1:
        return edge_feats
    L = int(leaf_size)
    assert L % pool == 0
    Ls = L // pool
    m, batch_shape = _merge_batch_blocks(edge_feats, edge_feats.dim() == 5)
    k_old = m.shape[-2]
    special = k_old - L
    if special not in (1, 2):
        raise ValueError(f"expected L+1 or L+2 key layout for edge_feats, got key_dim={k_old}")
    nb = m.shape[0]
    E = m.shape[-1]
    sub = m[:, :L, :L, :].reshape(nb, Ls, pool, Ls, pool, E).mean(dim=(2, 4))
    parts = [sub]
    for j in range(special):
        c = m[:, :L, L + j, :].reshape(nb, Ls, pool, E).mean(dim=2)
        parts.append(c.unsqueeze(2))
    out = torch.cat(parts, dim=2)
    return _unmerge_batch_blocks(out, batch_shape, tuple(out.shape[1:]))


def _normalize_attention_layout_string(layout: str) -> str:
    return layout.strip().lower().replace("×", "x")


def parse_attention_layout(layout: str, block_size: int) -> tuple[bool, bool, bool]:
    parts = _normalize_attention_layout_string(layout).split("x")
    if len(parts) != 2:
        raise ValueError(f"attention_layout {layout!r} must look like 'RxC' (e.g. LxL, (L+1)x(L+1)).")
    r, c = int(parts[0]), int(parts[1])
    L = int(block_size)
    query_has_block = r == L + 1
    if r not in (L, L + 1):
        raise ValueError(f"attention_layout first dimension must be L or L+1. Got {layout!r}")
    use_block_node = c >= L + 1
    use_matrix_node = c == L + 2
    return use_block_node, use_matrix_node, query_has_block


def attention_layout_choices(leaf_size: int) -> tuple[str, str, str, str]:
    L = int(leaf_size)
    return (f"{L}x{L}", f"{L}x{L + 1}", f"{L}x{L + 2}", f"{L + 1}x{L + 1}")


def default_attention_layout(leaf_size: int) -> str:
    L = int(leaf_size)
    return f"{L + 1}x{L + 1}"


def attention_layout_from_checkpoint_code(leaf_size: int, attention_layout_code: int) -> str:
    L = int(leaf_size)
    code = int(attention_layout_code)
    if code == 0:
        return f"{L}x{L}"
    if code == 1:
        return f"{L}x{L + 1}"
    if code == 2:
        return f"{L}x{L + 2}"
    if code == 3:
        return f"{L + 1}x{L + 1}"
    return default_attention_layout(L)


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
        self.block_size = int(block_size)
        if attention_layout is None:
            attention_layout = default_attention_layout(block_size)
        self.use_block_node, self.use_matrix_node, self.query_has_block = parse_attention_layout(attention_layout, block_size)
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
        L = self.block_size
        L_q = L + 1 if self.query_has_block else L

        node_pad = 0
        if N % L_q != 0:
            node_pad = L_q - (N % L_q)
            x = F.pad(x, (0, 0, 0, node_pad))
        N_pad = x.shape[1]
        num_blocks = N_pad // L_q
        x_blk = x.view(B, num_blocks, L_q, C)

        if attn_mask is None:
            if edge_feats is not None:
                raise ValueError("LeafBlockAttention: edge_feats set but attn_mask is None.")
            return x[:, :N, :] if node_pad > 0 else x
        if edge_feats is None:
            raise ValueError("LeafBlockAttention requires edge_feats when attn_mask is provided.")

        x_attn = x_blk
        kv_parts = [x_attn]
        if self.use_block_node and not self.query_has_block:
            block_node = x_attn.mean(dim=2, keepdim=True)
            kv_parts.append(block_node)
        if self.use_matrix_node:
            matrix_node = x.mean(dim=1, keepdim=True).unsqueeze(1).expand(-1, num_blocks, -1, -1)
            kv_parts.append(matrix_node)
        kv = torch.cat(kv_parts, dim=2)
        qkv_q = self.qkv(x_attn)
        qkv_kv = self.qkv(kv)
        q = qkv_q[..., :C]
        k = qkv_kv[..., C : 2 * C]
        v = qkv_kv[..., 2 * C : 3 * C]
        key_count = kv.shape[2]

        q = q.view(B, num_blocks, L_q, self.num_heads, self.head_dim)
        k = k.view(B, num_blocks, key_count, self.num_heads, self.head_dim)
        v = v.view(B, num_blocks, key_count, self.num_heads, self.head_dim)

        scores = torch.einsum("bnqhd,bnkhd->bnqkh", q, k) * self.scale

        if edge_feats.dim() == 4:
            bias_physics = edge_feats[..., :4].unsqueeze(0).expand(B, -1, -1, -1, -1).clone()
        else:
            bias_physics = edge_feats[..., :4].clone()

        if bias_physics.shape[2] < L_q:
            pad_q = L_q - bias_physics.shape[2]
            bp_pad = torch.zeros(
                *bias_physics.shape[:2], pad_q, bias_physics.shape[3], bias_physics.shape[4], device=device, dtype=dtype
            )
            bias_physics = torch.cat([bias_physics, bp_pad], dim=2)

        if bias_physics.shape[3] != key_count:
            if bias_physics.shape[3] > key_count:
                bias_physics = bias_physics[:, :, :, :key_count, :]
            else:
                key_pad = key_count - bias_physics.shape[3]
                pad_tensor = torch.zeros(*bias_physics.shape[:3], key_pad, bias_physics.shape[4], device=device, dtype=dtype)
                bias_physics = torch.cat([bias_physics, pad_tensor], dim=3)

        special_start = L
        if self.use_block_node and not self.query_has_block:
            bias_physics[:, :, :, special_start, :] = 0.0
            bias_physics[:, :, :, special_start, 3] = 1.0
            special_start += 1
        if self.use_matrix_node:
            bias_physics[:, :, :, special_start, :] = 0.0
            bias_physics[:, :, :, special_start, 3] = 1.0

        arange_s = torch.arange(L, device=device)
        bias_physics[:, :, arange_s, arange_s, :] = 0.0
        bias_physics[:, :, arange_s, arange_s, 3] = 1.0
        scores = scores + bias_physics[..., 3:4]

        if attn_mask.dim() == 3:
            mask_base = attn_mask.unsqueeze(0)
        else:
            mask_base = attn_mask

        if mask_base.shape[2] < L_q:
            pad_q = L_q - mask_base.shape[2]
            mb_pad = torch.ones(*mask_base.shape[:2], pad_q, mask_base.shape[3], device=device, dtype=dtype)
            mask_base = torch.cat([mask_base, mb_pad], dim=2)

        if mask_base.shape[3] != key_count:
            if mask_base.shape[3] > key_count:
                mask_base = mask_base[:, :, :, :key_count]
            else:
                key_pad = key_count - mask_base.shape[3]
                pad_tensor = torch.ones(*mask_base.shape[:3], key_pad, device=device, dtype=dtype)
                mask_base = torch.cat([mask_base, pad_tensor], dim=3)

        mask_expanded = mask_base.unsqueeze(-1)
        scores = scores.masked_fill(mask_expanded == 0, float("-inf"))

        if save_attention:
            with torch.no_grad():
                self.last_scores_matrix = scores.mean(dim=-1)[:, :, :, :L].cpu().float()
                self.last_bias_physics_matrix = bias_physics[:, :, :L].cpu().float()

        attn_probs = F.softmax(scores, dim=3)
        linear_edge_weights = self.edge_gate(bias_physics)
        linear_edge_weights = linear_edge_weights.masked_fill(mask_expanded == 0, 0.0)
        combined_weights = attn_probs + linear_edge_weights

        if not self.training and not torch.compiler.is_compiling():
            with torch.no_grad():
                attn_viz = combined_weights.mean(dim=-1)
                arange = torch.arange(L, device=attn_viz.device)
                self.last_attn_self = attn_viz[:, :, arange, arange].mean().item()
                if self.query_has_block and attn_viz.shape[2] > L:
                    self.last_attn_block = attn_viz[:, :, L, :].mean().item()
                elif self.use_block_node and attn_viz.shape[3] > L:
                    self.last_attn_block = attn_viz[:, :, :, L].mean().item()
                else:
                    self.last_attn_block = 0.0
                # Neighbor mass vs self: only for the L physical leaf queries (L_q may be L+1).
                attn_leaf_q = attn_viz[:, :, :L, :]
                to_nodes = attn_leaf_q[:, :, :, :L].sum(dim=3)
                self.last_attn_neighbor = (to_nodes - attn_viz[:, :, arange, arange]).mean().item()
                if save_attention:
                    self.last_attn_matrix = attn_viz[:, :, :, :L].cpu().float()

        x_out = torch.einsum("bnqkh,bnkhd->bnqhd", combined_weights, v)
        x_out = x_out.reshape(B, num_blocks, L_q, C)
        x_out = self.proj(x_out)
        x_out = x_out.reshape(B, N_pad, C)
        if node_pad > 0:
            x_out = x_out[:, :N, :]
        return x_out


class TransformerBlock(nn.Module):
    def __init__(self, dim, block_size, attn_module, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attn_module
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

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
        x = x + self.mlp(self.norm2(x))
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
        strip_build_mode: str = "no_einsum",
    ):
        super().__init__()
        if strip_build_mode not in ("einsum", "no_einsum"):
            raise ValueError(f"strip_build_mode must be 'einsum' or 'no_einsum', got {strip_build_mode!r}")
        self.strip_build_mode = strip_build_mode
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
        otp = int(OFF_DIAG_TOKEN_POOL)
        if L % otp != 0:
            raise ValueError(f"leaf_size {L} not divisible by OFF_DIAG_TOKEN_POOL {otp}")
        self.off_token_pool = otp
        Ls_off = L // otp
        self.leaf_apply_size = L
        self.leaf_apply_off = Ls_off
        La_d = L
        La_o = Ls_off
        off_attn_layout = f"{Ls_off}x{Ls_off + 1}"
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
        # Strip-pooled tokens are mean-pooled to Ls_off before this stack; attention + MLP stay at Ls_off (no upsample).
        self.off_diag_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    block_size=Ls_off,
                    attn_module=LeafBlockAttention(
                        d_model,
                        Ls_off,
                        num_heads=num_heads,
                        attention_layout=off_attn_layout,
                    ),
                )
                for _ in range(num_layers)
            ]
        )
        self.off_diag_head_U = nn.Linear(d_model, La_o)
        self.off_diag_head_V = nn.Linear(d_model, La_o)
        nn.init.normal_(self.off_diag_head_U.weight, std=0.001)
        nn.init.constant_(self.off_diag_head_U.bias, 0.0)
        nn.init.normal_(self.off_diag_head_V.weight, std=0.001)
        nn.init.constant_(self.off_diag_head_V.bias, 0.0)
        self.leaf_head = nn.Linear(d_model, La_d)
        nn.init.normal_(self.leaf_head.weight, std=0.001)
        nn.init.constant_(self.leaf_head.bias, 0.0)

        self.node_u = nn.Linear(d_model, La_o)
        self.node_v = nn.Linear(d_model, La_o)
        nn.init.normal_(self.node_u.weight, std=0.001)
        nn.init.constant_(self.node_u.bias, 0.0)
        nn.init.normal_(self.node_v.weight, std=0.001)
        nn.init.constant_(self.node_v.bias, 0.0)

        self.jacobi_gate = nn.Linear(d_model, 1)
        nn.init.normal_(self.jacobi_gate.weight, std=0.01)
        nn.init.constant_(self.jacobi_gate.bias, 0.0)
        self.use_jacobi = use_jacobi

        self.register_buffer("h_block_r0", HM_R0_CPU.clone(), persistent=False)
        self.register_buffer("h_block_c0", HM_C0_CPU.clone(), persistent=False)
        self.register_buffer("h_block_S", HM_S_CPU.clone(), persistent=False)
        self.num_h_off = int(self.h_block_r0.shape[0])
        wr, wc = hm_leaf_mean_pool_weights(
            self.h_block_r0.cpu(), self.h_block_c0.cpu(), self.h_block_S.cpu(), MAX_NUM_LEAVES
        )
        self.register_buffer("hm_pool_w_row", wr, persistent=False)
        self.register_buffer("hm_pool_w_col", wc, persistent=False)
        wr_sum, wc_sum = HM_SUM_W_ROW_CPU.clone(), HM_SUM_W_COL_CPU.clone()
        self.register_buffer("hm_sum_w_row", wr_sum, persistent=False)
        self.register_buffer("hm_sum_w_col", wc_sum, persistent=False)

        gnn_passes = max(1, 2 * int(math.ceil(math.log2(MAX_NUM_LEAVES))))
        self.tree_gnn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )
                for _ in range(gnn_passes)
            ]
        )
        self.tree_gnn_norm = nn.LayerNorm(d_model)
        self.register_buffer("hm_adj", HM_ADJ_CPU.clone(), persistent=False)
        self._legacy_no_gnn = False

    def _get_leaf_blocks(self, h, mode="diagonal"):
        # Diagonal: (B, N, C) with N = K * L. Off: (B, M_off, L, C) one L-token tile per H block — fold to (B*M, L, C).
        off_bm = None
        if mode == "off-diagonal" and h.dim() == 4:
            B0, M0, Ln, C = h.shape
            if Ln != self.leaf_apply_off:
                raise ValueError(f"off-diagonal h expected L={self.leaf_apply_off}, got {Ln}")
            off_bm = (B0, M0)
            h = h.reshape(B0 * M0, self.leaf_apply_off, C)
        Bf, N_or_P_len, _ = h.shape
        if mode == "diagonal":
            num_leaves = N_or_P_len // self.leaf_size
            h_leaves = h.view(Bf, num_leaves, self.leaf_size, -1)
            u_leaf = self.leaf_head(h_leaves)
            out = torch.matmul(u_leaf, u_leaf.transpose(-1, -2))
        else:
            num_tiles = N_or_P_len // self.leaf_apply_off
            h_leaves = h.view(Bf, num_tiles, self.leaf_apply_off, -1)
            u_leaf = self.off_diag_head_U(h_leaves)
            v_leaf = self.off_diag_head_V(h_leaves)
            out = torch.matmul(u_leaf, v_leaf.transpose(-1, -2))
        if off_bm is not None:
            B0, M0 = off_bm
            out = out.view(B0, M0, out.shape[-2], out.shape[-1])
        return out

    def _get_jacobi_scale(self, h):
        if not self.use_jacobi:
            return None
        B, N, C = h.shape
        num_leaves = N // self.leaf_size
        h_leaves = h.view(B, num_leaves, self.leaf_size, C)
        j_gate = torch.sigmoid(self.jacobi_gate(h_leaves)).squeeze(-1)
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
        off_om_pre, off_oe_pre = None, None

        if edge_index is not None and edge_values is not None:
            device, dtype = x.device, x.dtype
            if precomputed_leaf_connectivity is not None:
                pcs = precomputed_leaf_connectivity
                attn_mask, edge_feats = pcs[0], pcs[1]
                if len(pcs) >= 4:
                    off_om_pre, off_oe_pre = pcs[2], pcs[3]
                elif self.num_h_off > 0:
                    raise ValueError(
                        "precomputed_leaf_connectivity must be a 4-tuple "
                        "(diag_mask, diag_feats, off_attn_mask, off_edge_feats) when num_h_off > 0."
                    )
            else:
                attn_mask, edge_feats, off_om_pre, off_oe_pre = build_leaf_block_connectivity(
                    edge_index,
                    edge_values,
                    positions,
                    self.leaf_size,
                    device,
                    dtype,
                    num_hops=ATTENTION_HOPS,
                )

        B, N_h, C_h = h.shape
        L = self.leaf_size
        K = N_h // L
        if K != MAX_NUM_LEAVES:
            raise ValueError(
                f"LeafOnlyNet: expected N/leaf_size == MAX_NUM_LEAVES ({MAX_NUM_LEAVES}), got K={K} (N={N_h}). "
                "Pad/truncate to MAX_MIXED_SIZE nodes for a static H-grid."
            )

        off_attn_mask = off_edge_feats = None
        M_off = self.num_h_off
        if M_off > 0:
            # After strip sum + optional token mean-pool: (B*M_off, Ls_off, C), num_blocks=1; masks match Ls_off.
            if off_om_pre is None or off_om_pre.numel() == 0:
                raise ValueError(
                    "LeafOnlyNet: H off-diagonal attention mask missing. "
                    "Pass build_leaf_block_connectivity(...) (4-tuple) as precomputed_leaf_connectivity, "
                    "or provide edge_index/edge_values so it can be built."
                )
            if off_oe_pre is None or off_oe_pre.numel() == 0:
                raise ValueError(
                    "LeafOnlyNet: H off-diagonal edge_feats missing (precomputed[...,3] or build_leaf_block_connectivity)."
                )
            om_m = int(off_om_pre.shape[0]) if off_om_pre.dim() == 3 else int(off_om_pre.shape[1])
            if om_m != M_off:
                raise ValueError(
                    f"LeafOnlyNet: off attn_mask has M={om_m}, expected num_h_off={M_off}."
                )
            om = off_om_pre.to(device=h.device, dtype=h.dtype)
            oe = off_oe_pre.to(device=h.device, dtype=h.dtype)
            if self.off_token_pool > 1:
                om = pool_leaf_attn_mask(om, L, self.off_token_pool)
                oe = pool_leaf_edge_feats(oe, L, self.off_token_pool)
            if om.dim() == 3:
                # Keep expanded view strided; avoid .contiguous() (would materialize a full B×M copy each forward).
                om4 = om.unsqueeze(0).expand(B, M_off, -1, -1)
            else:
                om4 = om.contiguous()
                if om4.shape[0] == 1 and B > 1:
                    om4 = om4.expand(B, -1, -1, -1)
            Lq, Kq = om4.shape[-2], om4.shape[-1]
            if oe.dim() == 4:
                oe5 = oe.unsqueeze(0).expand(B, M_off, -1, -1, -1)
            else:
                oe5 = oe.contiguous()
                if oe5.shape[0] == 1 and B > 1:
                    oe5 = oe5.expand(B, -1, -1, -1, -1, -1)
            off_attn_mask = om4.reshape(B * M_off, 1, Lq, Kq)
            off_edge_feats = oe5.reshape(B * M_off, 1, Lq, Kq, 4)

        qhb = self.blocks[0].attn.query_has_block
        h_diag_view = h.view(B, K, L, C_h)

        if qhb:
            bn_diag = h_diag_view.mean(dim=2, keepdim=True)
            h_diag_in = torch.cat([h_diag_view, bn_diag], dim=2).view(B, K * (L + 1), C_h)
        else:
            h_diag_in = h.view(B, K * L, C_h)

        h_off_in = None
        if M_off > 0:
            if self.strip_build_mode == "einsum":
                Wr = self.hm_pool_w_row.to(device=h.device, dtype=h.dtype)
                Wc = self.hm_pool_w_col.to(device=h.device, dtype=h.dtype)
                row_p = torch.einsum("mk,bklc->bmlc", Wr, h_diag_view)
                col_p = torch.einsum("mk,bklc->bmlc", Wc, h_diag_view)
            else:
                ridx, cidx, gidx = _hm_prolong_scatter_indices(h.device)
                h_k_t = h_diag_view.transpose(0, 1)
                row_sum = torch.zeros(M_off, B, L, C_h, device=h.device, dtype=h.dtype)
                col_sum = torch.zeros(M_off, B, L, C_h, device=h.device, dtype=h.dtype)
                row_sum.index_add_(0, gidx, h_k_t[ridx])
                col_sum.index_add_(0, gidx, h_k_t[cidx])
                S_view = self.h_block_S.view(M_off, 1, 1, 1).to(device=h.device, dtype=h.dtype)
                row_p = (row_sum / S_view).transpose(0, 1)
                col_p = (col_sum / S_view).transpose(0, 1)

            h_off_base = (row_p + col_p).reshape(B * M_off, L, C_h)
            if self.off_token_pool > 1:
                Ls = self.leaf_apply_off
                h_off_base = h_off_base.view(B * M_off, Ls, self.off_token_pool, C_h).mean(dim=2)

            if qhb:
                h_off_view = h_off_base.view(B, M_off, self.leaf_apply_off, C_h)
                bn_off = h_off_view.mean(dim=2, keepdim=True)
                h_off_in = torch.cat([h_off_view, bn_off], dim=2).reshape(B * M_off, self.leaf_apply_off + 1, C_h)
            else:
                h_off_in = h_off_base

        if qhb:
            for i in range(len(self.blocks)):
                h_diag_in = self.blocks[i](
                    h_diag_in,
                    edge_index=edge_index,
                    edge_values=edge_values,
                    positions=positions,
                    save_attention=save_attention,
                    attn_mask=attn_mask,
                    edge_feats=edge_feats,
                )

                if M_off > 0:
                    h_off_in = self.off_diag_blocks[i](
                        h_off_in,
                        edge_index=edge_index,
                        edge_values=edge_values,
                        positions=positions,
                        save_attention=False,
                        attn_mask=off_attn_mask,
                        edge_feats=off_edge_feats,
                    )

                if i == 0 and not self._legacy_no_gnn:
                    h_d_v = h_diag_in.view(B, K, L + 1, C_h)
                    bnd = h_d_v[:, :, -1, :]
                    if M_off > 0:
                        h_o_v = h_off_in.view(B, M_off, self.leaf_apply_off + 1, C_h)
                        bno = h_o_v[:, :, -1, :]
                        V = torch.cat([bnd, bno], dim=1)
                    else:
                        V = bnd

                    adj = self.hm_adj.to(device=V.device, dtype=V.dtype)
                    for layer in self.tree_gnn_layers:
                        msg = torch.matmul(adj, V)
                        V = V + layer(msg)
                    V = self.tree_gnn_norm(V)

                    h_d_v = h_d_v.clone()
                    h_d_v[:, :, -1, :] = V[:, :K, :]
                    h_diag_in = h_d_v.view(B, K * (L + 1), C_h)

                    if M_off > 0:
                        h_o_v = h_o_v.clone()
                        h_o_v[:, :, -1, :] = V[:, K:, :]
                        h_off_in = h_o_v.reshape(B * M_off, self.leaf_apply_off + 1, C_h)

            h_diag = h_diag_in.view(B, K, L + 1, C_h)[:, :, :-1, :].reshape(B, K * L, C_h)
            diag_blocks = self._get_leaf_blocks(h_diag, mode="diagonal")
            if M_off > 0:
                h_off = h_off_in.view(B, M_off, self.leaf_apply_off + 1, C_h)[:, :, :-1, :].reshape(
                    B, M_off, self.leaf_apply_off, C_h
                )
                off_diag_blocks = self._get_leaf_blocks(h_off, mode="off-diagonal")
            else:
                off_diag_blocks = torch.empty(
                    (B, 0, self.leaf_apply_off, self.leaf_apply_off), device=h.device, dtype=h.dtype
                )
        else:
            for block in self.blocks:
                h_diag_in = block(
                    h_diag_in,
                    edge_index=edge_index,
                    edge_values=edge_values,
                    positions=positions,
                    save_attention=save_attention,
                    attn_mask=attn_mask,
                    edge_feats=edge_feats,
                )
            h_diag = h_diag_in
            diag_blocks = self._get_leaf_blocks(h_diag, mode="diagonal")
            if M_off > 0:
                for block in self.off_diag_blocks:
                    h_off_in = block(
                        h_off_in,
                        edge_index=edge_index,
                        edge_values=edge_values,
                        positions=positions,
                        save_attention=False,
                        attn_mask=off_attn_mask,
                        edge_feats=off_edge_feats,
                    )
                h_off = h_off_in.view(B, M_off, self.leaf_apply_off, C_h)
                off_diag_blocks = self._get_leaf_blocks(h_off, mode="off-diagonal")
            else:
                off_diag_blocks = torch.empty(
                    (B, 0, self.leaf_apply_off, self.leaf_apply_off), device=h.device, dtype=h.dtype
                )

        jacobi_scale = self._get_jacobi_scale(h_diag)

        node_U = self.node_u(h_diag)
        node_V = self.node_v(h_diag)

        Bb = diag_blocks.shape[0]
        packed_diag = diag_blocks.reshape(Bb, -1)
        packed_off = off_diag_blocks.reshape(Bb, -1)
        packed_U = node_U.reshape(Bb, -1)
        packed_V = node_V.reshape(Bb, -1)

        packed = torch.cat([packed_diag, packed_off, packed_U, packed_V], dim=1)
        if jacobi_scale is not None:
            packed = torch.cat([packed, jacobi_scale], dim=1)

        return packed


def unpack_precond(precond_packed, N, leaf_size=LEAF_SIZE, leaf_apply_size=None, leaf_apply_off=None):
    if leaf_apply_size is None:
        leaf_apply_size = leaf_size
    if leaf_apply_off is None:
        leaf_apply_off = leaf_apply_size
    B = precond_packed.shape[0]
    num_leaves = N // leaf_size
    La_d = leaf_apply_size
    La_o = leaf_apply_off
    diag_size = num_leaves * La_d * La_d
    M_h = NUM_HMATRIX_OFF_BLOCKS
    off_size = M_h * La_o * La_o
    node_size = num_leaves * leaf_size * La_o

    diag_blocks = precond_packed[:, :diag_size].view(B, num_leaves, La_d, La_d)
    off_diag_blocks = None
    idx = diag_size
    if M_h > 0:
        off_diag_blocks = precond_packed[:, idx : idx + off_size].view(B, M_h, La_o, La_o)
        idx += off_size
    node_U = precond_packed[:, idx : idx + node_size].view(B, num_leaves, leaf_size, La_o)
    idx += node_size
    node_V = precond_packed[:, idx : idx + node_size].view(B, num_leaves, leaf_size, La_o)
    idx += node_size

    jacobi_scale = precond_packed[:, idx:] if precond_packed.shape[1] > idx else None
    return diag_blocks, off_diag_blocks, node_U, node_V, jacobi_scale


def build_sparse_bsr_preconditioner(
    precond_packed,
    N,
    leaf_size,
    leaf_apply_size,
    leaf_apply_off,
    jacobi_inv_diag,
    device,
):
    """
    Assemble H-matrix diagonal/off blocks (unpooled), fuse optional Jacobi scaling into diagonals,
    return BSR (block size ``leaf_size``) for block SpMV ``z = M @ r``.

    Per ``(N, L, La_d, La_o, device)`` the first call builds COO, runs ``coalesce`` + ``to_sparse_bsr``
    once on dummy values to cache crow/col indices and the value permutation; later calls only
    ``all_values[perm]`` and ``sparse_bsr_tensor``. Uses padded forward packing (``diag_size_full`` gap),
    same as ``apply_block_diagonal_M_physical``.
    """
    if precond_packed.dim() == 1:
        precond_packed = precond_packed.unsqueeze(0)
    dtype = precond_packed.dtype
    precond_packed = precond_packed.to(device=device, dtype=dtype).contiguous()

    L = int(leaf_size)
    num_leaves = N // L
    if num_leaves * L != N:
        raise ValueError(f"build_sparse_bsr_preconditioner: N={N} not divisible by leaf_size={L}")
    La_d = int(leaf_apply_size)
    La_o = int(leaf_apply_off)
    pool_diag = L // La_d
    pool_off = L // La_o
    if pool_diag * La_d != L:
        raise ValueError(f"leaf_size {L} not divisible by leaf_apply_size {La_d}")
    if pool_off * La_o != L:
        raise ValueError(f"leaf_size {L} not divisible by leaf_apply_off {La_o}")

    diag_size_active = num_leaves * La_d * La_d
    diag_size_full = MAX_NUM_LEAVES * La_d * La_d
    M_h = NUM_HMATRIX_OFF_BLOCKS
    off_size = M_h * La_o * La_o
    node_size_full = MAX_NUM_LEAVES * L * La_o

    diag_blocks = precond_packed[:, :diag_size_active].view(1, num_leaves, La_d, La_d)

    off_diag_blocks = node_U = node_V = None
    idx = diag_size_full
    if M_h > 0:
        off_diag_blocks = precond_packed[:, idx : idx + off_size].view(1, M_h, La_o, La_o)
        idx += off_size
    node_U = precond_packed[:, idx : idx + node_size_full].view(1, MAX_NUM_LEAVES, L, La_o)
    idx += node_size_full
    node_V = precond_packed[:, idx : idx + node_size_full].view(1, MAX_NUM_LEAVES, L, La_o)
    idx += node_size_full

    jacobi_scale = precond_packed[:, idx :][:, :N] if precond_packed.shape[1] > idx else None

    if jacobi_inv_diag.dim() == 1:
        jinv = jacobi_inv_diag[:N].to(device=device, dtype=dtype)
    else:
        jinv = jacobi_inv_diag[0, :N].to(device=device, dtype=dtype)

    diag_b = diag_blocks[0, :num_leaves].clone()
    if pool_diag > 1:
        diag_b = diag_b.repeat_interleave(pool_diag, dim=1).repeat_interleave(pool_diag, dim=2)
        # repeat_interleave duplicates rows/cols like sum-pooling x; divide so SpMV matches mean-pool + matmul.
        diag_b = diag_b / float(pool_diag)

    if jacobi_scale is not None:
        j_scale = (jacobi_scale[0, :N] * jinv).view(num_leaves, L)
        diag_b = diag_b + torch.diag_embed(j_scale)

    indices_list = []
    values_list = []

    row_idx = torch.arange(N, device=device, dtype=torch.long).view(num_leaves, L, 1).expand(-1, -1, L)
    col_idx = torch.arange(N, device=device, dtype=torch.long).view(num_leaves, 1, L).expand(-1, L, -1)
    indices_list.append(torch.stack([row_idx.reshape(-1), col_idx.reshape(-1)], dim=0))
    values_list.append(diag_b.reshape(-1))

    if off_diag_blocks is not None and M_h > 0:
        for i in range(M_h):
            r0i = int(HM_R0_CPU[i].item())
            c0i = int(HM_C0_CPU[i].item())
            si = int(HM_S_CPU[i].item())
            if r0i + si > num_leaves or c0i + si > num_leaves:
                continue

            C_m = off_diag_blocks[0, i]
            U_strip = node_U[0, r0i : r0i + si].reshape(si * L, La_o)
            V_strip = node_V[0, c0i : c0i + si].reshape(si * L, La_o)

            U_C = torch.matmul(U_strip, C_m)
            oblk_dense = torch.matmul(U_C, V_strip.transpose(0, 1))

            S_px = si * L
            br0, bc0 = r0i * L, c0i * L

            r_idx_u = torch.arange(br0, br0 + S_px, device=device, dtype=torch.long).view(-1, 1).expand(-1, S_px)
            c_idx_u = torch.arange(bc0, bc0 + S_px, device=device, dtype=torch.long).view(1, -1).expand(S_px, -1)
            indices_list.append(torch.stack([r_idx_u.reshape(-1), c_idx_u.reshape(-1)], dim=0))
            values_list.append(oblk_dense.reshape(-1))

            oblk_t_dense = oblk_dense.t()
            r_idx_l = torch.arange(bc0, bc0 + S_px, device=device, dtype=torch.long).view(-1, 1).expand(-1, S_px)
            c_idx_l = torch.arange(br0, br0 + S_px, device=device, dtype=torch.long).view(1, -1).expand(S_px, -1)
            indices_list.append(torch.stack([r_idx_l.reshape(-1), c_idx_l.reshape(-1)], dim=0))
            values_list.append(oblk_t_dense.reshape(-1))

    all_indices = torch.cat(indices_list, dim=1)
    all_values = torch.cat(values_list, dim=0)

    global _BSR_TOPOLOGY_CACHE
    cache_key = (N, L, La_d, La_o, str(device))

    if cache_key not in _BSR_TOPOLOGY_CACHE:
        E = int(all_indices.shape[1])
        dummy_vals = torch.arange(E, device=device, dtype=torch.float64)
        dummy_coo = torch.sparse_coo_tensor(all_indices, dummy_vals, (N, N), device=device).coalesce()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dummy_bsr = dummy_coo.to_sparse_bsr(blocksize=(L, L))
        perm_src = dummy_bsr.values().reshape(-1)
        _BSR_TOPOLOGY_CACHE[cache_key] = {
            "crow": dummy_bsr.crow_indices().clone(),
            "col": dummy_bsr.col_indices().clone(),
            "perm": perm_src.long().clone(),
        }

    topo = _BSR_TOPOLOGY_CACHE[cache_key]
    bsr_flat = all_values[topo["perm"]]
    nnz_blk = bsr_flat.numel() // (L * L)
    if nnz_blk * L * L != bsr_flat.numel():
        raise ValueError(
            f"BSR value count {bsr_flat.numel()} not divisible by L*L={L * L} (N={N}, cache_key={cache_key})"
        )
    if int(topo["col"].numel()) != nnz_blk:
        raise ValueError("BSR crow/col nnz blocks mismatch vs permuted value length")
    # CUDA sparse BSR matmul asserts values.dim() == 3 (nnz, L, L); 1D values hit SparseBlasImpl.
    bsr_values = bsr_flat.reshape(nnz_blk, L, L)
    return torch.sparse_bsr_tensor(
        topo["crow"],
        topo["col"],
        bsr_values,
        size=(N, N),
        dtype=dtype,
        device=device,
    )


def apply_block_diagonal_M(
    precond_packed,
    x,
    leaf_size=LEAF_SIZE,
    leaf_apply_size=None,
    leaf_apply_off=None,
    jacobi_inv_diag=None,
):
    """
    Diagonal blocks are La_d×La_d; off-diagonal H tiles use FMM form U C V^T with coarse C (La_o×La_o);
    node U,V are per-node (leaf × La_o); Jacobi scaling optional at end of packed layout.
    """
    if leaf_apply_size is None:
        leaf_apply_size = leaf_size
    if leaf_apply_off is None:
        leaf_apply_off = leaf_apply_size
    B, N, K_dim = x.shape
    num_leaves = N // leaf_size
    La_d = leaf_apply_size
    La_o = leaf_apply_off
    pool_d = leaf_size // La_d
    pool_o = leaf_size // La_o
    if pool_d * La_d != leaf_size:
        raise ValueError(f"leaf_size {leaf_size} not divisible by leaf_apply_size {La_d}")
    if pool_o * La_o != leaf_size:
        raise ValueError(f"leaf_size {leaf_size} not divisible by leaf_apply_off {La_o}")

    diag_size = num_leaves * La_d * La_d
    M_h = NUM_HMATRIX_OFF_BLOCKS
    off_size = M_h * La_o * La_o

    diag_blocks = precond_packed[:, :diag_size].view(B, num_leaves, La_d, La_d)
    x_leaves = x.view(B, num_leaves, leaf_size, K_dim)
    x_pool_d = (
        x_leaves
        if pool_d == 1
        else x_leaves.view(B, num_leaves, La_d, pool_d, K_dim).mean(dim=3)
    )

    y_pool = torch.matmul(diag_blocks, x_pool_d)
    y_leaves = y_pool if pool_d == 1 else y_pool.repeat_interleave(pool_d, dim=2)

    idx = diag_size
    node_size = num_leaves * leaf_size * La_o
    if M_h > 0:
        off_diag_blocks = precond_packed[:, idx : idx + off_size].view(B, M_h, La_o, La_o)
        idx += off_size
        node_U = precond_packed[:, idx : idx + node_size].view(B, num_leaves, leaf_size, La_o)
        idx += node_size
        node_V = precond_packed[:, idx : idx + node_size].view(B, num_leaves, leaf_size, La_o)
        idx += node_size

        Wr_sum = HM_SUM_W_ROW_CPU.to(device=x.device, dtype=x.dtype)
        Wc_sum = HM_SUM_W_COL_CPU.to(device=x.device, dtype=x.dtype)

        x_proj_U = torch.matmul(node_U.transpose(-1, -2), x_leaves)
        x_proj_V = torch.matmul(node_V.transpose(-1, -2), x_leaves)

        x_row_strips = torch.einsum("mk,bkld->bmld", Wr_sum, x_proj_U)
        x_col_strips = torch.einsum("mk,bkld->bmld", Wc_sum, x_proj_V)

        flat = B * M_h
        Br = off_diag_blocks.reshape(flat, La_o, La_o)
        xr = x_row_strips.reshape(flat, La_o, K_dim)
        xc = x_col_strips.reshape(flat, La_o, K_dim)
        y_r_coarse = torch.bmm(Br, xc).view(B, M_h, La_o, K_dim)
        y_c_coarse = torch.bmm(Br.transpose(-1, -2), xr).view(B, M_h, La_o, K_dim)

        ridx, cidx, gidx = _hm_prolong_scatter_indices(x.device)
        y_r_leaves = torch.zeros(B, num_leaves, La_o, K_dim, device=x.device, dtype=x.dtype)
        y_c_leaves = torch.zeros(B, num_leaves, La_o, K_dim, device=x.device, dtype=x.dtype)

        y_r_leaves.index_add_(1, ridx, y_r_coarse[:, gidx, :, :])
        y_c_leaves.index_add_(1, cidx, y_c_coarse[:, gidx, :, :])

        y_r_final = torch.matmul(node_U, y_r_leaves)
        y_c_final = torch.matmul(node_V, y_c_leaves)

        y_leaves = y_leaves + y_r_final + y_c_final
    else:
        idx += 2 * node_size

    y = y_leaves.view(B, N, K_dim)

    if precond_packed.shape[1] > idx:
        if jacobi_inv_diag is None:
            raise ValueError("jacobi_inv_diag is required when jacobi_scale is present.")
        if jacobi_inv_diag.shape[:2] != (B, N):
            raise ValueError(f"jacobi_inv_diag shape {jacobi_inv_diag.shape} does not match (B,N)=({B},{N}).")
        jacobi_scale = precond_packed[:, idx:]
        y = y + (jacobi_scale * jacobi_inv_diag).unsqueeze(-1) * x
    return y


def apply_block_diagonal_M_physical(
    precond_packed,
    x,
    leaf_size=LEAF_SIZE,
    leaf_apply_size=None,
    leaf_apply_off=None,
    jacobi_inv_diag=None,
):
    """
    Apply the packed LeafOnly preconditioner to a **physical** residual (no padded tail).

    ``x`` has shape ``(B, N_active, K_dim)`` with ``N_active = num_leaves_active * leaf_size`` and
    ``num_leaves_active <= MAX_NUM_LEAVES``. ``precond_packed`` is the usual forward output built on
    the full ``MAX_MIXED_SIZE`` grid.

    Diagonal blocks for the first ``num_leaves_active`` leaves match the full packed layout.
    Off-diagonal H tiles that are not **fully** inside the active leaf index range are skipped
    (zeroed coefficients) so pooling and scatter never read past the active leaves.
    """
    if leaf_apply_size is None:
        leaf_apply_size = leaf_size
    if leaf_apply_off is None:
        leaf_apply_off = leaf_apply_size
    B, N, K_dim = x.shape
    L = int(leaf_size)
    if N % L != 0:
        raise ValueError(f"apply_block_diagonal_M_physical: N={N} not divisible by leaf_size={L}")
    num_leaves_active = N // L
    if num_leaves_active > MAX_NUM_LEAVES:
        raise ValueError(
            f"num_leaves_active={num_leaves_active} exceeds MAX_NUM_LEAVES={MAX_NUM_LEAVES}"
        )
    La_d = leaf_apply_size
    La_o = leaf_apply_off
    pool_d = L // La_d
    pool_o = L // La_o
    if pool_d * La_d != L:
        raise ValueError(f"leaf_size {L} not divisible by leaf_apply_size {La_d}")
    if pool_o * La_o != L:
        raise ValueError(f"leaf_size {L} not divisible by leaf_apply_off {La_o}")

    diag_size_active = num_leaves_active * La_d * La_d
    diag_size_full = MAX_NUM_LEAVES * La_d * La_d
    M_h = NUM_HMATRIX_OFF_BLOCKS
    off_size = M_h * La_o * La_o

    diag_blocks = precond_packed[:, :diag_size_active].view(B, num_leaves_active, La_d, La_d)
    x_leaves = x.view(B, num_leaves_active, L, K_dim)
    x_pool_d = (
        x_leaves
        if pool_d == 1
        else x_leaves.view(B, num_leaves_active, La_d, pool_d, K_dim).mean(dim=3)
    )
    y_pool = torch.matmul(diag_blocks, x_pool_d)
    y_leaves = y_pool if pool_d == 1 else y_pool.repeat_interleave(pool_d, dim=2)

    node_size_full = MAX_NUM_LEAVES * L * La_o
    idx = diag_size_full
    if M_h > 0:
        off_diag_blocks = precond_packed[:, idx : idx + off_size].view(B, M_h, La_o, La_o)
        idx += off_size
        node_U_full = precond_packed[:, idx : idx + node_size_full].view(B, MAX_NUM_LEAVES, L, La_o)
        idx += node_size_full
        node_V_full = precond_packed[:, idx : idx + node_size_full].view(B, MAX_NUM_LEAVES, L, La_o)
        idx += node_size_full

        na = int(num_leaves_active)
        valid = ((HM_R0_CPU + HM_S_CPU) <= na) & ((HM_C0_CPU + HM_S_CPU) <= na)
        valid = valid.to(device=off_diag_blocks.device, dtype=off_diag_blocks.dtype).view(1, M_h, 1, 1)
        off_diag_blocks = off_diag_blocks * valid

        node_U_active = node_U_full[:, :na]
        node_V_active = node_V_full[:, :na]

        x_proj_U = torch.matmul(node_U_active.transpose(-1, -2), x_leaves)
        x_proj_V = torch.matmul(node_V_active.transpose(-1, -2), x_leaves)

        x_full_U = x.new_zeros(B, MAX_NUM_LEAVES, La_o, K_dim)
        x_full_V = x.new_zeros(B, MAX_NUM_LEAVES, La_o, K_dim)
        x_full_U[:, :na] = x_proj_U
        x_full_V[:, :na] = x_proj_V

        Wr_sum = HM_SUM_W_ROW_CPU.to(device=x.device, dtype=x.dtype)
        Wc_sum = HM_SUM_W_COL_CPU.to(device=x.device, dtype=x.dtype)
        x_row_strips = torch.einsum("mk,bkld->bmld", Wr_sum, x_full_U)
        x_col_strips = torch.einsum("mk,bkld->bmld", Wc_sum, x_full_V)

        flat = B * M_h
        Br = off_diag_blocks.reshape(flat, La_o, La_o)
        xr = x_row_strips.reshape(flat, La_o, K_dim)
        xc = x_col_strips.reshape(flat, La_o, K_dim)

        y_r_coarse = torch.bmm(Br, xc).view(B, M_h, La_o, K_dim)
        y_c_coarse = torch.bmm(Br.transpose(-1, -2), xr).view(B, M_h, La_o, K_dim)

        ridx, cidx, gidx = _hm_prolong_scatter_indices(x.device)
        y_r_full = x.new_zeros(B, MAX_NUM_LEAVES, La_o, K_dim)
        y_c_full = x.new_zeros(B, MAX_NUM_LEAVES, La_o, K_dim)

        y_r_full.index_add_(1, ridx, y_r_coarse[:, gidx, :, :])
        y_c_full.index_add_(1, cidx, y_c_coarse[:, gidx, :, :])

        y_r_active = y_r_full[:, :na]
        y_c_active = y_c_full[:, :na]

        y_r_final = torch.matmul(node_U_active, y_r_active)
        y_c_final = torch.matmul(node_V_active, y_c_active)

        y_leaves.copy_(y_leaves + y_r_final + y_c_final)
    else:
        idx += 2 * node_size_full

    y = y_leaves.view(B, N, K_dim)

    if precond_packed.shape[1] > idx:
        if jacobi_inv_diag is None:
            raise ValueError("jacobi_inv_diag is required when jacobi_scale is present.")
        if jacobi_inv_diag.shape[:2] != (B, N):
            raise ValueError(f"jacobi_inv_diag shape {jacobi_inv_diag.shape} does not match (B,N)=({B},{N}).")
        jacobi_scale = precond_packed[:, idx:][:, :N]
        y = y + (jacobi_scale * jacobi_inv_diag).unsqueeze(-1) * x
    return y


def block_diagonal_m_apply_workspace(num_leaves: int, leaf_size: int, K_dim: int, M_h: int, La_o: int, device, dtype):
    la_c = La_o * K_dim
    t_pro = int(HM_PROLONG_GATHER_IDX.shape[0])
    n_flat = MAX_NUM_LEAVES if M_h > 0 else num_leaves
    out = {
        "x_proj_U_flat": torch.empty(n_flat, la_c, device=device, dtype=dtype),
        "x_proj_V_flat": torch.empty(n_flat, la_c, device=device, dtype=dtype),
        "row_flat": torch.empty(M_h, la_c, device=device, dtype=dtype),
        "col_flat": torch.empty(M_h, la_c, device=device, dtype=dtype),
        "y_r": torch.empty(1, M_h, La_o, K_dim, device=device, dtype=dtype),
        "y_c": torch.empty(1, M_h, La_o, K_dim, device=device, dtype=dtype),
        "y_scatter_r": torch.empty(1, t_pro, La_o, K_dim, device=device, dtype=dtype),
        "y_scatter_c": torch.empty(1, t_pro, La_o, K_dim, device=device, dtype=dtype),
        "x_gather_r": torch.empty(t_pro, la_c, device=device, dtype=dtype),
        "x_gather_c": torch.empty(t_pro, la_c, device=device, dtype=dtype),
    }
    if M_h > 0:
        out["y_r_leaves_pad"] = torch.empty(1, MAX_NUM_LEAVES, La_o, K_dim, device=device, dtype=dtype)
        out["y_c_leaves_pad"] = torch.empty(1, MAX_NUM_LEAVES, La_o, K_dim, device=device, dtype=dtype)
        if num_leaves < MAX_NUM_LEAVES:
            na = int(num_leaves)
            valid = ((HM_R0_CPU + HM_S_CPU) <= na) & ((HM_C0_CPU + HM_S_CPU) <= na)
            out["off_valid"] = valid.to(device=device, dtype=dtype).view(1, M_h, 1, 1)
    return out


def apply_block_diagonal_m_into(
    precond_packed,
    x,
    out,
    jacobi_inv_diag,
    ws,
    leaf_size=LEAF_SIZE,
    leaf_apply_size=None,
    leaf_apply_off=None,
):
    """
    B=1 only. Writes Mx into ``out`` (same shape as ``x``) using preallocated ``ws``.
    FMM off-diagonal: sum-pooled moments, coarse BMM, scatter, node re-projection.
    No internal tensor allocations — safe inside ``torch.cuda.graph`` replay when buffers are fixed.
    Requires ``leaf_apply_size == leaf_size`` (full diagonal blocks). ``leaf_apply_off`` may be smaller
    (H off-tile rank / pooled token count); must divide ``leaf_size``.
    """
    if leaf_apply_size is None:
        leaf_apply_size = leaf_size
    if leaf_apply_off is None:
        leaf_apply_off = leaf_apply_size
    if leaf_apply_size != leaf_size:
        raise ValueError("apply_block_diagonal_m_into expects leaf_apply_size == leaf_size.")
    La_o_chk = int(leaf_apply_off)
    if leaf_size % La_o_chk != 0:
        raise ValueError(f"leaf_size {leaf_size} not divisible by leaf_apply_off {La_o_chk}")
    B, N, K_dim = x.shape
    if B != 1:
        raise ValueError(f"apply_block_diagonal_m_into expects B=1, got {B}")
    num_leaves = N // leaf_size
    La_d, La_o = leaf_apply_size, leaf_apply_off
    M_h = NUM_HMATRIX_OFF_BLOCKS

    diag_size_active = num_leaves * La_d * La_d
    diag_size_full = MAX_NUM_LEAVES * La_d * La_d
    off_size = M_h * La_o * La_o
    node_size_full = MAX_NUM_LEAVES * leaf_size * La_o

    diag_blocks = precond_packed[:, :diag_size_active].view(B, num_leaves, La_d, La_d)
    x_leaves = x.view(B, num_leaves, leaf_size, K_dim)
    y_leaves = out.view(B, num_leaves, leaf_size, K_dim)
    torch.matmul(diag_blocks, x_leaves, out=y_leaves)

    idx = diag_size_full
    if M_h > 0:
        off_diag_blocks = precond_packed[:, idx : idx + off_size].view(B, M_h, La_o, La_o)
        idx += off_size
        node_U = precond_packed[:, idx : idx + node_size_full].view(B, MAX_NUM_LEAVES, leaf_size, La_o)
        idx += node_size_full
        node_V = precond_packed[:, idx : idx + node_size_full].view(B, MAX_NUM_LEAVES, leaf_size, La_o)
        idx += node_size_full

        na = int(num_leaves)
        if na < MAX_NUM_LEAVES:
            off_diag_blocks = off_diag_blocks * ws["off_valid"]

        node_U_active = node_U[:, :na]
        node_V_active = node_V[:, :na]

        x_proj_U = torch.matmul(node_U_active.transpose(-1, -2), x_leaves)
        x_proj_V = torch.matmul(node_V_active.transpose(-1, -2), x_leaves)

        la_c = La_o * K_dim
        ws["x_proj_U_flat"].zero_()
        ws["x_proj_V_flat"].zero_()
        ws["x_proj_U_flat"][:na].copy_(x_proj_U[0].reshape(na, la_c))
        ws["x_proj_V_flat"][:na].copy_(x_proj_V[0].reshape(na, la_c))

        ridx, cidx, gidx = _hm_prolong_scatter_indices(x.device)

        torch.index_select(ws["x_proj_U_flat"], 0, ridx, out=ws["x_gather_r"])
        ws["row_flat"].zero_()
        ws["row_flat"].index_add_(0, gidx, ws["x_gather_r"])

        torch.index_select(ws["x_proj_V_flat"], 0, cidx, out=ws["x_gather_c"])
        ws["col_flat"].zero_()
        ws["col_flat"].index_add_(0, gidx, ws["x_gather_c"])

        Br = off_diag_blocks.reshape(M_h, La_o, La_o)
        xc = ws["col_flat"].view(M_h, La_o, K_dim)
        xr = ws["row_flat"].view(M_h, La_o, K_dim)

        torch.bmm(Br, xc, out=ws["y_r"][0])
        torch.bmm(Br.transpose(-1, -2), xr, out=ws["y_c"][0])

        torch.index_select(ws["y_r"], 1, gidx, out=ws["y_scatter_r"])
        torch.index_select(ws["y_c"], 1, gidx, out=ws["y_scatter_c"])

        y_r_pad = ws["y_r_leaves_pad"]
        y_c_pad = ws["y_c_leaves_pad"]
        y_r_pad.zero_()
        y_c_pad.zero_()

        y_r_pad.index_add_(1, ridx, ws["y_scatter_r"])
        y_c_pad.index_add_(1, cidx, ws["y_scatter_c"])

        y_r_final = torch.matmul(node_U_active, y_r_pad[:, :na])
        y_c_final = torch.matmul(node_V_active, y_c_pad[:, :na])
        out.add_(y_r_final.view(1, N, K_dim) + y_c_final.view(1, N, K_dim))
    else:
        idx += 2 * node_size_full

    if precond_packed.shape[1] > idx:
        if jacobi_inv_diag is None:
            raise ValueError("jacobi_inv_diag is required when jacobi_scale is present.")
        if jacobi_inv_diag.shape[:2] != (B, N):
            raise ValueError(f"jacobi_inv_diag shape {jacobi_inv_diag.shape} does not match (B,N)=({B},{N}).")
        jacobi_scale = precond_packed[:, idx:][:, :N]
        out.addcmul_(x, (jacobi_scale * jacobi_inv_diag).unsqueeze(-1))
    return out
