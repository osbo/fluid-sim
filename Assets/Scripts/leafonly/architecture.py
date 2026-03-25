from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    ATTN_POOL_FACTOR_DIAG,
    ATTN_POOL_FACTOR_OFF,
    ATTENTION_HOPS,
    GLOBAL_FEATURES_DIM,
    LEAF_SIZE,
    MAX_NUM_LEAVES,
)
from .data import build_leaf_block_connectivity
from .hmatrix import (
    HM_C0_CPU,
    HM_POOL_W_COL_CPU,
    HM_POOL_W_ROW_CPU,
    HM_PROLONG_COL_LEAF_IDX,
    HM_PROLONG_GATHER_IDX,
    HM_PROLONG_ROW_LEAF_IDX,
    HM_R0_CPU,
    HM_S_CPU,
    NUM_HMATRIX_OFF_BLOCKS,
    hm_leaf_mean_pool_weights,
)

# Prolongation indices live on CPU in hmatrix; copy once per device. CUDAGraph capture must not call .to().
_HM_PROLONG_GPU: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


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
    """If merge and t is (B, nb, ...), flatten to (B*nb, ...)."""
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
    """2× (or `pool`×) or-pool on node×node slice; or-pool along query for special key columns.
    Supports (nb, L, K) or batched (B, nb, L, K)."""
    if pool <= 1:
        return attn_mask
    L = int(leaf_size)
    assert L % pool == 0, f"leaf_size {L} not divisible by attn pool {pool}"
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
    """Mean-pool masks; supports (nb, L, K, E) or (B, nb, L, K, E)."""
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


def mean_pool_leaf_tokens(x_leaves: torch.Tensor, leaf_size: int, pool: int) -> torch.Tensor:
    """(B, K, L, C) -> (B, K, L//pool, C) by averaging adjacent nodes within each leaf."""
    if pool <= 1:
        return x_leaves
    L = int(leaf_size)
    La = L // pool
    B, K, Ln, C = x_leaves.shape
    assert Ln == L
    return x_leaves.view(B, K, La, pool, C).mean(dim=3)


def pool_precomputed_leaf_connectivity(precomputed, leaf_size: int, pool: int):
    """Apply mean/amax pooling to cached (diag_mask, diag_feats, off_mask, off_feats). No-op if pool <= 1."""
    if pool <= 1 or precomputed[0] is None:
        return precomputed
    attn_mask, edge_feats, off_attn_mask, off_edge_feats = precomputed
    attn_mask = pool_leaf_attn_mask(attn_mask, leaf_size, pool)
    edge_feats = pool_leaf_edge_feats(edge_feats, leaf_size, pool)
    if off_attn_mask is not None and off_attn_mask.numel() > 0:
        off_attn_mask = pool_leaf_attn_mask(off_attn_mask, leaf_size, pool)
        off_edge_feats = pool_leaf_edge_feats(off_edge_feats, leaf_size, pool)
    return (attn_mask, edge_feats, off_attn_mask, off_edge_feats)


def pool_off_leaf_connectivity_if_needed(off_attn_mask, off_edge_feats, leaf_size: int, pool: int):
    """Pool off-diagonal masks from full leaf resolution when off-diagonal attention uses pool>1."""
    if pool <= 1 or off_attn_mask is None or off_edge_feats is None or off_attn_mask.numel() == 0:
        return off_attn_mask, off_edge_feats
    q = off_attn_mask.shape[2] if off_attn_mask.dim() == 4 else off_attn_mask.shape[1]
    if q != leaf_size:
        return off_attn_mask, off_edge_feats
    return (
        pool_leaf_attn_mask(off_attn_mask, leaf_size, pool),
        pool_leaf_edge_feats(off_edge_feats, leaf_size, pool),
    )


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
    def __init__(self, dim, block_size, num_heads=8, attention_layout=None, attn_pool_factor: int = 1):
        super().__init__()
        self.dim = dim
        self.block_size = int(block_size)
        self.attn_pool_factor = int(attn_pool_factor)
        if self.attn_pool_factor < 1:
            raise ValueError("attn_pool_factor must be >= 1")
        if self.block_size % self.attn_pool_factor != 0:
            raise ValueError(f"block_size {self.block_size} must be divisible by attn_pool_factor {self.attn_pool_factor}")
        self.attn_block_size = self.block_size // self.attn_pool_factor
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
        L_full = self.block_size
        Ls = self.attn_block_size
        pool = self.attn_pool_factor
        node_pad = 0
        if N % L_full != 0:
            node_pad = L_full - (N % L_full)
            x = F.pad(x, (0, 0, 0, node_pad))
        N_pad = x.shape[1]
        num_blocks = N_pad // L_full
        x_blk = x.view(B, num_blocks, L_full, C)

        if attn_mask is None:
            if edge_feats is not None:
                raise ValueError("LeafBlockAttention: edge_feats set but attn_mask is None.")
            return x[:, :N, :] if node_pad > 0 else x
        if edge_feats is None:
            raise ValueError("LeafBlockAttention requires edge_feats when attn_mask is provided.")

        if pool > 1:
            x_attn = x_blk.view(B, num_blocks, Ls, pool, C).mean(dim=3)
        else:
            x_attn = x_blk

        use_block_node = self.use_block_node
        use_matrix_node = self.use_matrix_node
        kv_parts = [x_attn]
        if use_block_node:
            block_node = x_attn.mean(dim=2, keepdim=True)
            kv_parts.append(block_node)
        if use_matrix_node:
            matrix_node = x.mean(dim=1, keepdim=True).unsqueeze(1).expand(-1, num_blocks, -1, -1)
            kv_parts.append(matrix_node)
        kv = torch.cat(kv_parts, dim=2)
        qkv_q = self.qkv(x_attn)
        qkv_kv = self.qkv(kv)
        q = qkv_q[..., :C]
        k = qkv_kv[..., C:2 * C]
        v = qkv_kv[..., 2 * C:3 * C]
        key_count = kv.shape[2]
        q = q.view(B, num_blocks, Ls, self.num_heads, self.head_dim)
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
        special_start = Ls
        if use_block_node:
            bias_physics[:, :, :, special_start, :] = 0.0
            bias_physics[:, :, :, special_start, 3] = 1.0
            special_start += 1
        if use_matrix_node:
            bias_physics[:, :, :, special_start, :] = 0.0
            bias_physics[:, :, :, special_start, 3] = 1.0
        arange_s = torch.arange(Ls, device=device)
        bias_physics[:, :, arange_s, arange_s, :] = 0.0
        bias_physics[:, :, arange_s, arange_s, 3] = 1.0
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
                self.last_scores_matrix = scores.mean(dim=-1)[:, :, :, :Ls].cpu().float()
                self.last_bias_physics_matrix = bias_physics[:, :, :Ls].cpu().float()

        attn_probs = F.softmax(scores, dim=3)
        linear_edge_weights = self.edge_gate(bias_physics)
        linear_edge_weights = linear_edge_weights.masked_fill(mask_expanded == 0, 0.0)
        combined_weights = attn_probs + linear_edge_weights

        if not self.training and not torch.compiler.is_compiling():
            with torch.no_grad():
                attn_viz = combined_weights.mean(dim=-1)
                arange = torch.arange(Ls, device=attn_viz.device)
                self.last_attn_self = attn_viz[:, :, arange, arange].mean().item()
                if use_block_node and attn_viz.shape[3] > Ls:
                    self.last_attn_block = attn_viz[:, :, :, Ls].mean().item()
                else:
                    self.last_attn_block = 0.0
                to_nodes = attn_viz[:, :, :, :Ls].sum(dim=3)
                self.last_attn_neighbor = (to_nodes - attn_viz[:, :, arange, arange]).mean().item()
                if save_attention:
                    self.last_attn_matrix = attn_viz[:, :, :, :Ls].cpu().float()

        x_out = torch.einsum("bnqkh,bnkhd->bnqhd", combined_weights, v)
        x_out = x_out.reshape(B, num_blocks, Ls, C)
        # Proj at length Ls then expand: same as expand-then-proj when upsample is repeat, fewer matmuls.
        x_out = self.proj(x_out)
        if pool > 1:
            x_out = x_out.repeat_interleave(pool, dim=2)
        x_out = x_out.reshape(B, N_pad, C)
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
        # H-matrix stack uses fixed L×(L+1) attention (block node); required for static compile.
        attention_layout = f"{self.leaf_size}x{self.leaf_size + 1}"
        self.embed = PhysicsAwareEmbedding(
            input_dim,
            d_model,
            use_gcn=use_gcn,
            global_features_dim=GLOBAL_FEATURES_DIM,
            num_gcn_layers=num_gcn_layers,
        )
        self.enc_input_proj = nn.Linear(d_model, d_model)
        L = self.leaf_size
        apd = int(ATTN_POOL_FACTOR_DIAG)
        apo = int(ATTN_POOL_FACTOR_OFF)
        if L % apd != 0 or L % apo != 0:
            raise ValueError(f"leaf_size {L} must be divisible by ATTN_POOL_FACTOR_DIAG {apd} and OFF {apo}")
        self.attn_pool_diag = apd
        self.attn_pool_off = apo
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
                        attn_pool_factor=apd,
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
                        attn_pool_factor=apo,
                    ),
                )
                for _ in range(num_layers)
            ]
        )
        # Must follow this instance's leaf_size + pool factors (not global LEAF_APPLY_* from config),
        # otherwise checkpoints built with a different config.LEAF_SIZE fail to load.
        self.leaf_apply_size = L // apd
        self.leaf_apply_off = L // apo
        La_d = self.leaf_apply_size
        La_o = self.leaf_apply_off
        self.off_diag_head_U = nn.Linear(d_model, La_o)
        self.off_diag_head_V = nn.Linear(d_model, La_o)
        nn.init.normal_(self.off_diag_head_U.weight, std=0.001)
        nn.init.constant_(self.off_diag_head_U.bias, 0.0)
        nn.init.normal_(self.off_diag_head_V.weight, std=0.001)
        nn.init.constant_(self.off_diag_head_V.bias, 0.0)
        self.leaf_head = nn.Linear(d_model, La_d)
        nn.init.normal_(self.leaf_head.weight, std=0.001)
        nn.init.constant_(self.leaf_head.bias, 0.0)
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

    def _get_leaf_blocks(self, h, mode="diagonal"):
        # Diagonal: (B, N, C) with N = K * L. Off: (B, M_off, L, C) one L-token tile per H block — fold to (B*M, L, C).
        off_bm = None
        if mode == "off-diagonal" and h.dim() == 4:
            B0, M0, Ln, C = h.shape
            if Ln != self.leaf_size:
                raise ValueError(f"off-diagonal h expected L={self.leaf_size}, got {Ln}")
            off_bm = (B0, M0)
            h = h.reshape(B0 * M0, self.leaf_size, C)
        Bf, N_or_P_len, _ = h.shape
        num_leaves = N_or_P_len // self.leaf_size
        h_leaves = h.view(Bf, num_leaves, self.leaf_size, -1)
        if mode == "diagonal":
            h_a = mean_pool_leaf_tokens(h_leaves, self.leaf_size, self.attn_pool_diag)
            u_leaf = self.leaf_head(h_a)
            out = torch.matmul(u_leaf, u_leaf.transpose(-1, -2))
        else:
            h_a = mean_pool_leaf_tokens(h_leaves, self.leaf_size, self.attn_pool_off)
            u_leaf = self.off_diag_head_U(h_a)
            v_leaf = self.off_diag_head_V(h_a)
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
            if self.attn_pool_diag > 1 and attn_mask is not None:
                attn_mask = pool_leaf_attn_mask(attn_mask, self.leaf_size, self.attn_pool_diag)
                edge_feats = pool_leaf_edge_feats(edge_feats, self.leaf_size, self.attn_pool_diag)

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
            # LeafBlockAttention sees x as (B*M_off, L, C) → num_blocks=1; masks must be (B*M_off, 1, L, L+1).
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
            if self.attn_pool_off > 1:
                om, oe = pool_off_leaf_connectivity_if_needed(om, oe, self.leaf_size, self.attn_pool_off)
            if om.dim() == 3:
                om4 = om.unsqueeze(0).expand(B, M_off, -1, -1).contiguous()
            else:
                om4 = om.contiguous()
                if om4.shape[0] == 1 and B > 1:
                    om4 = om4.expand(B, -1, -1, -1).contiguous()
            Lq, Kq = om4.shape[-2], om4.shape[-1]
            if oe.dim() == 4:
                oe5 = oe.unsqueeze(0).expand(B, M_off, -1, -1, -1).contiguous()
            else:
                oe5 = oe.contiguous()
                if oe5.shape[0] == 1 and B > 1:
                    oe5 = oe5.expand(B, -1, -1, -1, -1, -1).contiguous()
            off_attn_mask = om4.reshape(B * M_off, 1, Lq, Kq)
            off_edge_feats = oe5.reshape(B * M_off, 1, Lq, Kq, 4)

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

        if M_off > 0:
            h_k = h.view(B, K, L, C_h)
            Wr = self.hm_pool_w_row.to(device=h.device, dtype=h.dtype)
            Wc = self.hm_pool_w_col.to(device=h.device, dtype=h.dtype)
            row_p = torch.einsum("mk,bklc->bmlc", Wr, h_k)
            col_p = torch.einsum("mk,bklc->bmlc", Wc, h_k)
            h_off = (row_p + col_p).reshape(B * M_off, L, C_h)
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
            h_off = h_off.view(B, M_off, L, C_h)
            off_diag_blocks = self._get_leaf_blocks(h_off, mode="off-diagonal")
        else:
            off_diag_blocks = torch.empty(
                (B, 0, self.leaf_apply_off, self.leaf_apply_off), device=h.device, dtype=h.dtype
            )

        jacobi_scale = self._get_jacobi_scale(h_diag)

        Bb = diag_blocks.shape[0]
        packed_diag = diag_blocks.reshape(Bb, -1)
        packed_off = off_diag_blocks.reshape(Bb, -1)

        packed = torch.cat([packed_diag, packed_off], dim=1)
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

    diag_blocks = precond_packed[:, :diag_size].view(B, num_leaves, La_d, La_d)
    off_diag_blocks = None
    if M_h > 0:
        off_diag_blocks = precond_packed[:, diag_size : diag_size + off_size].view(B, M_h, La_o, La_o)

    jacobi_scale = precond_packed[:, diag_size + off_size :] if precond_packed.shape[1] > diag_size + off_size else None
    return diag_blocks, off_diag_blocks, jacobi_scale


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
    return BSR (block size ``leaf_size``) for block SpMV ``z = M @ r``. Built with
    ``torch.sparse_bsr_tensor`` on ``device`` (avoids PyTorch's CPU COO→BSR conversion). Uses padded
    forward packing (``diag_size_full`` gap), same as ``apply_block_diagonal_M_physical``.
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

    diag_blocks = precond_packed[:, :diag_size_active].view(1, num_leaves, La_d, La_d)

    jacobi_scale = None
    if precond_packed.shape[1] > diag_size_full + off_size:
        jacobi_scale = precond_packed[:, diag_size_full + off_size :][:, :N]

    off_diag_blocks = None
    if M_h > 0:
        off_diag_blocks = precond_packed[:, diag_size_full : diag_size_full + off_size].view(1, M_h, La_o, La_o)

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

    # Assemble BSR (L×L blocks) on ``device`` directly. PyTorch's COO→BSR path
    # (``sparse_coo_tensor(...).to_sparse_bsr``) runs conversion on CPU and warns.
    block_acc: Dict[Tuple[int, int], torch.Tensor] = {}

    def _bsr_block_add(br: int, bc: int, blk: torch.Tensor) -> None:
        k = (br, bc)
        if k in block_acc:
            block_acc[k] = block_acc[k] + blk
        else:
            block_acc[k] = blk.clone()

    for k in range(num_leaves):
        _bsr_block_add(k, k, diag_b[k])

    if off_diag_blocks is not None and M_h > 0:
        for i in range(M_h):
            r0i = int(HM_R0_CPU[i].item())
            c0i = int(HM_C0_CPU[i].item())
            si = int(HM_S_CPU[i].item())
            if r0i + si > num_leaves or c0i + si > num_leaves:
                continue

            oblk = off_diag_blocks[0, i]
            rep = si if La_o == L else max(1, (si * L) // La_o)
            if rep > 1:
                oblk = oblk.repeat_interleave(rep, dim=0).repeat_interleave(rep, dim=1)
                oblk = oblk / float(rep)

            for bi in range(si):
                for bj in range(si):
                    piece = oblk[bi * L : (bi + 1) * L, bj * L : (bj + 1) * L]
                    _bsr_block_add(r0i + bi, c0i + bj, piece)
                    _bsr_block_add(c0i + bj, r0i + bi, piece.t())

    crow: list[int] = [0]
    col_indices_list: list[int] = []
    values_list: list[torch.Tensor] = []
    for br in range(num_leaves):
        cols_br = sorted(bc for (brow, bc) in block_acc if brow == br)
        for bc in cols_br:
            values_list.append(block_acc[(br, bc)])
        crow.append(crow[-1] + len(cols_br))
        col_indices_list.extend(cols_br)

    crow_indices = torch.tensor(crow, device=device, dtype=torch.int64)
    col_indices = torch.tensor(col_indices_list, device=device, dtype=torch.int64)
    values = torch.stack(values_list, dim=0)
    return torch.sparse_bsr_tensor(
        crow_indices,
        col_indices,
        values,
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
    Diagonal blocks are La_d×La_d; off-diagonal blocks La_o×La_o; node layout is leaf_size per leaf.
    Per leaf: mean-pool x to La_*, matmul, repeat_interleave prolongation back to leaf_size.
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

    if M_h > 0:
        off_diag_blocks = precond_packed[:, diag_size : diag_size + off_size].view(B, M_h, La_o, La_o)
        Wr = HM_POOL_W_ROW_CPU.to(device=x.device, dtype=x.dtype)
        Wc = HM_POOL_W_COL_CPU.to(device=x.device, dtype=x.dtype)
        x_row_strips = torch.einsum("mk,bkle->bmle", Wr, x_leaves)
        x_col_strips = torch.einsum("mk,bkle->bmle", Wc, x_leaves)
        flat = B * M_h
        Br = off_diag_blocks.reshape(flat, La_o, La_o)
        xr = x_row_strips.reshape(flat, La_o, K_dim)
        xc = x_col_strips.reshape(flat, La_o, K_dim)
        y_r = torch.bmm(Br, xc)
        y_c = torch.bmm(Br.transpose(-1, -2), xr)
        y_r = y_r.view(B, M_h, La_o, K_dim)
        y_c = y_c.view(B, M_h, La_o, K_dim)
        ridx, cidx, gidx = _hm_prolong_scatter_indices(x.device)
        y_leaves.index_add_(1, ridx, y_r[:, gidx, :, :])
        y_leaves.index_add_(1, cidx, y_c[:, gidx, :, :])

    y = y_leaves.view(B, N, K_dim)

    if precond_packed.shape[1] > diag_size + off_size:
        if jacobi_inv_diag is None:
            raise ValueError("jacobi_inv_diag is required when jacobi_scale is present.")
        if jacobi_inv_diag.shape[:2] != (B, N):
            raise ValueError(f"jacobi_inv_diag shape {jacobi_inv_diag.shape} does not match (B,N)=({B},{N}).")
        jacobi_scale = precond_packed[:, diag_size + off_size :]
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

    if M_h > 0:
        off_diag_blocks = precond_packed[:, diag_size_full : diag_size_full + off_size].view(B, M_h, La_o, La_o)
        na = int(num_leaves_active)
        valid = ((HM_R0_CPU + HM_S_CPU) <= na) & ((HM_C0_CPU + HM_S_CPU) <= na)
        valid = valid.to(device=off_diag_blocks.device, dtype=off_diag_blocks.dtype).view(1, M_h, 1, 1)
        off_diag_blocks = off_diag_blocks * valid

        # Scatter indices address leaf slots up to MAX_NUM_LEAVES-1; y_leaves only has na leaves.
        # Pool with full Wr/Wc on zero-padded x, accumulate into y_full, then slice.
        Wr = HM_POOL_W_ROW_CPU.to(device=x.device, dtype=x.dtype)
        Wc = HM_POOL_W_COL_CPU.to(device=x.device, dtype=x.dtype)
        x_full = x.new_zeros(B, MAX_NUM_LEAVES, L, K_dim)
        x_full[:, :na] = x_leaves
        x_row_strips = torch.einsum("mk,bkle->bmle", Wr, x_full)
        x_col_strips = torch.einsum("mk,bkle->bmle", Wc, x_full)
        flat = B * M_h
        Br = off_diag_blocks.reshape(flat, La_o, La_o)
        xr = x_row_strips.reshape(flat, La_o, K_dim)
        xc = x_col_strips.reshape(flat, La_o, K_dim)
        y_r = torch.bmm(Br, xc)
        y_c = torch.bmm(Br.transpose(-1, -2), xr)
        y_r = y_r.view(B, M_h, La_o, K_dim)
        y_c = y_c.view(B, M_h, La_o, K_dim)
        ridx, cidx, gidx = _hm_prolong_scatter_indices(x.device)
        y_full = x.new_zeros(B, MAX_NUM_LEAVES, L, K_dim)
        y_full[:, :na] = y_leaves
        y_full.index_add_(1, ridx, y_r[:, gidx, :, :])
        y_full.index_add_(1, cidx, y_c[:, gidx, :, :])
        y_leaves.copy_(y_full[:, :na])

    y = y_leaves.view(B, N, K_dim)

    if precond_packed.shape[1] > diag_size_full + off_size:
        if jacobi_inv_diag is None:
            raise ValueError("jacobi_inv_diag is required when jacobi_scale is present.")
        if jacobi_inv_diag.shape[:2] != (B, N):
            raise ValueError(f"jacobi_inv_diag shape {jacobi_inv_diag.shape} does not match (B,N)=({B},{N}).")
        jacobi_scale = precond_packed[:, diag_size_full + off_size :][:, :N]
        y = y + (jacobi_scale * jacobi_inv_diag).unsqueeze(-1) * x
    return y


def block_diagonal_m_apply_workspace(num_leaves: int, leaf_size: int, K_dim: int, M_h: int, La_o: int, device, dtype):
    """Fixed buffers for apply_block_diagonal_m_into (B=1, no allocs in CUDAGraph replay)."""
    lc = leaf_size * K_dim
    t_pro = int(HM_PROLONG_GATHER_IDX.shape[0])
    # Off-diagonal path: pad x_flat to MAX_NUM_LEAVES, gather–scatter pooling (no dense Wr/Wc matmul).
    n_flat = MAX_NUM_LEAVES if M_h > 0 else num_leaves
    out = {
        "x_flat": torch.empty(n_flat, lc, device=device, dtype=dtype),
        "row_flat": torch.empty(M_h, lc, device=device, dtype=dtype),
        "col_flat": torch.empty(M_h, lc, device=device, dtype=dtype),
        "y_r": torch.empty(1, M_h, La_o, K_dim, device=device, dtype=dtype),
        "y_c": torch.empty(1, M_h, La_o, K_dim, device=device, dtype=dtype),
        "y_scatter_r": torch.empty(1, t_pro, La_o, K_dim, device=device, dtype=dtype),
        "y_scatter_c": torch.empty(1, t_pro, La_o, K_dim, device=device, dtype=dtype),
        "x_gather_r": torch.empty(t_pro, lc, device=device, dtype=dtype),
        "x_gather_c": torch.empty(t_pro, lc, device=device, dtype=dtype),
        "inv_S": (1.0 / HM_S_CPU.to(device=device, dtype=dtype)).view(M_h, 1),
    }
    if M_h > 0:
        out["y_leaves_pad"] = torch.empty(1, MAX_NUM_LEAVES, leaf_size, K_dim, device=device, dtype=dtype)
        # Precompute on device before CUDAGraph capture; .to(device) inside capture is not allowed.
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
    Pooling uses gather–scatter (HM prolongation indices); no dense Wr/Wc matmul.
    No internal tensor allocations — safe inside ``torch.cuda.graph`` replay when buffers are fixed.
    Requires leaf_apply_size == leaf_apply_off == leaf_size (no pool-down inside apply).
    """
    if leaf_apply_size is None:
        leaf_apply_size = leaf_size
    if leaf_apply_off is None:
        leaf_apply_off = leaf_apply_size
    if leaf_apply_size != leaf_size or leaf_apply_off != leaf_size:
        raise ValueError("apply_block_diagonal_m_into only supports full-resolution leaf apply (pool factor 1).")
    B, N, K_dim = x.shape
    if B != 1:
        raise ValueError(f"apply_block_diagonal_m_into expects B=1, got {B}")
    num_leaves = N // leaf_size
    La_d = leaf_apply_size
    La_o = leaf_apply_off
    M_h = NUM_HMATRIX_OFF_BLOCKS
    diag_size_active = num_leaves * La_d * La_d
    diag_size_full = MAX_NUM_LEAVES * La_d * La_d
    off_size = M_h * La_o * La_o

    diag_blocks = precond_packed[:, :diag_size_active].view(B, num_leaves, La_d, La_d)
    x_leaves = x.view(B, num_leaves, leaf_size, K_dim)
    y_leaves = out.view(B, num_leaves, leaf_size, K_dim)
    torch.matmul(diag_blocks, x_leaves, out=y_leaves)

    if M_h > 0:
        off_diag_blocks = precond_packed[:, diag_size_full : diag_size_full + off_size].view(
            B, M_h, La_o, La_o
        )
        na = int(num_leaves)
        if na < MAX_NUM_LEAVES:
            off_valid = ws["off_valid"]
            off_diag_blocks = off_diag_blocks * off_valid

        lc = leaf_size * K_dim
        ws["x_flat"].zero_()
        ws["x_flat"][:na].copy_(x_leaves[0].reshape(na, lc))
        ridx, cidx, gidx = _hm_prolong_scatter_indices(x.device)
        torch.index_select(ws["x_flat"], 0, ridx, out=ws["x_gather_r"])
        ws["row_flat"].zero_().index_add_(0, gidx, ws["x_gather_r"])
        ws["row_flat"].mul_(ws["inv_S"])
        torch.index_select(ws["x_flat"], 0, cidx, out=ws["x_gather_c"])
        ws["col_flat"].zero_().index_add_(0, gidx, ws["x_gather_c"])
        ws["col_flat"].mul_(ws["inv_S"])
        Br = off_diag_blocks.reshape(M_h, La_o, La_o)
        xc = ws["col_flat"].view(M_h, La_o, K_dim)
        xr = ws["row_flat"].view(M_h, La_o, K_dim)
        torch.bmm(Br, xc, out=ws["y_r"][0])
        torch.bmm(Br.transpose(-1, -2), xr, out=ws["y_c"][0])
        y_r = ws["y_r"]
        y_c = ws["y_c"]
        torch.index_select(y_r, 1, gidx, out=ws["y_scatter_r"])
        torch.index_select(y_c, 1, gidx, out=ws["y_scatter_c"])
        if na < MAX_NUM_LEAVES:
            y_pad = ws["y_leaves_pad"]
            y_pad.zero_()
            y_pad[0, :na].copy_(y_leaves[0])
            y_pad.index_add_(1, ridx, ws["y_scatter_r"])
            y_pad.index_add_(1, cidx, ws["y_scatter_c"])
            y_leaves[0].copy_(y_pad[0, :na])
        else:
            y_leaves.index_add_(1, ridx, ws["y_scatter_r"])
            y_leaves.index_add_(1, cidx, ws["y_scatter_c"])

    if precond_packed.shape[1] > diag_size_full + off_size:
        if jacobi_inv_diag is None:
            raise ValueError("jacobi_inv_diag is required when jacobi_scale is present.")
        if jacobi_inv_diag.shape[:2] != (B, N):
            raise ValueError(f"jacobi_inv_diag shape {jacobi_inv_diag.shape} does not match (B,N)=({B},{N}).")
        jacobi_scale = precond_packed[:, diag_size_full + off_size :][:, :N]
        out.addcmul_(x, (jacobi_scale * jacobi_inv_diag).unsqueeze(-1))
    return out
