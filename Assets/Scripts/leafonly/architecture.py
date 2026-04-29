import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    ATTENTION_HOPS,
    DIAG_TOKEN_POOL,
    GLOBAL_FEATURES_DIM,
    LEAF_SIZE,
    MAX_NUM_LEAVES,
    OFF_DIAG_TOKEN_POOL,
)
from .data import build_leaf_block_connectivity
from .hmatrix import (
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


def warmup_hmatrix_prolong_gpu(device: torch.device) -> None:
    """Materialize cached H-matrix prolongation indices on ``device`` before ``torch.cuda.graph`` capture.

    ``apply_block_diagonal_m_into`` calls ``_hm_prolong_scatter_indices`` lazily; calling this once up front
    avoids any first-touch ``.to(device)`` during graph recording.
    """
    if device.type == "cuda":
        _hm_prolong_scatter_indices(device)


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
    if special not in (0, 1, 2):
        raise ValueError(f"expected L, L+1, or L+2 key layout, got key_dim={k_old} for L={L}")
    nb = m.shape[0]
    sub = m[:, :L, :L].reshape(nb, Ls, pool, Ls, pool).amax(dim=(2, 4))
    if special == 0:
        out = sub
    else:
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
    if special not in (0, 1, 2):
        raise ValueError(f"expected L, L+1, or L+2 key layout for edge_feats, got key_dim={k_old}")
    nb = m.shape[0]
    E = m.shape[-1]
    sub = m[:, :L, :L, :].reshape(nb, Ls, pool, Ls, pool, E).mean(dim=(2, 4))
    if special == 0:
        out = sub
    else:
        parts = [sub]
        for j in range(special):
            c = m[:, :L, L + j, :].reshape(nb, Ls, pool, E).mean(dim=2)
            parts.append(c.unsqueeze(2))
        out = torch.cat(parts, dim=2)
    return _unmerge_batch_blocks(out, batch_shape, tuple(out.shape[1:]))


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


EDGE_GATE_HIDDEN_DIM = 16


class LeafBlockAttention(nn.Module):
    def __init__(self, dim, block_size, num_heads=8, attention_layout=None):
        super().__init__()
        self.dim = dim
        self.block_size = int(block_size)
        if attention_layout is None:
            attention_layout = default_attention_layout(block_size)
        self.use_block_node, self.use_matrix_node = parse_attention_layout(attention_layout, block_size)
        self.attention_layout = attention_layout
        self.num_heads = num_heads if dim % num_heads == 0 else 1
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.edge_gate = nn.Sequential(
            nn.Linear(4, EDGE_GATE_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(EDGE_GATE_HIDDEN_DIM, self.num_heads),
        )
        nn.init.normal_(self.edge_gate[0].weight, std=0.01)
        nn.init.zeros_(self.edge_gate[0].bias)
        nn.init.normal_(self.edge_gate[2].weight, std=0.01)
        nn.init.zeros_(self.edge_gate[2].bias)
        # Decoupled global routing: block/matrix values bypass spatial softmax; gates can → 0.
        self.block_route_gate = nn.Linear(dim, self.num_heads) if self.use_block_node else None
        self.matrix_route_gate = nn.Linear(dim, self.num_heads) if self.use_matrix_node else None
        if self.block_route_gate is not None:
            nn.init.normal_(self.block_route_gate.weight, std=0.01)
            nn.init.zeros_(self.block_route_gate.bias)
        if self.matrix_route_gate is not None:
            nn.init.normal_(self.matrix_route_gate.weight, std=0.01)
            nn.init.zeros_(self.matrix_route_gate.bias)
        self.last_attn_self = 0.0
        self.last_attn_neighbor = 0.0
        self.last_attn_block = 0.0
        self.last_attn_distribution_stats: Optional[dict] = None
        self.last_attn_matrix = None
        self.last_scores_matrix = None
        self.last_bias_physics_matrix = None

    def _align_prep_blocks(self, prep: torch.Tensor, B: int, num_blocks: int, C: int) -> torch.Tensor:
        """Slice or pad (B, *, 1, C) along block dim to match ``num_blocks``."""
        if prep.shape[0] != B or prep.shape[2] != 1 or prep.shape[3] != C:
            raise ValueError(
                f"prep node tensor expected shape (B={B}, num_blocks, 1, C={C}), got {tuple(prep.shape)}"
            )
        pb = prep
        nb = int(prep.shape[1])
        if nb == num_blocks:
            return pb
        if nb > num_blocks:
            return pb[:, :num_blocks]
        pad_n = num_blocks - nb
        return torch.cat([pb, pb.new_zeros(B, pad_n, 1, C)], dim=1)

    def forward(
        self,
        x,
        edge_index=None,
        edge_values=None,
        positions=None,
        save_attention=False,
        attn_mask=None,
        edge_feats=None,
        prep_block_node=None,
        prep_matrix_node=None,
        eager_attention: bool = False,
    ):
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype
        L = self.block_size
        node_pad = 0
        if N % L != 0:
            node_pad = L - (N % L)
            x = F.pad(x, (0, 0, 0, node_pad))
        N_pad = x.shape[1]
        num_blocks = N_pad // L
        x_blk = x.view(B, num_blocks, L, C)

        if attn_mask is None:
            if edge_feats is None:
                return x[:, :N, :] if node_pad > 0 else x
            no_binary_mask = True
        else:
            no_binary_mask = False
            if edge_feats is None:
                raise ValueError("LeafBlockAttention requires edge_feats when attn_mask is provided.")

        x_attn = x_blk
        use_block_node = self.use_block_node
        use_matrix_node = self.use_matrix_node

        # L×L spatial attention only: drop block/matrix key columns from mask and edge grid (global via gates).
        ef_sp = edge_feats[..., :L, :L, :4]
        if ef_sp.dim() == 4:
            bias_physics = ef_sp.unsqueeze(0).expand(B, -1, -1, -1, -1).clone()
        else:
            bias_physics = ef_sp.clone()
            if bias_physics.shape[0] == 1 and B > 1:
                bias_physics = bias_physics.expand(B, -1, -1, -1, -1)

        if not no_binary_mask:
            am = attn_mask
            if am.shape[-1] > L:
                am = am[..., :L]
            if am.dim() == 3:
                mask_base = am.unsqueeze(0)
            else:
                mask_base = am
                if mask_base.shape[0] == 1 and B > 1:
                    mask_base = mask_base.expand(B, -1, -1, -1)

            # Coupled layout used to always include a block key (column L) = 1, so no row was all-masked.
            # Spatial-only softmax rows must have ≥1 allowed key or softmax is all -inf → NaN (H-off tiles
            # with no reachability, batch-padded empty leaf blocks, etc.).
            mask_base = mask_base.clone()
            _need_diag = (mask_base.sum(dim=-1) < 1).to(dtype=mask_base.dtype)
            _diag_view = mask_base.diagonal(dim1=-2, dim2=-1)
            _diag_view.copy_(torch.maximum(_diag_view, _need_diag))
        else:
            mask_base = None

        key_count = L
        qkv_x = self.qkv(x_attn)
        q = qkv_x[..., :C].view(B, num_blocks, L, self.num_heads, self.head_dim)
        k = qkv_x[..., C : 2 * C].view(B, num_blocks, L, self.num_heads, self.head_dim)
        v_spatial = qkv_x[..., 2 * C : 3 * C].view(B, num_blocks, L, self.num_heads, self.head_dim)

        arange_s = torch.arange(L, device=device, dtype=torch.long)
        bias_physics[:, :, arange_s, arange_s, :] = 0.0
        bias_physics[:, :, arange_s, arange_s, 3] = 1.0

        bn = B * num_blocks
        Hh, Dd = self.num_heads, self.head_dim
        q_f = q.reshape(bn, L, Hh, Dd).transpose(1, 2).contiguous()
        k_f = k.reshape(bn, L, Hh, Dd).transpose(1, 2).contiguous()
        v_f = v_spatial.reshape(bn, L, Hh, Dd).transpose(1, 2).contiguous()
        bias_w3 = bias_physics[:, :, :, :, 3]
        logit_bias = (
            bias_w3.reshape(bn, 1, L, L).expand(bn, Hh, L, L).to(dtype=q_f.dtype).clone()
        )

        if no_binary_mask:
            linear_edge_weights = self.edge_gate(bias_physics)
            mask_expanded = None
        else:
            mask_expanded = mask_base.unsqueeze(-1)
            linear_edge_weights = self.edge_gate(bias_physics)
            linear_edge_weights = linear_edge_weights.masked_fill(mask_expanded == 0, 0.0)
            mb = mask_base.reshape(bn, 1, L, L).expand(bn, Hh, L, L)
            logit_bias = logit_bias.masked_fill(mb == 0, float("-inf"))

        # Fused SDPA is fast but does not materialize attention weights. Mass / distribution stats must use
        # either eager_attention (manual softmax @ V, same logits as SDPA) or a separate score pass only
        # when save_attention needs tensors — never treat ad-hoc score math as authoritative under SDPA alone.
        use_eager = bool(eager_attention)
        if use_eager:
            scores_f = torch.matmul(q_f, k_f.transpose(-2, -1)) * self.scale
            scores_f = scores_f + logit_bias
            attn_probs_f = F.softmax(scores_f, dim=-1)
            x_soft = torch.matmul(attn_probs_f, v_f)
            x_soft = x_soft.transpose(1, 2).reshape(B, num_blocks, L, Hh, Dd)
            attn_probs = attn_probs_f.reshape(B, num_blocks, Hh, L, L).permute(0, 1, 3, 4, 2).contiguous()
            combined_weights = attn_probs + linear_edge_weights
            if save_attention:
                scores = torch.einsum("bnqhd,bnkhd->bnqkh", q, k) * self.scale
                scores = scores + bias_physics[..., 3:4]
                if not no_binary_mask:
                    scores = scores.masked_fill(mask_expanded == 0, float("-inf"))
                with torch.no_grad():
                    self.last_scores_matrix = scores.mean(dim=-1)[:, :, :, :L].cpu().float()
                    self.last_bias_physics_matrix = bias_physics[:, :, :L].cpu().float()
        else:
            x_soft = F.scaled_dot_product_attention(
                q_f, k_f, v_f, attn_mask=logit_bias, dropout_p=0.0, is_causal=False
            )
            x_soft = x_soft.transpose(1, 2).reshape(B, num_blocks, L, Hh, Dd)
            combined_weights = None
            if save_attention:
                scores = torch.einsum("bnqhd,bnkhd->bnqkh", q, k) * self.scale
                scores = scores + bias_physics[..., 3:4]
                if not no_binary_mask:
                    scores = scores.masked_fill(mask_expanded == 0, float("-inf"))
                attn_probs = F.softmax(scores, dim=3)
                combined_weights = attn_probs + linear_edge_weights
                with torch.no_grad():
                    self.last_scores_matrix = scores.mean(dim=-1)[:, :, :, :L].cpu().float()
                    self.last_bias_physics_matrix = bias_physics[:, :, :L].cpu().float()

        x_mid = x_soft + torch.einsum("bnqkh,bnkhd->bnqhd", linear_edge_weights, v_spatial)

        g_block = None
        g_matrix = None
        if use_block_node:
            if prep_block_node is not None:
                block_node = self._align_prep_blocks(prep_block_node, B, num_blocks, C)
            else:
                block_node = x_attn.mean(dim=2, keepdim=True)
            v_block = self.qkv(block_node)[..., 2 * C : 3 * C].view(
                B, num_blocks, 1, self.num_heads, self.head_dim
            )
            g_block = torch.sigmoid(self.block_route_gate(x_attn))
            x_mid = x_mid + g_block.unsqueeze(-1) * v_block

        if use_matrix_node:
            if prep_matrix_node is not None:
                matrix_node = self._align_prep_blocks(prep_matrix_node, B, num_blocks, C)
            else:
                matrix_node = x.mean(dim=1, keepdim=True).unsqueeze(1).expand(-1, num_blocks, -1, -1)
            v_matrix = self.qkv(matrix_node)[..., 2 * C : 3 * C].view(
                B, num_blocks, 1, self.num_heads, self.head_dim
            )
            g_matrix = torch.sigmoid(self.matrix_route_gate(x_attn))
            x_mid = x_mid + g_matrix.unsqueeze(-1) * v_matrix

        if combined_weights is not None and not self.training and not torch.compiler.is_compiling():
            with torch.no_grad():
                attn_viz = combined_weights.mean(dim=-1)
                arange = torch.arange(L, device=attn_viz.device, dtype=torch.long)
                self.last_attn_self = attn_viz[:, :, arange, arange].mean().item()
                to_nodes = attn_viz[:, :, :, :L].sum(dim=3)
                self.last_attn_neighbor = (to_nodes - attn_viz[:, :, arange, arange]).mean().item()
                if use_block_node and g_block is not None:
                    self.last_attn_block = float(g_block.mean().item())
                else:
                    self.last_attn_block = 0.0
                m_mat_model = float(g_matrix.mean().item()) if use_matrix_node and g_matrix is not None else 0.0
                if save_attention:
                    self.last_attn_matrix = attn_viz[:, :, :, :L].cpu().float()

                # Spatial softmax uses K=L only; block/matrix are σ(gate)·V baselines (not in softmax sum).
                K = int(key_count)
                if no_binary_mask:
                    mask_b = torch.ones_like(attn_viz)
                else:
                    mask_b = (mask_base > 0).to(dtype=attn_viz.dtype)
                n_allowed = mask_b.sum(dim=-1, keepdim=True).clamp(min=1.0)
                unif = mask_b / n_allowed
                u_self = float(unif[:, :, arange, arange].mean().item())
                u_leaf = unif[..., :L]
                u_sum_leaf = u_leaf.sum(dim=-1)
                u_diag = unif[:, :, arange, arange]
                u_nei = float((u_sum_leaf - u_diag).mean().item())
                inv_k = 1.0 / float(K)
                b_self = inv_k
                b_nei = inv_k * float(max(L - 1, 0))
                if use_block_node and g_block is not None:
                    model_blk = float(g_block.mean().item())
                    u_blk = 0.5
                    b_blk = 0.0
                else:
                    model_blk = 0.0
                    u_blk = 0.0
                    b_blk = 0.0
                if use_matrix_node and g_matrix is not None:
                    model_mat = m_mat_model
                    u_mat = 0.5
                    b_mat = 0.0
                else:
                    model_mat = 0.0
                    u_mat = 0.0
                    b_mat = 0.0
                self.last_attn_distribution_stats = {
                    "key_count": K,
                    "model_self": self.last_attn_self,
                    "unif_self": u_self,
                    "blind_self": b_self,
                    "model_nei": self.last_attn_neighbor,
                    "unif_nei": u_nei,
                    "blind_nei": b_nei,
                    "model_blk": model_blk,
                    "unif_blk": u_blk,
                    "blind_blk": b_blk,
                    "model_mat": model_mat,
                    "unif_mat": u_mat,
                    "blind_mat": b_mat,
                    "excess_self": self.last_attn_self - u_self,
                    "excess_nei": self.last_attn_neighbor - u_nei,
                    "excess_blk": model_blk - u_blk,
                    "excess_mat": model_mat - u_mat,
                    "excess_self_vs_blind": self.last_attn_self - b_self,
                    "excess_nei_vs_blind": self.last_attn_neighbor - b_nei,
                    "excess_blk_vs_blind": model_blk - b_blk,
                    "excess_mat_vs_blind": model_mat - b_mat,
                }
        else:
            self.last_attn_distribution_stats = None
            if not self.training and not torch.compiler.is_compiling():
                self.last_attn_self = 0.0
                self.last_attn_neighbor = 0.0
                self.last_attn_block = 0.0

        x_out = x_mid
        x_out = x_out.reshape(B, num_blocks, L, C)
        x_out = self.proj(x_out)
        x_out = x_out.reshape(B, N_pad, C)
        if node_pad > 0:
            x_out = x_out[:, :N, :]
        return x_out


class TransformerBlock(nn.Module):
    def __init__(self, dim, block_size, attn_module, heads=4, mlp_ratio=4.0, mlp_in_dim: Optional[int] = None):
        super().__init__()
        self.token_dim = int(dim)
        self.mlp_in_dim = int(mlp_in_dim) if mlp_in_dim is not None else int(dim)
        if self.mlp_in_dim < self.token_dim or self.mlp_in_dim % self.token_dim != 0:
            raise ValueError(f"mlp_in_dim ({self.mlp_in_dim}) must be a positive multiple of dim ({self.token_dim})")
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attn_module
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_in_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward_attn(
        self,
        x,
        edge_index=None,
        edge_values=None,
        positions=None,
        save_attention=False,
        attn_mask=None,
        edge_feats=None,
        prep_block_node=None,
        prep_matrix_node=None,
        eager_attention: bool = False,
    ):
        return x + self.attn(
            self.norm1(x),
            edge_index=edge_index,
            edge_values=edge_values,
            positions=positions,
            save_attention=save_attention,
            attn_mask=attn_mask,
            edge_feats=edge_feats,
            prep_block_node=prep_block_node,
            prep_matrix_node=prep_matrix_node,
            eager_attention=eager_attention,
        )

    def forward_mlp(
        self,
        x_mid,
        mlp_row_hw: Optional[torch.Tensor] = None,
        mlp_col_hw: Optional[torch.Tensor] = None,
        mlp_global: Optional[torch.Tensor] = None,
    ):
        z = self.norm2(x_mid)
        if self.mlp_in_dim == self.token_dim:
            pass
        else:
            n_in = self.mlp_in_dim // self.token_dim
            if n_in == 4:
                if mlp_row_hw is None or mlp_col_hw is None or mlp_global is None:
                    raise ValueError("mlp_row_hw, mlp_col_hw, and mlp_global are required for 4× token_dim FFN input")
                z = torch.cat([z, mlp_row_hw, mlp_col_hw, mlp_global], dim=-1)
            elif n_in == 3:
                if mlp_row_hw is None or mlp_col_hw is None:
                    raise ValueError("mlp_row_hw and mlp_col_hw are required for 3× token_dim FFN input")
                z = torch.cat([z, mlp_row_hw, mlp_col_hw], dim=-1)
            else:
                raise ValueError(f"Unsupported mlp_in_dim={self.mlp_in_dim} for token_dim={self.token_dim}")
        return x_mid + self.mlp(z)

    def forward(
        self,
        x,
        edge_index=None,
        edge_values=None,
        positions=None,
        save_attention=False,
        attn_mask=None,
        edge_feats=None,
        prep_block_node=None,
        prep_matrix_node=None,
        eager_attention: bool = False,
        mlp_row_hw: Optional[torch.Tensor] = None,
        mlp_col_hw: Optional[torch.Tensor] = None,
        mlp_global: Optional[torch.Tensor] = None,
    ):
        x_mid = self.forward_attn(
            x,
            edge_index=edge_index,
            edge_values=edge_values,
            positions=positions,
            save_attention=save_attention,
            attn_mask=attn_mask,
            edge_feats=edge_feats,
            prep_block_node=prep_block_node,
            prep_matrix_node=prep_matrix_node,
            eager_attention=eager_attention,
        )
        return self.forward_mlp(x_mid, mlp_row_hw, mlp_col_hw, mlp_global)


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
        strip_build_mode: str = "einsum",
        off_diag_dense_attention: bool = True,
        diag_dense_attention: bool = True,
        use_highways: bool = True,
        ffn_concat_width: Optional[int] = None,
    ):
        super().__init__()
        if strip_build_mode not in ("einsum", "no_einsum"):
            raise ValueError(f"strip_build_mode must be 'einsum' or 'no_einsum', got {strip_build_mode!r}")
        self.strip_build_mode = strip_build_mode
        self.off_diag_dense_attention = bool(off_diag_dense_attention)
        self.diag_dense_attention = bool(diag_dense_attention)
        self.use_highways = bool(use_highways)
        # When True: FFN concat width 4 = [block, row hw, col hw, global DC]; 3 = legacy checkpoints without global.
        # Global = mean over tokens on the current stream, broadcast (0D / DC term vs 1D highways / 2D attention).
        self.highway_ffn = self.use_highways
        if self.use_highways:
            fcw = 4 if ffn_concat_width is None else int(ffn_concat_width)
            if fcw not in (3, 4):
                raise ValueError(f"ffn_concat_width must be 3 or 4 when use_highways, got {fcw}")
            self.ffn_concat_width = fcw
        else:
            if ffn_concat_width is not None and int(ffn_concat_width) != 1:
                raise ValueError("ffn_concat_width must be 1 (or omit) when use_highways=False")
            self.ffn_concat_width = 1
        self.leaf_size = int(leaf_size)
        L = self.leaf_size
        dtp = int(DIAG_TOKEN_POOL)
        if L % dtp != 0:
            raise ValueError(f"leaf_size {L} not divisible by DIAG_TOKEN_POOL {dtp}")
        Ls_diag = L // dtp
        self.diag_token_pool = dtp
        otp = int(OFF_DIAG_TOKEN_POOL)
        if L % otp != 0:
            raise ValueError(f"leaf_size {L} not divisible by OFF_DIAG_TOKEN_POOL {otp}")
        Ls_off = L // otp
        diag_layout = f"{Ls_diag}x{Ls_diag}" if self.diag_dense_attention else f"{Ls_diag}x{Ls_diag + 1}"
        off_layout = f"{Ls_off}x{Ls_off}" if self.off_diag_dense_attention else f"{Ls_off}x{Ls_off + 1}"
        self.embed = PhysicsAwareEmbedding(
            input_dim,
            d_model,
            use_gcn=use_gcn,
            global_features_dim=GLOBAL_FEATURES_DIM,
            num_gcn_layers=num_gcn_layers,
        )
        self.enc_input_proj = nn.Linear(d_model, d_model)
        self.off_token_pool = otp
        self.leaf_apply_size = Ls_diag
        self.leaf_apply_off = Ls_off
        La_d = Ls_diag
        La_o = Ls_off
        _ffn_in = int(self.ffn_concat_width) * d_model
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    block_size=Ls_diag,
                    attn_module=LeafBlockAttention(
                        d_model,
                        Ls_diag,
                        num_heads=num_heads,
                        attention_layout=diag_layout,
                    ),
                    mlp_in_dim=_ffn_in,
                )
                for _ in range(num_layers)
            ]
        )
        # H row/col strip features start at length L; if off_token_pool>1, mean-pool to Ls_off before this stack.
        self.off_diag_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    block_size=Ls_off,
                    attn_module=LeafBlockAttention(
                        d_model,
                        Ls_off,
                        num_heads=num_heads,
                        attention_layout=off_layout,
                    ),
                    mlp_in_dim=_ffn_in,
                )
                for _ in range(num_layers)
            ]
        )
        head_in = d_model
        self.off_diag_head_U = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, La_o),
        )
        self.off_diag_head_V = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, La_o),
        )
        nn.init.normal_(self.off_diag_head_U[0].weight, std=0.001)
        nn.init.constant_(self.off_diag_head_U[0].bias, 0.0)
        nn.init.normal_(self.off_diag_head_U[2].weight, std=0.001)
        nn.init.constant_(self.off_diag_head_U[2].bias, 0.0)
        nn.init.normal_(self.off_diag_head_V[0].weight, std=0.001)
        nn.init.constant_(self.off_diag_head_V[0].bias, 0.0)
        nn.init.normal_(self.off_diag_head_V[2].weight, std=0.001)
        nn.init.constant_(self.off_diag_head_V[2].bias, 0.0)
        self.leaf_head = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, La_d),
        )
        nn.init.normal_(self.leaf_head[0].weight, std=0.001)
        nn.init.constant_(self.leaf_head[0].bias, 0.0)
        nn.init.normal_(self.leaf_head[2].weight, std=0.001)
        nn.init.constant_(self.leaf_head[2].bias, 0.0)

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
        self.register_buffer("leaf_idx_full", torch.arange(MAX_NUM_LEAVES, dtype=torch.long), persistent=False)
        self._last_row_highway = None
        self._last_col_highway = None
        self._highway_cache: dict = {}

    def _pool_full_leaf_to_diag_tokens(self, h: torch.Tensor, B: int, K: int, L_full: int, C_h: int) -> torch.Tensor:
        """(B, K*L_full, C) → (B, K*Ls_diag, C) via uniform mean-pool over ``diag_token_pool`` consecutive tokens."""
        h_kv = h.view(B, K, L_full, C_h)
        p = self.diag_token_pool
        if p <= 1:
            return h_kv.reshape(B, K * L_full, C_h)
        Ls = self.leaf_apply_size
        hp = h_kv.view(B, K, Ls, p, C_h).mean(dim=3)
        return hp.reshape(B, K * Ls, C_h)

    def _diag_tokens_to_full_leaf(self, h_diag: torch.Tensor, B: int, K: int, L_full: int, C_h: int) -> torch.Tensor:
        """(B, K*Ls_diag, C) → (B, K*L_full, C) by repeating each pooled token across its pool group (strip / node heads)."""
        p = self.diag_token_pool
        if p <= 1:
            return h_diag
        Ls = self.leaf_apply_size
        h_kv = h_diag.view(B, K, Ls, C_h)
        h_kv = h_kv.repeat_interleave(p, dim=2)
        return h_kv.reshape(B, K * L_full, C_h)

    def _diag_head_tokens_from_highways(
        self, row_highway: torch.Tensor, col_highway: torch.Tensor, B: int, K: int, L_full: int, C_h: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_k = row_highway.view(B, K, L_full, C_h)
        col_k = col_highway.view(B, K, L_full, C_h)
        p = int(self.diag_token_pool)
        if p > 1:
            Ls = int(self.leaf_apply_size)
            row_k = row_k.view(B, K, Ls, p, C_h).mean(dim=3)
            col_k = col_k.view(B, K, Ls, p, C_h).mean(dim=3)
        return row_k, col_k

    def _off_head_tokens_from_highways(
        self, row_highway: torch.Tensor, col_highway: torch.Tensor, B: int, K: int, L_full: int, C_h: int, M_off: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_k = row_highway.view(B, K, L_full, C_h)
        col_k = col_highway.view(B, K, L_full, C_h)
        if K < MAX_NUM_LEAVES:
            row_k = F.pad(row_k, (0, 0, 0, 0, 0, MAX_NUM_LEAVES - K))
            col_k = F.pad(col_k, (0, 0, 0, 0, 0, MAX_NUM_LEAVES - K))
        if self.strip_build_mode == "einsum":
            Wr = self.hm_pool_w_row.to(device=row_highway.device, dtype=row_highway.dtype)
            Wc = self.hm_pool_w_col.to(device=row_highway.device, dtype=row_highway.dtype)
            row_t = torch.einsum("mk,bklc->bmlc", Wr, row_k)
            col_t = torch.einsum("mk,bklc->bmlc", Wc, col_k)
        else:
            ridx, cidx, gidx = _hm_prolong_scatter_indices(row_highway.device)
            row_kt = row_k.transpose(0, 1)
            col_kt = col_k.transpose(0, 1)
            row_sum = torch.zeros(M_off, B, L_full, C_h, device=row_highway.device, dtype=row_highway.dtype)
            col_sum = torch.zeros(M_off, B, L_full, C_h, device=row_highway.device, dtype=row_highway.dtype)
            row_sum.index_add_(0, gidx, row_kt[ridx])
            col_sum.index_add_(0, gidx, col_kt[cidx])
            S_view = self.h_block_S.view(M_off, 1, 1, 1).to(device=row_highway.device, dtype=row_highway.dtype)
            row_t = (row_sum / S_view).transpose(0, 1)
            col_t = (col_sum / S_view).transpose(0, 1)
        p = int(self.off_token_pool)
        if p > 1:
            Ls = int(self.leaf_apply_off)
            row_t = row_t.view(B, M_off, Ls, p, C_h).mean(dim=3)
            col_t = col_t.view(B, M_off, Ls, p, C_h).mean(dim=3)
        return row_t, col_t

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
            La = self.leaf_apply_size
            num_leaves = N_or_P_len // La
            h_leaves = h.view(Bf, num_leaves, La, -1)
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

    def _build_off_strip(self, h_diag: torch.Tensor, B: int, K: int, L: int, C_h: int, M_off: int) -> torch.Tensor:
        """H row/column strip aggregate for all off-blocks; shape (B*M_off, L or Ls_off, C_h)."""
        h_k = h_diag.view(B, K, L, C_h)
        if K < MAX_NUM_LEAVES:
            h_k = F.pad(h_k, (0, 0, 0, 0, 0, MAX_NUM_LEAVES - K))
        if self.strip_build_mode == "einsum":
            Wr = self.hm_pool_w_row.to(device=h_diag.device, dtype=h_diag.dtype)
            Wc = self.hm_pool_w_col.to(device=h_diag.device, dtype=h_diag.dtype)
            row_p = torch.einsum("mk,bklc->bmlc", Wr, h_k)
            col_p = torch.einsum("mk,bklc->bmlc", Wc, h_k)
        else:
            ridx, cidx, gidx = _hm_prolong_scatter_indices(h_diag.device)
            h_k_t = h_k.transpose(0, 1)
            row_sum = torch.zeros(M_off, B, L, C_h, device=h_diag.device, dtype=h_diag.dtype)
            col_sum = torch.zeros(M_off, B, L, C_h, device=h_diag.device, dtype=h_diag.dtype)
            row_sum.index_add_(0, gidx, h_k_t[ridx])
            col_sum.index_add_(0, gidx, h_k_t[cidx])
            S_view = self.h_block_S.view(M_off, 1, 1, 1).to(device=h_diag.device, dtype=h_diag.dtype)
            row_p = (row_sum / S_view).transpose(0, 1)
            col_p = (col_sum / S_view).transpose(0, 1)
        h_off = (row_p + col_p).reshape(B * M_off, L, C_h)
        if self.off_token_pool > 1:
            Ls = self.leaf_apply_off
            h_off = h_off.view(B * M_off, Ls, self.off_token_pool, C_h).mean(dim=2)
        return h_off

    def _accumulate_highway_from_blocks(
        self,
        row_highway: torch.Tensor,
        col_highway: torch.Tensor,
        block_tokens: torch.Tensor,
        row_leaf_idx: torch.Tensor,
        col_leaf_idx: torch.Tensor,
        gather_idx: Optional[torch.Tensor],
        pool: int,
        leaf_size: int,
    ) -> None:
        """
        Add per-block tokens into row/column highways in leaf-index space.
        block_tokens: (B, M, Ls, C) with Ls * pool == leaf_size.
        row_leaf_idx/col_leaf_idx: (T,) destination leaf indices.
        gather_idx: (T,) mapping from destination slot to source block (or None when T == M).
        """
        if block_tokens.numel() == 0:
            return
        B, M, Ls, C_h = block_tokens.shape
        if Ls * int(pool) != int(leaf_size):
            raise ValueError(
                f"highway scatter expected tokens_per_block * pool == leaf_size, got {Ls} * {pool} != {leaf_size}"
            )
        toks = block_tokens if pool <= 1 else block_tokens.repeat_interleave(int(pool), dim=2)
        if gather_idx is not None:
            toks = toks.index_select(1, gather_idx)
        T = int(row_leaf_idx.numel())
        if toks.shape[1] != T:
            raise ValueError(
                f"highway scatter mismatch: gathered tokens have T={toks.shape[1]}, "
                f"but destination leaf index length is {T}"
            )
        row_highway.index_add_(1, row_leaf_idx, toks)
        col_highway.index_add_(1, col_leaf_idx, toks)

    def _build_token_highways(
        self,
        h_ref: torch.Tensor,
        h_diag: torch.Tensor,
        off_stream: Optional[torch.Tensor],
        B: int,
        K: int,
        L: int,
        C_h: int,
        M_off: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build full row/column token highways (B, K*L, C) from diagonal and H off-diagonal token streams.
        """
        cache_key = (str(h_ref.device), str(h_ref.dtype), int(B), int(K), int(L), int(C_h), int(M_off))
        cache = self._highway_cache.get(cache_key)
        if cache is None:
            diag_leaf_idx = self.leaf_idx_full[:K].to(device=h_ref.device, dtype=torch.long)
            if M_off > 0:
                off_row_idx, off_col_idx, off_gather_idx = _hm_prolong_scatter_indices(h_ref.device)
            else:
                off_row_idx = off_col_idx = off_gather_idx = None
            cache = {
                "diag_leaf_idx": diag_leaf_idx,
                "off_row_idx": off_row_idx,
                "off_col_idx": off_col_idx,
                "off_gather_idx": off_gather_idx,
            }
            self._highway_cache[cache_key] = cache
        row_highway = h_ref.new_zeros(B, MAX_NUM_LEAVES, L, C_h)
        col_highway = h_ref.new_zeros(B, MAX_NUM_LEAVES, L, C_h)

        diag_tokens = h_diag.view(B, K, self.leaf_apply_size, C_h)
        self._accumulate_highway_from_blocks(
            row_highway=row_highway,
            col_highway=col_highway,
            block_tokens=diag_tokens,
            row_leaf_idx=cache["diag_leaf_idx"],
            col_leaf_idx=cache["diag_leaf_idx"],
            gather_idx=None,
            pool=self.diag_token_pool,
            leaf_size=L,
        )
        if M_off > 0 and off_stream is not None:
            off_tokens = off_stream.view(B, M_off, self.leaf_apply_off, C_h)
            self._accumulate_highway_from_blocks(
                row_highway=row_highway,
                col_highway=col_highway,
                block_tokens=off_tokens,
                row_leaf_idx=cache["off_row_idx"],
                col_leaf_idx=cache["off_col_idx"],
                gather_idx=cache["off_gather_idx"],
                pool=self.off_token_pool,
                leaf_size=L,
            )
        return (
            row_highway[:, :K, :, :].reshape(B, K * L, C_h),
            col_highway[:, :K, :, :].reshape(B, K * L, C_h),
        )

    def _build_diag_mlp_hw(
        self,
        h_diag_post_attn: torch.Tensor,
        off_stream: Optional[torch.Tensor],
        B: int,
        K: int,
        L: int,
        C_h: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Row/col tokens (each (B, K*Ls_diag, C)) for FFN concat after diagonal attention."""
        M_off = self.num_h_off
        off_for = None
        if off_stream is not None and M_off > 0:
            off_for = off_stream.reshape(B, M_off, self.leaf_apply_off, C_h)
        row_full, col_full = self._build_token_highways(
            h_ref=h_diag_post_attn,
            h_diag=h_diag_post_attn,
            off_stream=off_for,
            B=B,
            K=K,
            L=L,
            C_h=C_h,
            M_off=M_off,
        )
        row_k, col_k = self._diag_head_tokens_from_highways(row_full, col_full, B, K, L, C_h)
        Ls_d = int(self.leaf_apply_size)
        return row_k.reshape(B, K * Ls_d, C_h), col_k.reshape(B, K * Ls_d, C_h)

    def _build_off_mlp_hw(
        self,
        h_diag_current: torch.Tensor,
        off_post_attn: torch.Tensor,
        B: int,
        K: int,
        L: int,
        C_h: int,
        M_off: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Row/col tokens (each (B*M_off, Ls_off, C)) for FFN concat after H-off attention."""
        off_view = off_post_attn.view(B, M_off, self.leaf_apply_off, C_h)
        row_full, col_full = self._build_token_highways(
            h_ref=h_diag_current,
            h_diag=h_diag_current,
            off_stream=off_view,
            B=B,
            K=K,
            L=L,
            C_h=C_h,
            M_off=M_off,
        )
        r, c = self._off_head_tokens_from_highways(row_full, col_full, B, K, L, C_h, M_off)
        Bm = B * M_off
        return r.reshape(Bm, self.leaf_apply_off, C_h), c.reshape(Bm, self.leaf_apply_off, C_h)

    @staticmethod
    def _mlp_global_broadcast(x_mid: torch.Tensor) -> torch.Tensor:
        """Mean over the token dimension (DC / 0D), broadcast to every token — same shape as ``x_mid``."""
        if x_mid.dim() != 3:
            raise ValueError(f"_mlp_global_broadcast expects (B, N, C), got {tuple(x_mid.shape)}")
        g = x_mid.mean(dim=1, keepdim=True)
        return g.expand(-1, x_mid.shape[1], -1)

    def _apply_transformer_stacks(
        self,
        h: torch.Tensor,
        h_diag: torch.Tensor,
        B: int,
        K: int,
        L: int,
        C_h: int,
        edge_index,
        edge_values,
        positions,
        attn_mask,
        edge_feats,
        off_attn_mask,
        off_edge_feats,
        save_attention: bool,
        eager_attention: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run diagonal (+ interleaved H-off) transformer blocks. ``h`` is full post-embed (B,N,C); ``h_diag`` is pooled.
        Returns ``(h_diag_out, off_stream)`` where ``off_stream`` is (B*M_off, Ls_off, C) or None.
        """
        Ls_d = self.leaf_apply_size
        h_lv_diag = h_diag.view(B, K, Ls_d, C_h)
        diag_prep_block = None if self.diag_dense_attention else h_lv_diag.mean(dim=2, keepdim=True)
        diag_prep_matrix = h.mean(dim=1, keepdim=True).unsqueeze(1).expand(-1, K, -1, -1)
        M_off = self.num_h_off
        off_prep_block = off_prep_matrix = None
        if M_off > 0:
            strip0 = self._build_off_strip(h, B, K, L, C_h, M_off)
            Ls = self.leaf_apply_off
            Bm = B * M_off
            if not self.off_diag_dense_attention:
                off_prep_block = strip0.view(Bm, 1, Ls, C_h).mean(dim=2, keepdim=True)
            off_prep_matrix = strip0.mean(dim=1, keepdim=True).unsqueeze(1).unsqueeze(1)

        off_stream: Optional[torch.Tensor] = None
        n_l = len(self.blocks)
        attn_kw_diag = dict(
            edge_index=edge_index,
            edge_values=edge_values,
            positions=positions,
            save_attention=save_attention,
            attn_mask=attn_mask,
            edge_feats=edge_feats,
            prep_block_node=diag_prep_block,
            prep_matrix_node=diag_prep_matrix,
            eager_attention=eager_attention,
        )
        attn_kw_off = dict(
            edge_index=edge_index,
            edge_values=edge_values,
            positions=positions,
            save_attention=False,
            attn_mask=off_attn_mask,
            edge_feats=off_edge_feats,
            prep_block_node=off_prep_block,
            prep_matrix_node=off_prep_matrix,
            eager_attention=eager_attention,
        )
        if M_off > 0:
            if n_l == 0:
                off_stream = self._build_off_strip(
                    self._diag_tokens_to_full_leaf(h_diag, B, K, L, C_h), B, K, L, C_h, M_off
                )
            else:
                for i in range(n_l):
                    blk_d = self.blocks[i]
                    x_mid_d = blk_d.forward_attn(h_diag, **attn_kw_diag)
                    if self.highway_ffn:
                        rr, cc = self._build_diag_mlp_hw(x_mid_d, off_stream, B, K, L, C_h)
                        if self.ffn_concat_width == 4:
                            gg = self._mlp_global_broadcast(x_mid_d)
                            h_diag = blk_d.forward_mlp(x_mid_d, rr, cc, gg)
                        else:
                            h_diag = blk_d.forward_mlp(x_mid_d, rr, cc)
                    else:
                        h_diag = blk_d.forward_mlp(x_mid_d)
                    strip = self._build_off_strip(
                        self._diag_tokens_to_full_leaf(h_diag, B, K, L, C_h), B, K, L, C_h, M_off
                    )
                    off_in = strip if off_stream is None else strip + off_stream
                    blk_o = self.off_diag_blocks[i]
                    x_mid_o = blk_o.forward_attn(off_in, **attn_kw_off)
                    if self.highway_ffn:
                        orr, occ = self._build_off_mlp_hw(h_diag, x_mid_o, B, K, L, C_h, M_off)
                        if self.ffn_concat_width == 4:
                            og = self._mlp_global_broadcast(x_mid_o)
                            off_stream = blk_o.forward_mlp(x_mid_o, orr, occ, og)
                        else:
                            off_stream = blk_o.forward_mlp(x_mid_o, orr, occ)
                    else:
                        off_stream = blk_o.forward_mlp(x_mid_o)
        else:
            for block in self.blocks:
                if self.highway_ffn:
                    x_mid = block.forward_attn(h_diag, **attn_kw_diag)
                    rr, cc = self._build_diag_mlp_hw(x_mid, None, B, K, L, C_h)
                    if self.ffn_concat_width == 4:
                        gg = self._mlp_global_broadcast(x_mid)
                        h_diag = block.forward_mlp(x_mid, rr, cc, gg)
                    else:
                        h_diag = block.forward_mlp(x_mid, rr, cc)
                else:
                    h_diag = block(h_diag, **attn_kw_diag)
        return h_diag, off_stream

    def forward(
        self,
        x,
        edge_index,
        edge_values,
        save_attention=False,
        precomputed_leaf_connectivity=None,
        global_features=None,
        eager_attention: bool = False,
    ):
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
                    off_diag_dense_attention=self.off_diag_dense_attention,
                    diag_dense_attention=self.diag_dense_attention,
                )

        B, N_h, C_h = h.shape
        L = self.leaf_size
        K = N_h // L
        if K > MAX_NUM_LEAVES:
            raise ValueError(
                f"LeafOnlyNet: N/leaf_size={K} exceeds MAX_NUM_LEAVES ({MAX_NUM_LEAVES}) "
                f"(N={N_h}). Increase MAX_MIXED_SIZE in leafonly.config or use fewer nodes."
            )

        if self.diag_token_pool > 1:
            if self.diag_dense_attention:
                if edge_feats is not None:
                    edge_feats = pool_leaf_edge_feats(edge_feats, L, self.diag_token_pool)
            else:
                if attn_mask is not None:
                    attn_mask = pool_leaf_attn_mask(attn_mask, L, self.diag_token_pool)
                if edge_feats is not None:
                    edge_feats = pool_leaf_edge_feats(edge_feats, L, self.diag_token_pool)

        off_attn_mask = off_edge_feats = None
        M_off = self.num_h_off
        if M_off > 0:
            # After strip sum + optional token mean-pool: (B*M_off, Ls_off, C), num_blocks=1; masks match Ls_off.
            if not self.off_diag_dense_attention:
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
            oe = off_oe_pre.to(device=h.device, dtype=h.dtype)
            if self.off_diag_dense_attention:
                if self.off_token_pool > 1:
                    oe = pool_leaf_edge_feats(oe, L, self.off_token_pool)
                Ls = self.leaf_apply_off
                # Per-context oe is (M,Ls,Ls,4); batched precomputed is (B,M,Ls,Ls,4) — do not unsqueeze again.
                if oe.dim() == 4:
                    oe5 = oe.unsqueeze(0).expand(B, M_off, -1, -1, -1)
                else:
                    oe5 = oe.contiguous()
                    if oe5.shape[0] == 1 and B > 1:
                        oe5 = oe5.expand(B, -1, -1, -1, -1)
                off_attn_mask = None
                off_edge_feats = oe5.reshape(B * M_off, 1, Ls, Ls, 4)
            else:
                om_m = int(off_om_pre.shape[0]) if off_om_pre.dim() == 3 else int(off_om_pre.shape[1])
                if om_m != M_off:
                    raise ValueError(
                        f"LeafOnlyNet: off attn_mask has M={om_m}, expected num_h_off={M_off}."
                    )
                om = off_om_pre.to(device=h.device, dtype=h.dtype)
                if self.off_token_pool > 1:
                    om = pool_leaf_attn_mask(om, L, self.off_token_pool)
                    oe = pool_leaf_edge_feats(oe, L, self.off_token_pool)
                if om.dim() == 3:
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

        # Frozen KV extras (+1 / +2): from post-embed h; diagonal stream may be pooled (Ls_diag).
        h_diag = self._pool_full_leaf_to_diag_tokens(h, B, K, L, C_h)
        h_diag, off_stream = self._apply_transformer_stacks(
            h,
            h_diag,
            B,
            K,
            L,
            C_h,
            edge_index,
            edge_values,
            positions,
            attn_mask,
            edge_feats,
            off_attn_mask,
            off_edge_feats,
            save_attention,
            eager_attention,
        )

        if self.use_highways:
            off_for_last = None
            if M_off > 0 and off_stream is not None:
                off_for_last = off_stream.view(B, M_off, self.leaf_apply_off, C_h)
            self._last_row_highway, self._last_col_highway = self._build_token_highways(
                h_ref=h,
                h_diag=h_diag,
                off_stream=off_for_last,
                B=B,
                K=K,
                L=L,
                C_h=C_h,
                M_off=M_off,
            )
        else:
            self._last_row_highway, self._last_col_highway = None, None

        diag_blocks = self._get_leaf_blocks(h_diag, mode="diagonal")

        if M_off > 0:
            h_off = off_stream.view(B, M_off, self.leaf_apply_off, C_h)
            off_diag_blocks = self._get_leaf_blocks(h_off, mode="off-diagonal")
        else:
            off_diag_blocks = torch.empty(
                (B, 0, self.leaf_apply_off, self.leaf_apply_off), device=h.device, dtype=h.dtype
            )

        h_full = self._diag_tokens_to_full_leaf(h_diag, B, K, L, C_h)
        jacobi_scale = self._get_jacobi_scale(h_full)

        node_U = self.node_u(h_full)
        node_V = self.node_v(h_full)

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
    ``all_values[perm]`` and ``sparse_bsr_tensor``. Layout matches ``LeafOnlyNet.forward`` output:
    tight pack [diag active][off][node_U active][node_V active][jacobi optional].
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
    M_h = NUM_HMATRIX_OFF_BLOCKS
    off_size = M_h * La_o * La_o
    node_size_active = num_leaves * L * La_o

    diag_blocks = precond_packed[:, :diag_size_active].view(1, num_leaves, La_d, La_d)

    off_diag_blocks = node_U = node_V = None
    idx = diag_size_active
    if M_h > 0:
        off_diag_blocks = precond_packed[:, idx : idx + off_size].view(1, M_h, La_o, La_o)
        idx += off_size
    node_U = precond_packed[:, idx : idx + node_size_active].view(1, num_leaves, L, La_o)
    idx += node_size_active
    node_V = precond_packed[:, idx : idx + node_size_active].view(1, num_leaves, L, La_o)
    idx += node_size_active

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
        dummy_vals = torch.arange(E, device=device, dtype=torch.float32)
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
        if num_leaves < MAX_NUM_LEAVES:
            x_proj_U = F.pad(x_proj_U, (0, 0, 0, 0, 0, MAX_NUM_LEAVES - num_leaves))
            x_proj_V = F.pad(x_proj_V, (0, 0, 0, 0, 0, MAX_NUM_LEAVES - num_leaves))

        x_row_strips = torch.einsum("mk,bkld->bmld", Wr_sum, x_proj_U)
        x_col_strips = torch.einsum("mk,bkld->bmld", Wc_sum, x_proj_V)

        flat = B * M_h
        Br = off_diag_blocks.reshape(flat, La_o, La_o)
        xr = x_row_strips.reshape(flat, La_o, K_dim)
        xc = x_col_strips.reshape(flat, La_o, K_dim)
        y_r_coarse = torch.bmm(Br, xc).view(B, M_h, La_o, K_dim)
        y_c_coarse = torch.bmm(Br.transpose(-1, -2), xr).view(B, M_h, La_o, K_dim)

        ridx, cidx, gidx = _hm_prolong_scatter_indices(x.device)
        y_r_leaves = torch.zeros(B, MAX_NUM_LEAVES, La_o, K_dim, device=x.device, dtype=x.dtype)
        y_c_leaves = torch.zeros(B, MAX_NUM_LEAVES, La_o, K_dim, device=x.device, dtype=x.dtype)

        y_r_leaves.index_add_(1, ridx, y_r_coarse[:, gidx, :, :])
        y_c_leaves.index_add_(1, cidx, y_c_coarse[:, gidx, :, :])

        y_r_final = torch.matmul(node_U, y_r_leaves[:, :num_leaves])
        y_c_final = torch.matmul(node_V, y_c_leaves[:, :num_leaves])

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
    ``num_leaves_active <= MAX_NUM_LEAVES``. ``precond_packed`` is the usual ``LeafOnlyNet.forward`` output
    (tight pack for the active leaf count).

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
    M_h = NUM_HMATRIX_OFF_BLOCKS
    off_size = M_h * La_o * La_o
    node_size_active = num_leaves_active * L * La_o

    diag_blocks = precond_packed[:, :diag_size_active].view(B, num_leaves_active, La_d, La_d)
    x_leaves = x.view(B, num_leaves_active, L, K_dim)
    x_pool_d = (
        x_leaves
        if pool_d == 1
        else x_leaves.view(B, num_leaves_active, La_d, pool_d, K_dim).mean(dim=3)
    )
    y_pool = torch.matmul(diag_blocks, x_pool_d)
    y_leaves = y_pool if pool_d == 1 else y_pool.repeat_interleave(pool_d, dim=2)

    idx = diag_size_active
    if M_h > 0:
        off_diag_blocks = precond_packed[:, idx : idx + off_size].view(B, M_h, La_o, La_o)
        idx += off_size
        node_U_active = precond_packed[:, idx : idx + node_size_active].view(B, num_leaves_active, L, La_o)
        idx += node_size_active
        node_V_active = precond_packed[:, idx : idx + node_size_active].view(B, num_leaves_active, L, La_o)
        idx += node_size_active

        na = int(num_leaves_active)
        valid = ((HM_R0_CPU + HM_S_CPU) <= na) & ((HM_C0_CPU + HM_S_CPU) <= na)
        valid = valid.to(device=off_diag_blocks.device, dtype=off_diag_blocks.dtype).view(1, M_h, 1, 1)
        off_diag_blocks = off_diag_blocks * valid

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
        idx += 2 * node_size_active

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
    off_size = M_h * La_o * La_o
    node_size_active = num_leaves * leaf_size * La_o

    diag_blocks = precond_packed[:, :diag_size_active].view(B, num_leaves, La_d, La_d)
    x_leaves = x.view(B, num_leaves, leaf_size, K_dim)
    y_leaves = out.view(B, num_leaves, leaf_size, K_dim)
    torch.matmul(diag_blocks, x_leaves, out=y_leaves)

    idx = diag_size_active
    if M_h > 0:
        off_diag_blocks = precond_packed[:, idx : idx + off_size].view(B, M_h, La_o, La_o)
        idx += off_size
        node_U = precond_packed[:, idx : idx + node_size_active].view(B, num_leaves, leaf_size, La_o)
        idx += node_size_active
        node_V = precond_packed[:, idx : idx + node_size_active].view(B, num_leaves, leaf_size, La_o)
        idx += node_size_active

        na = int(num_leaves)
        if na < MAX_NUM_LEAVES:
            off_diag_blocks = off_diag_blocks * ws["off_valid"]

        node_U_active = node_U
        node_V_active = node_V

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
        idx += 2 * node_size_active

    if precond_packed.shape[1] > idx:
        if jacobi_inv_diag is None:
            raise ValueError("jacobi_inv_diag is required when jacobi_scale is present.")
        if jacobi_inv_diag.shape[:2] != (B, N):
            raise ValueError(f"jacobi_inv_diag shape {jacobi_inv_diag.shape} does not match (B,N)=({B},{N}).")
        jacobi_scale = precond_packed[:, idx:][:, :N]
        out.addcmul_(x, (jacobi_scale * jacobi_inv_diag).unsqueeze(-1))
    return out
