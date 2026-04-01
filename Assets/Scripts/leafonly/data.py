import math
from pathlib import Path

import numpy as np
import torch

from .config import ATTENTION_HOPS, LEAF_SIZE

# Imported where needed for H off masks (hmatrix does not import data).
def _hmatrix_static():
    from . import hmatrix as _hm

    return _hm.HM_R0_CPU, _hm.HM_C0_CPU, _hm.HM_S_CPU


NODE_DTYPE = np.dtype(
    [
        ("position", "3<f4"),
        ("velocity", "3<f4"),
        ("face_vels", "6<f4"),
        ("mass", "<f4"),
        ("layer", "<u4"),
        ("morton", "<u4"),
        ("active", "<u4"),
    ]
)


def _read_num_nodes(frame_path):
    p = Path(frame_path)
    meta = p / "meta.txt"
    if meta.exists():
        with open(meta, "r") as f:
            for line in f:
                if "numNodes" in line or "num_nodes" in line:
                    return int(line.split(":")[1].strip())
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

        positions = np.asarray(raw_nodes["position"], dtype=np.float32)
        layer = np.asarray(raw_nodes["layer"], dtype=np.float32)
        mass = np.asarray(raw_nodes["mass"], dtype=np.float32)
        diag_map = np.zeros(num_nodes, dtype=np.float32)
        for r, c, v in zip(rows, cols, vals):
            if r == c:
                diag_map[r] = v

        positions_n = positions / 1024.0
        layer_n = layer / 4.0

        row_sums = np.zeros(num_nodes, dtype=np.float64)
        col_sums = np.zeros(num_nodes, dtype=np.float64)
        for r, c, v in zip(rows, cols, vals):
            row_sums[r] += abs(v)
            col_sums[c] += abs(v)
        max_row = float(np.max(row_sums)) if row_sums.size else 1.0
        max_col = float(np.max(col_sums)) if col_sums.size else 1.0
        scale_A = float(min(max_row, max_col))
        if scale_A <= 0.0:
            scale_A = 1.0

        dg_path = frame_path / "diffusion_gradient.bin"
        if dg_path.exists():
            diffusion_grad = np.fromfile(dg_path, dtype=np.float32).reshape(num_nodes, 3)
        else:
            diffusion_grad = np.zeros((num_nodes, 3), dtype=np.float32)

        diag_map_n = diag_map / scale_A
        vals_n = vals / scale_A
        mass_sym = np.sign(mass) * np.log1p(np.abs(mass))
        mass_mean = float(np.mean(mass_sym))
        mass_std = float(np.std(mass_sym)) + 1e-8
        mass_n = (mass_sym - mass_mean) / mass_std
        diff_sym = np.sign(diffusion_grad) * np.log1p(np.abs(diffusion_grad))
        diff_mean = np.mean(diff_sym, axis=0)
        diff_std = np.std(diff_sym, axis=0) + 1e-8
        diffusion_grad_n = (diff_sym - diff_mean) / diff_std

        global_features = np.array(
            [
                math.log2(max(1, num_nodes)),
                0.0,
                mass_mean,
                mass_std,
                float(np.mean(diag_map_n)),
                float(np.std(diag_map_n)),
                float(diff_mean[0]),
                float(diff_std[0]),
                float(diff_mean[1]),
                float(diff_std[1]),
                float(diff_mean[2]),
                float(diff_std[2]),
            ],
            dtype=np.float32,
        )

        n_float = 9
        x = np.zeros((num_nodes, n_float), dtype=np.float32)
        x[:, :3] = positions_n
        x[:, 3] = layer_n
        x[:, 4] = mass_n
        x[:, 5:8] = diffusion_grad_n
        x[:, 8] = diag_map_n

        return {
            "x": torch.from_numpy(x).float(),
            "edge_index": torch.stack([torch.from_numpy(rows.astype(np.int64)), torch.from_numpy(cols.astype(np.int64))]),
            "edge_values": torch.from_numpy(vals_n.astype(np.float32)),
            "num_nodes": int(num_nodes),
            "scale_A": scale_A,
            "frame_path": str(frame_path),
            "global_features": torch.from_numpy(global_features).float(),
        }


def _edge_feats_LxL_mean_scatter(
    r_l: torch.Tensor,
    c_l: torch.Tensor,
    edge_feats_flat: torch.Tensor,
    leaf_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Mean-aggregate [dx(3), w(1)] into (L, L, 4) by local row/col index."""
    L = int(leaf_size)
    if r_l.numel() == 0:
        return torch.zeros(L, L, 4, device=device, dtype=dtype)
    idx = r_l.long() * L + c_l.long()
    sum_flat = torch.zeros(L * L, 4, device=device, dtype=dtype)
    cnt_flat = torch.zeros(L * L, device=device, dtype=dtype)
    f = edge_feats_flat.to(device=device, dtype=dtype)
    sum_flat.index_add_(0, idx, f)
    cnt_flat.index_add_(0, idx, torch.ones((idx.shape[0],), device=device, dtype=dtype))
    return (sum_flat / cnt_flat.unsqueeze(-1).clamp(min=1.0)).view(L, L, 4)


def _mean_A_ij_per_LxL_cell(
    r_l: torch.Tensor,
    c_l: torch.Tensor,
    edge_vals: torch.Tensor,
    leaf_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Mean ``A_ij`` per local (i, j) cell; cells with no edge stay 0."""
    L = int(leaf_size)
    if r_l.numel() == 0:
        return torch.zeros(L, L, device=device, dtype=dtype)
    idx = r_l.long() * L + c_l.long()
    sum_v = torch.zeros(L * L, device=device, dtype=dtype)
    cnt = torch.zeros(L * L, device=device, dtype=dtype)
    v = edge_vals.to(device=device, dtype=dtype)
    sum_v.index_add_(0, idx, v)
    cnt.index_add_(0, idx, torch.ones(idx.shape[0], device=device, dtype=dtype))
    mean_flat = sum_v / cnt.clamp(min=1.0)
    mean_flat = mean_flat * (cnt > 0).to(dtype)
    return mean_flat.view(L, L)


def _materialize_block_attn_and_edge_feats(
    reachable, b_l, r_l, c_l, edge_feats_flat, num_blocks, leaf_size, device, dtype
):
    L = leaf_size
    attn_mask = torch.zeros(num_blocks, L, L + 1, device=device, dtype=dtype)
    attn_mask[:, :, :L] = reachable
    attn_mask[:, :, L] = 1.0
    edge_feats = torch.zeros(num_blocks, L, L + 1, 4, device=device, dtype=dtype)
    if b_l.numel() > 0:
        for b in range(int(num_blocks)):
            sel = b_l == b
            if not sel.any():
                continue
            edge_feats[b, :L, :L, :] = _edge_feats_LxL_mean_scatter(
                r_l[sel], c_l[sel], edge_feats_flat[sel], L, device, dtype
            )
    return attn_mask, edge_feats


def _full_graph_hop_reachable(edge_index, num_nodes, num_hops, device, dtype):
    """
    Cumulative n-hop connectivity on the full node graph (before leaf tiling).

    Same semantics as the previous per-block hop loop: start from I, repeat num_hops times:
    reachable += cur; cur = cur @ A (boolean 0/1, clamped).

    Adjacency is stored as sparse COO; propagation uses torch.sparse.mm(sparse, dense) so we
    do not materialize a dense adjacency matrix (only the evolving dense reachability cur).
    """
    rows, cols = edge_index[0].long(), edge_index[1].long()
    valid = (rows >= 0) & (cols >= 0) & (rows < num_nodes) & (cols < num_nodes)
    rows, cols = rows[valid], cols[valid]

    reachable = torch.eye(num_nodes, device=device, dtype=dtype)
    if num_hops <= 0:
        return reachable

    if rows.numel() == 0:
        return reachable

    # MPS: sparse COO / sparse.mm is not reliably supported; use dense adjacency (typical N is modest).
    if device.type == "mps":
        A_dense = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)
        A_dense[rows, cols] = 1.0
        cur = A_dense.clone()
        for _ in range(num_hops):
            reachable = (reachable + cur).clamp(0.0, 1.0)
            cur = (cur @ A_dense).clamp(0.0, 1.0)
        return reachable

    ones = torch.ones(rows.shape[0], device=device, dtype=dtype)
    A_sp = torch.sparse_coo_tensor(
        torch.stack([rows, cols]),
        ones,
        (num_nodes, num_nodes),
        device=device,
        dtype=dtype,
    ).coalesce()
    # Collapse parallel edges to unweighted {0,1}
    A_sp = torch.sparse_coo_tensor(
        A_sp.indices(),
        (A_sp.values() > 0).to(dtype),
        (num_nodes, num_nodes),
        device=device,
        dtype=dtype,
    ).coalesce()

    eye = torch.eye(num_nodes, device=device, dtype=dtype)
    cur = torch.sparse.mm(A_sp, eye).clamp(0.0, 1.0)

    for _ in range(num_hops):
        reachable = (reachable + cur).clamp(0.0, 1.0)
        cur = torch.sparse.mm(A_sp, cur).clamp(0.0, 1.0)

    return reachable


def _slice_leaf_blocks_from_global_reachable(reachable, num_blocks, leaf_size):
    """
    reachable: (N, N) with N = num_blocks * leaf_size.

    Returns:
      diag: (num_blocks, L, L) — within-leaf slices (global n-hop restricted to same leaf)
      off:  (P, L, L) — upper-triangular cross-leaf slices (row leaf r, col leaf c), P = K(K-1)/2
    """
    L = leaf_size
    nb = num_blocks
    r4 = reachable.view(nb, L, nb, L)
    b = torch.arange(nb, device=reachable.device, dtype=torch.long)
    diag = r4[b, :, b, :]

    r_idx, c_idx = torch.triu_indices(nb, nb, offset=1, device=reachable.device)
    if r_idx.numel() == 0:
        off = reachable.new_zeros((0, L, L))
    else:
        off = r4[r_idx, :, c_idx, :]
    return diag, off


def _diag_in_block_edge_features(edge_index, edge_values, positions, leaf_size, num_blocks, device, dtype):
    """Physics edge features for same-leaf edges only (mask connectivity comes from global reachability)."""
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
    edge_feats_flat = torch.cat([dx, w.unsqueeze(1)], dim=1)
    return b_l, r_l, c_l, edge_feats_flat


def _reachable_and_diag_materialize(
    edge_index, edge_values, positions, leaf_size, device, dtype, num_hops
):
    """One full-graph hop pass; diagonal masks/feats; returns (R, dm, df, num_blocks) or (None, None, None, 0)."""
    N = positions.shape[0]
    num_blocks = N // leaf_size
    if num_blocks == 0:
        return None, None, None, 0
    R = _full_graph_hop_reachable(edge_index, N, num_hops, device, dtype)
    diag_reachable, _ = _slice_leaf_blocks_from_global_reachable(R, num_blocks, leaf_size)
    b_l, r_l, c_l, edge_feats_flat = _diag_in_block_edge_features(
        edge_index, edge_values, positions, leaf_size, num_blocks, device, dtype
    )
    dm, df = _materialize_block_attn_and_edge_feats(
        diag_reachable, b_l, r_l, c_l, edge_feats_flat, num_blocks, leaf_size, device, dtype
    )
    return R, dm, df, num_blocks


def build_diag_leaf_connectivity(
    edge_index, edge_values, positions, leaf_size, device, dtype=torch.float32, num_hops=ATTENTION_HOPS
):
    """
    Per-leaf diagonal tiles only: n-hop reachability restricted to same leaf + edge features on
    in-leaf edges. (H off masks come from the same reachability in build_leaf_block_connectivity.)
    """
    _, dm, df, num_blocks = _reachable_and_diag_materialize(
        edge_index, edge_values, positions, leaf_size, device, dtype, num_hops
    )
    if num_blocks == 0:
        return None, None
    return dm, df


def build_hmatrix_off_attn_masks_from_reachable(
    reachable: torch.Tensor,
    r0: torch.Tensor,
    c0: torch.Tensor,
    s: torch.Tensor,
    num_blocks: int,
    leaf_size: int,
    dtype_out: torch.dtype,
) -> torch.Tensor:
    """
    Aggregate global n-hop reachability into H-matrix off-block attention masks.

    For each admissible tile (r0, c0, S), OR together all L×L leaf-pair slices
    R[rr*L:(rr+1)*L, cc*L:(cc+1)*L] for rr in [r0, r0+S), cc in [c0, c0+S).

    Returns (M_off, L, L+1); last key column is 1 (block node), matching diagonal layout.
    """
    L = int(leaf_size)
    K = int(num_blocks)
    N = K * L
    if reachable.shape != (N, N):
        raise ValueError(f"reachable must be ({N}, {N}), got {tuple(reachable.shape)}")
    device = reachable.device
    M = int(r0.shape[0])
    if M == 0:
        return torch.zeros(0, L, L + 1, device=device, dtype=dtype_out)
    r0_l = r0.long().to(device)
    c0_l = c0.long().to(device)
    s_l = s.long().to(device)
    out = torch.zeros(M, L, L + 1, device=device, dtype=dtype_out)
    for m in range(M):
        rr0 = int(r0_l[m].item())
        cc0 = int(c0_l[m].item())
        sval = int(s_l[m].item())
        subs = []
        for rr in range(rr0, rr0 + sval):
            for cc in range(cc0, cc0 + sval):
                if 0 <= rr < K and 0 <= cc < K:
                    subs.append(reachable[rr * L : (rr + 1) * L, cc * L : (cc + 1) * L])
        if subs:
            acc = torch.stack(subs, dim=0).amax(dim=0).to(dtype_out)
            out[m, :, :L] = acc
        out[m, :, L] = 1.0
    return out


def build_hmatrix_off_dense_rpe_from_positions(
    positions: torch.Tensor,
    r0: torch.Tensor,
    c0: torch.Tensor,
    s: torch.Tensor,
    num_blocks: int,
    leaf_size: int,
    dtype: torch.dtype,
    edge_index: torch.Tensor | None = None,
    edge_values: torch.Tensor | None = None,
    *,
    with_block_key_column: bool = True,
) -> torch.Tensor:
    """
    Per H-tile (r0, c0, S): dense relative geometry for the L×L leaf-local attention grid.

    Row-strip nodes with local index i are ``pos[rr*L+i]`` for ``rr ∈ [r0, r0+S)``; column-strip
    similarly ``cc ∈ [c0, c0+S)``. For each (i, j), store the mean over all S×S pairwise differences
    ``pos[col] - pos[row]``, which equals ``mean_col(pos at slot j) - mean_row(pos at slot i)``.

    When ``edge_index`` / ``edge_values`` are provided, channel 3 is the mean ``A_ij`` over direct
    graph edges whose endpoints fall in that tile's row- and column-strips (same filtering as the
    sparse off-diag path); local (i, j) pairs with no such edge get 0. Block column (L) stays zero.
    """
    L = int(leaf_size)
    K = int(num_blocks)
    N = K * L
    device = positions.device
    pos = positions.to(device=device, dtype=dtype)
    if pos.shape[0] < N:
        raise ValueError(f"positions rows {pos.shape[0]} < N={N} (K*L)")
    pos = pos[:, :3]
    M = int(r0.shape[0])
    if M == 0:
        if with_block_key_column:
            return torch.zeros(0, L, L + 1, 4, device=device, dtype=dtype)
        return torch.zeros(0, L, L, 4, device=device, dtype=dtype)
    r0_l = r0.long().to(device)
    c0_l = c0.long().to(device)
    s_l = s.long().to(device)
    li = torch.arange(L, device=device, dtype=torch.long)
    if with_block_key_column:
        out = torch.zeros(M, L, L + 1, 4, device=device, dtype=dtype)
    else:
        out = torch.zeros(M, L, L, 4, device=device, dtype=dtype)
    for m in range(M):
        rr0 = int(r0_l[m].item())
        cc0 = int(c0_l[m].item())
        sv = int(s_l[m].item())
        rr_all = torch.arange(rr0, rr0 + sv, device=device, dtype=torch.long)
        cc_all = torch.arange(cc0, cc0 + sv, device=device, dtype=torch.long)
        # Static H-tiles are built for MAX_NUM_LEAVES; intersect strips with active leaves [0, K).
        rr = rr_all[(rr_all >= 0) & (rr_all < K)]
        cc = cc_all[(cc_all >= 0) & (cc_all < K)]
        if rr.numel() == 0 or cc.numel() == 0:
            continue
        idx_r = rr[:, None] * L + li[None, :]
        idx_c = cc[:, None] * L + li[None, :]
        pr = pos[idx_r].mean(dim=0)
        pc = pos[idx_c].mean(dim=0)
        out[m, :L, :L, :3] = pc.unsqueeze(0) - pr.unsqueeze(1)

    if edge_index is not None and edge_values is not None:
        rows = edge_index[0].long()
        cols = edge_index[1].long()
        valid = (rows >= 0) & (cols >= 0) & (rows < N) & (cols < N)
        rows = rows[valid]
        cols = cols[valid]
        ev = edge_values[valid].to(dtype=dtype)
        if rows.numel() > 0:
            br = rows // L
            bc = cols // L
            rl = rows % L
            cl = cols % L
            for m in range(M):
                rr0 = int(r0_l[m].item())
                cc0 = int(c0_l[m].item())
                sv = int(s_l[m].item())
                strip_r = (br >= rr0) & (br < rr0 + sv)
                strip_c = (bc >= cc0) & (bc < cc0 + sv)
                msk = strip_r & strip_c
                if msk.any():
                    out[m, :L, :L, 3] = _mean_A_ij_per_LxL_cell(
                        rl[msk], cl[msk], ev[msk], L, device, dtype
                    )
    return out


def build_diag_dense_edge_feats_from_positions(
    edge_index: torch.Tensor,
    edge_values: torch.Tensor,
    positions: torch.Tensor,
    num_blocks: int,
    leaf_size: int,
    dtype: torch.dtype,
    *,
    with_block_key_column: bool = True,
) -> torch.Tensor:
    """
    Per diagonal leaf block: dense L×L features [Δx, Δy, Δz, A_ij] for full (query i, key j) softmax.

    Geometry is pairwise ``pos[b*L+j] - pos[b*L+i]`` (same semantics as sparse in-leaf edges).
    Channel 3 is mean ``A_ij`` over direct graph edges in that cell, 0 when no edge (matches dense
    off-diagonal strip aggregation, but with per-node offsets because row/column strips coincide).
    """
    L = int(leaf_size)
    K = int(num_blocks)
    N = K * L
    device = positions.device
    pos = positions.to(device=device, dtype=dtype)
    if pos.shape[0] < N:
        raise ValueError(f"positions rows {pos.shape[0]} < N={N} (K*L)")
    pos = pos[:, :3]
    if with_block_key_column:
        out = torch.zeros(K, L, L + 1, 4, device=device, dtype=dtype)
    else:
        out = torch.zeros(K, L, L, 4, device=device, dtype=dtype)
    li = torch.arange(L, device=device, dtype=torch.long)
    for b in range(K):
        idx = b * L + li
        p = pos[idx]
        out[b, :L, :L, :3] = p.unsqueeze(0) - p.unsqueeze(1)

    rows = edge_index[0].long()
    cols = edge_index[1].long()
    valid = (rows >= 0) & (cols >= 0) & (rows < N) & (cols < N)
    rows = rows[valid]
    cols = cols[valid]
    ev = edge_values[valid].to(dtype=dtype)
    if rows.numel() == 0:
        return out
    br = rows // L
    bc = cols // L
    rl = rows % L
    cl = cols % L
    for b in range(K):
        in_b = (br == b) & (bc == b)
        if in_b.any():
            out[b, :L, :L, 3] = _mean_A_ij_per_LxL_cell(rl[in_b], cl[in_b], ev[in_b], L, device, dtype)
    return out


def build_hmatrix_off_edge_feats_from_edges(
    edge_index: torch.Tensor,
    edge_values: torch.Tensor,
    positions: torch.Tensor,
    r0: torch.Tensor,
    c0: torch.Tensor,
    s: torch.Tensor,
    num_blocks: int,
    leaf_size: int,
    dtype: torch.dtype,
    *,
    dense_position_rpe: bool = False,
    with_block_key_column: bool = True,
) -> torch.Tensor:
    """
    Per H-tile (r0, c0, S): off-diagonal edge features for LeafBlockAttention ``edge_gate``.

    If ``dense_position_rpe`` is False (default): aggregate direct graph edges whose src lies in the
    row-strip leaves and dst in the col-strip leaves into local (L, L, 4) with [Δx(3), A_ij(1)],
    mean over duplicates (sparse edge_index).

    If True: fill [Δx, Δy, Δz] from dense strip geometry and channel 3 from ``edge_values`` (mean per
    local (i,j) over edges in the tile; 0 where no direct edge), matching the sparse path's strip filter
    (see ``build_hmatrix_off_dense_rpe_from_positions``).

    Layout matches diagonal per-leaf cells; if ``with_block_key_column`` is True, last key column (block node) stays zero.
    """
    if dense_position_rpe:
        return build_hmatrix_off_dense_rpe_from_positions(
            positions,
            r0,
            c0,
            s,
            num_blocks,
            leaf_size,
            dtype,
            edge_index=edge_index,
            edge_values=edge_values,
            with_block_key_column=with_block_key_column,
        )

    L = int(leaf_size)
    K = int(num_blocks)
    N = K * L
    device = edge_index.device
    pos = positions.to(device=device, dtype=dtype)
    rows = edge_index[0].long()
    cols = edge_index[1].long()
    valid = (rows >= 0) & (cols >= 0) & (rows < N) & (cols < N)
    rows = rows[valid]
    cols = cols[valid]
    if rows.numel() == 0:
        M = int(r0.shape[0])
        return torch.zeros(M, L, L + 1, 4, device=device, dtype=dtype)
    ev = edge_values[valid].to(dtype=dtype)
    br = rows // L
    bc = cols // L
    rl = rows % L
    cl = cols % L
    pos_r = pos[rows]
    pos_c = pos[cols]
    dx = pos_c - pos_r
    feats = torch.cat([dx, ev.unsqueeze(1)], dim=1)

    M = int(r0.shape[0])
    if M == 0:
        return torch.zeros(0, L, L + 1, 4, device=device, dtype=dtype)
    r0_l = r0.long().to(device)
    c0_l = c0.long().to(device)
    s_l = s.long().to(device)
    out = torch.zeros(M, L, L + 1, 4, device=device, dtype=dtype)
    for m in range(M):
        rr0 = int(r0_l[m].item())
        cc0 = int(c0_l[m].item())
        sv = int(s_l[m].item())
        strip_r = (br >= rr0) & (br < rr0 + sv)
        strip_c = (bc >= cc0) & (bc < cc0 + sv)
        msk = strip_r & strip_c
        if not msk.any():
            continue
        out[m, :L, :L, :] = _edge_feats_LxL_mean_scatter(rl[msk], cl[msk], feats[msk], L, device, dtype)
    return out


def build_leaf_block_connectivity(
    edge_index,
    edge_values,
    positions,
    leaf_size,
    device,
    dtype=torch.float32,
    num_hops=ATTENTION_HOPS,
    *,
    off_diag_dense_attention: bool = False,
    diag_dense_attention: bool = False,
):
    """
    Returns (diag_mask, diag_feats, off_attn_mask, off_edge_feats).

    Diagonal: same-leaf n-hop reachability and in-leaf edge features (mean if multiple edges per cell),
    unless ``diag_dense_attention`` is True: then ``diag_mask`` is None (no binary mask; full L×L softmax)
    and ``diag_feats`` are dense [Δx, Δy, Δz, A_ij] per (i, j), shape (K, L, L, 4).

    Off-diagonal: H-tile reachability masks and edge features from direct edges across strips, unless
    ``off_diag_dense_attention`` is True: then ``off_attn_mask`` is None and ``off_edge_feats`` are dense
    strip RPE with shape (M, L, L, 4) (no block-key column).

    When both dense flags are True, skips n-hop reachability and H-tile mask construction.
    """
    N = int(positions.shape[0])
    L = int(leaf_size)
    num_blocks = N // leaf_size
    HM_R0_CPU, HM_C0_CPU, HM_S_CPU = _hmatrix_static()
    dd = bool(diag_dense_attention)
    od = bool(off_diag_dense_attention)

    if num_blocks == 0:
        zm = torch.zeros(0, L, L + 1, device=device, dtype=dtype)
        zf = torch.zeros(0, L, L + 1, 4, device=device, dtype=dtype)
        return None, None, zm, zf

    if dd and od:
        dm = None
        df = build_diag_dense_edge_feats_from_positions(
            edge_index,
            edge_values,
            positions,
            num_blocks,
            leaf_size,
            dtype,
            with_block_key_column=False,
        )
        hm_r0 = HM_R0_CPU.to(device)
        hm_c0 = HM_C0_CPU.to(device)
        hm_s = HM_S_CPU.to(device)
        om = None
        oe = build_hmatrix_off_dense_rpe_from_positions(
            positions,
            hm_r0,
            hm_c0,
            hm_s,
            num_blocks,
            leaf_size,
            dtype,
            edge_index=edge_index,
            edge_values=edge_values,
            with_block_key_column=False,
        )
        return dm, df, om, oe

    R, dm, df, num_blocks = _reachable_and_diag_materialize(
        edge_index, edge_values, positions, leaf_size, device, dtype, num_hops
    )
    if num_blocks == 0 or R is None:
        zm = torch.zeros(0, L, L + 1, device=device, dtype=dtype)
        zf = torch.zeros(0, L, L + 1, 4, device=device, dtype=dtype)
        return None, None, zm, zf

    if dd:
        dm = None
        df = build_diag_dense_edge_feats_from_positions(
            edge_index,
            edge_values,
            positions,
            num_blocks,
            leaf_size,
            dtype,
            with_block_key_column=False,
        )

    hm_r0 = HM_R0_CPU.to(device)
    hm_c0 = HM_C0_CPU.to(device)
    hm_s = HM_S_CPU.to(device)
    if od:
        om = None
        oe = build_hmatrix_off_edge_feats_from_edges(
            edge_index,
            edge_values,
            positions,
            hm_r0,
            hm_c0,
            hm_s,
            num_blocks,
            leaf_size,
            dtype,
            dense_position_rpe=True,
            with_block_key_column=False,
        )
    else:
        om = build_hmatrix_off_attn_masks_from_reachable(R, hm_r0, hm_c0, hm_s, num_blocks, leaf_size, dtype)
        oe = build_hmatrix_off_edge_feats_from_edges(
            edge_index,
            edge_values,
            positions,
            hm_r0,
            hm_c0,
            hm_s,
            num_blocks,
            leaf_size,
            dtype,
            dense_position_rpe=False,
        )
    return dm, df, om, oe


def most_recent_run_folder(base_path):
    base = Path(base_path)
    if not base.exists():
        return base
    runs = sorted([p for p in base.glob("Run_*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        return base
    return runs[0]
