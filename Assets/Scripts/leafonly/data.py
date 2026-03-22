import math
from pathlib import Path

import numpy as np
import torch

from .config import ATTENTION_HOPS, LEAF_SIZE


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


def _materialize_block_attn_and_edge_feats(
    reachable, b_l, r_l, c_l, edge_feats_flat, num_blocks, leaf_size, device, dtype
):
    L = leaf_size
    attn_mask = torch.zeros(num_blocks, L, L + 1, device=device, dtype=dtype)
    attn_mask[:, :, :L] = reachable
    attn_mask[:, :, L] = 1.0
    edge_feats = torch.zeros(num_blocks, L, L + 1, 4, device=device, dtype=dtype)
    if b_l.numel() > 0:
        edge_feats[b_l, r_l, c_l, :] = edge_feats_flat.to(device=device, dtype=dtype)
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


def _off_diag_pair_edge_features(edge_index, edge_values, positions, leaf_size, num_blocks, device, dtype):
    """
    Scatter lists for cross-leaf edges (block_r < block_c), mapped to upper-triangular pair index.
    Used only to fill edge_feats at matrix entries where a direct edge exists.
    """
    rows, cols = edge_index[0], edge_index[1]
    block_r = rows // leaf_size
    block_c = cols // leaf_size
    L = leaf_size

    r_idx, c_idx = torch.triu_indices(num_blocks, num_blocks, offset=1, device=device)
    P = r_idx.shape[0]
    if P == 0:
        empty = torch.tensor([], device=device, dtype=torch.long)
        zf = torch.zeros(0, 4, device=device, dtype=dtype)
        return empty, empty, empty, zf

    lookup = torch.full((num_blocks, num_blocks), -1, device=device, dtype=torch.long)
    lookup[r_idx, c_idx] = torch.arange(P, device=device)

    off_mask = (block_r < block_c) & (block_c < num_blocks)
    br_off, bc_off = block_r[off_mask], block_c[off_mask]
    p_ids = lookup[br_off, bc_off]
    valid_p = p_ids >= 0
    p_l = p_ids[valid_p]
    rl_off = rows[off_mask][valid_p] % leaf_size
    cl_off = cols[off_mask][valid_p] % leaf_size
    pos_r_off = positions[rows[off_mask][valid_p]]
    pos_c_off = positions[cols[off_mask][valid_p]]
    dx_off = (pos_c_off - pos_r_off).to(device=device, dtype=dtype)
    w_off = edge_values[off_mask][valid_p].to(device=device, dtype=dtype)
    edge_feats_flat_off = torch.cat([dx_off, w_off.unsqueeze(1)], dim=1)
    return p_l, rl_off, cl_off, edge_feats_flat_off


def build_leaf_block_connectivity(
    edge_index, edge_values, positions, leaf_size, device, dtype=torch.float32, num_hops=ATTENTION_HOPS
):
    """
    One global n-hop reachability matrix (ATTENTION_HOPS), then leaf-sized slices for
    diagonal blocks and upper-triangular off-diagonal leaf pairs. Attention masks use those
    slices; edge_feats still come from actual edges in each block/pair.
    """
    N = positions.shape[0]
    num_blocks = N // leaf_size
    if num_blocks == 0:
        return None, None, None, None

    R = _full_graph_hop_reachable(edge_index, N, num_hops, device, dtype)
    diag_reachable, off_reachable = _slice_leaf_blocks_from_global_reachable(R, num_blocks, leaf_size)

    b_l, r_l, c_l, edge_feats_flat = _diag_in_block_edge_features(
        edge_index, edge_values, positions, leaf_size, num_blocks, device, dtype
    )
    diag_mask, diag_feats = _materialize_block_attn_and_edge_feats(
        diag_reachable, b_l, r_l, c_l, edge_feats_flat, num_blocks, leaf_size, device, dtype
    )

    P = off_reachable.shape[0]
    p_l, rl_off, cl_off, edge_feats_off = _off_diag_pair_edge_features(
        edge_index, edge_values, positions, leaf_size, num_blocks, device, dtype
    )
    off_diag_mask, off_diag_feats = _materialize_block_attn_and_edge_feats(
        off_reachable, p_l, rl_off, cl_off, edge_feats_off, P, leaf_size, device, dtype
    )

    return diag_mask, diag_feats, off_diag_mask, off_diag_feats


def most_recent_run_folder(base_path):
    base = Path(base_path)
    if not base.exists():
        return base
    runs = sorted([p for p in base.glob("Run_*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        return base
    return runs[0]
