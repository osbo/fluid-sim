#!/usr/bin/env python3
"""
Cross-check LeafOnly GPU inference / apply against Unity [LeafOnlyParity] logs.

Unity: enable FluidSimulator.debugLeafOnlyPyTorchParity, run one pressure solve; copy console lines
starting with [LeafOnlyParity]. This script prints the same tensor names and summary stats (min, max,
mean, sum_abs, l2) from PyTorch on a recorded frame (default: dataset index 0 = lexicographically
first frame under --data_folder).

Known Unity vs dataset differences (see --match_unity_shader):
  • LiftEmbedding zeros diffusion features (x[:, 5:8]); FluidGraphDataset usually fills them from
    diffusion_gradient.bin.
  • FinalizeNodeStats uses scale_A = max_i(sum_j |A_ij|) (row sums). FluidGraphDataset uses
    min(max_row_sum, max_col_sum). Use --match_unity_shader to align both.

Run (from Assets/Scripts):
  python3 InspectLeafOnlyUnityParity.py --frame 0
  python3 InspectLeafOnlyUnityParity.py --frame 0 --match_unity_shader --weights leaf_only_weights.bytes

Optional: compare apply step with r=all_ones (default) or random (--r random).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import LeafOnly as _leaf_only_script  # noqa: E402

from leafonly.architecture import (  # noqa: E402
    LeafOnlyNet,
    apply_block_diagonal_M_physical,
    default_attention_layout,
    unpack_precond,
)
from leafonly.checkpoint import leaf_only_arch_from_checkpoint, load_leaf_only_weights  # noqa: E402
from leafonly.config import LEAF_SIZE, MAX_MIXED_SIZE, problem_padded_num_nodes  # noqa: E402


def _leaf_align_n(num_nodes_real: int, mode: str) -> int:
    """Unity buffers use ceil(numNodes/LEAF_SIZE) leaves; InspectModel uses floor (truncate)."""
    L = int(LEAF_SIZE)
    n = int(num_nodes_real)
    if mode == "ceil":
        return ((n + L - 1) // L) * L
    if mode == "floor":
        return (n // L) * L
    raise ValueError(mode)
from leafonly.data import NODE_DTYPE, FluidGraphDataset  # noqa: E402


def _summarize(name: str, t: torch.Tensor) -> None:
    x = t.detach().float().cpu().reshape(-1)
    n = int(x.numel())
    if n == 0:
        print(f"[LeafOnlyParity] phase=pytorch tensor={name} n=0 (empty)")
        return
    xmin = float(x.min())
    xmax = float(x.max())
    xmean = float(x.mean())
    sum_abs = float(x.abs().sum())
    l2 = float(torch.sqrt((x * x).sum()))
    print(
        f"[LeafOnlyParity] phase=pytorch tensor={name} n={n} min={xmin:.9g} max={xmax:.9g} "
        f"mean={xmean:.9g} sum_abs={sum_abs:.9g} l2={l2:.9g}"
    )


def _head(name: str, t: torch.Tensor, k: int) -> None:
    x = t.detach().float().cpu().reshape(-1)[:k]
    if x.numel() == 0:
        print(f"[LeafOnlyParity] phase=pytorch tensor={name}_head[0]=")
        return
    parts = ",".join(f"{float(v):.9g}" for v in x)
    print(f"[LeafOnlyParity] phase=pytorch tensor={name}_head[{int(x.numel())}]={parts}")


def _unity_stats_and_global(
    num_nodes: int,
    mass: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match LeafOnlyPreconditioner FinalizeNodeStats + globalFeatBuf (diffusion globals zero)."""
    n = int(num_nodes)
    sign_m = np.sign(mass.astype(np.float64))
    log_m = sign_m * np.log1p(np.abs(mass.astype(np.float64)))
    mass_mean = float(log_m.sum() / n)
    mass_var = max(0.0, float((log_m**2).mean() - mass_mean**2))
    mass_std = math.sqrt(mass_var) + 1e-6

    diag = np.zeros(n, dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        if int(r) == int(c):
            diag[int(r)] = float(v)

    diag_mean = float(diag.sum() / n)
    diag_var = max(0.0, float((diag**2).mean() - diag_mean**2))
    diag_std = math.sqrt(diag_var) + 1e-6

    row_sum = np.zeros(n, dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        row_sum[int(r)] += abs(float(v))
    rmx = float(row_sum.max()) if n else 1.0
    if rmx <= 0.0:
        rmx = 1.0

    stats = np.array(
        [mass_mean, mass_std, diag_mean, diag_std, rmx, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    global_feat = np.zeros(12, dtype=np.float32)
    global_feat[0] = math.log2(max(1, n))
    global_feat[1] = 0.0
    global_feat[2] = mass_mean
    global_feat[3] = mass_std
    global_feat[4] = diag_mean
    global_feat[5] = diag_std
    # [6..11] left 0 — matches Unity globalFeatBuf
    return stats, global_feat


def main() -> None:
    p = argparse.ArgumentParser(description="Print LeafOnly PyTorch parity stats for Unity comparison.")
    p.add_argument(
        "--data_folder",
        type=Path,
        default=None,
        help="Folder containing recorded runs (rglob nodes.bin). Default: ../StreamingAssets/TestData",
    )
    p.add_argument("--frame", type=int, default=0, help="Dataset index (sorted frame paths).")
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="leaf_only_weights.bytes path (default: script_dir/leaf_only_weights.bytes)",
    )
    p.add_argument(
        "--match_unity_shader",
        action="store_true",
        help="Zero diffusion in x and rebuild global_features like Unity FinalizeNodeStats (recommended).",
    )
    p.add_argument(
        "--align",
        choices=("floor", "ceil"),
        default="ceil",
        help="Leaf padding for PyTorch N: ceil matches Unity buffer leaf count when N%%128!=0; "
        "floor matches InspectModel problem_padded_num_nodes (truncate).",
    )
    p.add_argument(
        "--r",
        choices=("ones", "random"),
        default="ones",
        help="Test vector for apply_block_diagonal_M_physical.",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed when --r random.")
    p.add_argument("--no_apply", action="store_true", help="Skip M@r apply summaries.")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (may hide bugs).")
    args = p.parse_args()

    data_folder = args.data_folder
    if data_folder is None:
        data_folder = _script_dir.parent / "StreamingAssets" / "TestData"
    data_folder = Path(data_folder).resolve()
    weights_path = Path(args.weights) if args.weights else _script_dir / "leaf_only_weights.bytes"
    weights_path = weights_path.resolve()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if not weights_path.is_file():
        raise SystemExit(f"Weights not found: {weights_path}")

    ckpt = leaf_only_arch_from_checkpoint(weights_path)
    if ckpt is None:
        raise SystemExit(f"Bad checkpoint header: {weights_path}")
    if int(ckpt["leaf_size"]) != int(LEAF_SIZE):
        raise SystemExit(f"Checkpoint leaf_size {ckpt['leaf_size']} != LEAF_SIZE {LEAF_SIZE}")

    ck_hw = int(ckpt.get("highway_ffn_mlp", 0))
    cli_hw = bool(_leaf_only_script.DEFAULT_USE_HIGHWAYS)
    if ck_hw == 1 and not cli_hw:
        raise SystemExit(_leaf_only_script.CHECKPOINT_ERR_HIGHWAY_IN_FILE_NEED_CLI_ON)
    if ck_hw == 0 and cli_hw:
        raise SystemExit(_leaf_only_script.CHECKPOINT_ERR_NO_HIGHWAY_IN_FILE_NEED_CLI_OFF)
    ck_ffn = int(ckpt.get("ffn_concat_width", 3 if ck_hw else 1))

    dataset = FluidGraphDataset([data_folder])
    if len(dataset) == 0:
        raise SystemExit(f"No frames under {data_folder}")
    fi = max(0, min(int(args.frame), len(dataset) - 1))
    batch = dataset[fi]
    frame_path = batch.get("frame_path", "")
    print(f"[LeafOnlyParity] phase=pytorch frame_index={fi} frame_path={frame_path}")
    print(f"[LeafOnlyParity] phase=pytorch device={device} match_unity_shader={args.match_unity_shader}")

    num_nodes_real = int(batch["num_nodes"])
    if args.align == "floor":
        n_pad = int(problem_padded_num_nodes(num_nodes_real))
    else:
        n_pad = _leaf_align_n(num_nodes_real, "ceil")
    n_pad = min(n_pad, int(MAX_MIXED_SIZE))
    n_inspect_floor = int(problem_padded_num_nodes(num_nodes_real))
    print(
        f"[LeafOnlyParity] phase=pytorch num_nodes_real={num_nodes_real} align={args.align} n_pad={n_pad} "
        f"num_leaves={n_pad // int(LEAF_SIZE)} problem_padded_floor={n_inspect_floor} MAX_MIXED_SIZE={MAX_MIXED_SIZE}"
    )

    x = batch["x"].unsqueeze(0).to(device).clone()
    ei, ev = batch["edge_index"], batch["edge_values"]
    n_edge_active = min(num_nodes_real, n_pad)
    em = (ei[0] < n_edge_active) & (ei[1] < n_edge_active)
    edge_index_leaf = ei[:, em].to(device)
    edge_values_leaf = ev[em].to(device)

    if args.match_unity_shader:
        x[:, :, 5:8] = 0.0
        rows = ei[0].numpy()
        cols = ei[1].numpy()
        vals = ev.numpy()
        em_np = em.numpy()
        rows = rows[em_np]
        cols = cols[em_np]
        vals = vals[em_np]
        mass = np.fromfile(Path(frame_path) / "nodes.bin", dtype=NODE_DTYPE)["mass"][:num_nodes_real]
        stats_n = min(num_nodes_real, n_pad)
        stats_u, gf_u = _unity_stats_and_global(stats_n, mass[:stats_n], rows, cols, vals)
        gf_t = torch.from_numpy(gf_u).to(device=device, dtype=x.dtype)
        _summarize("statsBuffer_like", torch.from_numpy(stats_u))
        _head("statsBuffer_like", torch.from_numpy(stats_u), 8)
        _summarize("globalFeatBuf_like", gf_t)
        _head("globalFeatBuf_like", gf_t, 12)
        global_features = gf_t.unsqueeze(0)
    else:
        global_features = batch["global_features"].to(device)
        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)  # (1, 12)

    n_take = min(num_nodes_real, n_pad)
    x_leaf = x[:, :n_take, :].clone()
    if n_pad > n_take:
        x_leaf = F.pad(x_leaf, (0, 0, 0, n_pad - n_take), value=0.0)

    model = LeafOnlyNet(
        input_dim=int(ckpt["input_dim"]),
        d_model=int(ckpt["d_model"]),
        leaf_size=int(ckpt["leaf_size"]),
        num_layers=int(ckpt["num_layers"]),
        num_heads=int(ckpt["num_heads"]),
        use_gcn=bool(int(ckpt["use_gcn"])),
        attention_layout=default_attention_layout(int(ckpt["leaf_size"])),
        off_diag_dense_attention=True,
        diag_dense_attention=True,
        use_highways=cli_hw,
        ffn_concat_width=int(ck_ffn) if cli_hw else None,
    ).to(device)
    load_leaf_only_weights(model, weights_path)
    model.eval()
    if args.compile:
        model = torch.compile(model)

    with torch.inference_mode():
        packed = model(
            x_leaf,
            edge_index_leaf,
            edge_values_leaf,
            global_features=global_features,
        )

    _summarize("packed_precond", packed)
    _head("packed_precond", packed, 16)

    La_d = int(model.leaf_apply_size)
    La_o = int(model.leaf_apply_off)
    num_leaves = n_pad // int(LEAF_SIZE)
    diag_b, off_b, node_u, node_v, jac = unpack_precond(
        packed, n_pad, leaf_size=LEAF_SIZE, leaf_apply_size=La_d, leaf_apply_off=La_o
    )
    _summarize("precondDiag", diag_b)
    _head("precondDiag", diag_b, 16)
    if off_b is not None and off_b.numel() > 0:
        _summarize("precondOff", off_b)
        _head("precondOff", off_b, 16)
    _summarize("precondU", node_u)
    _head("precondU", node_u, 16)
    _summarize("precondV", node_v)
    _head("precondV", node_v, 16)
    if jac is not None:
        _summarize("precondJac", jac)
        _head("precondJac", jac, 16)

    # Token embedding path (after enc_input_proj in PyTorch = after NormAndProject in Unity shader).
    with torch.inference_mode():
        h0 = model.embed(x_leaf, edge_index_leaf, edge_values_leaf, global_features=global_features)
        h_enc = model.enc_input_proj(h0)
    _summarize("token_after_enc", h_enc)
    _head("token_after_enc", h_enc, 16)

    if args.no_apply:
        return

    A = torch.zeros(n_pad, n_pad, device=device, dtype=x.dtype)
    A[edge_index_leaf[0], edge_index_leaf[1]] = edge_values_leaf
    diag_a = torch.diagonal(A)
    jacobi_inv = torch.zeros(1, n_pad, device=device, dtype=x.dtype)
    jacobi_inv[0, :n_edge_active] = torch.where(
        diag_a[:n_edge_active].abs() > 1e-12,
        1.0 / diag_a[:n_edge_active],
        torch.zeros_like(diag_a[:n_edge_active]),
    )

    if args.r == "ones":
        r = torch.ones(1, n_pad, 1, device=device, dtype=x.dtype)
    else:
        g = torch.Generator(device=device)
        g.manual_seed(int(args.seed))
        r = torch.randn(1, n_pad, 1, device=device, dtype=x.dtype, generator=g)

    _summarize("apply_r", r)
    _head("apply_r", r, 16)

    with torch.inference_mode():
        z = apply_block_diagonal_M_physical(
            packed,
            r,
            leaf_size=LEAF_SIZE,
            leaf_apply_size=La_d,
            leaf_apply_off=La_o,
            jacobi_inv_diag=jacobi_inv,
        )
    _summarize("apply_z", z)
    _head("apply_z", z, 16)


if __name__ == "__main__":
    main()
