import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .architecture import (
    LeafOnlyNet,
    _hm_prolong_scatter_indices,
    apply_block_diagonal_M,
    pool_leaf_attn_mask,
    pool_leaf_edge_feats,
)
from .checkpoint import load_leaf_only_weights
from .config import (
    HUTCHINSON_PROBE_JACOBI_OMEGA,
    HUTCHINSON_PROBE_JACOBI_STEPS,
    LEAF_APPLY_SIZE,
    LEAF_APPLY_SIZE_OFF,
    LEAF_SIZE,
    MAX_MIXED_SIZE,
    MAX_NUM_LEAVES,
)
from .data import FluidGraphDataset, build_leaf_block_connectivity, most_recent_run_folder
from .probe_z import jacobi_smooth_hutchinson_z_inplace


def _sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _time_transformer_attn_and_mlp(
    timing_rows: list,
    label_prefix: str,
    block_idx: int,
    block,
    x_in: torch.Tensor,
    device,
    **attn_kw,
) -> torch.Tensor:
    """
    Times LeafBlockAttention (with its pre-norm) and MLP (with its pre-norm) separately; returns post-block x.
    Matches TransformerBlock residual layout: x_mid = x + attn(norm1(x)), x_out = x_mid + mlp(norm2(x_mid)).
    """
    ms_attn = _timed_ms(lambda: block.attn(block.norm1(x_in), **attn_kw), device)
    timing_rows.append([f"{label_prefix} {block_idx} attention (+ pre-norm)", ms_attn])
    with torch.no_grad():
        x_mid = x_in + block.attn(block.norm1(x_in), **attn_kw)
    ms_mlp = _timed_ms(lambda: block.mlp(block.norm2(x_mid)), device)
    timing_rows.append([f"{label_prefix} {block_idx} MLP (+ pre-norm)", ms_mlp])
    with torch.no_grad():
        x_out = x_mid + block.mlp(block.norm2(x_mid))
    return x_out


def _timed_ms(fn, device, warmup=3, repeat=10):
    warmup = max(0, int(warmup))
    repeat = max(1, int(repeat))
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        _sync_device(device)
        t0 = torch.tensor(0.0)  # dummy op to keep graph warm on some backends
        _ = t0 + 1.0
        start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if device.type == "cuda":
            start.record()
            for _ in range(repeat):
                fn()
            end.record()
            torch.cuda.synchronize()
            total_ms = start.elapsed_time(end)
        else:
            import time

            t_start = time.perf_counter()
            for _ in range(repeat):
                fn()
            _sync_device(device)
            total_ms = (time.perf_counter() - t_start) * 1000.0
    return float(total_ms / repeat)


def _hm_strip_pool_h_off_tokens(
    model: LeafOnlyNet,
    h_proj0: torch.Tensor,
    B_prof: int,
    Mh: int,
    Lf: int,
    Ls_off: int,
    otp: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """H-matrix row/col strip aggregate + optional token mean-pool → input to off-diagonal Transformer stack."""
    C_dim = int(h_proj0.shape[-1])
    h_k_t = h_proj0.view(B_prof, MAX_NUM_LEAVES, Lf, C_dim)
    if getattr(model, "strip_build_mode", "einsum") == "einsum":
        Wr_t = model.hm_pool_w_row.to(device=device, dtype=dtype)
        Wc_t = model.hm_pool_w_col.to(device=device, dtype=dtype)
        row_pt = torch.einsum("mk,bklc->bmlc", Wr_t, h_k_t)
        col_pt = torch.einsum("mk,bklc->bmlc", Wc_t, h_k_t)
    else:
        ridx, cidx, gidx = _hm_prolong_scatter_indices(device)
        hk_trans = h_k_t.transpose(0, 1)
        r_sum = torch.zeros(Mh, B_prof, Lf, C_dim, device=device, dtype=dtype)
        c_sum = torch.zeros(Mh, B_prof, Lf, C_dim, device=device, dtype=dtype)
        r_sum.index_add_(0, gidx, hk_trans[ridx])
        c_sum.index_add_(0, gidx, hk_trans[cidx])
        S_view = model.h_block_S.view(Mh, 1, 1, 1).to(device=device, dtype=dtype)
        row_pt = (r_sum / S_view).transpose(0, 1)
        col_pt = (c_sum / S_view).transpose(0, 1)
    h_ = (row_pt + col_pt).reshape(B_prof * Mh, Lf, C_dim)
    if otp > 1:
        h_ = h_.view(B_prof * Mh, Ls_off, otp, C_dim).mean(dim=2)
    return h_


def _attention_distribution_stats_row(stack: str, layer: int, st: Optional[dict]) -> list:
    """One table row: model / mask-aware uniform / blind baselines + excess vs each baseline."""
    if st is None:
        dash = "-"
        return [stack, layer, dash] + [dash] * 20
    return [
        stack,
        layer,
        st["key_count"],
        st["model_self"],
        st["unif_self"],
        st["blind_self"],
        st["model_nei"],
        st["unif_nei"],
        st["blind_nei"],
        st["model_blk"],
        st["unif_blk"],
        st["blind_blk"],
        st["model_mat"],
        st["unif_mat"],
        st["blind_mat"],
        st["excess_self"],
        st["excess_nei"],
        st["excess_blk"],
        st["excess_mat"],
        st["excess_self_vs_blind"],
        st["excess_nei_vs_blind"],
        st["excess_blk_vs_blind"],
        st["excess_mat_vs_blind"],
    ]


def print_comprehensive_attention_profiler(
    model: LeafOnlyNet,
    device: torch.device,
    h_proj0: torch.Tensor,
    leaf_attn_kw: dict,
    off_attn_kw: Optional[dict],
    *,
    B_prof: int,
    Mh: int,
    Lf: int,
    Ls_off: int,
    otp: int,
    x_dtype: torch.dtype,
    warmup: int = 3,
    repeat: int = 10,
) -> None:
    """
    Inventory, decomposed pre-attn LayerNorm vs attention ms, mean attention mass (self / neighbor / block),
    and optional torch.compile timings for full Leaf and H-off stacks. Expects ``model.eval()`` for correct
    ``last_attn_*`` buffers (caller typically runs this inside the component-timing section).
    """
    strip = getattr(model, "strip_build_mode", "einsum")
    _hoff = "dense L×L (all-ones mask)" if bool(getattr(model, "off_diag_dense_attention", True)) else "reachability mask"
    _edge_gate_note = ""
    if model.blocks:
        _h = int(model.blocks[0].attn.edge_gate[0].out_features)
        _edge_gate_note = f" LeafBlockAttention edge_gate: 2-layer MLP (4→{_h}→heads)."
    print(
        "\n=== Attention profiler (all Transformer stacks) ===\n"
        f"Strip build: {strip!r}; OFF_DIAG_TOKEN_POOL={otp} → off block tokens L={Ls_off}; "
        f"num_h_off={Mh}; Leaf stack layers={len(model.blocks)}; H-off stack layers={len(model.off_diag_blocks)}; "
        f"H-off softmax: {_hoff}.{_edge_gate_note}"
    )

    inv_headers = [
        "stack",
        "layer",
        "layout",
        "L_blk",
        "heads",
        "hdim",
        "blk_node",
        "mat_node",
        "n_param",
    ]
    inv_rows: list = []
    for i, block in enumerate(model.blocks):
        a = block.attn
        inv_rows.append(
            [
                "Leaf diag",
                i,
                a.attention_layout,
                a.block_size,
                a.num_heads,
                a.head_dim,
                int(a.use_block_node),
                int(a.use_matrix_node),
                sum(p.numel() for p in a.parameters()),
            ]
        )
    for i, block in enumerate(model.off_diag_blocks):
        a = block.attn
        inv_rows.append(
            [
                "H-off",
                i,
                a.attention_layout,
                a.block_size,
                a.num_heads,
                a.head_dim,
                int(a.use_block_node),
                int(a.use_matrix_node),
                sum(p.numel() for p in a.parameters()),
            ]
        )
    _print_table("Attention modules (per-layer inventory)", inv_headers, inv_rows)
    print(
        "blk_node/mat_node: LeafBlockAttention use_block_node / use_matrix_node from layout "
        "(Layout codes L×(L+1): softmax is L×L; block/matrix use gated V paths, not softmax keys.)"
    )

    dec_headers = ["stack", "layer", "pre_attn_LN_ms", "attention_ms", "LN+attn_ms"]
    dec_rows: list = []
    h = h_proj0
    for i, block in enumerate(model.blocks):
        ms_ln = _timed_ms(lambda b=block, x=h: b.norm1(x), device, warmup=warmup, repeat=repeat)
        with torch.no_grad():
            xn = block.norm1(h)
        ms_at = _timed_ms(
            lambda b=block, u=xn, kw=leaf_attn_kw: b.attn(u, **kw),
            device,
            warmup=warmup,
            repeat=repeat,
        )
        dec_rows.append(["Leaf diag", i, ms_ln, ms_at, ms_ln + ms_at])
        with torch.no_grad():
            x_mid = h + block.attn(xn, **leaf_attn_kw)
            h = x_mid + block.mlp(block.norm2(x_mid))

    if Mh > 0 and off_attn_kw is not None:
        h_off = _hm_strip_pool_h_off_tokens(model, h_proj0, B_prof, Mh, Lf, Ls_off, otp, device, x_dtype)
        for i, block in enumerate(model.off_diag_blocks):
            ms_ln = _timed_ms(lambda b=block, x=h_off: b.norm1(x), device, warmup=warmup, repeat=repeat)
            with torch.no_grad():
                xn = block.norm1(h_off)
            ms_at = _timed_ms(
                lambda b=block, u=xn, kw=off_attn_kw: b.attn(u, **kw),
                device,
                warmup=warmup,
                repeat=repeat,
            )
            dec_rows.append(["H-off", i, ms_ln, ms_at, ms_ln + ms_at])
            with torch.no_grad():
                x_mid = h_off + block.attn(xn, **off_attn_kw)
                h_off = x_mid + block.mlp(block.norm2(x_mid))

    _print_table("Attention micro-timing (eager, decomposed)", dec_headers, dec_rows)
    leaf_sum = sum(r[4] for r in dec_rows if r[0] == "Leaf diag")
    off_sum = sum(r[4] for r in dec_rows if r[0] == "H-off")
    print(
        f"Eager LN+attention subtotal: Leaf diag Σ={leaf_sum:.3f} ms; "
        f"H-off Σ={off_sum:.3f} ms (same warmup/repeat as component table)."
    )

    mass_headers = [
        "stack",
        "layer",
        "K",
        "self_m",
        "self_u",
        "self_b",
        "nei_m",
        "nei_u",
        "nei_b",
        "blk_m",
        "blk_u",
        "blk_b",
        "mat_m",
        "mat_u",
        "mat_b",
        "Δself_u",
        "Δnei_u",
        "Δblk_u",
        "Δmat_u",
        "Δself_b",
        "Δnei_b",
        "Δblk_b",
        "Δmat_b",
    ]
    mass_rows: list = []
    with torch.inference_mode():
        h = h_proj0
        for i, block in enumerate(model.blocks):
            xn = block.norm1(h)
            ya = block.attn(xn, **leaf_attn_kw)
            mass_rows.append(
                _attention_distribution_stats_row("Leaf diag", i, block.attn.last_attn_distribution_stats)
            )
            x_mid = h + ya
            h = x_mid + block.mlp(block.norm2(x_mid))
        if Mh > 0 and off_attn_kw is not None:
            h_off = _hm_strip_pool_h_off_tokens(model, h_proj0, B_prof, Mh, Lf, Ls_off, otp, device, x_dtype)
            for i, block in enumerate(model.off_diag_blocks):
                xn = block.norm1(h_off)
                ya = block.attn(xn, **off_attn_kw)
                mass_rows.append(
                    _attention_distribution_stats_row("H-off", i, block.attn.last_attn_distribution_stats)
                )
                x_mid = h_off + ya
                h_off = x_mid + block.mlp(block.norm2(x_mid))
    _print_table(
        "Attention mass vs baselines (eval; mean combined weight over heads, same mask as softmax)",
        mass_headers,
        mass_rows,
    )
    print(
        "Columns: K = spatial softmax width (L). "
        "self_u / nei_u = mask-aware uniform over allowed leaf keys; self_b / nei_b = blind 1/L priors on leaf keys. "
        "blk_*/mat_*: for block/matrix, model = mean σ(gate); unif = 0.5 (sigmoid-neutral); blind = 0 (no additive global). "
        "Δ*_u / Δ*_b = model minus those references."
    )

    compile_w, compile_r = 4, 4

    def _leaf_stack_forward(h0: torch.Tensor) -> torch.Tensor:
        h = h0
        for block in model.blocks:
            xn = block.norm1(h)
            h = h + block.attn(xn, **leaf_attn_kw)
            h = h + block.mlp(block.norm2(h))
        return h

    try:
        comp_leaf = torch.compile(_leaf_stack_forward, fullgraph=False)
        ms_leaf_c = _timed_ms(lambda: comp_leaf(h_proj0), device, warmup=compile_w, repeat=compile_r)
        print(f"\nLeaf diag stack: torch.compile(full blocks, {compile_w} warmup / {compile_r} repeat) → {ms_leaf_c:.3f} ms/call")
    except Exception as ex:
        print(f"\nLeaf diag stack torch.compile timing skipped: {ex}")

    if Mh > 0 and off_attn_kw is not None:
        h0_off = _hm_strip_pool_h_off_tokens(model, h_proj0, B_prof, Mh, Lf, Ls_off, otp, device, x_dtype)

        def _hoff_stack_forward(h0: torch.Tensor) -> torch.Tensor:
            h = h0
            for block in model.off_diag_blocks:
                xn = block.norm1(h)
                h = h + block.attn(xn, **off_attn_kw)
                h = h + block.mlp(block.norm2(h))
            return h

        try:
            comp_off = torch.compile(_hoff_stack_forward, fullgraph=False)
            ms_off_c = _timed_ms(lambda: comp_off(h0_off), device, warmup=compile_w, repeat=compile_r)
            print(f"H-off stack: torch.compile(full blocks, {compile_w} warmup / {compile_r} repeat) → {ms_off_c:.3f} ms/call")
        except Exception as ex:
            print(f"H-off stack torch.compile timing skipped: {ex}")


def _fmt_cell(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _print_table(title, headers, rows):
    str_rows = [[_fmt_cell(v) for v in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(f"\n{title}")
    print(sep)
    print("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    print(sep)
    for row in str_rows:
        print("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    print(sep)


def leafonly_grad_param_groups(model: LeafOnlyNet, num_blocks: Optional[int] = None):
    """
    One entry per LeafOnlyNet submodule with trainable weights, for gradient interference / Hutchinson variance.
    Order follows the forward data path (lift → GCN → norm → enc proj → …).
    """
    if num_blocks is None:
        num_blocks = len(model.blocks)
    num_gcn = len(model.embed.gcn) if model.embed.gcn else 0
    g = {}
    g["Lift (linear 0)"] = lambda m: list(m.embed.lift[0].parameters())
    g["Lift (linear 2)"] = lambda m: list(m.embed.lift[2].parameters())
    for gi in range(num_gcn):
        g[f"GCN layer {gi}"] = lambda m, gi=gi: list(m.embed.gcn[gi].parameters())
    g["Embedding LayerNorm"] = lambda m: list(m.embed.norm.parameters())
    g["Encoder input projection"] = lambda m: list(m.enc_input_proj.parameters())
    for b in range(num_blocks):
        g[f"Transformer block {b}"] = lambda m, b=b: list(m.blocks[b].parameters())
    for b in range(num_blocks):
        g[f"Off-diag Transformer block {b}"] = lambda m, b=b: list(m.off_diag_blocks[b].parameters())
    g["Diagonal Leaf Head"] = lambda m: list(m.leaf_head.parameters())
    g["Off-diag U/V heads"] = lambda m: list(m.off_diag_head_U.parameters()) + list(m.off_diag_head_V.parameters())
    g["Global node U linear"] = lambda m: list(m.node_u.parameters())
    g["Global node V linear"] = lambda m: list(m.node_v.parameters())
    g["Jacobi params"] = lambda m: list(m.jacobi_gate.parameters())
    return g


def _float_tensor_distribution_stats(t: torch.Tensor):
    """
    Mean, dispersion, quantiles, norms, and simple tail mass on CPU float32.
    Returns None if empty.
    """
    x = t.detach().float().cpu().reshape(-1)
    n = int(x.numel())
    if n == 0:
        return None
    mean = float(x.mean())
    std = float(x.std(unbiased=False))
    min_v = float(x.min())
    max_v = float(x.max())
    if n == 1:
        p01 = p50 = p99 = mean
    else:
        q = torch.quantile(x, torch.tensor([0.01, 0.5, 0.99], dtype=torch.float32))
        p01, p50, p99 = float(q[0].item()), float(q[1].item()), float(q[2].item())
    l2 = float(torch.linalg.norm(x))
    frac_abs_gt_1 = float((x.abs() > 1.0).float().mean())
    frac_near_zero = float((x.abs() < 1e-7).float().mean())
    return mean, std, min_v, max_v, p01, p50, p99, l2, frac_abs_gt_1, frac_near_zero


def print_leafonly_parameter_and_buffer_report(model: LeafOnlyNet) -> None:
    """
    Report means, spreads, quantiles, and tail fractions for every trainable parameter
    and every registered buffer (things you might initialize or precompute).
    """
    print(
        "\n=== Tensor statistics (after checkpoint load) ===\n"
        "Trainable parameters: compare std to your nn.init (e.g. normal std=0.001); "
        "p01/p50/p99 describe mass; frac|x|>1 and near-zero highlight tails/sparsity.\n"
        "Buffers: static H-matrix / pooling weights (not updated by optimizer)."
    )

    p_headers = [
        "name",
        "shape",
        "n",
        "mean",
        "std",
        "min",
        "max",
        "p01",
        "p50",
        "p99",
        "L2",
        "frac|x|>1",
        "frac|x|<1e-7",
    ]
    param_rows = []
    total_trainable = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        st = _float_tensor_distribution_stats(p)
        if st is None:
            continue
        mean, std, min_v, max_v, p01, p50, p99, l2, f1, fz = st
        total_trainable += int(p.numel())
        param_rows.append(
            [
                name,
                tuple(p.shape),
                int(p.numel()),
                mean,
                std,
                min_v,
                max_v,
                p01,
                p50,
                p99,
                l2,
                f1,
                fz,
            ]
        )
    _print_table("Trainable parameters", p_headers, param_rows)
    print(f"  Total trainable scalars: {total_trainable}")

    fbuf_headers = p_headers
    fbuf_rows = []
    for name, b in model.named_buffers():
        if not b.dtype.is_floating_point:
            continue
        st = _float_tensor_distribution_stats(b)
        if st is None:
            continue
        mean, std, min_v, max_v, p01, p50, p99, l2, f1, fz = st
        fbuf_rows.append(
            [
                name,
                tuple(b.shape),
                int(b.numel()),
                mean,
                std,
                min_v,
                max_v,
                p01,
                p50,
                p99,
                l2,
                f1,
                fz,
            ]
        )
    if fbuf_rows:
        _print_table("Floating-point buffers (fixed / precomputed)", fbuf_headers, fbuf_rows)

    ibuf_headers = ["name", "shape", "n", "dtype", "min", "max", "mean(as float)"]
    ibuf_rows = []
    for name, b in model.named_buffers():
        if b.dtype.is_floating_point:
            continue
        x = b.detach().cpu().reshape(-1)
        n = int(x.numel())
        if n == 0:
            continue
        xf = x.float()
        ibuf_rows.append(
            [
                name,
                tuple(b.shape),
                n,
                str(b.dtype).replace("torch.", ""),
                float(xf.min().item()),
                float(xf.max().item()),
                float(xf.mean().item()),
            ]
        )
    if ibuf_rows:
        _print_table("Integer / index buffers", ibuf_headers, ibuf_rows)

    nh = model.blocks[0].attn.num_heads if model.blocks else 0
    print(
        f"\nHyperparameters tied to default init / layout: "
        f"d_model={model.embed.lift[0].out_features}, "
        f"leaf_size={model.leaf_size}, "
        f"leaf_apply_size(diag)={model.leaf_apply_size}, "
        f"leaf_apply_off={model.leaf_apply_off}, "
        f"num_leaf_blocks={len(model.blocks)}, "
        f"num_off_blocks={len(model.off_diag_blocks)}, "
        f"num_heads={nh}, "
        f"num_h_off={model.num_h_off}, "
        f"use_jacobi={model.use_jacobi}."
    )


def evaluate_gradient_interference(args, runtime):
    data_folder = runtime["data_folder"]
    save_path = runtime["save_path"]
    runtime_use_gcn = runtime["use_gcn"]
    device = runtime["device"]
    args.num_layers = 2
    args.num_gcn_layers = 2
    args.use_jacobi = True
    use_jacobi = True
    attention_layout = str(args.attention_layout)
    if not runtime_use_gcn:
        raise ValueError("Runtime has use_gcn=False, but the fixed architecture requires 2 GCN layers.")
    requested_gcn_layers = 2
    effective_gcn_layers = requested_gcn_layers
    use_gcn = effective_gcn_layers > 0

    print(f"\n--- Starting Gradient Interference Analysis on {device} ---")
    print(
        f"Hutchinson probes: Z is low-passed with {HUTCHINSON_PROBE_JACOBI_STEPS} damped-Jacobi sweeps "
        f"(ω={HUTCHINSON_PROBE_JACOBI_OMEGA}) before the loss, matching training."
    )

    model = LeafOnlyNet(
        input_dim=9,
        d_model=args.d_model,
        leaf_size=LEAF_SIZE,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        attention_layout=attention_layout,
        use_gcn=use_gcn,
        num_gcn_layers=effective_gcn_layers,
        use_jacobi=use_jacobi,
        strip_build_mode=getattr(args, "strip_build_mode", "einsum"),
        off_diag_dense_attention=bool(getattr(args, "off_diag_dense_attn", True)),
    ).to(device)

    if save_path.exists():
        load_leaf_only_weights(model, str(save_path))
        print(f"Loaded weights from {save_path}")
    else:
        print("WARNING: No saved weights found. Analyzing randomly initialized gradients.")

    model.train()
    if getattr(args, "peek_parameters", False):
        print_leafonly_parameter_and_buffer_report(model)

    num_eval = 10
    contexts_per_step = max(1, int(args.contexts_per_step))
    eval_max_nodes = max(LEAF_SIZE * 2, int(MAX_MIXED_SIZE))
    data_path = Path(data_folder)
    run_folder = most_recent_run_folder(data_path)
    dataset = FluidGraphDataset([run_folder])
    rng = random.Random(args.seed)

    if args.use_single_frame:
        frame_idx = min(args.frame, len(dataset) - 1)
        frame_indices_per_pass = [[frame_idx] * contexts_per_step for _ in range(num_eval)]
        print(f"Evaluating {num_eval} passes, {contexts_per_step} contexts per pass (single frame {frame_idx})")
    else:
        frame_indices_per_pass = [rng.choices(range(len(dataset)), k=contexts_per_step) for _ in range(num_eval)]
        print(f"Evaluating {num_eval} passes, {contexts_per_step} frames per pass (gradient accumulation like training)")

    param_groups = leafonly_grad_param_groups(model, num_blocks=args.num_layers)

    group_gradients = {name: [] for name in param_groups.keys()}

    for step in range(num_eval):
        model.zero_grad()
        step_loss_sum = 0.0
        for micro in range(contexts_per_step):
            frame_idx = frame_indices_per_pass[step][micro]
            batch = dataset[frame_idx]
            n_pad = MAX_MIXED_SIZE
            n_orig = min(int(batch["num_nodes"]), eval_max_nodes, n_pad)
            x_input = batch["x"][:n_orig].unsqueeze(0).to(device)
            x_input = F.pad(x_input, (0, 0, 0, n_pad - n_orig), value=0.0)
            active_pos = x_input[0, :n_orig, :3]
            x_input[0, :n_orig, :3] = active_pos - active_pos.mean(dim=0, keepdim=True)
            rows, cols = batch["edge_index"][0], batch["edge_index"][1]
            mask = (rows < n_orig) & (cols < n_orig)
            edge_index = batch["edge_index"][:, mask].to(device)
            edge_values = batch["edge_values"][mask].to(device)
            A_sparse = torch.sparse_coo_tensor(edge_index, edge_values, (n_orig, n_orig)).coalesce()
            A_small = A_sparse.to_dense().to(device) if device.type == "mps" else A_sparse.to(device).to_dense()
            A_dense = torch.zeros(n_pad, n_pad, device=device, dtype=A_small.dtype)
            A_dense[:n_orig, :n_orig] = A_small
            A_dense[n_orig:, n_orig:] = torch.eye(n_pad - n_orig, device=device, dtype=A_small.dtype)
            dm, df, om, oe = build_leaf_block_connectivity(
                edge_index,
                edge_values,
                x_input[0, :n_pad, :3],
                LEAF_SIZE,
                device,
                x_input.dtype,
                off_diag_dense_attention=bool(model.off_diag_dense_attention),
            )
            pre_leaf = (dm, df, om, oe)
            global_feat = batch.get("global_features")
            if global_feat is None:
                raise ValueError(f"Missing global_features for frame: {batch.get('frame_path', '<unknown>')}")
            global_feat = global_feat.to(device)
            if global_feat.dim() == 1:
                global_feat = global_feat.unsqueeze(0)
            precond_out = model(
                x_input,
                edge_index=edge_index,
                edge_values=edge_values,
                precomputed_leaf_connectivity=pre_leaf,
                global_features=global_feat,
            )
            batch_vectors = max(1024, int(round(n_pad ** 0.5)))
            jacobi_inv_diag = torch.ones(1, n_pad, device=device, dtype=A_dense.dtype)
            diag_A = torch.diagonal(A_dense, 0)
            inv_mask = diag_A.abs() > 1e-6
            jacobi_inv_diag[0, inv_mask] = 1.0 / diag_A[inv_mask]
            Z = torch.randn(1, n_pad, batch_vectors, device=device, dtype=x_input.dtype)
            Z[:, n_orig:, :] = 0.0
            jacobi_smooth_hutchinson_z_inplace(
                Z,
                jacobi_inv_diag,
                [n_orig],
                n_pad,
                lambda Zt: (A_dense @ Zt.squeeze(0)).unsqueeze(0),
            )
            AZ = (A_dense @ Z.squeeze(0)).unsqueeze(0)
            MAZ = apply_block_diagonal_M(
                precond_out,
                AZ,
                leaf_size=LEAF_SIZE,
                leaf_apply_size=LEAF_APPLY_SIZE,
                leaf_apply_off=LEAF_APPLY_SIZE_OFF,
                jacobi_inv_diag=jacobi_inv_diag,
            )
            B_ctx = MAZ.size(0)
            MAZ_flat = MAZ.view(B_ctx, -1)
            Z_flat = Z.view(B_ctx, -1)
            cos_sim = F.cosine_similarity(MAZ_flat, Z_flat, dim=1)
            raw_loss = (1.0 - cos_sim).mean()
            step_loss_sum += raw_loss.item()
            loss = raw_loss / contexts_per_step
            loss.backward()

        step_loss_avg = step_loss_sum / contexts_per_step
        for group_name, get_params in param_groups.items():
            params = get_params(model)
            if not params:
                continue
            grads = []
            for p in params:
                if p.grad is not None:
                    grads.append(p.grad.detach().clone().view(-1))
                else:
                    grads.append(torch.zeros_like(p).view(-1))
            flat_grad = torch.cat(grads)
            group_gradients[group_name].append(flat_grad)

        print(f"  Processed pass {step + 1}/{num_eval} ({contexts_per_step} frames, avg loss: {step_loss_avg:.4f})")

    print("\n=== Gradient Interference Report ===")
    gradient_rows = []
    for group_name, grads in group_gradients.items():
        if not grads:
            continue
        grads_stack = torch.stack(grads)
        num_passes = grads_stack.shape[0]
        mean_grad = grads_stack.mean(dim=0)
        var_grad = grads_stack.var(dim=0, unbiased=False)
        signal_norm = mean_grad.norm().item()
        noise_norm = torch.sqrt(var_grad.mean() + 1e-12).item()
        snr = signal_norm / noise_norm if noise_norm > 0 else float("inf")
        cos_sims = []
        for i in range(num_passes):
            for j in range(i + 1, num_passes):
                sim = F.cosine_similarity(grads_stack[i].unsqueeze(0), grads_stack[j].unsqueeze(0), dim=1).item()
                cos_sims.append(sim)
        avg_cos_sim = sum(cos_sims) / len(cos_sims) if cos_sims else 1.0
        gradient_rows.append(
            [
                group_name,
                f"{avg_cos_sim:.4f}",
                f"{snr:.4f}",
                f"{signal_norm:.4e}",
            ]
        )
    _print_table(
        "Gradient Interference",
        ["Group", "Avg Cosine", "SNR", "Grad Magnitude"],
        gradient_rows,
    )
    print("Legend: cosine (1=aligned, 0=orthogonal, -1=opposing).")

    # Component-level timing on one representative context.
    profile_frame_idx = frame_indices_per_pass[0][0]
    batch = dataset[profile_frame_idx]
    n_pad = MAX_MIXED_SIZE
    n_orig = min(int(batch["num_nodes"]), eval_max_nodes, n_pad)
    x_input = batch["x"][:n_orig].unsqueeze(0).to(device)
    x_input = F.pad(x_input, (0, 0, 0, n_pad - n_orig), value=0.0)
    active_pos = x_input[0, :n_orig, :3]
    x_input[0, :n_orig, :3] = active_pos - active_pos.mean(dim=0, keepdim=True)
    rows, cols = batch["edge_index"][0], batch["edge_index"][1]
    mask = (rows < n_orig) & (cols < n_orig)
    edge_index = batch["edge_index"][:, mask].to(device)
    edge_values = batch["edge_values"][mask].to(device)
    dm, df, om, oe = build_leaf_block_connectivity(
        edge_index,
        edge_values,
        x_input[0, :n_pad, :3],
        LEAF_SIZE,
        device,
        x_input.dtype,
        off_diag_dense_attention=bool(model.off_diag_dense_attention),
    )
    pre_leaf = (dm, df, om, oe)
    global_feat = batch.get("global_features")
    if global_feat is None:
        raise ValueError(f"Missing global_features for frame: {batch.get('frame_path', '<unknown>')}")
    global_feat = global_feat.to(device)
    if global_feat.dim() == 1:
        global_feat = global_feat.unsqueeze(0)

    A_sparse = torch.sparse_coo_tensor(edge_index, edge_values, (n_orig, n_orig)).coalesce()
    A_small = A_sparse.to_dense().to(device) if device.type == "mps" else A_sparse.to(device).to_dense()
    A_dense = torch.zeros(n_pad, n_pad, device=device, dtype=A_small.dtype)
    A_dense[:n_orig, :n_orig] = A_small
    A_dense[n_orig:, n_orig:] = torch.eye(n_pad - n_orig, device=device, dtype=A_small.dtype)
    jacobi_inv_diag_profile = torch.ones(1, n_pad, device=device, dtype=A_dense.dtype)
    diag_A_profile = torch.diagonal(A_dense, 0)
    inv_mask_profile = diag_A_profile.abs() > 1e-6
    jacobi_inv_diag_profile[0, inv_mask_profile] = 1.0 / diag_A_profile[inv_mask_profile]
    batch_vectors = max(1024, int(round(n_pad ** 0.5)))

    attn_mask, edge_feats, om_prof, oe_prof = pre_leaf
    positions = x_input[0, :, :3]
    B_prof, N_prof = x_input.shape[0], x_input.shape[1]
    assert N_prof == MAX_NUM_LEAVES * LEAF_SIZE, "Profile expects N == MAX_MIXED_SIZE"
    Lf = LEAF_SIZE
    Mh = model.num_h_off
    Ls_off = int(model.leaf_apply_off)
    otp = int(model.off_token_pool)
    if Mh > 0:
        om = om_prof.to(device=device, dtype=x_input.dtype)
        oe = oe_prof.to(device=device, dtype=x_input.dtype)
        if otp > 1:
            om = pool_leaf_attn_mask(om, Lf, otp)
            oe = pool_leaf_edge_feats(oe, Lf, otp)
        off_attn_mask = om.unsqueeze(0).expand(B_prof, Mh, Ls_off, Ls_off + 1).contiguous().reshape(
            B_prof * Mh, 1, Ls_off, Ls_off + 1
        )
        if model.off_diag_dense_attention:
            off_attn_mask = torch.ones_like(off_attn_mask)
        off_edge_feats = (
            oe.unsqueeze(0)
            .expand(B_prof, Mh, Ls_off, Ls_off + 1, 4)
            .contiguous()
            .reshape(B_prof * Mh, 1, Ls_off, Ls_off + 1, 4)
        )
    else:
        off_attn_mask = off_edge_feats = None

    with torch.no_grad():
        h_lift = model.embed.lift(torch.cat([x_input[..., 3:], global_feat.unsqueeze(1).expand(-1, x_input.size(1), -1)], dim=-1))
        h_gcn = h_lift
        if model.embed.gcn is not None:
            for gcn_layer in model.embed.gcn:
                h_gcn = gcn_layer(h_gcn, edge_index, edge_values)
        h_norm = model.embed.norm(h_gcn)
        h_proj0 = model.enc_input_proj(h_norm)
        C_dim = h_proj0.shape[-1]
        h_diag = h_proj0
        for block in model.blocks:
            h_diag = block(
                h_diag,
                edge_index=edge_index,
                edge_values=edge_values,
                positions=positions,
                save_attention=False,
                attn_mask=attn_mask,
                edge_feats=edge_feats,
            )
        diag_blocks_profile = model._get_leaf_blocks(h_diag, mode="diagonal")
        if Mh > 0:
            h_off = _hm_strip_pool_h_off_tokens(model, h_proj0, B_prof, Mh, Lf, Ls_off, otp, device, x_input.dtype)
            for block in model.off_diag_blocks:
                h_off = block(
                    h_off,
                    edge_index=edge_index,
                    edge_values=edge_values,
                    positions=positions,
                    save_attention=False,
                    attn_mask=off_attn_mask,
                    edge_feats=off_edge_feats,
                )
            h_off = h_off.view(B_prof, Mh, Ls_off, C_dim)
            off_diag_blocks_profile = model._get_leaf_blocks(h_off, mode="off-diagonal")
        else:
            off_diag_blocks_profile = torch.empty(
                (B_prof, 0, LEAF_APPLY_SIZE_OFF, LEAF_APPLY_SIZE_OFF), device=device, dtype=h_proj0.dtype
            )
        node_U_profile = model.node_u(h_diag)
        node_V_profile = model.node_v(h_diag)
        jacobi_scale_profile = model._get_jacobi_scale(h_diag)

    was_training = model.training
    model.eval()
    try:
        compiled_model = torch.compile(model)
        with torch.inference_mode():
            ms_end_to_end = _timed_ms(
                lambda: compiled_model(
                    x_input,
                    edge_index=edge_index,
                    edge_values=edge_values,
                    global_features=global_feat,
                    precomputed_leaf_connectivity=pre_leaf,
                ),
                device,
                warmup=15,
                repeat=10,
            )
    except Exception as e:
        ms_end_to_end = None
        print(f"\nEnd-to-end torch.compile timing skipped: {e}")
    if was_training:
        model.train()

    if ms_end_to_end is not None:
        print(
            f"\nEnd-to-end forward (torch.compile, inference_mode, precomputed masks, n_pad={n_pad}): "
            f"{ms_end_to_end:.2f} ms — aligned with InspectModel 'Inference' (timed before BSR warmup)"
        )

    _timing_was_training = model.training
    model.eval()
    try:
        timing_rows = []

        lifted_in = torch.cat(
            [x_input[..., 3:], global_feat.unsqueeze(1).expand(-1, x_input.size(1), -1)], dim=-1
        )
        ms_lift0 = _timed_ms(lambda: model.embed.lift[0](lifted_in), device)
        timing_rows.append(["Lift linear 0", ms_lift0])
        with torch.no_grad():
            _lift_mid = F.gelu(model.embed.lift[0](lifted_in))
        ms_lift2 = _timed_ms(lambda: model.embed.lift[2](_lift_mid), device)
        timing_rows.append(["Lift linear 2 (+ preceding GELU)", ms_lift2])

        if model.embed.gcn is not None and len(model.embed.gcn) > 0:
            h_for_gcn = model.embed.lift(lifted_in)
            for i, gcn_layer in enumerate(model.embed.gcn):
                ms_gcn_i = _timed_ms(lambda g=gcn_layer, h=h_for_gcn: g(h, edge_index, edge_values), device)
                timing_rows.append([f"GCN layer {i}", ms_gcn_i])
                with torch.no_grad():
                    h_for_gcn = gcn_layer(h_for_gcn, edge_index, edge_values)

        ms_norm = _timed_ms(lambda: model.embed.norm(h_gcn), device)
        timing_rows.append(["Embedding LayerNorm", ms_norm])

        ms_enc_proj = _timed_ms(lambda: model.enc_input_proj(h_norm), device)
        timing_rows.append(["Encoder input projection", ms_enc_proj])

        leaf_attn_kw = dict(
            edge_index=edge_index,
            edge_values=edge_values,
            positions=positions,
            save_attention=False,
            attn_mask=attn_mask,
            edge_feats=edge_feats,
        )
        h_block_in = h_proj0
        for i, block in enumerate(model.blocks):
            h_block_in = _time_transformer_attn_and_mlp(
                timing_rows, "Leaf", i, block, h_block_in, device, **leaf_attn_kw
            )

        ms_leaf_head = _timed_ms(lambda: model._get_leaf_blocks(h_block_in, mode="diagonal"), device)
        timing_rows.append(["Diagonal leaf head (U U^T)", ms_leaf_head])

        off_attn_kw = dict(
            edge_index=edge_index,
            edge_values=edge_values,
            positions=positions,
            save_attention=False,
            attn_mask=off_attn_mask,
            edge_feats=off_edge_feats,
        )
        print_comprehensive_attention_profiler(
            model,
            device,
            h_proj0,
            leaf_attn_kw,
            off_attn_kw if Mh > 0 else None,
            B_prof=B_prof,
            Mh=Mh,
            Lf=Lf,
            Ls_off=Ls_off,
            otp=otp,
            x_dtype=x_input.dtype,
        )
        if Mh > 0:
            _strip_label = (
                "H off strip pool (einsum row+col + token mean)"
                if getattr(model, "strip_build_mode", "einsum") == "einsum"
                else "H off strip pool (index_add row+col + token mean)"
            )
            ms_strip = _timed_ms(
                lambda: _hm_strip_pool_h_off_tokens(
                    model, h_proj0, B_prof, Mh, Lf, Ls_off, otp, device, x_input.dtype
                ),
                device,
            )
            timing_rows.append([_strip_label, ms_strip])
            h_off_in = _hm_strip_pool_h_off_tokens(
                model, h_proj0, B_prof, Mh, Lf, Ls_off, otp, device, x_input.dtype
            )
            for i, block in enumerate(model.off_diag_blocks):
                h_off_in = _time_transformer_attn_and_mlp(
                    timing_rows, "H off", i, block, h_off_in, device, **off_attn_kw
                )
            h_off_in = h_off_in.view(B_prof, Mh, Ls_off, C_dim)
            ms_off_head = _timed_ms(lambda: model._get_leaf_blocks(h_off_in, mode="off-diagonal"), device)
            timing_rows.append(["H off head (U V^T)", ms_off_head])

        ms_node_u = _timed_ms(lambda: model.node_u(h_block_in), device)
        timing_rows.append(["Global node U linear", ms_node_u])
        ms_node_v = _timed_ms(lambda: model.node_v(h_block_in), device)
        timing_rows.append(["Global node V linear", ms_node_v])

        ms_jacobi = _timed_ms(lambda: model._get_jacobi_scale(h_block_in), device)
        timing_rows.append(["Jacobi gate (node_scalar)", ms_jacobi])

        def _bench_probe_z():
            z = torch.randn(1, n_pad, batch_vectors, device=device, dtype=x_input.dtype)
            z[:, n_orig:, :] = 0.0
            jacobi_smooth_hutchinson_z_inplace(
                z,
                jacobi_inv_diag_profile,
                [n_orig],
                n_pad,
                lambda Zt: (A_dense @ Zt.squeeze(0)).unsqueeze(0),
            )

        ms_probe_z = _timed_ms(_bench_probe_z, device)
        timing_rows.append(
            [
                f"Probe Z (+ Jacobi {HUTCHINSON_PROBE_JACOBI_STEPS}×ω={HUTCHINSON_PROBE_JACOBI_OMEGA})",
                ms_probe_z,
            ]
        )

        Z = torch.randn(1, n_pad, batch_vectors, device=device, dtype=x_input.dtype)
        Z[:, n_orig:, :] = 0.0
        jacobi_smooth_hutchinson_z_inplace(
            Z,
            jacobi_inv_diag_profile,
            [n_orig],
            n_pad,
            lambda Zt: (A_dense @ Zt.squeeze(0)).unsqueeze(0),
        )
        ms_AZ = _timed_ms(lambda: (A_dense @ Z.squeeze(0)).unsqueeze(0), device)
        timing_rows.append(["A @ Z (post-smooth)", ms_AZ])
        B_pack = diag_blocks_profile.shape[0]
        packed_profile = torch.cat(
            [
                diag_blocks_profile.reshape(B_pack, -1),
                off_diag_blocks_profile.reshape(B_pack, -1),
                node_U_profile.reshape(B_pack, -1),
                node_V_profile.reshape(B_pack, -1),
            ],
            dim=1,
        )
        if jacobi_scale_profile is not None:
            packed_profile = torch.cat([packed_profile, jacobi_scale_profile], dim=1)

        def _pack_precond_profile():
            parts = [
                diag_blocks_profile.reshape(B_pack, -1),
                off_diag_blocks_profile.reshape(B_pack, -1),
                node_U_profile.reshape(B_pack, -1),
                node_V_profile.reshape(B_pack, -1),
            ]
            if jacobi_scale_profile is not None:
                parts.append(jacobi_scale_profile)
            return torch.cat(parts, dim=1)

        ms_pack = _timed_ms(_pack_precond_profile, device)
        timing_rows.append(["Pack preconditioner (torch.cat)", ms_pack])
        ms_apply_m = _timed_ms(
            lambda: apply_block_diagonal_M(
                packed_profile,
                AZ,
                leaf_size=LEAF_SIZE,
                leaf_apply_size=LEAF_APPLY_SIZE,
                leaf_apply_off=LEAF_APPLY_SIZE_OFF,
                jacobi_inv_diag=jacobi_inv_diag_profile,
            ),
            device,
        )
        timing_rows.append(["Apply block-diagonal M", ms_apply_m])
    finally:
        if _timing_was_training:
            model.train()

    total_ms = sum(v for _, v in timing_rows) + 1e-12
    timed_rows = [[name, f"{ms:.3f}", f"{(100.0 * ms / total_ms):.1f}%"] for name, ms in timing_rows]
    _print_table(
        "Component Timing (--evaluate_gradients)",
        ["Component", "ms/call", "% of measured total"],
        timed_rows,
    )
    print(
        "\nTiming coverage: micro-benchmarks use eval() and omit build_leaf_block_connectivity only "
        "(masks are precomputed). Rows cover lift linears, GCN, embed norm, enc_input_proj, per-layer leaf and "
        "H-off attention vs MLP, strip pool, both low-rank heads, node U/V, Jacobi gate, pack cat, "
        "probe Z (randn + damped Jacobi, same as training), A@Z on smoothed Z, and M apply. "
        "Earlier in this run, Attention profiler: per-layer inventory, decomposed pre-attn LN vs attention ms, "
        "mean self/neighbor/block mass, and torch.compile full-stack lines for Leaf diag and H-off. "
        "Gradient interference uses leafonly_grad_param_groups() for trainable coverage."
    )
    if ms_end_to_end is not None:
        print(
            "\nNote: Rows above time each submodule alone in eager mode; their sum is not comparable to "
            f"the {ms_end_to_end:.2f} ms end-to-end line (torch.compile fuses the full graph, like InspectModel)."
        )
    else:
        print(
            "\nNote: Rows above time each submodule alone in eager mode; their sum is not comparable to "
            "a single fused torch.compile forward (see InspectModel 'Inference')."
        )


def evaluate_estimator_variance(args, runtime):
    """
    Fix one frame and resample probe matrix Z many times (with the same Jacobi smoothing as training).
    Measures gradient SNR across Z samples: if Hutchinson noise dominates, low-rank off-diagonal heads
    see signal-to-noise ratio below ~1.
    """
    data_folder = runtime["data_folder"]
    save_path = runtime["save_path"]
    runtime_use_gcn = runtime["use_gcn"]
    device = runtime["device"]
    args.num_layers = 2
    args.num_gcn_layers = 2
    args.use_jacobi = True
    use_jacobi = True
    attention_layout = str(args.attention_layout)
    if not runtime_use_gcn:
        raise ValueError("Runtime has use_gcn=False, but the fixed architecture requires 2 GCN layers.")

    print("\n--- Isolating Hutchinson Estimator Variance ---")

    model = LeafOnlyNet(
        input_dim=9,
        d_model=args.d_model,
        leaf_size=LEAF_SIZE,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        attention_layout=attention_layout,
        use_gcn=True,
        num_gcn_layers=2,
        use_jacobi=use_jacobi,
        strip_build_mode=getattr(args, "strip_build_mode", "einsum"),
        off_diag_dense_attention=bool(getattr(args, "off_diag_dense_attn", True)),
    ).to(device)

    if save_path.exists():
        load_leaf_only_weights(model, str(save_path))
        print(f"Loaded weights from {save_path}")
    else:
        print("WARNING: No saved weights found. Analyzing randomly initialized gradients.")

    model.train()

    num_samples = 20
    pv = int(getattr(args, "probe_vectors", -1))
    batch_vectors = pv if pv > 0 else 256

    eval_max_nodes = max(LEAF_SIZE * 2, int(MAX_MIXED_SIZE))
    data_path = Path(data_folder)
    run_folder = most_recent_run_folder(data_path)
    dataset = FluidGraphDataset([run_folder])
    if len(dataset) == 0:
        raise ValueError("No frames in dataset for estimator variance test.")

    frame_idx = 0
    batch = dataset[frame_idx]
    n_pad = MAX_MIXED_SIZE
    n_orig = min(int(batch["num_nodes"]), eval_max_nodes, n_pad)
    x_input = batch["x"][:n_orig].unsqueeze(0).to(device)
    x_input = F.pad(x_input, (0, 0, 0, n_pad - n_orig), value=0.0)
    active_pos = x_input[0, :n_orig, :3]
    x_input[0, :n_orig, :3] = active_pos - active_pos.mean(dim=0, keepdim=True)
    rows, cols = batch["edge_index"][0], batch["edge_index"][1]
    mask = (rows < n_orig) & (cols < n_orig)
    edge_index = batch["edge_index"][:, mask].to(device)
    edge_values = batch["edge_values"][mask].to(device)
    A_sparse = torch.sparse_coo_tensor(edge_index, edge_values, (n_orig, n_orig)).coalesce()
    A_small = A_sparse.to_dense().to(device) if device.type == "mps" else A_sparse.to(device).to_dense()
    A_dense = torch.zeros(n_pad, n_pad, device=device, dtype=A_small.dtype)
    A_dense[:n_orig, :n_orig] = A_small
    A_dense[n_orig:, n_orig:] = torch.eye(n_pad - n_orig, device=device, dtype=A_small.dtype)
    dm, df, om, oe = build_leaf_block_connectivity(
        edge_index,
        edge_values,
        x_input[0, :n_pad, :3],
        LEAF_SIZE,
        device,
        x_input.dtype,
        off_diag_dense_attention=bool(model.off_diag_dense_attention),
    )
    pre_leaf = (dm, df, om, oe)
    global_feat = batch.get("global_features")
    if global_feat is None:
        raise ValueError(f"Missing global_features for frame: {batch.get('frame_path', '<unknown>')}")
    global_feat = global_feat.to(device)
    if global_feat.dim() == 1:
        global_feat = global_feat.unsqueeze(0)

    print(
        f"Sampling Z {num_samples} times with {batch_vectors} vectors each on FIXED frame index {frame_idx} "
        f"(probe_vectors arg: {pv if pv > 0 else 'default 256'}); "
        f"each Z gets {HUTCHINSON_PROBE_JACOBI_STEPS} damped-Jacobi sweeps (ω={HUTCHINSON_PROBE_JACOBI_OMEGA}), "
        "matching training."
    )

    precond_out = model(
        x_input,
        edge_index=edge_index,
        edge_values=edge_values,
        precomputed_leaf_connectivity=pre_leaf,
        global_features=global_feat,
    )

    jacobi_inv_diag = torch.ones(1, n_pad, device=device, dtype=A_dense.dtype)
    diag_A = torch.diagonal(A_dense, 0)
    inv_mask = diag_A.abs() > 1e-6
    jacobi_inv_diag[0, inv_mask] = 1.0 / diag_A[inv_mask]

    param_groups = leafonly_grad_param_groups(model)
    group_gradients = {name: [] for name in param_groups.keys()}

    for step in range(num_samples):
        model.zero_grad()
        Z = torch.randn(1, n_pad, batch_vectors, device=device, dtype=x_input.dtype)
        Z[:, n_orig:, :] = 0.0
        jacobi_smooth_hutchinson_z_inplace(
            Z,
            jacobi_inv_diag,
            [n_orig],
            n_pad,
            lambda Zt: (A_dense @ Zt.squeeze(0)).unsqueeze(0),
        )
        AZ = (A_dense @ Z.squeeze(0)).unsqueeze(0)
        MAZ = apply_block_diagonal_M(
            precond_out,
            AZ,
            leaf_size=LEAF_SIZE,
            leaf_apply_size=LEAF_APPLY_SIZE,
            leaf_apply_off=LEAF_APPLY_SIZE_OFF,
            jacobi_inv_diag=jacobi_inv_diag,
        )
        B_ctx = MAZ.size(0)
        MAZ_flat = MAZ.view(B_ctx, -1)
        Z_flat = Z.view(B_ctx, -1)
        cos_sim = F.cosine_similarity(MAZ_flat, Z_flat, dim=1)
        loss = (1.0 - cos_sim).mean()
        loss.backward(retain_graph=(step < num_samples - 1))

        for group_name, get_params in param_groups.items():
            params = get_params(model)
            grads = []
            for p in params:
                if p.grad is not None:
                    grads.append(p.grad.detach().clone().view(-1))
                else:
                    grads.append(torch.zeros_like(p).view(-1))
            group_gradients[group_name].append(torch.cat(grads))

    print("\n=== Estimator Variance Report (Fixed Frame) ===")
    for group_name, grads in group_gradients.items():
        grads_stack = torch.stack(grads)
        mean_grad = grads_stack.mean(dim=0)
        var_grad = grads_stack.var(dim=0, unbiased=False)
        signal_norm = mean_grad.norm().item()
        noise_norm = torch.sqrt(var_grad.mean() + 1e-12).item()
        snr = signal_norm / noise_norm if noise_norm > 0 else float("inf")
        print(
            f"Group: {group_name:<30} | Signal Norm: {signal_norm:.4e} | "
            f"Noise Norm: {noise_norm:.4e} | SNR: {snr:.4f}"
        )
    print(
        "\nIf SNR < 1.0 for Off-diag U/V heads, global node U/V linears, or off-diag Transformer blocks, "
        "probe noise may dominate the low-rank / FMM signal; try more probe vectors (--probe-vectors) "
        "or a variance-reduced estimator."
    )
