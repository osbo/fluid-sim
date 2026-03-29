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
from .config import LEAF_APPLY_SIZE, LEAF_APPLY_SIZE_OFF, LEAF_SIZE, MAX_MIXED_SIZE, MAX_NUM_LEAVES
from .data import FluidGraphDataset, build_leaf_block_connectivity, most_recent_run_folder


def _diag_off_attention_category_rows(W, M, L_phys: int, num_extra: int):
    """
    For query rows 0..L_phys-1 (original mesh / strip tokens), report mean fraction of row mass
    on self, other same-block leaf indices, and each extra column. Compare to mask-aware uniform
    (uniform over keys with M[q,k]==1). Returns (table_rows, mean_total_variation).
    """
    eps = 1e-12
    U = M / M.sum(dim=-1, keepdim=True).clamp(min=eps)
    row_leaf_w = W[:, :, :L_phys, :L_phys]
    diag_w = torch.diagonal(row_leaf_w, dim1=2, dim2=3)
    other_leaf_w = row_leaf_w.sum(dim=-1) - diag_w
    self_m = diag_w.mean().item() * 100.0
    other_m = other_leaf_w.mean().item() * 100.0

    row_leaf_u = U[:, :, :L_phys, :L_phys]
    diag_u = torch.diagonal(row_leaf_u, dim1=2, dim2=3)
    other_leaf_u = row_leaf_u.sum(dim=-1) - diag_u
    self_u = diag_u.mean().item() * 100.0
    other_u = other_leaf_u.mean().item() * 100.0

    rows = [
        ["self", f"{self_m:.2f}", f"{self_u:.2f}", f"{self_m - self_u:+.2f}"],
        [
            "other_leaf_nodes",
            f"{other_m:.2f}",
            f"{other_u:.2f}",
            f"{other_m - other_u:+.2f}",
        ],
    ]
    for j in range(num_extra):
        if j == 0:
            label = "extra_block_token"
        elif j == 1:
            label = "extra_matrix_token"
        else:
            label = f"extra_{j}"
        col = L_phys + j
        wm = W[:, :, :L_phys, col].mean().item() * 100.0
        um = U[:, :, :L_phys, col].mean().item() * 100.0
        rows.append([label, f"{wm:.2f}", f"{um:.2f}", f"{wm - um:+.2f}"])

    diff = (W - U).abs()
    tv = 0.5 * diff[:, :, :L_phys, :].sum(dim=-1).mean().item()
    return rows, tv


def _report_evaluate_gradients_attention_breakdown(
    model: LeafOnlyNet,
    x_input: torch.Tensor,
    edge_index: torch.Tensor,
    edge_values: torch.Tensor,
    pre_leaf: tuple,
    global_feat: torch.Tensor,
    device: torch.device,
):
    """
    One no_grad pass mirroring LeafOnlyNet interleave: per diag/off layer, row-normalized
    mean(softmax + edge_gate) vs mask-aware uniform baseline on original query rows.
    """
    attn_mask, edge_feats, om_prof, oe_prof = pre_leaf
    positions = x_input[0, :, :3]
    B_prof, _N_prof, _ = x_input.shape
    Lf = LEAF_SIZE
    Mh = model.num_h_off
    T_off = int(model.off_tokens_per_block)
    nx = int(model.num_extra)
    otp = int(model.off_token_pool)

    if Mh > 0:
        om = om_prof.to(device=device, dtype=x_input.dtype)
        oe = oe_prof.to(device=device, dtype=x_input.dtype)
        if otp > 1:
            om = pool_leaf_attn_mask(om, Lf, otp, nx)
            oe = pool_leaf_edge_feats(oe, Lf, otp, nx)
        off_attn_mask = om.unsqueeze(0).expand(B_prof, Mh, T_off, T_off).contiguous().reshape(
            B_prof * Mh, 1, T_off, T_off
        )
        off_edge_feats = (
            oe.unsqueeze(0)
            .expand(B_prof, Mh, T_off, T_off, 4)
            .contiguous()
            .reshape(B_prof * Mh, 1, T_off, T_off, 4)
        )
    else:
        off_attn_mask = off_edge_feats = None

    K = MAX_NUM_LEAVES
    n_l = len(model.blocks)
    Ls = int(model.leaf_apply_off)

    print(
        "\n=== Attention mass vs mask-aware uniform (--evaluate_gradients) ===\n"
        "Per layer: rows = mean over batch, blocks, and queries 0..L_phys-1. "
        "Weights = row-normalized mean over heads of (softmax + edge_gate); "
        "Uniform = uniform over keys allowed by attn_mask for that query. "
        "Delta = learned % − uniform % (percentage points). "
        "TV = mean 0.5·Σ_k|p_k−u_k| over those query rows (full key axis).\n"
    )

    if n_l == 0:
        print("No transformer layers; skipping attention breakdown.")
        return

    leaf_kw = dict(
        edge_index=edge_index,
        edge_values=edge_values,
        positions=positions,
        save_attention=False,
        attn_mask=attn_mask,
        edge_feats=edge_feats,
    )
    off_kw = dict(
        edge_index=edge_index,
        edge_values=edge_values,
        positions=positions,
        save_attention=False,
        attn_mask=off_attn_mask,
        edge_feats=off_edge_feats,
    )

    with torch.no_grad():
        h_lift = model.embed.lift(
            torch.cat(
                [x_input[..., 3:], global_feat.unsqueeze(1).expand(-1, x_input.size(1), -1)],
                dim=-1,
            )
        )
        h_gcn = h_lift
        if model.embed.gcn is not None:
            for gcn_layer in model.embed.gcn:
                h_gcn = gcn_layer(h_gcn, edge_index, edge_values)
        h_norm = model.embed.norm(h_gcn)
        h_proj0 = model.enc_input_proj(h_norm)
        C_dim = h_proj0.shape[-1]
        h_diag = model._append_special_tokens_diagonal(h_proj0, B_prof, K, Lf, C_dim, model.num_extra)
        off_stream = None

        for i in range(n_l):
            block = model.blocks[i]
            x_mid = block.norm1(h_diag)
            W, M = block.attn.row_normalized_mean_weights(x_mid, attn_mask, edge_feats)
            rows, tv = _diag_off_attention_category_rows(W, M, Lf, nx)
            _print_table(
                f"Diagonal layer {i} (L_phys={Lf}, num_extra={nx})",
                ["Category", "Learned %", "Uniform %", "Δ (pp)"],
                rows + [["TV (full row)", f"{tv:.4f}", "—", "—"]],
            )
            h_diag = block(h_diag, **leaf_kw)

            if Mh > 0:
                strip = model._build_off_strip(h_diag, B_prof, K, Lf, C_dim, Mh)
                strip = model._append_special_tokens_off_strip(strip, h_diag, B_prof, K, Lf, C_dim, Mh)
                off_in = strip if off_stream is None else strip + off_stream
                ob = model.off_diag_blocks[i]
                x_off = ob.norm1(off_in)
                W_off, M_off = ob.attn.row_normalized_mean_weights(x_off, off_attn_mask, off_edge_feats)
                rows_o, tv_o = _diag_off_attention_category_rows(W_off, M_off, Ls, nx)
                _print_table(
                    f"Off-diagonal layer {i} (L_phys={Ls}, num_extra={nx})",
                    ["Category", "Learned %", "Uniform %", "Δ (pp)"],
                    rows_o + [["TV (full row)", f"{tv_o:.4f}", "—", "—"]],
                )
                off_stream = ob(off_in, **off_kw)


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
                num_extra=int(model.num_extra),
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
            Z = torch.randn(1, n_pad, batch_vectors, device=device, dtype=x_input.dtype)
            Z[:, n_orig:, :] = 0.0
            AZ = (A_dense @ Z.squeeze(0)).unsqueeze(0)
            jacobi_inv_diag = torch.ones(1, n_pad, device=device, dtype=A_dense.dtype)
            diag_A = torch.diagonal(A_dense, 0)
            inv_mask = diag_A.abs() > 1e-6
            jacobi_inv_diag[0, inv_mask] = 1.0 / diag_A[inv_mask]
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
        num_extra=int(model.num_extra),
    )
    pre_leaf = (dm, df, om, oe)
    global_feat = batch.get("global_features")
    if global_feat is None:
        raise ValueError(f"Missing global_features for frame: {batch.get('frame_path', '<unknown>')}")
    global_feat = global_feat.to(device)
    if global_feat.dim() == 1:
        global_feat = global_feat.unsqueeze(0)

    _report_evaluate_gradients_attention_breakdown(
        model, x_input, edge_index, edge_values, pre_leaf, global_feat, device
    )

    A_sparse = torch.sparse_coo_tensor(edge_index, edge_values, (n_orig, n_orig)).coalesce()
    A_small = A_sparse.to_dense().to(device) if device.type == "mps" else A_sparse.to(device).to_dense()
    A_dense = torch.zeros(n_pad, n_pad, device=device, dtype=A_small.dtype)
    A_dense[:n_orig, :n_orig] = A_small
    A_dense[n_orig:, n_orig:] = torch.eye(n_pad - n_orig, device=device, dtype=A_small.dtype)
    batch_vectors = max(1024, int(round(n_pad ** 0.5)))
    Z = torch.randn(1, n_pad, batch_vectors, device=device, dtype=x_input.dtype)
    Z[:, n_orig:, :] = 0.0
    AZ = (A_dense @ Z.squeeze(0)).unsqueeze(0)

    attn_mask, edge_feats, om_prof, oe_prof = pre_leaf
    positions = x_input[0, :, :3]
    B_prof, N_prof = x_input.shape[0], x_input.shape[1]
    assert N_prof == MAX_NUM_LEAVES * LEAF_SIZE, "Profile expects N == MAX_MIXED_SIZE"
    Lf = LEAF_SIZE
    Mh = model.num_h_off
    T_off = int(model.off_tokens_per_block)
    nx = int(model.num_extra)
    otp = int(model.off_token_pool)
    if Mh > 0:
        om = om_prof.to(device=device, dtype=x_input.dtype)
        oe = oe_prof.to(device=device, dtype=x_input.dtype)
        if otp > 1:
            om = pool_leaf_attn_mask(om, Lf, otp, nx)
            oe = pool_leaf_edge_feats(oe, Lf, otp, nx)
        off_attn_mask = om.unsqueeze(0).expand(B_prof, Mh, T_off, T_off).contiguous().reshape(
            B_prof * Mh, 1, T_off, T_off
        )
        off_edge_feats = (
            oe.unsqueeze(0)
            .expand(B_prof, Mh, T_off, T_off, 4)
            .contiguous()
            .reshape(B_prof * Mh, 1, T_off, T_off, 4)
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
        K = MAX_NUM_LEAVES
        h = model._append_special_tokens_diagonal(h_proj0, B_prof, K, Lf, C_dim, model.num_extra)
        h_diag = h
        off_stream = None
        n_l = len(model.blocks)
        T_off = int(model.off_tokens_per_block)
        if Mh > 0:
            if n_l == 0:
                strip_b = model._build_off_strip(h_diag, B_prof, K, Lf, C_dim, Mh)
                off_stream = model._append_special_tokens_off_strip(
                    strip_b, h_diag, B_prof, K, Lf, C_dim, Mh
                )
            else:
                for i in range(n_l):
                    h_diag = model.blocks[i](
                        h_diag,
                        edge_index=edge_index,
                        edge_values=edge_values,
                        positions=positions,
                        save_attention=False,
                        attn_mask=attn_mask,
                        edge_feats=edge_feats,
                    )
                    strip = model._build_off_strip(h_diag, B_prof, K, Lf, C_dim, Mh)
                    strip = model._append_special_tokens_off_strip(
                        strip, h_diag, B_prof, K, Lf, C_dim, Mh
                    )
                    off_in = strip if off_stream is None else strip + off_stream
                    off_stream = model.off_diag_blocks[i](
                        off_in,
                        edge_index=edge_index,
                        edge_values=edge_values,
                        positions=positions,
                        save_attention=False,
                        attn_mask=off_attn_mask,
                        edge_feats=off_edge_feats,
                    )
        else:
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
            h_off = off_stream.view(B_prof, Mh, T_off, C_dim)
            off_diag_blocks_profile = model._get_leaf_blocks(h_off, mode="off-diagonal")
        else:
            off_diag_blocks_profile = torch.empty(
                (B_prof, 0, LEAF_APPLY_SIZE_OFF, LEAF_APPLY_SIZE_OFF), device=device, dtype=h_proj0.dtype
            )
        T = int(model.tokens_per_block)
        h_phys_flat = h_diag.view(B_prof, K, T, C_dim)[:, :, :Lf, :].reshape(B_prof, K * Lf, C_dim)
        node_U_profile = model.node_u(h_phys_flat)
        node_V_profile = model.node_v(h_phys_flat)
        jacobi_scale_profile = model._get_jacobi_scale(h_diag)

    jacobi_inv_diag_profile = torch.ones(1, n_pad, device=device, dtype=A_dense.dtype)
    diag_A_profile = torch.diagonal(A_dense, 0)
    inv_mask_profile = diag_A_profile.abs() > 1e-6
    jacobi_inv_diag_profile[0, inv_mask_profile] = 1.0 / diag_A_profile[inv_mask_profile]

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
        h_block_in = model._append_special_tokens_diagonal(
            h_proj0, B_prof, MAX_NUM_LEAVES, Lf, C_dim, model.num_extra
        )
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
        T_off_t = int(model.off_tokens_per_block)
        if Mh > 0:

            def _build_h_off_strip():
                return model._build_off_strip(h_block_in, B_prof, MAX_NUM_LEAVES, Lf, C_dim, Mh)

            _strip_label = (
                "H off strip pool (einsum row+col + token mean)"
                if getattr(model, "strip_build_mode", "einsum") == "einsum"
                else "H off strip pool (index_add row+col + token mean)"
            )
            ms_strip = _timed_ms(_build_h_off_strip, device)
            timing_rows.append([_strip_label, ms_strip])
            strip_t = _build_h_off_strip()
            h_off_in = model._append_special_tokens_off_strip(
                strip_t, h_block_in, B_prof, MAX_NUM_LEAVES, Lf, C_dim, Mh
            )
            for i, block in enumerate(model.off_diag_blocks):
                h_off_in = _time_transformer_attn_and_mlp(
                    timing_rows, "H off", i, block, h_off_in, device, **off_attn_kw
                )
            h_off_4 = h_off_in.view(B_prof, Mh, T_off_t, C_dim)
            ms_off_head = _timed_ms(lambda: model._get_leaf_blocks(h_off_4, mode="off-diagonal"), device)
            timing_rows.append(["H off head (U V^T)", ms_off_head])

        Ttok = int(model.tokens_per_block)
        h_phys_timing = h_block_in.view(B_prof, MAX_NUM_LEAVES, Ttok, C_dim)[:, :, :Lf, :].reshape(
            B_prof, MAX_NUM_LEAVES * Lf, C_dim
        )
        ms_node_u = _timed_ms(lambda: model.node_u(h_phys_timing), device)
        timing_rows.append(["Global node U linear", ms_node_u])
        ms_node_v = _timed_ms(lambda: model.node_v(h_phys_timing), device)
        timing_rows.append(["Global node V linear", ms_node_v])

        ms_jacobi = _timed_ms(lambda: model._get_jacobi_scale(h_block_in), device)
        timing_rows.append(["Jacobi gate (node_scalar)", ms_jacobi])

        ms_AZ = _timed_ms(lambda: (A_dense @ Z.squeeze(0)).unsqueeze(0), device)
        timing_rows.append(["A @ Z", ms_AZ])
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
        "H-off attention vs MLP, strip pool, both low-rank heads, node U/V, Jacobi gate, pack cat, A@Z, and M apply. "
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
    Fix one frame and resample probe matrix Z many times. Measures gradient SNR across Z samples:
    if Hutchinson noise dominates, low-rank off-diagonal heads see signal-to-noise ratio below ~1.
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
        num_extra=int(model.num_extra),
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
        f"(probe_vectors arg: {pv if pv > 0 else 'default 256'})."
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
