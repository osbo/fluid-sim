import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .architecture import (
    LeafOnlyNet,
    apply_block_diagonal_M,
    next_valid_size,
)
from .checkpoint import load_leaf_only_weights
from .config import LEAF_SIZE, MAX_MIXED_SIZE
from .data import FluidGraphDataset, build_leaf_block_connectivity, most_recent_run_folder


def _sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


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
    ).to(device)

    if save_path.exists():
        load_leaf_only_weights(model, str(save_path))
        print(f"Loaded weights from {save_path}")
    else:
        print("WARNING: No saved weights found. Analyzing randomly initialized gradients.")

    model.train()

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

    num_blocks = args.num_layers
    num_gcn = len(model.embed.gcn) if model.embed.gcn else 0
    param_groups = {}
    param_groups["Lift (linear 0)"] = lambda m: list(m.embed.lift[0].parameters())
    param_groups["Lift (linear 2)"] = lambda m: list(m.embed.lift[2].parameters())
    for g in range(num_gcn):
        param_groups[f"GCN layer {g}"] = lambda m, g=g: list(m.embed.gcn[g].parameters())
    for b in range(num_blocks):
        param_groups[f"Transformer block {b}"] = lambda m, b=b: list(m.blocks[b].parameters())
    for b in range(num_blocks):
        param_groups[f"Off-diag Transformer block {b}"] = lambda m, b=b: list(m.off_diag_blocks[b].parameters())
    param_groups["Diagonal Leaf Head"] = lambda m: list(m.leaf_head.parameters())
    param_groups["Off-diag U/V heads"] = lambda m: list(m.off_diag_head_U.parameters()) + list(m.off_diag_head_V.parameters())
    param_groups["Jacobi params"] = lambda m: list(m.jacobi_gate.parameters())

    group_gradients = {name: [] for name in param_groups.keys()}

    for step in range(num_eval):
        model.zero_grad()
        step_loss_sum = 0.0
        for micro in range(contexts_per_step):
            frame_idx = frame_indices_per_pass[step][micro]
            batch = dataset[frame_idx]
            n_orig = min(int(batch["num_nodes"]), eval_max_nodes)
            n_pad = next_valid_size(n_orig, LEAF_SIZE)
            x_input = batch["x"][:n_orig].unsqueeze(0).to(device)
            if n_pad > n_orig:
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
            pre_leaf = build_leaf_block_connectivity(edge_index, edge_values, x_input[0, :n_pad, :3], LEAF_SIZE, device, x_input.dtype)
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
            MAZ = apply_block_diagonal_M(precond_out, AZ, leaf_size=LEAF_SIZE, jacobi_inv_diag=jacobi_inv_diag)
            raw_loss = ((MAZ - Z) ** 2).mean()
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
    n_orig = min(int(batch["num_nodes"]), eval_max_nodes)
    n_pad = next_valid_size(n_orig, LEAF_SIZE)
    x_input = batch["x"][:n_orig].unsqueeze(0).to(device)
    if n_pad > n_orig:
        x_input = F.pad(x_input, (0, 0, 0, n_pad - n_orig), value=0.0)
    active_pos = x_input[0, :n_orig, :3]
    x_input[0, :n_orig, :3] = active_pos - active_pos.mean(dim=0, keepdim=True)
    rows, cols = batch["edge_index"][0], batch["edge_index"][1]
    mask = (rows < n_orig) & (cols < n_orig)
    edge_index = batch["edge_index"][:, mask].to(device)
    edge_values = batch["edge_values"][mask].to(device)
    pre_leaf = build_leaf_block_connectivity(
        edge_index, edge_values, x_input[0, :n_pad, :3], LEAF_SIZE, device, x_input.dtype
    )
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
    batch_vectors = max(1024, int(round(n_pad ** 0.5)))
    Z = torch.randn(1, n_pad, batch_vectors, device=device, dtype=x_input.dtype)
    Z[:, n_orig:, :] = 0.0
    AZ = (A_dense @ Z.squeeze(0)).unsqueeze(0)

    attn_mask, edge_feats, off_attn_mask, off_edge_feats = pre_leaf
    positions = x_input[0, :, :3]
    B_prof, N_prof, C_prof = x_input.shape
    K_prof = N_prof // LEAF_SIZE
    r_idx_prof, c_idx_prof = torch.triu_indices(K_prof, K_prof, offset=1, device=device)
    P_prof = r_idx_prof.shape[0]

    with torch.no_grad():
        h_lift = model.embed.lift(torch.cat([x_input[..., 3:], global_feat.unsqueeze(1).expand(-1, x_input.size(1), -1)], dim=-1))
        h_gcn = h_lift
        if model.embed.gcn is not None:
            for gcn_layer in model.embed.gcn:
                h_gcn = gcn_layer(h_gcn, edge_index, edge_values)
        h_norm = model.embed.norm(h_gcn)
        h_proj0 = model.enc_input_proj(h_norm)
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
        if P_prof > 0:
            h_k = h_proj0.view(B_prof, K_prof, LEAF_SIZE, C_prof)
            h_pairs = (h_k[:, r_idx_prof] + h_k[:, c_idx_prof]).view(B_prof, P_prof * LEAF_SIZE, C_prof)
            h_off = h_pairs
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
            off_diag_blocks_profile = model._get_leaf_blocks(h_off, mode="off-diagonal")
        else:
            off_diag_blocks_profile = torch.empty((B_prof, 0, LEAF_SIZE, LEAF_SIZE), device=device, dtype=h_proj0.dtype)
        jacobi_scale_profile = model._get_jacobi_scale(h_diag)

    jacobi_inv_diag_profile = torch.ones(1, n_pad, device=device, dtype=A_dense.dtype)
    diag_A_profile = torch.diagonal(A_dense, 0)
    inv_mask_profile = diag_A_profile.abs() > 1e-6
    jacobi_inv_diag_profile[0, inv_mask_profile] = 1.0 / diag_A_profile[inv_mask_profile]

    was_training = model.training
    model.eval()
    try:
        compiled_model = torch.compile(model)
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
            f"\nEnd-to-end forward (torch.compile, precomputed masks, n_pad={n_pad}): "
            f"{ms_end_to_end:.2f} ms — same style as InspectModel 'Inference'"
        )

    timing_rows = []
    ms_lift = _timed_ms(
        lambda: model.embed.lift(
            torch.cat([x_input[..., 3:], global_feat.unsqueeze(1).expand(-1, x_input.size(1), -1)], dim=-1)
        ),
        device,
    )
    timing_rows.append(["Lift MLP", ms_lift])

    if model.embed.gcn is not None and len(model.embed.gcn) > 0:
        h_for_gcn = model.embed.lift(torch.cat([x_input[..., 3:], global_feat.unsqueeze(1).expand(-1, x_input.size(1), -1)], dim=-1))
        for i, gcn_layer in enumerate(model.embed.gcn):
            ms_gcn_i = _timed_ms(lambda g=gcn_layer, h=h_for_gcn: g(h, edge_index, edge_values), device)
            timing_rows.append([f"GCN layer {i}", ms_gcn_i])
            with torch.no_grad():
                h_for_gcn = gcn_layer(h_for_gcn, edge_index, edge_values)

    ms_norm = _timed_ms(lambda: model.embed.norm(h_gcn), device)
    timing_rows.append(["Embedding LayerNorm", ms_norm])

    ms_enc_proj = _timed_ms(lambda: model.enc_input_proj(h_norm), device)
    timing_rows.append(["Encoder input projection", ms_enc_proj])

    h_block_in = h_proj0
    for i, block in enumerate(model.blocks):
        ms_block = _timed_ms(
            lambda b=block, h=h_block_in: b(
                h,
                edge_index=edge_index,
                edge_values=edge_values,
                positions=positions,
                save_attention=False,
                attn_mask=attn_mask,
                edge_feats=edge_feats,
            ),
            device,
        )
        timing_rows.append([f"Leaf attention block {i}", ms_block])
        with torch.no_grad():
            h_block_in = block(
                h_block_in,
                edge_index=edge_index,
                edge_values=edge_values,
                positions=positions,
                save_attention=False,
                attn_mask=attn_mask,
                edge_feats=edge_feats,
            )

    ms_leaf_head = _timed_ms(lambda: model._get_leaf_blocks(h_block_in, mode="diagonal"), device)
    timing_rows.append(["Leaf head (block PSD)", ms_leaf_head])

    if P_prof > 0:
        h_k_t = h_proj0.view(B_prof, K_prof, LEAF_SIZE, C_prof)
        h_pairs_t = (h_k_t[:, r_idx_prof] + h_k_t[:, c_idx_prof]).view(B_prof, P_prof * LEAF_SIZE, C_prof)
        h_off_in = h_pairs_t
        for i, block in enumerate(model.off_diag_blocks):
            ms_ob = _timed_ms(
                lambda b=block, h=h_off_in: b(
                    h,
                    edge_index=edge_index,
                    edge_values=edge_values,
                    positions=positions,
                    save_attention=False,
                    attn_mask=off_attn_mask,
                    edge_feats=off_edge_feats,
                ),
                device,
            )
            timing_rows.append([f"Off-diag attention block {i}", ms_ob])
            with torch.no_grad():
                h_off_in = block(
                    h_off_in,
                    edge_index=edge_index,
                    edge_values=edge_values,
                    positions=positions,
                    save_attention=False,
                    attn_mask=off_attn_mask,
                    edge_feats=off_edge_feats,
                )
        ms_off_head = _timed_ms(lambda: model._get_leaf_blocks(h_off_in, mode="off-diagonal"), device)
        timing_rows.append(["Off-diag head (UV^T)", ms_off_head])

    ms_jacobi = _timed_ms(lambda: model._get_jacobi_scale(h_block_in), device)
    timing_rows.append(["Jacobi gate (node_scalar)", ms_jacobi])

    ms_AZ = _timed_ms(lambda: (A_dense @ Z.squeeze(0)).unsqueeze(0), device)
    timing_rows.append(["A @ Z", ms_AZ])
    B_pack = diag_blocks_profile.shape[0]
    packed_profile = torch.cat(
        [diag_blocks_profile.reshape(B_pack, -1), off_diag_blocks_profile.reshape(B_pack, -1)], dim=1
    )
    if jacobi_scale_profile is not None:
        packed_profile = torch.cat([packed_profile, jacobi_scale_profile], dim=1)
    ms_apply_m = _timed_ms(
        lambda: apply_block_diagonal_M(
            packed_profile,
            AZ,
            leaf_size=LEAF_SIZE,
            jacobi_inv_diag=jacobi_inv_diag_profile,
        ),
        device,
    )
    timing_rows.append(["Apply block-diagonal M", ms_apply_m])

    total_ms = sum(v for _, v in timing_rows) + 1e-12
    timed_rows = [[name, f"{ms:.3f}", f"{(100.0 * ms / total_ms):.1f}%"] for name, ms in timing_rows]
    _print_table(
        "Component Timing (--evaluate_gradients)",
        ["Component", "ms/call", "% of measured total"],
        timed_rows,
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
