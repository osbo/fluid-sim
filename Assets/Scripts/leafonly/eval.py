import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .architecture import (
    LeafOnlyNet,
    apply_block_structured_M,
    apply_block_structured_M_with_levels,
    build_hodlr_off_diag_structure,
    build_hodlr_operator,
    next_valid_size,
)
from .checkpoint import load_leaf_only_weights
from .config import LEAF_SIZE, MAX_MIXED_SIZE, RANK_BASE_LEVEL1
from .data import FluidGraphDataset, build_leaf_block_connectivity, get_or_compute_offdiag_super_data, most_recent_run_folder


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


def _pearson_corr(x, y):
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    x = x.float()
    y = y.float()
    xm = x - x.mean()
    ym = y - y.mean()
    denom = xm.norm() * ym.norm()
    if denom.item() <= 1e-12:
        return float("nan")
    return float((xm * ym).sum().item() / denom.item())


def evaluate_gradient_interference(args, runtime):
    data_folder = runtime["data_folder"]
    save_path = runtime["save_path"]
    use_global_node = runtime["use_global_node"]
    use_gcn = runtime["use_gcn"]
    max_levels = runtime["max_levels"]
    device = runtime["device"]

    print(f"\n--- Starting Gradient Interference Analysis on {device} ---")

    model = LeafOnlyNet(
        input_dim=9,
        d_model=args.d_model,
        leaf_size=LEAF_SIZE,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        rank_base=RANK_BASE_LEVEL1,
        mask_attention=True,
        use_global_node=use_global_node,
        use_gcn=use_gcn,
        max_levels=max_levels,
    ).to(device)

    if save_path.exists():
        load_leaf_only_weights(model, str(save_path))
        print(f"Loaded weights from {save_path}")
    else:
        print("WARNING: No saved weights found. Analyzing randomly initialized gradients.")

    model.train()

    num_eval = 10
    contexts_per_step = max(1, int(getattr(args, "contexts_per_step", 4)))
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

    num_blocks = getattr(args, "num_layers", 3)
    num_gcn = len(model.embed.gcn) if model.embed.gcn else 0
    param_groups = {}
    param_groups["Lift (linear 0)"] = lambda m: list(m.embed.lift[0].parameters())
    param_groups["Lift (linear 2)"] = lambda m: list(m.embed.lift[2].parameters())
    if getattr(model.embed, "lift_film", None) is not None:
        param_groups["Lift FiLM (global_features)"] = lambda m: list(m.embed.lift_film.parameters())
    for g in range(num_gcn):
        param_groups[f"GCN layer {g}"] = lambda m, g=g: list(m.embed.gcn[g].parameters())
    for b in range(num_blocks):
        param_groups[f"Transformer block {b}"] = lambda m, b=b: list(m.blocks[b].parameters())
    for lvl_idx in range(len(model.level_off_diag)):
        lvl = lvl_idx + 1
        param_groups[f"Off-diag L{lvl} Attention (Q/K/V)"] = lambda m, lvl_idx=lvl_idx: (
            list(m.level_off_diag[lvl_idx].attn_q.parameters())
            + list(m.level_off_diag[lvl_idx].attn_k.parameters())
            + list(m.level_off_diag[lvl_idx].attn_v.parameters())
        )
        param_groups[f"Off-diag L{lvl} Feature LN"] = lambda m, lvl_idx=lvl_idx: list(m.level_off_diag[lvl_idx].feature_ln.parameters())
        param_groups[f"Off-diag L{lvl} W_U/W_V"] = lambda m, lvl_idx=lvl_idx: [
            m.level_off_diag[lvl_idx].proj_U.weight,
            m.level_off_diag[lvl_idx].proj_U.bias,
            m.level_off_diag[lvl_idx].proj_V.weight,
            m.level_off_diag[lvl_idx].proj_V.bias,
        ]
    param_groups["Diagonal Leaf Head"] = lambda m: list(m.core.leaf_head.parameters())

    group_gradients = {name: [] for name in param_groups.keys()}
    global_features_list = []

    for step in range(num_eval):
        model.zero_grad()
        step_loss_sum = 0.0
        for micro in range(contexts_per_step):
            frame_idx = frame_indices_per_pass[step][micro]
            batch = dataset[frame_idx]
            if batch.get("global_features") is not None:
                global_features_list.append(batch["global_features"].detach().cpu().numpy())
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
            scale_A = batch.get("scale_A")
            if scale_A is not None and not isinstance(scale_A, torch.Tensor):
                scale_A = torch.tensor(scale_A, device=device, dtype=x_input.dtype)
            A_sparse = torch.sparse_coo_tensor(edge_index, edge_values, (n_orig, n_orig)).coalesce()
            A_small = A_sparse.to_dense().to(device) if device.type == "mps" else A_sparse.to(device).to_dense()
            A_dense = torch.zeros(n_pad, n_pad, device=device, dtype=A_small.dtype)
            A_dense[:n_orig, :n_orig] = A_small
            A_dense[n_orig:, n_orig:] = torch.eye(n_pad - n_orig, device=device, dtype=A_small.dtype)
            dummy_struct = build_hodlr_off_diag_structure(n_pad, LEAF_SIZE, RANK_BASE_LEVEL1)
            pre_masks = get_or_compute_offdiag_super_data(batch["frame_path"], edge_index, edge_values, dummy_struct, device, x_input.dtype, LEAF_SIZE)
            pre_leaf = build_leaf_block_connectivity(edge_index, edge_values, x_input[0, :n_pad, :3], LEAF_SIZE, device, x_input.dtype)
            global_feat = batch.get("global_features")
            if global_feat is None:
                raise ValueError(f"Missing global_features for frame: {batch.get('frame_path', '<unknown>')}")
            global_feat = global_feat.to(device)
            if global_feat.dim() == 1:
                global_feat = global_feat.unsqueeze(0)
            diag_blocks, off_diag_list = model(
                x_input,
                edge_index=edge_index,
                edge_values=edge_values,
                scale_A=scale_A,
                precomputed_masks=pre_masks,
                precomputed_leaf_connectivity=pre_leaf,
                global_features=global_feat,
            )
            batch_vectors = max(1024, int(round(n_pad ** 0.5)))
            Z = torch.randn(1, n_pad, batch_vectors, device=device, dtype=x_input.dtype)
            Z[:, n_orig:, :] = 0.0
            AZ = (A_dense @ Z.squeeze(0)).unsqueeze(0)
            hodlr_op = build_hodlr_operator(diag_blocks, off_diag_list, model.off_diag_struct, leaf_size=LEAF_SIZE, scale_A=1.0)
            MAZ = hodlr_op.apply(AZ)
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
        is_key_group = (
            group_name.startswith("Off-diag")
            or group_name == "Diagonal Leaf Head"
        )
        if not is_key_group:
            continue
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

    frame_idx = frame_indices_per_pass[0][0]
    batch = dataset[frame_idx]
    n_orig = min(int(batch["num_nodes"]), eval_max_nodes)
    n_pad = next_valid_size(n_orig, LEAF_SIZE)
    x_input = batch["x"][:n_orig].unsqueeze(0).to(device)
    if n_pad > n_orig:
        x_input = F.pad(x_input, (0, 0, 0, n_pad - n_orig), value=0.0)
    x_input[0, :n_orig, :3] = x_input[0, :n_orig, :3] - x_input[0, :n_orig, :3].mean(dim=0, keepdim=True)
    rows, cols = batch["edge_index"][0], batch["edge_index"][1]
    mask = (rows < n_orig) & (cols < n_orig)
    edge_index = batch["edge_index"][:, mask].to(device)
    edge_values = batch["edge_values"][mask].to(device)
    scale_A = batch.get("scale_A")
    if scale_A is not None and not isinstance(scale_A, torch.Tensor):
        scale_A = torch.tensor(scale_A, device=device, dtype=x_input.dtype)
    A_sparse = torch.sparse_coo_tensor(edge_index, edge_values, (n_orig, n_orig)).coalesce()
    A_small = A_sparse.to_dense().to(device) if device.type == "mps" else A_sparse.to(device).to_dense()
    A_dense = torch.zeros(n_pad, n_pad, device=device, dtype=A_small.dtype)
    A_dense[:n_orig, :n_orig] = A_small
    A_dense[n_orig:, n_orig:] = torch.eye(n_pad - n_orig, device=device, dtype=A_small.dtype)
    dummy_struct = build_hodlr_off_diag_structure(n_pad, LEAF_SIZE, RANK_BASE_LEVEL1)
    pre_masks = get_or_compute_offdiag_super_data(batch["frame_path"], edge_index, edge_values, dummy_struct, device, x_input.dtype, LEAF_SIZE)
    pre_leaf = build_leaf_block_connectivity(edge_index, edge_values, x_input[0, :n_pad, :3], LEAF_SIZE, device, x_input.dtype)
    global_feat = batch.get("global_features")
    if global_feat is None:
        raise ValueError(f"Missing global_features for frame: {batch.get('frame_path', '<unknown>')}")
    global_feat = global_feat.to(device)
    if global_feat.dim() == 1:
        global_feat = global_feat.unsqueeze(0)
    model.zero_grad()
    diag_blocks, off_diag_list = model(
        x_input,
        edge_index=edge_index,
        edge_values=edge_values,
        scale_A=scale_A,
        precomputed_masks=pre_masks,
        precomputed_leaf_connectivity=pre_leaf,
        global_features=global_feat,
    )
    struct_list = model.off_diag_struct
    levels_in_struct = sorted(set(s["level"] for s in struct_list)) if struct_list else []
    if not levels_in_struct:
        print("\n(Skipping HODLR level analysis: no off-diagonal levels for this graph size.)")
        return

    n_trained = next_valid_size(MAX_MIXED_SIZE, model.leaf_size)
    trained_struct = build_hodlr_off_diag_structure(n_trained, model.leaf_size, model.rank_base)
    levels_trained = sorted(set(s["level"] for s in trained_struct))
    level_min = levels_trained[0]
    level_max = min(levels_trained[-1], max(levels_in_struct))
    if level_max < level_min:
        level_max = level_min
    print(f"\n  (Trained levels: {levels_trained[0]}..{levels_trained[-1]} from N={n_trained}; evaluating L{level_min} vs L{level_max})")
    batch_vectors = max(128, int(round(n_pad ** 0.5)))
    Z = torch.randn(1, n_pad, batch_vectors, device=device, dtype=x_input.dtype)
    Z[:, n_orig:, :] = 0.0
    AZ = (A_dense @ Z.squeeze(0)).unsqueeze(0)
    scale_A_scalar = scale_A.item() if isinstance(scale_A, torch.Tensor) and scale_A.numel() == 1 else (float(scale_A) if scale_A is not None else 1.0)

    print("\n=== Cross-Level Gradient Comparison (Level-Specific Heads) ===")
    M_L1 = apply_block_structured_M_with_levels(diag_blocks, off_diag_list, AZ, struct_list, leaf_size=LEAF_SIZE, levels_to_include={level_min}, scale_A=scale_A_scalar)
    M_Lmax = apply_block_structured_M_with_levels(diag_blocks, off_diag_list, AZ, struct_list, leaf_size=LEAF_SIZE, levels_to_include={level_max}, scale_A=scale_A_scalar)
    blk_min = model.level_off_diag[level_min - 1]
    blk_max = model.level_off_diag[level_max - 1]

    def _flat_grad(params):
        grads = []
        for p in params:
            if p.grad is not None:
                grads.append(p.grad.detach().clone().view(-1))
            else:
                grads.append(torch.zeros_like(p).view(-1))
        return torch.cat(grads) if grads else torch.zeros(1, device=device)

    shared_params_min = list(blk_min.attn_q.parameters()) + list(blk_min.attn_k.parameters()) + list(blk_min.attn_v.parameters()) + list(blk_min.feature_ln.parameters())
    shared_params_max = list(blk_max.attn_q.parameters()) + list(blk_max.attn_k.parameters()) + list(blk_max.attn_v.parameters()) + list(blk_max.feature_ln.parameters())
    model.zero_grad()
    loss_L1 = ((M_L1 - Z) ** 2).mean()
    loss_L1.backward(retain_graph=True)
    grad_lvl_L1 = _flat_grad(shared_params_min)
    model.zero_grad()
    loss_Lmax = ((M_Lmax - Z) ** 2).mean()
    loss_Lmax.backward()
    grad_lvl_Lmax = _flat_grad(shared_params_max)
    cos_sim_levels = F.cosine_similarity(grad_lvl_L1.unsqueeze(0), grad_lvl_Lmax.unsqueeze(0), dim=1).item()
    _print_table(
        "Cross-Level Gradient",
        ["Levels", "Cosine(Q/K/V+FeatureLN)"],
        [[f"L{level_min} vs L{level_max}", f"{cos_sim_levels:.4f}"]],
    )
    print("Projection gradients are shown separately (different exact-rank sizes).")

    diag_blocks, off_diag_list = model(
        x_input,
        edge_index=edge_index,
        edge_values=edge_values,
        scale_A=scale_A,
        precomputed_masks=pre_masks,
        precomputed_leaf_connectivity=pre_leaf,
        global_features=global_feat,
    )
    M_L1 = apply_block_structured_M_with_levels(diag_blocks, off_diag_list, AZ, struct_list, leaf_size=LEAF_SIZE, levels_to_include={level_min}, scale_A=scale_A_scalar)
    M_Lmax = apply_block_structured_M_with_levels(diag_blocks, off_diag_list, AZ, struct_list, leaf_size=LEAF_SIZE, levels_to_include={level_max}, scale_A=scale_A_scalar)

    def _flat_grad_proj(proj):
        w = proj.weight.grad.clone() if proj.weight.grad is not None else torch.zeros_like(proj.weight)
        b = proj.bias.grad.clone() if proj.bias.grad is not None else torch.zeros_like(proj.bias)
        return torch.cat([w.view(-1), b.view(-1)])

    model.zero_grad()
    loss_L1 = ((M_L1 - Z) ** 2).mean()
    loss_L1.backward(retain_graph=True)
    grad_proj_U_L1 = _flat_grad_proj(blk_min.proj_U)
    grad_proj_V_L1 = _flat_grad_proj(blk_min.proj_V)
    model.zero_grad()
    loss_Lmax = ((M_Lmax - Z) ** 2).mean()
    loss_Lmax.backward()
    grad_proj_U_Lmax = _flat_grad_proj(blk_max.proj_U)
    grad_proj_V_Lmax = _flat_grad_proj(blk_max.proj_V)

    print("\n=== Component Reliance Probe ===")
    def _zero_precomputed_masks(pre_masks_in):
        out = []
        for pm in pre_masks_in:
            mask_raw, strength_raw = pm
            mask_new = torch.zeros_like(mask_raw, dtype=torch.bool)
            out.append((mask_new, torch.zeros_like(strength_raw)))
        return out

    def _zero_pre_leaf(pre_leaf_in):
        leaf_mask_in, leaf_feats_in = pre_leaf_in
        leaf_mask_new = torch.zeros_like(leaf_mask_in)
        leaf_mask_new[..., -1] = 1.0
        leaf_feats_new = torch.zeros_like(leaf_feats_in)
        return (leaf_mask_new, leaf_feats_new)

    was_training = model.training
    model.eval()
    try:
        num_probe = 16
        probe_vectors = max(64, int(round(n_pad ** 0.5)))
        Z_list = []
        for _ in range(num_probe):
            Zp = torch.randn(1, n_pad, probe_vectors, device=device, dtype=x_input.dtype)
            Zp[:, n_orig:, :] = 0.0
            Z_list.append(Zp)

        hodlr_op_base = build_hodlr_operator(diag_blocks, off_diag_list, struct_list, leaf_size=LEAF_SIZE, scale_A=scale_A_scalar)

        def _eval_residual_mse(hodlr_op_local, A_dense_local):
            mses = []
            for Zp in Z_list:
                AZp = (A_dense_local @ Zp.squeeze(0)).unsqueeze(0)
                MAz = hodlr_op_local.apply(AZp)
                resid = MAz - Zp
                mses.append((resid ** 2).mean().item())
            return np.asarray(mses, dtype=np.float64)

        with torch.no_grad():
            base_mse = _eval_residual_mse(hodlr_op_base, A_dense)
            diag_mask_edges = edge_index[0] == edge_index[1]
            offdiag_mask_edges = ~diag_mask_edges
            idx_Lmin = next(idx for idx, s in enumerate(struct_list) if s["level"] == level_min)
            idx_Lmax = next(idx for idx, s in enumerate(struct_list) if s["level"] == level_max) if level_max != level_min else idx_Lmin

            scenarios = []
            x_zero = x_input.clone()
            x_zero.zero_()
            pre_leaf_x_zero = build_leaf_block_connectivity(edge_index, edge_values, x_zero[0, :n_pad, :3], LEAF_SIZE, device, x_zero.dtype)
            scenarios.append(
                (
                    "x_input zeroed (features+positions)",
                    dict(x_input=x_zero, edge_values=edge_values, pre_masks=pre_masks, pre_leaf=pre_leaf_x_zero, global_features=global_feat, A_dense=A_dense),
                )
            )

            global_zero = torch.zeros_like(global_feat)
            scenarios.append(
                (
                    "global_features zeroed",
                    dict(x_input=x_input, edge_values=edge_values, pre_masks=pre_masks, pre_leaf=pre_leaf, global_features=global_zero, A_dense=A_dense),
                )
            )

            pre_masks_zero = _zero_precomputed_masks(pre_masks)
            scenarios.append(
                (
                    "precomputed off-diag masks zeroed",
                    dict(x_input=x_input, edge_values=edge_values, pre_masks=pre_masks_zero, pre_leaf=pre_leaf, global_features=global_feat, A_dense=A_dense),
                )
            )

            pre_leaf_zero = _zero_pre_leaf(pre_leaf)
            scenarios.append(
                (
                    "precomputed leaf connectivity zeroed",
                    dict(x_input=x_input, edge_values=edge_values, pre_masks=pre_masks, pre_leaf=pre_leaf_zero, global_features=global_feat, A_dense=A_dense),
                )
            )

            edge_values_diag = edge_values * diag_mask_edges.to(edge_values.dtype)
            pre_masks_diag_strength = get_or_compute_offdiag_super_data(batch["frame_path"], edge_index, edge_values_diag, dummy_struct, device, x_input.dtype, LEAF_SIZE)
            scenarios.append(
                (
                    "off-diag attention strength: diag-only (AZ fixed)",
                    dict(x_input=x_input, edge_values=edge_values, pre_masks=pre_masks_diag_strength, pre_leaf=pre_leaf, global_features=global_feat, A_dense=A_dense),
                )
            )

            edge_values_offdiag = edge_values * offdiag_mask_edges.to(edge_values.dtype)
            pre_masks_offdiag_strength = get_or_compute_offdiag_super_data(batch["frame_path"], edge_index, edge_values_offdiag, dummy_struct, device, x_input.dtype, LEAF_SIZE)
            scenarios.append(
                (
                    "off-diag attention strength: off-diag-only (AZ fixed)",
                    dict(x_input=x_input, edge_values=edge_values, pre_masks=pre_masks_offdiag_strength, pre_leaf=pre_leaf, global_features=global_feat, A_dense=A_dense),
                )
            )

            def _zero_precomputed_masks_strength_only(pre_masks_in):
                out = []
                for pm in pre_masks_in:
                    mask_raw, strength_raw = pm
                    out.append((mask_raw, torch.zeros_like(strength_raw)))
                return out

            pre_masks_strength_zero = _zero_precomputed_masks_strength_only(pre_masks)
            scenarios.append(
                (
                    "off-diag attention strength: zeroed (mask kept, AZ fixed)",
                    dict(x_input=x_input, edge_values=edge_values, pre_masks=pre_masks_strength_zero, pre_leaf=pre_leaf, global_features=global_feat, A_dense=A_dense),
                )
            )

            print(f"  Probe count: {num_probe}, probe_vectors={probe_vectors}")
            _print_table(
                "Component Reliance Baseline",
                ["Metric", "Value"],
                [
                    ["Base residual MSE mean", f"{base_mse.mean():.6e}"],
                    ["Base residual MSE std", f"{base_mse.std():.6e}"],
                ],
            )

            scenario_rows = []
            for scenario_name, sc in scenarios:
                diag_blocks_ab, off_diag_list_ab = model(
                    sc["x_input"],
                    edge_index=edge_index,
                    edge_values=sc["edge_values"],
                    scale_A=scale_A,
                    precomputed_masks=sc["pre_masks"],
                    precomputed_leaf_connectivity=sc["pre_leaf"],
                    global_features=sc["global_features"],
                )
                hodlr_op_ab = build_hodlr_operator(diag_blocks_ab, off_diag_list_ab, struct_list, leaf_size=LEAF_SIZE, scale_A=scale_A_scalar)
                mse_ab = _eval_residual_mse(hodlr_op_ab, sc["A_dense"])
                if np.isnan(mse_ab).any():
                    print(f"  - {scenario_name}: produced NaNs in residual MSE (component ablation likely masks all attention keys).")
                    continue
                delta = mse_ab - base_mse
                delta_rel = delta / (base_mse + 1e-12)

                h_ab = model.core.forward_features(
                    sc["x_input"],
                    edge_index=edge_index,
                    edge_values=sc["edge_values"],
                    precomputed_leaf_connectivity=sc["pre_leaf"],
                    global_features=sc["global_features"],
                )

                def _attn_debug_for_spec(spec_idx):
                    spec = struct_list[spec_idx]
                    rs, re = spec["row_start"], spec["row_end"]
                    cs, ce = spec["col_start"], spec["col_end"]
                    h_row = h_ab[:, rs:re]
                    h_col = h_ab[:, cs:ce]
                    pos_row = sc["x_input"][:, rs:re, :3]
                    pos_col = sc["x_input"][:, cs:ce, :3]
                    mask_raw, strength_raw = sc["pre_masks"][spec_idx]
                    mask = mask_raw.unsqueeze(0) if mask_raw.dim() == 2 else mask_raw
                    edge_strength = strength_raw.unsqueeze(0) if strength_raw.dim() == 2 else strength_raw
                    blk = model.level_off_diag[spec["level"] - 1]
                    _, _, dbg = blk(
                        h_row,
                        h_col,
                        pos_row,
                        pos_col,
                        mask,
                        super_edge_strength=edge_strength,
                        return_attn_debug=True,
                        global_features=sc["global_features"],
                    )
                    return dbg

                dbg_min = _attn_debug_for_spec(idx_Lmin)
                dbg_max = _attn_debug_for_spec(idx_Lmax) if idx_Lmax != idx_Lmin else dbg_min

                def _fmt_dbg(dbg):
                    return (
                        f"H={dbg['attn_entropy_mean']:.3f}+/-{dbg['attn_entropy_std']:.3f}, "
                        f"E={dbg['edge_scale_mean_on_mask']:.3e}+/-{dbg['edge_scale_std_on_mask']:.3e}"
                    )
                row = [
                    scenario_name,
                    f"{delta_rel.mean():+.4%}",
                    f"{delta_rel.std():.4%}",
                    _fmt_dbg(dbg_min),
                ]
                if level_min != level_max:
                    row.append(_fmt_dbg(dbg_max))
                scenario_rows.append(row)

            headers = ["Scenario", "Delta MSE mean", "Delta MSE std", f"L{level_min} attn (entropy/edge)"]
            if level_min != level_max:
                headers.append(f"L{level_max} attn (entropy/edge)")
            _print_table("Component Reliance Scenarios", headers, scenario_rows)
            print("Interpretation: Delta MSE shows loss sensitivity; attn columns summarize entropy/edge-scale shifts.")

            # Focused A-path diagnostics:
            # Does off-diagonal strength actually change supernode attention and UV^T blocks?
            print("\n=== A-Path Diagnostics (offdiag strength -> attn -> UV^T) ===")

            def _attn_probs_for_spec(spec_idx, strength_override=None):
                spec = struct_list[spec_idx]
                rs, re = spec["row_start"], spec["row_end"]
                cs, ce = spec["col_start"], spec["col_end"]
                h_row = h_base[:, rs:re]
                h_col = h_base[:, cs:ce]
                pos_row = x_input[:, rs:re, :3]
                pos_col = x_input[:, cs:ce, :3]
                mask_raw, strength_raw = pre_masks[spec_idx]
                mask = mask_raw.unsqueeze(0) if mask_raw.dim() == 2 else mask_raw
                edge_strength = strength_raw.unsqueeze(0) if strength_raw.dim() == 2 else strength_raw
                if strength_override is not None:
                    edge_strength = strength_override
                blk = model.level_off_diag[spec["level"] - 1]

                B_local = h_row.shape[0]
                side = h_row.shape[1]
                g = side // blk.n_super
                down_row = h_row.view(B_local, blk.n_super, g, -1).amax(dim=2)
                down_col = h_col.view(B_local, blk.n_super, g, -1).amax(dim=2)
                down_pos_row = pos_row.view(B_local, blk.n_super, g, -1).amax(dim=2)
                down_pos_col = pos_col.view(B_local, blk.n_super, g, -1).amax(dim=2)
                delta_x = down_pos_col - down_pos_row
                dist = torch.norm(delta_x, dim=-1, keepdim=True).clamp(min=1e-8)
                gf = global_feat if global_feat.dim() == 2 else global_feat.unsqueeze(0)
                gf_exp = gf.unsqueeze(1).expand(-1, blk.n_super, -1)
                context_U = torch.cat([delta_x, dist, gf_exp], dim=-1)
                context_V = torch.cat([-delta_x, dist, gf_exp], dim=-1)
                down_row_cond = torch.cat([down_row, context_U], dim=-1)
                down_col_cond = torch.cat([down_col, context_V], dim=-1)
                block_avg_row = torch.cat([down_row.amax(dim=1, keepdim=True), context_U.amax(dim=1, keepdim=True)], dim=-1)
                block_avg_col = torch.cat([down_col.amax(dim=1, keepdim=True), context_V.amax(dim=1, keepdim=True)], dim=-1)
                row_tokens = torch.cat([down_row_cond, block_avg_row], dim=1)
                col_tokens = torch.cat([down_col_cond, block_avg_col], dim=1)
                Q = blk.attn_q(row_tokens)
                K = blk.attn_k(col_tokens)
                scores = (Q @ K.transpose(-2, -1)) * blk._scale_attn
                edge_bias = blk.super_edge_gate(edge_strength.to(dtype=scores.dtype).unsqueeze(-1)).squeeze(-1)
                edge_bias_33 = F.pad(edge_bias, (0, blk.NUM_GLOBAL_TOKENS, 0, blk.NUM_GLOBAL_TOKENS), value=0.0)
                edge_scale_33 = torch.exp(edge_bias_33.clamp(min=-10.0, max=10.0))
                mask_33 = F.pad(mask, (0, blk.NUM_GLOBAL_TOKENS, 0, blk.NUM_GLOBAL_TOKENS), value=True)
                scores_masked = scores.masked_fill(~mask_33, float("-inf"))
                probs = F.softmax(scores_masked, dim=-1)
                probs = probs * edge_scale_33
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
                return probs[:, : blk.n_super, : blk.n_super], mask[:, : blk.n_super, : blk.n_super], edge_strength[:, : blk.n_super, : blk.n_super], edge_bias[:, : blk.n_super, : blk.n_super]

            def _block_uv_matrix(spec_idx, strength_override=None):
                spec = struct_list[spec_idx]
                rs, re = spec["row_start"], spec["row_end"]
                cs, ce = spec["col_start"], spec["col_end"]
                h_row = h_base[:, rs:re]
                h_col = h_base[:, cs:ce]
                pos_row = x_input[:, rs:re, :3]
                pos_col = x_input[:, cs:ce, :3]
                mask_raw, strength_raw = pre_masks[spec_idx]
                mask = mask_raw.unsqueeze(0) if mask_raw.dim() == 2 else mask_raw
                edge_strength = strength_raw.unsqueeze(0) if strength_raw.dim() == 2 else strength_raw
                if strength_override is not None:
                    edge_strength = strength_override
                blk = model.level_off_diag[spec["level"] - 1]
                U, V = blk(h_row, h_col, pos_row, pos_col, mask, super_edge_strength=edge_strength, global_features=global_feat)
                return torch.bmm(U, V.transpose(1, 2))

            with torch.no_grad():
                h_base = model.core.forward_features(
                    x_input,
                    edge_index=edge_index,
                    edge_values=edge_values,
                    precomputed_leaf_connectivity=pre_leaf,
                    global_features=global_feat,
                )

                selected_specs = [idx_Lmin]
                if idx_Lmax != idx_Lmin:
                    selected_specs.append(idx_Lmax)

                a_path_rows = []
                eps = 0.10
                for spec_idx in selected_specs:
                    spec = struct_list[spec_idx]
                    probs_base, mask_base, strength_base, edge_bias_base = _attn_probs_for_spec(spec_idx)
                    strength_pert = strength_base * (1.0 + eps)
                    probs_pert, _, _, _ = _attn_probs_for_spec(spec_idx, strength_override=strength_pert)

                    delta_strength_norm = (strength_pert - strength_base).norm().item() + 1e-12
                    attn_sensitivity = (probs_pert - probs_base).norm().item() / delta_strength_norm

                    M_base = _block_uv_matrix(spec_idx, strength_override=strength_base)
                    M_pert = _block_uv_matrix(spec_idx, strength_override=strength_pert)
                    M_zero = _block_uv_matrix(spec_idx, strength_override=torch.zeros_like(strength_base))
                    uv_sensitivity = (M_pert - M_base).norm().item() / delta_strength_norm
                    strength_attr_ratio = (M_base - M_zero).norm().item() / (M_base.norm().item() + 1e-12)

                    sel = mask_base.bool()
                    s_sel = strength_base.masked_select(sel)
                    p_sel = probs_base.masked_select(sel)
                    align_corr = _pearson_corr(s_sel, p_sel)

                    edge_bias_sel = edge_bias_base.masked_select(sel)
                    edge_bias_mean = edge_bias_sel.mean().item() if edge_bias_sel.numel() > 0 else float("nan")
                    edge_bias_std = edge_bias_sel.std(unbiased=False).item() if edge_bias_sel.numel() > 1 else float("nan")

                    a_path_rows.append(
                        [
                            f"L{spec['level']}",
                            f"{attn_sensitivity:.4e}",
                            f"{uv_sensitivity:.4e}",
                            f"{strength_attr_ratio:.4e}",
                            f"{align_corr:.4f}",
                            f"{edge_bias_mean:.4e}",
                            f"{edge_bias_std:.4e}",
                        ]
                    )

                _print_table(
                    "A-Path Local Sensitivity/Attribution",
                    [
                        "Level",
                        "||dAttn||/||dStrength||",
                        "||dUV||/||dStrength||",
                        "||UV(s)-UV(0)||/||UV(s)||",
                        "corr(strength,attn)",
                        "gate bias mean",
                        "gate bias std",
                    ],
                    a_path_rows,
                )

            # True-loss gradient flow into offdiag strength inputs.
            with torch.enable_grad():
                grad_rows = []
                for spec_idx in selected_specs:
                    spec = struct_list[spec_idx]
                    mask_raw, strength_raw = pre_masks[spec_idx]
                    strength_var = strength_raw.clone().detach().requires_grad_(True)
                    pre_masks_var = list(pre_masks)
                    pre_masks_var[spec_idx] = (mask_raw, strength_var)
                    model.zero_grad()
                    diag_tmp, off_tmp = model(
                        x_input,
                        edge_index=edge_index,
                        edge_values=edge_values,
                        scale_A=scale_A,
                        precomputed_masks=pre_masks_var,
                        precomputed_leaf_connectivity=pre_leaf,
                        global_features=global_feat,
                    )
                    z_one = Z_list[0]
                    az_one = (A_dense @ z_one.squeeze(0)).unsqueeze(0)
                    op_tmp = build_hodlr_operator(diag_tmp, off_tmp, struct_list, leaf_size=LEAF_SIZE, scale_A=scale_A_scalar)
                    loss_tmp = ((op_tmp.apply(az_one) - z_one) ** 2).mean()
                    loss_tmp.backward()
                    grad_norm = strength_var.grad.norm().item() if strength_var.grad is not None else 0.0
                    grad_rows.append([f"L{spec['level']}", f"{grad_norm:.4e}", f"{loss_tmp.item():.6e}"])

                _print_table(
                    "A-Path True-Loss Gradient Flow",
                    ["Level", "||dLoss/dStrength||", "reference loss"],
                    grad_rows,
                )
                model.zero_grad()
    finally:
        if was_training:
            model.train()

