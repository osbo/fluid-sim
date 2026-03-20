import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .architecture import (
    LeafOnlyNet,
    apply_block_diagonal_M,
    next_valid_size,
)
from .checkpoint import load_leaf_only_weights, save_leaf_only_weights
from .config import LEAF_SIZE, MIN_MIXED_SIZE, MAX_MIXED_SIZE
from .data import (
    FluidGraphDataset,
    build_leaf_block_connectivity,
    most_recent_run_folder,
)


def train_leaf_only(args, runtime):
    data_folder = runtime["data_folder"]
    save_path = runtime["save_path"]
    runtime_use_gcn = runtime["use_gcn"]
    print_timing = runtime["print_timing"]
    max_grad_norm = runtime["max_grad_norm"]
    device = runtime["device"]

    print(f"Using device: {device}")

    data_path = Path(data_folder)
    if not data_path.exists():
        raise SystemExit(f"Data folder not found: {data_path}")
    run_folder = most_recent_run_folder(data_path)
    if run_folder != data_path:
        print(f"  [startup] Using most recent run: {run_folder.name}")
    dataset = FluidGraphDataset([run_folder])
    if len(dataset) == 0:
        raise SystemExit(f"No frames found under {run_folder}")

    if args.use_single_frame:
        frame_idx = min(args.frame, len(dataset) - 1)
        frame_indices = [frame_idx]
        print(f"  [startup] Using single frame index {frame_idx} (--use_single_frame True, --frame {args.frame})")
    else:
        rng = random.Random(args.seed)
        if args.num_frames <= 0:
            frame_indices = list(range(len(dataset)))
            print(f"  [startup] Using all {len(frame_indices)} frames (--num_frames 0)")
        else:
            n_sample = min(args.num_frames, len(dataset))
            frame_indices = sorted(rng.sample(range(len(dataset)), n_sample))
            print(f"  [startup] Random sample of {n_sample} frames (--num_frames {args.num_frames})")

    base_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    training_contexts = []
    for frame_idx in frame_indices:
        batch = dataset[frame_idx]
        num_nodes_real = int(batch["num_nodes"])
        target_sizes = []
        for s in base_sizes:
            if MIN_MIXED_SIZE <= s <= MAX_MIXED_SIZE and s <= num_nodes_real:
                target_sizes.append(s)
        if MIN_MIXED_SIZE <= num_nodes_real <= MAX_MIXED_SIZE:
            target_sizes.append(num_nodes_real)
        target_sizes = sorted(set(target_sizes))

        for n in target_sizes:
            if n > num_nodes_real:
                continue
            n_pad = next_valid_size(n, LEAF_SIZE)
            x_full = batch["x"]
            x_input = x_full[:n].unsqueeze(0).to(device)
            if n_pad > n:
                x_input = F.pad(x_input, (0, 0, 0, n_pad - n), value=0.0)
            active_pos = x_input[0, :n, :3]
            centroid = active_pos.mean(dim=0, keepdim=True)
            x_input[0, :n, :3] = active_pos - centroid

            rows, cols = batch["edge_index"][0], batch["edge_index"][1]
            mask = (rows < n) & (cols < n)
            edge_index = batch["edge_index"][:, mask].to(device)
            edge_values = batch["edge_values"][mask].to(device)

            A_indices = batch["edge_index"][:, mask]
            A_vals = batch["edge_values"][mask]
            A_sparse = torch.sparse_coo_tensor(A_indices, A_vals, (n, n)).coalesce()
            A_small = A_sparse.to_dense().to(device) if device.type == "mps" else A_sparse.to(device).to_dense()
            A_dense = torch.zeros(n_pad, n_pad, device=device, dtype=A_small.dtype)
            A_dense[:n, :n] = A_small
            A_dense[n:, n:] = torch.eye(n_pad - n, device=device, dtype=A_small.dtype)

            positions_ctx = x_input[0, :n_pad, :3]
            leaf_attn_mask, leaf_edge_feats, off_attn_mask, off_edge_feats = build_leaf_block_connectivity(
                edge_index, edge_values, positions_ctx, LEAF_SIZE, device, x_input.dtype
            )
            precomputed_leaf_connectivity = (leaf_attn_mask, leaf_edge_feats, off_attn_mask, off_edge_feats)
            batch_vectors = max(128, int(round(n_pad ** 0.5)))
            global_feat = batch.get("global_features")
            if global_feat is None:
                raise ValueError(f"Missing global_features for frame: {batch.get('frame_path', '<unknown>')}")
            global_feat = global_feat.to(device)

            inv_diag = torch.ones(n_pad, device=device, dtype=A_dense.dtype)
            diag_A = torch.diagonal(A_dense, 0)
            inv_mask = diag_A.abs() > 1e-6
            inv_diag[inv_mask] = 1.0 / diag_A[inv_mask]

            training_contexts.append(
                {
                    "n_pad": n_pad,
                    "n_orig": n,
                    "num_leaves": n_pad // LEAF_SIZE,
                    "x_input": x_input,
                    "edge_index": edge_index,
                    "edge_values": edge_values,
                    "A_dense": A_dense,
                    "precomputed_leaf_connectivity": precomputed_leaf_connectivity,
                    "batch_vectors": batch_vectors,
                    "global_features": global_feat,
                    "jacobi_inv_diag": inv_diag,
                }
            )

    if len(training_contexts) == 0:
        raise SystemExit("No valid (frame, size) pairs: ensure frames have at least MIN_MIXED_SIZE nodes.")
    global_max_edges = max(ctx["edge_index"].shape[1] for ctx in training_contexts)
    print(f"  [startup] Cached {len(training_contexts)} training contexts")
    print(f"  [startup] global_max_edges = {global_max_edges}")
    contexts_per_step = max(1, int(args.contexts_per_step))
    print(f"  [startup] contexts_per_step = {contexts_per_step} (gradient accumulation)")

    args.num_layers = 2
    args.num_gcn_layers = 2
    args.use_jacobi = True
    attention_layout = str(args.attention_layout)

    torch.manual_seed(args.seed)
    max_n_pad = max(ctx["n_pad"] for ctx in training_contexts)
    if not runtime_use_gcn:
        raise ValueError("Runtime has use_gcn=False, but the fixed architecture requires 2 GCN layers.")
    requested_gcn_layers = 2
    effective_gcn_layers = requested_gcn_layers
    use_gcn = effective_gcn_layers > 0
    model = LeafOnlyNet(
        input_dim=9,
        d_model=args.d_model,
        leaf_size=LEAF_SIZE,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        attention_layout=attention_layout,
        use_gcn=use_gcn,
        num_gcn_layers=effective_gcn_layers,
        use_jacobi=True,
    ).to(device)
    print(
        "  [startup] Ablation config:"
        f" layers={args.num_layers}, gcn_layers={effective_gcn_layers}, jacobi=True (node_scalar),"
        f" attention_layout={attention_layout}"
    )

    if device.type == "cuda":
        model = torch.compile(model)
        print("  [startup] torch.compile: enabled")

    if args.continue_training:
        if save_path.exists():
            load_leaf_only_weights(model, str(save_path))
            print(f"  [startup] continue_training: Loaded initial state from {save_path}")
        else:
            raise SystemExit(f"--continue_training given but save file not found: {save_path}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        threshold=5e-3,
        threshold_mode="rel",
        cooldown=1,
        min_lr=max(args.lr * 1e-2, 1e-6),
    )
    print(
        "  [startup] LR scheduler: ReduceLROnPlateau"
        f" (factor=0.5, patience=5, threshold=5e-3, min_lr={max(args.lr * 1e-2, 1e-6):.2e})"
    )
    num_leaves_max = max_n_pad // LEAF_SIZE
    print(f"  Block-diagonal preconditioner: {num_leaves_max} leaves of {LEAF_SIZE}x{LEAF_SIZE}")
    model.train()
    print_interval = 100
    loss_history = deque(maxlen=print_interval)
    t_start = time.perf_counter()
    t_start_avg = None

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    TIMING_STEP = 300
    target_step = int(args.target_step)
    target_loss_mean = None
    target_loss_std = None
    target_lr = None
    for step in range(args.steps):
        do_timing = print_timing and (step == TIMING_STEP - 1)
        do_log = step % print_interval == 0
        step_loss_sum = 0.0

        if do_timing:
            _sync()
            t0 = time.perf_counter()
            print(f"\n--- Timing triggered on step {TIMING_STEP} (contexts_per_step={contexts_per_step}) ---")

        optimizer.zero_grad()
        if do_timing:
            _sync()
            t_zero = time.perf_counter() - t0
            t_forward = t_sample_z = t_az = t_apply_m = t_loss = t_backward = 0.0
            n_orig_t, n_pad_t = None, None

        batch_ctx = [random.choice(training_contexts) for _ in range(contexts_per_step)]
        B_step = len(batch_ctx)
        max_n_pad_step = max(ctx["n_pad"] for ctx in batch_ctx)
        max_num_blocks = max_n_pad_step // LEAF_SIZE
        max_P = (max_num_blocks * (max_num_blocks - 1)) // 2
        n_orig_t = batch_ctx[0]["n_orig"]
        n_pad_t = max_n_pad_step
        x_list, A_list, gf_list, inv_diag_list = [], [], [], []
        edge_idx_parts, edge_val_parts = [], []
        leaf_masks_list, leaf_feats_list = [], []
        off_masks_list, off_feats_list = [], []
        n_orig_list = []

        for b_idx, ctx in enumerate(batch_ctx):
            x_ctx = ctx["x_input"]
            n_pad_ctx = ctx["n_pad"]
            n_orig_ctx = ctx["n_orig"]
            n_orig_list.append(n_orig_ctx)
            edge_index_ctx = ctx["edge_index"]
            edge_values_ctx = ctx["edge_values"]
            if n_pad_ctx < max_n_pad_step:
                pad_nodes = max_n_pad_step - n_pad_ctx
                x_ctx = F.pad(x_ctx, (0, 0, 0, pad_nodes), value=0.0)
                A_ctx = torch.zeros(max_n_pad_step, max_n_pad_step, device=device, dtype=ctx["A_dense"].dtype)
                A_ctx[:n_pad_ctx, :n_pad_ctx] = ctx["A_dense"]
                A_ctx[n_pad_ctx:, n_pad_ctx:] = torch.eye(pad_nodes, device=device, dtype=ctx["A_dense"].dtype)
            else:
                A_ctx = ctx["A_dense"]
            x_list.append(x_ctx)
            A_list.append(A_ctx)

            E_ctx = edge_index_ctx.shape[1]
            pad_e = global_max_edges - E_ctx
            if pad_e > 0:
                edge_index_ctx = F.pad(edge_index_ctx, (0, pad_e), value=0)
                edge_values_ctx = F.pad(edge_values_ctx, (0, pad_e), value=0.0)
            offset = b_idx * max_n_pad_step
            edge_idx_parts.append(edge_index_ctx + offset)
            edge_val_parts.append(edge_values_ctx)

            pre_leaf = ctx["precomputed_leaf_connectivity"]
            leaf_mask, leaf_feats, off_mask, off_feats = pre_leaf

            pad_blocks = max_num_blocks - leaf_mask.shape[0]
            if pad_blocks > 0:
                leaf_mask = F.pad(leaf_mask, (0, 0, 0, 0, 0, pad_blocks), value=0.0)
                leaf_feats = F.pad(leaf_feats, (0, 0, 0, 0, 0, 0, 0, pad_blocks), value=0.0)

            pad_P = max_P - (off_mask.shape[0] if off_mask is not None else 0)
            if pad_P > 0:
                if off_mask is None or off_mask.shape[0] == 0:
                    off_mask = torch.zeros(max_P, LEAF_SIZE, LEAF_SIZE + 1, device=device, dtype=x_ctx.dtype)
                    off_feats = torch.zeros(max_P, LEAF_SIZE, LEAF_SIZE + 1, 4, device=device, dtype=x_ctx.dtype)
                else:
                    off_mask = F.pad(off_mask, (0, 0, 0, 0, 0, pad_P), value=0.0)
                    off_feats = F.pad(off_feats, (0, 0, 0, 0, 0, 0, 0, pad_P), value=0.0)

            leaf_masks_list.append(leaf_mask)
            leaf_feats_list.append(leaf_feats)
            off_masks_list.append(off_mask)
            off_feats_list.append(off_feats)

            gf = ctx["global_features"]
            gf_list.append(gf if gf.dim() == 1 else gf.squeeze(0))
            inv_diag_ctx = ctx["jacobi_inv_diag"]
            if n_pad_ctx < max_n_pad_step:
                inv_diag_ctx = F.pad(inv_diag_ctx, (0, max_n_pad_step - n_pad_ctx), value=1.0)
            inv_diag_list.append(inv_diag_ctx)

        x_batched = torch.cat(x_list, dim=0)
        A_batched = torch.stack(A_list, dim=0)
        edge_index_batched = torch.cat(edge_idx_parts, dim=1)
        edge_values_batched = torch.cat(edge_val_parts, dim=0)
        global_features_batched = torch.stack(gf_list, dim=0)
        jacobi_inv_diag_batched = torch.stack(inv_diag_list, dim=0)
        pre_leaf_batched = (
            torch.stack(leaf_masks_list, dim=0),
            torch.stack(leaf_feats_list, dim=0),
            torch.stack(off_masks_list, dim=0),
            torch.stack(off_feats_list, dim=0),
        )

        batch_vectors = max(256, int(round(max_n_pad_step ** 0.5)))
        if do_timing:
            _sync()
            t0 = time.perf_counter()
        precond_out = model(
            x_batched,
            edge_index=edge_index_batched,
            edge_values=edge_values_batched,
            precomputed_leaf_connectivity=pre_leaf_batched,
            global_features=global_features_batched,
        )
        if do_timing:
            _sync()
            t_forward += time.perf_counter() - t0

        if do_timing:
            _sync()
            t0 = time.perf_counter()
        Z = torch.randn(B_step, max_n_pad_step, batch_vectors, device=device, dtype=x_batched.dtype)
        for b_idx, n_orig_ctx in enumerate(n_orig_list):
            if n_orig_ctx < max_n_pad_step:
                Z[b_idx, n_orig_ctx:, :] = 0.0
        if do_timing:
            _sync()
            t_sample_z += time.perf_counter() - t0

        if do_timing:
            _sync()
            t0 = time.perf_counter()
        AZ = torch.bmm(A_batched, Z)
        if do_timing:
            _sync()
            t_az += time.perf_counter() - t0

        if do_timing:
            _sync()
            t0 = time.perf_counter()
        MAZ = apply_block_diagonal_M(precond_out, AZ, leaf_size=LEAF_SIZE, jacobi_inv_diag=jacobi_inv_diag_batched)
        if do_timing:
            _sync()
            t_apply_m += time.perf_counter() - t0

        if do_timing:
            _sync()
            t0 = time.perf_counter()
        residual = MAZ - Z
        raw_loss = (residual ** 2).mean()
        step_loss_sum += raw_loss.item()
        loss = raw_loss
        if do_timing:
            _sync()
            t_loss += time.perf_counter() - t0

        if do_timing:
            _sync()
            t0 = time.perf_counter()
        loss.backward()
        if do_timing:
            _sync()
            t_backward += time.perf_counter() - t0
            t0 = time.perf_counter()

        step_loss = step_loss_sum
        loss_history.append(step_loss)

        if step % print_interval == 0:
            def _grad_norm(params):
                total = torch.tensor(0.0, device=device)
                for p in params:
                    if p.grad is not None:
                        total += p.grad.data.pow(2).sum()
                return total.sqrt().item()

            _all = list(model.parameters())
            _leaf = [p for n, p in model.named_parameters() if "leaf_head" in n]
            log_tot = _grad_norm(_all)
            log_leaf = _grad_norm(_leaf)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if do_timing:
            _sync()
            t_clip = time.perf_counter() - t0
            t0 = time.perf_counter()

        optimizer.step()
        if do_timing:
            _sync()
            t_optim = time.perf_counter() - t0
            total = t_zero + t_forward + t_sample_z + t_az + t_apply_m + t_loss + t_backward + t_clip + t_optim
            if n_orig_t is not None and n_pad_t is not None:
                print(f"  first micro-batch size: n={n_orig_t} (padded to {n_pad_t})")
            print(f"--- Step {TIMING_STEP} detailed timing (ms) ---")
            print(f"  zero_grad:        {t_zero*1000:8.2f}")
            print(f"  model forward:    {t_forward*1000:8.2f}")
            print(f"  sample Z:         {t_sample_z*1000:8.2f}")
            print(f"  A @ Z:            {t_az*1000:8.2f}")
            print(f"  apply M (MAZ):    {t_apply_m*1000:8.2f}")
            print(f"  residual + loss:  {t_loss*1000:8.2f}")
            print(f"  backward:         {t_backward*1000:8.2f}")
            print(f"  clip_grad_norm:   {t_clip*1000:8.2f}")
            print(f"  optimizer.step:   {t_optim*1000:8.2f}")
            print(f"  total:            {total*1000:8.2f}")
            print("----------------------------------------\n")

        if step % print_interval == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_start
            n = len(loss_history)
            if n > 0:
                arr = np.array(loss_history)
                loss_avg, loss_std = float(arr.mean()), float(arr.std())
                loss_str = f"Loss avg={loss_avg:.4f} ± {loss_std:.4f}"
            else:
                loss_avg = float(step_loss)
                loss_str = f"Loss {step_loss:.6f}"

            lr_before = optimizer.param_groups[0]["lr"]
            if step > 0:
                scheduler.step(loss_avg)
            lr_after = optimizer.param_groups[0]["lr"]
            if lr_after < lr_before:
                print(f"  [lr] plateau detected: lr {lr_before:.2e} -> {lr_after:.2e}")

            if step >= TIMING_STEP:
                now = time.perf_counter()
                if step == TIMING_STEP:
                    t_start_avg = now
                    avg_per_100 = elapsed
                else:
                    avg_per_100 = (now - t_start_avg) * 100 / (step - TIMING_STEP)
                print(
                    f"{step:05d}: {loss_str}  ({elapsed:.3f}s, avg: {avg_per_100:.3f}s)"
                    f" lr={lr_after:.2e} gTot={log_tot:.2e} gL={log_leaf:.2e}"
                )
            else:
                print(
                    f"{step:05d}: {loss_str}  ({elapsed:.3f}s)"
                    f" lr={lr_after:.2e} gTot={log_tot:.2e} gL={log_leaf:.2e}"
                )
            if step > 0 and step % (print_interval * 10) == 0:
                save_leaf_only_weights(model, str(save_path), input_dim=9)
            if step == target_step:
                target_loss_mean = loss_avg
                target_loss_std = loss_std if n > 0 else 0.0
                target_lr = lr_after
            t_start = time.perf_counter()

    save_leaf_only_weights(model, str(save_path), input_dim=9)
    print(f"Saved to {save_path}")
    return {
        "target_step": target_step,
        "loss_mean_at_target": target_loss_mean,
        "loss_std_at_target": target_loss_std,
        "lr_at_target": target_lr,
        "save_path": str(save_path),
    }
