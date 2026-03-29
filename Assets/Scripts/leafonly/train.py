import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .architecture import LeafOnlyNet, apply_block_diagonal_M
from .checkpoint import load_leaf_only_weights, save_leaf_only_weights
from .config import (
    LEAF_APPLY_SIZE,
    LEAF_APPLY_SIZE_OFF,
    LEAF_SIZE,
    MAX_MIXED_SIZE,
    MIN_MIXED_SIZE,
    effective_aligned_num_nodes,
)
from .context_cache import (
    build_training_context_cache_meta,
    load_training_contexts_from_cache,
    move_training_context_entries_to_device,
    save_training_contexts_to_cache,
)
from .data import (
    FluidGraphDataset,
    build_leaf_block_connectivity,
    most_recent_run_folder,
)
from .hmatrix import NUM_HMATRIX_OFF_BLOCKS


def _bucket_cuda_profiler_key(key: str) -> str:
    """Coarse labels for explaining where backward GPU time goes (heuristic on op names)."""
    k = str(key).lower()
    if "backward" in k or "autograd" in k:
        return "autograd-labeled"
    if "triton" in k or "inductor" in k or "compiled_autograd" in k:
        return "inductor/triton"
    if "mm" in k or "matmul" in k or "bmm" in k or "addmm" in k or "mv" in k:
        return "matmul/gemm"
    if "softmax" in k or "sdp" in k or "flash" in k or "scaled_dot" in k:
        return "softmax/attn"
    if "norm" in k or "layer_norm" in k or "batch_norm" in k:
        return "norm"
    if "add" in k or "mul" in k or "sub" in k or "div" in k:
        return "elementwise"
    return "other"


def _profiler_self_gpu_time_us(event_avg: Any) -> float:
    """Microseconds of self GPU time; PyTorch 2.6+ uses self_device_time_total, older uses self_cuda_time_total."""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(event_avg, attr):
            return float(getattr(event_avg, attr))
    return 0.0


def _print_backward_cuda_bucket_summary(prof: Any) -> None:
    """Sum self GPU time by coarse bucket (backward mixes many aten ops without 'backward' in name)."""
    from collections import defaultdict

    buckets = defaultdict(float)
    for e in prof.key_averages():
        dt = _profiler_self_gpu_time_us(e)
        if dt <= 0:
            continue
        buckets[_bucket_cuda_profiler_key(e.key)] += dt
    if not buckets:
        return
    total = sum(buckets.values())
    print("  Backward GPU self-time by coarse bucket (heuristic):")
    for name in sorted(buckets.keys(), key=lambda x: -buckets[x]):
        pct = 100.0 * buckets[name] / total if total > 0 else 0.0
        print(f"    {name:22s}  {buckets[name]:8.1f} µs  ({pct:5.1f}%)")


def train_leaf_only(args, runtime):
    data_folder = runtime["data_folder"]
    save_path = runtime["save_path"]
    runtime_use_gcn = runtime["use_gcn"]
    print_timing = runtime["print_timing"]
    max_grad_norm = runtime["max_grad_norm"]
    device = runtime["device"]

    t_wall0 = time.perf_counter()
    print(f"Using device: {device}")

    t_seg = time.perf_counter()
    data_path = Path(data_folder)
    if not data_path.exists():
        raise SystemExit(f"Data folder not found: {data_path}")
    run_folder = most_recent_run_folder(data_path)
    if run_folder != data_path:
        print(f"  [startup] Using most recent run: {run_folder.name}")
    dataset = FluidGraphDataset([run_folder])
    if len(dataset) == 0:
        raise SystemExit(f"No frames found under {run_folder}")
    ms_dataset = (time.perf_counter() - t_seg) * 1000.0

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

    rebuild_context_cache = bool(getattr(args, "rebuild_context_cache", False))
    cache_dir = data_path / ".leafonly_training_context_cache"
    cache_meta = build_training_context_cache_meta(dataset, run_folder, args, frame_indices)
    contexts_from_cache = False
    training_contexts = None
    ctx_tf_ms = ctx_a_ms = ctx_conn_ms = ctx_other_ms = 0.0

    if not rebuild_context_cache:
        t_try = time.perf_counter()
        raw_entries = load_training_contexts_from_cache(cache_dir, cache_meta)
        if raw_entries is not None:
            training_contexts = move_training_context_entries_to_device(raw_entries, device)
            contexts_from_cache = True
            ms_contexts_total = (time.perf_counter() - t_try) * 1000.0
            print(
                f"  [startup] Loaded {len(training_contexts)} training contexts from disk cache "
                f"({ms_contexts_total:.2f} ms) → {cache_dir.name}/"
            )

    if training_contexts is None:
        t_seg = time.perf_counter()
        training_contexts = []
        for frame_idx in frame_indices:
            batch = dataset[frame_idx]
            num_nodes_real = int(batch["num_nodes"])
            n = effective_aligned_num_nodes(num_nodes_real)
            if n < MIN_MIXED_SIZE:
                continue
            n_pad = MAX_MIXED_SIZE
            t_o0 = time.perf_counter()
            x_full = batch["x"]
            x_input = x_full[:n].unsqueeze(0).to(device)
            x_input = F.pad(x_input, (0, 0, 0, n_pad - n), value=0.0)
            active_pos = x_input[0, :n, :3]
            centroid = active_pos.mean(dim=0, keepdim=True)
            x_input[0, :n, :3] = active_pos - centroid

            rows, cols = batch["edge_index"][0], batch["edge_index"][1]
            mask = (rows < n) & (cols < n)
            edge_index = batch["edge_index"][:, mask].to(device)
            edge_values = batch["edge_values"][mask].to(device)
            ctx_tf_ms += (time.perf_counter() - t_o0) * 1000.0

            t_a0 = time.perf_counter()
            A_indices = batch["edge_index"][:, mask]
            A_vals = batch["edge_values"][mask]
            A_sparse = torch.sparse_coo_tensor(A_indices, A_vals, (n, n)).coalesce()
            A_small = A_sparse.to_dense().to(device) if device.type == "mps" else A_sparse.to(device).to_dense()
            A_dense = torch.zeros(n_pad, n_pad, device=device, dtype=A_small.dtype)
            A_dense[:n, :n] = A_small
            A_dense[n:, n:] = torch.eye(n_pad - n, device=device, dtype=A_small.dtype)
            ctx_a_ms += (time.perf_counter() - t_a0) * 1000.0

            t_c0 = time.perf_counter()
            positions_ctx = x_input[0, :n_pad, :3]
            dm, df, om, oe = build_leaf_block_connectivity(
                edge_index, edge_values, positions_ctx, LEAF_SIZE, device, x_input.dtype
            )
            precomputed_leaf_connectivity = (dm, df, om, oe)
            ctx_conn_ms += (time.perf_counter() - t_c0) * 1000.0

            t_r0 = time.perf_counter()
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
            ctx_other_ms += (time.perf_counter() - t_r0) * 1000.0

        ms_contexts_total = (time.perf_counter() - t_seg) * 1000.0
        saved = save_training_contexts_to_cache(cache_dir, cache_meta, training_contexts)
        if saved is not None:
            print(f"  [startup] Wrote disk cache {saved.name} under {cache_dir.name}/")
        else:
            print("  [startup] Could not write context disk cache (permissions/disk); next run will rebuild from frames.")
        print(f"  [startup] Cached {len(training_contexts)} training contexts in memory")

    n_ctx = len(training_contexts)

    if len(training_contexts) == 0:
        raise SystemExit(
            "No valid training contexts: every frame was skipped. "
            f"Need aligned size n ≥ MIN_MIXED_SIZE ({MIN_MIXED_SIZE}) after capping by MAX_MIXED_SIZE ({MAX_MIXED_SIZE}); "
            "lower MIN_MIXED_SIZE in leafonly/config.py if your frames are smaller."
        )
    global_max_edges = max(ctx["edge_index"].shape[1] for ctx in training_contexts)
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
    t_seg = time.perf_counter()
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
    ms_model = (time.perf_counter() - t_seg) * 1000.0
    print(
        "  [startup] Ablation config:"
        f" layers={args.num_layers}, gcn_layers={effective_gcn_layers}, jacobi=True (node_scalar),"
        f" attention_layout={attention_layout}"
    )

    ms_compile = 0.0
    if device.type == "cuda":
        t_seg = time.perf_counter()
        model = torch.compile(model)
        ms_compile = (time.perf_counter() - t_seg) * 1000.0
        print("  [startup] torch.compile: enabled")

    leafonly_pcg = str(getattr(args, "leafonly_pcg", "bsr"))

    la_d_model = int(model.leaf_apply_size)
    la_o_model = int(model.leaf_apply_off)

    def _probe_bmm(A: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        return torch.bmm(A, Z)

    def _probe_apply_m(
        precond_out: torch.Tensor,
        AZ: torch.Tensor,
        jacobi_inv_diag_batched: torch.Tensor,
    ) -> torch.Tensor:
        return apply_block_diagonal_M(
            precond_out,
            AZ,
            leaf_size=LEAF_SIZE,
            leaf_apply_size=la_d_model,
            leaf_apply_off=la_o_model,
            jacobi_inv_diag=jacobi_inv_diag_batched,
        )

    if device.type == "cuda":
        compiled_probe_bmm = torch.compile(_probe_bmm)
        # Training cannot use cuSPARSE BSR sparse.mm for MAZ: backward is unsupported for
        # SparseBsr @ dense (see sparse_addmm_sparse_backward). Both modes use batched
        # apply_block_diagonal_M (same operator as expanded M); matrix_free also compiles it.
        compiled_probe_apply_m = torch.compile(_probe_apply_m) if leafonly_pcg == "matrix_free" else None
    else:
        compiled_probe_bmm = _probe_bmm
        compiled_probe_apply_m = _probe_apply_m if leafonly_pcg == "matrix_free" else None

    ms_load = 0.0
    if args.continue_training:
        if save_path.exists():
            t_seg = time.perf_counter()
            load_leaf_only_weights(model, str(save_path))
            ms_load = (time.perf_counter() - t_seg) * 1000.0
            print(f"  [startup] continue_training: Loaded initial state from {save_path}")
        else:
            raise SystemExit(f"--continue_training given but save file not found: {save_path}")

    t_seg = time.perf_counter()
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
    ms_optim = (time.perf_counter() - t_seg) * 1000.0
    print(
        "  [startup] LR scheduler: ReduceLROnPlateau"
        f" (factor=0.5, patience=5, threshold=5e-3, min_lr={max(args.lr * 1e-2, 1e-6):.2e})"
    )
    num_leaves_max = max_n_pad // LEAF_SIZE
    print(
        f"  Block preconditioner: {num_leaves_max} leaves, diag {LEAF_APPLY_SIZE}×{LEAF_APPLY_SIZE}, "
        f"off {LEAF_APPLY_SIZE_OFF}×{LEAF_APPLY_SIZE_OFF} ({LEAF_SIZE} nodes/leaf)"
    )
    _pv = int(getattr(args, "probe_vectors", -1))
    if _pv >= 0:
        print(f"  Probe vectors (loss MC columns): fixed K={_pv} (--probe-vectors)")
    else:
        print(
            "  Probe vectors (loss MC columns): auto max(256, ⌈√n_pad⌉) per step "
            "(override with --probe-vectors K)"
        )
    if getattr(args, "profile_backward", False):
        if device.type != "cuda":
            print("  --profile-backward: ignored (CUDA only)")
        elif print_timing:
            print("  --profile-backward: CUDA kernel table after loss.backward() at detailed timing step")
        else:
            print("  --profile-backward: no effect (enable print_timing in runtime for detailed step)")
    _pcg_tail = (
        "apply_block_diagonal_M torch.compile (matrix_free)"
        if leafonly_pcg == "matrix_free"
        else "apply_block_diagonal_M eager (bsr; matches expanded M, autograd-safe)"
    )
    _bmm_desc = "bmm torch.compile" if device.type == "cuda" else "bmm eager"
    print(f"  Probe MAZ path: --leafonly-pcg={leafonly_pcg} ({_bmm_desc}; {_pcg_tail})")
    ms_startup_to_loop = (time.perf_counter() - t_wall0) * 1000.0
    if print_timing:
        print("\n=== Startup timing (wall clock, ms) ===")
        print(f"  Dataset + run folder scan:     {ms_dataset:10.2f}")
        if contexts_from_cache:
            print(f"  Load training contexts (disk→device): {ms_contexts_total:10.2f} ms  ({n_ctx} contexts)")
        else:
            print(
                f"  Build {n_ctx} training contexts: {ms_contexts_total:10.2f}  (avg {ms_contexts_total / max(1, n_ctx):.2f} ms/context)"
            )
            print(f"    x/edges → device + centroid:   {ctx_tf_ms:10.2f}")
            print(f"    A_sparse → A_dense + pad:    {ctx_a_ms:10.2f}")
            print(f"    n-hop connectivity:          {ctx_conn_ms:10.2f}")
            print(f"    jacobi vec + append dict:    {ctx_other_ms:10.2f}")
        print(f"  LeafOnlyNet + .to(device):     {ms_model:10.2f}")
        if device.type == "cuda":
            print(f"  torch.compile(model) call:     {ms_compile:10.2f}  (Inductor runs on 1st forward/backward)")
        else:
            print(f"  torch.compile:                  skipped (not CUDA)")
        if ms_load > 0:
            print(f"  load_leaf_only_weights:        {ms_load:10.2f}")
        else:
            print(f"  load_leaf_only_weights:         skipped")
        print(f"  AdamW + ReduceLROnPlateau:     {ms_optim:10.2f}")
        print(f"  ---")
        print(f"  Total before training loop:    {ms_startup_to_loop:10.2f}")
        print(
            "  Note: First loss line’s (elapsed) is wall time for that step only (after startup); "
            "with torch.compile, the first forward+backward triggers Inductor codegen (often ~10–20s).\n"
            "  At step 299, a one-time detailed breakdown (zero_grad / forward / apply M / backward / …) is printed.\n"
        )
    model.train()
    print_interval = 100
    loss_history = deque(maxlen=print_interval)
    t_start = time.perf_counter()
    t_start_avg = None

    def _cuda_sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    # Rolling `avg:` column (matches historical LeafOnly logs).
    TIMING_STEP = 300
    # One-time wall-clock breakdown of substeps (same idea as LeafOnlyOld.py).
    DETAILED_TIMING_STEP = 300
    target_step = int(args.target_step)
    target_loss_mean = None
    target_loss_std = None
    target_lr = None
    for step in range(args.steps):
        do_log = step % print_interval == 0
        step_loss_sum = 0.0
        do_detailed = bool(print_timing) and (step == DETAILED_TIMING_STEP - 1)
        wall_iter_start = None
        if do_detailed:
            wall_iter_start = time.perf_counter()
            _cuda_sync()
            print(
                f"\n--- Detailed timing for step {DETAILED_TIMING_STEP} "
                f"(contexts_per_step={contexts_per_step}, batched forward) ---"
            )
            t_mark = time.perf_counter()
            t_zero_grad = t_batch = t_forward = t_z = t_az = t_apply = t_loss = t_backward = t_clip = t_optim = 0.0
            n_pad_t = B_step_t = None

        optimizer.zero_grad()
        if do_detailed:
            _cuda_sync()
            t_zero_grad = time.perf_counter() - t_mark
            t_mark = time.perf_counter()

        batch_ctx = [random.choice(training_contexts) for _ in range(contexts_per_step)]
        B_step = len(batch_ctx)
        max_n_pad_step = max(ctx["n_pad"] for ctx in batch_ctx)
        max_num_blocks = max_n_pad_step // LEAF_SIZE
        max_M_off = NUM_HMATRIX_OFF_BLOCKS
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

            pad_M = max_M_off - (off_mask.shape[0] if off_mask is not None else 0)
            if off_mask is None or off_mask.shape[0] == 0:
                raise RuntimeError(
                    "Context missing H off-diagonal masks/feats (expected from build_leaf_block_connectivity). "
                    "Rebuild training context cache with --rebuild-context-cache."
                )
            if pad_M > 0:
                off_mask = F.pad(off_mask, (0, 0, 0, 0, 0, pad_M), value=0.0)
                off_feats = F.pad(off_feats, (0, 0, 0, 0, 0, 0, 0, pad_M), value=0.0)

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

        if do_detailed:
            _cuda_sync()
            t_batch = time.perf_counter() - t_mark
            t_mark = time.perf_counter()
            n_pad_t, B_step_t = max_n_pad_step, B_step

        probe_arg = int(getattr(args, "probe_vectors", -1))
        if probe_arg < 0:
            batch_vectors = max(256, int(round(max_n_pad_step ** 0.5)))
        else:
            batch_vectors = max(1, probe_arg)
        precond_out = model(
            x_batched,
            edge_index=edge_index_batched,
            edge_values=edge_values_batched,
            precomputed_leaf_connectivity=pre_leaf_batched,
            global_features=global_features_batched,
        )

        if do_detailed:
            _cuda_sync()
            t_forward = time.perf_counter() - t_mark
            t_mark = time.perf_counter()

        Z = torch.randn(B_step, max_n_pad_step, batch_vectors, device=device, dtype=x_batched.dtype)
        for b_idx, n_orig_ctx in enumerate(n_orig_list):
            if n_orig_ctx < max_n_pad_step:
                Z[b_idx, n_orig_ctx:, :] = 0.0

        if do_detailed:
            _cuda_sync()
            t_z = time.perf_counter() - t_mark
            t_mark = time.perf_counter()

        AZ = compiled_probe_bmm(A_batched, Z)

        if do_detailed:
            _cuda_sync()
            t_az = time.perf_counter() - t_mark
            t_mark = time.perf_counter()

        if leafonly_pcg == "matrix_free":
            assert compiled_probe_apply_m is not None
            MAZ = compiled_probe_apply_m(precond_out, AZ, jacobi_inv_diag_batched)
        else:
            MAZ = _probe_apply_m(precond_out, AZ, jacobi_inv_diag_batched)

        if do_detailed:
            _cuda_sync()
            t_apply = time.perf_counter() - t_mark
            t_mark = time.perf_counter()

        MAZ_flat = MAZ.view(B_step, -1)
        Z_flat = Z.view(B_step, -1)
        cos_sim = F.cosine_similarity(MAZ_flat, Z_flat, dim=1)
        raw_loss = (1.0 - cos_sim).mean()
        step_loss_sum += raw_loss.item()
        loss = raw_loss

        if do_detailed:
            _cuda_sync()
            t_loss = time.perf_counter() - t_mark
            t_mark = time.perf_counter()

        prof: Optional[Any] = None
        profile_bw = (
            bool(getattr(args, "profile_backward", False))
            and do_detailed
            and device.type == "cuda"
        )
        if profile_bw:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=False,
            ) as prof:
                loss.backward()
            _cuda_sync()
        else:
            # Backward: chain rule from loss → MAZ → apply_block_diagonal_M(precond, AZ) → precond_out,
            # then through compiled LeafOnlyNet (attention/GCN/heads). AZ has no grad; matmul backward
            # is ~2× forward FLOPs per gemm; full net backward often several× forward wall time.
            loss.backward()
            if do_detailed:
                _cuda_sync()

        if do_detailed:
            t_backward = time.perf_counter() - t_mark
            t_mark = time.perf_counter()

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

        if do_detailed:
            _cuda_sync()
            t_clip = time.perf_counter() - t_mark
            t_mark = time.perf_counter()

        optimizer.step()

        if do_detailed:
            _cuda_sync()
            t_optim = time.perf_counter() - t_mark
            total_s = t_zero_grad + t_batch + t_forward + t_z + t_az + t_apply + t_loss + t_backward + t_clip + t_optim
            if n_pad_t is not None:
                print(f"  batch: B={B_step_t}, n_pad={n_pad_t}, batch_vectors={batch_vectors}")
            print(f"--- Step {DETAILED_TIMING_STEP} detailed timing (ms) ---")
            print(f"  zero_grad:        {t_zero_grad * 1000:8.2f}")
            print(f"  batch assembly:   {t_batch * 1000:8.2f}")
            print(f"  model forward:    {t_forward * 1000:8.2f}")
            print(f"  sample Z:         {t_z * 1000:8.2f}")
            print(f"  A @ Z (bmm):      {t_az * 1000:8.2f}")
            print(f"  apply M (MAZ):    {t_apply * 1000:8.2f}")
            print(f"  residual + loss:  {t_loss * 1000:8.2f}")
            print(f"  backward:         {t_backward * 1000:8.2f}")
            if profile_bw:
                print(
                    "     (profiler on: this wall time is dominated by Kineto, not GPU work — "
                    "omit --profile-backward to measure real backward; see table footer Self CUDA total.)"
                )
            print(f"  clip_grad_norm:   {t_clip * 1000:8.2f}")
            print(f"  optimizer.step:   {t_optim * 1000:8.2f}")
            print(f"  total:            {total_s * 1000:8.2f}")
            if wall_iter_start is not None:
                _cuda_sync()
                wall_ms = (time.perf_counter() - wall_iter_start) * 1000.0
                overhead_ms = wall_ms - total_s * 1000.0
                print(
                    f"  full iter (wall): {wall_ms:8.2f} ms  "
                    f"(Python / scheduler / print overhead ≈ {overhead_ms:+.2f} ms vs sum above)"
                )
            if prof is not None:
                print(
                    "  --- torch.profiler: backward() only (op mix / relative cost; "
                    "aggregated Self CUDA total can be << wall ms due to launch overlap + profiler CPU) ---"
                )
                print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=35))
                print("  --- same events, top by inclusive cuda_time_total ---")
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                _print_backward_cuda_bucket_summary(prof)
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
