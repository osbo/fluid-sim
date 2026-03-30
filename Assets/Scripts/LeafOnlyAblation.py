import argparse
import csv
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from leafonly import (
    LeafOnlyNet,
    apply_block_diagonal_M,
    load_leaf_only_weights,
)
from leafonly.config import (
    LEAF_APPLY_SIZE,
    LEAF_APPLY_SIZE_OFF,
    LEAF_SIZE,
    MAX_MIXED_SIZE,
    fixed_runtime_config,
    require_cuda_or_mps_device,
)
from leafonly.data import FluidGraphDataset, most_recent_run_folder
from leafonly.train import train_leaf_only


def _sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _avg_ms(fn, device, warmup=10, repeat=50):
    with torch.no_grad():
        for _ in range(max(0, int(warmup))):
            fn()
        _sync_device(device)
        t0 = time.perf_counter()
        for _ in range(max(1, int(repeat))):
            fn()
        _sync_device(device)
        total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms / max(1, int(repeat))


def _measure_inference_ms(save_path, cfg, runtime, frame_idx=600):
    device = runtime["device"]
    data_folder = runtime["data_folder"]
    data_path = Path(data_folder)
    run_folder = most_recent_run_folder(data_path)
    dataset = FluidGraphDataset([run_folder])
    if len(dataset) == 0:
        raise SystemExit(f"No frames found under {run_folder}")

    frame_idx = min(int(frame_idx), len(dataset) - 1)
    batch = dataset[frame_idx]
    num_nodes_real = int(batch["num_nodes"])
    n_requested = min(MAX_MIXED_SIZE, num_nodes_real)
    n_pad = MAX_MIXED_SIZE

    x = batch["x"].unsqueeze(0).to(device)
    x_input = x[:, :n_requested, :].clone()
    x_input = F.pad(x_input, (0, 0, 0, n_pad - n_requested), value=0.0)
    active_pos = x_input[0, :n_requested, :3]
    x_input[0, :n_requested, :3] = active_pos - active_pos.mean(dim=0, keepdim=True)

    ei, ev = batch["edge_index"], batch["edge_values"]
    em = (ei[0] < n_requested) & (ei[1] < n_requested)
    edge_index = ei[:, em].to(device)
    edge_values = ev[em].to(device)

    global_feat = batch.get("global_features")
    if global_feat is None:
        raise ValueError(f"Missing global_features for frame: {batch.get('frame_path', '<unknown>')}")
    global_feat = global_feat.to(device)
    if global_feat.dim() == 1:
        global_feat = global_feat.unsqueeze(0)

    model = LeafOnlyNet(
        input_dim=9,
        d_model=cfg.d_model,
        leaf_size=LEAF_SIZE,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        attention_layout=cfg.attention_layout,
        use_gcn=cfg.num_gcn_layers > 0,
        num_gcn_layers=cfg.num_gcn_layers,
        use_jacobi=cfg.use_jacobi,
        off_diag_dense_attention=bool(getattr(cfg, "off_diag_dense_attention", True)),
        diag_dense_attention=bool(getattr(cfg, "diag_dense_attention", True)),
    ).to(device)
    load_leaf_only_weights(model, str(save_path))
    model.eval()

    if device.type == "cuda":
        model = torch.compile(model, dynamic=False)

    batch_vectors = max(1024, int(round(n_pad ** 0.5)))
    Z = torch.randn(1, n_pad, batch_vectors, device=device, dtype=x_input.dtype)
    Z[:, n_requested:, :] = 0.0
    A_small = torch.sparse_coo_tensor(ei[:, em], ev[em], (n_requested, n_requested)).coalesce().to(device).to_dense()
    A_dense = torch.zeros(n_pad, n_pad, device=device, dtype=A_small.dtype)
    A_dense[:n_requested, :n_requested] = A_small
    if n_pad > n_requested:
        A_dense[n_requested:, n_requested:] = torch.eye(n_pad - n_requested, device=device, dtype=A_small.dtype)
    AZ = (A_dense @ Z.squeeze(0)).unsqueeze(0)
    jacobi_inv_diag = torch.ones(1, n_pad, device=device, dtype=A_dense.dtype)
    diag_A = torch.diagonal(A_dense, 0)
    inv_mask = diag_A.abs() > 1e-6
    jacobi_inv_diag[0, inv_mask] = 1.0 / diag_A[inv_mask]

    with torch.no_grad():
        precond_out = model(
            x_input,
            edge_index=edge_index,
            edge_values=edge_values,
            global_features=global_feat,
        )

    forward_ms = _avg_ms(
        lambda: model(
            x_input,
            edge_index=edge_index,
            edge_values=edge_values,
            global_features=global_feat,
        ),
        device,
        warmup=10,
        repeat=10,
    )

    apply_ms = _avg_ms(
        lambda: apply_block_diagonal_M(
            precond_out,
            AZ,
            leaf_size=LEAF_SIZE,
            leaf_apply_size=LEAF_APPLY_SIZE,
            leaf_apply_off=LEAF_APPLY_SIZE_OFF,
            jacobi_inv_diag=jacobi_inv_diag,
        ),
        device,
        warmup=5,
        repeat=40,
    )
    return forward_ms, apply_ms


def _build_train_args(base_args, d_model, num_layers, num_gcn_layers, use_jacobi, attention_layout):
    return SimpleNamespace(
        steps=base_args.target_step + 1,
        lr=base_args.lr,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=base_args.num_heads,
        frame=base_args.frame,
        use_single_frame=base_args.use_single_frame,
        num_frames=base_args.num_frames,
        seed=base_args.seed,
        contexts_per_step=base_args.contexts_per_step,
        continue_training=False,
        evaluate_gradients=False,
        num_gcn_layers=num_gcn_layers,
        use_jacobi=use_jacobi,
        attention_layout=attention_layout,
        target_step=base_args.target_step,
    )


def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--target_step", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--frame", type=int, default=600)
    p.add_argument("--use_single_frame", action="store_true")
    p.add_argument("--num_frames", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--contexts_per_step", type=int, default=4)
    p.add_argument("--output_csv", type=str, default="leafonly_ablation_results.csv")
    p.add_argument("--weights_dir", type=str, default="leafonly_ablation_weights")
    p.add_argument("--limit", type=int, default=0, help="Optional: run only first N configs (0 = all).")
    return p


def main():
    args = _build_parser().parse_args()
    runtime = fixed_runtime_config(__file__)
    runtime["device"] = require_cuda_or_mps_device()
    runtime["use_gcn"] = True

    L = LEAF_SIZE
    attention_variants = [
        ("baseline", f"{L}x{L + 1}"),
        ("only_LxL", f"{L}x{L}"),
        ("with_matrix_global_LxL2", f"{L}x{L + 2}"),
    ]
    baseline_num_layers = 2
    baseline_num_gcn_layers = 2
    baseline_use_jacobi = True
    baseline_d_model = args.d_model
    grid = attention_variants
    if args.limit > 0:
        grid = grid[: args.limit]

    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(grid)} ablation configs on device={runtime['device']}")
    print(f"Stopping metric at step={args.target_step}")

    rows = []
    for i, (variant_name, attention_layout) in enumerate(grid, start=1):
        d_model = baseline_d_model
        num_layers = baseline_num_layers
        num_gcn_layers = baseline_num_gcn_layers
        use_jacobi = baseline_use_jacobi
        tag = (
            f"d{d_model}"
            "_"
            f"L{num_layers}"
            f"_gcn{num_gcn_layers}"
            f"_jac{int(use_jacobi)}"
            f"_{attention_layout}"
        )
        save_path = weights_dir / f"{tag}.bytes"
        runtime_run = dict(runtime)
        runtime_run["save_path"] = save_path

        cfg = _build_train_args(
            args,
            d_model,
            num_layers,
            num_gcn_layers,
            use_jacobi,
            attention_layout,
        )
        print(f"\n[{i}/{len(grid)}] {tag}")
        metrics = train_leaf_only(cfg, runtime_run)
        fwd_ms, apply_ms = _measure_inference_ms(save_path, cfg, runtime_run, frame_idx=args.frame)
        total_ms = fwd_ms + apply_ms

        row = {
            "tag": tag,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_gcn_layers": num_gcn_layers,
            "use_jacobi": int(use_jacobi),
            "variant": variant_name,
            "attention_layout": attention_layout,
            "target_step": metrics.get("target_step"),
            "loss_mean_at_target": metrics.get("loss_mean_at_target"),
            "loss_std_at_target": metrics.get("loss_std_at_target"),
            "lr_at_target": metrics.get("lr_at_target"),
            "forward_inference_ms": fwd_ms,
            "apply_M_ms": apply_ms,
            "total_inference_plus_apply_ms": total_ms,
            "weights_path": str(save_path),
        }
        rows.append(row)
        print(
            "  Result:"
            f" loss@{args.target_step}={row['loss_mean_at_target']} ± {row['loss_std_at_target']},"
            f" infer={fwd_ms:.3f}ms, apply={apply_ms:.3f}ms"
        )

        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSaved ablation results to {output_csv}")


if __name__ == "__main__":
    main()

