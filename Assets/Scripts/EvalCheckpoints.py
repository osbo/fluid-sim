"""Evaluate training checkpoints: append pcg_iters and sai_loss to the training CSV.

Run after LeafOnly.py --track-training to evaluate each saved *_step*.bytes checkpoint.
For each checkpoint, this script:
  - Loads weights into the LeafOnly model
  - Runs LeafOnly PCG on a fixed consistent problem → records iteration count
  - Computes the SAI loss (equation 8): mean_k ||(A/||A||_F) M^{-1} w_k - w_k||_2^2
    where w_k are fixed random probe vectors and M^{-1} is the learned preconditioner

PCG is run in float32 (vs InspectModel's float64 default); iteration counts are comparable
across checkpoints but may differ slightly from InspectModel --test-only output.

Usage:
    python EvalCheckpoints.py --csv leaf_only_weights_training_profile.csv
"""
import argparse
import csv
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*torch\._prims_common\.check.*", category=FutureWarning)

_DEFAULT_LEAF_SIZE = 128
_DEFAULT_MAX_MIXED_SIZE = 8192

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))


def _bootstrap_leafonly_config(leaf_size: int, max_mixed_size: int) -> None:
    """Apply leaf_size / max_mixed_size before leafonly imports (mirrors LeafOnly.py bootstrap)."""
    import importlib.util
    import types

    leafonly_dir = _scripts_dir / "leafonly"
    cfg_path = leafonly_dir / "config.py"
    pkg = "leafonly"
    if pkg not in sys.modules:
        m_pkg = types.ModuleType(pkg)
        m_pkg.__path__ = [str(leafonly_dir)]
        sys.modules[pkg] = m_pkg
    cfg_name = "leafonly.config"
    spec = importlib.util.spec_from_file_location(cfg_name, cfg_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load leafonly config from {cfg_path}")
    cfg_mod = importlib.util.module_from_spec(spec)
    sys.modules[cfg_name] = cfg_mod
    spec.loader.exec_module(cfg_mod)
    cfg_mod.apply_runtime_sizes(leaf_size, max_mixed_size)


def _step_from_filename(name: str):
    m = re.search(r"_step(\d+)\.bytes$", name)
    return int(m.group(1)) if m else None


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--leaf-size", type=int, default=_DEFAULT_LEAF_SIZE)
    pre.add_argument("--max-mixed-size", type=int, default=_DEFAULT_MAX_MIXED_SIZE)
    ns, _ = pre.parse_known_args()
    _bootstrap_leafonly_config(ns.leaf_size, ns.max_mixed_size)

    import numpy as np
    import torch
    import torch.nn.functional as F

    from leafonly import (
        LeafOnlyNet,
        apply_block_diagonal_m_into,
        block_diagonal_m_apply_workspace,
        build_leaf_block_connectivity,
        default_attention_layout,
        FluidGraphDataset,
        load_leaf_only_weights,
        warmup_hmatrix_prolong_gpu,
    )
    from leafonly.checkpoint import leaf_only_arch_from_checkpoint
    from leafonly.config import (
        LEAF_APPLY_SIZE,
        LEAF_APPLY_SIZE_OFF,
        LEAF_SIZE,
        problem_padded_num_nodes,
    )
    from leafonly.hmatrix import NUM_HMATRIX_OFF_BLOCKS

    parser = argparse.ArgumentParser(
        description="Evaluate training checkpoints: append pcg_iters and sai_loss to training CSV.",
    )
    parser.add_argument(
        "--csv",
        required=True,
        metavar="PATH",
        help="Training profile CSV produced by LeafOnly.py --track-training.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default=None,
        metavar="DIR",
        help="Directory containing *_step*.bytes checkpoint files. Default: same directory as --csv.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=600,
        metavar="N",
        help="Dataset frame index (consistent across all checkpoints). Default: 600.",
    )
    parser.add_argument(
        "--data-folder",
        default=None,
        metavar="DIR",
        help="Frame root (rglob nodes.bin). Default: StreamingAssets/TestData beside Assets/.",
    )
    parser.add_argument(
        "--num-probes",
        type=int,
        default=64,
        metavar="K",
        help="Number of Monte-Carlo SAI loss probe vectors. Default: 64.",
    )
    parser.add_argument(
        "--pcg-tol",
        type=float,
        default=1e-8,
        help="PCG convergence tolerance (same as InspectModel default). Default: 1e-8.",
    )
    parser.add_argument(
        "--pcg-max-iter",
        type=int,
        default=50000,
        help="PCG maximum iterations. Default: 50000.",
    )
    parser.add_argument("--leaf-size", type=int, default=ns.leaf_size)
    parser.add_argument("--max-mixed-size", type=int, default=ns.max_mixed_size)
    parser.add_argument(
        "--use-highways",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Must match the --use-highways flag used during training.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    ckpts_dir = (
        Path(args.checkpoints_dir).expanduser().resolve()
        if args.checkpoints_dir
        else csv_path.parent
    )
    ckpt_files = sorted(ckpts_dir.glob("*_step*.bytes"))
    if not ckpt_files:
        raise SystemExit(f"No *_step*.bytes checkpoint files found in {ckpts_dir}")

    step_to_file = {}
    for f in ckpt_files:
        s = _step_from_filename(f.name)
        if s is not None:
            step_to_file[s] = f
    if not step_to_file:
        raise SystemExit("Could not parse step numbers from checkpoint filenames.")

    sorted_steps = sorted(step_to_file)
    preview = sorted_steps[:5]
    suffix = "..." if len(sorted_steps) > 5 else ""
    print(f"Found {len(sorted_steps)} checkpoint(s): steps {preview}{suffix}")

    # --- Load CSV ---
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        old_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    new_cols = ["pcg_iters", "sai_loss"]
    fieldnames = old_fieldnames + [c for c in new_cols if c not in old_fieldnames]

    # --- Dataset ---
    if args.data_folder:
        data_folder = Path(args.data_folder).expanduser().resolve()
    else:
        data_folder = _scripts_dir.parent / "StreamingAssets" / "TestData"
    dataset = FluidGraphDataset([data_folder])
    if len(dataset) == 0:
        raise SystemExit(f"No frames found in {data_folder}")
    frame_idx = max(0, min(int(args.frame), len(dataset) - 1))
    batch = dataset[frame_idx]
    num_nodes_real = int(batch["num_nodes"])
    print(f"Frame {frame_idx}: N={num_nodes_real}")

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    leaf_L = int(LEAF_SIZE)
    leaf_apply_diag_L = int(LEAF_APPLY_SIZE)
    leaf_apply_off_L = int(LEAF_APPLY_SIZE_OFF)
    viz_n = problem_padded_num_nodes(num_nodes_real)
    num_leaves = viz_n // leaf_L
    print(f"Problem size: n={viz_n}, leaves={num_leaves}")

    # --- Build A (sparse, float32) ---
    ei, ev = batch["edge_index"], batch["edge_values"]
    em = (ei[0] < viz_n) & (ei[1] < viz_n)
    A_sparse = torch.sparse_coo_tensor(
        ei[:, em], ev[em], (viz_n, viz_n)
    ).coalesce().to(device=device, dtype=torch.float32)

    # Frobenius norm: sqrt(sum of squared nonzero values)
    a_frob_norm = float(ev[em].to(torch.float64).pow(2).sum().sqrt().item())
    print(f"||A||_F = {a_frob_norm:.6f}")

    # Jacobi diagonal inverse
    A_diag_np = torch.sparse_coo_tensor(
        ei[:, em], ev[em], (viz_n, viz_n)
    ).coalesce().to_dense().diagonal().numpy().astype(np.float32)
    jacobi_inv_diag = torch.ones(1, viz_n, device=device, dtype=torch.float32)
    diag_t = torch.from_numpy(A_diag_np).to(device)
    jacobi_mask = diag_t.abs() > 1e-6
    jacobi_inv_diag[0, jacobi_mask] = 1.0 / diag_t[jacobi_mask]
    jacobi_s_phys = jacobi_inv_diag[:, :viz_n].contiguous()

    # --- Build node features and connectivity (fixed across checkpoints) ---
    x_base = batch["x"].unsqueeze(0).to(device)
    n_feat = x_base.shape[1]
    if n_feat < viz_n:
        x_leaf = F.pad(x_base, (0, 0, 0, viz_n - n_feat), value=0.0)
    elif n_feat > viz_n:
        x_leaf = x_base[:, :viz_n, :]
    else:
        x_leaf = x_base

    edge_index_gpu = ei[:, em].to(device)
    edge_values_gpu = ev[em].to(device)
    positions = x_leaf[0, :, :3]

    pre_connectivity = build_leaf_block_connectivity(
        edge_index_gpu,
        edge_values_gpu,
        positions,
        leaf_L,
        device,
        x_leaf.dtype,
        off_diag_dense_attention=True,
        diag_dense_attention=True,
    )
    pre_connectivity = tuple(
        t.contiguous() if isinstance(t, torch.Tensor) else t for t in pre_connectivity
    )

    # --- Fixed PCG right-hand side (seeded, same as InspectModel) ---
    np.random.seed(123)
    b_np = np.random.randn(viz_n).astype(np.float32)
    b_np /= np.linalg.norm(b_np) + 1e-12
    b_gpu = torch.from_numpy(b_np).to(device=device, dtype=torch.float32).unsqueeze(-1)

    # --- Fixed SAI probe vectors (seeded) ---
    torch.manual_seed(777)
    W_sai = torch.randn(1, viz_n, args.num_probes, device=device, dtype=torch.float32)
    col_norms = W_sai.norm(dim=1, keepdim=True).clamp(min=1e-12)
    W_sai = W_sai / col_norms  # shape (1, viz_n, K)
    W_sai_sq = W_sai.squeeze(0)  # (viz_n, K)

    # --- Workspaces (reused across checkpoints) ---
    pcg_ws = block_diagonal_m_apply_workspace(
        num_leaves=num_leaves,
        leaf_size=leaf_L,
        K_dim=1,
        M_h=NUM_HMATRIX_OFF_BLOCKS,
        La_o=leaf_apply_off_L,
        device=device,
        dtype=torch.float32,
    )
    pcg_r_buf = torch.zeros(1, viz_n, 1, device=device, dtype=torch.float32)
    pcg_z_buf = torch.zeros(1, viz_n, 1, device=device, dtype=torch.float32)

    sai_ws = block_diagonal_m_apply_workspace(
        num_leaves=num_leaves,
        leaf_size=leaf_L,
        K_dim=args.num_probes,
        M_h=NUM_HMATRIX_OFF_BLOCKS,
        La_o=leaf_apply_off_L,
        device=device,
        dtype=torch.float32,
    )

    warmup_hmatrix_prolong_gpu(device)

    # --- Read arch from first checkpoint (all checkpoints share the same arch) ---
    first_ckpt = step_to_file[sorted_steps[0]]
    ckpt_arch = leaf_only_arch_from_checkpoint(first_ckpt)
    if ckpt_arch is None:
        raise SystemExit(f"Cannot read architecture from {first_ckpt}")
    d_model_lo = int(ckpt_arch["d_model"])
    num_layers_lo = int(ckpt_arch["num_layers"])
    num_heads_lo = int(ckpt_arch["num_heads"])
    use_gcn_lo = bool(ckpt_arch["use_gcn"])
    ck_hw = int(ckpt_arch.get("highway_ffn_mlp", 0))
    ck_ffn = int(ckpt_arch.get("ffn_concat_width", 3 if ck_hw else 1))
    use_highways = bool(args.use_highways)

    model = LeafOnlyNet(
        input_dim=9,
        d_model=d_model_lo,
        leaf_size=leaf_L,
        num_layers=num_layers_lo,
        num_heads=num_heads_lo,
        use_gcn=use_gcn_lo,
        attention_layout=default_attention_layout(leaf_L),
        off_diag_dense_attention=True,
        diag_dense_attention=True,
        use_highways=use_highways,
        ffn_concat_width=ck_ffn if ck_hw else None,
    ).to(device)
    model = torch.compile(model)

    results = {}  # step -> {"pcg_iters": int, "sai_loss": float}
    total = len(sorted_steps)

    for i, step in enumerate(sorted_steps):
        ckpt_path = step_to_file[step]
        print(f"\n[{i + 1}/{total}] step={step}  {ckpt_path.name}", flush=True)

        load_leaf_only_weights(model, str(ckpt_path))
        model.eval()

        with torch.inference_mode():
            # Warmup forward (primes torch.compile / inductor)
            for _ in range(3 if i == 0 else 1):
                precond_out = model(
                    x_leaf,
                    edge_index=edge_index_gpu,
                    edge_values=edge_values_gpu,
                    precomputed_leaf_connectivity=pre_connectivity,
                )
            precond_s = precond_out.detach().contiguous()

            # ---- PCG ----
            def _apply_m(r_col, z_col):
                pcg_r_buf.copy_(r_col.reshape(1, viz_n, 1))
                apply_block_diagonal_m_into(
                    precond_s,
                    pcg_r_buf,
                    pcg_z_buf,
                    jacobi_s_phys,
                    pcg_ws,
                    leaf_size=leaf_L,
                    leaf_apply_size=leaf_apply_diag_L,
                    leaf_apply_off=leaf_apply_off_L,
                )
                z_col.copy_(pcg_z_buf.reshape(viz_n, 1))
                return z_col

            x_pcg = torch.zeros(viz_n, 1, device=device, dtype=torch.float32)
            r = b_gpu - (A_sparse @ x_pcg.squeeze(-1)).unsqueeze(-1)
            z = torch.empty_like(r)
            _apply_m(r, z)
            p = z.clone()
            rho = (r * z).sum().to(torch.float64)
            b_norm_sq = (b_gpu * b_gpu).sum().to(torch.float64)
            tol_sq = args.pcg_tol ** 2 * b_norm_sq.item()
            iters = args.pcg_max_iter
            for k in range(args.pcg_max_iter):
                Ap = (A_sparse @ p.squeeze(-1)).unsqueeze(-1)
                pAp = (p * Ap).sum().to(torch.float64)
                alpha = (rho / pAp).to(torch.float32)
                x_pcg = x_pcg + alpha * p
                r = r - alpha * Ap
                _apply_m(r, z)
                rho_new = (r * z).sum().to(torch.float64)
                beta = (rho_new / rho).to(torch.float32)
                p = z + beta * p
                rho = rho_new
                if (k + 1) % 3 == 0:
                    r_sq = (r * r).sum().to(torch.float64)
                    if r_sq.item() <= tol_sq:
                        iters = k + 1
                        break
            print(f"  PCG iters: {iters}")

            # ---- SAI loss: mean_k ||(A/||A||_F) M^{-1} w_k - w_k||_2^2 ----
            Z_sai = torch.empty_like(W_sai)
            apply_block_diagonal_m_into(
                precond_s,
                W_sai,
                Z_sai,
                jacobi_s_phys,
                sai_ws,
                leaf_size=leaf_L,
                leaf_apply_size=leaf_apply_diag_L,
                leaf_apply_off=leaf_apply_off_L,
            )
            Z_sq = Z_sai.squeeze(0)  # (viz_n, K)
            AZ = A_sparse @ Z_sq     # (viz_n, K)
            diff = AZ / a_frob_norm - W_sai_sq  # (viz_n, K)
            sai_loss = float((diff * diff).sum(dim=0).mean().item())
            print(f"  SAI loss:  {sai_loss:.6f}")

        results[step] = {"pcg_iters": iters, "sai_loss": sai_loss}

    # --- Update CSV ---
    checkpoint_steps = set(results.keys())
    for row in rows:
        step = int(row["step"])
        if step in checkpoint_steps:
            row["pcg_iters"] = str(results[step]["pcg_iters"])
            row["sai_loss"] = f"{results[step]['sai_loss']:.6f}"
        else:
            row.setdefault("pcg_iters", "")
            row.setdefault("sai_loss", "")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nUpdated CSV written to: {csv_path}")
    print(f"Columns: {fieldnames}")


if __name__ == "__main__":
    main()
