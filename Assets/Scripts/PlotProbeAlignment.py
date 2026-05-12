"""
Visualize aligned probe vectors (early vs late checkpoints) in shared 3D PCA space.

Workflow:
  1) Load one frame and sparse operator A.
  2) Draw fixed probe vectors Z (seeded).
  3) For each checkpoint, compute MAZ = M(AZ) using the same learned preconditioner apply path.
  4) Per probe i, build a Householder alignment that maps z_i / ||z_i|| to a fixed reference e.
     Apply that transform to MAz_i to get aligned vectors.
  5) Fit PCA basis on the early aligned cloud only; project both clouds with shared axis limits.
  6) Plot side-by-side 3D arrow tips (optionally arrows from origin).

Usage example:
  python PlotProbeAlignment.py \
    --csv /path/to/v2_4096_d128_L3_hw_training_profile.csv \
    --num-probes 200 \
    --output /path/to/probe_alignment_4096.png
"""

import argparse
import csv
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*torch\._prims_common\.check.*", category=FutureWarning)

_SCALE_LEAF_SIZE = {1024: 128, 2048: 128, 4096: 128, 8192: 128, 16384: 256}
_DEFAULT_LEAF_SIZE = 128
_DEFAULT_MAX_MIXED_SIZE = 8192

_scripts_dir = Path(__file__).resolve().parent
_repo_root = _scripts_dir.parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))


def _bootstrap_leafonly_config(leaf_size: int, max_mixed_size: int) -> None:
    """Apply leaf_size / max_mixed_size before leafonly imports."""
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


def _infer_scale_from_name(name: str):
    m = re.search(r"v2_(\d+)_", Path(name).name)
    return int(m.group(1)) if m else None


def _step_from_filename(name: str):
    m = re.search(r"_step(\d+)\.bytes$", name)
    return int(m.group(1)) if m else None


def _householder_align_apply(z_hat, y, ref, eps=1e-12):
    """
    Apply a numerically stable Householder-based alignment transform to y so that z_hat maps to ref.
    Returns transformed y.
    """
    import numpy as np

    dot = float(np.dot(z_hat, ref))
    if dot > 1.0 - 1e-7:
        return y.copy()

    if dot < -1.0 + 1e-7:
        # Anti-parallel corner case: reflect with n=(z+ref) would be unstable (near zero).
        # Reflect with n=(z-ref) maps z -> -ref, then flip sign to map z -> ref.
        n = z_hat - ref
        nn = float(np.linalg.norm(n))
        if nn < eps:
            return -y
        n = n / nn
        hy = y - 2.0 * n * float(np.dot(n, y))
        return -hy

    n = z_hat - ref
    nn = float(np.linalg.norm(n))
    if nn < eps:
        return y.copy()
    n = n / nn
    return y - 2.0 * n * float(np.dot(n, y))


def _align_maz_columns(Z, MAZ, length_mode: str):
    """Align each MAz column to a fixed reference direction via per-probe Householder transforms."""
    import numpy as np

    n, k = Z.shape
    ref = np.zeros((n,), dtype=np.float32)
    ref[0] = 1.0
    aligned = np.zeros_like(MAZ, dtype=np.float32)

    z_norms = np.linalg.norm(Z, axis=0) + 1e-12
    m_norms = np.linalg.norm(MAZ, axis=0) + 1e-12
    for i in range(k):
        z_hat = (Z[:, i] / z_norms[i]).astype(np.float32, copy=False)
        y = MAZ[:, i].astype(np.float32, copy=False)

        if length_mode == "unit":
            y = y / m_norms[i]
        elif length_mode == "match_probe_norm":
            y = y * (z_norms[i] / m_norms[i])

        aligned[:, i] = _householder_align_apply(z_hat, y, ref)
    return aligned, ref


def _fit_pca_basis_early(early_points):
    """Fit PCA basis from early cloud only. Input shape: (K, N)."""
    import numpy as np

    mean = early_points.mean(axis=0, keepdims=True)
    x = early_points - mean
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    basis = vt[:3].T
    return basis, mean


def _project(points, basis, mean):
    return (points - mean) @ basis


def _build_parser(_auto_data):
    parser = argparse.ArgumentParser(description="Plot aligned MAz probe clouds (early vs late) in shared 3D PCA.")
    parser.add_argument(
        "--csv",
        required=False,
        metavar="PATH",
        help=(
            "Training profile CSV path; used to infer checkpoint prefix and default data folder. "
            "Optional if --checkpoints-dir and --model-prefix are provided."
        ),
    )
    parser.add_argument(
        "--checkpoints-dir",
        default=None,
        metavar="DIR",
        help="Directory containing checkpoint files. Default: same directory as --csv.",
    )
    parser.add_argument(
        "--model-prefix",
        default=None,
        metavar="NAME",
        help="Checkpoint filename prefix before _stepXXXXXX.bytes. Default: inferred from CSV stem.",
    )
    parser.add_argument(
        "--early-step",
        type=int,
        default=None,
        metavar="N",
        help="Explicit early step. Default: in-progress checkpoint chosen by --early-fraction.",
    )
    parser.add_argument(
        "--late-step",
        type=int,
        default=None,
        metavar="N",
        help="Explicit late step. Default: latest available step.",
    )
    parser.add_argument(
        "--early-fraction",
        type=float,
        default=0.4,
        metavar="F",
        help=(
            "When --early-step is omitted, choose an in-progress checkpoint at fraction F of training "
            "(clamped away from first/last). Default: 0.4."
        ),
    )
    parser.add_argument("--frame", type=int, default=0, metavar="N", help="Dataset frame index. Default: 0.")
    parser.add_argument(
        "--data-folder",
        default=_auto_data,
        metavar="DIR",
        help=f"Dataset root (rglob nodes.bin). Auto from CSV scale: {_auto_data or '(unknown; pass explicitly)'}",
    )
    parser.add_argument("--num-probes", type=int, default=200, metavar="K", help="Number of probes. Default: 200.")
    parser.add_argument("--probe-seed", type=int, default=777, metavar="S", help="Probe RNG seed. Default: 777.")
    parser.add_argument(
        "--length-mode",
        type=str,
        choices=("keep", "unit", "match_probe_norm"),
        default="keep",
        help=(
            "How to scale MAz vectors before alignment: "
            "keep (preserve magnitude), unit (direction-only), "
            "match_probe_norm (scale each MAz to ||z||)."
        ),
    )
    parser.add_argument(
        "--center-on-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Subtract reference direction e before PCA. Default: true.",
    )
    parser.add_argument(
        "--draw-arrows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw quiver arrows from origin to tips (can clutter if K is large). Default: false.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Aligned 200 MAz probes in shared PCA basis",
        help="Figure title.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Output PNG path. Default: Paper/figures/probe_alignment_<scale>.png",
    )
    parser.add_argument(
        "--dump-npz",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional NPZ output with raw/aligned/projected clouds and norm stats.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI. Default: 220.")
    parser.add_argument("--leaf-size", type=int, default=None)
    parser.add_argument("--max-mixed-size", type=int, default=None)
    return parser


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--csv", default=None)
    pre.add_argument("--leaf-size", type=int, default=None)
    pre.add_argument("--max-mixed-size", type=int, default=None)
    ns, _ = pre.parse_known_args()

    inferred_scale = _infer_scale_from_name(ns.csv) if ns.csv else None
    auto_leaf = _SCALE_LEAF_SIZE.get(inferred_scale, _DEFAULT_LEAF_SIZE)
    auto_max = inferred_scale if inferred_scale else _DEFAULT_MAX_MIXED_SIZE
    leaf_size = ns.leaf_size if ns.leaf_size is not None else auto_leaf
    max_mixed = ns.max_mixed_size if ns.max_mixed_size is not None else auto_max
    _bootstrap_leafonly_config(leaf_size, max_mixed)

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    from leafonly.architecture import (
        LeafOnlyNet,
        apply_block_diagonal_m_into,
        block_diagonal_m_apply_workspace,
        default_attention_layout,
        warmup_hmatrix_prolong_gpu,
    )
    from leafonly.checkpoint import leaf_only_arch_from_checkpoint, load_leaf_only_weights
    from leafonly.config import LEAF_APPLY_SIZE, LEAF_APPLY_SIZE_OFF, LEAF_SIZE, problem_padded_num_nodes
    from leafonly.data import FluidGraphDataset, build_leaf_block_connectivity
    from leafonly.hmatrix import NUM_HMATRIX_OFF_BLOCKS

    auto_data = (
        str(_scripts_dir / "data" / f"multiphase_v2_{inferred_scale}" / "test")
        if inferred_scale
        else None
    )
    parser = _build_parser(auto_data)
    args = parser.parse_args()
    csv_path = None
    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
        if not csv_path.exists():
            raise SystemExit(
                f"CSV not found: {csv_path}\n"
                "Tip: either pass a real CSV path, or run without --csv and pass both "
                "--checkpoints-dir and --model-prefix."
            )

    if args.checkpoints_dir:
        ckpt_dir = Path(args.checkpoints_dir).expanduser().resolve()
    elif csv_path is not None:
        ckpt_dir = csv_path.parent
    else:
        raise SystemExit("Missing checkpoints location. Pass --checkpoints-dir or --csv.")

    if args.model_prefix:
        model_prefix = args.model_prefix
    elif csv_path is not None:
        model_prefix = csv_path.stem.replace("_training_profile", "")
    else:
        raise SystemExit("Missing model prefix. Pass --model-prefix when --csv is omitted.")

    ckpt_files = sorted(ckpt_dir.glob(f"{model_prefix}_step*.bytes"))
    if not ckpt_files:
        raise SystemExit(f"No checkpoints matching '{model_prefix}_step*.bytes' in {ckpt_dir}")
    step_to_file = {}
    for f in ckpt_files:
        s = _step_from_filename(f.name)
        if s is not None:
            step_to_file[s] = f
    if not step_to_file:
        raise SystemExit("Could not parse checkpoint steps.")
    steps = sorted(step_to_file)

    def _default_early_step(all_steps, frac):
        if len(all_steps) <= 2:
            return all_steps[0]
        frac = max(0.0, min(1.0, float(frac)))
        idx = int(round(frac * (len(all_steps) - 1)))
        idx = max(1, min(len(all_steps) - 2, idx))
        return all_steps[idx]

    early_step = args.early_step if args.early_step is not None else _default_early_step(steps, args.early_fraction)
    late_step = args.late_step if args.late_step is not None else steps[-1]
    if early_step not in step_to_file:
        raise SystemExit(f"--early-step {early_step} not found. Available: {steps[:8]}{'...' if len(steps) > 8 else ''}")
    if late_step not in step_to_file:
        raise SystemExit(f"--late-step {late_step} not found. Available: {steps[:8]}{'...' if len(steps) > 8 else ''}")
    if early_step == late_step:
        raise SystemExit("Early and late steps are identical. Pick two distinct checkpoints.")

    pcg_iters_by_step = {}
    if csv_path is not None:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames or [])
            if "step" in fields and "pcg_iters" in fields:
                for row in reader:
                    s_raw = str(row.get("step", "")).strip()
                    p_raw = str(row.get("pcg_iters", "")).strip()
                    if not s_raw or not p_raw:
                        continue
                    try:
                        pcg_iters_by_step[int(s_raw)] = int(float(p_raw))
                    except ValueError:
                        continue

    if args.data_folder:
        _df = Path(args.data_folder).expanduser()
        data_folder = _df.resolve() if _df.is_absolute() else (_repo_root / _df).resolve()
    else:
        raise SystemExit(
            "Could not infer data folder. Pass --data-folder data/multiphase_v2_<scale>/test."
        )

    dataset = FluidGraphDataset([data_folder])
    if len(dataset) == 0:
        raise SystemExit(f"No frames found in {data_folder}")
    frame_idx = max(0, min(int(args.frame), len(dataset) - 1))
    batch = dataset[frame_idx]
    num_nodes_real = int(batch["num_nodes"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    leaf_L = int(LEAF_SIZE)
    leaf_apply_diag_L = int(LEAF_APPLY_SIZE)
    leaf_apply_off_L = int(LEAF_APPLY_SIZE_OFF)
    viz_n = problem_padded_num_nodes(num_nodes_real)
    num_leaves = viz_n // leaf_L

    print(f"Device: {device}")
    print(f"Frame {frame_idx}: n_real={num_nodes_real}, n_pad={viz_n}, leaves={num_leaves}")
    print(f"Using steps early={early_step}, late={late_step}")

    ei, ev = batch["edge_index"], batch["edge_values"]
    em = (ei[0] < viz_n) & (ei[1] < viz_n)

    A_sparse_f32 = torch.sparse_coo_tensor(ei[:, em], ev[em], (viz_n, viz_n)).coalesce().to(device=device, dtype=torch.float32)

    A_diag_np = torch.sparse_coo_tensor(ei[:, em], ev[em], (viz_n, viz_n)).coalesce().to_dense().diagonal().numpy().astype(np.float32)
    jacobi_inv_diag = torch.ones(1, viz_n, device=device, dtype=torch.float32)
    diag_t = torch.from_numpy(A_diag_np).to(device)
    jacobi_mask = diag_t.abs() > 1e-6
    jacobi_inv_diag[0, jacobi_mask] = 1.0 / diag_t[jacobi_mask]
    jacobi_s_phys = jacobi_inv_diag[:, :viz_n].contiguous()

    x_base = batch["x"].unsqueeze(0).to(device)
    n_feat = x_base.shape[1]
    if n_feat < viz_n:
        x_leaf = F.pad(x_base, (0, 0, 0, viz_n - n_feat), value=0.0)
    elif n_feat > viz_n:
        x_leaf = x_base[:, :viz_n, :]
    else:
        x_leaf = x_base

    global_feat = batch.get("global_features")
    if global_feat is not None:
        global_feat = global_feat.to(device)
        if global_feat.dim() == 1:
            global_feat = global_feat.unsqueeze(0)

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
    pre_connectivity = tuple(t.contiguous() if isinstance(t, torch.Tensor) else t for t in pre_connectivity)

    torch.manual_seed(int(args.probe_seed))
    Z = torch.randn(viz_n, int(args.num_probes), device=device, dtype=torch.float32)
    z_col_norms = Z.norm(dim=0, keepdim=True).clamp(min=1e-12)
    Z = Z / z_col_norms
    AZ = A_sparse_f32 @ Z
    Z_np = Z.detach().cpu().numpy()

    ws = block_diagonal_m_apply_workspace(
        num_leaves=num_leaves,
        leaf_size=leaf_L,
        K_dim=int(args.num_probes),
        M_h=NUM_HMATRIX_OFF_BLOCKS,
        La_o=leaf_apply_off_L,
        device=device,
        dtype=torch.float32,
    )
    az_buf = AZ.unsqueeze(0).contiguous()
    maz_buf = torch.empty_like(az_buf)
    warmup_hmatrix_prolong_gpu(device)

    def _compute_maz_for_checkpoint(ckpt_path: Path):
        ckpt_arch = leaf_only_arch_from_checkpoint(ckpt_path)
        if ckpt_arch is None:
            raise RuntimeError(f"Cannot read architecture from {ckpt_path}")

        d_model_lo = int(ckpt_arch["d_model"])
        num_layers_lo = int(ckpt_arch["num_layers"])
        num_heads_lo = int(ckpt_arch["num_heads"])
        use_gcn_lo = bool(ckpt_arch["use_gcn"])
        ck_hw = int(ckpt_arch.get("highway_ffn_mlp", 0))
        ck_ffn = int(ckpt_arch.get("ffn_concat_width", 3 if ck_hw else 1))
        use_highways = bool(ck_hw)

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
        load_leaf_only_weights(model, str(ckpt_path))
        model.eval()

        with torch.inference_mode():
            precond = model(
                x_leaf,
                edge_index=edge_index_gpu,
                edge_values=edge_values_gpu,
                global_features=global_feat,
                precomputed_leaf_connectivity=pre_connectivity,
            ).detach().contiguous()

            apply_block_diagonal_m_into(
                precond,
                az_buf,
                maz_buf,
                jacobi_s_phys,
                ws,
                leaf_size=leaf_L,
                leaf_apply_size=leaf_apply_diag_L,
                leaf_apply_off=leaf_apply_off_L,
            )
            return maz_buf.squeeze(0).detach().cpu().numpy()

    early_ckpt = step_to_file[early_step]
    late_ckpt = step_to_file[late_step]
    print(f"Loading early checkpoint: {early_ckpt.name}")
    MAZ_early = _compute_maz_for_checkpoint(early_ckpt)
    print(f"Loading late checkpoint:  {late_ckpt.name}")
    MAZ_late = _compute_maz_for_checkpoint(late_ckpt)

    aligned_early, ref = _align_maz_columns(Z_np, MAZ_early, args.length_mode)
    aligned_late, _ = _align_maz_columns(Z_np, MAZ_late, args.length_mode)

    pts_early = aligned_early.T
    pts_late = aligned_late.T
    ref_pt = ref.reshape(1, -1)
    if args.center_on_reference:
        pts_early = pts_early - ref_pt
        pts_late = pts_late - ref_pt

    basis, mean_early = _fit_pca_basis_early(pts_early)
    proj_early = _project(pts_early, basis, mean_early)
    proj_late = _project(pts_late, basis, mean_early)

    mins = proj_early.min(axis=0)
    maxs = proj_early.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.08 * span
    lo = mins - pad
    hi = maxs + pad

    if args.output:
        _out = Path(args.output).expanduser()
        out_path = _out.resolve() if _out.is_absolute() else (_repo_root / _out).resolve()
    else:
        out_path = (_repo_root / "Paper" / "figures" / f"probe_alignment_{inferred_scale or 'unknown'}.png").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    all_proj = np.vstack([proj_early, proj_late])
    all_r = np.linalg.norm(all_proj, axis=1)
    r_min = float(all_r.min())
    r_max = float(all_r.max())
    if r_max <= r_min + 1e-12:
        r_max = r_min + 1e-12
    norm = plt.Normalize(vmin=r_min, vmax=r_max)
    cmap = plt.get_cmap("magma")

    def _plot_cloud(ax, proj, ttl):
        r = np.linalg.norm(proj, axis=1)
        colors = cmap(norm(r))
        ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], s=12, alpha=0.9, c=colors)
        if args.draw_arrows:
            zeros = np.zeros((proj.shape[0],))
            ax.quiver(
                zeros,
                zeros,
                zeros,
                proj[:, 0],
                proj[:, 1],
                proj[:, 2],
                length=1.0,
                normalize=False,
                linewidths=0.4,
                alpha=0.25,
                color=colors,
            )
        ax.set_title(ttl)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])

    def _label_with_pcg(tag, step):
        pit = pcg_iters_by_step.get(step)
        if pit is None:
            return f"{tag} (step {step})"
        return f"{tag} (step {step}, PCG={pit})"

    _plot_cloud(ax1, proj_early, _label_with_pcg("Early M", early_step))
    _plot_cloud(ax2, proj_late, _label_with_pcg("Late M", late_step))
    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(args.dpi))
    print(f"Saved figure: {out_path}")

    if args.dump_npz:
        _dump = Path(args.dump_npz).expanduser()
        dump_npz = _dump.resolve() if _dump.is_absolute() else (_repo_root / _dump).resolve()
    else:
        dump_npz = out_path.with_suffix(".npz")
    np.savez_compressed(
        dump_npz,
        steps=np.array([early_step, late_step], dtype=np.int64),
        z=Z_np,
        maz_early=MAZ_early,
        maz_late=MAZ_late,
        aligned_early=aligned_early,
        aligned_late=aligned_late,
        pca_basis=basis,
        pca_mean=mean_early,
        proj_early=proj_early,
        proj_late=proj_late,
        norms_maz_early=np.linalg.norm(MAZ_early, axis=0),
        norms_maz_late=np.linalg.norm(MAZ_late, axis=0),
        norms_z=np.linalg.norm(Z_np, axis=0),
        length_mode=np.array(args.length_mode),
    )
    print(f"Saved data dump: {dump_npz}")


if __name__ == "__main__":
    main()
