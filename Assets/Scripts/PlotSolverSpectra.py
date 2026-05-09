#!/usr/bin/env python3
"""
Plot 1D spectral clustering for four solver variants on one test frame:
  1) Unpreconditioned CG operator (A)
  2) Jacobi-left-preconditioned operator (D^{-1}A)
  3) Baseline neural preconditioner operator (M_base A)
  4) SAI-trained neural preconditioner operator (M_sai A)

Condition number is printed below each subplot.

Default dataset path targets multiphase v2 8192 test split:
  data/multiphase_v2_8192/test
"""

import argparse
import sys
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigs, eigsh
from scipy.stats import gaussian_kde

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import leafonly.config as lo_cfg
from leafonly.architecture import LeafOnlyNet, apply_block_diagonal_M_physical, default_attention_layout
from leafonly.checkpoint import (
    apply_leaf_only_runtime_from_checkpoint,
    leaf_only_arch_from_checkpoint,
    load_leaf_only_weights,
)
from leafonly.data import FluidGraphDataset


_C_UNPRE = "#888888"
_C_JAC   = "#aaaaaa"
_C_BASE  = "#D55E00"
_C_SAI   = "#009E73"


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )


def _resolve_path(p: str | None, default: Path) -> Path:
    if p is None:
        return default.resolve()
    return Path(p).expanduser().resolve()


def _build_A_from_frame(dataset: FluidGraphDataset, frame_idx: int) -> tuple[csr_matrix, dict, int]:
    batch = dataset[frame_idx]
    num_nodes_real = int(batch["num_nodes"])
    n = int(lo_cfg.problem_padded_num_nodes(num_nodes_real))
    if n <= 0:
        raise ValueError(f"Frame {frame_idx} produced n={n}; check LEAF_SIZE/MAX_MIXED_SIZE vs dataset.")
    ei = batch["edge_index"]
    ev = batch["edge_values"]
    em = (ei[0] < n) & (ei[1] < n)
    rows = ei[0, em].cpu().numpy().astype(np.int64)
    cols = ei[1, em].cpu().numpy().astype(np.int64)
    vals = ev[em].cpu().numpy().astype(np.float64)
    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return A, batch, n


def _make_model_from_checkpoint(weights_path: Path, device: torch.device) -> tuple[LeafOnlyNet, dict]:
    meta = apply_leaf_only_runtime_from_checkpoint(weights_path)
    arch = leaf_only_arch_from_checkpoint(weights_path)
    if arch is None:
        raise FileNotFoundError(f"Could not parse checkpoint header: {weights_path}")
    model = LeafOnlyNet(
        input_dim=int(arch["input_dim"]),
        d_model=int(arch["d_model"]),
        leaf_size=int(arch["leaf_size"]),
        num_layers=int(arch["num_layers"]),
        num_heads=int(arch["num_heads"]),
        use_gcn=bool(int(arch["use_gcn"])),
        attention_layout=default_attention_layout(int(arch["leaf_size"])),
        off_diag_dense_attention=True,
        diag_dense_attention=True,
        use_highways=bool(int(arch.get("highway_ffn_mlp", 0))),
        ffn_concat_width=int(arch.get("ffn_concat_width", 4 if int(arch.get("highway_ffn_mlp", 0)) else 1)),
    ).to(device)
    load_leaf_only_weights(model, str(weights_path))
    model.eval()
    merged = dict(arch)
    if isinstance(meta, dict):
        merged.update(meta)
    return model, merged


def _prepare_frame_tensors(batch: dict, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"].unsqueeze(0).to(device).clone()
    x_in = x[:, :n, :].clone()
    pos = x_in[0, :, :3]
    x_in[0, :, :3] = pos - pos.mean(dim=0, keepdim=True)
    ei = batch["edge_index"]
    ev = batch["edge_values"]
    em = (ei[0] < n) & (ei[1] < n)
    edge_index = ei[:, em].to(device)
    edge_values = ev[em].to(device)
    gf = batch.get("global_features")
    if gf is None:
        raise ValueError("Frame missing global_features.")
    gf = gf.to(device)
    if gf.dim() == 1:
        gf = gf.unsqueeze(0)
    return x_in, edge_index, edge_values, gf


def _neural_M_apply_fn(
    model: LeafOnlyNet,
    x_in: torch.Tensor,
    edge_index: torch.Tensor,
    edge_values: torch.Tensor,
    global_features: torch.Tensor,
    jacobi_inv_diag_t: torch.Tensor,
    n: int,
    device: torch.device,
) -> Callable[[np.ndarray], np.ndarray]:
    with torch.inference_mode():
        packed = model(
            x_in,
            edge_index=edge_index,
            edge_values=edge_values,
            global_features=global_features,
        )
    la_d = int(model.leaf_apply_size)
    la_o = int(model.leaf_apply_off)
    leaf_size = int(lo_cfg.LEAF_SIZE)

    def apply(v_np: np.ndarray) -> np.ndarray:
        v_t = torch.from_numpy(v_np.astype(np.float32, copy=False)).to(device=device).view(1, n, 1)
        with torch.inference_mode():
            z_t = apply_block_diagonal_M_physical(
                packed,
                v_t,
                leaf_size=leaf_size,
                leaf_apply_size=la_d,
                leaf_apply_off=la_o,
                jacobi_inv_diag=jacobi_inv_diag_t,
            )
        return z_t.view(n).detach().cpu().numpy().astype(np.float64, copy=False)

    return apply


def _operator_and_spectrum(
    A: csr_matrix,
    M_apply: Callable[[np.ndarray], np.ndarray] | None,
    k_eigs: int,
) -> tuple[np.ndarray, float]:
    n = A.shape[0]
    k = max(8, min(k_eigs, n - 2))

    if M_apply is None:
        lin = LinearOperator((n, n), matvec=lambda x: A @ x, dtype=np.float64)
        # eigsh with "BE" (both ends) gives the k/2 smallest and k/2 largest eigenvalues
        vals = eigsh(lin, k=k, which="BE", return_eigenvectors=False, tol=1e-3)
        vals = np.sort(vals.real)
        pos = vals[vals > 1e-12]
        kappa = float(vals[-1] / pos[0]) if pos.size >= 2 else float("nan")
        return vals, kappa

    def matvec(x: np.ndarray) -> np.ndarray:
        return M_apply(A @ x)

    lin = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    try:
        vals = eigs(
            lin, k=k, which="LM", return_eigenvectors=False, tol=1e-3, maxiter=max(2000, 20 * n)
        ).real
    except Exception:
        return np.array([float("nan")]), float("nan")

    vals = np.sort(vals[np.isfinite(vals)])
    # Condition number from sampled eigenvalue range; underestimate when smallest
    # eigenvalues fall outside the sampled k largest, but reliable for clustered spectra.
    pos = np.abs(vals[np.abs(vals) > 1e-10])
    kappa = float(pos.max() / pos.min()) if pos.size >= 2 else float("nan")
    return vals, kappa


def _plot_kde(ax: plt.Axes, vals: np.ndarray, color: str, title: str, cond_num: float) -> None:
    vals = np.asarray(vals, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        ax.text(0.5, 0.5, "No finite eigenvalues", ha="center", va="center", transform=ax.transAxes)
        return

    span = float(finite.max() - finite.min())
    pad = max(span * 0.06, 1e-10)
    x_lo = finite.min() - pad
    x_hi = finite.max() + pad
    x_grid = np.linspace(x_lo, x_hi, 600)

    # Gaussian KDE with Silverman bandwidth
    try:
        kde = gaussian_kde(finite, bw_method="silverman")
        density = kde(x_grid)
    except Exception:
        # Degenerate data (all values identical) — draw a spike
        density = np.zeros_like(x_grid)
        density[len(x_grid) // 2] = 1.0

    # Normalize so peak = 1 for a clean relative-density read
    peak = density.max()
    if peak > 0:
        density = density / peak

    # Filled area + curve
    ax.fill_between(x_grid, density, alpha=0.22, color=color, lw=0, zorder=2)
    ax.plot(x_grid, density, color=color, lw=1.8, zorder=3)

    # Subtle rug marks along x-axis
    rug_y = np.full_like(finite, -0.04)
    ax.plot(finite, rug_y, "|", color=color, alpha=0.28, markersize=7, markeredgewidth=0.7, zorder=4, clip_on=False)

    ax.set_ylim(-0.10, 1.18)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.axhline(0.0, color="#bbbbbb", lw=0.6, zorder=1)
    ax.grid(True, axis="x", color="#d8d8d8", alpha=0.7, lw=0.5, zorder=0)
    ax.grid(True, axis="y", color="#efefef", alpha=0.7, lw=0.5, zorder=0)
    ax.set_xlim(x_lo, x_hi)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.set_ylabel("rel. density")
    ax.set_title(title, fontsize=10.5, fontweight="bold")

    cond_str = f"{cond_num:.3g}" if np.isfinite(cond_num) else "n/a"
    ax.text(
        0.5,
        -0.22,
        rf"$\kappa \approx {cond_str}$",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Spectral clustering comparison across solver preconditioners.")
    parser.add_argument("--data-folder", type=str, default="data/multiphase_v2_8192/test")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--baseline-weights",
        type=str,
        default=str(_SCRIPT_DIR / "weights" / "v2_8192_d128_L3_hw.bytes"),
    )
    parser.add_argument(
        "--sai-weights",
        type=str,
        default=str(_SCRIPT_DIR / "weights" / "v2_8192_d128_L3_hw_sai.bytes"),
    )
    parser.add_argument("--eig-count", type=int, default=140)
    parser.add_argument(
        "--output",
        type=str,
        default=str(_SCRIPT_DIR / "solver_spectral_clustering.png"),
    )
    args = parser.parse_args()

    _apply_style()

    data_folder = _resolve_path(args.data_folder, _SCRIPT_DIR / "data" / "multiphase_v2_8192" / "test")
    baseline_weights = _resolve_path(args.baseline_weights, _SCRIPT_DIR / "weights" / "v2_8192_d128_L3_hw.bytes")
    sai_weights = _resolve_path(args.sai_weights, _SCRIPT_DIR / "weights" / "v2_8192_d128_L3_hw_sai.bytes")
    out_path = _resolve_path(args.output, _SCRIPT_DIR / "solver_spectral_clustering.png")

    if not data_folder.exists():
        raise SystemExit(f"Data folder not found: {data_folder}")
    if not baseline_weights.is_file():
        raise SystemExit(f"Baseline checkpoint not found: {baseline_weights}")
    if not sai_weights.is_file():
        raise SystemExit(f"SAI checkpoint not found: {sai_weights}")

    base_arch = leaf_only_arch_from_checkpoint(baseline_weights)
    sai_arch = leaf_only_arch_from_checkpoint(sai_weights)
    if base_arch is None or sai_arch is None:
        raise SystemExit("Could not parse one or both checkpoint headers.")
    if int(base_arch["leaf_size"]) != int(sai_arch["leaf_size"]):
        raise SystemExit(
            f"Checkpoint leaf_size mismatch: baseline={base_arch['leaf_size']} sai={sai_arch['leaf_size']}."
        )

    apply_leaf_only_runtime_from_checkpoint(baseline_weights)
    dataset = FluidGraphDataset([Path(data_folder)])
    if len(dataset) == 0:
        raise SystemExit(f"No frames found under {data_folder}")
    frame_idx = max(0, min(int(args.frame), len(dataset) - 1))
    A, batch, n = _build_A_from_frame(dataset, frame_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_in, edge_index, edge_values, global_features = _prepare_frame_tensors(batch, n, device)

    diag = A.diagonal().astype(np.float64, copy=False)
    jacobi_inv_diag = np.ones(n, dtype=np.float64)
    mask = np.abs(diag) > 1e-12
    jacobi_inv_diag[mask] = 1.0 / diag[mask]
    jacobi_inv_diag_t = torch.from_numpy(jacobi_inv_diag.astype(np.float32)).to(device=device).view(1, n)

    print(f"[spectra] frame={frame_idx} n={n} device={device}")

    base_model, _ = _make_model_from_checkpoint(baseline_weights, device)
    M_base = _neural_M_apply_fn(
        base_model, x_in, edge_index, edge_values, global_features, jacobi_inv_diag_t, n, device,
    )

    sai_model, _ = _make_model_from_checkpoint(sai_weights, device)
    if int(lo_cfg.problem_padded_num_nodes(int(batch["num_nodes"]))) != n:
        A, batch, n = _build_A_from_frame(dataset, frame_idx)
        x_in, edge_index, edge_values, global_features = _prepare_frame_tensors(batch, n, device)
        diag = A.diagonal().astype(np.float64, copy=False)
        jacobi_inv_diag = np.ones(n, dtype=np.float64)
        mask = np.abs(diag) > 1e-12
        jacobi_inv_diag[mask] = 1.0 / diag[mask]
        jacobi_inv_diag_t = torch.from_numpy(jacobi_inv_diag.astype(np.float32)).to(device=device).view(1, n)
    M_sai = _neural_M_apply_fn(
        sai_model, x_in, edge_index, edge_values, global_features, jacobi_inv_diag_t, n, device,
    )

    def M_jac(v: np.ndarray) -> np.ndarray:
        return jacobi_inv_diag * v

    eig_unpre, cond_unpre = _operator_and_spectrum(A, None,    args.eig_count)
    eig_jac,   cond_jac   = _operator_and_spectrum(A, M_jac,  args.eig_count)
    eig_base,  cond_base  = _operator_and_spectrum(A, M_base, args.eig_count)
    eig_sai,   cond_sai   = _operator_and_spectrum(A, M_sai,  args.eig_count)

    fig, axs = plt.subplots(2, 2, figsize=(11.0, 7.2))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.14, wspace=0.32, hspace=0.52)

    _plot_kde(axs[0, 0], eig_unpre, _C_UNPRE, r"Unpreconditioned CG  ($A$)",              cond_unpre)
    _plot_kde(axs[0, 1], eig_jac,   _C_JAC,   r"Jacobi-Preconditioned  ($D^{-1}A$)",      cond_jac)
    _plot_kde(axs[1, 0], eig_base,  _C_BASE,  r"Baseline Neural  ($M_\mathrm{base}\,A$)", cond_base)
    _plot_kde(axs[1, 1], eig_sai,   _C_SAI,   r"SAI-Trained Neural  ($M_\mathrm{sai}\,A$)", cond_sai)

    for ax in axs.flat:
        ax.set_xlabel("eigenvalue (real part)")

    fig.suptitle(
        rf"Eigenvalue Spectra — solver preconditioners  (frame={frame_idx}, $n={n}$)",
        fontsize=11,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[spectra] wrote {out_path}")
    print(
        "[spectra] condition numbers: "
        f"unpre={cond_unpre:.3g}, jacobi={cond_jac:.3g}, baseline={cond_base:.3g}, sai={cond_sai:.3g}"
    )


if __name__ == "__main__":
    main()
