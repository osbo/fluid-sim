#!/usr/bin/env python3
"""
Plot 1D spectral clustering for four solver variants on one test frame:
  1) Unpreconditioned CG operator (A)
  2) Jacobi-left-preconditioned operator (D^{-1}A)
  3) Cosine-Hutchinson neural preconditioner operator (M_cos A)
  4) SAI neural preconditioner operator (M_sai A)

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


def _sample_eigenvalues_symmetric(A: csr_matrix, k: int) -> np.ndarray:
    """Sample eigenvalues from both spectral ends of a symmetric SPD matrix."""
    n = A.shape[0]
    k = max(8, min(k, n - 2))
    lin = LinearOperator((n, n), matvec=lambda x: A @ x, dtype=np.float64)
    vals = eigsh(lin, k=k, which="BE", return_eigenvectors=False, tol=1e-4)
    return np.sort(vals.real)


def _sample_eigenvalues_jacobi(A: csr_matrix, inv_diag: np.ndarray, k: int) -> np.ndarray:
    """Sample eigenvalues of D^{-1}A via the symmetric similarity D^{-1/2} A D^{-1/2}."""
    n = A.shape[0]
    k = max(8, min(k, n - 2))
    d_isqrt = np.sqrt(np.clip(inv_diag.astype(np.float64, copy=False), 0.0, None))
    lin = LinearOperator((n, n), matvec=lambda x: d_isqrt * (A @ (d_isqrt * x)), dtype=np.float64)
    vals = eigsh(lin, k=k, which="BE", return_eigenvectors=False, tol=1e-4)
    return np.sort(vals.real)


def _sample_eigenvalues_nonsym(
    A: csr_matrix,
    M_apply: Callable[[np.ndarray], np.ndarray],
    k: int,
) -> np.ndarray:
    """Sample complex eigenvalues of M·A from both real ends (LR and SR)."""
    n = A.shape[0]
    k_each = max(4, min(k // 2, n - 2))

    def matvec(x: np.ndarray) -> np.ndarray:
        return M_apply(A @ x)

    lin = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    parts: list[np.ndarray] = []
    for which in ("LR", "SR"):
        try:
            v = eigs(lin, k=k_each, which=which, return_eigenvectors=False,
                     tol=1e-3, maxiter=max(3000, 30 * n))
            finite = v[np.isfinite(v.real) & np.isfinite(v.imag)]
            if finite.size > 0:
                parts.append(finite)
        except Exception:
            continue
    if not parts:
        try:
            v = eigs(lin, k=k_each, which="LM", return_eigenvectors=False, tol=1e-3)
            finite = v[np.isfinite(v.real) & np.isfinite(v.imag)]
            if finite.size > 0:
                parts = [finite]
        except Exception:
            return np.array([float("nan") + 0j])
    return np.concatenate(parts)


def _kappa_symmetric(A: csr_matrix) -> float:
    """Condition number of symmetric SPD A from dedicated SA+LA eigsh calls."""
    n = A.shape[0]
    lin = LinearOperator((n, n), matvec=lambda x: A @ x, dtype=np.float64)
    lam_min = float(eigsh(lin, k=1, which="SA", return_eigenvectors=False, tol=1e-4)[0])
    lam_max = float(eigsh(lin, k=1, which="LA", return_eigenvectors=False, tol=1e-4)[0])
    return float(lam_max / max(lam_min, 1e-12))


def _kappa_jacobi(A: csr_matrix, inv_diag: np.ndarray) -> float:
    """Condition number of D^{-1}A via symmetric similarity D^{-1/2} A D^{-1/2}."""
    n = A.shape[0]
    d_isqrt = np.sqrt(np.clip(inv_diag.astype(np.float64, copy=False), 0.0, None))
    lin = LinearOperator((n, n), matvec=lambda x: d_isqrt * (A @ (d_isqrt * x)), dtype=np.float64)
    lam_min = float(eigsh(lin, k=1, which="SA", return_eigenvectors=False, tol=1e-4)[0])
    lam_max = float(eigsh(lin, k=1, which="LA", return_eigenvectors=False, tol=1e-4)[0])
    return float(lam_max / max(lam_min, 1e-12))


def _kappa_nonsym(A: csr_matrix, M_apply: Callable[[np.ndarray], np.ndarray], k: int) -> float:
    """Condition number estimate for M·A using LR+SR eigs with abs-value thresholding."""
    n = A.shape[0]
    k_each = max(4, min(k // 2, n - 2))

    def matvec(x: np.ndarray) -> np.ndarray:
        return M_apply(A @ x)

    lin = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    parts: list[np.ndarray] = []
    for which in ("LR", "SR"):
        try:
            v = eigs(lin, k=k_each, which=which, return_eigenvectors=False,
                     tol=1e-3, maxiter=max(3000, 30 * n))
            parts.append(v.real)
        except Exception:
            continue
    if not parts:
        return float("nan")
    vals = np.concatenate(parts)
    vals = vals[np.isfinite(vals)]
    pos = np.abs(vals[np.abs(vals) > 1e-10])
    return float(pos.max() / pos.min()) if pos.size >= 2 else float("nan")



def _plot_eigenvalue_scatter(
    ax: plt.Axes,
    vals: np.ndarray,
    color: str,
    title: str,
    ref_x: float | None = None,
    kappa_this: float | None = None,
    kappa_ref: float | None = None,
    reduction_green: bool = False,
) -> None:
    """Scatter plot of eigenvalues in the complex plane with rug marks along the real axis.

    kappa_this: condition number of this operator (pre-computed externally).
    kappa_ref:  condition number of the unpreconditioned A.  When both are supplied the
                annotation shows the improvement factor kappa_ref / kappa_this.
    """
    c = np.asarray(vals, dtype=np.complex128)
    ok = np.isfinite(c.real) & np.isfinite(c.imag)
    c = c[ok]
    if c.size == 0:
        ax.text(0.5, 0.5, "No finite eigenvalues", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        return

    re, im = c.real, c.imag
    im_max = float(np.abs(im).max())
    is_real = im_max < 1e-8  # symmetric operator — eigenvalues are numerically real

    if is_real:
        rng = np.random.default_rng(42)
        jitter_scale = max((re.max() - re.min()) * 0.012, 1e-12)
        im_plot = rng.normal(0.0, jitter_scale, size=re.size)
        y_half = float(np.abs(im_plot).max()) * 3.0
    else:
        im_plot = im
        y_half = max(im_max * 1.35, float(np.sqrt(np.mean(im**2))) * 4.0)

    ax.scatter(re, im_plot, c=color, alpha=0.65, s=22, edgecolors="none", zorder=3, rasterized=True)
    ax.axhline(0.0, color="#aaaaaa", lw=0.6, alpha=0.55, zorder=1)

    if ref_x is not None:
        ax.axvline(ref_x, color="#333333", lw=0.9, ls="--", alpha=0.5, zorder=2)

    ax.set_ylim(-y_half, y_half)

    # Rug marks — tick marks projected onto the real axis at the bottom of the plot
    rug_y = -y_half * 0.88
    ax.plot(re, np.full_like(re, rug_y), "|", color=color, alpha=0.5,
            markersize=7, markeredgewidth=0.8, zorder=4)

    x_span = float(re.max() - re.min()) if re.size > 1 else 1.0
    x_pad = max(x_span * 0.08, 1e-10)
    ax.set_xlim(float(re.min()) - x_pad, float(re.max()) + x_pad)

    ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
    if is_real:
        ax.set_ylabel("jittered", fontsize=7.5, color="#999999")
        ax.tick_params(axis="y", labelsize=0, length=0)
    else:
        ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.grid(True, alpha=0.18, lw=0.45, zorder=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))

    # Condition number annotation (kappa_this and kappa_ref computed externally)
    if kappa_this is not None and np.isfinite(kappa_this):
        if kappa_ref is not None and np.isfinite(kappa_ref) and kappa_ref > 0:
            impr = kappa_ref / kappa_this
            ann = rf"${impr:.0f}\times$ $\kappa$ reduction"
            ann_color = "#1a6b1a" if reduction_green else "#444444"
        else:
            ann = rf"$\tilde{{\kappa}} \approx {kappa_this:.2g}$"
            ann_color = "#444444"
        ax.text(0.5, 0.96, ann,
                ha="center", va="top", transform=ax.transAxes,
                fontsize=11.5, fontweight="bold", color=ann_color)


def main() -> None:
    parser = argparse.ArgumentParser(description="Spectral clustering comparison across solver preconditioners.")
    parser.add_argument("--data-folder", type=str, default="data/multiphase_v2_8192/test")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--baseline-weights",
        type=str,
        default=str(_SCRIPT_DIR.parent / "weights" / "v2_8192_d128_L3_hw.bytes"),
    )
    parser.add_argument(
        "--sai-weights",
        type=str,
        default=str(_SCRIPT_DIR.parent / "weights" / "v2_8192_d128_L3_hw_sai.bytes"),
    )
    parser.add_argument("--eig-count", type=int, default=140)
    parser.add_argument(
        "--output",
        type=str,
        default=str(_SCRIPT_DIR / "solver_spectral_clustering.png"),
    )
    args = parser.parse_args()

    _apply_style()

    data_folder = _resolve_path(args.data_folder, _SCRIPT_DIR.parent / "data" / "multiphase_v2_8192" / "test")
    baseline_weights = _resolve_path(args.baseline_weights, _SCRIPT_DIR.parent / "weights" / "v2_8192_d128_L3_hw.bytes")
    sai_weights = _resolve_path(args.sai_weights, _SCRIPT_DIR.parent / "weights" / "v2_8192_d128_L3_hw_sai.bytes")
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

    print(f"[spectra] sampling eigenvalues …")
    eig_unpre = _sample_eigenvalues_symmetric(A, args.eig_count)
    eig_jac   = _sample_eigenvalues_jacobi(A, jacobi_inv_diag, args.eig_count)
    eig_base  = _sample_eigenvalues_nonsym(A, M_base, args.eig_count)
    eig_sai   = _sample_eigenvalues_nonsym(A, M_sai, args.eig_count)

    print(f"[spectra] computing condition numbers …")
    kappa_A    = _kappa_symmetric(A)
    kappa_jac  = _kappa_jacobi(A, jacobi_inv_diag)
    kappa_base = _kappa_nonsym(A, M_base, args.eig_count)
    kappa_sai  = _kappa_nonsym(A, M_sai, args.eig_count)
    print(f"[spectra] κ̃: A={kappa_A:.3g}  jac={kappa_jac:.3g}  base={kappa_base:.3g}  sai={kappa_sai:.3g}")

    fig, axs = plt.subplots(2, 2, figsize=(11.0, 7.2))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.11, wspace=0.38, hspace=0.50)

    _plot_eigenvalue_scatter(axs[0, 0], eig_unpre, _C_UNPRE, r"Unpreconditioned CG  ($A$)",
                             kappa_this=kappa_A)
    _plot_eigenvalue_scatter(axs[0, 1], eig_jac,   _C_JAC,   r"Jacobi-Preconditioned  ($D^{-1}A$)",
                             kappa_this=kappa_jac,  kappa_ref=kappa_A)
    # Bottom-left: SAI Neural
    _plot_eigenvalue_scatter(axs[1, 0], eig_sai,   _C_SAI,   r"SAI Neural  ($M_\mathrm{sai}\,A$)",
                             ref_x=1.0, kappa_this=kappa_sai,  kappa_ref=kappa_A, reduction_green=False)
    # Bottom-right: Cosine-Hutchinson Neural (baseline)
    _plot_eigenvalue_scatter(axs[1, 1], eig_base,  _C_BASE,  r"Cosine-Hutchinson Neural  ($M_\mathrm{cos}\,A$)",
                             ref_x=1.0, kappa_this=kappa_base, kappa_ref=kappa_A, reduction_green=True)

    fig.suptitle("Eigenvalue spectra - solver preconditioners (multiphase Poisson).", fontsize=11, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[spectra] wrote {out_path}")


if __name__ == "__main__":
    main()
