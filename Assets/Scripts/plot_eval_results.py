"""
Plot evaluation summaries for multiphase Poisson benchmark runs.

Creates figures:
1) Baseline-vs-methods across scales (1024..16384, includes 32768 if present)
2) Ablation impact across scales (2048/4096/8192) for:
   total/inference/solve time, iteration reduction, parameter count, training time.

Neural-preconditioner ablation rows with extreme pooled σ/μ (typically PCG-cap frames mixed into the CSV average)
are excluded from ablation figures by default; run EvalScaleSweep.py again to re-average without those frames.

Visual language follows plot_training.py: serif fonts, colored lines; scale-methods ribbons use ±0.5σ.

Usage:
  python Assets/Scripts/plot_eval_results.py
  python Assets/Scripts/plot_eval_results.py --results-dir Assets/Scripts/results
  python Assets/Scripts/plot_eval_results.py --no-ablation-outlier-filter
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

from train_configs import CONFIGS


SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
WEIGHTS_DIR = SCRIPTS_DIR / "weights"
DEFAULT_RESULTS_DIR = SCRIPTS_DIR / "results"
DEFAULT_OUT_METHODS = SCRIPTS_DIR.parent.parent / "Paper" / "figures" / "scale_methods.png"
DEFAULT_OUT_ABLATION = SCRIPTS_DIR.parent.parent / "Paper" / "figures" / "ablation_impact.png"
DEFAULT_OUT_ABLATION_CONFIGS = SCRIPTS_DIR.parent.parent / "Paper" / "figures" / "ablation_configs_scale.png"
DEFAULT_OUT_TRAINING_WALL = SCRIPTS_DIR.parent.parent / "Paper" / "figures" / "ablation_training_wallclock.png"

# Color/style language aligned with plot_training.py
C_BLUE = "#0072B2"
C_GREEN = "#009E73"
C_ORANGE = "#D55E00"
C_PURPLE = "#9467bd"
C_BROWN = "#8C564B"
C_GOLD = "#E69F00"
C_RED = "#E41A1C"


@dataclass(frozen=True)
class MethodStyle:
    label: str
    color: str
    linestyle: str


# Order preserved for legend. AMG+CPU intentionally excluded from the scale-sweep plot:
# its setup cost scales nonlinearly with contrast under PyAMG, which makes the curve
# noisy and not directly comparable to the GPU-resident methods that dominate the
# interactive regime we evaluate. Numbers are preserved in the supplementary table.
METHOD_STYLES: dict[tuple[str, str], MethodStyle] = {
    ("amgx_gpu", "gpu"): MethodStyle("AMGX SPAI (GPU)", C_BLUE, "-"),
    ("ic", "cpu"): MethodStyle("IC+CPU", C_GOLD, "--"),
    ("ic_amgx", "gpu"): MethodStyle("IC+CUDA", "#00BA38", ":"),
    ("jacobi", "gpu"): MethodStyle("Diag+CUDA", C_RED, "--"),
    ("cg", "gpu"): MethodStyle("None+CUDA", C_PURPLE, "-."),
    # Neural SPAI baseline (Yang2025sparse, re-trained per scale; CUDA solve path)
    # is loaded from lsp_scale_infer_summary.csv and injected as a synthetic
    # (method, device) tuple in _load_neural_spai_rows below.
    ("neural_spai", "gpu"): MethodStyle("Neural SPAI (GPU)", C_ORANGE, (0, (4, 2))),
    ("leafonly", "gpu"): MethodStyle("Ours", C_BROWN, (0, (2, 1, 1, 1))),
}

# Frame-budget reference lines (full-frame Poisson+solve budget, single threaded assumption).
FPS_REFERENCE_MS = (
    (60, 1000.0 / 60.0),
    (24, 1000.0 / 24.0),
)
TIME_Y_MIN_MS = 5.0
TIME_Y_MAX_MS = 1000.0

# Pooled CSV rows with extreme σ/mean usually mix converged frames with PCG-cap failures; drop from ablation figures.
ABLATION_POOL_REL_SIGMA_MAX_DEFAULT = 3.0


ABLATION_SERIES = [
    ("baseline", "Baseline (d=128, L=3, hw)", C_BROWN, "-"),
    ("d64", "d_model=64", C_BLUE, "--"),
    ("d256", "d_model=256", C_ORANGE, ":"),
    ("L1", "layers=1", C_GREEN, "-."),
    ("L2", "layers=2", C_PURPLE, (0, (3, 1, 1, 1))),
    ("nohw", "no highways", C_RED, (0, (5, 2))),
]

ABLATION_COLOR_BY_FAMILY = {k: c for k, _lbl, c, _ls in ABLATION_SERIES}
ABLATION_LINESTYLE_BY_FAMILY = {k: ls for k, _lbl, _c, ls in ABLATION_SERIES}


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


def _to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _drop_interior_scale_outliers(
    x: np.ndarray,
    y: np.ndarray,
    y_std: np.ndarray,
    *,
    dip_vs_neighbor: float = 0.55,
    spike_vs_neighbor: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop interior scale points that are obvious single-grid dips/spikes vs neighbors.

    Endpoints are kept so legitimate blow-ups at the largest scale remain visible.
    """

    n = int(len(x))
    if n < 3:
        return x, y, y_std
    order = np.argsort(x.astype(float))
    xs = x[order].astype(float)
    ys = y[order].astype(float)
    sig = y_std[order].astype(float)
    keep = np.ones(n, dtype=bool)
    for i in range(1, n - 1):
        yi, ym, yp = ys[i], ys[i - 1], ys[i + 1]
        if yi < ym and yi < yp and yi < dip_vs_neighbor * float(min(ym, yp)):
            keep[i] = False
        elif yi > ym and yi > yp and yi > spike_vs_neighbor * float(max(ym, yp)):
            keep[i] = False
    if bool(np.all(keep)):
        return x, y, y_std
    return xs[keep], ys[keep], sig[keep]


def _ablation_pool_rel_sigma(leaf: pd.DataFrame) -> pd.Series:
    """max(σ/μ) over total_ms and PCG iters — flags pooled stats polluted by rare PCG-cap frames."""

    t = leaf["total_ms"].to_numpy(dtype=float)
    ts = leaf["total_ms_std"].fillna(0.0).to_numpy(dtype=float)
    it = leaf["iters"].to_numpy(dtype=float)
    its = leaf["iters_std"].fillna(0.0).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rt = np.where(np.isfinite(t) & (t > 0), ts / t, 0.0)
        ri = np.where(np.isfinite(it) & (it > 0), its / it, 0.0)
    return pd.Series(np.maximum(rt, ri), index=leaf.index, dtype=float)


def _partition_ablation_leaf_for_plots(
    leaf: pd.DataFrame,
    *,
    max_rel_sigma: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept_for_plots, excluded_rows_with_sigma_column)."""

    sig = _ablation_pool_rel_sigma(leaf)
    work = leaf.copy()
    work["_pool_rel_sigma"] = sig
    bad = sig >= float(max_rel_sigma)
    excluded = work.loc[bad].copy()
    kept = work.loc[~bad].drop(columns=["_pool_rel_sigma"])
    return kept, excluded


def _print_ablation_leaf_data_audit(kept: pd.DataFrame, excluded: pd.DataFrame, *, max_rel_sigma: float) -> None:
    cols = [
        "family",
        "config",
        "scale",
        "total_ms",
        "total_ms_std",
        "solve_ms",
        "solve_ms_std",
        "iters",
        "iters_std",
        "n_frames",
        "iter_reduction_pct",
        "iter_reduction_pct_std",
    ]
    present = [c for c in cols if c in kept.columns]
    gate_disabled = max_rel_sigma != max_rel_sigma  # NaN sentinel when filter off
    if gate_disabled:
        print(
            "\nAblation plots — stability gate: disabled (--no-ablation-outlier-filter).\n"
            "All neural ablation rows from the CSV are plotted as-is."
        )
    else:
        print(
            "\nAblation plots — stability gate: max(σ/μ over total_ms and iters) "
            f"< {max_rel_sigma:g} (excludes pooled CSV rows dominated by PCG-cap outliers).\n"
            "Note: Means±σ in the CSV still mix outlier frames until you re-run EvalScaleSweep.py "
            "(it drops PCG-cap frames before averaging)."
        )
    if not excluded.empty:
        print("\nExcluded from plots (unstable pooled statistics):")
        ex_cols = present + [c for c in ("_pool_rel_sigma",) if c in excluded.columns]
        print(excluded[ex_cols].sort_values(["family", "config", "scale"]).to_string(index=False))
    else:
        print("\nExcluded from plots: (none)")
    print("\nIncluded in plots (all metrics below are taken from these rows only):")
    if kept.empty:
        print("  (none — check results CSV or loosen --ablation-pool-rel-sigma-max)")
    else:
        print(kept[present].sort_values(["family", "config", "scale"]).to_string(index=False))


def _load_neural_spai_rows(results_dir: pathlib.Path) -> pd.DataFrame:
    """Read Neural SPAI (CUDA) timings from the lsp_scale_infer_summary CSV.

    The neural-SPAI sweep records mean totals only (no per-frame std), so the
    returned rows fill total_ms_std/iters_std with NaN — the plotting code
    handles missing std gracefully (no ribbon).
    """
    p = results_dir / "lsp_scale_infer_summary.csv"
    if not p.exists():
        return pd.DataFrame(
            columns=[
                "scale", "config", "method", "device",
                "total_ms", "total_ms_std", "iters", "iters_std",
            ]
        )
    raw = pd.read_csv(p)
    rows = raw[raw["method"].astype(str) == "Neural+CUDA"].copy()
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "scale", "config", "method", "device",
                "total_ms", "total_ms_std", "iters", "iters_std",
            ]
        )
    out = pd.DataFrame(
        {
            "scale": pd.to_numeric(rows["scale"], errors="coerce"),
            "config": rows["exp_name"].astype(str).str.replace("multiphase_v2_", "v2_") + "_neural_spai",
            "method": "neural_spai",
            "device": "gpu",
            "total_ms": pd.to_numeric(rows["total_time_ms"], errors="coerce"),
            "total_ms_std": float("nan"),
            "iters": pd.to_numeric(rows["iterations"], errors="coerce"),
            "iters_std": float("nan"),
        }
    )
    return out


def _load_results(results_dir: pathlib.Path) -> pd.DataFrame:
    task_files = sorted(results_dir.glob("ablation_sweep_task_*.csv"))
    if task_files:
        frames = [pd.read_csv(p) for p in task_files]
        df = pd.concat(frames, ignore_index=True)
    else:
        merged = results_dir / "ablation_sweep.csv"
        if not merged.exists():
            raise FileNotFoundError(
                f"No ablation results found in {results_dir} "
                "(expected ablation_sweep_task_*.csv or ablation_sweep.csv)."
            )
        df = pd.read_csv(merged)
    return _to_num(
        df,
        [
            "scale",
            "setup_ms",
            "setup_ms_std",
            "inference_ms",
            "inference_ms_std",
            "solve_ms",
            "solve_ms_std",
            "iters",
            "iters_std",
            "total_ms",
            "total_ms_std",
            "n_frames",
        ],
    )


def _config_map() -> dict[str, dict]:
    return {cfg["name"]: dict(cfg) for cfg in CONFIGS}


def _training_time_hours(config_name: str) -> float:
    p = WEIGHTS_DIR / f"{config_name}_training_profile.csv"
    if not p.exists():
        return float("nan")
    df = pd.read_csv(p)
    if "elapsed_s" not in df.columns or df.empty:
        return float("nan")
    return float(df["elapsed_s"].cumsum().iloc[-1] / 3600.0)


def _count_params(cfg: dict) -> int:
    # Imported lazily to keep script startup fast.
    from leafonly import config as cfg_mod
    from leafonly.architecture import LeafOnlyNet

    cfg_mod.apply_runtime_sizes(int(cfg["leaf"]), int(cfg["scale"]))
    model = LeafOnlyNet(
        input_dim=9,
        d_model=int(cfg["d"]),
        leaf_size=int(cfg_mod.LEAF_SIZE),
        num_layers=int(cfg["L"]),
        num_heads=8,
        use_gcn=True,
        num_gcn_layers=2,
        use_jacobi=True,
        use_highways=bool(cfg["hw"]),
    )
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _build_leafonly_ablation_table(df: pd.DataFrame, cfg_map: dict[str, dict]) -> pd.DataFrame:
    target_scales = {2048, 4096, 8192}
    leaf = df[(df["method"] == "leafonly") & (df["device"] == "gpu") & (df["scale"].isin(target_scales))].copy()
    cg = df[(df["method"] == "cg") & (df["device"] == "gpu") & (df["scale"].isin(target_scales))].copy()

    if leaf.empty or cg.empty:
        raise ValueError("Missing required leafonly/cg gpu rows for ablation scales 2048/4096/8192.")

    # Parse variant family from config name
    def family(name: str) -> str:
        if "_d64_" in name:
            return "d64"
        if "_d256_" in name:
            return "d256"
        if "_L1_" in name:
            return "L1"
        if "_L2_" in name:
            return "L2"
        if "_nohw" in name:
            return "nohw"
        return "baseline"

    leaf["family"] = leaf["config"].astype(str).map(family)
    leaf["params"] = leaf["config"].map(lambda n: _count_params(cfg_map[str(n)]))
    leaf["train_hours"] = leaf["config"].map(lambda n: _training_time_hours(str(n)))

    # Iteration reduction against same-scale unpreconditioned CG (GPU)
    cg_small = cg[["scale", "iters", "iters_std"]].rename(columns={"iters": "cg_iters", "iters_std": "cg_iters_std"})
    leaf = leaf.merge(cg_small, on="scale", how="left")
    leaf["iter_reduction_pct"] = 100.0 * (1.0 - (leaf["iters"] / leaf["cg_iters"]))
    # First-order uncertainty propagation for r = 1 - a/b
    a = leaf["iters"].to_numpy(dtype=float)
    sa = leaf["iters_std"].fillna(0.0).to_numpy(dtype=float)
    b = leaf["cg_iters"].to_numpy(dtype=float)
    sb = leaf["cg_iters_std"].fillna(0.0).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.sqrt((sa / b) ** 2 + ((a * sb) / (b * b)) ** 2)
    leaf["iter_reduction_pct_std"] = 100.0 * sr

    return leaf


def _plot_methods_across_scale(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    # Baseline configs for method-vs-scale plot
    baseline = df[
        df["config"].astype(str).str.contains(r"_d128_L3_hw$|_neural_spai$")
        & df["scale"].isin([1024, 2048, 4096, 8192, 16384, 32768])
    ].copy()

    if baseline.empty:
        raise ValueError("No baseline rows found for d128/L3/highways scale sweep.")

    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    # Leave figure margin on the right so FPS labels (outside axes in data-y) are not clipped.
    fig.subplots_adjust(left=0.14, right=0.88, top=0.88, bottom=0.17)

    handles = []
    y_for_limits: list[float] = []

    for key, sty in METHOD_STYLES.items():
        method, device = key
        sub = baseline[(baseline["method"] == method) & (baseline["device"] == device)].sort_values("scale")
        if sub.empty:
            continue
        x = sub["scale"].to_numpy(dtype=float)
        y = sub["total_ms"].to_numpy(dtype=float)
        ys = sub["total_ms_std"].fillna(0.0).to_numpy(dtype=float)

        x, y, ys = _drop_interior_scale_outliers(x, y, ys)

        # Half-sigma uncertainty band (±0.5σ) for readability with many overlapping methods.
        ys = 0.5 * ys
        # Robust uncertainty band: cap sigma to 45% of mean to avoid one noisy run
        # blowing up the axis (especially on log plots).
        ys = np.minimum(ys, 0.45 * np.maximum(y, 1e-9))

        line, = ax.plot(
            x,
            y,
            color=sty.color,
            linestyle=sty.linestyle,
            label=sty.label,
            zorder=3,
        )
        lo = np.maximum(1e-6, y - ys)
        hi = y + ys
        ax.fill_between(x, lo, hi, color=sty.color, alpha=0.18, zorder=2)
        y_for_limits.extend(lo.tolist())
        y_for_limits.extend(hi.tolist())
        y_for_limits.extend(y.tolist())
        handles.append(line)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    x_ticks = sorted(baseline["scale"].dropna().unique().tolist())
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{int(v)}" for v in x_ticks])
    # Log-style uneven vertical grid: dense minor ticks across decades.
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10, dtype=float)))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    # Human-readable log y ticks in milliseconds: 10, 20, 50, 100, 200, ...
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(1, 10, dtype=float)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _p: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # Horizontal FPS budgets (behind curves). Budget = full-frame time for that refresh rate.
    trans_axes_x_data_y = blended_transform_factory(ax.transAxes, ax.transData)
    _fps_outline = [
        path_effects.withStroke(linewidth=3.5, foreground="white", alpha=1.0),
    ]

    for fps, budget_ms in FPS_REFERENCE_MS:
        ax.axhline(budget_ms, color="#555555", ls=(0, (6, 3)), lw=0.85, alpha=0.65, zorder=1)
        # Labels outside plotting region (x > 1 in axes coords) so data ribbons cannot obscure them.
        txt = ax.text(
            1.012,
            budget_ms,
            f"{fps} fps ({budget_ms:.2f} ms)",
            transform=trans_axes_x_data_y,
            ha="left",
            va="center",
            fontsize=8,
            color="#1a1a1a",
            clip_on=False,
            zorder=6,
        )
        txt.set_path_effects(_fps_outline)

    # Log scale cannot include true y=0; extend ymin downward vs percentile clipping so fast curves
    # do not sit on the axis edge (readable separation).
    if y_for_limits:
        y_arr = np.asarray([v for v in y_for_limits if np.isfinite(v) and v > 0], dtype=float)
        if y_arr.size:
            ymin_data = float(np.nanmin(y_arr))
            ymax_data = float(np.nanpercentile(y_arr, 98))
            # Floor at 5 ms (readable log lower bound; avoids implying sub-ms totals).
            y_bottom = max(5.0, ymin_data * 0.22)
            y_top = max(ymax_data * 1.22, np.nanmax(y_arr) * 1.08)
            # Include FPS guides inside view when reasonable.
            y_top = max(y_top, max(ms for _, ms in FPS_REFERENCE_MS) * 1.06)
            y_top = min(y_top, 1000.0)
            ax.set_ylim(y_bottom, y_top)
    ax.set_xlabel("Matrix size")
    ax.set_ylabel("Total time (ms)")
    ax.set_title("Scale Sweep: Baseline vs Methods (multiphase Poisson)")
    ax.grid(True, which="major", color="#d8d8d8", zorder=0)
    ax.grid(True, which="minor", color="#efefef", zorder=0)
    ax.legend(
        handles=handles,
        title="Method",
        loc="upper left",
        ncol=2,
        framealpha=0.9,
        edgecolor="#cccccc",
        fontsize=8,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")


def _plot_ablation_impact(leaf: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.3))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.14, wspace=0.28, hspace=0.33)
    axs = axes.flatten()

    metrics = [
        ("total_ms", "total_ms_std", "Total time (ms)"),
        ("inference_ms", "inference_ms_std", "Inference time (ms)"),
        ("solve_ms", "solve_ms_std", "Solve time (ms)"),
        ("iter_reduction_pct", "iter_reduction_pct_std", "Iteration reduction (%)"),
        ("params", None, "Parameters (millions)"),
        ("train_hours", None, "Training time (hours)"),
    ]

    legend_handles = []
    for fam, label, color, ls in ABLATION_SERIES:
        sub = leaf[leaf["family"] == fam].sort_values("scale")
        if sub.empty:
            continue
        x = sub["scale"].to_numpy(dtype=float)
        for i, (m, s, ylab) in enumerate(metrics):
            y = sub[m].to_numpy(dtype=float)
            if m == "params":
                y = y / 1e6
            ax = axs[i]
            kw = {"color": color, "linestyle": ls, "zorder": 3, "label": label}
            if len(x) == 1:
                kw["marker"] = "o"
                kw["markersize"] = 5.0
            line, = ax.plot(x, y, **kw)
            if i == 0:
                legend_handles.append(line)
            if s is not None and s in sub.columns and len(x) >= 2:
                ys = sub[s].fillna(0.0).to_numpy(dtype=float)
                lo = y - ys
                hi = y + ys
                ax.fill_between(x, lo, hi, color=color, alpha=0.18, zorder=2)

    for i, (m, _s, ylab) in enumerate(metrics):
        ax = axs[i]
        ax.set_xscale("log", base=2)
        ax.set_xticks([2048, 4096, 8192])
        ax.set_xticklabels(["2048", "4096", "8192"])
        ax.set_xlabel("Matrix size")
        ax.set_ylabel(ylab)
        if m in ("total_ms", "solve_ms", "inference_ms"):
            ax.set_yscale("log", base=10)
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(1, 10, dtype=float)))
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _p: f"{v:,.0f}" if v >= 1 else f"{v:g}")
            )
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            # Use same axis family as scale_methods for apples-to-apples read:
            #  - total/solve panels: fixed [5, 1000] ms
            #  - inference: keep same tick style, adaptive lower range (typically < 10 ms).
            if m in ("total_ms", "solve_ms"):
                ax.set_ylim(TIME_Y_MIN_MS, TIME_Y_MAX_MS)
            else:
                yvals = np.asarray(ax.get_lines()[0].get_ydata() if ax.get_lines() else [1.0], dtype=float)
                ymin = float(max(np.nanmin(yvals), 0.5))
                ymax = float(max(np.nanmax(yvals), 1.0))
                ax.set_ylim(max(0.5, ymin * 0.75), min(TIME_Y_MAX_MS, ymax * 1.35))
        if m in ("total_ms", "solve_ms"):
            for fps, budget_ms in FPS_REFERENCE_MS:
                ax.axhline(budget_ms, color="#555555", ls=(0, (6, 3)), lw=0.75, alpha=0.6, zorder=1)
        ax.grid(True, which="major", color="#dddddd", zorder=0)
        ax.grid(True, which="minor", color="#f0f0f0", zorder=0)
        ax.set_title(ylab)

    fig.suptitle("Ablation Impact Across Scales (multiphase Poisson, GCN layers fixed at 2)")
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")


def _plot_ablation_configs_scale(leaf: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Single-panel, scale-vs-time plot for ablation configurations (same visual language as scale_methods)."""
    fig, ax = plt.subplots(figsize=(7.0, 5.1))
    fig.subplots_adjust(left=0.14, right=0.88, top=0.88, bottom=0.17)

    y_vals: list[float] = []
    legend_handles = []
    for fam, label, color, ls in ABLATION_SERIES:
        sub = leaf[leaf["family"] == fam].sort_values("scale")
        if sub.empty:
            continue
        x = sub["scale"].to_numpy(dtype=float)
        y = sub["solve_ms"].to_numpy(dtype=float)
        ys = sub["solve_ms_std"].fillna(0.0).to_numpy(dtype=float)
        ys = 0.5 * np.minimum(ys, 0.45 * np.maximum(y, 1e-9))
        kw = {"color": color, "linestyle": ls, "label": label, "zorder": 3}
        if len(x) == 1:
            kw["marker"] = "o"
            kw["markersize"] = 5.0
        ln, = ax.plot(x, y, **kw)
        if len(x) >= 2:
            lo = np.maximum(1e-6, y - ys)
            hi = y + ys
            ax.fill_between(x, lo, hi, color=color, alpha=0.18, zorder=2)
            y_vals.extend(lo.tolist())
            y_vals.extend(hi.tolist())
        y_vals.extend(y.tolist())
        legend_handles.append(ln)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xticks([2048, 4096, 8192])
    ax.set_xticklabels(["2048", "4096", "8192"])
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10, dtype=float)))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(1, 10, dtype=float)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _p: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    for fps, budget_ms in FPS_REFERENCE_MS:
        ax.axhline(budget_ms, color="#555555", ls=(0, (6, 3)), lw=0.85, alpha=0.65, zorder=1)

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    outline = [path_effects.withStroke(linewidth=3.5, foreground="white", alpha=1.0)]
    for fps, budget_ms in FPS_REFERENCE_MS:
        t = ax.text(
            1.012,
            budget_ms,
            f"{fps} fps ({budget_ms:.2f} ms)",
            transform=trans,
            ha="left",
            va="center",
            fontsize=8,
            color="#1a1a1a",
            clip_on=False,
            zorder=6,
        )
        t.set_path_effects(outline)

    if y_vals:
        arr = np.asarray([v for v in y_vals if np.isfinite(v) and v > 0], dtype=float)
        if arr.size:
            ymin_data = float(np.nanmin(arr))
            ymax_data = float(np.nanpercentile(arr, 98))
            y_bottom = max(TIME_Y_MIN_MS, ymin_data * 0.22)
            y_top = max(ymax_data * 1.22, np.nanmax(arr) * 1.08)
            y_top = max(y_top, max(ms for _, ms in FPS_REFERENCE_MS) * 1.06)
            y_top = min(y_top, TIME_Y_MAX_MS)
            ax.set_ylim(y_bottom, y_top)

    ax.set_xlabel("Matrix size")
    ax.set_ylabel("Our method solve time (ms, log scale)")
    ax.set_title("Ablation by Configuration (multiphase Poisson)")
    ax.grid(True, which="major", color="#d8d8d8", zorder=0)
    ax.grid(True, which="minor", color="#efefef", zorder=0)
    ax.legend(
        handles=legend_handles,
        title="Configuration",
        loc="upper left",
        ncol=2,
        framealpha=0.9,
        edgecolor="#cccccc",
        fontsize=8,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")


def _family_from_config_name(name: str) -> str:
    if "_d64_" in name:
        return "d64"
    if "_d256_" in name:
        return "d256"
    if "_L1_" in name:
        return "L1"
    if "_L2_" in name:
        return "L2"
    if "_nohw" in name:
        return "nohw"
    return "baseline"


def _plot_training_loss_wallclock(cfg_map: dict[str, dict], out_path: pathlib.Path) -> None:
    """Training loss vs wall-clock time for all available training profiles."""
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.17)

    handles = []
    for cfg_name in sorted(cfg_map.keys()):
        p = WEIGHTS_DIR / f"{cfg_name}_training_profile.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if df.empty or "elapsed_s" not in df.columns or "loss_avg" not in df.columns:
            continue
        family = _family_from_config_name(cfg_name)
        color = ABLATION_COLOR_BY_FAMILY.get(family, "#666666")
        ls = ABLATION_LINESTYLE_BY_FAMILY.get(family, "-")
        wall_min = df["elapsed_s"].cumsum().to_numpy(dtype=float) / 60.0
        loss = df["loss_avg"].to_numpy(dtype=float)
        # Keep this readable with many curves: no ribbons; baseline thicker.
        lw = 2.0 if family == "baseline" else 1.15
        alpha = 0.95 if family == "baseline" else 0.55
        ln, = ax.plot(wall_min, loss, color=color, linestyle=ls, lw=lw, alpha=alpha, zorder=3)
        if family == "baseline":
            handles.append(ln)

    # Legend describes families rather than every config to avoid clutter.
    proxy_handles = []
    for fam, label, color, ls in ABLATION_SERIES:
        proxy, = ax.plot([], [], color=color, linestyle=ls, lw=1.8, label=label)
        proxy_handles.append(proxy)
    ax.legend(
        handles=proxy_handles,
        title="Family",
        ncol=2,
        loc="upper right",
        framealpha=0.9,
        edgecolor="#cccccc",
        fontsize=8,
    )

    ax.set_xlabel("Wall-clock time (min)")
    ax.set_ylabel(r"$\mathcal{L}_{\cos}$ (linear)")
    ax.set_title("Training Loss vs Wall-clock Across Configurations")
    ax.grid(True, which="major", color="#d8d8d8", zorder=0)
    ax.grid(True, which="minor", color="#efefef", zorder=0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot scale sweep + ablation figures from results CSVs.")
    parser.add_argument("--results-dir", type=pathlib.Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-methods", type=pathlib.Path, default=DEFAULT_OUT_METHODS)
    parser.add_argument("--out-ablation", type=pathlib.Path, default=DEFAULT_OUT_ABLATION)
    parser.add_argument("--out-ablation-configs", type=pathlib.Path, default=DEFAULT_OUT_ABLATION_CONFIGS)
    parser.add_argument("--out-training-wall", type=pathlib.Path, default=DEFAULT_OUT_TRAINING_WALL)
    parser.add_argument("--methods-only", action="store_true", help="Generate only the scale-methods figure.")
    parser.add_argument(
        "--no-ablation-outlier-filter",
        action="store_true",
        help="Use every neural ablation CSV row (disable σ/μ stability gate for plots).",
    )
    parser.add_argument(
        "--ablation-pool-rel-sigma-max",
        type=float,
        default=ABLATION_POOL_REL_SIGMA_MAX_DEFAULT,
        metavar="R",
        help=(
            "Exclude neural-method rows when max(σ/μ) over total_ms and iters is ≥ R "
            "(default: %(default)s; guards against PCG-cap frames in pooled CSVs)."
        ),
    )
    args = parser.parse_args()

    _apply_style()
    df = _load_results(args.results_dir)
    neural_spai = _load_neural_spai_rows(args.results_dir)
    if not neural_spai.empty:
        df = pd.concat([df, neural_spai], ignore_index=True)
    cfg_map = _config_map()
    _plot_methods_across_scale(df, args.out_methods)
    if args.methods_only:
        return
    leaf = _build_leafonly_ablation_table(df, cfg_map)
    if args.no_ablation_outlier_filter:
        _print_ablation_leaf_data_audit(leaf, pd.DataFrame(), max_rel_sigma=float("nan"))
    else:
        leaf_kept, leaf_excluded = _partition_ablation_leaf_for_plots(
            leaf, max_rel_sigma=args.ablation_pool_rel_sigma_max
        )
        _print_ablation_leaf_data_audit(leaf_kept, leaf_excluded, max_rel_sigma=args.ablation_pool_rel_sigma_max)
        leaf = leaf_kept
    _plot_ablation_impact(leaf, args.out_ablation)
    _plot_ablation_configs_scale(leaf, args.out_ablation_configs)
    _plot_training_loss_wallclock(cfg_map, args.out_training_wall)


if __name__ == "__main__":
    main()
