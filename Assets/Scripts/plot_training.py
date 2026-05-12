"""
Training profile figure for the default trained checkpoint (d=128, L=3, hw, N=8192).

Single plot, three y-axes:
  Left  (primary)   – cosine Hutchinson loss  L_cos  (mean ± 1σ ribbon)
  Left  (secondary) – SAI loss (offset outward)
  Right             – PCG iteration count (log y; left axes are linear)

Dual x-axes: training steps (bottom) and cumulative wall-clock time (minutes, top).

Horizontal reference lines on the PCG axis for every classical/baseline
method whose iteration count exceeds the trained model's final value.

Usage:
    python Assets/Scripts/plot_training.py [--out Paper/figures/training.png]
"""

import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
WEIGHTS_DIR = pathlib.Path(__file__).parent / "weights"
DEFAULT_OUT  = pathlib.Path(__file__).parent.parent.parent / "Paper" / "figures" / "training.png"
CSV_STEM     = "v2_8192_d128_L3_hw"

# Baseline iteration counts at N=8192 that are WORSE than our trained model.
# Only draw lines for baselines above our final PCG count.
BASELINES = [
    ("Unprecond. CG",  2163, "#888888"),
    ("Jacobi (GPU)",   1248, "#aaaaaa"),
]

# Colours
C_LOSS = "#0072B2"   # blue  – cosine loss
C_SAI  = "#009E73"   # green – SAI loss
C_PCG  = "#D55E00"   # vermilion – PCG iterations

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load() -> pd.DataFrame:
    path = WEIGHTS_DIR / f"{CSV_STEM}_training_profile.csv"
    df = pd.read_csv(path)
    df["wall_min"] = df["elapsed_s"].cumsum() / 60.0
    return df


def _pcg_log_ticks(ymin: float, ymax: float, max_ticks: int = 12) -> np.ndarray:
    """Major y-ticks for PCG iteration counts on a log axis: {1,2,5}×10^k inside [ymin, ymax]."""
    ymin = float(max(ymin, 1.0))
    ymax = float(max(ymax, ymin * 1.001))
    lo = np.log10(ymin * 0.98)
    hi = np.log10(ymax * 1.02)
    k0 = int(np.floor(lo))
    k1 = int(np.ceil(hi))
    ticks: list[float] = []
    for k in range(k0, k1 + 1):
        for m in (1, 2, 5):
            t = m * (10.0 ** k)
            if ymin * 0.97 <= t <= ymax * 1.03:
                ticks.append(t)
    ticks = sorted(set(ticks))
    if len(ticks) <= max_ticks:
        return np.asarray(ticks, dtype=float)
    idx = np.unique(np.linspace(0, len(ticks) - 1, max_ticks, dtype=int))
    return np.asarray([ticks[i] for i in idx], dtype=float)


def _fmt_pcg_tick(x: float, _pos: object = None) -> str:
    if not np.isfinite(x):
        return ""
    if abs(x - round(x)) < 1e-5 * max(x, 1.0) and x >= 10.0:
        return f"{int(round(x)):,}"
    return f"{x:g}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load()

    # Full loss trace (every 100 steps, including step 0)
    steps_all  = df["step"].values
    loss_mean  = df["loss_avg"].values
    loss_std   = df["loss_std"].values
    wall_all   = df["wall_min"].values

    # Checkpoint rows — drop step 0 (untrained model; PCG=5000, SAI=56 are outliers)
    ck = df[df["pcg_iters"].notna() & (df["step"] > 0)].copy()
    steps_ck  = ck["step"].values
    pcg_iters = ck["pcg_iters"].values
    sai_loss  = ck["sai_loss"].values

    # For SAI axis scaling, skip the first checkpoint (step 2000, ~21.9) which is
    # an extreme transient that makes the rest of the curve unreadable.
    sai_for_scale = sai_loss[1:]

    # -----------------------------------------------------------------------
    # Figure and axes
    # -----------------------------------------------------------------------
    mpl.rcParams.update({
        "font.family":       "serif",
        "font.size":         9,
        "axes.labelsize":    10,
        "axes.titlesize":    10.5,
        "legend.fontsize":   8.5,
        "xtick.labelsize":   8.5,
        "ytick.labelsize":   8.5,
        "axes.linewidth":    0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "grid.linewidth":    0.45,
        "lines.linewidth":   1.6,
        "figure.dpi":        180,
        "savefig.dpi":       300,
    })

    # Larger canvas vs. original script → more plot area; type sized for readability
    fig, ax_loss = plt.subplots(figsize=(9.0, 5.6))
    fig.subplots_adjust(left=0.14, right=0.88, top=0.88, bottom=0.16)

    # Right axis – PCG iterations
    ax_pcg = ax_loss.twinx()

    # Second left axis – SAI loss (offset outward to the left)
    ax_sai = ax_loss.twinx()
    ax_sai.spines["left"].set_position(("axes", -0.12))
    ax_sai.spines["left"].set_visible(True)
    ax_sai.yaxis.set_label_position("left")
    ax_sai.yaxis.tick_left()
    # Hide the extra right spine that twinx() adds
    ax_sai.spines["right"].set_visible(False)

    # -----------------------------------------------------------------------
    # Plot: cosine loss ribbon + line
    # -----------------------------------------------------------------------
    ax_loss.fill_between(steps_all,
                         loss_mean - loss_std,
                         loss_mean + loss_std,
                         color=C_LOSS, alpha=0.18, linewidth=0, zorder=2)
    ax_loss.plot(steps_all, loss_mean,
                 color=C_LOSS, lw=2.0, zorder=3,
                 label=r"$\mathcal{L}_{\cos}$ (mean $\pm$ 1$\sigma$, linear)")

    # -----------------------------------------------------------------------
    # Plot: SAI loss
    # -----------------------------------------------------------------------
    ax_sai.plot(steps_ck, sai_loss,
                color=C_SAI, lw=1.5, ls=(0, (5, 2)),
                marker="s", markersize=4.5, markerfacecolor="white",
                markeredgecolor=C_SAI, markeredgewidth=1.0,
                zorder=3, label="SAI loss (linear)")

    # -----------------------------------------------------------------------
    # Plot: PCG iterations
    # -----------------------------------------------------------------------
    ax_pcg.semilogy(steps_ck, pcg_iters,
                    color=C_PCG, lw=1.8,
                    marker="o", markersize=5.5, markerfacecolor="white",
                    markeredgecolor=C_PCG, markeredgewidth=1.2,
                    zorder=4, label="PCG iterations (log)")

    # -----------------------------------------------------------------------
    # Axis limits, labels, ticks
    # -----------------------------------------------------------------------
    x_max = steps_all[-1]
    ax_loss.set_xlim(0, x_max)

    # Baseline reference lines. With the y-axis capped at 600 (set below), baselines that
    # exceed the cap are off-scale; we draw their labels just inside the top of the visible
    # region with an "off-scale" indicator instead of letting them float outside the axes.
    final_pcg = pcg_iters[-1]
    visible_baselines = [(lbl, it, col) for lbl, it, col in BASELINES if it > final_pcg]
    pcg_cap = 600.0
    off_scale_anchor = pcg_cap * 0.97
    off_scale_offset = 0.78  # multiplicative spacing for multiple off-scale labels on log axis
    n_off_scale = 0
    for bl_label, iters, color in visible_baselines:
        if iters <= pcg_cap:
            ax_pcg.axhline(iters, color=color, ls="--", lw=0.9, zorder=1)
            y_lbl = iters * 0.97
            text = f"{bl_label} ({iters:,} iterations)"
            va = "top"
        else:
            y_lbl = off_scale_anchor * (off_scale_offset ** n_off_scale)
            text = f"{bl_label}: {iters:,} iters (off-scale)"
            va = "top"
            n_off_scale += 1
        ax_pcg.text(
            x_max * 0.99,
            y_lbl,
            text,
            color=color,
            fontsize=8.5,
            ha="right",
            va=va,
            zorder=5,
        )

    # PCG y-limits: cap at 600 to keep the model's iteration range readable; baselines that
    # exceed this are still drawn but their labels are positioned at the cap.
    pcg_min = float(np.nanmin(pcg_iters))
    ax_pcg.set_ylim(max(1.0, pcg_min * 0.88), 600.0)

    # Loss axis
    ax_loss.set_ylim(bottom=-0.005, top=loss_mean.max() * 1.08)
    ax_loss.set_ylabel(r"$\mathcal{L}_{\cos}$ (linear)", color=C_LOSS)
    ax_loss.tick_params(axis="y", colors=C_LOSS)
    ax_loss.spines["left"].set_edgecolor(C_LOSS)
    ax_loss.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

    # SAI axis — log scale so the dip-and-rebound pattern is clearly visible: the trace
    # falls from ~22 to ~0.15 around step ~10k and then climbs back to ~0.5, which on a
    # linear axis compressed against the cosine-loss range visually flattens to a line
    # right on top of the cosine curve. Log makes the rebound a visibly distinct second
    # phase and keeps the curve on its own band, well above the cosine-loss range.
    ax_sai.set_ylabel("SAI loss (log)", color=C_SAI, labelpad=6)
    ax_sai.tick_params(axis="y", colors=C_SAI)
    ax_sai.spines["left"].set_edgecolor(C_SAI)
    ax_sai.set_yscale("log")
    ax_sai.set_ylim(bottom=0.1, top=30.0)

    # PCG axis (log): explicit {1,2,5}×10^k majors + plain integers — LogLocator on twinx often
    # drops to a single $10^k$ label in practice.
    ax_pcg.set_ylabel("PCG iterations (log)", color=C_PCG)
    ax_pcg.tick_params(axis="y", colors=C_PCG)
    ax_pcg.spines["right"].set_edgecolor(C_PCG)
    y0, y1 = ax_pcg.get_ylim()
    pcg_ticks = _pcg_log_ticks(y0, y1, max_ticks=12)
    ax_pcg.yaxis.set_major_locator(ticker.FixedLocator(pcg_ticks))
    ax_pcg.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_pcg_tick))
    ax_pcg.yaxis.set_minor_locator(ticker.NullLocator())

    # Grid (only on the main axis to avoid clutter)
    ax_loss.grid(True, which="major", color="#dddddd", zorder=0)
    ax_loss.grid(True, which="minor", color="#f0f0f0", zorder=0)

    # -----------------------------------------------------------------------
    # Primary x-axis (bottom): wall-clock minutes. The underlying data is plotted against
    # step indices, so we set tick positions at step-coordinates corresponding to round
    # minute values and re-label them in minutes.
    # -----------------------------------------------------------------------
    total_wall = wall_all[-1]
    raw_iv     = total_wall / 5
    scale      = 10 ** np.floor(np.log10(max(raw_iv, 1e-9)))
    interval   = max(np.ceil(raw_iv / scale) * scale, 1.0)
    wall_ticks = np.arange(0, total_wall + interval * 0.5, interval)
    step_pos   = np.interp(wall_ticks, wall_all, steps_all)

    def _fmt_min(m: float) -> str:
        if m >= 100 or abs(m - round(m)) < 0.05 * max(interval, 1.0):
            return f"{m:.0f}"
        return f"{m:g}"

    ax_loss.set_xticks(step_pos)
    ax_loss.set_xticklabels([_fmt_min(m) for m in wall_ticks])
    ax_loss.set_xlabel("Wall-clock time (min)")

    # -----------------------------------------------------------------------
    # Secondary x-axis (top): training step.
    # -----------------------------------------------------------------------
    ax_wall = ax_loss.twiny()
    ax_wall.set_xlim(ax_loss.get_xlim())
    ax_wall.set_xlabel("Training step", labelpad=5)
    ax_wall.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))

    # -----------------------------------------------------------------------
    # Title and combined legend
    # -----------------------------------------------------------------------
    ax_loss.set_title(
        r"Training profile  ($d{=}128,\;L{=}3$, hw,  $N{=}8\,192$)",
        pad=10,
    )

    handles = [
        Line2D([0], [0], color=C_LOSS, lw=2.0,
               label=r"$\mathcal{L}_{\cos}$ (mean $\pm$ 1$\sigma$, linear)"),
        Line2D([0], [0], color=C_SAI, lw=1.5, ls=(0, (5, 2)),
               marker="s", markersize=4.5, markerfacecolor="white",
               markeredgecolor=C_SAI, markeredgewidth=1.0,
               label="SAI loss (log)"),
        Line2D([0], [0], color=C_PCG, lw=1.8,
               marker="o", markersize=5.5, markerfacecolor="white",
               markeredgecolor=C_PCG, markeredgewidth=1.2,
               label="PCG iterations (log)"),
    ]
    # Legend below the axes, centred, three items in one row
    ax_loss.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        framealpha=0.88,
        edgecolor="#cccccc",
        handlelength=2.0,
        columnspacing=1.0,
    )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.out)
