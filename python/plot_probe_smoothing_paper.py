#!/usr/bin/env python3
"""Paper-style probe smoothing figure: smoothed on/off ratio + checkpoint PCG iterations."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


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


def _normalized_progress(step: pd.Series) -> pd.Series:
    s0 = float(step.min())
    s1 = float(step.max())
    if s1 <= s0:
        return pd.Series(np.zeros(len(step), dtype=float), index=step.index)
    return (step - s0) / (s1 - s0)


def _smooth_on_over_off(on_over_off: pd.Series, window: int) -> pd.Series:
    # Smooth in log-space so multiplicative spikes are handled robustly.
    safe = np.maximum(on_over_off.to_numpy(dtype=float), 1e-12)
    logv = np.log10(safe)
    ser = pd.Series(logv, index=on_over_off.index)
    med = ser.rolling(window=window, min_periods=1, center=True).median()
    avg = med.rolling(window=window, min_periods=1, center=True).mean()
    return pd.Series(np.power(10.0, avg.to_numpy(dtype=float)), index=on_over_off.index)


def _smooth_linear(v: pd.Series, window: int) -> pd.Series:
    ser = pd.Series(v.to_numpy(dtype=float), index=v.index)
    med = ser.rolling(window=window, min_periods=1, center=True).median()
    avg = med.rolling(window=window, min_periods=1, center=True).mean()
    return avg


STEP_RE = re.compile(r"^\s*(\d+):")
KV_RE = re.compile(r"\b([A-Za-z0-9_]+)=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")


def _parse_group_l1_from_log(log_path: Path, run_name: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        m_step = STEP_RE.search(line)
        if not m_step:
            continue
        kv = {k: float(v) for k, v in KV_RE.findall(line)}
        on_sum = 0.0
        off_sum = 0.0
        for k, v in kv.items():
            if k.startswith("gDiag"):
                on_sum += abs(v)
            elif k.startswith("gOffTf"):
                off_sum += abs(v)
        on_sum += abs(kv.get("gL", 0.0))
        off_sum += abs(kv.get("gOffUV", 0.0))
        if on_sum <= 0.0 and off_sum <= 0.0:
            continue
        rows.append(
            {
                "run": run_name,
                "step": float(m_step.group(1)),
                "on_group_l1_sum": on_sum,
                "off_group_l1_sum": off_sum,
                "on_over_off_l1": on_sum / max(off_sum, 1e-12),
                "off_over_on_l1": off_sum / max(on_sum, 1e-12),
                "off_share_l1": off_sum / max(on_sum + off_sum, 1e-12),
                "gTot": abs(kv.get("gTot", np.nan)),
            }
        )
    if not rows:
        raise ValueError(f"No gradient-group rows parsed from {log_path}")
    return pd.DataFrame(rows)


def _metric_config(metric: str) -> tuple[str, str, bool]:
    # returns (column_name, y_label, use_log_scale)
    if metric == "on_over_off_l1":
        return "on_over_off_l1", "on / off grad magnitude (group-L1)", True
    if metric == "off_over_on_l1":
        return "off_over_on_l1", "off / on grad magnitude (group-L1)", True
    if metric == "off_share_l1":
        return "off_share_l1", "off share of grad magnitude (group-L1)", False
    if metric == "gTot":
        return "gTot", "total gradient magnitude gTot", True
    if metric == "loss_avg_profile":
        return "loss_avg", "training loss avg", False
    raise ValueError(f"Unsupported metric: {metric}")


def _checkpoint_points(profile_csv: Path) -> pd.DataFrame:
    prof = pd.read_csv(profile_csv)
    if "pcg_iters" not in prof.columns:
        raise ValueError(
            f"{profile_csv} is missing 'pcg_iters'. Run EvalCheckpoints.py first to append checkpoint PCG iterations."
        )
    prof["step"] = pd.to_numeric(prof["step"], errors="coerce")
    prof["pcg_iters"] = pd.to_numeric(prof["pcg_iters"], errors="coerce")
    pts = prof.dropna(subset=["step", "pcg_iters"]).copy()
    pts["progress"] = _normalized_progress(pts["step"])
    return pts


def _step_ticks_for_axis(step_min: float, step_max: float, n: int = 5) -> tuple[np.ndarray, list[str]]:
    p = np.linspace(0.0, 1.0, n)
    steps = step_min + p * (step_max - step_min)
    labels = [f"{int(round(s / 100.0) * 100):d}" for s in steps]
    return p, labels


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    results_dir = scripts_dir.parent / "results" / "probe_smoothing_8192"
    paper_fig_dir = scripts_dir.parent.parent / "Paper" / "figures"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--default-log", type=Path, default=results_dir / "train_smooth_default.log")
    p.add_argument("--zero-log", type=Path, default=results_dir / "train_smooth_zero.log")
    p.add_argument(
        "--default-profile-csv",
        type=Path,
        default=scripts_dir.parent / "weights" / "v2_8192_d128_L3_hw_smooth_default_training_profile.csv",
    )
    p.add_argument(
        "--zero-profile-csv",
        type=Path,
        default=scripts_dir.parent / "weights" / "v2_8192_d128_L3_hw_smooth_zero_training_profile.csv",
    )
    p.add_argument(
        "--out-png",
        type=Path,
        default=paper_fig_dir / "probe_smoothing_ratio_and_pcg.png",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=results_dir / "probe_smoothing_ratio_and_pcg_plot_data.csv",
    )
    p.add_argument("--smooth-window", type=int, default=11)
    p.add_argument(
        "--metric",
        choices=["on_over_off_l1", "off_over_on_l1", "off_share_l1", "gTot", "loss_avg_profile"],
        default="loss_avg_profile",
        help="Gradient summary metric used in the top panel.",
    )
    args = p.parse_args()

    _apply_style()

    col_metric, ylabel_metric, metric_log = _metric_config(args.metric)

    df_log = pd.concat(
        [
            _parse_group_l1_from_log(args.default_log, "default_smoothing"),
            _parse_group_l1_from_log(args.zero_log, "zero_smoothing"),
        ],
        ignore_index=True,
    )

    runs = {"default_smoothing": "#1f77b4", "zero_smoothing": "#d62728"}
    labels = {"default_smoothing": "default smoothing", "zero_smoothing": "zero smoothing"}

    ratio_rows: list[pd.DataFrame] = []
    step_ranges: dict[str, tuple[float, float]] = {}
    if args.metric == "loss_avg_profile":
        for run, prof_csv in [
            ("default_smoothing", args.default_profile_csv),
            ("zero_smoothing", args.zero_profile_csv),
        ]:
            prof = pd.read_csv(prof_csv)
            prof["step"] = pd.to_numeric(prof["step"], errors="coerce")
            prof["loss_avg"] = pd.to_numeric(prof["loss_avg"], errors="coerce")
            sub = prof.dropna(subset=["step", "loss_avg"]).sort_values("step").copy()
            sub["run"] = run
            sub["progress"] = _normalized_progress(sub["step"])
            sub["metric_raw"] = sub["loss_avg"]
            sub["metric_smooth"] = _smooth_linear(sub["metric_raw"], max(3, int(args.smooth_window)))
            ratio_rows.append(sub)
            step_ranges[run] = (float(sub["step"].min()), float(sub["step"].max()))
    else:
        for run in runs:
            sub = df_log[df_log["run"] == run].sort_values("step").copy()
            if sub.empty:
                continue
            sub["progress"] = _normalized_progress(sub["step"])
            sub["metric_raw"] = sub[col_metric]
            if metric_log:
                sub["metric_smooth"] = _smooth_on_over_off(sub["metric_raw"], max(3, int(args.smooth_window)))
            else:
                sub["metric_smooth"] = _smooth_linear(sub["metric_raw"], max(3, int(args.smooth_window)))
            ratio_rows.append(sub)
            step_ranges[run] = (float(sub["step"].min()), float(sub["step"].max()))

    if len(ratio_rows) != 2:
        raise ValueError("Expected both default_smoothing and zero_smoothing in combined CSV.")
    ratio_df = pd.concat(ratio_rows, ignore_index=True)

    ckpt_default = _checkpoint_points(args.default_profile_csv)
    ckpt_zero = _checkpoint_points(args.zero_profile_csv)
    ckpt_default["run"] = "default_smoothing"
    ckpt_zero["run"] = "zero_smoothing"
    ckpt = pd.concat([ckpt_default, ckpt_zero], ignore_index=True)

    fig, (ax_ratio, ax_pcg) = plt.subplots(
        2,
        1,
        figsize=(7.6, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.16},
    )

    for run, color in runs.items():
        sub = ratio_df[ratio_df["run"] == run].sort_values("progress")
        ax_ratio.plot(
            sub["progress"],
            sub["metric_smooth"],
            color=color,
            label=labels[run],
        )
    if metric_log:
        ax_ratio.set_yscale("log")
    ax_ratio.set_ylabel(ylabel_metric)
    ax_ratio.set_title("Probe smoothing effect through training progress")
    ax_ratio.grid(True, which="major", color="#d8d8d8")
    ax_ratio.grid(True, which="minor", color="#efefef")
    ax_ratio.legend(loc="upper right", framealpha=0.9, edgecolor="#cccccc")

    for run, color in runs.items():
        sub = ckpt[ckpt["run"] == run].sort_values("progress")
        ax_pcg.plot(
            sub["progress"],
            sub["pcg_iters"],
            color=color,
            marker="o",
            markersize=4.0,
            linestyle="-",
            label=labels[run],
        )
    ax_pcg.set_ylabel("PCG iterations")
    ax_pcg.set_yscale("log")
    ax_pcg.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax_pcg.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(1, 10, dtype=float)))
    ax_pcg.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    ax_pcg.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_pcg.set_xlabel("normalized training progress (start -> end)")
    ax_pcg.grid(True, which="major", color="#d8d8d8")
    ax_pcg.grid(True, which="minor", color="#efefef")
    ax_pcg.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax_pcg.legend(loc="upper right", framealpha=0.9, edgecolor="#cccccc")

    ax_ratio.set_xlim(0.0, 1.0)
    ax_pcg.set_xticks(np.linspace(0.0, 1.0, 6))
    ax_pcg.set_xticklabels([f"{x:.1f}" for x in np.linspace(0.0, 1.0, 6)])

    # Top axis 1: default-smoothing step scale.
    ax_top_default = ax_ratio.twiny()
    ax_top_default.set_xlim(ax_ratio.get_xlim())
    p_ticks, default_step_labels = _step_ticks_for_axis(*step_ranges["default_smoothing"], n=6)
    ax_top_default.set_xticks(p_ticks)
    ax_top_default.set_xticklabels(default_step_labels)
    ax_top_default.tick_params(axis="x", colors=runs["default_smoothing"], pad=1)
    ax_top_default.spines["top"].set_color(runs["default_smoothing"])
    ax_top_default.set_xlabel("default-smoothing training step", color=runs["default_smoothing"])

    # Top axis 2: zero-smoothing step scale (offset upward).
    ax_top_zero = ax_ratio.twiny()
    ax_top_zero.set_xlim(ax_ratio.get_xlim())
    ax_top_zero.spines["top"].set_position(("outward", 22))
    p_ticks2, zero_step_labels = _step_ticks_for_axis(*step_ranges["zero_smoothing"], n=6)
    ax_top_zero.set_xticks(p_ticks2)
    ax_top_zero.set_xticklabels(zero_step_labels)
    ax_top_zero.tick_params(axis="x", colors=runs["zero_smoothing"], pad=1)
    ax_top_zero.spines["top"].set_color(runs["zero_smoothing"])
    ax_top_zero.set_xlabel("zero-smoothing training step", color=runs["zero_smoothing"])

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, bbox_inches="tight")
    plt.close(fig)

    export = ratio_df[
        [
            "run",
            "step",
            "progress",
            "metric_raw",
            "metric_smooth",
        ]
    ].copy()
    if "on_group_l1_sum" in ratio_df.columns:
        export["on_group_l1_sum"] = ratio_df["on_group_l1_sum"]
    if "off_group_l1_sum" in ratio_df.columns:
        export["off_group_l1_sum"] = ratio_df["off_group_l1_sum"]
    if "on_over_off_l1" in ratio_df.columns:
        export["on_over_off_l1"] = ratio_df["on_over_off_l1"]
    if "off_over_on_l1" in ratio_df.columns:
        export["off_over_on_l1"] = ratio_df["off_over_on_l1"]
    if "off_share_l1" in ratio_df.columns:
        export["off_share_l1"] = ratio_df["off_share_l1"]
    if "gTot" in ratio_df.columns:
        export["gTot"] = ratio_df["gTot"]
    export["pcg_iters"] = np.nan
    for run in runs:
        pts = ckpt[ckpt["run"] == run][["step", "pcg_iters"]].copy()
        pts["run"] = run
        export = export.merge(pts, on=["run", "step"], how="left", suffixes=("", "_ckpt"))
        export["pcg_iters"] = export["pcg_iters"].fillna(export["pcg_iters_ckpt"])
        export = export.drop(columns=["pcg_iters_ckpt"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(args.out_csv, index=False)

    print(f"Saved: {args.out_png}")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
