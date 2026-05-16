#!/usr/bin/env python3
"""
Parse LeafOnly training logs with --grad-mags and compare on-/off-diagonal gradient scales.

Default usage (for run_probe_smoothing_gradients_8192_engaging.sh outputs):
  python3 Assets/Scripts/plot_probe_smoothing_grad_balance.py
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STEP_RE = re.compile(r"^\s*(\d+):")
ELAPSED_RE = re.compile(r"\(([0-9.]+)s")
KV_RE = re.compile(r"\b([A-Za-z0-9_]+)=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")


def _sum_sq(prefix: str, metrics: dict[str, float]) -> float:
    return sum(v * v for k, v in metrics.items() if k.startswith(prefix))


def parse_grad_log(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m_step = STEP_RE.search(line)
        if not m_step:
            continue

        kvs = {k: float(v) for k, v in KV_RE.findall(line)}
        on_sq = _sum_sq("gDiag", kvs)
        off_sq = _sum_sq("gOffTf", kvs)

        # Include head-level terms so "on/off diagonal" reflects both stack + output blocks.
        if "gL" in kvs:
            on_sq += kvs["gL"] * kvs["gL"]
        if "gOffUV" in kvs:
            off_sq += kvs["gOffUV"] * kvs["gOffUV"]

        on = math.sqrt(on_sq)
        off = math.sqrt(off_sq)
        ratio = off / max(on, 1e-30)
        m_elapsed = ELAPSED_RE.search(line)
        elapsed = float(m_elapsed.group(1)) if m_elapsed else float("nan")

        rows.append(
            {
                "step": float(m_step.group(1)),
                "elapsed_s": elapsed,
                "on_diag_grad_l2": on,
                "off_diag_grad_l2": off,
                "off_over_on": ratio,
            }
        )
    return rows


def write_series_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["step", "elapsed_s", "on_diag_grad_l2", "off_diag_grad_l2", "off_over_on"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_combined_csv(
    path: Path,
    default_rows: list[dict[str, float]],
    zero_rows: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "step", "elapsed_s", "on_diag_grad_l2", "off_diag_grad_l2", "off_over_on"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in default_rows:
            w.writerow({"run": "default_smoothing", **r})
        for r in zero_rows:
            w.writerow({"run": "zero_smoothing", **r})


def summarize(rows: list[dict[str, float]], label: str) -> dict[str, str]:
    if not rows:
        return {
            "run": label,
            "n_points": "0",
            "start_step": "",
            "end_step": "",
            "off_over_on_start": "",
            "off_over_on_end": "",
            "off_over_on_max": "",
            "off_over_on_max_step": "",
        }
    arr_ratio = np.array([r["off_over_on"] for r in rows], dtype=float)
    arr_step = np.array([r["step"] for r in rows], dtype=float)
    imax = int(np.argmax(arr_ratio))
    return {
        "run": label,
        "n_points": str(len(rows)),
        "start_step": f"{arr_step[0]:.0f}",
        "end_step": f"{arr_step[-1]:.0f}",
        "off_over_on_start": f"{arr_ratio[0]:.6g}",
        "off_over_on_end": f"{arr_ratio[-1]:.6g}",
        "off_over_on_max": f"{arr_ratio[imax]:.6g}",
        "off_over_on_max_step": f"{arr_step[imax]:.0f}",
    }


def plot(
    default_rows: list[dict[str, float]],
    zero_rows: list[dict[str, float]],
    out_png: Path,
) -> None:
    def _xy(rows: list[dict[str, float]], ykey: str) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([r["step"] for r in rows], dtype=float)
        y = np.array([r[ykey] for r in rows], dtype=float)
        return x, y

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10.0, 7.2), sharex=True)

    x_d, y_d = _xy(default_rows, "off_over_on")
    x_z, y_z = _xy(zero_rows, "off_over_on")
    ax0.plot(x_d, y_d, color="#1f77b4", label="default smoothing")
    ax0.plot(x_z, y_z, color="#d62728", label="zero smoothing")
    ax0.set_ylabel("off / on grad L2")
    ax0.grid(True, alpha=0.3)
    ax0.legend()
    ax0.set_title("8192 baseline: gradient balance under probe smoothing")

    x_d, on_d = _xy(default_rows, "on_diag_grad_l2")
    _, off_d = _xy(default_rows, "off_diag_grad_l2")
    x_z, on_z = _xy(zero_rows, "on_diag_grad_l2")
    _, off_z = _xy(zero_rows, "off_diag_grad_l2")
    ax1.plot(x_d, on_d, color="#1f77b4", linestyle="-", label="on (default)")
    ax1.plot(x_d, off_d, color="#1f77b4", linestyle="--", label="off (default)")
    ax1.plot(x_z, on_z, color="#d62728", linestyle="-", label="on (zero)")
    ax1.plot(x_z, off_z, color="#d62728", linestyle="--", label="off (zero)")
    ax1.set_yscale("log")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("grad L2 (log)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(ncol=2)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    default_results = scripts_dir.parent / "results" / "probe_smoothing_8192"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--default-log",
        type=Path,
        default=default_results / "train_smooth_default.log",
    )
    p.add_argument(
        "--zero-log",
        type=Path,
        default=default_results / "train_smooth_zero.log",
    )
    p.add_argument(
        "--out-png",
        type=Path,
        default=default_results / "gradient_balance_8192.png",
    )
    p.add_argument(
        "--out-summary-csv",
        type=Path,
        default=default_results / "gradient_balance_8192_summary.csv",
    )
    args = p.parse_args()

    default_rows = parse_grad_log(args.default_log)
    zero_rows = parse_grad_log(args.zero_log)
    if not default_rows:
        raise SystemExit(f"No gradient rows parsed from {args.default_log}")
    if not zero_rows:
        raise SystemExit(f"No gradient rows parsed from {args.zero_log}")

    write_series_csv(args.out_summary_csv.with_name("gradient_balance_default_series.csv"), default_rows)
    write_series_csv(args.out_summary_csv.with_name("gradient_balance_zero_series.csv"), zero_rows)
    write_combined_csv(args.out_summary_csv.with_name("gradient_balance_combined_series.csv"), default_rows, zero_rows)

    summary_rows = [summarize(default_rows, "default_smoothing"), summarize(zero_rows, "zero_smoothing")]
    args.out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    plot(default_rows, zero_rows, args.out_png)
    print(f"Saved: {args.out_png}")
    print(f"Saved: {args.out_summary_csv}")


if __name__ == "__main__":
    main()
