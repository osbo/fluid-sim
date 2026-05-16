#!/usr/bin/env python3
"""Render a 2D residual-comparison video (2x4 grid, multiphase style).

Top row:
  Density rho | Input b | Pressure p | Velocity |v| + stream arrows

Bottom row (left->right):
  Jacobi | NeuralSPAI | AMG | Ours

Each method advances by normalized progress alpha in [0, 1], mapping to that method's own
iteration count with k = ceil(alpha * K_method), so all panels share the same wall-clock
video duration while stepping through iterations at different rates.
"""

import argparse
import csv
import math
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from MakeMultiphaseFigures import (
    NeuralSPAIPreconditioner,
    OursPreconditioner,
    _active_mask,
    _apply_paper_style_compact,
    _build_amg_apply,
    _build_jacobi_apply,
    _find_neural_spai_checkpoint,
    _pressure_velocity_fields,
    _resolve_frame,
    buoyancy_rhs,
    gridify,
    imshow_density,
    load_frame,
    pcg_preconditioned_full,
)

_RE_SETUP_CPU_GPU = re.compile(
    r"Setup:\s*([\d.]+)\s*ms,\s*solve\s*\((\w+)\):\s*([\d.]+)\s*ms,\s*(\d+)\s*iterations"
)
_RE_SETUP_SOLVE = re.compile(
    r"Setup:\s*([\d.]+)\s*ms,\s*solve:\s*([\d.]+)\s*ms,\s*(\d+)\s*iterations"
)
_RE_LEAFONLY = re.compile(
    r"Inference:\s*([\d.]+)\s*ms,\s*solve:\s*([\d.]+)\s*ms,\s*(\d+)\s*iterations"
)


def _run_method_with_timing(name, A, b, apply_M, *, rtol, max_iter):
    t0 = time.perf_counter()
    _, meta = pcg_preconditioned_full(
        A,
        b,
        apply_M,
        rtol=float(rtol),
        max_iter=int(max_iter),
        max_snapshots=8,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {"name": name, "A": A, "b": b, "apply": apply_M, "K": int(meta["iters"]), "solve_ms": elapsed_ms}


def _collect_snaps_for_video(method, n_frames, *, rtol, max_iter, full_n):
    K = max(1, int(method["K"]))
    alphas = np.linspace(0.0, 1.0, int(n_frames))
    ks = sorted({max(1, min(K, int(math.ceil(a * K)))) for a in alphas} | {1, K})
    snaps, _ = pcg_preconditioned_full(
        method["A"],
        method["b"],
        method["apply"],
        rtol=float(rtol),
        max_iter=int(max_iter),
        snap_iters_explicit=ks,
        max_snapshots=len(ks),
    )
    lifted = {}
    n_local = int(method["A"].shape[0])
    for k, r_k in snaps.items():
        rr = np.asarray(r_k, dtype=np.float64).ravel()
        if n_local == full_n:
            lifted[k] = rr
        else:
            out = np.zeros(full_n, dtype=np.float64)
            n_copy = min(full_n, n_local, rr.shape[0])
            out[:n_copy] = rr[:n_copy]
            lifted[k] = out
    method["ks_for_frames"] = [max(1, min(K, int(math.ceil(a * K)))) for a in alphas]
    method["snaps"] = lifted
    return method


def _infer_ours_effective_n(ours_prec, full_n: int) -> int:
    n_full = int(full_n)
    try:
        _ = ours_prec.apply_M(np.zeros(n_full, dtype=np.float64))
        return n_full
    except ValueError as exc:
        msg = str(exc)
        m = re.search(r"into shape \((\d+),\)", msg)
        if m:
            n = int(m.group(1))
            return max(1, min(n, n_full))
    n_attr = int(getattr(ours_prec, "_real_n", n_full))
    return max(1, min(n_attr, n_full))


def _sweep_reference_metrics(scale: int, config_name: str, sweep_csv: Path) -> dict:
    """Return Jacobi/AMG/Ours solve_ms from scale sweep CSV (same source as plots)."""
    try:
        with sweep_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}

    out = {}
    for row in rows:
        try:
            row_scale = int(float(row.get("scale", -1)))
        except Exception:
            continue
        if row_scale != int(scale):
            continue
        if row.get("config", "").strip() != str(config_name):
            continue
        method = row.get("method", "").strip()
        device = row.get("device", "").strip()
        try:
            solve_ms = float(row.get("solve_ms", ""))
        except Exception:
            continue
        if method == "jacobi" and device == "gpu":
            out["Jacobi"] = {"solve_ms": solve_ms}
        elif method == "amg_cpu" and device == "cpu":
            out["AMG"] = {"solve_ms": solve_ms}
        elif method == "leafonly" and device == "gpu":
            out["Ours"] = {"solve_ms": solve_ms}
    return out


def _neural_spai_reference_metrics(scale: int, summary_csv: Path) -> dict:
    """Return NeuralSPAI CUDA timing from lsp_scale_infer_summary.csv."""
    try:
        with summary_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}
    chosen = None
    for row in rows:
        if int(float(row.get("scale", -1))) != int(scale):
            continue
        if row.get("method", "").strip() == "Neural+CUDA":
            chosen = row
            break
        if chosen is None and row.get("method", "").strip() == "Neural":
            chosen = row
    if chosen is None:
        return {}
    try:
        return {
            "solve_ms": float(chosen.get("solve_time_ms", "")),
            "iters": int(float(chosen.get("iterations", ""))),
        }
    except Exception:
        return {}


def _render_video(
    *,
    frame_dir: Path,
    out_video: Path,
    ours_weights: Path,
    neuralspai_checkpoint: Path,
    lsp_repo: Path,
    scale: int,
    lsp_summary_csv: Path,
    scale_sweep_csv: Path,
    n_frames: int,
    fps: int,
    rtol: float,
    max_iter: int,
):
    _apply_paper_style_compact()
    info = load_frame(str(frame_dir))
    A = info["A"]
    b = buoyancy_rhs(info)
    full_n = int(info["N"])
    active = _active_mask(info)

    print(f"[video-2d] frame={frame_dir}  N={full_n}  ratio={info['ratio']:.1f}x")
    print("[video-2d] computing pressure + velocity context via AMG ...", flush=True)
    _, P_top, vx_top, vy_top, mag_top = _pressure_velocity_fields(info)

    print(f"[video-2d] loading Ours checkpoint: {ours_weights}", flush=True)
    ours = OursPreconditioner(info, str(ours_weights))
    ours_n = _infer_ours_effective_n(ours, full_n)
    print(f"[video-2d] Ours effective system size: {ours_n}", flush=True)
    A_ours = A[:ours_n, :ours_n].tocsr()
    b_ours = np.asarray(b[:ours_n], dtype=np.float64)

    print(f"[video-2d] loading NeuralSPAI checkpoint: {neuralspai_checkpoint}", flush=True)
    neural_spai = NeuralSPAIPreconditioner(info, str(neuralspai_checkpoint), lsp_repo=str(lsp_repo))

    methods = [
        _run_method_with_timing("Jacobi", A, b, _build_jacobi_apply(A), rtol=rtol, max_iter=max_iter),
        _run_method_with_timing("NeuralSPAI", A, b, neural_spai.apply_M, rtol=rtol, max_iter=max_iter),
        _run_method_with_timing("AMG", A, b, _build_amg_apply(A), rtol=rtol, max_iter=max_iter),
        _run_method_with_timing("Ours", A_ours, b_ours, ours.apply_M, rtol=rtol, max_iter=max_iter),
    ]
    for m in methods:
        print(f"[video-2d] {m['name']}: K={m['K']}  solve={m['solve_ms']:.2f} ms", flush=True)

    methods = [
        _collect_snaps_for_video(m, n_frames, rtol=rtol, max_iter=max_iter, full_n=full_n) for m in methods
    ]
    config_name = f"v2_{int(scale)}_d128_L3_hw"
    ref_metrics = _sweep_reference_metrics(int(scale), config_name, Path(scale_sweep_csv))
    ref_neural = _neural_spai_reference_metrics(int(scale), Path(lsp_summary_csv))
    if ref_neural:
        ref_metrics["NeuralSPAI"] = ref_neural
    if ref_metrics:
        print(f"[video-2d] reference timing source loaded for: {sorted(ref_metrics.keys())}", flush=True)
        print(f"[video-2d] timing CSVs: sweep={scale_sweep_csv}, lsp={lsp_summary_csv}", flush=True)

    all_pos = []
    for m in methods:
        for rr in m["snaps"].values():
            vals = np.abs(np.asarray(rr, dtype=np.float64).ravel())
            vals = vals[vals > 0]
            if vals.size:
                all_pos.append(vals)
    if not all_pos:
        raise RuntimeError("No positive residual magnitudes available for video rendering.")
    cat = np.concatenate(all_pos)
    r_vmin = max(float(np.percentile(cat, 0.5)), 1e-30)
    r_vmax = max(float(np.percentile(cat, 99.5)), r_vmin * 1.01)
    r_norm = LogNorm(vmin=r_vmin, vmax=r_vmax)

    b_grid = gridify(b, info)
    b_abs = max(float(np.percentile(np.abs(b_grid[active]), 99.0)) if active.any() else 1.0, 1e-12)
    p_abs = max(float(np.percentile(np.abs(P_top[active]), 99.0)) if active.any() else 1.0, 1e-12)
    v_hi = max(float(np.percentile(mag_top[active], 99.5)) if active.any() else 1e-8, 1e-12)
    v_pos = mag_top[active & (mag_top > 0)] if active.any() else np.array([], dtype=np.float64)
    v_lo = float(np.percentile(v_pos, 5.0)) if v_pos.size else (v_hi * 1e-3)
    v_lo = max(min(v_lo, v_hi * 0.5), v_hi * 1e-3)

    with tempfile.TemporaryDirectory(prefix="residual_compare_video_2d_") as td:
        tmp = Path(td)
        last_im = None
        for fi in range(int(n_frames)):
            alpha = fi / max(1, int(n_frames) - 1)
            fig, axes = plt.subplots(2, 4, figsize=(16, 8.3), dpi=180)
            fig.subplots_adjust(left=0.02, right=0.935, top=0.90, bottom=0.05, wspace=0.04, hspace=0.10)

            ax = axes[0, 0]
            imshow_density(ax, info, title=r"Density $\rho$")
            ax.set_box_aspect(1)

            ax = axes[0, 1]
            cmap_b = matplotlib.colormaps["RdBu_r"].copy()
            cmap_b.set_bad(color="lightgray")
            ax.imshow(
                np.where(active, b_grid, np.nan),
                origin="lower",
                cmap=cmap_b,
                vmin=-b_abs,
                vmax=b_abs,
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(r"Input $b$")
            ax.set_box_aspect(1)

            ax = axes[0, 2]
            cmap_p = matplotlib.colormaps["RdBu_r"].copy()
            cmap_p.set_bad(color="lightgray")
            ax.imshow(
                np.where(active, P_top, np.nan),
                origin="lower",
                cmap=cmap_p,
                vmin=-p_abs,
                vmax=p_abs,
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(r"Pressure $p$")
            ax.set_box_aspect(1)

            ax = axes[0, 3]
            cmap_v = matplotlib.colormaps["viridis"].copy()
            cmap_v.set_bad(color="lightgray")
            mag_masked = np.where(active, mag_top, np.nan)
            ax.imshow(
                mag_masked,
                origin="lower",
                cmap=cmap_v,
                norm=LogNorm(vmin=v_lo, vmax=v_hi),
                interpolation="nearest",
                aspect="equal",
            )
            H, W = vx_top.shape
            yg, xg = np.mgrid[0:H, 0:W]
            vx_safe = np.where(active, vx_top, 0.0)
            vy_safe = np.where(active, vy_top, 0.0)
            try:
                ax.streamplot(
                    xg,
                    yg,
                    vx_safe,
                    vy_safe,
                    color="white",
                    linewidth=0.65,
                    density=1.35,
                    arrowsize=0.7,
                )
            except Exception:
                pass
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(r"Velocity $|v|$ + arrows")
            ax.set_box_aspect(1)

            for ci, m in enumerate(methods):
                ax = axes[1, ci]
                k = int(m["ks_for_frames"][fi])
                rr = np.abs(np.asarray(m["snaps"][k], dtype=np.float64).ravel())
                rg = gridify(rr, info)
                rg = np.where(active, rg, np.nan)
                cmap_r = matplotlib.colormaps["magma"].copy()
                cmap_r.set_bad(color="lightgray")
                last_im = ax.imshow(
                    rg,
                    origin="lower",
                    cmap=cmap_r,
                    norm=r_norm,
                    interpolation="nearest",
                    aspect="equal",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_box_aspect(1)
                ax.set_title(
                    f"{m['name']}  k={k}/{m['K']}  t={ref_metrics.get(m['name'], {}).get('solve_ms', m['solve_ms']):.1f} ms",
                    fontsize=matplotlib.rcParams["axes.titlesize"],
                )

            if last_im is not None:
                cb = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.020, pad=0.010)
                cb.set_label(r"$|r_k|$")

            fig.suptitle(
                rf"2D residual comparison  ($\rho_H/\rho_L\!\approx\!{info['ratio']:.0f}\times$)  "
                rf"progress={alpha:.2f}",
                y=0.955,
            )
            png_path = tmp / f"frame_{fi:04d}.png"
            fig.savefig(png_path, dpi=180)
            plt.close(fig)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise SystemExit("ffmpeg not found on PATH.")
        out_video.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(int(fps)),
            "-i",
            str(tmp / "frame_%04d.png"),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_video),
        ]
        print("[video-2d] encoding with ffmpeg ...", flush=True)
        subprocess.run(cmd, check=True)
        print(f"[video-2d] wrote {out_video}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(Path(__file__).resolve().parent.parent / "data"))
    p.add_argument("--scale", type=int, default=16384, help="Use multiphase_v2_<scale> for 2D frame selection.")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--frame", type=int, default=73)
    p.add_argument("--frame-dir", default=None, help="Explicit 2D frame directory override.")
    p.add_argument("--ours-weights", required=True, help="Path to Ours .bytes checkpoint.")
    p.add_argument("--neuralspai-checkpoint", default=None, help="Optional explicit NeuralSPAI checkpoint path.")
    p.add_argument(
        "--lsp-summary-csv",
        default=str(Path(__file__).resolve().parent.parent / "results" / "lsp_scale_infer_summary.csv"),
        help="Used to auto-resolve NeuralSPAI checkpoint by --scale.",
    )
    p.add_argument(
        "--scale-sweep-csv",
        default=str(Path(__file__).resolve().parent.parent / "results" / "scale_sweep_task_004.csv"),
        help="Scale-sweep CSV for Jacobi/AMG/Ours timing labels (same source as scale sweep plots).",
    )
    p.add_argument("--lsp-repo", default="/orcd/home/002/osbo/ondemand/LearningSparsePreconditioner4GPU")
    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--max-iter", type=int, default=6000)
    p.add_argument("--video-frames", type=int, default=144, help="Default doubled vs previous (72 -> 144).")
    p.add_argument("--fps", type=int, default=18)
    p.add_argument(
        "--out-video",
        default=str(Path(__file__).resolve().parent.parent / "results" / "residual_compare_2d_jacobi_neuralspai_amg_ours.mp4"),
    )
    args = p.parse_args()

    frame_dir = (
        Path(args.frame_dir).expanduser().resolve()
        if args.frame_dir
        else Path(_resolve_frame(args.data_root, args.scale, args.split, args.frame)).resolve()
    )
    neural_ckpt = args.neuralspai_checkpoint
    if neural_ckpt is None:
        neural_ckpt = _find_neural_spai_checkpoint(int(args.scale), Path(args.lsp_summary_csv))
        print(f"[video-2d] auto-resolved NeuralSPAI checkpoint: {neural_ckpt}")

    _render_video(
        frame_dir=frame_dir,
        out_video=Path(args.out_video).expanduser().resolve(),
        ours_weights=Path(args.ours_weights).expanduser().resolve(),
        neuralspai_checkpoint=Path(neural_ckpt).expanduser().resolve(),
        lsp_repo=Path(args.lsp_repo).expanduser().resolve(),
        scale=int(args.scale),
        lsp_summary_csv=Path(args.lsp_summary_csv).expanduser().resolve(),
        scale_sweep_csv=Path(args.scale_sweep_csv).expanduser().resolve(),
        n_frames=int(args.video_frames),
        fps=int(args.fps),
        rtol=float(args.rtol),
        max_iter=int(args.max_iter),
    )


if __name__ == "__main__":
    main()
