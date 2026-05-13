#!/usr/bin/env python3
"""Render an orbit video around a residual-3d plot."""

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt

from MakeMultiphaseFigures import make_residual_3d


def _render_orbit(
    *,
    frame_dir: Path,
    ours_weights: Path,
    out_video: Path,
    n_frames: int,
    fps: int,
    k: int,
    cmap_name: str,
    render_mode: str,
    field: str,
    clean_view: bool,
    show_colorbar: bool,
    trace_seeds: int,
    trace_steps: int,
    trace_step_size: float,
    trace_sigma: float,
    alpha_power: float,
    alpha_scurve: bool,
    zoom: float,
    tight_image: bool,
    elev: float,
    azim_start: float,
    save_dpi: int,
):
    with tempfile.TemporaryDirectory(prefix="residual_3d_orbit_") as td:
        tmp = Path(td)
        # Build the 3D scene once, then only rotate the camera per frame.
        captured_figs = []
        orig_close = plt.close

        def _capture_close(fig=None):
            if fig is None:
                fig = plt.gcf()
            captured_figs.append(fig)

        plt.close = _capture_close
        try:
            make_residual_3d(
                str(frame_dir),
                str(tmp / "frame_init.png"),
                str(ours_weights),
                k=int(k),
                cmap_name=str(cmap_name),
                elev=float(elev),
                azim=float(azim_start),
                render_mode=str(render_mode),
                field=str(field),
                clean_view=bool(clean_view),
                show_colorbar=bool(show_colorbar),
                alpha_power=float(alpha_power),
                alpha_scurve=bool(alpha_scurve),
                trace_seeds=int(trace_seeds),
                trace_steps=int(trace_steps),
                trace_step_size=float(trace_step_size),
                trace_sigma=float(trace_sigma),
                zoom=float(zoom),
                tight_image=bool(tight_image),
            )
        finally:
            plt.close = orig_close

        if not captured_figs:
            raise RuntimeError("Failed to capture residual-3d scene figure for orbit rendering.")
        fig = captured_figs[-1]
        if not fig.axes:
            raise RuntimeError("Captured figure has no axes.")
        ax = fig.axes[0]

        t0 = time.perf_counter()
        for i in range(int(n_frames)):
            phase = i / max(1, int(n_frames) - 1)
            azim = float(azim_start) + 360.0 * phase
            ax.view_init(elev=float(elev), azim=float(azim))
            png = tmp / f"frame_{i:04d}.png"
            if tight_image:
                fig.savefig(png, dpi=int(save_dpi), bbox_inches="tight", pad_inches=0.0)
            else:
                fig.savefig(png, dpi=int(save_dpi), bbox_inches="tight")
            if (i + 1) % max(1, int(n_frames) // 12) == 0 or (i + 1) == int(n_frames):
                elapsed = time.perf_counter() - t0
                print(f"[orbit] rendered {i+1}/{n_frames} frames in {elapsed:.1f}s", flush=True)

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
        subprocess.run(cmd, check=True)
        print(f"[orbit] wrote {out_video}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frame-dir", required=True)
    p.add_argument("--ours-weights", required=True)
    p.add_argument("--out-video", required=True)

    # Match previous video length by default: 8s at 18 fps.
    p.add_argument("--video-frames", type=int, default=144)
    p.add_argument("--fps", type=int, default=18)

    # Residual-3d settings.
    p.add_argument("--residual-3d-k", type=int, default=1)
    p.add_argument("--residual-3d-cmap", default="viridis")
    p.add_argument("--residual-3d-render", choices=["scatter", "voxels", "traces"], default="traces")
    p.add_argument("--residual-3d-field", choices=["residual", "correction", "dual"], default="correction")
    p.add_argument("--residual-3d-clean", action="store_true")
    p.add_argument("--residual-3d-no-colorbar", action="store_true")
    p.add_argument("--residual-3d-trace-seeds", type=int, default=750)
    p.add_argument("--residual-3d-trace-steps", type=int, default=96)
    p.add_argument("--residual-3d-trace-step-size", type=float, default=0.7)
    p.add_argument("--residual-3d-trace-sigma", type=float, default=1.2)
    p.add_argument("--residual-3d-alpha-power", type=float, default=2.4)
    p.add_argument("--residual-3d-alpha-scurve", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--residual-3d-zoom", type=float, default=1.0)
    p.add_argument("--residual-3d-tight-image", action="store_true")
    p.add_argument("--residual-3d-elev", type=float, default=22.0)
    p.add_argument("--orbit-azim-start", type=float, default=-55.0)
    p.add_argument("--orbit-save-dpi", type=int, default=160, help="Per-frame save DPI (lower=faster).")
    args = p.parse_args()

    _render_orbit(
        frame_dir=Path(args.frame_dir).expanduser().resolve(),
        ours_weights=Path(args.ours_weights).expanduser().resolve(),
        out_video=Path(args.out_video).expanduser().resolve(),
        n_frames=int(args.video_frames),
        fps=int(args.fps),
        k=int(args.residual_3d_k),
        cmap_name=str(args.residual_3d_cmap),
        render_mode=str(args.residual_3d_render),
        field=str(args.residual_3d_field),
        clean_view=bool(args.residual_3d_clean),
        show_colorbar=not bool(args.residual_3d_no_colorbar),
        trace_seeds=int(args.residual_3d_trace_seeds),
        trace_steps=int(args.residual_3d_trace_steps),
        trace_step_size=float(args.residual_3d_trace_step_size),
        trace_sigma=float(args.residual_3d_trace_sigma),
        alpha_power=float(args.residual_3d_alpha_power),
        alpha_scurve=bool(args.residual_3d_alpha_scurve),
        zoom=float(args.residual_3d_zoom),
        tight_image=bool(args.residual_3d_tight_image),
        elev=float(args.residual_3d_elev),
        azim_start=float(args.orbit_azim_start),
        save_dpi=int(args.orbit_save_dpi),
    )


if __name__ == "__main__":
    main()

