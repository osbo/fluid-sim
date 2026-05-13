#!/usr/bin/env python3
"""Video: LeafOnly inverse construction by off-diagonal layers (2D).

Renders a zoomed-out, full-system view (e.g. 16384x16384) and progressively reveals
off-diagonal structure by increasing block-band distance from the diagonal.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from MakeMultiphaseFigures import OursPreconditioner, _apply_paper_style_compact, _resolve_frame, load_frame


def _build_block_strength_map(ours: OursPreconditioner) -> np.ndarray:
    """Build a zoomed-out leaf-block strength map for the full system."""
    from leafonly.architecture import unpack_precond
    from leafonly import hmatrix as _hm

    pre = ours._precond_s.to(device=ours.device).contiguous()
    jac = ours._jacobi_s.to(device=ours.device).contiguous()
    viz_n = int(ours._viz_n)
    leaf_L = int(ours._leaf_L)
    leaf_apply_diag_L = int(ours._leaf_apply_diag_L)
    leaf_apply_off_L = int(ours._leaf_apply_off_L)
    num_leaves = viz_n // leaf_L
    pool_diag_to_full = leaf_L // leaf_apply_diag_L

    diag_blocks, off_diag_blocks, node_U, node_V, jacobi_scale = unpack_precond(
        pre,
        viz_n,
        leaf_size=leaf_L,
        leaf_apply_size=leaf_apply_diag_L,
        leaf_apply_off=leaf_apply_off_L,
    )

    B = np.zeros((num_leaves, num_leaves), dtype=np.float64)
    for l in range(num_leaves):
        blk = diag_blocks[0, l]
        if pool_diag_to_full > 1:
            blk = blk.repeat_interleave(pool_diag_to_full, dim=0).repeat_interleave(pool_diag_to_full, dim=1)
        B[l, l] = float(torch.mean(torch.abs(blk)).item())

    if int(_hm.NUM_HMATRIX_OFF_BLOCKS) > 0 and off_diag_blocks is not None and node_U is not None and node_V is not None:
        # Reduce to one scalar strength per leaf (avoid vector-valued comparisons).
        u = torch.mean(torch.abs(node_U[0]), dim=tuple(range(1, node_U[0].dim()))).detach().cpu().numpy().astype(np.float64)
        v = torch.mean(torch.abs(node_V[0]), dim=tuple(range(1, node_V[0].dim()))).detach().cpu().numpy().astype(np.float64)
        for i in range(int(_hm.NUM_HMATRIX_OFF_BLOCKS)):
            r0i = int(_hm.HM_R0_CPU[i].item())
            c0i = int(_hm.HM_C0_CPU[i].item())
            si = int(_hm.HM_S_CPU[i].item())
            c_scale = float(torch.mean(torch.abs(off_diag_blocks[0, i])).item())
            for rr in range(si):
                gr = r0i + rr
                if not (0 <= gr < num_leaves):
                    continue
                for cc in range(si):
                    gc = c0i + cc
                    if not (0 <= gc < num_leaves):
                        continue
                    val = c_scale * u[gr] * v[gc]
                    if val > B[gr, gc]:
                        B[gr, gc] = val
                    if val > B[gc, gr]:
                        B[gc, gr] = val

    if jacobi_scale is not None:
        js = torch.abs(jacobi_scale[0] * jac[0]).detach().cpu().numpy().astype(np.float64)
        for l in range(num_leaves):
            a = l * leaf_L
            b = min((l + 1) * leaf_L, js.shape[0])
            if a < b:
                B[l, l] += float(np.mean(js[a:b]))
    return B


def _upsample_block_map(B: np.ndarray, out_size: int) -> np.ndarray:
    n = int(B.shape[0])
    out_size = int(max(out_size, n))
    rep = int(np.ceil(out_size / n))
    img = np.kron(B, np.ones((rep, rep), dtype=np.float64))
    return img[:out_size, :out_size]


def _build_video(
    *,
    frame_dir: Path,
    weights: Path,
    out_video: Path,
    n_frames: int,
    fps: int,
    render_size: int,
):
    _apply_paper_style_compact()
    info = load_frame(str(frame_dir))
    ours = OursPreconditioner(info, str(weights))
    viz_n = int(ours._viz_n)
    leaf_L = int(ours._leaf_L)
    num_leaves = viz_n // leaf_L
    render_size = int(max(256, int(render_size)))

    print(f"[leaf-build] frame={frame_dir} N={info['N']} ratio={info['ratio']:.1f}x")
    print(f"[leaf-build] full system view={viz_n}x{viz_n}, leaves={num_leaves}, leaf_size={leaf_L}")
    print("[leaf-build] assembling static LeafOnly block-strength map ...", flush=True)

    B_full = _build_block_strength_map(ours)
    pos = B_full[B_full > 0]
    if pos.size == 0:
        raise RuntimeError("LeafOnly block map has no positive entries.")
    vmin = max(float(np.percentile(pos, 1.0)), 1e-12)
    vmax = max(float(np.percentile(pos, 99.5)), vmin * 1.01)

    with tempfile.TemporaryDirectory(prefix="leafonly_inverse_build_") as td:
        tmp = Path(td)
        for fi in range(int(n_frames)):
            alpha = fi / max(1, int(n_frames) - 1)
            band = int(round(alpha * (num_leaves - 1)))
            idx = np.arange(num_leaves, dtype=np.int32)
            mask = np.abs(idx[:, None] - idx[None, :]) <= band
            B_layer = np.where(mask, B_full, 0.0)
            img = _upsample_block_map(B_layer, render_size)

            fig, ax = plt.subplots(1, 1, figsize=(10.5, 10.0), dpi=180, constrained_layout=False)
            fig.subplots_adjust(left=0.06, right=0.88, top=0.90, bottom=0.08)
            im = ax.imshow(
                img,
                cmap="magma",
                origin="lower",
                norm=LogNorm(vmin=vmin, vmax=vmax),
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                rf"LeafOnly $M$ (zoomed-out full view {viz_n}\times{viz_n})"
                "\n"
                rf"revealed block-band distance: {band}/{num_leaves - 1}",
                pad=10,
            )
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(r"$|M_{ij}|$ proxy")
            fig.suptitle(
                rf"Off-diagonal layer build  ($\rho_H/\rho_L\!\approx\!{info['ratio']:.0f}\times$)  progress={alpha:.2f}",
                y=0.97,
            )
            png = tmp / f"frame_{fi:04d}.png"
            fig.savefig(png, dpi=180)
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
        print("[leaf-build] encoding with ffmpeg ...")
        subprocess.run(cmd, check=True)
        print(f"[leaf-build] wrote {out_video}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(Path(__file__).resolve().parent / "data"))
    p.add_argument("--scale", type=int, default=16384)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--frame", type=int, default=73)
    p.add_argument("--frame-dir", default=None)
    p.add_argument(
        "--weights",
        default=str(Path(__file__).resolve().parent / "weights" / "v2_16384_d128_L3_hw.bytes"),
        help="LeafOnly .bytes checkpoint (same as fig 1b teaser).",
    )
    p.add_argument("--video-frames", type=int, default=144, help="Match previous 8s video at 18 fps.")
    p.add_argument("--fps", type=int, default=18)
    p.add_argument(
        "--render-size",
        type=int,
        default=1024,
        help="Rendered matrix image size for zoomed-out full-system view (e.g. 1024).",
    )
    p.add_argument(
        "--out-video",
        default=str(Path(__file__).resolve().parent / "results" / "leafonly_inverse_layer_build_16384.mp4"),
    )
    args = p.parse_args()

    frame_dir = (
        Path(args.frame_dir).expanduser().resolve()
        if args.frame_dir
        else Path(_resolve_frame(args.data_root, args.scale, args.split, args.frame)).resolve()
    )
    _build_video(
        frame_dir=frame_dir,
        weights=Path(args.weights).expanduser().resolve(),
        out_video=Path(args.out_video).expanduser().resolve(),
        n_frames=int(args.video_frames),
        fps=int(args.fps),
        render_size=int(args.render_size),
    )


if __name__ == "__main__":
    main()

