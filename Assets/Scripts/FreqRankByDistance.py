"""
FreqRankByDistance: visualize how off-diagonal blocks of A^{-1} change with
their distance from the diagonal in the H-matrix partition. Two panels:

  (A) Radially-averaged 2D Fourier power spectrum of A^{-1} blocks, grouped by
      tile size S (in leaf units). Near tiles concentrate energy at high
      relative frequencies; distant tiles concentrate at low frequencies.

  (B) Singular-value decay of A^{-1} blocks, grouped by tile size S, normalized
      by sigma_1 per block then averaged. A vertical reference at k = L_s
      shows the rank the architecture provides per tile, regardless of S.

Run with the fluid conda env, e.g.
  /orcd/home/002/osbo/.conda/envs/fluid/bin/python FreqRankByDistance.py
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from leafonly.config import HMATRIX_ETA, LEAF_SIZE, LEAF_APPLY_SIZE_OFF, problem_padded_num_nodes
from leafonly.data import FluidGraphDataset
from leafonly.hmatrix import standard_admissible_unique_blocks


def load_dense_A(data_folder: Path, frame_idx: int):
    dataset = FluidGraphDataset([data_folder])
    if len(dataset) == 0:
        raise SystemExit(f"No frames found under {data_folder}")
    frame_idx = min(frame_idx, len(dataset) - 1)
    batch = dataset[frame_idx]
    n_real = int(batch["num_nodes"])
    n = problem_padded_num_nodes(n_real)
    ei = batch["edge_index"].numpy()
    ev = batch["edge_values"].numpy()
    mask = (ei[0] < n) & (ei[1] < n)
    A = np.zeros((n, n), dtype=np.float64)
    A[ei[0, mask], ei[1, mask]] = ev[mask]
    # Pad any unused diagonal entries so the inversion is well-defined; the
    # data loader already pads by zero so unused rows have zero diagonal.
    diag = np.diag(A)
    bad = diag <= 0
    if bad.any():
        idx = np.where(bad)[0]
        A[idx, idx] = 1.0
    return A, n_real, n


def radial_power_density(block: np.ndarray, edges: np.ndarray):
    """2D FFT, |.|^2 / total_energy, then sum power into shared absolute-frequency
    edges (cycles per node). Returns power *density* per unit radial frequency
    so blocks of different sizes are directly comparable.
    """
    h, w = block.shape
    F = np.fft.fft2(block)
    P = np.abs(F) ** 2
    total = float(P.sum())
    if total <= 0:
        return np.zeros(len(edges) - 1)
    P = P / total
    fy = np.fft.fftfreq(h)  # cycles per sample == cycles per node here
    fx = np.fft.fftfreq(w)
    fy2, fx2 = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(fy2 ** 2 + fx2 ** 2)
    n_bins = len(edges) - 1
    bin_idx = np.digitize(radius, edges) - 1
    in_range = (bin_idx >= 0) & (bin_idx < n_bins)
    sums = np.bincount(bin_idx[in_range].ravel(),
                       weights=P[in_range].ravel(),
                       minlength=n_bins)[:n_bins]
    widths = (edges[1:] - edges[:-1])
    return sums / widths  # power density (probability mass / bin width)


def main():
    parser = argparse.ArgumentParser()
    default_data = _script_dir / "data" / "multiphase_v2_8192" / "test"
    parser.add_argument("--data-folder", type=str, default=str(default_data))
    parser.add_argument("--frames", type=int, default=4,
                        help="Number of frames to average over.")
    parser.add_argument("--leaf-size", type=int, default=LEAF_SIZE)
    parser.add_argument("--ls", type=int, default=LEAF_APPLY_SIZE_OFF,
                        help="Per-tile rank L_s the architecture emits.")
    parser.add_argument("--n-freq-bins", type=int, default=28)
    parser.add_argument("--cache", type=str, default=str(_script_dir / ".freq_rank_cache.npz"),
                        help="Cache for per-tile spectra/SVDs so plot iterations are cheap.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--output", type=str,
                        default=str(_script_dir.parent.parent / "Paper" / "figures" / "rank_provided_vs_needed.png"))
    parser.add_argument("--multi-panel", action="store_true",
                        help="Also emit the wide three-panel debugging figure to <output>_3panel.png")
    args = parser.parse_args()

    data_folder = Path(args.data_folder)
    leaf_L = int(args.leaf_size)
    L_s = int(args.ls)

    # Shared absolute frequency grid (cycles per node), log-spaced from a tiny
    # value to Nyquist=0.5. The lowest absolute frequency a block of side T can
    # resolve is 1/T, so larger blocks contribute to lower bins.
    edges = np.geomspace(1e-4, 0.5, args.n_freq_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    n_bins = args.n_freq_bins

    spec_acc = defaultdict(lambda: np.zeros(n_bins))   # S -> sum of densities
    spec_cnt = defaultdict(int)                         # S -> #tiles seen
    sv_acc = defaultdict(list)                          # S -> list of sv arrays (each length S*leaf_L)
    rank_eps_acc = defaultdict(lambda: defaultdict(list))  # S -> {eps: [ranks]}
    EPS_LIST = (1e-3, 1e-6, 1e-9)

    cache_path = Path(args.cache)
    cache_key = (args.frames, leaf_L, args.n_freq_bins, str(data_folder.resolve()))
    cache_loaded = False
    if cache_path.exists() and not args.rebuild_cache and not args.no_cache:
        try:
            cdata = np.load(cache_path, allow_pickle=True)
            stored = tuple(cdata["key"].tolist())
            if stored == cache_key:
                S_vals_cached = cdata["S_values"].tolist()
                edges = cdata["edges"]
                centers = np.sqrt(edges[:-1] * edges[1:])
                n_bins = len(centers)
                for S in S_vals_cached:
                    S = int(S)
                    spec_acc[S] = cdata[f"spec_{S}"]
                    spec_cnt[S] = int(cdata[f"cnt_{S}"])
                    svs = cdata[f"sv_{S}"]
                    sv_acc[S] = [svs[i] for i in range(svs.shape[0])]
                    for eps in EPS_LIST:
                        ranks = (svs > eps).sum(axis=1)
                        rank_eps_acc[S][eps] = ranks.tolist()
                cache_loaded = True
                print(f"Loaded cached spectra/SVDs from {cache_path} (key matches).")
        except Exception as ex:
            print(f"Cache read failed ({ex}); recomputing.")

    if not cache_loaded:
        for f_idx in range(args.frames):
            print(f"\nFrame {f_idx}: loading + inverting...")
            A, n_real, n = load_dense_A(data_folder, f_idx)
            if f_idx == 0:
                print(f"  N_real={n_real}, N_padded={n}, leaf_size={leaf_L}, L_s={L_s}")
            A_inv = np.linalg.inv(A)
            del A

            num_units = n // leaf_L
            blocks_t = standard_admissible_unique_blocks(
                num_units, float(HMATRIX_ETA), torch.device("cpu"), dtype=torch.float32
            )
            blocks = blocks_t.numpy()
            off_mask = (blocks[:, 0] != blocks[:, 1])
            upper_mask = (blocks[:, 0] <= blocks[:, 1])
            tiles = blocks[off_mask & upper_mask]

            for r_leaf, c_leaf, S in tiles:
                S = int(S)
                r0, c0 = int(r_leaf) * leaf_L, int(c_leaf) * leaf_L
                sz = S * leaf_L
                if r0 + sz > n or c0 + sz > n:
                    continue
                B = A_inv[r0:r0 + sz, c0:c0 + sz]
                spec_acc[S] += radial_power_density(B, edges)
                spec_cnt[S] += 1
                s = np.linalg.svd(B, compute_uv=False)
                s = s / (s[0] + 1e-300)
                sv_acc[S].append(s)
                for eps in EPS_LIST:
                    rank_eps_acc[S][eps].append(int(np.sum(s > eps)))
            del A_inv

        if not args.no_cache:
            payload = {"key": np.array(list(cache_key), dtype=object),
                       "edges": edges,
                       "S_values": np.array(sorted(spec_acc.keys()), dtype=np.int64)}
            for S in spec_acc:
                payload[f"spec_{S}"] = spec_acc[S]
                payload[f"cnt_{S}"] = spec_cnt[S]
                payload[f"sv_{S}"] = np.stack(sv_acc[S], axis=0)
            np.savez(cache_path, **payload)
            print(f"Cached spectra/SVDs to {cache_path}")

    S_values = sorted(spec_acc.keys())
    print(f"\ndistance classes (S in leaves): {S_values}")
    spectra = {}
    svals = {}
    rank_eps = {}
    for S in S_values:
        spectra[S] = spec_acc[S] / max(1, spec_cnt[S])
        sv = np.stack(sv_acc[S], axis=0)
        svals[S] = sv.mean(axis=0)
        rank_eps[S] = {eps: (float(np.mean(rank_eps_acc[S][eps])),
                              float(np.std(rank_eps_acc[S][eps])))
                       for eps in EPS_LIST}
        rs = ", ".join(f"@{eps:.0e}: {rank_eps[S][eps][0]:.1f}" for eps in EPS_LIST)
        print(f"  S={S} ({S*leaf_L}x{S*leaf_L}, {spec_cnt[S]} tiles): "
              f"L_s={L_s} of {S*leaf_L} provided ({100.0*L_s/(S*leaf_L):.2f}% rank-fraction); rank {rs}")

    # ---- Plot (style mirrors plot_eval_results.py / plot_training.py) ----
    mpl.rcParams.update({
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
    })
    GRID_MAJOR = "#d8d8d8"
    GRID_MINOR = "#efefef"
    LEGEND_EDGE = "#cccccc"
    REF_LINE = "#555555"
    HALO = [path_effects.withStroke(linewidth=2.4, foreground="white", alpha=1.0)]
    cmap = plt.get_cmap("viridis")
    n_S = len(S_values)
    colors = [cmap(0.10 + 0.70 * i / max(1, n_S - 1)) for i in range(n_S)]

    def _grid(ax):
        ax.grid(True, which="major", color=GRID_MAJOR, zorder=0)
        ax.grid(True, which="minor", color=GRID_MINOR, zorder=0)
        ax.set_axisbelow(True)

    # ---- Single-panel figure (sized to match scale_methods.png / training.png:
    # ~6.5"x4.0" so that when LaTeX scales it to \linewidth, fonts/markers end
    # up the same physical size as in the rest of the paper). ----
    fig_c, axC = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)

    S_arr = np.array(S_values, dtype=float)
    block_dim = S_arr * leaf_L  # tile side length in NODES
    provided_frac = L_s / block_dim
    eps_styles = [
        (1e-3, "s--", "#1f9d4d"),
        (1e-6, "^--", "#197aa6"),
        (1e-9, "d--", "#3b1da6"),
    ]
    needed_means = {}
    for eps, marker, color in eps_styles:
        means = np.array([rank_eps[S][eps][0] for S in S_values])
        stds = np.array([rank_eps[S][eps][1] for S in S_values])
        frac = means / block_dim
        frac_lo = np.maximum(means - stds, 0.5) / block_dim
        frac_hi = (means + stds) / block_dim
        axC.loglog(block_dim, frac, marker, color=color, lw=1.6, markersize=5.5, zorder=3)
        axC.fill_between(block_dim, frac_lo, frac_hi, color=color, alpha=0.16, zorder=2)
        needed_means[eps] = frac

    axC.loglog(block_dim, provided_frac, "o-", color="#b03030", lw=2.2, markersize=6.5, zorder=4)
    axC.fill_between(block_dim, needed_means[1e-9], provided_frac,
                     where=(provided_frac > needed_means[1e-9]),
                     color="#b03030", alpha=0.07, zorder=1)

    def _ilbl(x, y, text, color, ha="left", va="center", weight="normal", size=9.0):
        t = axC.text(x, y, text, color=color, fontsize=size, ha=ha, va=va,
                     zorder=6, fontweight=weight)
        t.set_path_effects(HALO)
        return t

    # Curve labels stacked at the LEFT edge (above their first marker).
    x_lbl = block_dim[0] * 1.12
    _ilbl(x_lbl, provided_frac[0] * 1.20,
          r"provided rank: $L_s\,{=}\,32$  (constant, regardless of tile size)",
          "#b03030", ha="left", va="bottom", weight="bold")
    _ilbl(x_lbl, needed_means[1e-9][0] * 1.20,
          r"rank required to capture $A^{-1}$ tile to  $\epsilon\,{=}\,10^{-9}$",
          "#3b1da6", ha="left", va="bottom")
    _ilbl(x_lbl, needed_means[1e-6][0] * 1.32,
          r"$\dots$ to  $\epsilon\,{=}\,10^{-6}$", "#197aa6", ha="left", va="bottom")
    _ilbl(x_lbl, needed_means[1e-3][0] * 0.55,
          r"$\dots$ to  $\epsilon\,{=}\,10^{-3}$", "#1f9d4d", ha="left", va="top")

    # Headroom callout in the right-side gap (S=8 in default config).
    s_idx = len(S_arr) - 2
    hr_mid = provided_frac[s_idx] / max(needed_means[1e-9][s_idx], 1e-30)
    y_mid = np.sqrt(provided_frac[s_idx] * needed_means[1e-9][s_idx])
    t = axC.text(block_dim[s_idx], y_mid, f"{hr_mid:.1f}$\\times$ headroom",
                 color="#1a1a1a", fontsize=9.5, ha="center", va="center", zorder=6,
                 bbox=dict(boxstyle="round,pad=0.26", fc="white",
                           ec="#b03030", lw=0.8, alpha=0.95))
    t.set_path_effects(HALO)

    # Note explaining the shaded bands.
    axC.text(0.985, 0.025,
             r"shaded bands: $\pm\,1\sigma$ across tiles in the same size class",
             transform=axC.transAxes, fontsize=8, color="#555555",
             ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.20", fc="white",
                       ec=LEGEND_EDGE, lw=0.5, alpha=0.92))

    axC.set_xlabel(r"Off-diagonal tile size  (nodes per side, $\,=S\!\cdot\!L$ with leaf $L\,{=}\,128$)")
    axC.set_ylabel(r"Numerical rank, as fraction of tile size")
    axC.set_title(r"Off-diagonal rank: emitted by architecture vs required to capture $A^{-1}$")
    axC.set_xticks(block_dim)
    axC.set_xticklabels([f"{int(b):,}" for b in block_dim])
    axC.set_xlim(block_dim[0] * 0.88, block_dim[-1] * 1.12)

    y_top = max(provided_frac.max() * 2.4, needed_means[1e-9].max() * 3.0)
    y_bot = min(needed_means[1e-3].min() * 0.30, 4e-4)
    axC.set_ylim(y_bot, y_top)
    _grid(axC)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig_c.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved single-panel figure -> {out}")

    if not args.multi_panel:
        return

    # ---- Optional 3-panel (debug / supplementary) ----
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)

    # ---- Panel A ----
    bin_widths = edges[1:] - edges[:-1]
    F_REF = 1e-2  # cycles/node
    cdf_at_fref = {}
    for color, S in zip(colors, S_values):
        p_density = spectra[S]
        p_mass = p_density * bin_widths
        cdf = np.cumsum(p_mass)
        cdf = cdf / max(cdf[-1], 1e-300)
        f_min = 1.0 / (S * leaf_L)
        m = centers >= f_min
        axA.semilogx(centers[m], cdf[m], "-", color=color, lw=1.8,
                     label=f"$S\\,{{=}}\\,{S}$  (${S*leaf_L}\\!\\times\\!{S*leaf_L}$)", zorder=3)
        cdf_interp = float(np.interp(np.log(F_REF), np.log(centers), cdf))
        cdf_at_fref[S] = cdf_interp
        axA.plot(F_REF, cdf_interp, "o", color=color, markersize=5.5,
                 markeredgecolor="black", markeredgewidth=0.6, zorder=5)
    axA.axvline(F_REF, color=REF_LINE, ls=(0, (4, 2)), lw=0.85, alpha=0.7, zorder=1)
    s_lo, s_hi = S_values[0], S_values[-1]
    c_lo, c_hi = cdf_at_fref[s_lo], cdf_at_fref[s_hi]
    axA.annotate("", xy=(F_REF, c_hi), xytext=(F_REF, c_lo),
                 arrowprops=dict(arrowstyle="<->", color="#1a1a1a", lw=1.0),
                 zorder=4)
    txt_a = axA.text(F_REF * 1.35, 0.5 * (c_lo + c_hi),
                     f"at $f\\,{{=}}\\,{F_REF:g}$ cyc/node:\n"
                     f"$S\\,{{=}}\\,{s_lo}$: {100*c_lo:.0f}% energy\n"
                     f"$S\\,{{=}}\\,{s_hi}$: {100*c_hi:.0f}% energy",
                     color="#1a1a1a", fontsize=8.5, ha="left", va="center",
                     bbox=dict(boxstyle="round,pad=0.28", fc="white",
                               ec=LEGEND_EDGE, lw=0.6, alpha=0.95))
    txt_a.set_path_effects(HALO)

    axA.set_xlabel(r"Spatial radial frequency  $f$  (cycles / node)")
    axA.set_ylabel(r"Cumulative spectral energy  $\Pr[\,f' \leq f\,]$")
    axA.set_title(r"(A)  Long-range tiles concentrate energy at lower frequencies")
    axA.set_xlim(3e-4, 0.5)
    axA.set_ylim(0.20, 1.02)
    _grid(axA)
    axA.legend(title="distance ($=$ tile size in leaves)",
               loc="lower right", framealpha=0.9, edgecolor=LEGEND_EDGE,
               title_fontsize=8.5)

    # ---- Panel B ----
    for color, S in zip(colors, S_values):
        s_mean = svals[S]
        k = np.arange(1, len(s_mean) + 1)
        axB.semilogy(k, s_mean + 1e-16, "-", color=color, lw=1.8,
                     label=f"$S\\,{{=}}\\,{S}$", zorder=3)
    axB.axvline(L_s, color="#b03030", linestyle="--", lw=1.4, zorder=4,
                label=f"$L_s\\,{{=}}\\,{L_s}\\,({{=}}\\,L/p_{{\\mathrm{{off}}}})$, rank emitted per tile")
    for eps in (1e-3, 1e-6, 1e-9):
        axB.axhline(eps, color=REF_LINE, ls=":", lw=0.7, alpha=0.7, zorder=1)
        t = axB.text(int(1.4 * L_s) - 0.6, eps * 1.6,
                     f"$\\epsilon{{=}}10^{{{int(np.log10(eps))}}}$",
                     color=REF_LINE, fontsize=8, va="bottom", ha="right", zorder=2)
        t.set_path_effects(HALO)

    axB.set_xlabel(r"Singular value index  $k$")
    axB.set_ylabel(r"$\sigma_k / \sigma_1$  (mean over tiles)")
    axB.set_title(r"(B)  Every $A^{-1}$ tile is essentially low-rank, well below $L_s$")
    axB.set_ylim(1e-11, 2.0)
    axB.set_xlim(0, int(1.4 * L_s))
    _grid(axB)
    axB.legend(loc="upper right", ncol=2, framealpha=0.9, edgecolor=LEGEND_EDGE,
               columnspacing=1.0, handlelength=1.4)

    # ---- Panel C ----
    S_arr = np.array(S_values, dtype=float)
    block_dim = S_arr * leaf_L
    provided_frac = L_s / block_dim
    eps_styles = [
        (1e-3, "s--", "#1f9d4d", r"$\epsilon\,{=}\,10^{-3}$"),
        (1e-6, "^--", "#197aa6", r"$\epsilon\,{=}\,10^{-6}$"),
        (1e-9, "d--", "#3b1da6", r"$\epsilon\,{=}\,10^{-9}$"),
    ]
    needed_means = {}
    for eps, marker, color, label in eps_styles:
        means = np.array([rank_eps[S][eps][0] for S in S_values])
        stds = np.array([rank_eps[S][eps][1] for S in S_values])
        frac = means / block_dim
        frac_lo = np.maximum(means - stds, 0.5) / block_dim
        frac_hi = (means + stds) / block_dim
        axC.loglog(S_arr, frac, marker, color=color, lw=1.4, markersize=6, zorder=3)
        axC.fill_between(S_arr, frac_lo, frac_hi, color=color, alpha=0.15, zorder=2)
        needed_means[eps] = frac

    axC.loglog(S_arr, provided_frac, "o-", color="#b03030", lw=2.0, markersize=7, zorder=4)
    axC.fill_between(S_arr, needed_means[1e-9], provided_frac,
                     where=(provided_frac > needed_means[1e-9]),
                     color="#b03030", alpha=0.08, zorder=1)

    # Inline curve labels (match plot_training.py style — no boxed legend).
    def _label(ax, x, y, text, color, dy=1.0, ha="left", va="center"):
        t = ax.text(x, y * dy, text, color=color, fontsize=8.5, ha=ha, va=va, zorder=6)
        t.set_path_effects(HALO)

    _label(axC, S_arr[1] * 1.05, provided_frac[1] * 1.18,
           r"provided $L_s/(SL)$,  $L_s\,{=}\,32\,{=}\,L/p_{\mathrm{off}}$",
           "#b03030", ha="left", va="bottom")
    # Needed labels at the right end of each curve
    for eps, _marker, color, lab in eps_styles:
        y_end = needed_means[eps][-1]
        _label(axC, S_arr[-1] * 1.04, y_end, f"needed  {lab}",
               color, ha="left", va="center")

    # Headroom callout in the shaded gap (at S=4, where the gap is widest).
    s_idx = len(S_arr) // 2
    hr_mid = provided_frac[s_idx] / max(needed_means[1e-9][s_idx], 1e-30)
    y_mid = np.sqrt(provided_frac[s_idx] * needed_means[1e-9][s_idx])
    t = axC.text(S_arr[s_idx], y_mid, f"{hr_mid:.1f}$\\times$ headroom",
                 color="#1a1a1a", fontsize=9, ha="center", va="center", zorder=6,
                 bbox=dict(boxstyle="round,pad=0.22", fc="white",
                           ec="#b03030", lw=0.7, alpha=0.95))
    t.set_path_effects(HALO)

    axC.set_xlabel(r"Distance class  $S$  (tile size in leaves)")
    axC.set_ylabel(r"Rank fraction  $r/(SL)$")
    axC.set_title(r"(C)  Architecture-provided rank exceeds what's needed at every $S$")
    axC.set_xticks(S_arr)
    axC.set_xticklabels([str(int(s)) for s in S_arr])
    # Extra right-side margin for inline labels
    axC.set_xlim(S_arr[0] * 0.85, S_arr[-1] * 1.55)
    _grid(axC)

    out = Path(args.output)
    out_3p = out.with_name(out.stem + "_3panel" + out.suffix)
    out_3p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_3p, dpi=170, bbox_inches="tight")
    print(f"Saved 3-panel debug figure -> {out_3p}")


if __name__ == "__main__":
    main()
