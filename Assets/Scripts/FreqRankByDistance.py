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
import matplotlib.pyplot as plt

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
                        default=str(_script_dir.parent.parent / "Paper" / "figures" / "freq_rank_by_distance.png"))
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

    # ---- Plot ----
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "legend.fontsize": 9.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.titlepad": 8,
    })
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    n_S = len(S_values)
    # Wider lightness range so smallest and largest S are clearly distinct;
    # avoid the brightest yellow at the top end.
    colors = [cmap(0.08 + 0.70 * i / max(1, n_S - 1)) for i in range(n_S)]

    # ---- Panel A ----
    bin_widths = edges[1:] - edges[:-1]
    F_REF = 1e-2  # cycles/node — about one cycle per 100 nodes
    cdf_at_fref = {}
    for color, S in zip(colors, S_values):
        p_density = spectra[S]
        p_mass = p_density * bin_widths
        cdf = np.cumsum(p_mass)
        cdf = cdf / max(cdf[-1], 1e-300)
        f_min = 1.0 / (S * leaf_L)
        m = centers >= f_min
        axA.semilogx(centers[m], cdf[m], "-", color=color, lw=2.4,
                     label=f"S={S}  ({S*leaf_L}×{S*leaf_L})")
        # cdf at F_REF
        cdf_interp = np.interp(np.log(F_REF), np.log(centers), cdf)
        cdf_at_fref[S] = float(cdf_interp)
        # Mark each curve's intersection with the F_REF vertical line
        axA.plot(F_REF, cdf_interp, "o", color=color, markersize=8,
                 markeredgecolor="black", markeredgewidth=0.8, zorder=5)
    axA.axvline(F_REF, color="black", linestyle=":", lw=1.1, alpha=0.8)
    # Annotation: fraction of energy below F_REF for the extremes
    s_lo, s_hi = S_values[0], S_values[-1]
    c_lo, c_hi = cdf_at_fref[s_lo], cdf_at_fref[s_hi]
    axA.annotate(
        "",
        xy=(F_REF, c_hi), xytext=(F_REF, c_lo),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.6),
    )
    axA.text(F_REF * 1.6, 0.5 * (c_lo + c_hi),
             f"S={s_lo}: {100*c_lo:.0f}% of energy\n"
             f"S={s_hi}: {100*c_hi:.0f}% of energy\n"
             f"below f={F_REF:g} cyc/node",
             color="black", fontsize=9.5, ha="left", va="center",
             bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="black", lw=0.8))
    axA.set_xlabel("Spatial radial frequency  $f$  (cycles / node)")
    axA.set_ylabel(r"Cumulative spectral energy  $\Pr[f' \leq f]$")
    axA.set_title(r"(A)  Long-range tiles concentrate energy at lower frequencies")
    axA.set_xlim(centers[0], 0.5)
    axA.set_ylim(0.0, 1.06)
    axA.grid(True, which="both", alpha=0.22)
    axA.legend(title="distance  (=tile size in leaves)", loc="lower right")

    # ---- Panel B ----
    for color, S in zip(colors, S_values):
        s_mean = svals[S]
        k = np.arange(1, len(s_mean) + 1)
        rk6, _ = rank_eps[S][1e-6]
        axB.semilogy(k, s_mean + 1e-16, "-", color=color, lw=2.4,
                     label=f"S={S}: rank @1e-6 ≈ {rk6:.1f}")
    axB.axvline(L_s, color="crimson", linestyle="--", lw=2.0,
                label=f"$L_s={L_s}$  (rank emitted per tile)")
    for eps in (1e-3, 1e-6, 1e-9):
        axB.axhline(eps, color="grey", linestyle=":", lw=0.8)
        axB.text(0.3, eps * 1.4, f" $\\epsilon={eps:g}$",
                 color="grey", fontsize=9, va="bottom", ha="left")
    axB.set_xlabel("Singular value index  $k$")
    axB.set_ylabel(r"$\sigma_k / \sigma_1$  (mean over tiles)")
    axB.set_title("(B)  Every tile is essentially low-rank, well below $L_s$")
    axB.set_ylim(1e-12, 2.0)
    axB.set_xlim(0, int(1.4 * L_s))
    axB.grid(True, which="both", alpha=0.22)
    axB.legend(loc="lower left")

    # ---- Panel C ----
    S_arr = np.array(S_values, dtype=float)
    block_dim = S_arr * leaf_L
    provided_frac = L_s / block_dim
    # mean and per-tile std for needed-rank (in absolute rank units)
    eps_styles = [
        (1e-3, "s--", "#1f9d4d", r"needed for $\epsilon=10^{-3}$"),
        (1e-6, "^--", "#197aa6", r"needed for $\epsilon=10^{-6}$"),
        (1e-9, "d--", "#3b1da6", r"needed for $\epsilon=10^{-9}$"),
    ]
    needed_means = {}
    for eps, marker, color, label in eps_styles:
        means = np.array([rank_eps[S][eps][0] for S in S_values])
        stds = np.array([rank_eps[S][eps][1] for S in S_values])
        frac = means / block_dim
        frac_lo = np.maximum(means - stds, 0.5) / block_dim
        frac_hi = (means + stds) / block_dim
        axC.loglog(S_arr, frac, marker, color=color, lw=1.7, markersize=8, label=label)
        axC.fill_between(S_arr, frac_lo, frac_hi, color=color, alpha=0.12)
        needed_means[eps] = frac

    axC.loglog(S_arr, provided_frac, "o-", color="crimson", lw=2.6, markersize=10,
               label=f"provided by architecture  ($L_s/(SL)$,  $L_s\\!=\\!{L_s}$)")
    axC.fill_between(S_arr, needed_means[1e-9], provided_frac,
                     where=(provided_frac > needed_means[1e-9]),
                     color="crimson", alpha=0.10,
                     label="headroom (provided $-$ needed @ $10^{-9}$)")
    # Annotate headroom factor at the largest S
    hr = provided_frac[-1] / max(needed_means[1e-9][-1], 1e-30)
    axC.annotate(f"{hr:.1f}× headroom",
                 xy=(S_arr[-1], np.sqrt(provided_frac[-1] * needed_means[1e-9][-1])),
                 xytext=(S_arr[-1] * 0.95, np.sqrt(provided_frac[-1] * needed_means[1e-9][-1]) * 0.6),
                 ha="right", va="top", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="crimson", lw=0.8))
    axC.set_xlabel("Distance class  $S$  (tile size in leaves)")
    axC.set_ylabel(r"Rank fraction  $r/(SL)$")
    axC.set_title(r"(C)  Architecture-provided rank exceeds what's needed at every distance")
    axC.set_xticks(S_arr)
    axC.set_xticklabels([str(int(s)) for s in S_arr])
    axC.grid(True, which="both", alpha=0.22)
    axC.legend(loc="lower left")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"Saved figure -> {out}")


if __name__ == "__main__":
    main()
