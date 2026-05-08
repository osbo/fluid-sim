"""
EvalScaleSweep.py — Run InspectModel --test-only across scales and collect results.

For each (scale, model) combination this script:
  1. Detects how many test frames exist (typically 20)
  2. Invokes InspectModel.py --test-only once per frame
  3. Parses the PCG benchmark summary from stdout
  4. Averages timing and iterations across all frames and writes one row per method

Default mode: scale sweep with the baseline config (d128, L3, hw) at scales 1024–16384.

Usage:
  python3 EvalScaleSweep.py                         # all 5 baseline scales, all frames
  python3 EvalScaleSweep.py --scales 2048 4096      # subset of scales
  python3 EvalScaleSweep.py --ablation              # all 20 configs (arch ablation)
  python3 EvalScaleSweep.py --frames 0 1 2          # specific frames only
  python3 EvalScaleSweep.py --out results/sweep.csv
  python3 EvalScaleSweep.py --index 3               # one config (for SLURM array tasks)
  python3 EvalScaleSweep.py --ablation --index 7    # one ablation config
  python3 EvalScaleSweep.py --merge                 # merge results/sweep_task_*.csv → scale_sweep.csv
  python3 EvalScaleSweep.py --ablation --merge      # merge ablation task CSVs

CSV schema:
  scale, config, method, device,
  setup_ms, setup_ms_std,
  inference_ms, inference_ms_std,
  solve_ms, solve_ms_std,
  iters, iters_std,
  total_ms, total_ms_std,
  n_frames

  - setup_ms:     preconditioner build time (LeafOnly → blank)
  - inference_ms: neural forward pass time  (LeafOnly only)
  - solve_ms:     PCG solve time after preconditioner is applied
  - iters:        PCG iterations to convergence
  - total_ms:     setup + inference + solve (as reported by InspectModel)
  - *_std:        sample std-dev across test frames (blank when n_frames=1)
  - n_frames:     number of frames successfully measured
"""

import argparse
import csv
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SCRIPTS_DIR / "weights"
DATA_BASE   = SCRIPTS_DIR / "data"

# ── Config tables ─────────────────────────────────────────────────────────────

SCALE_CONFIGS = [
    dict(scale=1024,  leaf=128, d=128, L=3, hw=True,  name="v2_1024_d128_L3_hw"),
    dict(scale=2048,  leaf=128, d=128, L=3, hw=True,  name="v2_2048_d128_L3_hw"),
    dict(scale=4096,  leaf=128, d=128, L=3, hw=True,  name="v2_4096_d128_L3_hw"),
    dict(scale=8192,  leaf=128, d=128, L=3, hw=True,  name="v2_8192_d128_L3_hw"),
    dict(scale=16384, leaf=256, d=128, L=3, hw=True,  name="v2_16384_d128_L3_hw"),
]

ALL_CONFIGS = [
    dict(scale=1024,  leaf=128, d=128, L=3, hw=True,  name="v2_1024_d128_L3_hw"),
    dict(scale=2048,  leaf=128, d=128, L=3, hw=True,  name="v2_2048_d128_L3_hw"),
    dict(scale=4096,  leaf=128, d=128, L=3, hw=True,  name="v2_4096_d128_L3_hw"),
    dict(scale=8192,  leaf=128, d=128, L=3, hw=True,  name="v2_8192_d128_L3_hw"),
    dict(scale=16384, leaf=256, d=128, L=3, hw=True,  name="v2_16384_d128_L3_hw"),
    dict(scale=2048,  leaf=128, d=64,  L=3, hw=True,  name="v2_2048_d64_L3_hw"),
    dict(scale=4096,  leaf=128, d=64,  L=3, hw=True,  name="v2_4096_d64_L3_hw"),
    dict(scale=8192,  leaf=128, d=64,  L=3, hw=True,  name="v2_8192_d64_L3_hw"),
    dict(scale=2048,  leaf=128, d=256, L=3, hw=True,  name="v2_2048_d256_L3_hw"),
    dict(scale=4096,  leaf=128, d=256, L=3, hw=True,  name="v2_4096_d256_L3_hw"),
    dict(scale=8192,  leaf=128, d=256, L=3, hw=True,  name="v2_8192_d256_L3_hw"),
    dict(scale=2048,  leaf=128, d=128, L=1, hw=True,  name="v2_2048_d128_L1_hw"),
    dict(scale=4096,  leaf=128, d=128, L=1, hw=True,  name="v2_4096_d128_L1_hw"),
    dict(scale=8192,  leaf=128, d=128, L=1, hw=True,  name="v2_8192_d128_L1_hw"),
    dict(scale=2048,  leaf=128, d=128, L=2, hw=True,  name="v2_2048_d128_L2_hw"),
    dict(scale=4096,  leaf=128, d=128, L=2, hw=True,  name="v2_4096_d128_L2_hw"),
    dict(scale=8192,  leaf=128, d=128, L=2, hw=True,  name="v2_8192_d128_L2_hw"),
    dict(scale=2048,  leaf=128, d=128, L=3, hw=False, name="v2_2048_d128_L3_nohw"),
    dict(scale=4096,  leaf=128, d=128, L=3, hw=False, name="v2_4096_d128_L3_nohw"),
    dict(scale=8192,  leaf=128, d=128, L=3, hw=False, name="v2_8192_d128_L3_nohw"),
]

# ── Regex patterns ─────────────────────────────────────────────────────────────

# "  Setup: 0.12 ms, solve (CPU): 34.56 ms, 42 iterations, total: 34.68 ms"
_RE_SETUP_CPU_GPU = re.compile(
    r"Setup:\s*([\d.]+)\s*ms,\s*solve\s*\((\w+)\):\s*([\d.]+)\s*ms,\s*(\d+)\s*iterations"
    r"(?:,\s*total:\s*([\d.]+)\s*ms)?"
)

# "  Setup: X ms, solve: Y ms, N iterations, total: Z ms"  (AMG, AMGX)
_RE_SETUP_SOLVE = re.compile(
    r"Setup:\s*([\d.]+)\s*ms,\s*solve:\s*([\d.]+)\s*ms,\s*(\d+)\s*iterations"
    r"(?:,\s*total:\s*([\d.]+)\s*ms)?"
)

# "  Inference: X ms, solve: Y ms, N iterations, total: Z ms; ..."
_RE_LEAFONLY = re.compile(
    r"Inference:\s*([\d.]+)\s*ms,\s*solve:\s*([\d.]+)\s*ms,\s*(\d+)\s*iterations"
    r"(?:,\s*total:\s*([\d.]+)\s*ms)?"
)

_SECTION_HEADERS = [
    ("Unpreconditioned (CG)",  "cg"),
    ("Diag (Jacobi",           "jacobi"),
    ("IC (CPU:",               "ic"),
    ("LeafOnly:",              "leafonly"),
    ("AMG (CPU):",             "amg_cpu"),
    ("AMGX (GPU)",             "amgx_gpu"),
]

# Fields that are averaged; keyed by method type
_NUMERIC_FIELDS = ["setup_ms", "inference_ms", "solve_ms", "iters", "total_ms"]


def _parse_output(stdout: str) -> "list[dict]":
    """Parse one InspectModel --test-only stdout into a list of measurement dicts.

    Each dict has keys: method, device, setup_ms, inference_ms, solve_ms, iters, total_ms.
    Missing values are None (not blank string) so averaging can skip them cleanly.
    """
    rows = []
    current_method = None
    cpu_row_pending = None

    def _flush_pending():
        nonlocal cpu_row_pending
        if cpu_row_pending is not None:
            rows.append(cpu_row_pending)
            cpu_row_pending = None

    def _f(s):
        return float(s) if s is not None else None

    for raw_line in stdout.splitlines():
        line = raw_line.strip()

        for header, key in _SECTION_HEADERS:
            if line.startswith(header):
                _flush_pending()
                current_method = key
                break
        else:
            if current_method is None:
                continue

            if current_method == "leafonly":
                m = _RE_LEAFONLY.search(line)
                if m:
                    inf, sol, itr, tot = m.groups()
                    rows.append(dict(method="leafonly", device="gpu",
                                    setup_ms=None, inference_ms=_f(inf),
                                    solve_ms=_f(sol), iters=_f(itr), total_ms=_f(tot)))
                continue

            if current_method in ("amg_cpu", "amgx_gpu"):
                m = _RE_SETUP_SOLVE.search(line)
                if m:
                    setup, sol, itr, tot = m.groups()
                    dev = "cpu" if current_method == "amg_cpu" else "gpu"
                    rows.append(dict(method=current_method, device=dev,
                                    setup_ms=_f(setup), inference_ms=None,
                                    solve_ms=_f(sol), iters=_f(itr), total_ms=_f(tot)))
                continue

            m = _RE_SETUP_CPU_GPU.search(line)
            if m:
                setup, dev_str, sol, itr, tot = m.groups()
                dev_str = dev_str.lower()
                mkey = "ic_amgx" if (current_method == "ic" and "AMGX" in line) else current_method
                row = dict(method=mkey, device=dev_str,
                           setup_ms=_f(setup), inference_ms=None,
                           solve_ms=_f(sol), iters=_f(itr), total_ms=_f(tot))
                if dev_str == "cpu":
                    _flush_pending()
                    cpu_row_pending = row
                else:
                    rows.append(row)
                continue

    _flush_pending()
    return rows


def _detect_frames(data_folder: Path) -> "list[int]":
    """Return sorted list of available frame indices in a test folder."""
    frames = sorted(
        int(p.name.split("_")[1])
        for p in data_folder.iterdir()
        if p.is_dir() and p.name.startswith("frame_")
    )
    return frames


def _run_inspect(cfg: dict, frame: int, extra_args: "list[str]") -> str:
    """Run InspectModel --test-only for one frame; return stdout (empty on failure)."""
    weights = WEIGHTS_DIR / f"{cfg['name']}.bytes"
    data_folder = DATA_BASE / f"multiphase_v2_{cfg['scale']}" / "test"

    cmd = [
        "python3", "-u",
        str(SCRIPTS_DIR / "InspectModel.py"),
        "--test-only",
        "--weights",     str(weights),
        "--data-folder", str(data_folder),
        "--frame",       str(frame),
    ]
    if not cfg["hw"]:
        cmd.append("--no-use-highways")
    cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [ERROR frame {frame}] returncode={result.returncode}", file=sys.stderr)
        print(result.stderr[-1000:], file=sys.stderr)
        return ""
    return result.stdout


def _mean_std(values: "list[float]") -> "tuple[str, str]":
    """Return (mean_str, std_str) for a list of floats. std is '' if len < 2."""
    n = len(values)
    if n == 0:
        return "", ""
    mean = sum(values) / n
    if n < 2:
        return f"{mean:.4f}", ""
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return f"{mean:.4f}", f"{math.sqrt(variance):.4f}"


def _aggregate(per_frame_rows: "list[list[dict]]", scale: int, config_name: str) -> "list[dict]":
    """Average per-frame measurements into one summary row per (method, device)."""
    # Group all scalar values by (method, device, field)
    buckets: dict = defaultdict(list)
    for frame_rows in per_frame_rows:
        for r in frame_rows:
            key = (r["method"], r["device"])
            for field in _NUMERIC_FIELDS:
                v = r.get(field)
                if v is not None:
                    buckets[(key, field)].append(v)

    # Collect unique (method, device) pairs in stable order
    seen = {}
    for frame_rows in per_frame_rows:
        for r in frame_rows:
            k = (r["method"], r["device"])
            if k not in seen:
                seen[k] = True

    out = []
    n_frames = len(per_frame_rows)
    for (method, device) in seen:
        row = {"scale": scale, "config": config_name, "method": method, "device": device,
               "n_frames": n_frames}
        for field in _NUMERIC_FIELDS:
            vals = buckets[((method, device), field)]
            mean_s, std_s = _mean_std(vals)
            row[field]              = mean_s
            row[f"{field}_std"]     = std_s
        out.append(row)
    return out


def _merge(results_dir: Path, stem: str, out_path: Path) -> None:
    """Merge all results/{stem}_task_*.csv into out_path, preserving config order."""
    task_files = sorted(results_dir.glob(f"{stem}_task_*.csv"))
    if not task_files:
        print(f"No task CSVs found matching {stem}_task_*.csv in {results_dir}", file=sys.stderr)
        sys.exit(1)

    with open(out_path, "w", newline="") as fout:
        writer = None
        for tf in task_files:
            with open(tf, newline="") as fin:
                reader = csv.DictReader(fin)
                if writer is None:
                    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
    print(f"Merged {len(task_files)} task CSVs → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scales", nargs="+", type=int, default=None,
                        metavar="N", help="Subset of scales to run (default: all 5 baseline scales).")
    parser.add_argument("--ablation", action="store_true",
                        help="Run all 20 configs instead of just the 5 baseline scales.")
    parser.add_argument("--index", type=int, default=None,
                        metavar="I", help="Run a single config by 0-based index (for SLURM array tasks).")
    parser.add_argument("--merge", action="store_true",
                        help="Merge per-task CSVs (results/*_task_*.csv) into the final CSV and exit.")
    parser.add_argument("--frames", nargs="+", type=int, default=None,
                        metavar="N", help="Explicit frame indices to use (default: all frames found in test/).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path (default: results/scale_sweep.csv or results/ablation_sweep.csv).")
    parser.add_argument("--pcg-backend", choices=("cudagraph", "compile", "eager"), default="cudagraph",
                        help="GPU PCG backend for InspectModel (default: cudagraph).")
    parser.add_argument("--pcg-precision", choices=("f32", "mixed", "f64"), default="f64",
                        help="PCG precision (default: f64).")
    parser.add_argument("--pcg-check-freq", type=int, default=3,
                        metavar="K", help="Residual check frequency (default: 3).")
    args = parser.parse_args()

    stem = "ablation_sweep" if args.ablation else "scale_sweep"
    results_dir = SCRIPTS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else results_dir / f"{stem}.csv"

    if args.merge:
        _merge(results_dir, stem, out_path)
        return

    configs = ALL_CONFIGS if args.ablation else SCALE_CONFIGS
    if args.scales:
        configs = [c for c in configs if c["scale"] in args.scales]

    if args.index is not None:
        n = len(configs)
        if not (0 <= args.index < n):
            print(f"Error: --index {args.index} out of range [0, {n-1}]", file=sys.stderr)
            sys.exit(1)
        configs = [configs[args.index]]
        # Write to a per-task file so SLURM array tasks don't collide
        out_path = results_dir / f"{stem}_task_{args.index:03d}.csv"
        print(f"Task {args.index}: {configs[0]['name']} → {out_path}", flush=True)

    if not configs:
        print("No matching configs. Check --scales.", file=sys.stderr)
        sys.exit(1)

    extra = [
        "--pcg-cuda-backend", args.pcg_backend,
        "--pcg-precision",    args.pcg_precision,
        "--pcg-check-freq",   str(args.pcg_check_freq),
    ]

    fieldnames = [
        "scale", "config", "method", "device",
        "setup_ms", "setup_ms_std",
        "inference_ms", "inference_ms_std",
        "solve_ms", "solve_ms_std",
        "iters", "iters_std",
        "total_ms", "total_ms_std",
        "n_frames",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ci, cfg in enumerate(configs):
            print(f"\n[{ci+1}/{len(configs)}] {cfg['name']}  (scale={cfg['scale']})", flush=True)

            data_folder = DATA_BASE / f"multiphase_v2_{cfg['scale']}" / "test"
            weights     = WEIGHTS_DIR / f"{cfg['name']}.bytes"

            if not weights.exists():
                print(f"  [SKIP] weights not found: {weights}", file=sys.stderr)
                continue
            if not data_folder.exists():
                print(f"  [SKIP] data folder not found: {data_folder}", file=sys.stderr)
                continue

            frames = args.frames if args.frames is not None else _detect_frames(data_folder)
            print(f"  Frames: {frames}", flush=True)

            per_frame_rows = []
            for fi, frame in enumerate(frames):
                print(f"  [{fi+1}/{len(frames)}] frame {frame}", end="", flush=True)
                stdout = _run_inspect(cfg, frame, extra)
                if not stdout:
                    print("  → skipped (error)", flush=True)
                    continue
                rows = _parse_output(stdout)
                if not rows:
                    print(f"  → [WARN] no rows parsed", flush=True)
                else:
                    per_frame_rows.append(rows)
                    leafonly = next((r for r in rows if r["method"] == "leafonly"), None)
                    if leafonly:
                        print(f"  → LeafOnly {leafonly['iters']:.0f} iters  "
                              f"inf={leafonly['inference_ms']:.2f} ms  "
                              f"solve={leafonly['solve_ms']:.2f} ms", flush=True)
                    else:
                        print(f"  → {len(rows)} rows", flush=True)

            if not per_frame_rows:
                print(f"  [SKIP] no successful frames", file=sys.stderr)
                continue

            agg_rows = _aggregate(per_frame_rows, cfg["scale"], cfg["name"])
            for row in agg_rows:
                writer.writerow(row)
            f.flush()

            leafonly = next((r for r in agg_rows if r["method"] == "leafonly"), None)
            if leafonly:
                print(f"  → Avg LeafOnly: {leafonly['iters']} iters "
                      f"(±{leafonly['iters_std']}), "
                      f"total={leafonly['total_ms']} ms "
                      f"(±{leafonly['total_ms_std']} ms) "
                      f"over {leafonly['n_frames']} frames")

    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
