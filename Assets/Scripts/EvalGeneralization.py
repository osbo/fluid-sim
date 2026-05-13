#!/usr/bin/env python3
"""
Run InspectModel --test-only on generalization experiment cells and aggregate CSVs.

Usage:
  python3 EvalGeneralization.py
  python3 EvalGeneralization.py --index 2
  python3 EvalGeneralization.py --merge
  python3 EvalGeneralization.py --frames 0 1 2
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
DATA_BASE = SCRIPTS_DIR / "data" / "generalization_4096"
RESULTS_DIR = SCRIPTS_DIR / "results"

_INSPECT_PCG_MAX_ITER = 50000
_NUMERIC_FIELDS = ["setup_ms", "inference_ms", "solve_ms", "iters", "total_ms"]

_RE_SETUP_CPU_GPU = re.compile(
    r"Setup:\s*([\d.]+)\s*ms,\s*solve\s*\((\w+)\):\s*([\d.]+)\s*ms,\s*(\d+)\s*iterations"
    r"(?:,\s*total:\s*([\d.]+)\s*ms)?"
)
_RE_SETUP_SOLVE = re.compile(
    r"Setup:\s*([\d.]+)\s*ms,\s*solve:\s*([\d.]+)\s*ms,\s*(\d+)\s*iterations"
    r"(?:,\s*total:\s*([\d.]+)\s*ms)?"
)
_RE_LEAFONLY = re.compile(
    r"Inference:\s*([\d.]+)\s*ms,\s*solve:\s*([\d.]+)\s*ms,\s*(\d+)\s*iterations"
    r"(?:,\s*total:\s*([\d.]+)\s*ms)?"
)

_SECTION_HEADERS = [
    ("Unpreconditioned (CG)", "cg"),
    ("Diag (Jacobi", "jacobi"),
    ("IC (CPU:", "ic"),
    ("LeafOnly:", "leafonly"),
    ("AMG (CPU):", "amg_cpu"),
    ("AMGX (GPU)", "amgx_gpu"),
]

BASE_MODEL_CONFIGS = [
    dict(
        model_name="g4096_id_d128_L3_hw",
        train_split="g4096_train_id",
        hw=True,
        eval_cells=["g4096_eval_id"],
    ),
    dict(
        model_name="g4096_topo_no_closed_d128_L3_hw",
        train_split="g4096_train_topo_no_closed",
        hw=True,
        eval_cells=["g4096_eval_id", "g4096_eval_closed_only"],
    ),
    dict(
        model_name="g4096_param_low_d128_L3_hw",
        train_split="g4096_train_param_low",
        hw=True,
        eval_cells=["g4096_eval_id", "g4096_eval_param_high"],
    ),
    dict(
        model_name="g4096_combo_no_closed_low_d128_L3_hw",
        train_split="g4096_train_combo_no_closed_low",
        hw=True,
        eval_cells=[
            "g4096_eval_id",
            "g4096_eval_closed_only",
            "g4096_eval_param_high",
            "g4096_eval_combo_closed_high",
        ],
    ),
]


def _model_configs_with_suffix(model_suffix: str):
    out = []
    for cfg in BASE_MODEL_CONFIGS:
        c = dict(cfg)
        c["model_name"] = f"{c['model_name']}{model_suffix}"
        out.append(c)
    return out


def _parse_output(stdout):
    rows = []
    current_method = None
    cpu_row_pending = None

    def _flush_pending() -> None:
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
                    rows.append(
                        dict(
                            method="leafonly",
                            device="gpu",
                            setup_ms=None,
                            inference_ms=_f(inf),
                            solve_ms=_f(sol),
                            iters=_f(itr),
                            total_ms=_f(tot),
                        )
                    )
                continue

            if current_method in ("amg_cpu", "amgx_gpu"):
                m = _RE_SETUP_SOLVE.search(line)
                if m:
                    setup, sol, itr, tot = m.groups()
                    dev = "cpu" if current_method == "amg_cpu" else "gpu"
                    rows.append(
                        dict(
                            method=current_method,
                            device=dev,
                            setup_ms=_f(setup),
                            inference_ms=None,
                            solve_ms=_f(sol),
                            iters=_f(itr),
                            total_ms=_f(tot),
                        )
                    )
                continue

            m = _RE_SETUP_CPU_GPU.search(line)
            if m:
                setup, dev_str, sol, itr, tot = m.groups()
                dev_str = dev_str.lower()
                mkey = "ic_amgx" if (current_method == "ic" and "AMGX" in line) else current_method
                row = dict(
                    method=mkey,
                    device=dev_str,
                    setup_ms=_f(setup),
                    inference_ms=None,
                    solve_ms=_f(sol),
                    iters=_f(itr),
                    total_ms=_f(tot),
                )
                if dev_str == "cpu":
                    _flush_pending()
                    cpu_row_pending = row
                else:
                    rows.append(row)

    _flush_pending()
    return rows


def _detect_frames(data_folder):
    return sorted(
        int(p.name.split("_")[1])
        for p in data_folder.iterdir()
        if p.is_dir() and p.name.startswith("frame_")
    )


def _mean_std(values):
    n = len(values)
    if n == 0:
        return "", ""
    mean = sum(values) / n
    if n < 2:
        return f"{mean:.4f}", ""
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return f"{mean:.4f}", f"{math.sqrt(variance):.4f}"


def _drop_leafonly_nonconverged(per_frame_rows):
    kept = []
    dropped = 0
    for rows in per_frame_rows:
        lo = next((r for r in rows if r["method"] == "leafonly"), None)
        if lo is None or lo.get("iters") is None:
            kept.append(rows)
            continue
        if float(lo["iters"]) >= float(_INSPECT_PCG_MAX_ITER):
            dropped += 1
            continue
        kept.append(rows)
    return kept, dropped


def _aggregate(per_frame_rows, model_name, train_split, eval_cell):
    buckets: dict = defaultdict(list)
    for frame_rows in per_frame_rows:
        for r in frame_rows:
            key = (r["method"], r["device"])
            for field in _NUMERIC_FIELDS:
                v = r.get(field)
                if v is not None:
                    buckets[(key, field)].append(v)

    seen = {}
    for frame_rows in per_frame_rows:
        for r in frame_rows:
            k = (r["method"], r["device"])
            if k not in seen:
                seen[k] = True

    out = []
    n_frames = len(per_frame_rows)
    for method, device in seen:
        row = {
            "model_name": model_name,
            "train_split": train_split,
            "eval_cell": eval_cell,
            "method": method,
            "device": device,
            "n_frames": n_frames,
        }
        for field in _NUMERIC_FIELDS:
            vals = buckets[((method, device), field)]
            mean_s, std_s = _mean_std(vals)
            row[field] = mean_s
            row[f"{field}_std"] = std_s
        out.append(row)
    return out


def _run_inspect(model_cfg, data_folder, frame, extra_args):
    weights = WEIGHTS_DIR / f"{model_cfg['model_name']}.bytes"
    cmd = [
        "python3",
        "-u",
        str(SCRIPTS_DIR / "InspectModel.py"),
        "--test-only",
        "--weights",
        str(weights),
        "--data-folder",
        str(data_folder),
        "--frame",
        str(frame),
    ]
    if not model_cfg["hw"]:
        cmd.append("--no-use-highways")
    cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [ERROR frame {frame}] returncode={result.returncode}", file=sys.stderr)
        print(result.stderr[-1000:], file=sys.stderr)
        return ""
    return result.stdout


def _merge(results_dir, stem, out_path):
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
    print(f"Merged {len(task_files)} task CSVs -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=None, help="Run a single model config index.")
    parser.add_argument("--merge", action="store_true", help="Merge per-task CSVs and exit.")
    parser.add_argument("--frames", nargs="+", type=int, default=None, help="Optional frame subset.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap evaluation to the first N detected test frames (ignored when --frames is provided).",
    )
    parser.add_argument("--out", type=str, default=None, help="Output CSV path.")
    parser.add_argument(
        "--out-stem",
        type=str,
        default="generalization_sweep",
        help="Output stem for task/merged CSV names (default: generalization_sweep).",
    )
    parser.add_argument(
        "--model-suffix",
        type=str,
        default="",
        help="Optional suffix appended to model_name when resolving weights (example: _sai).",
    )
    parser.add_argument(
        "--pcg-backend",
        choices=("cudagraph", "compile", "eager"),
        default="cudagraph",
        help="GPU PCG backend for InspectModel.",
    )
    parser.add_argument(
        "--pcg-precision",
        choices=("f32", "mixed", "f64"),
        default="f64",
        help="PCG precision for InspectModel.",
    )
    parser.add_argument("--pcg-check-freq", type=int, default=3, help="Residual check frequency.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = args.out_stem
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{stem}.csv"

    if args.merge:
        _merge(RESULTS_DIR, stem, out_path)
        return

    configs = _model_configs_with_suffix(args.model_suffix)
    if args.index is not None:
        n = len(configs)
        if not (0 <= args.index < n):
            print(f"Error: --index {args.index} out of range [0, {n - 1}]", file=sys.stderr)
            sys.exit(1)
        configs = [configs[args.index]]
        out_path = RESULTS_DIR / f"{stem}_task_{args.index:03d}.csv"
        print(f"Task {args.index}: {configs[0]['model_name']} -> {out_path}", flush=True)

    extra = [
        "--pcg-cuda-backend",
        args.pcg_backend,
        "--pcg-precision",
        args.pcg_precision,
        "--pcg-check-freq",
        str(args.pcg_check_freq),
    ]

    fieldnames = [
        "model_name",
        "train_split",
        "eval_cell",
        "method",
        "device",
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
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ci, cfg in enumerate(configs):
            weights = WEIGHTS_DIR / f"{cfg['model_name']}.bytes"
            if not weights.exists():
                print(f"[SKIP {ci + 1}/{len(configs)}] missing weights: {weights}", file=sys.stderr)
                continue

            print(f"\n[{ci + 1}/{len(configs)}] {cfg['model_name']}", flush=True)
            for eval_cell in cfg["eval_cells"]:
                data_folder = DATA_BASE / eval_cell / "test"
                if not data_folder.exists():
                    print(f"  [SKIP cell={eval_cell}] missing folder: {data_folder}", file=sys.stderr)
                    continue

                if args.frames is not None:
                    frames = args.frames
                else:
                    frames = _detect_frames(data_folder)
                    if args.max_frames is not None:
                        frames = frames[: max(0, int(args.max_frames))]
                print(f"  cell={eval_cell} frames={len(frames)}", flush=True)

                per_frame_rows = []
                for i, frame in enumerate(frames):
                    print(f"    [{i + 1}/{len(frames)}] frame {frame}", end="", flush=True)
                    stdout = _run_inspect(cfg, data_folder, frame, extra)
                    if not stdout:
                        print(" -> skipped (error)", flush=True)
                        continue
                    rows = _parse_output(stdout)
                    if not rows:
                        print(" -> no rows parsed", flush=True)
                        continue
                    per_frame_rows.append(rows)
                    lo = next((r for r in rows if r["method"] == "leafonly"), None)
                    if lo:
                        print(
                            f" -> LeafOnly {lo['iters']:.0f} iters, "
                            f"total={lo['total_ms'] if lo['total_ms'] is not None else float('nan'):.2f} ms",
                            flush=True,
                        )
                    else:
                        print(f" -> parsed {len(rows)} rows", flush=True)

                if not per_frame_rows:
                    print(f"  [SKIP cell={eval_cell}] no successful frames", file=sys.stderr)
                    continue

                per_frame_rows, ndrop = _drop_leafonly_nonconverged(per_frame_rows)
                if ndrop:
                    print(
                        f"  dropped {ndrop} frame(s) at LeafOnly PCG cap {_INSPECT_PCG_MAX_ITER}",
                        flush=True,
                    )
                if not per_frame_rows:
                    print(f"  [SKIP cell={eval_cell}] all frames dropped at PCG cap", file=sys.stderr)
                    continue

                agg_rows = _aggregate(per_frame_rows, cfg["model_name"], cfg["train_split"], eval_cell)
                for row in agg_rows:
                    writer.writerow(row)
                f.flush()

                lo_agg = next((r for r in agg_rows if r["method"] == "leafonly"), None)
                if lo_agg:
                    print(
                        f"  avg LeafOnly on {eval_cell}: iters={lo_agg['iters']} "
                        f"total_ms={lo_agg['total_ms']} over {lo_agg['n_frames']} frames",
                        flush=True,
                    )

    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
