#!/usr/bin/env python3
"""
Autonomous experiment harness: sweep problem sizes and model configs for LeafOnly.

Grid: all use --use-highways
  N       in {8192, 16384, 32768}
  d_model in {128, 256, 512}
  layers  in {3, 5}

Stopping criterion: train until LR scheduler reaches 1e-6 (--stop-at-min-lr).
If --steps is exhausted before LR converges, continue from final LR with
--continue-training + --lr <final_lr> until convergence.

Usage:
  python3 run_size_experiments.py                  # run all
  python3 run_size_experiments.py --dry-run        # print plan only
  python3 run_size_experiments.py --only N16384    # filter by substring of exp name
  python3 run_size_experiments.py --no-skip-done   # re-run already-completed experiments
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
CONFIG_PY = SCRIPTS_DIR / "leafonly" / "config.py"
CONFIG_BACKUP = SCRIPTS_DIR / "leafonly" / "config.py.experiment_bak"
EXPERIMENTS_ROOT = SCRIPTS_DIR.parent.parent / "experiments"

# ─── Experiment grid ─────────────────────────────────────────────────────────
# All use --use-highways. Steps = chunk size per training call; harness will
# continue from final LR until LR hits 1e-6.
STEPS_PER_CHUNK = 20000
MAX_CHUNKS = 6  # safety cap: at most 6 × 20k = 120k steps total
INITIAL_LR = 2e-4
LR_CONVERGED = 1e-6

# (N, d_model, layers)  —  all use highways=True
EXPERIMENTS = [
    # N = 8192  (64 leaves — current training size)
    ( 8192,  128, 3),
    ( 8192,  128, 5),
    ( 8192,  256, 3),
    ( 8192,  256, 5),
    ( 8192,  512, 3),
    ( 8192,  512, 5),
    # N = 16384  (128 leaves)
    (16384,  128, 3),
    (16384,  128, 5),
    (16384,  256, 3),
    (16384,  256, 5),
    (16384,  512, 3),
    (16384,  512, 5),
    # N = 32768  (256 leaves)
    (32768,  128, 3),
    (32768,  128, 5),
    (32768,  256, 3),
    (32768,  256, 5),
    (32768,  512, 3),
    (32768,  512, 5),
]


def exp_name(N, d_model, layers):
    return f"N{N}_d{d_model}_L{layers}_hw"


# ─── Config patching ─────────────────────────────────────────────────────────
def backup_config():
    shutil.copy2(CONFIG_PY, CONFIG_BACKUP)
    print(f"  [config] Backed up {CONFIG_PY.name} → {CONFIG_BACKUP.name}")


def patch_config(N: int):
    text = CONFIG_PY.read_text()
    new_text = re.sub(
        r"^(MAX_MIXED_SIZE\s*=\s*)\d+",
        f"\\g<1>{N}",
        text,
        flags=re.MULTILINE,
    )
    if new_text == text:
        raise RuntimeError(f"Pattern MAX_MIXED_SIZE not found in {CONFIG_PY}")
    CONFIG_PY.write_text(new_text)
    pycache = CONFIG_PY.parent / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache)
    print(f"  [config] MAX_MIXED_SIZE = {N}  (NUM_LEAVES = {N // 128})")


def restore_config():
    if CONFIG_BACKUP.exists():
        shutil.copy2(CONFIG_BACKUP, CONFIG_PY)
        pycache = CONFIG_PY.parent / "__pycache__"
        if pycache.exists():
            shutil.rmtree(pycache)
        print("  [config] Restored original config.py")


# ─── Subprocess runner ────────────────────────────────────────────────────────
def run_cmd(cmd, log_path: Path, timeout_s=None):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(log_path, "w") as fh:
        fh.write(f"# CMD: {' '.join(str(c) for c in cmd)}\n")
        fh.write(f"# TIME: {datetime.now().isoformat()}\n\n")
        fh.flush()
        proc = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(SCRIPTS_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=timeout_s,
            text=True,
        )
        fh.write(proc.stdout)
    elapsed = time.perf_counter() - t0
    print(f"    → exit={proc.returncode}  elapsed={elapsed:.0f}s  log={log_path.name}")
    return proc.returncode, proc.stdout


# ─── LR extraction from training log ─────────────────────────────────────────
def parse_final_lr(text: str) -> float | None:
    """Return the last lr=X.XXe-XX value printed by the training loop."""
    matches = re.findall(r"\blr=([\d.e+\-]+)", text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def lr_converged(lr: float | None) -> bool:
    return lr is not None and lr <= LR_CONVERGED * 1.01


# ─── Output parser ────────────────────────────────────────────────────────────
def parse_inspect_output(text: str) -> dict:
    """Extract timing/iteration numbers from InspectModel --test-only output."""
    r = {}

    # Unpreconditioned CG
    block = re.search(r"Unpreconditioned \(CG\):(.*?)(?=\n\S|\Z)", text, re.S)
    if block:
        b = block.group(1)
        m = re.search(r"solve \(CPU\): ([\d.]+) ms, (\d+) iter", b)
        if m:
            r["unprec_cpu_ms"] = float(m.group(1)); r["unprec_cpu_iters"] = int(m.group(2))
        m = re.search(r"solve \(GPU\): ([\d.]+) ms, (\d+) iter", b)
        if m:
            r["unprec_gpu_ms"] = float(m.group(1)); r["unprec_gpu_iters"] = int(m.group(2))

    # Diag (Jacobi)
    block = re.search(r"Diag \(Jacobi.*?\):(.*?)(?=\n\S|\Z)", text, re.S)
    if block:
        b = block.group(1)
        m = re.search(r"solve \(CPU\): ([\d.]+) ms, (\d+) iter", b)
        if m:
            r["diag_cpu_ms"] = float(m.group(1)); r["diag_cpu_iters"] = int(m.group(2))
        m = re.search(r"solve \(GPU\): ([\d.]+) ms, (\d+) iter", b)
        if m:
            r["diag_gpu_ms"] = float(m.group(1)); r["diag_gpu_iters"] = int(m.group(2))

    # IC — identified by [ilupp IChol0] and [AMGX MULTICOLOR_DILU] markers
    for line in text.splitlines():
        if "IChol" in line or "[ilupp" in line:
            m = re.search(
                r"Setup: ([\d.]+) ms, solve \(CPU\): ([\d.]+) ms, (\d+) iter.*?total: ([\d.]+) ms",
                line,
            )
            if m:
                r["ic_cpu_setup_ms"] = float(m.group(1)); r["ic_cpu_solve_ms"] = float(m.group(2))
                r["ic_cpu_iters"] = int(m.group(3)); r["ic_cpu_total_ms"] = float(m.group(4))
        if "MULTICOLOR_DILU" in line:
            m = re.search(
                r"Setup: ([\d.]+) ms, solve \(GPU\): ([\d.]+) ms, (\d+) iter.*?total: ([\d.]+) ms",
                line,
            )
            if m:
                r["ic_gpu_setup_ms"] = float(m.group(1)); r["ic_gpu_solve_ms"] = float(m.group(2))
                r["ic_gpu_iters"] = int(m.group(3)); r["ic_gpu_total_ms"] = float(m.group(4))

    # AMG CPU
    m = re.search(
        r"AMG \(CPU\):.*?Setup: ([\d.]+) ms, solve: ([\d.]+) ms.*?total: ([\d.]+) ms",
        text, re.S,
    )
    if m:
        r["amg_cpu_total_ms"] = float(m.group(3))

    # LeafOnly
    m = re.search(
        r"LeafOnly:.*?Inference: ([\d.]+) ms, solve: ([\d.]+) ms, (\d+) iter.*?total: ([\d.]+) ms",
        text, re.S,
    )
    if m:
        r["lo_infer_ms"] = float(m.group(1))
        r["lo_solve_ms"] = float(m.group(2))
        r["lo_iters"] = int(m.group(3))
        r["lo_total_ms"] = float(m.group(4))
        m2 = re.search(r"true rel residual.*?=([\d.e+\-]+)", text)
        if m2:
            r["lo_true_resid"] = float(m2.group(1))

    # AMGX GPU
    m = re.search(
        r"AMGX \(GPU\).*?Setup: ([\d.]+) ms, solve: ([\d.]+) ms.*?total: ([\d.]+) ms",
        text, re.S,
    )
    if m:
        r["amgx_setup_ms"] = float(m.group(1))
        r["amgx_solve_ms"] = float(m.group(2))
        r["amgx_total_ms"] = float(m.group(3))

    return r


def speedup_summary(res: dict) -> str:
    lo = res.get("lo_total_ms")
    if lo is None:
        return "LeafOnly not found in output"
    parts = []
    for label, key in [("IC-GPU", "ic_gpu_total_ms"), ("AMGX", "amgx_total_ms"),
                       ("IC-CPU", "ic_cpu_total_ms")]:
        v = res.get(key)
        if v:
            parts.append(f"{label}: {v / lo:.1f}×")
    return f"LeafOnly {lo:.1f}ms  speedup: " + "  ".join(parts)


# ─── Training with LR-convergence continuation ───────────────────────────────
def train_to_convergence(N, d_model, layers, train_dir, weights_path, exp_dir):
    """
    Run training in chunks, continuing from the last LR until it hits 1e-6.
    Returns (final_lr, total_chunks_run).
    """
    lr = INITIAL_LR
    total_chunks = 0

    for chunk in range(MAX_CHUNKS):
        is_continuation = chunk > 0
        log_path = exp_dir / f"train_chunk{chunk:02d}.log"

        cmd = [
            sys.executable, SCRIPTS_DIR / "LeafOnly.py",
            "--data-folder", str(train_dir),
            "--weights-out", str(weights_path),
            "--d-model", str(d_model),
            "--num-layers", str(layers),
            "--steps", str(STEPS_PER_CHUNK),
            "--num-frames", "40",
            "--lr", f"{lr:.6e}",
            "--use-highways",
            "--stop-at-min-lr",
        ]
        if is_continuation:
            cmd.append("--continue-training")
        else:
            cmd.append("--rebuild-context-cache")

        chunk_label = f"chunk {chunk}" + (" (continuation)" if is_continuation else "")
        print(f"  Training {chunk_label}: lr_start={lr:.2e}, steps={STEPS_PER_CHUNK}...")
        rc, out = run_cmd(cmd, log_path=log_path, timeout_s=18000)
        total_chunks += 1

        if rc != 0:
            print(f"  ERROR: training failed rc={rc} on chunk {chunk}")
            return None, total_chunks

        final_lr = parse_final_lr(out)
        print(f"    final lr: {final_lr:.2e}" if final_lr else "    final lr: unknown")

        if lr_converged(final_lr):
            print(f"  LR converged to {final_lr:.2e} after {total_chunks} chunk(s).")
            break

        if final_lr is not None:
            lr = final_lr  # next chunk starts optimizer at the decayed lr

        if chunk == MAX_CHUNKS - 1:
            print(f"  WARNING: hit MAX_CHUNKS={MAX_CHUNKS}, lr={final_lr}; stopping.")

    return final_lr, total_chunks


# ─── Per-experiment runner ────────────────────────────────────────────────────
def run_experiment(N, d_model, layers, dry_run=False):
    name = exp_name(N, d_model, layers)
    exp_dir = EXPERIMENTS_ROOT / name
    shared_data_dir = EXPERIMENTS_ROOT / f"data_N{N}"
    train_dir = shared_data_dir / "train"
    test_dir = shared_data_dir / "test"
    weights_path = exp_dir / "weights.bytes"
    result_file = exp_dir / "result.json"

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  N={N}  d_model={d_model}  layers={layers}  highways=True")
    print(f"{'='*60}")

    if dry_run:
        print("  [dry-run] skipping")
        return None

    exp_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Patch config ──
    patch_config(N)

    # ── 2. Generate data (shared per N) ──
    existing_train = list(train_dir.rglob("nodes.bin")) if train_dir.exists() else []
    if len(existing_train) < 10:
        print(f"  Generating {N}-node data (40 train / 10 test)...")
        rc, _ = run_cmd([
            sys.executable, SCRIPTS_DIR / "generate_dataset.py",
            "--dataset-type", "hard-multiphase",
            "--n-target", str(N),
            "--num-train", "40",
            "--num-test", "10",
            "--train-dir", str(train_dir),
            "--test-dir", str(test_dir),
        ], log_path=shared_data_dir / "generate.log", timeout_s=600)
        if rc != 0:
            print(f"  ERROR: data generation failed (rc={rc})")
            return None
    else:
        print(f"  Data already present ({len(existing_train)} train frames)")

    # ── 3. Train until LR converges ──
    final_lr, chunks = train_to_convergence(N, d_model, layers, train_dir, weights_path, exp_dir)
    if not weights_path.exists():
        print(f"  ERROR: weights not found at {weights_path}")
        return None

    # ── 4. Test on first test frame ──
    inspect_cmd = [
        sys.executable, SCRIPTS_DIR / "InspectModel.py",
        "--test-only",
        "--data-folder", str(test_dir),
        "--weights", str(weights_path),
        "--frame", "0",
        "--use-highways",
    ]
    print("  Testing with InspectModel --test-only...")
    rc, out = run_cmd(inspect_cmd, log_path=exp_dir / "test.log", timeout_s=600)
    if rc != 0:
        print(f"  WARNING: InspectModel returned rc={rc}")

    # ── 5. Parse & save ──
    parsed = parse_inspect_output(out)
    result = {
        "name": name,
        "N": N, "d_model": d_model, "layers": layers, "highways": True,
        "final_lr": final_lr,
        "chunks": chunks,
        "timestamp": datetime.now().isoformat(),
        "metrics": parsed,
    }
    result_file.write_text(json.dumps(result, indent=2))
    print(f"  {speedup_summary(parsed)}")
    return result


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only", type=str, default=None,
        help="Run only experiments whose name contains this substring (e.g. N16384 or d256).",
    )
    parser.add_argument(
        "--skip-done", action="store_true", default=True,
        help="Skip experiments that already have a result.json (default: True).",
    )
    parser.add_argument("--no-skip-done", dest="skip_done", action="store_false")
    args = parser.parse_args()

    EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = EXPERIMENTS_ROOT / "results_summary.json"

    print(f"Experiments root: {EXPERIMENTS_ROOT}")
    print(f"Config:           {CONFIG_PY}")
    print(f"Grid:             {len(EXPERIMENTS)} experiments")
    print(f"Steps/chunk:      {STEPS_PER_CHUNK}  (up to {MAX_CHUNKS} chunks per experiment)")
    print(f"LR convergence:   {LR_CONVERGED:.0e}")

    all_results = []
    if summary_path.exists():
        try:
            all_results = json.loads(summary_path.read_text())
        except Exception:
            pass
    done_names = {r["name"] for r in all_results}

    if not args.dry_run:
        backup_config()

    try:
        for N, d_model, layers in EXPERIMENTS:
            name = exp_name(N, d_model, layers)
            if args.only and args.only.replace("=", "") not in name:
                continue
            if args.skip_done and name in done_names and not args.dry_run:
                if (EXPERIMENTS_ROOT / name / "result.json").exists():
                    print(f"  SKIP (done): {name}")
                    continue

            result = run_experiment(N, d_model, layers, dry_run=args.dry_run)
            if result is not None:
                all_results = [r for r in all_results if r["name"] != name]
                all_results.append(result)
                summary_path.write_text(json.dumps(all_results, indent=2))

    finally:
        if not args.dry_run:
            restore_config()

    # ── Summary table ──
    print(f"\n{'='*90}")
    print("RESULTS SUMMARY")
    print(f"{'='*90}")
    hdr = (f"{'Experiment':<30} {'LO-ms':>8} {'vs IC-GPU':>10} {'vs AMGX':>9}"
           f" {'LO-iters':>9} {'IC-iters':>9} {'final-lr':>10} {'chunks':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(all_results, key=lambda x: (x["N"], x["d_model"], x["layers"])):
        m = r.get("metrics", {})
        lo = m.get("lo_total_ms")
        ic_gpu = m.get("ic_gpu_total_ms")
        amgx = m.get("amgx_total_ms")
        lo_s = f"{lo:.1f}" if lo else "—"
        ic_s = f"{ic_gpu / lo:.1f}×" if (lo and ic_gpu) else "—"
        amgx_s = f"{amgx / lo:.1f}×" if (lo and amgx) else "—"
        lo_it = str(int(m["lo_iters"])) if m.get("lo_iters") else "—"
        ic_it = str(int(m["ic_gpu_iters"])) if m.get("ic_gpu_iters") else "—"
        flr = f"{r['final_lr']:.1e}" if r.get("final_lr") else "—"
        ch = str(r.get("chunks", "—"))
        print(f"{r['name']:<30} {lo_s:>8} {ic_s:>10} {amgx_s:>9} {lo_it:>9} {ic_it:>9} {flr:>10} {ch:>7}")

    print(f"\nFull results: {summary_path}")


if __name__ == "__main__":
    main()
