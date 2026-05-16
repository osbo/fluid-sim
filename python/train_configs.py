"""
All training configurations for the LeafOnly ablation study.

Evaluations covered:
  1. Scale sweep (1024–16384): baseline config (d128, L3, highways)
  2. Architecture ablation at scales 2048/4096/8192:
       d_model in {64, 128, 256}
       num_layers in {1, 2, 3}
       use_highways in {True, False}
     NOTE: GCN layers are fixed at 2 — no CLI flag to disable GCN exists yet.
           Add --no-gcn if that ablation is needed.
  3. Training curves / SAI loss: all runs use --track-training so EvalCheckpoints.py
     can be run afterward on the *_step*.bytes checkpoints.

Usage:
  python3 train_configs.py --list              # print all 20 configs
  python3 train_configs.py --index 7           # run one config (for SLURM array)
  python3 train_configs.py                     # run all 20 sequentially (long!)
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_BASE = SCRIPTS_DIR.parent / "data"
WEIGHTS_DIR = SCRIPTS_DIR.parent / "weights"

# ──────────────────────────────────────────────────────────────────────────────
# Config table  (index order = SLURM array task ID)
# ──────────────────────────────────────────────────────────────────────────────
# leaf_size notes:
#   scales 1024–8192  → leaf_size=128 (64 leaves at 8192, fewer at smaller scales)
#   scale   16384     → leaf_size=256 (64 leaves; matches existing run_leaf_only_engaging.sh)
# The ablation (eval 2) uses only scales 2048/4096/8192 with leaf_size=128 throughout,
# so cross-config comparisons are architecturally consistent.

CONFIGS = [
    # ── Scale sweep: baseline config d128 / 3 layers / highways ──────────────
    # [0]
    dict(name="v2_1024_d128_L3_hw",   scale=1024,  leaf=128, d=128, L=3, hw=True),
    # [1]  ← also ablation baseline at scale 2048
    dict(name="v2_2048_d128_L3_hw",   scale=2048,  leaf=128, d=128, L=3, hw=True),
    # [2]  ← also ablation baseline at scale 4096
    dict(name="v2_4096_d128_L3_hw",   scale=4096,  leaf=128, d=128, L=3, hw=True),
    # [3]  ← also ablation baseline at scale 8192
    dict(name="v2_8192_d128_L3_hw",   scale=8192,  leaf=128, d=128, L=3, hw=True),
    # [4]  (leaf_size=256 → 64 leaves, same H-grid depth as [3])
    dict(name="v2_16384_d128_L3_hw",  scale=16384, leaf=256, d=128, L=3, hw=True),

    # ── Ablation: d_model=64 at 2048 / 4096 / 8192 ───────────────────────────
    # [5]
    dict(name="v2_2048_d64_L3_hw",    scale=2048,  leaf=128, d=64,  L=3, hw=True),
    # [6]
    dict(name="v2_4096_d64_L3_hw",    scale=4096,  leaf=128, d=64,  L=3, hw=True),
    # [7]
    dict(name="v2_8192_d64_L3_hw",    scale=8192,  leaf=128, d=64,  L=3, hw=True),

    # ── Ablation: d_model=256 at 2048 / 4096 / 8192 ──────────────────────────
    # [8]
    dict(name="v2_2048_d256_L3_hw",   scale=2048,  leaf=128, d=256, L=3, hw=True),
    # [9]
    dict(name="v2_4096_d256_L3_hw",   scale=4096,  leaf=128, d=256, L=3, hw=True),
    # [10]
    dict(name="v2_8192_d256_L3_hw",   scale=8192,  leaf=128, d=256, L=3, hw=True),

    # ── Ablation: num_layers=1 at 2048 / 4096 / 8192 ─────────────────────────
    # [11]
    dict(name="v2_2048_d128_L1_hw",   scale=2048,  leaf=128, d=128, L=1, hw=True),
    # [12]
    dict(name="v2_4096_d128_L1_hw",   scale=4096,  leaf=128, d=128, L=1, hw=True),
    # [13]
    dict(name="v2_8192_d128_L1_hw",   scale=8192,  leaf=128, d=128, L=1, hw=True),

    # ── Ablation: num_layers=2 at 2048 / 4096 / 8192 ─────────────────────────
    # [14]
    dict(name="v2_2048_d128_L2_hw",   scale=2048,  leaf=128, d=128, L=2, hw=True),
    # [15]
    dict(name="v2_4096_d128_L2_hw",   scale=4096,  leaf=128, d=128, L=2, hw=True),
    # [16]
    dict(name="v2_8192_d128_L2_hw",   scale=8192,  leaf=128, d=128, L=2, hw=True),

    # ── Ablation: no highways at 2048 / 4096 / 8192 ──────────────────────────
    # [17]
    dict(name="v2_2048_d128_L3_nohw", scale=2048,  leaf=128, d=128, L=3, hw=False),
    # [18]
    dict(name="v2_4096_d128_L3_nohw", scale=4096,  leaf=128, d=128, L=3, hw=False),
    # [19]
    dict(name="v2_8192_d128_L3_nohw", scale=8192,  leaf=128, d=128, L=3, hw=False),
]

assert len(CONFIGS) == 20, f"Expected 20 configs, got {len(CONFIGS)}"


def build_cmd(cfg: dict) -> list:
    weights_out = WEIGHTS_DIR / f"{cfg['name']}.bytes"
    data_folder = DATA_BASE / f"multiphase_v2_{cfg['scale']}" / "train"
    cmd = [
        "python3", "-u",
        str(SCRIPTS_DIR / "LeafOnly.py"),
        "--data-folder",    str(data_folder),
        "--leaf-size",      str(cfg["leaf"]),
        "--max-mixed-size", str(cfg["scale"]),
        "--d-model",        str(cfg["d"]),
        "--num-layers",     str(cfg["L"]),
        "--num-heads",      "8",
        "--lr",             "2e-4",
        "--steps",          "100000",
        "--weights-out",    str(weights_out),
        "--track-training",
        "--auto-stop",
    ]
    if not cfg["hw"]:
        cmd.append("--no-use-highways")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=None,
                        help="Run a single config by 0-based index (for SLURM array jobs).")
    parser.add_argument("--list", action="store_true",
                        help="Print all configs and exit.")
    args = parser.parse_args()

    if args.list:
        print(f"{'Idx':>3}  {'Name':<30}  scale   leaf  d_model  layers  highways")
        print("─" * 75)
        for i, cfg in enumerate(CONFIGS):
            hw_str = "yes" if cfg["hw"] else "no"
            print(f"[{i:2d}]  {cfg['name']:<30}  {cfg['scale']:>5}  "
                  f"{cfg['leaf']:>4}    {cfg['d']:>3}       {cfg['L']}       {hw_str}")
        return

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.index is not None:
        n = len(CONFIGS)
        if not (0 <= args.index < n):
            print(f"Error: --index {args.index} out of range [0, {n - 1}]", file=sys.stderr)
            sys.exit(1)
        cfg = CONFIGS[args.index]
        print(f"[{args.index}/{n - 1}] {cfg['name']}", flush=True)
        subprocess.run(build_cmd(cfg), check=True)
    else:
        for i, cfg in enumerate(CONFIGS):
            print(f"\n{'='*60}", flush=True)
            print(f"[{i + 1}/{len(CONFIGS)}] {cfg['name']}", flush=True)
            print(f"{'='*60}", flush=True)
            subprocess.run(build_cmd(cfg), check=True)


if __name__ == "__main__":
    main()
