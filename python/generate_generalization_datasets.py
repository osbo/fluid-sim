#!/usr/bin/env python3
"""
Generate 4096-node multiphase-v2 datasets for generalization testing.

This script builds a small set of named splits used by train/eval generalization sweeps:
  - id: in-distribution reference
  - topo holdout: train excludes "closed", eval closed-only
  - param holdout: train lower contrast, eval higher contrast
  - compositional: train excludes closed + lower contrast, eval closed + higher contrast

Usage:
  python3 generate_generalization_datasets.py
  python3 generate_generalization_datasets.py --pilot
  python3 generate_generalization_datasets.py --only g4096_eval_closed_only g4096_train_id
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
GENERATOR = SCRIPTS_DIR / "generate_dataset.py"
DATA_ROOT = SCRIPTS_DIR / "data" / "generalization_4096"


def _specs():
    return [
        dict(
            name="g4096_train_id",
            num_train=100,
            num_test=20,
            seed=1000,
            rho_heavy_min=5.0,
            rho_heavy=100.0,
            min_barriers=1,
            max_barriers=3,
            allowed_gap_types="top,bottom,middle_hole,closed",
            allowed_orientations="vertical,horizontal",
        ),
        dict(
            name="g4096_train_topo_no_closed",
            num_train=100,
            num_test=20,
            seed=1100,
            rho_heavy_min=5.0,
            rho_heavy=100.0,
            min_barriers=1,
            max_barriers=3,
            allowed_gap_types="top,bottom,middle_hole",
            allowed_orientations="vertical,horizontal",
        ),
        dict(
            name="g4096_train_param_low",
            num_train=100,
            num_test=20,
            seed=1200,
            rho_heavy_min=5.0,
            rho_heavy=50.0,
            min_barriers=1,
            max_barriers=3,
            allowed_gap_types="top,bottom,middle_hole,closed",
            allowed_orientations="vertical,horizontal",
        ),
        dict(
            name="g4096_train_combo_no_closed_low",
            num_train=100,
            num_test=20,
            seed=1300,
            rho_heavy_min=5.0,
            rho_heavy=50.0,
            min_barriers=1,
            max_barriers=3,
            allowed_gap_types="top,bottom,middle_hole",
            allowed_orientations="vertical,horizontal",
        ),
        dict(
            name="g4096_eval_id",
            num_train=0,
            num_test=100,
            seed=2000,
            rho_heavy_min=5.0,
            rho_heavy=100.0,
            min_barriers=1,
            max_barriers=3,
            allowed_gap_types="top,bottom,middle_hole,closed",
            allowed_orientations="vertical,horizontal",
        ),
        dict(
            name="g4096_eval_closed_only",
            num_train=0,
            num_test=100,
            seed=2100,
            rho_heavy_min=5.0,
            rho_heavy=100.0,
            min_barriers=1,
            max_barriers=3,
            allowed_gap_types="closed",
            allowed_orientations="vertical,horizontal",
        ),
        dict(
            name="g4096_eval_param_high",
            num_train=0,
            num_test=100,
            seed=2200,
            rho_heavy_min=50.0,
            rho_heavy=200.0,
            min_barriers=1,
            max_barriers=3,
            allowed_gap_types="top,bottom,middle_hole,closed",
            allowed_orientations="vertical,horizontal",
        ),
        dict(
            name="g4096_eval_combo_closed_high",
            num_train=0,
            num_test=100,
            seed=2300,
            rho_heavy_min=50.0,
            rho_heavy=200.0,
            min_barriers=1,
            max_barriers=3,
            allowed_gap_types="closed",
            allowed_orientations="vertical,horizontal",
        ),
    ]


def _run_one(spec: dict, pilot: bool) -> None:
    train_dir = DATA_ROOT / spec["name"] / "train"
    test_dir = DATA_ROOT / spec["name"] / "test"
    num_train = 8 if pilot and spec["num_train"] > 0 else spec["num_train"]
    num_test = 8 if pilot else spec["num_test"]

    cmd = [
        sys.executable,
        str(GENERATOR),
        "--dataset-type",
        "multiphase-v2",
        "--n-target",
        "4096",
        "--train-dir",
        str(train_dir),
        "--test-dir",
        str(test_dir),
        "--num-train",
        str(num_train),
        "--num-test",
        str(num_test),
        "--seed",
        str(spec["seed"]),
        "--rho-light",
        "1.0",
        "--rho-heavy-min",
        str(spec["rho_heavy_min"]),
        "--rho-heavy",
        str(spec["rho_heavy"]),
        "--min-barriers",
        str(spec["min_barriers"]),
        "--max-barriers",
        str(spec["max_barriers"]),
        "--allowed-gap-types",
        str(spec["allowed_gap_types"]),
        "--allowed-orientations",
        str(spec["allowed_orientations"]),
    ]
    print(f"\n=== {spec['name']} ===", flush=True)
    subprocess.run(cmd, check=True, cwd=str(SCRIPTS_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Optional subset of dataset names to generate.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List dataset names and exit.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Small dry-run: 8 train (if applicable) and 8 test frames per split.",
    )
    args = parser.parse_args()

    specs = _specs()
    if args.list:
        for s in specs:
            print(s["name"])
        return

    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if s["name"] in wanted]
        missing = wanted.difference({s["name"] for s in specs})
        if missing:
            raise SystemExit(f"Unknown split(s): {', '.join(sorted(missing))}")

    for spec in specs:
        _run_one(spec, pilot=bool(args.pilot))

    print("\nDone generating generalization datasets.")


if __name__ == "__main__":
    main()
