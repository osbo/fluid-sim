#!/usr/bin/env python3
"""
Training configs for 4096 generalization experiments.

Usage:
  python3 train_generalization_configs.py --list
  python3 train_generalization_configs.py --index 2
  python3 train_generalization_configs.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_BASE = SCRIPTS_DIR / "data" / "generalization_4096"
WEIGHTS_DIR = SCRIPTS_DIR / "weights"


BASE_CONFIGS = [
    dict(
        name="g4096_id_d128_L3_hw",
        train_split="g4096_train_id",
        scale=4096,
        leaf=128,
        d=128,
        L=3,
        hw=True,
    ),
    dict(
        name="g4096_topo_no_closed_d128_L3_hw",
        train_split="g4096_train_topo_no_closed",
        scale=4096,
        leaf=128,
        d=128,
        L=3,
        hw=True,
    ),
    dict(
        name="g4096_param_low_d128_L3_hw",
        train_split="g4096_train_param_low",
        scale=4096,
        leaf=128,
        d=128,
        L=3,
        hw=True,
    ),
    dict(
        name="g4096_combo_no_closed_low_d128_L3_hw",
        train_split="g4096_train_combo_no_closed_low",
        scale=4096,
        leaf=128,
        d=128,
        L=3,
        hw=True,
    ),
]


def _configs_with_suffix(name_suffix: str):
    out = []
    for cfg in BASE_CONFIGS:
        c = dict(cfg)
        c["name"] = f"{c['name']}{name_suffix}"
        out.append(c)
    return out


def build_cmd(cfg, *, loss_mode: str):
    weights_out = WEIGHTS_DIR / f"{cfg['name']}.bytes"
    data_folder = DATA_BASE / cfg["train_split"] / "train"
    cmd = [
        "python3",
        "-u",
        str(SCRIPTS_DIR / "LeafOnly.py"),
        "--data-folder",
        str(data_folder),
        "--leaf-size",
        str(cfg["leaf"]),
        "--max-mixed-size",
        str(cfg["scale"]),
        "--d-model",
        str(cfg["d"]),
        "--num-layers",
        str(cfg["L"]),
        "--num-heads",
        "8",
        "--lr",
        "2e-4",
        "--steps",
        "100000",
        "--weights-out",
        str(weights_out),
        "--track-training",
        "--auto-stop",
        "--loss-mode",
        str(loss_mode),
    ]
    if not cfg["hw"]:
        cmd.append("--no-use-highways")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=None, help="Run a single config by 0-based index.")
    parser.add_argument("--list", action="store_true", help="Print all configs and exit.")
    parser.add_argument(
        "--loss-mode",
        choices=("hutchinson", "sai"),
        default="hutchinson",
        help="LeafOnly loss mode for training.",
    )
    parser.add_argument(
        "--name-suffix",
        type=str,
        default="",
        help="Optional suffix appended to each output model name (example: _sai).",
    )
    args = parser.parse_args()
    configs = _configs_with_suffix(args.name_suffix)

    if args.list:
        print(f"{'Idx':>3}  {'Name':<42}  {'Train Split':<32}  d  L  hw")
        print("-" * 100)
        for i, cfg in enumerate(configs):
            print(
                f"[{i:2d}]  {cfg['name']:<42}  {cfg['train_split']:<32}  "
                f"{cfg['d']:>3}  {cfg['L']}  {'yes' if cfg['hw'] else 'no'}"
            )
        print(f"\nloss_mode={args.loss_mode}, name_suffix='{args.name_suffix}'")
        return

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.index is not None:
        n = len(configs)
        if not (0 <= args.index < n):
            print(f"Error: --index {args.index} out of range [0, {n - 1}]", file=sys.stderr)
            sys.exit(1)
        cfg = configs[args.index]
        print(f"[{args.index}/{n - 1}] {cfg['name']} ({cfg['train_split']})", flush=True)
        subprocess.run(build_cmd(cfg, loss_mode=args.loss_mode), check=True)
    else:
        for i, cfg in enumerate(configs):
            print(f"\n{'=' * 72}", flush=True)
            print(f"[{i + 1}/{len(configs)}] {cfg['name']} ({cfg['train_split']})", flush=True)
            print(f"{'=' * 72}", flush=True)
            subprocess.run(build_cmd(cfg, loss_mode=args.loss_mode), check=True)


if __name__ == "__main__":
    main()
