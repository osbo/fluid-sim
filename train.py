#!/usr/bin/env python3
"""
Runs ``Assets/Scripts/LeafOnly.py`` (the real training entry point).

Maps common names to LeafOnly flags:
  --dataset-folders DIR / --dataset_folders DIR  ->  --data-folder DIR
  --batch-size N / --batch_size N                ->  --contexts-per-step N

From the repo root:
  python3 train.py --dataset-folders data/hard_multiphase_train --batch-size 4 --lr 1e-4

Equivalent from ``Assets/Scripts`` (do not run ``leafonly/train.py`` directly):
  python3 LeafOnly.py --data-folder ../../data/hard_multiphase_train --contexts-per-step 4 --lr 1e-4
"""
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
LEAF_ONLY = REPO / "python" / "LeafOnly.py"


def _remap_argv(argv: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--dataset-folders", "--dataset_folders") and i + 1 < len(argv):
            out.extend(["--data-folder", argv[i + 1]])
            i += 2
            continue
        if a.startswith("--dataset-folders="):
            out.extend(["--data-folder", a.split("=", 1)[1]])
            i += 1
            continue
        if a.startswith("--dataset_folders="):
            out.extend(["--data-folder", a.split("=", 1)[1]])
            i += 1
            continue
        if a in ("--batch-size", "--batch_size") and i + 1 < len(argv):
            out.extend(["--contexts-per-step", argv[i + 1]])
            i += 2
            continue
        if a.startswith("--batch-size="):
            out.extend(["--contexts-per-step", a.split("=", 1)[1]])
            i += 1
            continue
        if a.startswith("--batch_size="):
            out.extend(["--contexts-per-step", a.split("=", 1)[1]])
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def main() -> None:
    if not LEAF_ONLY.is_file():
        raise SystemExit(f"Missing {LEAF_ONLY}")
    new_argv = [str(LEAF_ONLY)] + _remap_argv(sys.argv[1:])
    sys.argv = new_argv
    runpy.run_path(str(LEAF_ONLY), run_name="__main__")


if __name__ == "__main__":
    main()
