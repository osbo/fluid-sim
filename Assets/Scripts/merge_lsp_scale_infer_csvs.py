#!/usr/bin/env python3
"""Merge LSP scale-sweep inference task CSV files into one summary CSV."""

import argparse
import csv
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/home/osbo/ondemand/fluid-sim/Assets/Scripts/results"),
        help="Directory containing lsp_scale_infer_task_*.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <results-dir>/lsp_scale_infer_summary.csv)",
    )
    args = p.parse_args()

    results_dir = args.results_dir
    out = args.out or (results_dir / "lsp_scale_infer_summary.csv")
    parts = sorted(results_dir.glob("lsp_scale_infer_task_*.csv"))
    if not parts:
        raise SystemExit(f"No task files found in {results_dir}")

    with out.open("w", newline="", encoding="utf-8") as fout:
        writer = None
        for pth in parts:
            with pth.open(newline="", encoding="utf-8") as fin:
                reader = csv.DictReader(fin)
                if writer is None:
                    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)

    print(f"Merged {len(parts)} files -> {out}")


if __name__ == "__main__":
    main()
