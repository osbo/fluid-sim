#!/usr/bin/env python3
"""Convert LeafOnly frame folders into LSP infer.py generated-format dataset.

Input format (per frame):
  frame_XXXX/{edge_index_rows.bin, edge_index_cols.bin, A_values.bin, meta.txt}

Output format:
  <out_dir>/demo.mtx
  <out_dir>/mat/000000.npy, 000001.npy, ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np

try:
    from scipy.io import mmwrite
    from scipy.sparse import coo_matrix
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"scipy is required for conversion (missing: {exc})")


_META_RE = re.compile(r"^\s*numNodes\s*:\s*(\d+)\s*$")


def _read_num_nodes(meta_path: Path) -> int:
    for line in meta_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _META_RE.match(line)
        if m:
            return int(m.group(1))
    raise ValueError(f"Could not parse numNodes from {meta_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True, help="LeafOnly frame root (contains frame_XXXX dirs).")
    p.add_argument("--out-dir", type=Path, required=True, help="Target LSP generated-format directory.")
    args = p.parse_args()

    in_dir = args.input_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    mat_dir = out_dir / "mat"
    mat_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted([d for d in in_dir.iterdir() if d.is_dir() and d.name.startswith("frame_")])
    if not frames:
        raise SystemExit(f"No frame_XXXX directories found in {in_dir}")

    first = frames[0]
    rows0 = np.fromfile(first / "edge_index_rows.bin", dtype=np.uint32)
    cols0 = np.fromfile(first / "edge_index_cols.bin", dtype=np.uint32)
    n = _read_num_nodes(first / "meta.txt")
    if rows0.size != cols0.size:
        raise SystemExit("edge_index row/col size mismatch in first frame")

    # demo.mtx stores fixed sparsity pattern; values are irrelevant for infer.py data loading.
    demo_vals = np.ones(rows0.size, dtype=np.float32)
    demo = coo_matrix((demo_vals, (rows0.astype(np.int64), cols0.astype(np.int64))), shape=(n, n))
    mmwrite(str(out_dir / "demo.mtx"), demo)

    for i, fr in enumerate(frames):
        rows = np.fromfile(fr / "edge_index_rows.bin", dtype=np.uint32)
        cols = np.fromfile(fr / "edge_index_cols.bin", dtype=np.uint32)
        vals = np.fromfile(fr / "A_values.bin", dtype=np.float32)
        if rows.size != cols.size or rows.size != vals.size:
            raise SystemExit(f"Index/value size mismatch in {fr}")
        if not (np.array_equal(rows0, rows) and np.array_equal(cols0, cols)):
            raise SystemExit(f"Topology mismatch in {fr}; expected fixed topology.")
        np.save(mat_dir / f"{i:06d}.npy", vals.astype(np.float32))

    print(f"Converted {len(frames)} frames -> {out_dir}")


if __name__ == "__main__":
    main()
