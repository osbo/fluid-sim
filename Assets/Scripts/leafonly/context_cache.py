"""Disk cache for precomputed training contexts (expensive n-hop connectivity)."""

import hashlib
import json
from pathlib import Path
from typing import Optional

import torch

from .config import (
    ATTENTION_HOPS,
    LEAF_SIZE,
    MAX_MIXED_SIZE,
    MAX_NUM_LEAVES,
    MIN_MIXED_SIZE,
    OFF_DIAG_TOKEN_POOL,
)
from .hmatrix import NUM_HMATRIX_OFF_BLOCKS

CONTEXT_CACHE_VERSION = 10


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def build_training_context_cache_meta(dataset, run_folder: Path, args, frame_indices: list[int]) -> dict:
    fingerprints = []
    for i in frame_indices:
        d = Path(dataset.frame_paths[i])
        fingerprints.append(
            [
                str(d.resolve()),
                _mtime_ns(d / "nodes.bin"),
                _mtime_ns(d / "edge_index_rows.bin"),
                _mtime_ns(d / "A_values.bin"),
            ]
        )
    return {
        "v": CONTEXT_CACHE_VERSION,
        "run": str(Path(run_folder).resolve()),
        "frame_indices": [int(x) for x in frame_indices],
        "use_single_frame": bool(getattr(args, "use_single_frame", False)),
        "single_frame": int(getattr(args, "frame", 0)),
        "seed": int(getattr(args, "seed", 0)),
        "num_frames": int(getattr(args, "num_frames", 0)),
        "leaf_size": int(LEAF_SIZE),
        "off_diag_token_pool": int(OFF_DIAG_TOKEN_POOL),
        "attention_hops": int(ATTENTION_HOPS),
        "min_mixed": int(MIN_MIXED_SIZE),
        "max_mixed": int(MAX_MIXED_SIZE),
        "max_num_leaves": int(MAX_NUM_LEAVES),
        "hmatrix_off_blocks": int(NUM_HMATRIX_OFF_BLOCKS),
        "off_diag_dense_attn": bool(getattr(args, "off_diag_dense_attn", True)),
        "dataset_len": int(len(dataset)),
        "fingerprints": fingerprints,
    }


def _canonical_json(meta: dict) -> str:
    return json.dumps(meta, sort_keys=True, separators=(",", ":"), default=str)


def _meta_digest(meta: dict) -> str:
    return hashlib.sha256(_canonical_json(meta).encode("utf-8")).hexdigest()[:20]


def training_context_cache_path(cache_dir: Path, meta: dict) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"ctx_v{meta['v']}_{_meta_digest(meta)}.pt"


def _serialize_context(ctx: dict) -> dict:
    lm, lf, om, off = ctx["precomputed_leaf_connectivity"]
    out = {
        "n_pad": int(ctx["n_pad"]),
        "n_orig": int(ctx["n_orig"]),
        "num_leaves": int(ctx["num_leaves"]),
        "batch_vectors": int(ctx["batch_vectors"]),
        "x_input": ctx["x_input"].detach().cpu().contiguous(),
        "edge_index": ctx["edge_index"].detach().cpu().contiguous(),
        "edge_values": ctx["edge_values"].detach().cpu().contiguous(),
        "precomputed_leaf_connectivity": (
            lm.detach().cpu().contiguous(),
            lf.detach().cpu().contiguous(),
            om.detach().cpu().contiguous(),
            off.detach().cpu().contiguous(),
        ),
        "global_features": ctx["global_features"].detach().cpu().contiguous(),
        "jacobi_inv_diag": ctx["jacobi_inv_diag"].detach().cpu().contiguous(),
    }
    if ctx.get("use_dense_A"):
        out["use_dense_A"] = True
        out["A_dense"] = ctx["A_dense"].detach().cpu().contiguous()
    else:
        out["use_dense_A"] = False
        sp = ctx["A_sp"].detach().cpu().coalesce()
        out["A_sp_indices"] = sp.indices().contiguous()
        out["A_sp_values"] = sp.values().contiguous()
        out["A_sp_size"] = [int(sp.size(0)), int(sp.size(1))]
    return out


def _deserialize_context(entry: dict, device: torch.device) -> dict:
    lm, lf, om, off = entry["precomputed_leaf_connectivity"]
    use_dense = bool(entry.get("use_dense_A", True))
    if use_dense:
        A_dense = entry["A_dense"].to(device)
        A_sp = None
    else:
        A_dense = None
        A_sp = torch.sparse_coo_tensor(
            entry["A_sp_indices"],
            entry["A_sp_values"],
            tuple(entry["A_sp_size"]),
            dtype=entry["A_sp_values"].dtype,
            device=device,
        ).coalesce()
    return {
        "n_pad": entry["n_pad"],
        "n_orig": entry["n_orig"],
        "num_leaves": entry["num_leaves"],
        "batch_vectors": entry["batch_vectors"],
        "x_input": entry["x_input"].to(device),
        "edge_index": entry["edge_index"].to(device),
        "edge_values": entry["edge_values"].to(device),
        "use_dense_A": use_dense,
        "A_dense": A_dense,
        "A_sp": A_sp,
        "precomputed_leaf_connectivity": (
            lm.to(device),
            lf.to(device),
            om.to(device),
            off.to(device),
        ),
        "global_features": entry["global_features"].to(device),
        "jacobi_inv_diag": entry["jacobi_inv_diag"].to(device),
    }


def load_training_contexts_from_cache(cache_dir: Path, meta: dict):
    path = training_context_cache_path(cache_dir, meta)
    if not path.is_file():
        return None
    try:
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if blob.get("v") != CONTEXT_CACHE_VERSION:
        return None
    blob_meta = blob.get("meta")
    if blob_meta is None:
        return None
    if _canonical_json(blob_meta) != _canonical_json(meta):
        return None
    entries = blob.get("entries")
    if not isinstance(entries, list) or len(entries) == 0:
        return None
    return entries


def move_training_context_entries_to_device(entries, device: torch.device):
    return [_deserialize_context(e, device) for e in entries]


def save_training_contexts_to_cache(cache_dir: Path, meta: dict, training_contexts: list) -> Optional[Path]:
    path = training_context_cache_path(cache_dir, meta)
    try:
        entries = [_serialize_context(c) for c in training_contexts]
        torch.save({"v": CONTEXT_CACHE_VERSION, "meta": meta, "entries": entries}, path)
        return path
    except Exception:
        return None
