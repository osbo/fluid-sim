from pathlib import Path

import torch


def _validate_leaf_size(L: int) -> int:
    L = int(L)
    if L < 1 or (L & (L - 1)) != 0:
        raise ValueError(f"LEAF_SIZE must be a positive power of 2, got {L}")
    return L


# Set to any power of 2; training checkpoints store this in the header — match when loading.
# H-matrix static buffers use MAX_NUM_LEAVES = MAX_MIXED_SIZE // LEAF_SIZE; wrong LEAF_SIZE breaks apply vs checkpoint.
LEAF_SIZE = _validate_leaf_size(32)
# Diagonal leaf attention + diagonal preconditioner blocks (no downsample when 1).
ATTN_POOL_FACTOR_DIAG = 1
# Off-diagonal pair attention + off blocks (2 ⇒ half resolution inside each 32-node chunk).
ATTN_POOL_FACTOR_OFF = 1
for _name, _pf in (("ATTN_POOL_FACTOR_DIAG", ATTN_POOL_FACTOR_DIAG), ("ATTN_POOL_FACTOR_OFF", ATTN_POOL_FACTOR_OFF)):
    if LEAF_SIZE % _pf != 0:
        raise ValueError(f"LEAF_SIZE {LEAF_SIZE} must be divisible by {_name} {_pf}")
LEAF_APPLY_SIZE = LEAF_SIZE // ATTN_POOL_FACTOR_DIAG
LEAF_APPLY_SIZE_OFF = LEAF_SIZE // ATTN_POOL_FACTOR_OFF
# Back-compat alias (single factor); prefer ATTN_POOL_FACTOR_DIAG / _OFF.
ATTN_POOL_FACTOR = ATTN_POOL_FACTOR_DIAG
ATTENTION_HOPS = 1
GLOBAL_FEATURES_DIM = 12

# Padded problem size for LeafOnlyNet (static H-grid). Checkpoints and MAX_NUM_LEAVES follow this.
MAX_MIXED_SIZE = 1024 
# Minimum **aligned** active nodes to keep a frame: n_active = ⌊min(num_nodes, MAX)/LEAF⌋·LEAF.
# Must be ≤ your smallest frame's aligned count. Do not set this to MAX_MIXED_SIZE unless every frame has ≥ that many nodes.
MIN_MIXED_SIZE = LEAF_SIZE
# Leaf grid for H-matrix partition (must match padded training N = MAX_MIXED_SIZE).
MAX_NUM_LEAVES = MAX_MIXED_SIZE // LEAF_SIZE


def effective_aligned_num_nodes(num_nodes_real: int) -> int:
    """How many nodes from a frame are used before padding to MAX_MIXED_SIZE (full leaves only)."""
    n = int(num_nodes_real)
    cap = min(n, int(MAX_MIXED_SIZE))
    return (cap // int(LEAF_SIZE)) * int(LEAF_SIZE)
# Weak admissibility parameter (same as analytical reference).
HMATRIX_ETA = 1.0


def require_cuda_or_mps_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    raise SystemExit("LeafOnly requires CUDA or MPS; CPU execution path has been removed.")


def fixed_runtime_config(script_file):
    script_dir = Path(script_file).resolve().parent
    return {
        "data_folder": script_dir.parent / "StreamingAssets" / "TestData",
        "save_path": script_dir / "leaf_only_weights.bytes",
        "use_global_node": True,
        "use_gcn": True,
        "print_timing": True,
        "max_grad_norm": 1.0,
    }
