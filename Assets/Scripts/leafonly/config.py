from pathlib import Path

import torch


def _validate_leaf_size(L: int) -> int:
    L = int(L)
    if L < 1 or (L & (L - 1)) != 0:
        raise ValueError(f"LEAF_SIZE must be a positive power of 2, got {L}")
    return L


def _validate_off_diag_token_pool(leaf: int, p: int) -> int:
    """Optional uniform mean-pool factor along each H off-tile strip (after H row/col aggregation).

    ``p == 1``: full ``LEAF_SIZE`` tokens (no extra downsampling). ``p > 1`` must divide ``LEAF_SIZE``;
    attention + off heads run at ``LEAF_SIZE // p`` (power of 2).
    """
    p = int(p)
    if p < 1:
        raise ValueError(f"OFF_DIAG_TOKEN_POOL must be >= 1, got {p}")
    if (p & (p - 1)) != 0:
        raise ValueError(f"OFF_DIAG_TOKEN_POOL must be a power of 2, got {p}")
    if leaf % p != 0:
        raise ValueError(f"LEAF_SIZE {leaf} not divisible by OFF_DIAG_TOKEN_POOL {p}")
    return p


# Power of 2; must match ``leaf_size`` in leaf_only_weights header (``read_leaf_only_header``).
# Padded graph size is MAX_MIXED_SIZE; leaf count is MAX_NUM_LEAVES = MAX_MIXED_SIZE // LEAF_SIZE (static H-grid).
LEAF_SIZE = _validate_leaf_size(32)
# Diagonal preconditioner blocks use full leaf tokens.
LEAF_APPLY_SIZE = LEAF_SIZE
# H off-diagonal Transformer + off_diag heads at LEAF_APPLY_SIZE_OFF (checkpoint ``leaf_apply_off``).
# 1 = only H-matrix strip aggregation; >1 adds uniform mean-pool along the strip before the off stack.
OFF_DIAG_TOKEN_POOL = _validate_off_diag_token_pool(LEAF_SIZE, 1)
LEAF_APPLY_SIZE_OFF = LEAF_SIZE // OFF_DIAG_TOKEN_POOL
ATTENTION_HOPS = 1
GLOBAL_FEATURES_DIM = 12

# Padded problem size for LeafOnlyNet / training contexts / InspectModel (single source of truth).
# MAX_NUM_LEAVES = MAX_MIXED_SIZE // LEAF_SIZE must match the checkpoint layout (same as at train time).
MAX_MIXED_SIZE = 512
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
