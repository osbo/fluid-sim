from pathlib import Path

import torch


def _validate_leaf_size(L: int) -> int:
    L = int(L)
    if L < 1 or (L & (L - 1)) != 0:
        raise ValueError(f"LEAF_SIZE must be a positive power of 2, got {L}")
    return L


def _validate_token_pool(leaf: int, p: int, name: str) -> int:
    """Uniform mean-pool factor along leaf tokens: ``p == 1`` is full resolution; ``p > 1`` divides ``LEAF_SIZE`` (power of 2)."""
    p = int(p)
    if p < 1:
        raise ValueError(f"{name} must be >= 1, got {p}")
    if (p & (p - 1)) != 0:
        raise ValueError(f"{name} must be a power of 2, got {p}")
    if leaf % p != 0:
        raise ValueError(f"LEAF_SIZE {leaf} not divisible by {name} {p}")
    return p


# Power of 2; must match ``leaf_size`` in leaf_only_weights header (``read_leaf_only_header``).
# Padded graph size is MAX_MIXED_SIZE; leaf count is MAX_NUM_LEAVES = MAX_MIXED_SIZE // LEAF_SIZE (static H-grid).
LEAF_SIZE = _validate_leaf_size(128)
# Diagonal Transformer + packed diag blocks at LEAF_APPLY_SIZE (checkpoint ``leaf_apply_diag``).
# 1 = full ``LEAF_SIZE`` tokens per leaf; >1 mean-pools within each leaf before the on-diagonal stack (same idea as off).
DIAG_TOKEN_POOL = _validate_token_pool(LEAF_SIZE, 1, "DIAG_TOKEN_POOL")
LEAF_APPLY_SIZE = LEAF_SIZE // DIAG_TOKEN_POOL
# H off-diagonal Transformer + off_diag heads at LEAF_APPLY_SIZE_OFF (checkpoint ``leaf_apply_off``).
# 1 = only H-matrix strip aggregation; >1 adds uniform mean-pool along the strip before the off stack.
OFF_DIAG_TOKEN_POOL = _validate_token_pool(LEAF_SIZE, 4, "OFF_DIAG_TOKEN_POOL")
LEAF_APPLY_SIZE_OFF = LEAF_SIZE // OFF_DIAG_TOKEN_POOL
ATTENTION_HOPS = 1
GLOBAL_FEATURES_DIM = 12

# Hutchinson / probe loss: damped Jacobi on random Z (aligned with train.py and leafonly.eval).
HUTCHINSON_PROBE_JACOBI_STEPS = 2
HUTCHINSON_PROBE_JACOBI_OMEGA = 0.6

# Upper cap on nodes; leaf grid in checkpoints is still sized for this maximum (MAX_NUM_LEAVES = MAX_MIXED_SIZE // LEAF_SIZE).
# Per-frame padded size is ``problem_padded_num_nodes(num_nodes)`` (aligned, min(frame, MAX_MIXED_SIZE)) — no identity tail when smaller.
MAX_MIXED_SIZE = 8192
# Minimum **aligned** active nodes to keep a frame: n_active = ⌊min(num_nodes, MAX)/LEAF⌋·LEAF.
# Must be ≤ your smallest frame's aligned count. Do not set this to MAX_MIXED_SIZE unless every frame has ≥ that many nodes.
MIN_MIXED_SIZE = LEAF_SIZE
# Leaf grid for H-matrix partition (must match padded training N = MAX_MIXED_SIZE).
MAX_NUM_LEAVES = MAX_MIXED_SIZE // LEAF_SIZE


def apply_runtime_sizes(
    leaf_size: int,
    max_mixed_size: int,
    *,
    leaf_apply_diag: int | None = None,
    leaf_apply_off: int | None = None,
) -> None:
    """
    Update LEAF_SIZE / MAX_MIXED_SIZE / token pools and derived globals.

    When ``leaf_apply_diag`` / ``leaf_apply_off`` are set (e.g. from a checkpoint header), token pools are
    derived so ``LEAF_APPLY_*`` match those values. Otherwise existing ``DIAG_TOKEN_POOL`` /
    ``OFF_DIAG_TOKEN_POOL`` are re-validated against the new ``LEAF_SIZE``.

    After changing ``MAX_NUM_LEAVES``, call ``leafonly.hmatrix.rebuild_hmatrix_globals()`` if ``hmatrix``
    was already imported.
    """
    global LEAF_SIZE, DIAG_TOKEN_POOL, OFF_DIAG_TOKEN_POOL
    global LEAF_APPLY_SIZE, LEAF_APPLY_SIZE_OFF, MAX_MIXED_SIZE, MAX_NUM_LEAVES, MIN_MIXED_SIZE
    LEAF_SIZE = _validate_leaf_size(leaf_size)
    if leaf_apply_diag is not None:
        lad = int(leaf_apply_diag)
        if lad <= 0 or LEAF_SIZE % lad != 0:
            raise ValueError(f"leaf_apply_diag={lad} must divide LEAF_SIZE={LEAF_SIZE}")
        DIAG_TOKEN_POOL = LEAF_SIZE // lad
        LEAF_APPLY_SIZE = lad
    else:
        _validate_token_pool(LEAF_SIZE, DIAG_TOKEN_POOL, "DIAG_TOKEN_POOL")
        LEAF_APPLY_SIZE = LEAF_SIZE // DIAG_TOKEN_POOL
    if leaf_apply_off is not None:
        lao = int(leaf_apply_off)
        if lao <= 0 or LEAF_SIZE % lao != 0:
            raise ValueError(f"leaf_apply_off={lao} must divide LEAF_SIZE={LEAF_SIZE}")
        OFF_DIAG_TOKEN_POOL = LEAF_SIZE // lao
        LEAF_APPLY_SIZE_OFF = lao
    else:
        _validate_token_pool(LEAF_SIZE, OFF_DIAG_TOKEN_POOL, "OFF_DIAG_TOKEN_POOL")
        LEAF_APPLY_SIZE_OFF = LEAF_SIZE // OFF_DIAG_TOKEN_POOL
    MAX_MIXED_SIZE = int(max_mixed_size)
    if MAX_MIXED_SIZE < LEAF_SIZE:
        raise ValueError(f"MAX_MIXED_SIZE ({MAX_MIXED_SIZE}) must be >= LEAF_SIZE ({LEAF_SIZE})")
    if MAX_MIXED_SIZE % LEAF_SIZE != 0:
        raise ValueError(
            f"MAX_MIXED_SIZE ({MAX_MIXED_SIZE}) must be divisible by LEAF_SIZE ({LEAF_SIZE}) "
            "so the padded grid is an integer number of leaves."
        )
    MAX_NUM_LEAVES = MAX_MIXED_SIZE // LEAF_SIZE
    MIN_MIXED_SIZE = LEAF_SIZE


def effective_aligned_num_nodes(num_nodes_real: int) -> int:
    """How many nodes from a frame are used before padding to MAX_MIXED_SIZE (full leaves only)."""
    n = int(num_nodes_real)
    cap = min(n, int(MAX_MIXED_SIZE))
    return (cap // int(LEAF_SIZE)) * int(LEAF_SIZE)


def problem_padded_num_nodes(num_nodes_real: int) -> int:
    """
    Padded graph size N for LeafOnly forward, A, and Jacobi: same as the aligned active count
    (no identity tail past the frame). At most MAX_MIXED_SIZE; smaller frames use fewer nodes.
    """
    return int(effective_aligned_num_nodes(num_nodes_real))
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
        "data_folder": script_dir.parent / "Assets" / "StreamingAssets" / "TestData",
        "save_path": script_dir / "leaf_only_weights.bytes",
        "use_global_node": True,
        "use_gcn": True,
        "print_timing": True,
        "max_grad_norm": 1.0,
    }
