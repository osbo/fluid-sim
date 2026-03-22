from pathlib import Path

import torch


def _validate_leaf_size(L: int) -> int:
    L = int(L)
    if L < 1 or (L & (L - 1)) != 0:
        raise ValueError(f"LEAF_SIZE must be a positive power of 2, got {L}")
    return L


# Set to any power of 2; training checkpoints store this in the header — match when loading.
LEAF_SIZE = _validate_leaf_size(64)
ATTN_POOL_FACTOR = 2
if LEAF_SIZE % ATTN_POOL_FACTOR != 0:
    raise ValueError(f"LEAF_SIZE {LEAF_SIZE} must be divisible by ATTN_POOL_FACTOR {ATTN_POOL_FACTOR}")
# Packed preconditioner blocks and apply_block_diagonal_M matmuls use this tile (e.g. 32×32 for 64 nodes/leaf).
LEAF_APPLY_SIZE = LEAF_SIZE // ATTN_POOL_FACTOR
# Attention runs at LEAF_APPLY_SIZE per leaf; activations are repeat_interleave’d back to LEAF_SIZE for residuals.
ATTENTION_HOPS = 1
GLOBAL_FEATURES_DIM = 12

MIN_MIXED_SIZE = 256
MAX_MIXED_SIZE = 256


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
