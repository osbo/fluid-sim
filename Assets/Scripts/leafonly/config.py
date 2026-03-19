import math
from pathlib import Path

import torch


LEAF_SIZE = 32
ATTENTION_HOPS = 1
RANK_BASE_LEVEL1 = 16
OFF_DIAG_SUPER = 32
GLOBAL_FEATURES_DIM = 12

# Trained graph size range
MIN_MIXED_SIZE = 256
MAX_MIXED_SIZE = 256
DEFAULT_MAX_LEVELS = max(1, int(math.log2(max(2, MAX_MIXED_SIZE // LEAF_SIZE))))


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
        "max_levels": DEFAULT_MAX_LEVELS,
        "max_grad_norm": 1.0,
    }
