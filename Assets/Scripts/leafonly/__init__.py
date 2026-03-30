import warnings

# torch.compile / Inductor: internal FutureWarning on deprecated torch._prims_common.check (PyTorch issue).
warnings.filterwarnings(
    "ignore",
    message=r".*torch\._prims_common\.check.*",
    category=FutureWarning,
)

from .architecture import (
    LeafOnlyNet,
    apply_block_diagonal_m_into,
    apply_block_diagonal_M,
    apply_block_diagonal_M_physical,
    block_diagonal_m_apply_workspace,
    build_sparse_bsr_preconditioner,
    attention_layout_choices,
    default_attention_layout,
    next_valid_size,
    parse_attention_layout,
    unpack_precond,
)
from .checkpoint import load_leaf_only_weights, save_leaf_only_weights
from .config import (
    LEAF_APPLY_SIZE,
    LEAF_APPLY_SIZE_OFF,
    LEAF_SIZE,
    OFF_DIAG_TOKEN_POOL,
)
from .data import FluidGraphDataset, build_leaf_block_connectivity
from .eval import evaluate_estimator_variance, evaluate_gradient_interference
from .train import train_leaf_only
