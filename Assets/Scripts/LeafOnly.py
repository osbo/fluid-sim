import argparse

from leafonly.architecture import attention_layout_choices, default_attention_layout
from leafonly.config import LEAF_SIZE, fixed_runtime_config, require_cuda_or_mps_device
from leafonly.eval import evaluate_estimator_variance, evaluate_gradient_interference
from leafonly.train import train_leaf_only as _train_leaf_only_impl


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2, help="Fixed architecture depth for LeafOnly (kept at 2).")
    parser.add_argument("--num_heads", type=int, default=8, help="LeafBlockAttention heads; must divide d_model")
    parser.add_argument("--frame", type=int, default=600, help="Frame index to use when --use_single_frame is True. Default: 600.")
    parser.add_argument("--use_single_frame", action="store_true", help="Train on a single frame (--frame) instead of random-frame sampling.")
    parser.add_argument("--num_frames", type=int, default=50, help="When --use_single_frame False: number of frames to randomly sample; 0 = use all frames.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contexts_per_step", type=int, default=4, help="Gradient accumulation: number of random cached contexts per optimizer step.")
    parser.add_argument("--continue_training", action="store_true", help="Load initial weights from the saved .bytes file and continue training from that state.")
    parser.add_argument(
        "--rebuild-context-cache",
        action="store_true",
        help="Ignore TestData/.leafonly_training_context_cache/ and rebuild contexts from frame files.",
    )
    parser.add_argument("--evaluate_gradients", action="store_true", help="Run gradient interference analysis then exit.")
    parser.add_argument(
        "--peek_parameters",
        action="store_true",
        help=(
            "With --evaluate_gradients: print tensor statistics (trainable weights, float/index buffers, "
            "layout hyperparameters). No effect without --evaluate_gradients."
        ),
    )
    parser.add_argument("--num_gcn_layers", type=int, default=2, choices=[2], help="Fixed number of GCN layers (kept at 2).")
    _al = attention_layout_choices(LEAF_SIZE)
    parser.add_argument(
        "--attention_layout",
        type=str,
        default=default_attention_layout(LEAF_SIZE),
        choices=list(_al),
        help=(
            f"Attention layout: {LEAF_SIZE}×{LEAF_SIZE} (nodes only), "
            f"{LEAF_SIZE}×{LEAF_SIZE + 1} (+ block key), "
            f"{LEAF_SIZE}×{LEAF_SIZE + 2} (+ block + matrix key), "
            f"{LEAF_SIZE + 1}×{LEAF_SIZE + 1} (query+key block token; flat-tree GNN). "
            f"Must match leafonly.config.LEAF_SIZE."
        ),
    )
    parser.add_argument("--target_step", type=int, default=10000, help="Step index to track in logs/metrics.")
    parser.add_argument(
        "--probe-vectors",
        type=int,
        default=-1,
        metavar="K",
        help=(
            "Monte-Carlo probe count for ||M A Z - Z||^2 loss (columns of Z). "
            "Default -1: use max(256, ceil(sqrt(n_pad))) as before. "
            "Smaller K (e.g. 64, 128) speeds AZ/MAZ and backward roughly linearly; noisier gradient. "
            "With --evaluate_gradients, the fixed-frame Hutchinson variance probe uses K=256 when this is -1."
        ),
    )
    parser.add_argument(
        "--profile-backward",
        action="store_true",
        help=(
            "On the same step as the detailed timing breakdown (step 300), run torch.profiler around "
            "loss.backward() only and print top CUDA kernels (self time vs total time). CUDA only."
        ),
    )
    parser.add_argument(
        "--leafonly-pcg",
        type=str,
        choices=("bsr", "matrix_free"),
        default="bsr",
        help=(
            "How to apply M in the probe loss (AZ then MAZ): both use batched apply_block_diagonal_M "
            "(same operator as InspectModel’s expanded M; autograd cannot use BSR sparse.mm). "
            "bsr = default, eager apply; matrix_free = torch.compile on bmm + apply (faster on CUDA)."
        ),
    )
    parser.add_argument(
        "--strip_build_mode",
        type=str,
        choices=("einsum", "no_einsum"),
        default="no_einsum",
        help=(
            "How to build H off-diagonal row/column strip means from per-leaf features: "
            "einsum (default) or no_einsum (gather/scatter index_add_, lower asymptotic work for large K)."
        ),
    )
    return parser


def train_leaf_only():
    args = _build_parser().parse_args()
    args.num_layers = 2
    args.num_gcn_layers = 2
    args.use_jacobi = True
    runtime = fixed_runtime_config(__file__)
    runtime["device"] = require_cuda_or_mps_device()
    if args.evaluate_gradients:
        evaluate_gradient_interference(args, runtime)
        evaluate_estimator_variance(args, runtime)
        return
    _train_leaf_only_impl(args, runtime)


if __name__ == "__main__":
    train_leaf_only()