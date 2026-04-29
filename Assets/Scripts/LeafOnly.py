import argparse
from pathlib import Path

from leafonly.architecture import attention_layout_choices, default_attention_layout
from leafonly.config import LEAF_SIZE, fixed_runtime_config, require_cuda_or_mps_device
from leafonly.eval import evaluate_estimator_variance, evaluate_gradient_interference
from leafonly.train import train_leaf_only as _train_leaf_only_impl

# Single source of truth for InspectModel.py and training CLI (default, help, checkpoint errors).
DEFAULT_USE_HIGHWAYS = True

USE_HIGHWAYS_HELP = (
    "Use highway + global DC features in each Transformer FFN: concat 4×d_model "
    "(block token, row highway, col highway, mean-pooled stream broadcast) into the MLP; "
    "legacy checkpoints use 3× (no global). "
    "True (--use-highways): after attention every layer; output heads use block tokens only (d_model). "
    "False (default): 1×d_model FFN, no highway gather. "
    "Checkpoint highway_ffn_mlp and ffn_concat_width must match when loading weights."
)

CHECKPOINT_ERR_HIGHWAY_IN_FILE_NEED_CLI_ON = (
    "Checkpoint was saved with highway-in-FFN (header highway_ffn_mlp=1). "
    "Do not pass --no-use-highways when loading this file."
)

CHECKPOINT_ERR_NO_HIGHWAY_IN_FILE_NEED_CLI_OFF = (
    "Checkpoint has no highway-in-FFN flag (legacy or trained with --no-use-highways). "
    "Use LeafOnly.py default (omit --use-highways), or pass --no-use-highways explicitly; "
    "retrain with --use-highways for weights that include highway FFN."
)


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-folder",
        "--dataset-folders",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Directory to scan for frames (rglob nodes.bin under it). "
            "Default: StreamingAssets/TestData from fixed_runtime_config. "
            "Repo train.py maps --dataset-folders to this flag."
        ),
    )
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Transformer depth: number of diagonal-leaf blocks and H-off blocks (each stack). Checkpoint must match.",
    )
    parser.add_argument("--num-heads", type=int, default=8, help="LeafBlockAttention heads; must divide d_model")
    parser.add_argument("--frame", type=int, default=600, help="Frame index to use when --use-single-frame is True. Default: 600.")
    parser.add_argument("--use-single-frame", action="store_true", help="Train on a single frame (--frame) instead of random-frame sampling.")
    parser.add_argument("--num-frames", type=int, default=50, help="When --use-single-frame False: number of frames to randomly sample; 0 = use all frames.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contexts-per-step", type=int, default=4, help="Gradient accumulation: number of random cached contexts per optimizer step.")
    parser.add_argument("--continue-training", action="store_true", help="Load initial weights from the saved .bytes file and continue training from that state.")
    parser.add_argument(
        "--grad-mags",
        action="store_true",
        help=(
            "On each loss log line (every 100 steps): print per-module L2 grad norms (gTot, gDiag0, …). "
            "When omitted, skip those prints and skip extra device sync used only for those metrics."
        ),
    )
    parser.add_argument(
        "--rebuild-context-cache",
        action="store_true",
        help="Ignore TestData/.leafonly_training_context_cache/ and rebuild contexts from frame files.",
    )
    parser.add_argument(
        "--evaluate-gradients",
        action="store_true",
        help="Run gradient interference analysis then exit (includes component timing + attention profiler for all Leaf/H-off layers).",
    )
    parser.add_argument(
        "--profile-attention",
        action="store_true",
        help=(
            "With --evaluate-gradients: run attention with eager materialized softmax (not fused SDPA) when "
            "building the attention mass table so masses match the forward path. Omit for faster profiling "
            "(mass rows show '-'; fused SDPA does not expose weights for that table)."
        ),
    )
    parser.add_argument(
        "--peek-parameters",
        action="store_true",
        help=(
            "With --evaluate-gradients: print tensor statistics (trainable weights, float/index buffers, "
            "layout hyperparameters). No effect without --evaluate-gradients."
        ),
    )
    parser.add_argument("--num-gcn-layers", type=int, default=2, choices=[2], help="Fixed number of GCN layers (kept at 2).")
    _al = attention_layout_choices(LEAF_SIZE)
    parser.add_argument(
        "--attention-layout",
        type=str,
        default=default_attention_layout(LEAF_SIZE),
        choices=list(_al),
        help=(
            f"Attention keys: {LEAF_SIZE}×{LEAF_SIZE} (nodes only), "
            f"{LEAF_SIZE}×{LEAF_SIZE + 1} (+ block node), "
            f"{LEAF_SIZE}×{LEAF_SIZE + 2} (+ block + matrix node). Must match leafonly.config.LEAF_SIZE."
        ),
    )
    parser.add_argument("--target-step", type=int, default=10000, help="Step index to track in logs/metrics.")
    parser.add_argument(
        "--probe-vectors",
        type=int,
        default=-1,
        metavar="K",
        help=(
            "Monte-Carlo probe count for ||M A Z - Z||^2 loss (columns of Z). "
            "Default -1: use max(64, ceil(sqrt(n_pad))) as before. "
            "Smaller K (e.g. 64, 128) speeds AZ/MAZ and backward roughly linearly; noisier gradient. "
            "With --evaluate-gradients, the fixed-frame Hutchinson variance probe uses K=64 when this is -1."
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
        "--loss-mode",
        type=str,
        choices=("hutchinson",),
        default="hutchinson",
        help=(
            "Training loss. Only hutchinson is supported: cosine-similarity Hutchinson probe loss."
        ),
    )
    parser.add_argument(
        "--leafonly-pcg",
        type=str,
        choices=("bsr", "matrix_free"),
        default="matrix_free",
        help=(
            "How to apply M in the probe loss (AZ then MAZ): both use batched apply_block_diagonal_M "
            "(same operator as InspectModel’s expanded M; autograd cannot use BSR sparse.mm). "
            "matrix_free (default) = torch.compile on bmm + apply (faster on CUDA); bsr = eager apply."
        ),
    )
    parser.add_argument(
        "--strip-build-mode",
        type=str,
        choices=("einsum", "no_einsum"),
        default="einsum",
        help=(
            "How to build H off-diagonal row/column strip means from per-leaf features: "
            "einsum (default) or no_einsum (gather/scatter index_add_, lower asymptotic work for large K)."
        ),
    )
    parser.add_argument(
        "--off-diag-dense-attn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "H off-diagonal Transformer: True (default) = all-ones attention mask (full L×L softmax), ignoring "
            "H-matrix reachability; False (--no-off-diag-dense-attn) = reachability mask. Edge physics features unchanged."
        ),
    )
    parser.add_argument(
        "--diag-dense-attn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Diagonal leaf Transformer: True (default) = full dense L×L softmax (all-ones mask), with edge_feats "
            "[Δx,Δy,Δz,A] per (i,j) like dense off-diag; False (--no-diag-dense-attn) = n-hop reachability mask "
            "and sparse in-leaf edge aggregation."
        ),
    )
    parser.add_argument(
        "--use-highways",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_HIGHWAYS,
        help=USE_HIGHWAYS_HELP,
    )
    return parser


def train_leaf_only():
    args = _build_parser().parse_args()
    args.num_gcn_layers = 2
    args.use_jacobi = True
    runtime = fixed_runtime_config(__file__)
    if args.data_folder is not None:
        runtime["data_folder"] = str(Path(args.data_folder).expanduser().resolve())
    runtime["device"] = require_cuda_or_mps_device()
    if args.evaluate_gradients:
        evaluate_gradient_interference(args, runtime)
        evaluate_estimator_variance(args, runtime)
        return
    _train_leaf_only_impl(args, runtime)


if __name__ == "__main__":
    train_leaf_only()