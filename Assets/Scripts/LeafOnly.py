import argparse

from leafonly.config import fixed_runtime_config, require_cuda_or_mps_device
from leafonly.eval import evaluate_gradient_interference
from leafonly.train import train_leaf_only as _train_leaf_only_impl


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2, help="Fixed architecture depth for LeafOnly (kept at 2).")
    parser.add_argument("--num_heads", type=int, default=2, help="LeafBlockAttention heads; must divide d_model")
    parser.add_argument("--frame", type=int, default=600, help="Frame index to use when --use_single_frame is True. Default: 600.")
    parser.add_argument("--use_single_frame", action="store_true", help="Train on a single frame (--frame) instead of random-frame sampling.")
    parser.add_argument("--num_frames", type=int, default=50, help="When --use_single_frame False: number of frames to randomly sample; 0 = use all frames.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contexts_per_step", type=int, default=4, help="Gradient accumulation: number of random cached contexts per optimizer step.")
    parser.add_argument("--continue_training", action="store_true", help="Load initial weights from the saved .bytes file and continue training from that state.")
    parser.add_argument("--evaluate_gradients", action="store_true", help="Run gradient interference analysis then exit.")
    parser.add_argument("--num_gcn_layers", type=int, default=2, choices=[2], help="Fixed number of GCN layers (kept at 2).")
    parser.add_argument("--target_step", type=int, default=10000, help="Step index to track in logs/metrics.")
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
        return
    _train_leaf_only_impl(args, runtime)


if __name__ == "__main__":
    train_leaf_only()