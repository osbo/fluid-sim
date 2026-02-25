"""
LeafOnly: train a single 32x32 leaf block (U@U.T output) with no HODLR/hierarchy.
Uses first 32 nodes only; for dialing in leaf architecture and comparing with full HGT_OL.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import argparse
import struct
from pathlib import Path
import time

from NeuralPreconditioner import (
    FluidGraphDataset,
    build_leaf_block_connectivity,
    LeafBlockAttention,
    TransformerBlock,
    PhysicsAwareEmbedding,
)

LEAF_SIZE = 32


class LeafOnlyNet(nn.Module):
    """
    Single leaf (32 nodes): embed -> transformer blocks (leaf attention) -> leaf_head -> U@U.T + Jacobi diag.
    No HODLR, no downsampling. Output shape: (B, 1, 32, 32) for the one block.
    """
    def __init__(self, input_dim=4, d_model=128, leaf_size=32, num_layers=3, num_heads=4):
        super().__init__()
        self.leaf_size = leaf_size
        self.embed = PhysicsAwareEmbedding(input_dim, d_model)
        self.enc_input_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, block_size=leaf_size, attn_module=LeafBlockAttention(d_model, leaf_size, num_heads=num_heads))
            for _ in range(num_layers)
        ])
        self.leaf_head = nn.Linear(d_model, leaf_size)
        nn.init.normal_(self.leaf_head.weight, std=0.01)
        nn.init.constant_(self.leaf_head.bias, 0.0)
        self.log_hodlr_scale_leaf = nn.Parameter(torch.ones(1) * math.log(1e-2))

    def forward(self, x, edge_index=None, edge_values=None, scale_A=None, save_attention=False):
        """
        x: (B, 32, input_dim). Returns leaf_blocks (B, 1, 32, 32).
        """
        B, N, _ = x.shape
        assert N == self.leaf_size, f"LeafOnly expects exactly {self.leaf_size} nodes, got {N}"
        positions = x[..., :3]

        h = self.embed(x, edge_index, edge_values, scale_A)
        h = self.enc_input_proj(h)

        for block in self.blocks:
            h = block(h, edge_index=edge_index, edge_values=edge_values, positions=positions, scale_A=scale_A, save_attention=save_attention)

        # Single leaf: (B, 1, 32, d_model) -> leaf_head -> (B, 1, 32, 32)
        h_leaves = h.view(B, 1, self.leaf_size, -1)
        u_leaf = self.leaf_head(h_leaves)  # (B, 1, 32, leaf_size)
        leaf_base = torch.matmul(u_leaf, u_leaf.transpose(-1, -2)) * torch.exp(self.log_hodlr_scale_leaf)

        # Jacobi diagonal from input diagonal
        diag_A_n = x[..., 3]
        s = scale_A if scale_A is not None else 1.0
        if isinstance(s, torch.Tensor):
            diag_A_real = diag_A_n * s
        else:
            diag_A_real = diag_A_n * s
        jacobi_diag = torch.zeros_like(diag_A_real)
        mask = diag_A_real.abs() > 1e-6
        jacobi_diag[mask] = 1.0 / diag_A_real[mask]
        jacobi_diag_blocks = jacobi_diag.view(B, 1, self.leaf_size)
        mask_blocks = mask.view(B, 1, self.leaf_size)
        new_diag = torch.where(mask_blocks, jacobi_diag_blocks, torch.ones_like(jacobi_diag_blocks))
        dense_blocks = leaf_base + torch.diag_embed(new_diag)

        return dense_blocks  # (B, 1, 32, 32)


def apply_leaf_only(leaf_blocks, x):
    """leaf_blocks (B, 1, 32, 32), x (B, 32, K). Returns (B, 32, K)."""
    B, N, K = x.shape
    x_leaves = x.view(B, 1, N, K)
    y_leaves = torch.matmul(leaf_blocks, x_leaves)
    return y_leaves.view(B, N, K)


# --- Save / Load (for InspectModel) ---

def read_leaf_only_header(path):
    path = Path(path)
    with open(path, 'rb') as f:
        header = f.read(16)
    if len(header) < 16:
        raise ValueError("LeafOnly weights file too short")
    # d_model, leaf_size, input_dim, num_layers
    d_model, leaf_size, input_dim, num_layers = struct.unpack('<iiii', header)
    return d_model, leaf_size, input_dim, num_layers


def save_leaf_only_weights(model, path, input_dim=4):
    path = Path(path)
    with open(path, 'wb') as f:
        f.write(struct.pack('<iiii', model.embed.lift[0].weight.shape[0], model.leaf_size, input_dim, len(model.blocks)))
        # Embed: lift, gcn, norm
        _write_packed_tensor(f, model.embed.lift[0].weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.lift[0].bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.lift[2].weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.lift[2].bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.gcn.linear_self.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.gcn.linear_self.bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.gcn.linear_neighbor.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.gcn.linear_neighbor.bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.gcn.update_gate[0].weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.gcn.update_gate[0].bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.gcn.update_gate[2].weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.gcn.update_gate[2].bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.norm.weight.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.norm.bias.detach().cpu().float(), transpose=False)
        # enc_input_proj
        _write_packed_tensor(f, model.enc_input_proj.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.enc_input_proj.bias.detach().cpu().float(), transpose=False)
        # Blocks
        for block in model.blocks:
            _write_packed_tensor(f, block.norm1.weight.detach().cpu().float(), False)
            _write_packed_tensor(f, block.norm1.bias.detach().cpu().float(), False)
            _write_packed_tensor(f, block.attn.qkv.weight.detach().cpu().float(), True)
            _write_packed_tensor(f, block.attn.qkv.bias.detach().cpu().float(), False)
            _write_packed_tensor(f, block.attn.proj.weight.detach().cpu().float(), True)
            _write_packed_tensor(f, block.attn.proj.bias.detach().cpu().float(), False)
            _write_packed_tensor(f, block.attn.edge_gate.weight.detach().cpu().float(), True)
            _write_packed_tensor(f, block.attn.edge_gate.bias.detach().cpu().float(), False)
        # leaf_head + scale
        _write_packed_tensor(f, model.leaf_head.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.leaf_head.bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, torch.exp(model.log_hodlr_scale_leaf).detach().cpu().float(), transpose=False)


def _write_packed_tensor(f, param, transpose=False):
    import numpy as np
    t = param.float() if isinstance(param, torch.Tensor) else torch.tensor(param)
    if transpose and t.dim() == 2:
        t = t.t()
    arr = t.numpy().astype(np.float16)
    n = arr.size
    pad = (1 if n % 2 else 0)
    f.write(arr.tobytes())
    if pad:
        f.write(np.zeros(1, dtype=np.float16).tobytes())


def load_leaf_only_weights(model, path):
    import numpy as np
    path = Path(path)
    with open(path, 'rb') as f:
        header = f.read(16)
        d_model, leaf_size, input_dim, num_layers = struct.unpack('<iiii', header)
    if model.leaf_size != leaf_size or len(model.blocks) != num_layers:
        raise ValueError(f"Checkpoint leaf_size={leaf_size} num_layers={num_layers} != model {model.leaf_size} {len(model.blocks)}")

    def read_tensor(f, shape, transpose=False):
        num_elements = int(torch.Size(shape).numel())
        read_len = num_elements + (1 if num_elements % 2 else 0)
        buf = f.read(read_len * 2)
        packed = np.frombuffer(buf, dtype=np.uint32)
        data_fp16 = packed.view(np.float16)
        if num_elements % 2 != 0:
            data_fp16 = data_fp16[:-1]
        data_fp32 = torch.from_numpy(data_fp16.astype(np.float32))
        if transpose and len(shape) == 2:
            data_fp32 = data_fp32.view(shape[1], shape[0]).t()
        else:
            data_fp32 = data_fp32.view(shape)
        return data_fp32

    with open(path, 'rb') as f:
        f.seek(16)
        # Embed
        _read_into(f, model.embed.lift[0].weight, read_tensor, transpose=True)
        _read_into(f, model.embed.lift[0].bias, read_tensor, transpose=False)
        _read_into(f, model.embed.lift[2].weight, read_tensor, transpose=True)
        _read_into(f, model.embed.lift[2].bias, read_tensor, transpose=False)
        _read_into(f, model.embed.gcn.linear_self.weight, read_tensor, transpose=True)
        _read_into(f, model.embed.gcn.linear_self.bias, read_tensor, transpose=False)
        _read_into(f, model.embed.gcn.linear_neighbor.weight, read_tensor, transpose=True)
        _read_into(f, model.embed.gcn.linear_neighbor.bias, read_tensor, transpose=False)
        _read_into(f, model.embed.gcn.update_gate[0].weight, read_tensor, transpose=True)
        _read_into(f, model.embed.gcn.update_gate[0].bias, read_tensor, transpose=False)
        _read_into(f, model.embed.gcn.update_gate[2].weight, read_tensor, transpose=True)
        _read_into(f, model.embed.gcn.update_gate[2].bias, read_tensor, transpose=False)
        _read_into(f, model.embed.norm.weight, read_tensor, transpose=False)
        _read_into(f, model.embed.norm.bias, read_tensor, transpose=False)
        _read_into(f, model.enc_input_proj.weight, read_tensor, transpose=True)
        _read_into(f, model.enc_input_proj.bias, read_tensor, transpose=False)
        for block in model.blocks:
            _read_into(f, block.norm1.weight, read_tensor, transpose=False)
            _read_into(f, block.norm1.bias, read_tensor, transpose=False)
            _read_into(f, block.attn.qkv.weight, read_tensor, transpose=True)
            _read_into(f, block.attn.qkv.bias, read_tensor, transpose=False)
            _read_into(f, block.attn.proj.weight, read_tensor, transpose=True)
            _read_into(f, block.attn.proj.bias, read_tensor, transpose=False)
            _read_into(f, block.attn.edge_gate.weight, read_tensor, transpose=True)
            _read_into(f, block.attn.edge_gate.bias, read_tensor, transpose=False)
        _read_into(f, model.leaf_head.weight, read_tensor, transpose=True)
        _read_into(f, model.leaf_head.bias, read_tensor, transpose=False)
        scale = read_tensor(f, (1,), transpose=False).to(model.log_hodlr_scale_leaf.device)
        with torch.no_grad():
            model.log_hodlr_scale_leaf.copy_(torch.log(scale.clamp(min=1e-12)))


def _read_into(f, param, read_fn, transpose=False):
    t = read_fn(f, param.shape, transpose=transpose).to(param.device)
    with torch.no_grad():
        param.copy_(t)


def train_leaf_only():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    default_data = script_dir.parent / "StreamingAssets" / "TestData"
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--data_folder', type=str, default=str(default_data))
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--frame', type=int, default=600)
    parser.add_argument('--save', type=str, default=str(script_dir / "leaf_only_weights.bytes"))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using device: {device}")

    data_path = Path(args.data_folder)
    if not data_path.exists():
        raise SystemExit(f"Data folder not found: {data_path}")
    dataset = FluidGraphDataset([data_path])
    if len(dataset) == 0:
        raise SystemExit("No frames found.")
    frame_idx = min(args.frame, len(dataset) - 1)
    batch = dataset[frame_idx]

    n = LEAF_SIZE
    x_full = batch['x']  # (num_nodes, 4)
    x_input = x_full[:n].unsqueeze(0).to(device)  # (1, 32, 4)
    rows, cols = batch['edge_index'][0], batch['edge_index'][1]
    mask = (rows < n) & (cols < n)
    edge_index = batch['edge_index'][:, mask].to(device)
    edge_values = batch['edge_values'][mask].to(device)
    scale_A = batch.get('scale_A')
    if scale_A is not None and not isinstance(scale_A, torch.Tensor):
        scale_A = torch.tensor(scale_A, device=device, dtype=x_input.dtype)

    A_indices = batch['edge_index'][:, mask]
    A_vals = batch['edge_values'][mask]
    A_sparse = torch.sparse_coo_tensor(A_indices, A_vals, (n, n)).coalesce()
    if device.type == 'mps':
        A_dense = A_sparse.to_dense().to(device)
    else:
        A_dense = A_sparse.to(device).to_dense()

    model = LeafOnlyNet(input_dim=4, d_model=args.d_model, leaf_size=LEAF_SIZE, num_layers=args.num_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"LeafOnly: first {n} nodes, {edge_index.shape[1]} edges, {args.num_layers} layers, d_model={args.d_model}")

    model.train()
    batch_vectors = 32  # SAI loss: batch of 32 random vectors for stochastic trace estimation
    for step in range(args.steps):
        optimizer.zero_grad()
        leaf_blocks = model(x_input, edge_index=edge_index, edge_values=edge_values, scale_A=scale_A)
        M_block = leaf_blocks[0, 0]  # (32, 32)
        # SAI loss: E_z || M A z - z ||^2 with batch of 32 vectors (all ops MPS-friendly, no SVD)
        Z = torch.randn(n, batch_vectors, device=device, dtype=x_input.dtype)
        AZ = A_dense @ Z  # (32, 32)
        MAZ = M_block @ AZ  # (32, 32)
        residual = MAZ - Z
        loss = (residual ** 2).mean()
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step}: SAI loss (E||MAz-z||²) {loss.item():.6f}")
            save_leaf_only_weights(model, args.save, input_dim=4)

    save_leaf_only_weights(model, args.save, input_dim=4)
    print(f"Saved to {args.save}")


if __name__ == "__main__":
    train_leaf_only()
