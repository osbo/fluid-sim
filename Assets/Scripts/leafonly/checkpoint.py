import struct
from pathlib import Path

import torch


LEAF_ONLY_HEADER_BYTES = 32
OFF_DIAG_SENTINEL = 0xFFFFFFEA


def read_leaf_only_header(path):
    path = Path(path)
    with open(path, "rb") as f:
        header = f.read(LEAF_ONLY_HEADER_BYTES)
    if len(header) < LEAF_ONLY_HEADER_BYTES:
        raise ValueError("LeafOnly weights file too short")
    d_model, leaf_size, input_dim, num_layers, num_heads, use_gcn, num_gcn_layers, _ = struct.unpack("<iiiiiiii", header)
    return d_model, leaf_size, input_dim, num_layers, num_heads, use_gcn, num_gcn_layers, LEAF_ONLY_HEADER_BYTES


def _write_packed_tensor(f, param, transpose=False):
    import numpy as np

    t = param.float() if isinstance(param, torch.Tensor) else torch.tensor(param)
    if transpose and t.dim() == 2:
        t = t.t()
    arr = t.numpy().astype(np.float16)
    n = arr.size
    pad = 1 if n % 2 else 0
    f.write(arr.tobytes())
    if pad:
        f.write(np.zeros(1, dtype=np.float16).tobytes())


def _read_into(f, param, read_fn, transpose=False):
    t = read_fn(f, param.shape, transpose=transpose).to(param.device)
    with torch.no_grad():
        param.copy_(t)


def _write_gcn_layer(f, gcn_layer):
    _write_packed_tensor(f, gcn_layer.linear_self.weight.detach().cpu().float(), transpose=True)
    _write_packed_tensor(f, gcn_layer.linear_self.bias.detach().cpu().float(), transpose=False)
    _write_packed_tensor(f, gcn_layer.linear_neighbor.weight.detach().cpu().float(), transpose=True)
    _write_packed_tensor(f, gcn_layer.linear_neighbor.bias.detach().cpu().float(), transpose=False)
    _write_packed_tensor(f, gcn_layer.update_gate[0].weight.detach().cpu().float(), transpose=True)
    _write_packed_tensor(f, gcn_layer.update_gate[0].bias.detach().cpu().float(), transpose=False)
    _write_packed_tensor(f, gcn_layer.update_gate[2].weight.detach().cpu().float(), transpose=True)
    _write_packed_tensor(f, gcn_layer.update_gate[2].bias.detach().cpu().float(), transpose=False)


def _read_gcn_layer_into(f, gcn_layer, read_tensor):
    _read_into(f, gcn_layer.linear_self.weight, read_tensor, transpose=True)
    _read_into(f, gcn_layer.linear_self.bias, read_tensor, transpose=False)
    _read_into(f, gcn_layer.linear_neighbor.weight, read_tensor, transpose=True)
    _read_into(f, gcn_layer.linear_neighbor.bias, read_tensor, transpose=False)
    _read_into(f, gcn_layer.update_gate[0].weight, read_tensor, transpose=True)
    _read_into(f, gcn_layer.update_gate[0].bias, read_tensor, transpose=False)
    _read_into(f, gcn_layer.update_gate[2].weight, read_tensor, transpose=True)
    _read_into(f, gcn_layer.update_gate[2].bias, read_tensor, transpose=False)


def save_leaf_only_weights(model, path, input_dim=9):
    path = Path(path)
    d_model = model.embed.lift[0].weight.shape[0]
    num_heads = model.blocks[0].attn.num_heads if model.blocks else 4
    num_gcn_layers = len(model.embed.gcn)
    with open(path, "wb") as f:
        f.write(struct.pack("<iiiiiiii", d_model, model.leaf_size, input_dim, len(model.blocks), num_heads, 1, num_gcn_layers, 0))
        _write_packed_tensor(f, model.embed.lift[0].weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.lift[0].bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.lift[2].weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.lift[2].bias.detach().cpu().float(), transpose=False)
        if getattr(model.embed, "lift_film", None) is not None:
            _write_packed_tensor(f, model.embed.lift_film[0].weight.detach().cpu().float(), transpose=True)
            _write_packed_tensor(f, model.embed.lift_film[0].bias.detach().cpu().float(), transpose=False)
            _write_packed_tensor(f, model.embed.lift_film[2].weight.detach().cpu().float(), transpose=True)
            _write_packed_tensor(f, model.embed.lift_film[2].bias.detach().cpu().float(), transpose=False)
        for gcn_layer in model.embed.gcn:
            _write_gcn_layer(f, gcn_layer)
        _write_packed_tensor(f, model.embed.norm.weight.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.norm.bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.enc_input_proj.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.enc_input_proj.bias.detach().cpu().float(), transpose=False)
        for block in model.blocks:
            _write_packed_tensor(f, block.norm1.weight.detach().cpu().float(), False)
            _write_packed_tensor(f, block.norm1.bias.detach().cpu().float(), False)
            _write_packed_tensor(f, block.attn.qkv.weight.detach().cpu().float(), True)
            _write_packed_tensor(f, block.attn.qkv.bias.detach().cpu().float(), False)
            _write_packed_tensor(f, block.attn.proj.weight.detach().cpu().float(), True)
            _write_packed_tensor(f, block.attn.proj.bias.detach().cpu().float(), False)
            _write_packed_tensor(f, block.attn.edge_gate.weight.detach().cpu().float(), True)
            _write_packed_tensor(f, block.attn.edge_gate.bias.detach().cpu().float(), False)
        _write_packed_tensor(f, model.leaf_head.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.leaf_head.bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.core.jacobi_gate.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.core.jacobi_gate.bias.detach().cpu().float(), transpose=False)
        if getattr(model, "level_off_diag", None) is not None and len(model.level_off_diag) > 0:
            f.write(struct.pack("<II", OFF_DIAG_SENTINEL, int(len(model.level_off_diag))))
            for blk in model.level_off_diag:
                _write_packed_tensor(f, blk.attn_q.weight.detach().cpu().float(), transpose=True)
                _write_packed_tensor(f, blk.attn_q.bias.detach().cpu().float(), transpose=False)
                _write_packed_tensor(f, blk.attn_k.weight.detach().cpu().float(), transpose=True)
                _write_packed_tensor(f, blk.attn_k.bias.detach().cpu().float(), transpose=False)
                _write_packed_tensor(f, blk.attn_v.weight.detach().cpu().float(), transpose=True)
                _write_packed_tensor(f, blk.attn_v.bias.detach().cpu().float(), transpose=False)
                _write_packed_tensor(f, blk.super_edge_gate.weight.detach().cpu().float(), transpose=True)
                _write_packed_tensor(f, blk.super_edge_gate.bias.detach().cpu().float(), transpose=False)
                _write_packed_tensor(f, blk.feature_ln.weight.detach().cpu().float(), transpose=False)
                _write_packed_tensor(f, blk.feature_ln.bias.detach().cpu().float(), transpose=False)
                _write_packed_tensor(f, blk.proj_U.weight.detach().cpu().float(), transpose=True)
                _write_packed_tensor(f, blk.proj_U.bias.detach().cpu().float(), transpose=False)
                _write_packed_tensor(f, blk.proj_V.weight.detach().cpu().float(), transpose=True)
                _write_packed_tensor(f, blk.proj_V.bias.detach().cpu().float(), transpose=False)


def load_leaf_only_weights(model, path):
    import numpy as np

    path = Path(path)
    result = read_leaf_only_header(path)
    d_model_lo, leaf_size_lo, input_dim_lo, num_layers_lo, num_heads_lo, use_gcn_file, num_gcn_layers_file, header_bytes = result
    expected_d_model = model.embed.lift[0].weight.shape[0]
    expected_num_heads = model.blocks[0].attn.num_heads if model.blocks else 4
    expected_input_dim = 9
    if d_model_lo != expected_d_model:
        raise ValueError(f"Checkpoint d_model={d_model_lo} != model {expected_d_model}")
    if model.leaf_size != leaf_size_lo or len(model.blocks) != num_layers_lo:
        raise ValueError(f"Checkpoint leaf_size={leaf_size_lo} num_layers={num_layers_lo} != model {model.leaf_size} {len(model.blocks)}")
    if input_dim_lo != expected_input_dim:
        raise ValueError(f"Checkpoint input_dim={input_dim_lo} != expected {expected_input_dim}")
    if num_heads_lo != expected_num_heads:
        raise ValueError(f"Checkpoint num_heads={num_heads_lo} != model {expected_num_heads}")
    if int(use_gcn_file) != 1:
        raise ValueError(f"Checkpoint use_gcn={use_gcn_file} is unsupported; expected 1")

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

    with open(path, "rb") as f:
        f.seek(header_bytes)
        _read_into(f, model.embed.lift[0].weight, read_tensor, transpose=True)
        _read_into(f, model.embed.lift[0].bias, read_tensor, transpose=False)
        _read_into(f, model.embed.lift[2].weight, read_tensor, transpose=True)
        _read_into(f, model.embed.lift[2].bias, read_tensor, transpose=False)
        if getattr(model.embed, "lift_film", None) is not None:
            _read_into(f, model.embed.lift_film[0].weight, read_tensor, transpose=True)
            _read_into(f, model.embed.lift_film[0].bias, read_tensor, transpose=False)
            _read_into(f, model.embed.lift_film[2].weight, read_tensor, transpose=True)
            _read_into(f, model.embed.lift_film[2].bias, read_tensor, transpose=False)
        if num_gcn_layers_file != len(model.embed.gcn):
            raise ValueError(f"Checkpoint num_gcn_layers={num_gcn_layers_file} != model {len(model.embed.gcn)}")
        for i in range(num_gcn_layers_file):
            _read_gcn_layer_into(f, model.embed.gcn[i], read_tensor)
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
        _read_into(f, model.core.jacobi_gate.weight, read_tensor, transpose=True)
        _read_into(f, model.core.jacobi_gate.bias, read_tensor, transpose=False)
        extra = f.read(8)
        if len(extra) != 8:
            raise ValueError("Checkpoint missing off-diagonal tail.")
        sentinel, n_blocks_file = struct.unpack("<II", extra)
        if sentinel != OFF_DIAG_SENTINEL:
            raise ValueError(f"Unsupported off-diagonal sentinel: {sentinel}")
        if int(n_blocks_file) != len(model.level_off_diag):
            raise ValueError(f"Checkpoint off-diagonal block count {int(n_blocks_file)} != model {len(model.level_off_diag)}")
        for i in range(int(n_blocks_file)):
            blk = model.level_off_diag[i]
            _read_into(f, blk.attn_q.weight, read_tensor, transpose=True)
            _read_into(f, blk.attn_q.bias, read_tensor, transpose=False)
            _read_into(f, blk.attn_k.weight, read_tensor, transpose=True)
            _read_into(f, blk.attn_k.bias, read_tensor, transpose=False)
            _read_into(f, blk.attn_v.weight, read_tensor, transpose=True)
            _read_into(f, blk.attn_v.bias, read_tensor, transpose=False)
            _read_into(f, blk.super_edge_gate.weight, read_tensor, transpose=True)
            _read_into(f, blk.super_edge_gate.bias, read_tensor, transpose=False)
            _read_into(f, blk.feature_ln.weight, read_tensor, transpose=False)
            _read_into(f, blk.feature_ln.bias, read_tensor, transpose=False)
            _read_into(f, blk.proj_U.weight, read_tensor, transpose=True)
            _read_into(f, blk.proj_U.bias, read_tensor, transpose=False)
            _read_into(f, blk.proj_V.weight, read_tensor, transpose=True)
            _read_into(f, blk.proj_V.bias, read_tensor, transpose=False)

