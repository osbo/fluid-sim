import struct
from pathlib import Path

import torch
import torch.nn as nn

from .architecture import EDGE_GATE_HIDDEN_DIM


# v1: 32 bytes (8 ints). v2: 36 bytes (9 ints), diag + off apply sizes.
# v3: 40 bytes (10 ints), adds attention_layout_code + TransformerBlock FFN params in body.
#   attention_layout_code: 0 = LxL (no special nodes), 1 = Lx(L+1) (+ block node), 2 = Lx(L+2) (+ matrix node).
# v4: 44 bytes (11 ints), adds decoupled_route_gates (1 = block_route_gate / matrix_route_gate tensors after edge_gate).
# v4+edge_mlp: 52-byte header = v4 + 8 bytes (magic + ver) before tensor blob; LeafBlockAttention edge_gate is 2-layer MLP.
LEAF_ONLY_HEADER_BYTES = 44
CHECKPOINT_EDGE_GATE_EXT_MAGIC = 0x4544474D  # b"MDGE"
CHECKPOINT_EDGE_GATE_EXT_VER = 1


def read_leaf_only_header(path):
    path = Path(path)
    with open(path, "rb") as f:
        header = f.read(LEAF_ONLY_HEADER_BYTES)
    if len(header) < 32:
        raise ValueError("LeafOnly weights file too short")
    decoupled_route_gates = 0
    if len(header) >= 44:
        (
            d_model,
            leaf_size,
            input_dim,
            num_layers,
            num_heads,
            use_gcn,
            num_gcn_layers,
            leaf_apply_diag,
            leaf_apply_off,
            attention_layout_code,
            decoupled_route_gates,
        ) = struct.unpack("<iiiiiiiiiii", header[:44])
        header_bytes = 44
    elif len(header) >= 40:
        d_model, leaf_size, input_dim, num_layers, num_heads, use_gcn, num_gcn_layers, leaf_apply_diag, leaf_apply_off, attention_layout_code = (
            struct.unpack("<iiiiiiiiii", header[:40])
        )
        header_bytes = 40
    elif len(header) >= 36:
        d_model, leaf_size, input_dim, num_layers, num_heads, use_gcn, num_gcn_layers, leaf_apply_diag, leaf_apply_off = (
            struct.unpack("<iiiiiiiii", header[:36])
        )
        header_bytes = 36
        attention_layout_code = -1
    else:
        d_model, leaf_size, input_dim, num_layers, num_heads, use_gcn, num_gcn_layers, leaf_apply_diag = struct.unpack(
            "<iiiiiiii", header[:32]
        )
        leaf_apply_off = leaf_apply_diag
        header_bytes = 32
        attention_layout_code = -1
    if leaf_apply_diag <= 0:
        leaf_apply_diag = leaf_size
    if leaf_apply_off <= 0:
        leaf_apply_off = leaf_apply_diag

    edge_gate_hidden_dim = 0
    if header_bytes == 44:
        with open(path, "rb") as f:
            f.seek(44)
            ext = f.read(8)
        if len(ext) == 8:
            mag, ver = struct.unpack("<ii", ext)
            if mag == CHECKPOINT_EDGE_GATE_EXT_MAGIC and ver == CHECKPOINT_EDGE_GATE_EXT_VER:
                header_bytes = 52
                edge_gate_hidden_dim = int(EDGE_GATE_HIDDEN_DIM)

    return (
        d_model,
        leaf_size,
        input_dim,
        num_layers,
        num_heads,
        use_gcn,
        num_gcn_layers,
        header_bytes,
        leaf_apply_diag,
        leaf_apply_off,
        attention_layout_code,
        decoupled_route_gates,
        edge_gate_hidden_dim,
    )


def leaf_only_arch_from_checkpoint(path) -> dict | None:
    """
    Return architecture fields from ``leaf_only_weights.bytes`` header, or None if ``path`` is missing.

    Used to build ``LeafOnlyNet`` so it matches ``load_leaf_only_weights`` (same as InspectModel).
    """
    p = Path(path)
    if not p.is_file():
        return None
    (
        d_model,
        leaf_size,
        input_dim,
        num_layers,
        num_heads,
        use_gcn,
        num_gcn_layers,
        header_bytes,
        leaf_apply_diag,
        leaf_apply_off,
        attention_layout_code,
        decoupled_route_gates,
        edge_gate_hidden_dim,
    ) = read_leaf_only_header(p)
    return {
        "d_model": int(d_model),
        "leaf_size": int(leaf_size),
        "input_dim": int(input_dim),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "use_gcn": int(use_gcn),
        "num_gcn_layers": int(num_gcn_layers),
        "header_bytes": int(header_bytes),
        "leaf_apply_diag": int(leaf_apply_diag),
        "leaf_apply_off": int(leaf_apply_off),
        "attention_layout_code": int(attention_layout_code),
        "decoupled_route_gates": int(decoupled_route_gates),
        "edge_gate_hidden_dim": int(edge_gate_hidden_dim),
    }


def apply_leaf_only_arch_from_checkpoint_args(args, save_path) -> None:
    """
    If ``save_path`` exists, set ``args.num_layers``, ``args.d_model``, ``args.num_heads`` from the header
    so the model matches the checkpoint (CLI values are overridden when they differ).
    """
    arch = leaf_only_arch_from_checkpoint(save_path)
    if arch is None:
        return
    cli_nl = getattr(args, "num_layers", None)
    cli_dm = getattr(args, "d_model", None)
    cli_nh = getattr(args, "num_heads", None)
    args.num_layers = arch["num_layers"]
    args.d_model = arch["d_model"]
    args.num_heads = arch["num_heads"]
    changed = []
    if cli_nl is not None and int(cli_nl) != arch["num_layers"]:
        changed.append(f"num_layers {cli_nl}→{arch['num_layers']}")
    if cli_dm is not None and int(cli_dm) != arch["d_model"]:
        changed.append(f"d_model {cli_dm}→{arch['d_model']}")
    if cli_nh is not None and int(cli_nh) != arch["num_heads"]:
        changed.append(f"num_heads {cli_nh}→{arch['num_heads']}")
    if changed:
        name = Path(save_path).name
        print(f"  [checkpoint] Using architecture from {name}: " + ", ".join(changed))


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


def _write_edge_gate_mlp(f, attn):
    _write_packed_tensor(f, attn.edge_gate[0].weight.detach().cpu().float(), transpose=True)
    _write_packed_tensor(f, attn.edge_gate[0].bias.detach().cpu().float(), transpose=False)
    _write_packed_tensor(f, attn.edge_gate[2].weight.detach().cpu().float(), transpose=True)
    _write_packed_tensor(f, attn.edge_gate[2].bias.detach().cpu().float(), transpose=False)


def _read_edge_gate_into(f, attn, read_tensor, *, legacy_linear: bool):
    if legacy_linear:
        _read_into(f, attn.edge_gate[2].weight, read_tensor, transpose=True)
        _read_into(f, attn.edge_gate[2].bias, read_tensor, transpose=False)
        nn.init.normal_(attn.edge_gate[0].weight, std=0.01)
        nn.init.zeros_(attn.edge_gate[0].bias)
    else:
        _read_into(f, attn.edge_gate[0].weight, read_tensor, transpose=True)
        _read_into(f, attn.edge_gate[0].bias, read_tensor, transpose=False)
        _read_into(f, attn.edge_gate[2].weight, read_tensor, transpose=True)
        _read_into(f, attn.edge_gate[2].bias, read_tensor, transpose=False)


def _read_gcn_layer_into(f, gcn_layer, read_tensor):
    _read_into(f, gcn_layer.linear_self.weight, read_tensor, transpose=True)
    _read_into(f, gcn_layer.linear_self.bias, read_tensor, transpose=False)
    _read_into(f, gcn_layer.linear_neighbor.weight, read_tensor, transpose=True)
    _read_into(f, gcn_layer.linear_neighbor.bias, read_tensor, transpose=False)
    _read_into(f, gcn_layer.update_gate[0].weight, read_tensor, transpose=True)
    _read_into(f, gcn_layer.update_gate[0].bias, read_tensor, transpose=False)
    _read_into(f, gcn_layer.update_gate[2].weight, read_tensor, transpose=True)
    _read_into(f, gcn_layer.update_gate[2].bias, read_tensor, transpose=False)


def _write_transformer_block(f, block, *, write_route_gates: bool = True):
    _write_packed_tensor(f, block.norm1.weight.detach().cpu().float(), False)
    _write_packed_tensor(f, block.norm1.bias.detach().cpu().float(), False)
    _write_packed_tensor(f, block.attn.qkv.weight.detach().cpu().float(), True)
    _write_packed_tensor(f, block.attn.qkv.bias.detach().cpu().float(), False)
    _write_packed_tensor(f, block.attn.proj.weight.detach().cpu().float(), True)
    _write_packed_tensor(f, block.attn.proj.bias.detach().cpu().float(), False)
    _write_edge_gate_mlp(f, block.attn)
    if write_route_gates and block.attn.block_route_gate is not None:
        _write_packed_tensor(f, block.attn.block_route_gate.weight.detach().cpu().float(), True)
        _write_packed_tensor(f, block.attn.block_route_gate.bias.detach().cpu().float(), False)
    if write_route_gates and block.attn.matrix_route_gate is not None:
        _write_packed_tensor(f, block.attn.matrix_route_gate.weight.detach().cpu().float(), True)
        _write_packed_tensor(f, block.attn.matrix_route_gate.bias.detach().cpu().float(), False)
    _write_packed_tensor(f, block.norm2.weight.detach().cpu().float(), False)
    _write_packed_tensor(f, block.norm2.bias.detach().cpu().float(), False)
    _write_packed_tensor(f, block.mlp[0].weight.detach().cpu().float(), True)
    _write_packed_tensor(f, block.mlp[0].bias.detach().cpu().float(), False)
    _write_packed_tensor(f, block.mlp[2].weight.detach().cpu().float(), True)
    _write_packed_tensor(f, block.mlp[2].bias.detach().cpu().float(), False)


def save_leaf_only_weights(model, path, input_dim=9):
    path = Path(path)
    d_model = model.embed.lift[0].weight.shape[0]
    num_heads = model.blocks[0].attn.num_heads if model.blocks else 4
    gcn_layers = model.embed.gcn if model.embed.gcn is not None else []
    num_gcn_layers = len(gcn_layers)
    use_block_node = model.blocks[0].attn.use_block_node if model.blocks else False
    use_matrix_node = model.blocks[0].attn.use_matrix_node if model.blocks else False
    attention_layout_code = (1 if use_block_node else 0) + (1 if use_matrix_node else 0)
    has_route_gates = bool(
        model.blocks
        and (
            model.blocks[0].attn.block_route_gate is not None
            or model.blocks[0].attn.matrix_route_gate is not None
        )
    )
    with open(path, "wb") as f:
        f.write(
            struct.pack(
                "<iiiiiiiiiii",
                d_model,
                model.leaf_size,
                input_dim,
                len(model.blocks),
                num_heads,
                1,
                num_gcn_layers,
                int(model.leaf_apply_size),
                int(model.leaf_apply_off),
                attention_layout_code,
                1 if has_route_gates else 0,
            )
        )
        f.write(struct.pack("<ii", CHECKPOINT_EDGE_GATE_EXT_MAGIC, CHECKPOINT_EDGE_GATE_EXT_VER))
        _write_packed_tensor(f, model.embed.lift[0].weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.lift[0].bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.lift[2].weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.embed.lift[2].bias.detach().cpu().float(), transpose=False)
        for gcn_layer in gcn_layers:
            _write_gcn_layer(f, gcn_layer)
        _write_packed_tensor(f, model.embed.norm.weight.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.embed.norm.bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.enc_input_proj.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.enc_input_proj.bias.detach().cpu().float(), transpose=False)
        for block in model.blocks:
            _write_transformer_block(f, block, write_route_gates=has_route_gates)
        for block in model.off_diag_blocks:
            _write_transformer_block(f, block, write_route_gates=has_route_gates)
        _write_packed_tensor(f, model.off_diag_head_U.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.off_diag_head_U.bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.off_diag_head_V.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.off_diag_head_V.bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.leaf_head.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.leaf_head.bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.node_u.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.node_u.bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.node_v.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.node_v.bias.detach().cpu().float(), transpose=False)
        _write_packed_tensor(f, model.jacobi_gate.weight.detach().cpu().float(), transpose=True)
        _write_packed_tensor(f, model.jacobi_gate.bias.detach().cpu().float(), transpose=False)


def _read_transformer_block_into(f, block, read_tensor, *, load_route_gates: bool = False, legacy_edge_gate_linear: bool = False):
    _read_into(f, block.norm1.weight, read_tensor, transpose=False)
    _read_into(f, block.norm1.bias, read_tensor, transpose=False)
    _read_into(f, block.attn.qkv.weight, read_tensor, transpose=True)
    _read_into(f, block.attn.qkv.bias, read_tensor, transpose=False)
    _read_into(f, block.attn.proj.weight, read_tensor, transpose=True)
    _read_into(f, block.attn.proj.bias, read_tensor, transpose=False)
    _read_edge_gate_into(f, block.attn, read_tensor, legacy_linear=legacy_edge_gate_linear)
    if load_route_gates:
        if block.attn.block_route_gate is not None:
            _read_into(f, block.attn.block_route_gate.weight, read_tensor, transpose=True)
            _read_into(f, block.attn.block_route_gate.bias, read_tensor, transpose=False)
        if block.attn.matrix_route_gate is not None:
            _read_into(f, block.attn.matrix_route_gate.weight, read_tensor, transpose=True)
            _read_into(f, block.attn.matrix_route_gate.bias, read_tensor, transpose=False)
    _read_into(f, block.norm2.weight, read_tensor, transpose=False)
    _read_into(f, block.norm2.bias, read_tensor, transpose=False)
    _read_into(f, block.mlp[0].weight, read_tensor, transpose=True)
    _read_into(f, block.mlp[0].bias, read_tensor, transpose=False)
    _read_into(f, block.mlp[2].weight, read_tensor, transpose=True)
    _read_into(f, block.mlp[2].bias, read_tensor, transpose=False)


def load_leaf_only_weights(model, path):
    import numpy as np

    path = Path(path)
    result = read_leaf_only_header(path)
    (
        d_model_lo,
        leaf_size_lo,
        input_dim_lo,
        num_layers_lo,
        num_heads_lo,
        use_gcn_file,
        num_gcn_layers_file,
        header_bytes,
        leaf_apply_diag_lo,
        leaf_apply_off_lo,
        attention_layout_code,
        decoupled_route_gates,
        edge_gate_hidden_dim,
    ) = result
    legacy_edge_gate_linear = int(edge_gate_hidden_dim) != int(EDGE_GATE_HIDDEN_DIM)
    if header_bytes < 40:
        raise ValueError(
            f"Checkpoint '{path}' uses old format (header v{'1' if header_bytes < 36 else '2'}); "
            "it predates the TransformerBlock FFN and cannot be loaded into the current model. Retrain from scratch."
        )
    expected_d_model = model.embed.lift[0].weight.shape[0]
    expected_num_heads = model.blocks[0].attn.num_heads if model.blocks else 4
    expected_input_dim = 9
    if d_model_lo != expected_d_model:
        raise ValueError(f"Checkpoint d_model={d_model_lo} != model {expected_d_model}")
    if model.leaf_size != leaf_size_lo or len(model.blocks) != num_layers_lo:
        raise ValueError(f"Checkpoint leaf_size={leaf_size_lo} num_layers={num_layers_lo} != model {model.leaf_size} {len(model.blocks)}")
    if int(model.leaf_apply_size) != int(leaf_apply_diag_lo):
        raise ValueError(
            f"Checkpoint leaf_apply_diag={leaf_apply_diag_lo} != model.leaf_apply_size {model.leaf_apply_size}"
        )
    if int(model.leaf_apply_off) != int(leaf_apply_off_lo):
        raise ValueError(
            f"Checkpoint leaf_apply_off={leaf_apply_off_lo} != model.leaf_apply_off {model.leaf_apply_off}"
        )
    if input_dim_lo != expected_input_dim:
        raise ValueError(f"Checkpoint input_dim={input_dim_lo} != expected {expected_input_dim}")
    if num_heads_lo != expected_num_heads:
        raise ValueError(f"Checkpoint num_heads={num_heads_lo} != model {expected_num_heads}")
    if int(use_gcn_file) != 1:
        raise ValueError(f"Checkpoint use_gcn={use_gcn_file} is unsupported; expected 1")
    if model.blocks:
        ck_ub = attention_layout_code >= 1
        ck_um = attention_layout_code >= 2
        if ck_ub != model.blocks[0].attn.use_block_node or ck_um != model.blocks[0].attn.use_matrix_node:
            layout_names = [f"{leaf_size_lo}x{leaf_size_lo}", f"{leaf_size_lo}x{leaf_size_lo + 1}", f"{leaf_size_lo}x{leaf_size_lo + 2}"]
            raise ValueError(
                f"Checkpoint attention_layout '{layout_names[min(attention_layout_code, 2)]}' "
                f"!= model '{model.blocks[0].attn.attention_layout}'"
            )

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
        model_gcn_layers = model.embed.gcn if model.embed.gcn is not None else []
        if num_gcn_layers_file != len(model_gcn_layers):
            raise ValueError(f"Checkpoint num_gcn_layers={num_gcn_layers_file} != model {len(model_gcn_layers)}")
        for i in range(num_gcn_layers_file):
            _read_gcn_layer_into(f, model_gcn_layers[i], read_tensor)
        _read_into(f, model.embed.norm.weight, read_tensor, transpose=False)
        _read_into(f, model.embed.norm.bias, read_tensor, transpose=False)
        _read_into(f, model.enc_input_proj.weight, read_tensor, transpose=True)
        _read_into(f, model.enc_input_proj.bias, read_tensor, transpose=False)
        _load_rg = int(decoupled_route_gates) == 1
        for block in model.blocks:
            _read_transformer_block_into(
                f, block, read_tensor, load_route_gates=_load_rg, legacy_edge_gate_linear=legacy_edge_gate_linear
            )
        for block in model.off_diag_blocks:
            _read_transformer_block_into(
                f, block, read_tensor, load_route_gates=_load_rg, legacy_edge_gate_linear=legacy_edge_gate_linear
            )
        _read_into(f, model.off_diag_head_U.weight, read_tensor, transpose=True)
        _read_into(f, model.off_diag_head_U.bias, read_tensor, transpose=False)
        _read_into(f, model.off_diag_head_V.weight, read_tensor, transpose=True)
        _read_into(f, model.off_diag_head_V.bias, read_tensor, transpose=False)
        _read_into(f, model.leaf_head.weight, read_tensor, transpose=True)
        _read_into(f, model.leaf_head.bias, read_tensor, transpose=False)
        _read_into(f, model.node_u.weight, read_tensor, transpose=True)
        _read_into(f, model.node_u.bias, read_tensor, transpose=False)
        _read_into(f, model.node_v.weight, read_tensor, transpose=True)
        _read_into(f, model.node_v.bias, read_tensor, transpose=False)
        _read_into(f, model.jacobi_gate.weight, read_tensor, transpose=True)
        _read_into(f, model.jacobi_gate.bias, read_tensor, transpose=False)
        if f.read(1):
            raise ValueError("Checkpoint has trailing bytes; unsupported legacy format.")
