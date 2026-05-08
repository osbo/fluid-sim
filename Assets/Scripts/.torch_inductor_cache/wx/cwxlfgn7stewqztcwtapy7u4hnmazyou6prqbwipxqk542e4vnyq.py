# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, grid_combo_kernels, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ff/cffoojsmcwtz4th7o5yxfltqww7xewalzndogsqhpnijyyatbrux.py
# Topologically Sorted Source Nodes: [node_feats_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   node_feats_1 => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_1, %expand], -1), kwargs = {})
triton_poi_fused_cat_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 18
    x1 = (xindex // 18)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 6, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (3 + (9*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 18, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((-6) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/nz/cnzdatoqsvt42dccka5txesdryrjvljdihuoal6r4m75c6ezg37q.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_2 => add, erf, mul, mul_1, mul_2
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.5), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_1,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %add), kwargs = {})
triton_poi_fused_gelu_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_1', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/yj/cyjpschwxqi42mnge43ewxlvm2bblgakxzgfbhgrzoordhbowm3y.py
# Topologically Sorted Source Nodes: [aggr], Original ATen: [aten.zeros_like]
# Source node to ATen node mapping:
#   aggr => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([512, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_like_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_like_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (256*x1)), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/6c/c6cikx6sfis57gvtbsanjef63j3wotagkzx7ptimi7hvmzdddfel.py
# Topologically Sorted Source Nodes: [aggr, getitem_4, messages, index_add_], Original ATen: [aten.zeros_like, aten.index, aten.mul, aten.index_add]
# Source node to ATen node mapping:
#   aggr => full_default
#   getitem_4 => index
#   index_add_ => index_put
#   messages => mul_3
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([512, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%addmm_2, [%select_1]), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index, %unsqueeze_1), kwargs = {})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default, [%select], %mul_3, True), kwargs = {})
triton_poi_fused_index_index_add_mul_zeros_like_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_index_add_mul_zeros_like_3', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 424064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128)
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3313 + x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 512, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 512)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 512")
    tmp7 = tmp6 + tmp1
    tmp8 = tmp6 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp6)
    tl.device_assert(((0 <= tmp9) & (tmp9 < 512)) | ~(xmask), "index out of bounds: 0 <= tmp9 < 512")
    tmp11 = tl.load(in_ptr1 + (x0 + (128*tmp9)), xmask)
    tmp13 = tmp11 * tmp12
    tl.atomic_add(out_ptr0 + (x0 + (256*tmp4)), tmp13, xmask, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/t4/ct4nf2caubp4fbedrd5npa4uog5sojvb6734drbodyvt5tm2k56e.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   out => add_2
# Graph fragment:
#   %add_tensor_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_43, %arg15_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze, %add_tensor_43), kwargs = {})
triton_poi_fused_add_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_4', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/5a/c5a3o37ccjd37on4xpberivdl5lh7ufjzx3z4daujccy2kxtx47i.py
# Topologically Sorted Source Nodes: [h_2], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   h_2 => add_5, add_6, mul_11, mul_12, rsqrt, sub, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%unsqueeze_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_4, %getitem_1), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %arg24_1), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %arg25_1), kwargs = {})
triton_per_fused_native_layer_norm_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 128.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp31, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ih/cihx7sew4ooxinpsba2bcondbnn4zisdbxc3ihiuvm635sko776n.py
# Topologically Sorted Source Nodes: [bn_diag], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   bn_diag => mean_2
# Graph fragment:
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_12, [2], True), kwargs = {})
triton_per_fused_mean_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (4096*x1)), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = 32.0
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr1 + (x0 + (4224*x1)), tmp5, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/uw/cuwtmkui5jyzb3ek2hxskdfua6n7mcwlq65eigjlzibvvfddkxbh.py
# Topologically Sorted Source Nodes: [cat_5], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_5 => cat_5
# Graph fragment:
#   %cat_5 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_12, %mean_2], 2), kwargs = {})
triton_poi_fused_cat_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 4096
    x1 = (xindex // 4096)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (4224*x1)), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/vo/cvobundw5jzthn7e6capknc66s2pghhsqcc27kuosdy77v3cwayb.py
# Topologically Sorted Source Nodes: [layer_norm_1], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_1 => add_8, add_9, mul_13, mul_14, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_13, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_13, %getitem_3), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %arg36_1), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %arg37_1), kwargs = {})
triton_per_fused_native_layer_norm_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 528
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 128.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/5m/c5mctk42eh5kfykz7wuyj2gonsnwc7os377meumpp3dp5huyoejl.py
# Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   einsum => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_18,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16) % 33
    x2 = (xindex // 528) % 8
    x3 = (xindex // 4224)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*x2) + (384*x1) + (12672*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/m3/cm3e7kmy6jkbi5n25sjhvfrofk7w5w4qmnb2gurab4c5fxg5wdci.py
# Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   einsum => clone_2
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_19,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 33
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (128 + y0 + (384*x2) + (12672*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (128 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (33*y3)), tmp2, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/u6/cu6k2puav6fg6wybgq5gr4orls43rwdka5g4messg5qkyxl4ilro.py
# Topologically Sorted Source Nodes: [bias_physics_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   bias_physics_1 => cat_7
# Graph fragment:
#   %cat_7 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_3, %full_default_4], 2), kwargs = {})
triton_poi_fused_cat_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 69696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 132) % 33
    x0 = xindex % 132
    x2 = (xindex // 4356)
    x3 = xindex % 4356
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (132*x1) + (4224*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 33, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = 0.0
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.where(tmp4, tmp5, tmp11)
    tl.store(out_ptr0 + (x3 + (4384*x2)), tmp12, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/3a/c3awy7cg3av5jquw6dusvgishptwtoxnvpjvixxhwzeg3yruof57.py
# Topologically Sorted Source Nodes: [bias_physics_1, setitem], Original ATen: [aten.cat, aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   bias_physics_1 => cat_7
#   setitem => full_default_5, index_put_4
# Graph fragment:
#   %cat_7 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_3, %full_default_4], 2), kwargs = {})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_4 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%cat_7, [None, None, %iota, %iota], %full_default_5), kwargs = {})
triton_poi_fused_cat_index_put_lift_fresh_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_index_put_lift_fresh_12', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 4
    x1 = (xindex // 4) % 32
    x2 = (xindex // 128)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (136*x1) + (4384*x2)), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/tu/ctuecadgq4w5podi43uaf2niwjswatqcn5qtsppnr2bfswnf6fvi.py
# Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_1 => full_default_6, index_put_5
# Graph fragment:
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_5 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_7, [None, None, %iota, %iota], %full_default_6), kwargs = {})
triton_poi_fused_index_put_lift_fresh_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_13', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1089
    x1 = (xindex // 1089)
    tmp0 = tl.load(in_ptr0 + (3 + (4*x0) + (4384*x1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0 + (1120*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/e6/ce65ecqtsi77b6sesuksd6t2bd3mm7zgrhuc5z5f6fscfiu64fkf.py
# Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_1 => full_default_6, index_put_5
# Graph fragment:
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_5 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_7, [None, None, %iota, %iota], %full_default_6), kwargs = {})
triton_poi_fused_index_put_lift_fresh_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_14', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp0 = 1.0
    tl.store(out_ptr0 + ((34*x0) + (1120*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/jf/cjf676oxvd7ecqlpy74ifrm4g64piydmmkxhc2dk2frvcquhvbwe.py
# Topologically Sorted Source Nodes: [eq, scores_2, scores, scores_1, attn_probs], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_probs => amax_2, exp, sub_2, sum_1
#   eq => eq
#   scores => mul_15
#   scores_1 => add_10
#   scores_2 => clone_4, full_default_8, where
# Graph fragment:
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%unsqueeze_13, 0), kwargs = {})
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_30, 0.25), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %slice_30), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default_8, %add_10), kwargs = {})
#   %clone_4 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%where,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_2 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_4, [3], True), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_4, %amax_2), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [3], True), kwargs = {})
triton_per_fused__softmax_add_eq_masked_fill_mul_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_eq_masked_fill_mul_15', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4224
    rnumel = 33
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex % 33
    r3 = rindex
    x2 = (xindex // 264)
    x5 = xindex
    x4 = (xindex // 33)
    tmp15 = tl.load(in_ptr1 + (r3 + (33*x5)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr2 + (r3 + (33*x0) + (1120*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr3 + (3 + (4*r3) + (132*x0) + (4384*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r3 + (33*x0) + (1056*x2)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 33, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = 1.0
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.where(tmp4, tmp5, tmp11)
    tmp13 = 0.0
    tmp14 = tmp12 == tmp13
    tmp16 = 0.25
    tmp17 = tmp15 * tmp16
    tmp18 = tl.full([1, 1], 3, tl.int32)
    tmp19 = tmp18 == tmp18
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp17 + tmp22
    tmp24 = float("-inf")
    tmp25 = tl.where(tmp14, tmp24, tmp23)
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, float("-inf"))
    tmp29 = triton_helpers.max2(tmp28, 1)[:, None]
    tmp30 = tmp25 - tmp29
    tmp31 = tl_math.exp(tmp30)
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
    tmp34 = tl.where(rmask & xmask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None]
    tl.store(out_ptr1 + (r3 + (33*x0) + (1120*x4)), tmp31, rmask & xmask)
    tl.store(out_ptr2 + (x5), tmp35, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/cr/ccroqllml5354e7eqacywdcnif2l57gk3wdmfd2nbqrqtsg4mg23.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %select_scatter_default : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%index_put_4, %index_put_5, 4, 3), kwargs = {})
triton_poi_fused_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_16', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 69696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = (xindex // 4) % 1089
    x2 = (xindex // 4356)
    x3 = xindex % 4356
    tmp3 = tl.load(in_ptr0 + (x1 + (1120*x2)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x3 + (4384*x2)), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x3 + (4384*x2)), tmp5, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/6t/c6trfk6rvxrwtx6mg5gptpi4xpmbb22djh4t44bztqtivua6ds4e.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_default_38 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_32, %permute_21), kwargs = {})
triton_poi_fused_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_17', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 69696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = (xindex // 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (4*(x1 % 1089)) + (4384*(x1 // 1089))), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/hb/chbwrm3u6on27gxjiddjmugokwcqidhau3xjhz3fzdswigovf5lw.py
# Topologically Sorted Source Nodes: [x_out], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_out => clone_5
# Graph fragment:
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_24,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[128, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_18', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 1089
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x5 = xindex
    y4 = yindex
    x3 = (xindex // 33)
    x2 = xindex % 33
    y1 = (yindex // 8)
    y0 = yindex % 8
    tmp0 = tl.load(in_out_ptr0 + (x5 + (1120*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x3 + (33*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr2 + (y0 + (8*x5) + (8712*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 / tmp1
    tmp3 = x3
    tmp4 = tl.full([1, 1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1, 1], 32, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tl.load(in_ptr1 + (x2 + (33*x3) + (1056*y1)), tmp7 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp3 >= tmp6
    tmp10 = tl.full([1, 1], 33, tl.int64)
    tmp11 = tmp3 < tmp10
    tmp12 = 1.0
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp7, tmp8, tmp14)
    tmp16 = 0.0
    tmp17 = tmp15 == tmp16
    tmp20 = tmp18 + tmp19
    tmp21 = tl.where(tmp17, tmp16, tmp20)
    tmp22 = tmp2 + tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5 + (1120*y4)), tmp22, xmask & ymask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ym/cymdgor45kntekmoxixw2j2ugeh7svj3bwqlktv33afcu663bm3h.py
# Topologically Sorted Source Nodes: [x_out], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_out => clone_6
# Graph fragment:
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_25,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_19', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16) % 33
    x2 = (xindex // 528) % 8
    x3 = (xindex // 4224)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + (16*x2) + (384*x1) + (12672*x3)), None)
    tmp1 = tl.load(in_ptr1 + (256 + x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/wr/cwr2xvsgipgx4gzacu4d44sv2osom7yiecuazn5xvvnjabiwmu5c.py
# Topologically Sorted Source Nodes: [x_out_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_out_1 => clone_7
# Graph fragment:
#   %clone_7 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_37,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16) % 8
    x2 = (xindex // 128) % 33
    x3 = (xindex // 4224)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*x2) + (528*x1) + (4224*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/66/c6646w4xoqxzv4g7kmf4cw35f24vyz6jg5ehmgihp2kbmesftylg.py
# Topologically Sorted Source Nodes: [x, layer_norm_2], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_2 => add_13, add_14, mul_16, mul_17, rsqrt_2, sub_3, var_mean_2
#   x => add_12
# Graph fragment:
#   %add_12 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_13, %view_41), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_12, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_12, %getitem_5), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %arg44_1), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %arg45_1), kwargs = {})
triton_per_fused_add_native_layer_norm_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_21', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 528
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 128.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp31, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/jc/cjclawmccc2ol7ocsncsdnjztna6fbxvfsb7flvion2vnrim6jhc.py
# Topologically Sorted Source Nodes: [row_sum], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   row_sum => full_default_2
# Graph fragment:
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 1, 32, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_22', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/p7/cp7o6gtkwbdzjbzahejqa3zdbzysr2xdfev6myxgya7vdnj6qs64.py
# Topologically Sorted Source Nodes: [row_sum, getitem_12, index_add__2, col_sum, getitem_13, index_add__3], Original ATen: [aten.zeros, aten.index, aten.index_add]
# Source node to ATen node mapping:
#   col_sum => full_default_3
#   getitem_12 => index_2
#   getitem_13 => index_3
#   index_add__2 => index_put_2
#   index_add__3 => index_put_3
#   row_sum => full_default_2
# Graph fragment:
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 1, 32, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_2 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%permute_11, [%arg32_1]), kwargs = {})
#   %index_put_2 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_2, [%arg34_1], %index_2, True), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 1, 32, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_3 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%permute_11, [%arg33_1]), kwargs = {})
#   %index_put_3 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_3, [%arg34_1], %index_3, True), kwargs = {})
triton_poi_fused_index_index_add_zeros_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_index_add_zeros_23', 'mutated_arg_names': ['out_ptr0', 'out_ptr1'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 270336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 4096)
    x0 = xindex % 4096
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 48, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 48), "index out of bounds: 0 <= tmp4 < 48")
    tmp7 = tl.full([XBLOCK], 16, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert((0 <= tmp10) & (tmp10 < 16), "index out of bounds: 0 <= tmp10 < 16")
    tmp12 = tl.load(in_ptr2 + (x0 + (4096*tmp10)), None)
    tmp14 = tmp13 + tmp7
    tmp15 = tmp13 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp13)
    tl.device_assert((0 <= tmp16) & (tmp16 < 16), "index out of bounds: 0 <= tmp16 < 16")
    tmp18 = tl.load(in_ptr2 + (x0 + (4096*tmp16)), None)
    tl.atomic_add(out_ptr0 + (x0 + (4096*tmp4)), tmp12, None, sem='relaxed')
    tl.atomic_add(out_ptr1 + (x0 + (4096*tmp4)), tmp18, None, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/5l/c5lufh63jitzzsewhi7hn4zfm3vpzfkt2f5uoafdac5llxhux27t.py
# Topologically Sorted Source Nodes: [h_off_base_1, cat_6], Original ATen: [aten.mean, aten.cat]
# Source node to ATen node mapping:
#   cat_6 => cat_6
#   h_off_base_1 => mean_3
# Graph fragment:
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_16, [2]), kwargs = {})
#   %cat_6 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_17, %mean_4], 2), kwargs = {})
triton_poi_fused_cat_mean_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_mean_24', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x4 = (xindex // 128)
    x2 = (xindex // 2048)
    x5 = xindex
    x3 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (x0 + (256*x4)), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0 + (256*x4)), None)
    tmp7 = tl.load(in_ptr0 + (128 + x0 + (256*x4)), None)
    tmp9 = tl.load(in_ptr2 + (128 + x0 + (256*x4)), None)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 / tmp2
    tmp5 = tmp4 / tmp2
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7 / tmp2
    tmp10 = tmp9 / tmp2
    tmp11 = tmp8 + tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = 2.0
    tmp14 = tmp12 / tmp13
    tl.store(out_ptr0 + (x5), tmp14, None)
    tl.store(out_ptr1 + (x3 + (2176*x2)), tmp14, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/vj/cvj5sx5mmhaavnz5flcfevowglkqzzpszldlt7aebq3dkxsk25ei.py
# Topologically Sorted Source Nodes: [bn_off], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   bn_off => mean_4
# Graph fragment:
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_17, [2], True), kwargs = {})
triton_per_fused_mean_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_25', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (2048*x1)), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = 16.0
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr1 + (x0 + (2176*x1)), tmp5, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/y2/cy2djmbhmmylgr46ifm4a2ivirqmebhcewaxxlzk3jhjn7at5n3s.py
# Topologically Sorted Source Nodes: [layer_norm_3], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_3 => var_mean_3
# Graph fragment:
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_18, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_layer_norm_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_26', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 816
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/5w/c5wwia5pyoy3xwdk2gg6wi6gpgauxvaitcbsteo3xrx3cocqtpxx.py
# Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_3 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%mean, %unsqueeze_6], 2), kwargs = {})
triton_poi_fused_cat_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_27', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 52224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 4) % 17
    x0 = xindex % 4
    x2 = (xindex // 68)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (8*x1) + (264*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr0 + (4 + x0 + (8*x1) + (264*x2)), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr0 + (132 + x0 + (8*x1) + (264*x2)), tmp4 & xmask, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr0 + (136 + x0 + (8*x1) + (264*x2)), tmp4 & xmask, other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = 4.0
    tmp13 = tmp11 / tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 17, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr0 + (128 + x0 + (264*x2)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr0 + (260 + x0 + (264*x2)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = 2.0
    tmp23 = tmp21 / tmp22
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp16, tmp23, tmp24)
    tmp26 = tl.where(tmp4, tmp15, tmp25)
    tl.store(out_ptr0 + (x3), tmp26, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/jl/cjlh6hatubgf2taimjxnozx7yqpa5brs5zfdsd6ahyose7qyisq4.py
# Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_2 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%amax, %unsqueeze_5], -1), kwargs = {})
triton_poi_fused_cat_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_28', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 17
    x1 = (xindex // 17)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((2*x0) + (66*x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (1 + (2*x0) + (66*x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tl.load(in_ptr0 + (33 + (2*x0) + (66*x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = tl.load(in_ptr0 + (34 + (2*x0) + (66*x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 17, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr0 + (32 + (66*x1)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr0 + (65 + (66*x1)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp14, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp13, tmp21)
    tl.store(out_ptr0 + (x2), tmp22, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/wu/cwuor3wgsepuaf5doydwt5ebqyh6fgseagyud7syptgmi6abete6.py
# Topologically Sorted Source Nodes: [layer_norm_3, x_2, kv_1], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd, aten.cat]
# Source node to ATen node mapping:
#   kv_1 => cat_9
#   layer_norm_3 => add_17, add_18, mul_21, mul_22, rsqrt_3, sub_4, var_mean_3
#   x_2 => constant_pad_nd
# Graph fragment:
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_18, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_18, %getitem_7), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_17,), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %arg50_1), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %arg51_1), kwargs = {})
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_18, [0, 0, 0, 15], 0.0), kwargs = {})
#   %cat_9 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_46, %mean_5], 2), kwargs = {})
triton_poi_fused_cat_constant_pad_nd_native_layer_norm_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_constant_pad_nd_native_layer_norm_29', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 128) % 32
    x2 = (xindex // 4096)
    x5 = xindex % 4096
    x0 = xindex % 128
    x6 = xindex
    x3 = xindex % 2048
    x4 = (xindex // 2048)
    tmp0 = x1
    tmp1 = tl.full([1], 17, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x5 + (2176*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (x1 + (17*x2)), tmp2, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 - tmp4
    tmp6 = tl.load(in_ptr2 + (x1 + (17*x2)), tmp2, eviction_policy='evict_last', other=0.0)
    tmp7 = 128.0
    tmp8 = tmp6 / tmp7
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp13 = tl.load(in_ptr3 + (x0), tmp2, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 * tmp13
    tmp15 = tl.load(in_ptr4 + (x0), tmp2, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp2, tmp16, tmp17)
    tl.store(out_ptr0 + (x6), tmp18, None)
    tl.store(out_ptr1 + (x3 + (2176*x4)), tmp18, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/d6/cd6yofeeti66i2trymvulmls5hwfn5r7dvgylpaujdpbuu5vjqjh.py
# Topologically Sorted Source Nodes: [block_node], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   block_node => mean_5
# Graph fragment:
#   %mean_5 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_46, [2], True), kwargs = {})
triton_per_fused_mean_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_30', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (2048*x1)), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = 16.0
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr1 + (x0 + (2176*x1)), tmp5, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/be/cbeqxud6zaapfhhuihws27gm2bky2bklb6mm5plkzhu76lflp5gi.py
# Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   einsum_2 => clone_8
# Graph fragment:
#   %clone_8 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_34,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_31', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16) % 16
    x2 = (xindex // 256) % 8
    x3 = (xindex // 2048)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*x2) + (384*x1) + (6144*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/tt/cttwq6qvgjzddzt65osobhpvubo6zzr543svxcei4plhjkngcwew.py
# Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   einsum_2 => clone_9
# Graph fragment:
#   %clone_9 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_35,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16384, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_32', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 17
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (128 + y0 + (384*x2) + (6528*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (128 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (17*y3)), tmp2, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ye/cyehlkcayafimrwcmpyt23sgxvnqjwvroxkikxnrbr5co5panndb.py
# Topologically Sorted Source Nodes: [setitem_2, setitem_3], Original ATen: [aten.lift_fresh, aten.fill]
# Source node to ATen node mapping:
#   setitem_2 => copy, full_default_10
#   setitem_3 => copy_1, full_default_11
# Graph fragment:
#   %full_default_10 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_9, %full_default_10), kwargs = {})
#   %select_scatter_default_1 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%view_11, %copy, 3, 16), kwargs = {})
#   %full_default_11 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_15, %full_default_11), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int, %copy_1, 3, 3), kwargs = {})
#   %select_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_1, %select_scatter_default_2, 3, 16), kwargs = {})
triton_poi_fused_fill_lift_fresh_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_fill_lift_fresh_33', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 52224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 4) % 17
    x0 = xindex % 4
    x2 = (xindex // 68)
    x3 = xindex
    tmp7 = tl.load(in_ptr0 + (64 + x0 + (68*x2)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (x3), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 16, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 3, tl.int32)
    tmp5 = tmp3 == tmp4
    tmp6 = tmp1 == tmp1
    tmp8 = 0.0
    tmp9 = tl.where(tmp6, tmp8, tmp7)
    tmp10 = 1.0
    tmp11 = tl.where(tmp5, tmp10, tmp9)
    tmp13 = tl.where(tmp2, tmp8, tmp12)
    tmp14 = tl.where(tmp2, tmp11, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/uu/cuuw4b4atgbddcw5dcq26xk5rbqguiqgqlutxeb2pxcj7bpmyvqq.py
# Topologically Sorted Source Nodes: [setitem_2, setitem_3, setitem_4], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
# Source node to ATen node mapping:
#   setitem_2 => copy, full_default_10
#   setitem_3 => copy_1, full_default_11
#   setitem_4 => full_default_12, index_put_6
# Graph fragment:
#   %full_default_10 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_9, %full_default_10), kwargs = {})
#   %select_scatter_default_1 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%view_11, %copy, 3, 16), kwargs = {})
#   %full_default_11 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_15, %full_default_11), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int, %copy_1, 3, 3), kwargs = {})
#   %select_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_1, %select_scatter_default_2, 3, 16), kwargs = {})
#   %full_default_12 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_6 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%select_scatter_default_3, [None, None, %iota_1, %iota_1], %full_default_12), kwargs = {})
triton_poi_fused_fill_index_put_lift_fresh_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_fill_index_put_lift_fresh_34', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = (xindex // 4) % 16
    x2 = (xindex // 64)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (72*x1) + (1088*x2)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/a6/ca6yxnf23rebbbdmcb5i32ni3xzhyaxlrw5nho7d6bijws7j7jdj.py
# Topologically Sorted Source Nodes: [setitem_5], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_5 => full_default_13, index_put_7
# Graph fragment:
#   %full_default_13 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_7 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_20, [None, None, %iota_1, %iota_1], %full_default_13), kwargs = {})
triton_poi_fused_index_put_lift_fresh_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_35', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + (4*x0)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/gd/cgdia4k6mso33kotelxlpcdzxxdew6npzox5fqsxbcyraoqqcfkl.py
# Topologically Sorted Source Nodes: [setitem_5], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_5 => full_default_13, index_put_7
# Graph fragment:
#   %full_default_13 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_7 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_20, [None, None, %iota_1, %iota_1], %full_default_13), kwargs = {})
triton_poi_fused_index_put_lift_fresh_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_36', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp0 = 1.0
    tl.store(out_ptr0 + ((18*x0) + (272*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/x5/cx52waiqboffubcjylil2yrhc3rtiqeow3dphl7h475h2ssbxict.py
# Topologically Sorted Source Nodes: [eq_2, scores_5, scores_3, scores_4, attn_probs_1], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_probs_1 => amax_3, exp_1, sub_5
#   eq_2 => eq_2
#   scores_3 => mul_23
#   scores_4 => add_19
#   scores_5 => clone_11, full_default_14, where_2
# Graph fragment:
#   %eq_2 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%unsqueeze_18, 0), kwargs = {})
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_57, 0.25), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %slice_77), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_2, %full_default_14, %add_19), kwargs = {})
#   %clone_11 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%where_2,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_3 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_11, [3], True), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_11, %amax_3), kwargs = {})
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_5,), kwargs = {})
triton_per_fused__softmax_add_eq_masked_fill_mul_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_eq_masked_fill_mul_37', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 17
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 16
    x2 = (xindex // 256)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (17*x0) + (272*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (r3 + (17*x4)), rmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r3 + (17*x0) + (272*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (3 + (4*r3) + (68*x0) + (1088*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tmp4 = 0.25
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full([1, 1], 3, tl.int32)
    tmp7 = tmp6 == tmp6
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp5 + tmp10
    tmp12 = float("-inf")
    tmp13 = tl.where(tmp2, tmp12, tmp11)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask, tmp14, float("-inf"))
    tmp17 = triton_helpers.max2(tmp16, 1)[:, None]
    tmp18 = tmp13 - tmp17
    tmp19 = tl_math.exp(tmp18)
    tl.store(out_ptr1 + (r3 + (17*x4)), tmp19, rmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/d5/cd5qdvxc4b3tkgow2xy73bl2joiw6sug6c7otsaum25hud267hw2.py
# Topologically Sorted Source Nodes: [attn_probs_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_probs_1 => sum_2
# Graph fragment:
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [3], True), kwargs = {})
triton_per_fused__softmax_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_38', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 17
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 8
    x1 = (xindex // 8) % 16
    x2 = (xindex // 128)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (17*x1) + (272*x0) + (2176*x2)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/2t/c2taffrheppxszung4wrwyszohhudiquwtlct2nvtdydhe2te3x2.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %select_scatter_default_4 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%index_put_6, %index_put_7, 4, 3), kwargs = {})
triton_poi_fused_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_39', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 52224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = (xindex // 4)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x2), tmp5, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/tv/ctv2ayykhmvzmxbabsqp53gy5bkbk3ii637qfzbzsiujyspjhdsq.py
# Topologically Sorted Source Nodes: [x_out_4], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_out_4 => clone_12
# Graph fragment:
#   %clone_12 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_40,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1024, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_40', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 272
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x6 = xindex
    y5 = yindex
    x4 = (xindex // 17)
    y0 = yindex % 8
    y7 = (yindex // 8)
    y2 = (yindex // 16)
    tmp0 = tl.load(in_out_ptr0 + (x6 + (272*y5)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (8*x4) + (128*y7)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x6 + (272*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (y0 + (8*x6) + (2176*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 / tmp1
    tmp4 = 0.0
    tmp5 = tmp3 == tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = tl.where(tmp5, tmp4, tmp8)
    tmp10 = tmp2 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x6 + (272*y5)), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/lc/clcxxfbjttf6ifms7qbkwzcxgkprhwwr6suofmiqizwoplev3n2x.py
# Topologically Sorted Source Nodes: [x_out_4], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_out_4 => clone_13
# Graph fragment:
#   %clone_13 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_41,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_41', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 208896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16) % 17
    x2 = (xindex // 272) % 8
    x3 = (xindex // 2176)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + (16*x2) + (384*x1) + (6528*x3)), None)
    tmp1 = tl.load(in_ptr1 + (256 + x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/2q/c2qkts3w7xswpeeh4ddurhnpmt2hhsaewoohwjo2wfh4zadumrkw.py
# Topologically Sorted Source Nodes: [x_out_5], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_out_5 => clone_14
# Graph fragment:
#   %clone_14 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_64,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_42', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16) % 8
    x2 = (xindex // 128) % 16
    x3 = (xindex // 2048)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*x2) + (256*x1) + (2048*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/va/cva7hw62bzazelh6sou7n5nra4mbkvr376z37gtehkxumxb5pxmb.py
# Topologically Sorted Source Nodes: [x_3, layer_norm_4], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_4 => add_22, add_23, mul_24, mul_25, rsqrt_4, sub_6, var_mean_4
#   x_3 => add_21
# Graph fragment:
#   %add_21 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_18, %slice_79), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_21, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_21, %getitem_9), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_22,), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_4), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, %arg58_1), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_25, %arg59_1), kwargs = {})
triton_per_fused_add_native_layer_norm_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_43', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 816
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 17
    x1 = (xindex // 17)
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (128*x0) + (4096*x1)), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 128.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp31, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/y2/cy27xkchvhdbv7wjdr6nverfpol2cq22357waa2r5an4oi2kflea.py
# Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_11 => add_15, erf_3, mul_18, mul_19, mul_20
# Graph fragment:
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_43, 0.5), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_43, 0.7071067811865476), kwargs = {})
#   %erf_3 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_19,), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_3, 1), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %add_15), kwargs = {})
triton_poi_fused_gelu_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_44', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 270336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/b5/cb5vleyz7plmktsefikr3jzgm62gt7tnyke44zfmrqab2plrizea.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x => add_12
#   x_1 => add_16
# Graph fragment:
#   %add_12 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_13, %view_41), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %view_45), kwargs = {})
triton_poi_fused_add_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_45', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ch/cchrtb2xyt3o6dklxbulopzgci2w7lgs4tzstmiwoaqfrgclqg7h.py
# Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_14 => add_24, erf_4, mul_26, mul_27, mul_28
# Graph fragment:
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_70, 0.5), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_70, 0.7071067811865476), kwargs = {})
#   %erf_4 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_27,), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_4, 1), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, %add_24), kwargs = {})
triton_poi_fused_gelu_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_46', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 417792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/vv/cvvsgtmkbvej4naf3u4flv5zumszrqbqztccxq23ohbxi22aw2ua.py
# Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_3 => add_21
#   x_4 => add_25
# Graph fragment:
#   %add_21 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_18, %slice_79), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %view_72), kwargs = {})
triton_poi_fused_add_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_47', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 104448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x2 = (xindex // 2176)
    x4 = xindex % 2176
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 + (4096*x2)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x3), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ud/cudnqlp3jsovxdyxhfmi6xhrerkagdnbdxgao3y5k6fvh3yfs674.py
# Topologically Sorted Source Nodes: [V], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   V => cat_10
# Graph fragment:
#   %cat_10 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%select_22, %select_23], 1), kwargs = {})
triton_poi_fused_cat_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_48', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (4096 + x0 + (4224*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ld/cldhqfrqghn75zigatqj4kgicjrkaip5dxdf32rmjxefa6ttv3xw.py
# Topologically Sorted Source Nodes: [V], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   V => cat_10
# Graph fragment:
#   %cat_10 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%select_22, %select_23], 1), kwargs = {})
triton_poi_fused_cat_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_49', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2048 + x0 + (2176*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ri/criv4rj5etrdzzswpieozzu6sfrp6hq46du7px5sianvq3yzijxx.py
# Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_17 => add_26, erf_5, mul_29, mul_30, mul_31
# Graph fragment:
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_79, 0.5), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_79, 0.7071067811865476), kwargs = {})
#   %erf_5 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_30,), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_5, 1), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %add_26), kwargs = {})
triton_poi_fused_gelu_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_50', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/45/c45ijvqwm5gpzxcfewd6n43ktqbw6gefrl725oigiga4clyw3quv.py
# Topologically Sorted Source Nodes: [V_1], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   V_1 => add_27
# Graph fragment:
#   %add_27 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_10, %view_81), kwargs = {})
triton_poi_fused_add_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_51', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/7l/c7l4ldgrozyhyulicww4dqbhtaxgctregl7ule3sa56wvryhml2a.py
# Topologically Sorted Source Nodes: [V_2], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   V_2 => add_29
# Graph fragment:
#   %add_29 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %view_88), kwargs = {})
triton_poi_fused_add_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_52', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/72/c72h2oogtfx47pbdrwt46gx4umqsp3vn22q45o2uh6dc73dw7iv7.py
# Topologically Sorted Source Nodes: [V_8, V_9], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   V_8 => add_41
#   V_9 => add_42, add_43, mul_53, mul_54, rsqrt_5, sub_7, var_mean_5
# Graph fragment:
#   %add_41 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_39, %view_130), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_41, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_41, %getitem_11), kwargs = {})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_42,), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_5), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_53, %arg97_1), kwargs = {})
#   %add_43 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_54, %arg98_1), kwargs = {})
triton_per_fused_add_native_layer_norm_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[64, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_53', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 128.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp31, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ax/caxrlonb24xar32pd54lkexpzacg3tkjx25y7grv7grvn4a3cbkg.py
# Topologically Sorted Source Nodes: [layer_norm_6], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_6 => add_44, add_45, mul_55, mul_56, rsqrt_6, sub_8, var_mean_6
# Graph fragment:
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_133, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_133, %getitem_13), kwargs = {})
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-05), kwargs = {})
#   %rsqrt_6 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_44,), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_6), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %arg99_1), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %arg100_1), kwargs = {})
triton_per_fused_native_layer_norm_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_54', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 528
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex
    r1 = rindex
    tmp3 = tl.load(in_ptr0 + (r1 + (128*(x0 // 33))), xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask, other=0.0)
    tmp29 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp0 = x0 % 33
    tmp1 = tl.full([1, 1], 32, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 / tmp14
    tmp16 = tmp6 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = tmp5 - tmp15
    tmp23 = 128.0
    tmp24 = tmp21 / tmp23
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp32, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/gt/cgtf6q5vcs76i2wzapepfos422c4tne5dhxhzk6yrwihmpk2qtmk.py
# Topologically Sorted Source Nodes: [x_5, layer_norm_7], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_7 => add_49, add_50, mul_58, mul_59, rsqrt_7, sub_10, var_mean_7
#   x_5 => add_48
# Graph fragment:
#   %add_48 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_133, %view_156), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_48, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_48, %getitem_15), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-05), kwargs = {})
#   %rsqrt_7 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_49,), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %rsqrt_7), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_58, %arg107_1), kwargs = {})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_59, %arg108_1), kwargs = {})
triton_per_fused_add_native_layer_norm_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_55', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 528
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex
    r1 = rindex
    tmp3 = tl.load(in_ptr0 + (r1 + (128*(x0 // 33))), xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp0 = x0 % 33
    tmp1 = tl.full([1, 1], 32, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 / tmp18
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp9 - tmp19
    tmp27 = 128.0
    tmp28 = tmp25 / tmp27
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp36, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/e3/ce357wprago5gi2c5repqf6hekprflg4uuglv5znywmgg7k6p3da.py
# Topologically Sorted Source Nodes: [layer_norm_8], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_8 => var_mean_8
# Graph fragment:
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_161, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_layer_norm_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_56', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 816
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex % 17
    r2 = rindex
    x1 = (xindex // 17)
    x3 = xindex
    tmp3 = tl.load(in_ptr0 + (2048 + r2 + (128*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr1 + (r2 + (128*x3)), xmask, other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1, 1], 16, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 / tmp14
    tmp16 = tmp6 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp21, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/dt/cdtxrgw5elulz4lwfat6wih2oovcp5eyqcx6uok3ogcy74l5aen7.py
# Topologically Sorted Source Nodes: [layer_norm_8, x_7, kv_3], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd, aten.cat]
# Source node to ATen node mapping:
#   kv_3 => cat_13
#   layer_norm_8 => add_53, add_54, mul_63, mul_64, rsqrt_8, sub_11, var_mean_8
#   x_7 => constant_pad_nd_1
# Graph fragment:
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_161, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_161, %getitem_17), kwargs = {})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_8 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_53,), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %rsqrt_8), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_63, %arg113_1), kwargs = {})
#   %add_54 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_64, %arg114_1), kwargs = {})
#   %constant_pad_nd_1 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_54, [0, 0, 0, 15], 0.0), kwargs = {})
#   %cat_13 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_162, %mean_6], 2), kwargs = {})
triton_poi_fused_cat_constant_pad_nd_native_layer_norm_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_constant_pad_nd_native_layer_norm_57', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 128) % 32
    x0 = xindex % 128
    x2 = (xindex // 4096)
    x5 = xindex % 4096
    x6 = xindex
    x3 = xindex % 2048
    x4 = (xindex // 2048)
    tmp0 = x1
    tmp1 = tl.full([1], 17, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 16, tl.int32)
    tmp4 = tmp0 == tmp3
    tmp5 = tl.load(in_ptr0 + (2048 + x0 + (128*x2)), tmp2, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x5 + (2176*x2)), tmp2, other=0.0)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tl.load(in_ptr2 + (x1 + (17*x2)), tmp2, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 - tmp8
    tmp10 = tl.load(in_ptr3 + (x1 + (17*x2)), tmp2, eviction_policy='evict_last', other=0.0)
    tmp11 = 128.0
    tmp12 = tmp10 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp9 * tmp15
    tmp17 = tl.load(in_ptr4 + (x0), tmp2, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr5 + (x0), tmp2, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tl.store(out_ptr0 + (x6), tmp22, None)
    tl.store(out_ptr1 + (x3 + (2176*x4)), tmp22, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ik/cik3dnmvirls2xckiv74trapww4sn7s7w7beagxsyblg2tfjxgy7.py
# Topologically Sorted Source Nodes: [x_8, layer_norm_9], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_9 => add_58, add_59, mul_66, mul_67, rsqrt_9, sub_13, var_mean_9
#   x_8 => add_57
# Graph fragment:
#   %add_57 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_161, %slice_178), kwargs = {})
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_57, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_57, %getitem_19), kwargs = {})
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_18, 1e-05), kwargs = {})
#   %rsqrt_9 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_58,), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %rsqrt_9), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_66, %arg121_1), kwargs = {})
#   %add_59 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_67, %arg122_1), kwargs = {})
triton_per_fused_add_native_layer_norm_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_58', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 816
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex % 17
    r2 = rindex
    x1 = (xindex // 17)
    x3 = xindex
    tmp3 = tl.load(in_ptr0 + (2048 + r2 + (128*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr1 + (r2 + (128*x3)), xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (r2 + (128*x0) + (4096*x1)), xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 16, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 / tmp18
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp9 - tmp19
    tmp27 = 128.0
    tmp28 = tmp25 / tmp27
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp36, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/og/cog2ayr6ltaynmxln22rxtgnlearsvsulwz3qm2h3gsclelvp3de.py
# Topologically Sorted Source Nodes: [h_diag], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   h_diag => clone_32
# Graph fragment:
#   %clone_32 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_182,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_59', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 128) % 32
    x0 = xindex % 128
    x2 = (xindex // 4096)
    x3 = xindex % 4096
    x4 = xindex
    tmp3 = tl.load(in_ptr0 + (x0 + (128*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x3 + (4224*x2)), None)
    tmp6 = tl.load(in_ptr2 + (x3 + (4224*x2)), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x3 + (4224*x2)), None)
    tmp11 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 32, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/u4/cu456mp2egnvqahufokfiudilomrra4i4wzzggiovjrxr6a4y5xg.py
# Topologically Sorted Source Nodes: [u_leaf], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   u_leaf => addmm_51
# Graph fragment:
#   %addmm_51 : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg128_1, %view_192, %permute_94), kwargs = {})
#   %mm_default : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_212, %permute_99), kwargs = {})
triton_poi_fused_addmm_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_60', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x1) + (4096*((x1 % 32) // 32))), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
    tl.store(out_ptr1 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/a3/ca35fqjmzx42pilk5j3rsws6dgnljhlpzptakzgbuifqlu7olwhc.py
# Topologically Sorted Source Nodes: [x_8, x_9], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_8 => add_57
#   x_9 => add_61
# Graph fragment:
#   %add_57 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_161, %slice_178), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_57, %view_188), kwargs = {})
triton_poi_fused_add_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_61', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 104448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 128) % 17
    x0 = xindex % 128
    x2 = (xindex // 2176)
    x3 = xindex
    x4 = xindex % 2176
    tmp3 = tl.load(in_ptr0 + (2048 + x0 + (128*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x4 + (4096*x2)), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_out_ptr0 + (x3), None)
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 16, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/w7/cw7e477gef3cv746ow73fauuci7ywaliflpjthvlg6nohdy52fvw.py
# Topologically Sorted Source Nodes: [u_leaf_1], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   u_leaf_1 => add_62
# Graph fragment:
#   %add_62 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_203, %arg130_1), kwargs = {})
triton_poi_fused_add_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_62', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/b7/cb7wqveuhwsd5orzk3eipsshqpai75rgx5yxixfsbtefhgh5uq54.py
# Topologically Sorted Source Nodes: [sigmoid], Original ATen: [aten.sigmoid]
# Source node to ATen node mapping:
#   sigmoid => sigmoid
# Graph fragment:
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_213,), kwargs = {})
triton_poi_fused_sigmoid_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_63', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ul/culngr7myzck6q3irq6leqisyszm3eo7jvsgo3czhygqn5yexeyq.py
# Topologically Sorted Source Nodes: [packed_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   packed_1 => cat_15
# Graph fragment:
#   %cat_15 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_14, %view_214], 1), kwargs = {})
triton_poi_fused_cat_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_64', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 45568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 45056, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 16384, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.load(in_ptr0 + (x0), tmp7 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp0 >= tmp5
    tmp10 = tl.full([1], 28672, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tmp12 & tmp4
    tmp14 = tl.load(in_ptr1 + ((-16384) + x0), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp0 >= tmp10
    tmp16 = tl.full([1], 36864, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr2 + ((-28672) + x0), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp0 >= tmp16
    tmp22 = tmp21 & tmp4
    tmp23 = tl.load(in_ptr3 + ((-36864) + x0), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp18, tmp20, tmp23)
    tmp25 = tl.where(tmp12, tmp14, tmp24)
    tmp26 = tl.where(tmp6, tmp8, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp4, tmp26, tmp27)
    tmp29 = tmp0 >= tmp3
    tmp30 = tl.full([1], 45568, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = tl.load(in_ptr4 + ((-45056) + x0), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.where(tmp4, tmp28, tmp32)
    tl.store(out_ptr0 + (x0), tmp33, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 512, 9), (67716, 9, 1))
    assert_size_stride(arg1_1, (1, 12), (12, 1))
    assert_size_stride(arg2_1, (128, 18), (18, 1))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, 128), (128, 1))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (2, 3313), (3313, 1))
    assert_size_stride(arg7_1, (3313, ), (1, ))
    assert_size_stride(arg8_1, (128, 128), (128, 1))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (128, 128), (128, 1))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (128, 256), (256, 1))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, 128), (128, 1))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (128, 128), (128, 1))
    assert_size_stride(arg17_1, (128, ), (1, ))
    assert_size_stride(arg18_1, (128, 128), (128, 1))
    assert_size_stride(arg19_1, (128, ), (1, ))
    assert_size_stride(arg20_1, (128, 256), (256, 1))
    assert_size_stride(arg21_1, (128, ), (1, ))
    assert_size_stride(arg22_1, (128, 128), (128, 1))
    assert_size_stride(arg23_1, (128, ), (1, ))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (128, 128), (128, 1))
    assert_size_stride(arg27_1, (128, ), (1, ))
    assert_size_stride(arg28_1, (16, 32, 33, 4), (4224, 132, 4, 1))
    assert_size_stride(arg29_1, (16, 32, 33), (1056, 33, 1))
    assert_size_stride(arg30_1, (48, 32, 33, 4), (4224, 132, 4, 1))
    assert_size_stride(arg31_1, (48, 32, 33), (1056, 33, 1))
    assert_size_stride(arg32_1, (66, ), (1, ))
    assert_size_stride(arg33_1, (66, ), (1, ))
    assert_size_stride(arg34_1, (66, ), (1, ))
    assert_size_stride(arg35_1, (48, ), (1, ))
    assert_size_stride(arg36_1, (128, ), (1, ))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (384, 128), (128, 1))
    assert_size_stride(arg39_1, (384, ), (1, ))
    assert_size_stride(arg40_1, (8, 4), (4, 1))
    assert_size_stride(arg41_1, (8, ), (1, ))
    assert_size_stride(arg42_1, (128, 128), (128, 1))
    assert_size_stride(arg43_1, (128, ), (1, ))
    assert_size_stride(arg44_1, (128, ), (1, ))
    assert_size_stride(arg45_1, (128, ), (1, ))
    assert_size_stride(arg46_1, (512, 128), (128, 1))
    assert_size_stride(arg47_1, (512, ), (1, ))
    assert_size_stride(arg48_1, (128, 512), (512, 1))
    assert_size_stride(arg49_1, (128, ), (1, ))
    assert_size_stride(arg50_1, (128, ), (1, ))
    assert_size_stride(arg51_1, (128, ), (1, ))
    assert_size_stride(arg52_1, (384, 128), (128, 1))
    assert_size_stride(arg53_1, (384, ), (1, ))
    assert_size_stride(arg54_1, (8, 4), (4, 1))
    assert_size_stride(arg55_1, (8, ), (1, ))
    assert_size_stride(arg56_1, (128, 128), (128, 1))
    assert_size_stride(arg57_1, (128, ), (1, ))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (128, ), (1, ))
    assert_size_stride(arg60_1, (512, 128), (128, 1))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (128, 512), (512, 1))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (64, 64), (64, 1))
    assert_size_stride(arg65_1, (128, 128), (128, 1))
    assert_size_stride(arg66_1, (128, ), (1, ))
    assert_size_stride(arg67_1, (128, 128), (128, 1))
    assert_size_stride(arg68_1, (128, ), (1, ))
    assert_size_stride(arg69_1, (128, 128), (128, 1))
    assert_size_stride(arg70_1, (128, ), (1, ))
    assert_size_stride(arg71_1, (128, 128), (128, 1))
    assert_size_stride(arg72_1, (128, ), (1, ))
    assert_size_stride(arg73_1, (128, 128), (128, 1))
    assert_size_stride(arg74_1, (128, ), (1, ))
    assert_size_stride(arg75_1, (128, 128), (128, 1))
    assert_size_stride(arg76_1, (128, ), (1, ))
    assert_size_stride(arg77_1, (128, 128), (128, 1))
    assert_size_stride(arg78_1, (128, ), (1, ))
    assert_size_stride(arg79_1, (128, 128), (128, 1))
    assert_size_stride(arg80_1, (128, ), (1, ))
    assert_size_stride(arg81_1, (128, 128), (128, 1))
    assert_size_stride(arg82_1, (128, ), (1, ))
    assert_size_stride(arg83_1, (128, 128), (128, 1))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (128, 128), (128, 1))
    assert_size_stride(arg86_1, (128, ), (1, ))
    assert_size_stride(arg87_1, (128, 128), (128, 1))
    assert_size_stride(arg88_1, (128, ), (1, ))
    assert_size_stride(arg89_1, (128, 128), (128, 1))
    assert_size_stride(arg90_1, (128, ), (1, ))
    assert_size_stride(arg91_1, (128, 128), (128, 1))
    assert_size_stride(arg92_1, (128, ), (1, ))
    assert_size_stride(arg93_1, (128, 128), (128, 1))
    assert_size_stride(arg94_1, (128, ), (1, ))
    assert_size_stride(arg95_1, (128, 128), (128, 1))
    assert_size_stride(arg96_1, (128, ), (1, ))
    assert_size_stride(arg97_1, (128, ), (1, ))
    assert_size_stride(arg98_1, (128, ), (1, ))
    assert_size_stride(arg99_1, (128, ), (1, ))
    assert_size_stride(arg100_1, (128, ), (1, ))
    assert_size_stride(arg101_1, (384, 128), (128, 1))
    assert_size_stride(arg102_1, (384, ), (1, ))
    assert_size_stride(arg103_1, (8, 4), (4, 1))
    assert_size_stride(arg104_1, (8, ), (1, ))
    assert_size_stride(arg105_1, (128, 128), (128, 1))
    assert_size_stride(arg106_1, (128, ), (1, ))
    assert_size_stride(arg107_1, (128, ), (1, ))
    assert_size_stride(arg108_1, (128, ), (1, ))
    assert_size_stride(arg109_1, (512, 128), (128, 1))
    assert_size_stride(arg110_1, (512, ), (1, ))
    assert_size_stride(arg111_1, (128, 512), (512, 1))
    assert_size_stride(arg112_1, (128, ), (1, ))
    assert_size_stride(arg113_1, (128, ), (1, ))
    assert_size_stride(arg114_1, (128, ), (1, ))
    assert_size_stride(arg115_1, (384, 128), (128, 1))
    assert_size_stride(arg116_1, (384, ), (1, ))
    assert_size_stride(arg117_1, (8, 4), (4, 1))
    assert_size_stride(arg118_1, (8, ), (1, ))
    assert_size_stride(arg119_1, (128, 128), (128, 1))
    assert_size_stride(arg120_1, (128, ), (1, ))
    assert_size_stride(arg121_1, (128, ), (1, ))
    assert_size_stride(arg122_1, (128, ), (1, ))
    assert_size_stride(arg123_1, (512, 128), (128, 1))
    assert_size_stride(arg124_1, (512, ), (1, ))
    assert_size_stride(arg125_1, (128, 512), (512, 1))
    assert_size_stride(arg126_1, (128, ), (1, ))
    assert_size_stride(arg127_1, (32, 128), (128, 1))
    assert_size_stride(arg128_1, (32, ), (1, ))
    assert_size_stride(arg129_1, (16, 128), (128, 1))
    assert_size_stride(arg130_1, (16, ), (1, ))
    assert_size_stride(arg131_1, (16, 128), (128, 1))
    assert_size_stride(arg132_1, (16, ), (1, ))
    assert_size_stride(arg133_1, (1, 128), (128, 1))
    assert_size_stride(arg134_1, (1, ), (1, ))
    assert_size_stride(arg135_1, (16, 128), (128, 1))
    assert_size_stride(arg136_1, (16, ), (1, ))
    assert_size_stride(arg137_1, (16, 128), (128, 1))
    assert_size_stride(arg138_1, (16, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 512, 18), (9216, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [node_feats_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(arg0_1, arg1_1, buf0, 9216, grid=grid(9216), stream=stream0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf0, (512, 18), (18, 1), 0), reinterpret_tensor(arg2_1, (18, 128), (1, 18), 0), out=buf1)
        del arg2_1
        del buf0
        buf2 = reinterpret_tensor(buf1, (1, 512, 128), (65536, 128, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_1.run(buf2, arg3_1, 65536, grid=grid(65536), stream=stream0)
        del arg3_1
        buf3 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg5_1, reinterpret_tensor(buf2, (512, 128), (128, 1), 0), reinterpret_tensor(arg4_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf3)
        del arg4_1
        del arg5_1
        buf8 = empty_strided_cuda((512, 256), (256, 1), torch.float32)
        buf4 = reinterpret_tensor(buf8, (512, 128), (256, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [self_features], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, buf3, reinterpret_tensor(arg10_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf4)
        del arg10_1
        del arg11_1
        buf5 = reinterpret_tensor(buf2, (512, 128), (128, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [neighbor_features], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg9_1, buf3, reinterpret_tensor(arg8_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf5)
        del arg8_1
        del arg9_1
        buf6 = reinterpret_tensor(buf8, (512, 128), (256, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [aggr], Original ATen: [aten.zeros_like]
        triton_poi_fused_zeros_like_2.run(buf6, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [aggr, getitem_4, messages, index_add_], Original ATen: [aten.zeros_like, aten.index, aten.mul, aten.index_add]
        triton_poi_fused_index_index_add_mul_zeros_like_3.run(arg6_1, buf5, arg7_1, buf6, 424064, grid=grid(424064), stream=stream0)
        del buf4
        del buf6
        buf9 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf8, reinterpret_tensor(arg12_1, (256, 128), (1, 256), 0), out=buf9)
        del arg12_1
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_1.run(buf10, arg13_1, 65536, grid=grid(65536), stream=stream0)
        del arg13_1
        buf11 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.gelu]
        extern_kernels.mm(buf10, reinterpret_tensor(arg14_1, (128, 128), (1, 128), 0), out=buf11)
        del arg14_1
        buf12 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.add]
        triton_poi_fused_add_4.run(buf12, buf3, arg15_1, 65536, grid=grid(65536), stream=stream0)
        del arg15_1
        buf17 = buf8; del buf8  # reuse
        buf13 = reinterpret_tensor(buf17, (512, 128), (256, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [self_features_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg19_1, buf12, reinterpret_tensor(arg18_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf13)
        del arg18_1
        del arg19_1
        buf14 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [neighbor_features_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg17_1, buf12, reinterpret_tensor(arg16_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf14)
        del arg16_1
        del arg17_1
        buf15 = reinterpret_tensor(buf17, (512, 128), (256, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [aggr_1], Original ATen: [aten.zeros_like]
        triton_poi_fused_zeros_like_2.run(buf15, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [aggr_1, getitem_7, messages_1, index_add__1], Original ATen: [aten.zeros_like, aten.index, aten.mul, aten.index_add]
        triton_poi_fused_index_index_add_mul_zeros_like_3.run(arg6_1, buf14, arg7_1, buf15, 424064, grid=grid(424064), stream=stream0)
        del arg6_1
        del arg7_1
        del buf13
        del buf15
        buf18 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf17, reinterpret_tensor(arg20_1, (256, 128), (1, 256), 0), out=buf18)
        del arg20_1
        del buf17
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_1.run(buf19, arg21_1, 65536, grid=grid(65536), stream=stream0)
        del arg21_1
        buf20 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.gelu]
        extern_kernels.mm(buf19, reinterpret_tensor(arg22_1, (128, 128), (1, 128), 0), out=buf20)
        del arg22_1
        buf24 = reinterpret_tensor(buf19, (1, 512, 128), (65536, 128, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [h_2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_5.run(buf12, buf20, arg23_1, arg24_1, arg25_1, buf24, 512, 128, grid=grid(512), stream=stream0)
        del arg23_1
        del arg24_1
        del arg25_1
        buf25 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg27_1, reinterpret_tensor(buf24, (512, 128), (128, 1), 0), reinterpret_tensor(arg26_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf25)
        del arg26_1
        del arg27_1
        buf29 = empty_strided_cuda((1, 16, 33, 128), (67584, 4224, 128, 1), torch.float32)
        buf28 = reinterpret_tensor(buf29, (1, 16, 1, 128), (67584, 4224, 128, 1), 4096)  # alias
        # Topologically Sorted Source Nodes: [bn_diag], Original ATen: [aten.mean]
        triton_per_fused_mean_6.run(buf25, buf28, 2048, 32, grid=grid(2048), stream=stream0)
        buf27 = reinterpret_tensor(buf29, (1, 16, 32, 128), (67584, 4224, 128, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [cat_5], Original ATen: [aten.cat]
        triton_poi_fused_cat_7.run(buf25, buf27, 65536, grid=grid(65536), stream=stream0)
        buf33 = empty_strided_cuda((1, 528, 128), (67584, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_8.run(buf29, arg36_1, arg37_1, buf33, 528, 128, grid=grid(528), stream=stream0)
        del arg36_1
        del arg37_1
        del buf27
        del buf28
        buf34 = empty_strided_cuda((528, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf33, (528, 128), (128, 1), 0), reinterpret_tensor(arg38_1, (128, 384), (1, 128), 0), out=buf34)
        buf35 = empty_strided_cuda((528, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf33, (528, 128), (128, 1), 0), reinterpret_tensor(arg38_1, (128, 384), (1, 128), 0), out=buf35)
        del arg38_1
        buf36 = reinterpret_tensor(buf33, (16, 8, 33, 16, 1, 1), (4224, 528, 16, 1, 1, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf34, arg39_1, buf36, 67584, grid=grid(67584), stream=stream0)
        buf37 = empty_strided_cuda((16, 8, 16, 1, 33, 1), (4224, 528, 33, 33, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf35, arg39_1, buf37, 2048, 33, grid=grid(2048, 33), stream=stream0)
        buf38 = empty_strided_cuda((128, 33, 33), (1089, 33, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf36, (128, 33, 16), (528, 16, 1), 0), reinterpret_tensor(buf37, (128, 16, 33), (528, 33, 1), 0), out=buf38)
        buf39 = empty_strided_cuda((1, 16, 33, 33, 4), (70144, 4384, 132, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bias_physics_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(arg28_1, buf39, 69696, grid=grid(69696), stream=stream0)
        # Topologically Sorted Source Nodes: [bias_physics_1, setitem], Original ATen: [aten.cat, aten.lift_fresh, aten.index_put]
        triton_poi_fused_cat_index_put_lift_fresh_12.run(buf39, 2048, grid=grid(2048), stream=stream0)
        buf41 = empty_strided_cuda((1, 16, 33, 33), (17920, 1120, 33, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_13.run(buf39, buf41, 17424, grid=grid(17424), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_14.run(buf41, 512, grid=grid(512), stream=stream0)
        buf44 = empty_strided_cuda((1, 16, 33, 33, 8), (143360, 8960, 33, 1, 1120), torch.float32)
        buf45 = empty_strided_cuda((1, 16, 33, 1, 8), (4224, 264, 1, 4224, 33), torch.float32)
        # Topologically Sorted Source Nodes: [eq, scores_2, scores, scores_1, attn_probs], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_eq_masked_fill_mul_15.run(arg29_1, buf38, buf41, buf39, buf44, buf45, 4224, 33, grid=grid(4224), stream=stream0)
        buf46 = empty_strided_cuda((1, 16, 33, 33, 4), (70144, 4384, 132, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(buf41, buf39, buf46, 69696, grid=grid(69696), stream=stream0)
        buf47 = empty_strided_cuda((17424, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_17.run(buf46, buf47, 69696, grid=grid(69696), stream=stream0)
        buf48 = reinterpret_tensor(buf38, (17424, 8), (8, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf47, reinterpret_tensor(arg40_1, (4, 8), (1, 4), 0), out=buf48)
        del arg40_1
        buf49 = reinterpret_tensor(buf44, (16, 8, 33, 33, 1, 1), (8960, 1120, 33, 1, 1, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [x_out], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf49, buf45, arg29_1, buf48, arg41_1, 128, 1089, grid=grid(128, 1089), stream=stream0)
        del arg41_1
        buf50 = reinterpret_tensor(buf37, (16, 8, 33, 1, 16, 1), (4224, 528, 16, 16, 1, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [x_out], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf35, arg39_1, buf50, 67584, grid=grid(67584), stream=stream0)
        del arg39_1
        buf51 = reinterpret_tensor(buf36, (128, 33, 16), (528, 16, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [x_out], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf49, (128, 33, 33), (1120, 33, 1), 0), reinterpret_tensor(buf50, (128, 33, 16), (528, 16, 1), 0), out=buf51)
        buf52 = reinterpret_tensor(buf50, (1, 16, 33, 8, 16), (67584, 4224, 128, 16, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_out_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf51, buf52, 67584, grid=grid(67584), stream=stream0)
        buf53 = reinterpret_tensor(buf51, (528, 128), (128, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf52, (528, 128), (128, 1), 0), reinterpret_tensor(arg42_1, (128, 128), (1, 128), 0), out=buf53)
        del arg42_1
        buf98 = reinterpret_tensor(buf52, (1, 528, 128), (67584, 128, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [x, layer_norm_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_21.run(buf29, buf53, arg43_1, arg44_1, arg45_1, buf98, 528, 128, grid=grid(528), stream=stream0)
        del arg44_1
        del arg45_1
        buf57 = empty_strided_cuda((48, 1, 32, 128), (4096, 4096, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [row_sum], Original ATen: [aten.zeros]
        triton_poi_fused_zeros_22.run(buf57, 196608, grid=grid(196608), stream=stream0)
        buf59 = empty_strided_cuda((48, 1, 32, 128), (4096, 4096, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [col_sum], Original ATen: [aten.zeros]
        triton_poi_fused_zeros_22.run(buf59, 196608, grid=grid(196608), stream=stream0)
        # Topologically Sorted Source Nodes: [row_sum, getitem_12, index_add__2, col_sum, getitem_13, index_add__3], Original ATen: [aten.zeros, aten.index, aten.index_add]
        triton_poi_fused_index_index_add_zeros_23.run(arg34_1, arg32_1, buf25, arg33_1, buf57, buf59, 270336, grid=grid(270336), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        buf61 = empty_strided_cuda((48, 16, 128), (2048, 128, 1), torch.float32)
        buf65 = empty_strided_cuda((1, 48, 17, 128), (104448, 2176, 128, 1), torch.float32)
        buf63 = reinterpret_tensor(buf65, (1, 48, 16, 128), (104448, 2176, 128, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [h_off_base_1, cat_6], Original ATen: [aten.mean, aten.cat]
        triton_poi_fused_cat_mean_24.run(buf57, arg35_1, buf59, buf61, buf63, 98304, grid=grid(98304), stream=stream0)
        del arg35_1
        buf64 = reinterpret_tensor(buf65, (1, 48, 1, 128), (104448, 2176, 128, 1), 2048)  # alias
        # Topologically Sorted Source Nodes: [bn_off], Original ATen: [aten.mean]
        triton_per_fused_mean_25.run(buf61, buf64, 6144, 16, grid=grid(6144), stream=stream0)
        del buf61
        buf66 = empty_strided_cuda((48, 17, 1), (17, 1, 816), torch.float32)
        buf67 = empty_strided_cuda((48, 17, 1), (17, 1, 816), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_3], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_26.run(buf65, buf66, buf67, 816, 128, grid=grid(816), stream=stream0)
        del buf63
        del buf64
        buf69 = empty_strided_cuda((48, 16, 17, 4), (1088, 68, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.cat]
        triton_poi_fused_cat_27.run(arg30_1, buf69, 52224, grid=grid(52224), stream=stream0)
        del arg30_1
        buf70 = empty_strided_cuda((48, 16, 17), (272, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.cat]
        triton_poi_fused_cat_28.run(arg31_1, buf70, 13056, grid=grid(13056), stream=stream0)
        del arg31_1
        buf71 = reinterpret_tensor(buf59, (48, 32, 128), (4096, 128, 1), 0); del buf59  # reuse
        buf76 = empty_strided_cuda((48, 2, 17, 128), (4352, 2176, 128, 1), torch.float32)
        buf74 = reinterpret_tensor(buf76, (48, 2, 16, 128), (4352, 2176, 128, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [layer_norm_3, x_2, kv_1], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd, aten.cat]
        triton_poi_fused_cat_constant_pad_nd_native_layer_norm_29.run(buf65, buf66, buf67, arg50_1, arg51_1, buf71, buf74, 196608, grid=grid(196608), stream=stream0)
        del arg50_1
        del arg51_1
        buf72 = empty_strided_cuda((1536, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf71, (1536, 128), (128, 1), 0), reinterpret_tensor(arg52_1, (128, 384), (1, 128), 0), out=buf72)
        buf75 = reinterpret_tensor(buf76, (48, 2, 1, 128), (4352, 2176, 128, 1), 2048)  # alias
        # Topologically Sorted Source Nodes: [block_node], Original ATen: [aten.mean]
        triton_per_fused_mean_30.run(buf71, buf75, 12288, 16, grid=grid(12288), stream=stream0)
        del buf74
        del buf75
        buf77 = empty_strided_cuda((1632, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (1632, 128), (128, 1), 0), reinterpret_tensor(arg52_1, (128, 384), (1, 128), 0), out=buf77)
        del arg52_1
        buf78 = reinterpret_tensor(buf71, (48, 2, 8, 16, 16, 1), (4096, 2048, 256, 16, 1, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf72, arg53_1, buf78, 196608, grid=grid(196608), stream=stream0)
        buf79 = reinterpret_tensor(buf76, (48, 2, 8, 16, 17, 1), (4352, 2176, 272, 17, 1, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf77, arg53_1, buf79, 12288, 17, grid=grid(12288, 17), stream=stream0)
        buf80 = empty_strided_cuda((768, 16, 17), (272, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf78, (768, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf79, (768, 16, 17), (272, 17, 1), 0), out=buf80)
        buf81 = empty_strided_cuda((48, 1, 16, 17, 4), (1088, 52224, 68, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_2, setitem_3], Original ATen: [aten.lift_fresh, aten.fill]
        triton_poi_fused_fill_lift_fresh_33.run(buf69, buf81, 52224, grid=grid(52224), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_2, setitem_3, setitem_4], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
        triton_poi_fused_fill_index_put_lift_fresh_34.run(buf81, 3072, grid=grid(3072), stream=stream0)
        buf83 = empty_strided_cuda((48, 1, 16, 17), (272, 272, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_5], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_35.run(buf81, buf83, 13056, grid=grid(13056), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_5], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_36.run(buf83, 768, grid=grid(768), stream=stream0)
        buf86 = reinterpret_tensor(buf79, (48, 2, 16, 17, 8), (4352, 2176, 17, 1, 272), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [eq_2, scores_5, scores_3, scores_4, attn_probs_1], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_eq_masked_fill_mul_37.run(buf70, buf80, buf83, buf81, buf86, 12288, 17, grid=grid(12288), stream=stream0)
        buf87 = empty_strided_cuda((48, 2, 16, 1, 8), (256, 128, 8, 12288, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_probs_1], Original ATen: [aten._softmax]
        triton_per_fused__softmax_38.run(buf86, buf87, 12288, 17, grid=grid(12288), stream=stream0)
        buf88 = empty_strided_cuda((48, 1, 16, 17, 4), (1088, 1, 68, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_39.run(buf83, buf81, buf88, 52224, grid=grid(52224), stream=stream0)
        del buf81
        buf89 = empty_strided_cuda((13056, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf88, (13056, 4), (4, 1), 0), reinterpret_tensor(arg54_1, (4, 8), (1, 4), 0), out=buf89)
        del arg54_1
        buf90 = reinterpret_tensor(buf86, (48, 2, 8, 16, 17, 1), (4352, 2176, 272, 17, 1, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [x_out_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf90, buf87, buf70, buf89, arg55_1, 768, 272, grid=grid(768, 272), stream=stream0)
        del arg55_1
        buf91 = reinterpret_tensor(buf80, (48, 2, 8, 17, 16, 1), (4352, 2176, 272, 16, 1, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [x_out_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf77, arg53_1, buf91, 208896, grid=grid(208896), stream=stream0)
        del arg53_1
        buf92 = reinterpret_tensor(buf78, (768, 16, 16), (256, 16, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [x_out_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf90, (768, 16, 17), (272, 17, 1), 0), reinterpret_tensor(buf91, (768, 17, 16), (272, 16, 1), 0), out=buf92)
        buf93 = reinterpret_tensor(buf57, (48, 2, 16, 8, 16), (4096, 2048, 128, 16, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [x_out_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_42.run(buf92, buf93, 196608, grid=grid(196608), stream=stream0)
        buf94 = reinterpret_tensor(buf92, (1536, 128), (128, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf93, (1536, 128), (128, 1), 0), reinterpret_tensor(arg56_1, (128, 128), (1, 128), 0), out=buf94)
        del arg56_1
        buf103 = reinterpret_tensor(buf89, (48, 17, 128), (2176, 128, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [x_3, layer_norm_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_43.run(buf65, buf94, arg57_1, arg58_1, arg59_1, buf103, 816, 128, grid=grid(816), stream=stream0)
        del arg58_1
        del arg59_1
        buf99 = empty_strided_cuda((528, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf98, (528, 128), (128, 1), 0), reinterpret_tensor(arg46_1, (128, 512), (1, 128), 0), out=buf99)
        del arg46_1
        buf100 = reinterpret_tensor(buf99, (1, 528, 512), (270336, 512, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf100, arg47_1, 270336, grid=grid(270336), stream=stream0)
        del arg47_1
        buf101 = reinterpret_tensor(buf98, (528, 128), (128, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf100, (528, 512), (512, 1), 0), reinterpret_tensor(arg48_1, (512, 128), (1, 512), 0), out=buf101)
        del arg48_1
        buf102 = reinterpret_tensor(buf101, (1, 528, 128), (67584, 128, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.add]
        triton_poi_fused_add_45.run(buf102, buf29, buf53, arg43_1, arg49_1, 67584, grid=grid(67584), stream=stream0)
        del arg43_1
        del arg49_1
        buf104 = empty_strided_cuda((816, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (816, 128), (128, 1), 0), reinterpret_tensor(arg60_1, (128, 512), (1, 128), 0), out=buf104)
        del arg60_1
        buf105 = reinterpret_tensor(buf104, (48, 17, 512), (8704, 512, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf105, arg61_1, 417792, grid=grid(417792), stream=stream0)
        del arg61_1
        buf106 = reinterpret_tensor(buf103, (816, 128), (128, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf105, (816, 512), (512, 1), 0), reinterpret_tensor(arg62_1, (512, 128), (1, 512), 0), out=buf106)
        del arg62_1
        buf107 = reinterpret_tensor(buf106, (48, 17, 128), (2176, 128, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(buf107, buf65, buf94, arg57_1, arg63_1, 104448, grid=grid(104448), stream=stream0)
        del arg57_1
        del arg63_1
        buf110 = empty_strided_cuda((1, 64, 128), (8192, 128, 1), torch.float32)
        buf108 = reinterpret_tensor(buf110, (1, 16, 128), (8192, 128, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [V], Original ATen: [aten.cat]
        triton_poi_fused_cat_48.run(buf102, buf108, 2048, grid=grid(2048), stream=stream0)
        buf109 = reinterpret_tensor(buf110, (1, 48, 128), (8192, 128, 1), 2048)  # alias
        # Topologically Sorted Source Nodes: [V], Original ATen: [aten.cat]
        triton_poi_fused_cat_49.run(buf107, buf109, 6144, grid=grid(6144), stream=stream0)
        del buf108
        del buf109
        buf111 = empty_strided_cuda((1, 64, 128), (8192, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [msg], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg64_1, (1, 64, 64), (0, 64, 1), 0), buf110, out=buf111)
        buf112 = empty_strided_cuda((64, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf111, (64, 128), (128, 1), 0), reinterpret_tensor(arg65_1, (128, 128), (1, 128), 0), out=buf112)
        del arg65_1
        buf113 = reinterpret_tensor(buf112, (1, 64, 128), (8192, 128, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_50.run(buf113, arg66_1, 8192, grid=grid(8192), stream=stream0)
        del arg66_1
        buf114 = reinterpret_tensor(buf111, (64, 128), (128, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf113, (64, 128), (128, 1), 0), reinterpret_tensor(arg67_1, (128, 128), (1, 128), 0), out=buf114)
        del arg67_1
        buf115 = reinterpret_tensor(buf114, (1, 64, 128), (8192, 128, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [V_1], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(buf115, buf110, arg68_1, 8192, grid=grid(8192), stream=stream0)
        del arg68_1
        buf116 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [V_1, msg_1], Original ATen: [aten.add, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg64_1, (1, 64, 64), (0, 64, 1), 0), buf115, out=buf116)
        buf117 = reinterpret_tensor(buf113, (64, 128), (128, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (64, 128), (128, 1), 0), reinterpret_tensor(arg69_1, (128, 128), (1, 128), 0), out=buf117)
        del arg69_1
        buf118 = reinterpret_tensor(buf117, (1, 64, 128), (8192, 128, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_50.run(buf118, arg70_1, 8192, grid=grid(8192), stream=stream0)
        del arg70_1
        buf119 = reinterpret_tensor(buf116, (64, 128), (128, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (64, 128), (128, 1), 0), reinterpret_tensor(arg71_1, (128, 128), (1, 128), 0), out=buf119)
        del arg71_1
        buf120 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [V_2], Original ATen: [aten.add]
        triton_poi_fused_add_52.run(buf120, buf119, arg72_1, 8192, grid=grid(8192), stream=stream0)
        del arg72_1
        buf121 = reinterpret_tensor(buf119, (1, 64, 128), (8192, 128, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [V_2, msg_2], Original ATen: [aten.add, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg64_1, (1, 64, 64), (0, 64, 1), 0), buf120, out=buf121)
        buf122 = reinterpret_tensor(buf118, (64, 128), (128, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf121, (64, 128), (128, 1), 0), reinterpret_tensor(arg73_1, (128, 128), (1, 128), 0), out=buf122)
        del arg73_1
        buf123 = reinterpret_tensor(buf122, (1, 64, 128), (8192, 128, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_50.run(buf123, arg74_1, 8192, grid=grid(8192), stream=stream0)
        del arg74_1
        buf124 = reinterpret_tensor(buf121, (64, 128), (128, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (64, 128), (128, 1), 0), reinterpret_tensor(arg75_1, (128, 128), (1, 128), 0), out=buf124)
        del arg75_1
        buf125 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [V_3], Original ATen: [aten.add]
        triton_poi_fused_add_52.run(buf125, buf124, arg76_1, 8192, grid=grid(8192), stream=stream0)
        del arg76_1
        buf126 = reinterpret_tensor(buf124, (1, 64, 128), (8192, 128, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [V_3, msg_3], Original ATen: [aten.add, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg64_1, (1, 64, 64), (0, 64, 1), 0), buf125, out=buf126)
        buf127 = reinterpret_tensor(buf123, (64, 128), (128, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf126, (64, 128), (128, 1), 0), reinterpret_tensor(arg77_1, (128, 128), (1, 128), 0), out=buf127)
        del arg77_1
        buf128 = reinterpret_tensor(buf127, (1, 64, 128), (8192, 128, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_50.run(buf128, arg78_1, 8192, grid=grid(8192), stream=stream0)
        del arg78_1
        buf129 = reinterpret_tensor(buf126, (64, 128), (128, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (64, 128), (128, 1), 0), reinterpret_tensor(arg79_1, (128, 128), (1, 128), 0), out=buf129)
        del arg79_1
        buf130 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [V_4], Original ATen: [aten.add]
        triton_poi_fused_add_52.run(buf130, buf129, arg80_1, 8192, grid=grid(8192), stream=stream0)
        del arg80_1
        buf131 = reinterpret_tensor(buf129, (1, 64, 128), (8192, 128, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [V_4, msg_4], Original ATen: [aten.add, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg64_1, (1, 64, 64), (0, 64, 1), 0), buf130, out=buf131)
        buf132 = reinterpret_tensor(buf128, (64, 128), (128, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf131, (64, 128), (128, 1), 0), reinterpret_tensor(arg81_1, (128, 128), (1, 128), 0), out=buf132)
        del arg81_1
        buf133 = reinterpret_tensor(buf132, (1, 64, 128), (8192, 128, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_50.run(buf133, arg82_1, 8192, grid=grid(8192), stream=stream0)
        del arg82_1
        buf134 = reinterpret_tensor(buf131, (64, 128), (128, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf133, (64, 128), (128, 1), 0), reinterpret_tensor(arg83_1, (128, 128), (1, 128), 0), out=buf134)
        del arg83_1
        buf135 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [V_5], Original ATen: [aten.add]
        triton_poi_fused_add_52.run(buf135, buf134, arg84_1, 8192, grid=grid(8192), stream=stream0)
        del arg84_1
        buf136 = reinterpret_tensor(buf134, (1, 64, 128), (8192, 128, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [V_5, msg_5], Original ATen: [aten.add, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg64_1, (1, 64, 64), (0, 64, 1), 0), buf135, out=buf136)
        buf137 = reinterpret_tensor(buf133, (64, 128), (128, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf136, (64, 128), (128, 1), 0), reinterpret_tensor(arg85_1, (128, 128), (1, 128), 0), out=buf137)
        del arg85_1
        buf138 = reinterpret_tensor(buf137, (1, 64, 128), (8192, 128, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_50.run(buf138, arg86_1, 8192, grid=grid(8192), stream=stream0)
        del arg86_1
        buf139 = reinterpret_tensor(buf136, (64, 128), (128, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf138, (64, 128), (128, 1), 0), reinterpret_tensor(arg87_1, (128, 128), (1, 128), 0), out=buf139)
        del arg87_1
        buf140 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [V_6], Original ATen: [aten.add]
        triton_poi_fused_add_52.run(buf140, buf139, arg88_1, 8192, grid=grid(8192), stream=stream0)
        del arg88_1
        buf141 = reinterpret_tensor(buf139, (1, 64, 128), (8192, 128, 1), 0); del buf139  # reuse
        # Topologically Sorted Source Nodes: [V_6, msg_6], Original ATen: [aten.add, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg64_1, (1, 64, 64), (0, 64, 1), 0), buf140, out=buf141)
        buf142 = reinterpret_tensor(buf138, (64, 128), (128, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf141, (64, 128), (128, 1), 0), reinterpret_tensor(arg89_1, (128, 128), (1, 128), 0), out=buf142)
        del arg89_1
        buf143 = reinterpret_tensor(buf142, (1, 64, 128), (8192, 128, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_50.run(buf143, arg90_1, 8192, grid=grid(8192), stream=stream0)
        del arg90_1
        buf144 = reinterpret_tensor(buf141, (64, 128), (128, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (64, 128), (128, 1), 0), reinterpret_tensor(arg91_1, (128, 128), (1, 128), 0), out=buf144)
        del arg91_1
        buf145 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [V_7], Original ATen: [aten.add]
        triton_poi_fused_add_52.run(buf145, buf144, arg92_1, 8192, grid=grid(8192), stream=stream0)
        del arg92_1
        buf146 = reinterpret_tensor(buf144, (1, 64, 128), (8192, 128, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [V_7, msg_7], Original ATen: [aten.add, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg64_1, (1, 64, 64), (0, 64, 1), 0), buf145, out=buf146)
        del arg64_1
        buf147 = reinterpret_tensor(buf143, (64, 128), (128, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf146, (64, 128), (128, 1), 0), reinterpret_tensor(arg93_1, (128, 128), (1, 128), 0), out=buf147)
        del arg93_1
        buf148 = reinterpret_tensor(buf147, (1, 64, 128), (8192, 128, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_50.run(buf148, arg94_1, 8192, grid=grid(8192), stream=stream0)
        del arg94_1
        buf149 = reinterpret_tensor(buf146, (64, 128), (128, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf148, (64, 128), (128, 1), 0), reinterpret_tensor(arg95_1, (128, 128), (1, 128), 0), out=buf149)
        del arg95_1
        buf153 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [V_8, V_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_53.run(buf145, buf149, arg96_1, arg97_1, arg98_1, buf153, 64, 128, grid=grid(64), stream=stream0)
        del arg96_1
        del arg97_1
        del arg98_1
        del buf145
        buf157 = reinterpret_tensor(buf53, (1, 528, 128), (67584, 128, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_6], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_54.run(buf153, buf102, arg99_1, arg100_1, buf157, 528, 128, grid=grid(528), stream=stream0)
        del arg100_1
        del arg99_1
        buf158 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (528, 128), (128, 1), 0), reinterpret_tensor(arg101_1, (128, 384), (1, 128), 0), out=buf158)
        buf159 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (528, 128), (128, 1), 0), reinterpret_tensor(arg101_1, (128, 384), (1, 128), 0), out=buf159)
        del arg101_1
        buf160 = reinterpret_tensor(buf157, (16, 8, 33, 16, 1, 1), (4224, 528, 16, 1, 1, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [einsum_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf158, arg102_1, buf160, 67584, grid=grid(67584), stream=stream0)
        del buf158
        buf161 = reinterpret_tensor(buf29, (16, 8, 16, 1, 33, 1), (4224, 528, 33, 33, 1, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [einsum_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf159, arg102_1, buf161, 2048, 33, grid=grid(2048, 33), stream=stream0)
        buf162 = reinterpret_tensor(buf48, (128, 33, 33), (1089, 33, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [einsum_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf160, (128, 33, 16), (528, 16, 1), 0), reinterpret_tensor(buf161, (128, 16, 33), (528, 33, 1), 0), out=buf162)
        buf163 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [bias_physics_4], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(arg28_1, buf163, 69696, grid=grid(69696), stream=stream0)
        del arg28_1
        # Topologically Sorted Source Nodes: [bias_physics_4, setitem_8], Original ATen: [aten.cat, aten.lift_fresh, aten.index_put]
        triton_poi_fused_cat_index_put_lift_fresh_12.run(buf163, 2048, grid=grid(2048), stream=stream0)
        buf165 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [setitem_9], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_13.run(buf163, buf165, 17424, grid=grid(17424), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_9], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_14.run(buf165, 512, grid=grid(512), stream=stream0)
        buf168 = reinterpret_tensor(buf49, (1, 16, 33, 33, 8), (143360, 8960, 33, 1, 1120), 0); del buf49  # reuse
        buf169 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [eq_4, scores_8, scores_6, scores_7, attn_probs_2], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_eq_masked_fill_mul_15.run(arg29_1, buf162, buf165, buf163, buf168, buf169, 4224, 33, grid=grid(4224), stream=stream0)
        buf170 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(buf165, buf163, buf170, 69696, grid=grid(69696), stream=stream0)
        del buf163
        del buf165
        buf171 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_17.run(buf170, buf171, 69696, grid=grid(69696), stream=stream0)
        del buf170
        buf172 = reinterpret_tensor(buf162, (17424, 8), (8, 1), 0); del buf162  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf171, reinterpret_tensor(arg103_1, (4, 8), (1, 4), 0), out=buf172)
        del arg103_1
        del buf171
        buf173 = reinterpret_tensor(buf168, (16, 8, 33, 33, 1, 1), (8960, 1120, 33, 1, 1, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [x_out_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf173, buf169, arg29_1, buf172, arg104_1, 128, 1089, grid=grid(128, 1089), stream=stream0)
        del arg104_1
        del arg29_1
        del buf169
        del buf172
        buf174 = reinterpret_tensor(buf161, (16, 8, 33, 1, 16, 1), (4224, 528, 16, 16, 1, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [x_out_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf159, arg102_1, buf174, 67584, grid=grid(67584), stream=stream0)
        del arg102_1
        del buf159
        buf175 = reinterpret_tensor(buf160, (128, 33, 16), (528, 16, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [x_out_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf173, (128, 33, 33), (1120, 33, 1), 0), reinterpret_tensor(buf174, (128, 33, 16), (528, 16, 1), 0), out=buf175)
        del buf173
        buf176 = reinterpret_tensor(buf174, (1, 16, 33, 8, 16), (67584, 4224, 128, 16, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [x_out_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf175, buf176, 67584, grid=grid(67584), stream=stream0)
        buf177 = reinterpret_tensor(buf175, (528, 128), (128, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf176, (528, 128), (128, 1), 0), reinterpret_tensor(arg105_1, (128, 128), (1, 128), 0), out=buf177)
        del arg105_1
        buf211 = reinterpret_tensor(buf176, (1, 528, 128), (67584, 128, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [x_5, layer_norm_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_55.run(buf153, buf102, buf177, arg106_1, arg107_1, arg108_1, buf211, 528, 128, grid=grid(528), stream=stream0)
        del arg107_1
        del arg108_1
        buf181 = buf67; del buf67  # reuse
        buf182 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_8], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_56.run(buf153, buf107, buf181, buf182, 816, 128, grid=grid(816), stream=stream0)
        buf184 = reinterpret_tensor(buf94, (48, 32, 128), (4096, 128, 1), 0); del buf94  # reuse
        buf189 = reinterpret_tensor(buf91, (48, 2, 17, 128), (4352, 2176, 128, 1), 0); del buf91  # reuse
        buf187 = reinterpret_tensor(buf189, (48, 2, 16, 128), (4352, 2176, 128, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [layer_norm_8, x_7, kv_3], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd, aten.cat]
        triton_poi_fused_cat_constant_pad_nd_native_layer_norm_57.run(buf153, buf107, buf181, buf182, arg113_1, arg114_1, buf184, buf187, 196608, grid=grid(196608), stream=stream0)
        del arg113_1
        del arg114_1
        del buf181
        del buf182
        buf185 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf184, (1536, 128), (128, 1), 0), reinterpret_tensor(arg115_1, (128, 384), (1, 128), 0), out=buf185)
        buf188 = reinterpret_tensor(buf189, (48, 2, 1, 128), (4352, 2176, 128, 1), 2048)  # alias
        # Topologically Sorted Source Nodes: [block_node_1], Original ATen: [aten.mean]
        triton_per_fused_mean_30.run(buf184, buf188, 12288, 16, grid=grid(12288), stream=stream0)
        del buf187
        del buf188
        buf190 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (1632, 128), (128, 1), 0), reinterpret_tensor(arg115_1, (128, 384), (1, 128), 0), out=buf190)
        del arg115_1
        buf191 = reinterpret_tensor(buf184, (48, 2, 8, 16, 16, 1), (4096, 2048, 256, 16, 1, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [einsum_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf185, arg116_1, buf191, 196608, grid=grid(196608), stream=stream0)
        del buf185
        buf192 = reinterpret_tensor(buf189, (48, 2, 8, 16, 17, 1), (4352, 2176, 272, 17, 1, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [einsum_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf190, arg116_1, buf192, 12288, 17, grid=grid(12288, 17), stream=stream0)
        buf193 = reinterpret_tensor(buf90, (768, 16, 17), (272, 17, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [einsum_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf191, (768, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf192, (768, 16, 17), (272, 17, 1), 0), out=buf193)
        buf194 = reinterpret_tensor(buf88, (48, 1, 16, 17, 4), (1088, 52224, 68, 4, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [setitem_10, setitem_11], Original ATen: [aten.lift_fresh, aten.fill]
        triton_poi_fused_fill_lift_fresh_33.run(buf69, buf194, 52224, grid=grid(52224), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_10, setitem_11, setitem_12], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
        triton_poi_fused_fill_index_put_lift_fresh_34.run(buf194, 3072, grid=grid(3072), stream=stream0)
        buf196 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [setitem_13], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_35.run(buf194, buf196, 13056, grid=grid(13056), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_13], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_36.run(buf196, 768, grid=grid(768), stream=stream0)
        buf199 = reinterpret_tensor(buf192, (48, 2, 16, 17, 8), (4352, 2176, 17, 1, 272), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [eq_6, scores_11, scores_9, scores_10, attn_probs_3], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_eq_masked_fill_mul_37.run(buf70, buf193, buf196, buf194, buf199, 12288, 17, grid=grid(12288), stream=stream0)
        buf200 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [attn_probs_3], Original ATen: [aten._softmax]
        triton_per_fused__softmax_38.run(buf199, buf200, 12288, 17, grid=grid(12288), stream=stream0)
        buf201 = reinterpret_tensor(buf69, (48, 1, 16, 17, 4), (1088, 1, 68, 4, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_39.run(buf196, buf194, buf201, 52224, grid=grid(52224), stream=stream0)
        del buf194
        del buf196
        buf202 = reinterpret_tensor(buf65, (13056, 8), (8, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf201, (13056, 4), (4, 1), 0), reinterpret_tensor(arg117_1, (4, 8), (1, 4), 0), out=buf202)
        del arg117_1
        del buf201
        buf203 = reinterpret_tensor(buf199, (48, 2, 8, 16, 17, 1), (4352, 2176, 272, 17, 1, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [x_out_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf203, buf200, buf70, buf202, arg118_1, 768, 272, grid=grid(768, 272), stream=stream0)
        del arg118_1
        del buf70
        buf204 = reinterpret_tensor(buf193, (48, 2, 8, 17, 16, 1), (4352, 2176, 272, 16, 1, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [x_out_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf190, arg116_1, buf204, 208896, grid=grid(208896), stream=stream0)
        del arg116_1
        del buf190
        buf205 = reinterpret_tensor(buf191, (768, 16, 16), (256, 16, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [x_out_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf203, (768, 16, 17), (272, 17, 1), 0), reinterpret_tensor(buf204, (768, 17, 16), (272, 16, 1), 0), out=buf205)
        del buf203
        del buf204
        buf206 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [x_out_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_42.run(buf205, buf206, 196608, grid=grid(196608), stream=stream0)
        buf207 = reinterpret_tensor(buf205, (1536, 128), (128, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (1536, 128), (128, 1), 0), reinterpret_tensor(arg119_1, (128, 128), (1, 128), 0), out=buf207)
        del arg119_1
        del buf206
        buf219 = reinterpret_tensor(buf202, (48, 17, 128), (2176, 128, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [x_8, layer_norm_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_58.run(buf153, buf107, buf207, arg120_1, arg121_1, arg122_1, buf219, 816, 128, grid=grid(816), stream=stream0)
        del arg121_1
        del arg122_1
        buf212 = reinterpret_tensor(buf100, (528, 512), (512, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf211, (528, 128), (128, 1), 0), reinterpret_tensor(arg109_1, (128, 512), (1, 128), 0), out=buf212)
        del arg109_1
        buf213 = reinterpret_tensor(buf212, (1, 528, 512), (270336, 512, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf213, arg110_1, 270336, grid=grid(270336), stream=stream0)
        del arg110_1
        buf214 = reinterpret_tensor(buf211, (528, 128), (128, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf213, (528, 512), (512, 1), 0), reinterpret_tensor(arg111_1, (512, 128), (1, 512), 0), out=buf214)
        del arg111_1
        del buf213
        buf215 = reinterpret_tensor(buf25, (1, 16, 32, 128), (65536, 4096, 128, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [h_diag], Original ATen: [aten.clone]
        triton_poi_fused_clone_59.run(buf153, buf102, buf177, arg106_1, buf214, arg112_1, buf215, 65536, grid=grid(65536), stream=stream0)
        del arg106_1
        del arg112_1
        del buf102
        del buf177
        del buf214
        buf216 = reinterpret_tensor(buf24, (512, 128), (128, 1), 0); del buf24  # reuse
        buf231 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [u_leaf], Original ATen: [aten.addmm]
        triton_poi_fused_addmm_60.run(buf215, buf216, buf231, 65536, grid=grid(65536), stream=stream0)
        buf217 = empty_strided_cuda((512, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_leaf], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg128_1, buf216, reinterpret_tensor(arg127_1, (128, 32), (1, 128), 0), alpha=1, beta=1, out=buf217)
        del arg127_1
        del arg128_1
        del buf216
        buf218 = empty_strided_cuda((16, 32, 32), (1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf217, (16, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf217, (16, 32, 32), (1024, 1, 32), 0), out=buf218)
        del buf217
        buf220 = reinterpret_tensor(buf105, (816, 512), (512, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (816, 128), (128, 1), 0), reinterpret_tensor(arg123_1, (128, 512), (1, 128), 0), out=buf220)
        del arg123_1
        buf221 = reinterpret_tensor(buf220, (48, 17, 512), (8704, 512, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_46.run(buf221, arg124_1, 417792, grid=grid(417792), stream=stream0)
        del arg124_1
        buf222 = reinterpret_tensor(buf219, (816, 128), (128, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (816, 512), (512, 1), 0), reinterpret_tensor(arg125_1, (512, 128), (1, 512), 0), out=buf222)
        del arg125_1
        del buf221
        buf223 = reinterpret_tensor(buf222, (48, 17, 128), (2176, 128, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [x_8, x_9], Original ATen: [aten.add]
        triton_poi_fused_add_61.run(buf223, buf153, buf107, buf207, arg120_1, arg126_1, 104448, grid=grid(104448), stream=stream0)
        del arg120_1
        del arg126_1
        del buf107
        del buf207
        buf224 = reinterpret_tensor(buf200, (48, 16, 16), (256, 16, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [u_leaf_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf223, (48, 16, 128), (2176, 128, 1), 0), reinterpret_tensor(arg129_1, (48, 128, 16), (0, 1, 128), 0), out=buf224)
        del arg129_1
        buf225 = empty_strided_cuda((48, 16, 16), (256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_leaf], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf223, (48, 16, 128), (2176, 128, 1), 0), reinterpret_tensor(arg131_1, (48, 128, 16), (0, 1, 128), 0), out=buf225)
        del arg131_1
        del buf223
        buf226 = reinterpret_tensor(buf224, (48, 1, 16, 16), (256, 1, 16, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [u_leaf_1], Original ATen: [aten.add]
        triton_poi_fused_add_62.run(buf226, arg130_1, 12288, grid=grid(12288), stream=stream0)
        del arg130_1
        buf227 = reinterpret_tensor(buf225, (48, 1, 16, 16), (256, 1, 16, 1), 0); del buf225  # reuse
        # Topologically Sorted Source Nodes: [v_leaf], Original ATen: [aten.add]
        triton_poi_fused_add_62.run(buf227, arg132_1, 12288, grid=grid(12288), stream=stream0)
        del arg132_1
        buf228 = empty_strided_cuda((48, 16, 16), (256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf226, (48, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf227, (48, 16, 16), (256, 1, 16), 0), out=buf228)
        del buf226
        del buf227
        buf229 = reinterpret_tensor(buf153, (512, 16), (16, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [node_U], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg136_1, reinterpret_tensor(buf215, (512, 128), (128, 1), 0), reinterpret_tensor(arg135_1, (128, 16), (1, 128), 0), alpha=1, beta=1, out=buf229)
        del arg135_1
        del arg136_1
        buf230 = reinterpret_tensor(buf149, (512, 16), (16, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [node_V], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg138_1, reinterpret_tensor(buf215, (512, 128), (128, 1), 0), reinterpret_tensor(arg137_1, (128, 16), (1, 128), 0), alpha=1, beta=1, out=buf230)
        del arg137_1
        del arg138_1
        del buf215
        buf232 = empty_strided_cuda((512, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf231, reinterpret_tensor(arg133_1, (128, 1), (1, 128), 0), out=buf232)
        del arg133_1
        del buf231
        buf233 = reinterpret_tensor(buf232, (1, 16, 32, 1), (512, 32, 1, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [sigmoid], Original ATen: [aten.sigmoid]
        triton_poi_fused_sigmoid_63.run(buf233, arg134_1, 512, grid=grid(512), stream=stream0)
        del arg134_1
        buf234 = empty_strided_cuda((1, 45568), (45568, 1), torch.float32)
        # Topologically Sorted Source Nodes: [packed_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_64.run(buf218, buf228, buf229, buf230, buf233, buf234, 45568, grid=grid(45568), stream=stream0)
        del buf218
        del buf228
        del buf229
        del buf230
    return (buf234, reinterpret_tensor(buf233, (1, 16, 32), (512, 32, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 512, 9), (67716, 9, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 18), (18, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((2, 3313), (3313, 1), device='cuda:0', dtype=torch.int64)
    arg7_1 = rand_strided((3313, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((16, 32, 33, 4), (4224, 132, 4, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((16, 32, 33), (1056, 33, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((48, 32, 33, 4), (4224, 132, 4, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((48, 32, 33), (1056, 33, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((66, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg33_1 = rand_strided((66, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg34_1 = rand_strided((66, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg35_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg36_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((8, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((8, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((8, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((8, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((32, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((16, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((16, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((16, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((16, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
