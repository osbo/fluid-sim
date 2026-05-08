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
#   %add_tensor_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_35, %arg15_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze, %add_tensor_35), kwargs = {})
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/s5/cs5n35upzsij3edkxal2dgatygs2wbbam55qor3u2uz2f5cuxzb6.py
# Topologically Sorted Source Nodes: [layer_norm_1], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_1 => add_8, add_9, mul_13, mul_14, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %getitem_3), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %arg36_1), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %arg37_1), kwargs = {})
triton_per_fused_native_layer_norm_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/7v/c7v7ooho2ctpdngvpor4ulzqi6aedkhx5bbhkkcuqa7j27wrknnf.py
# Topologically Sorted Source Nodes: [mask_base_1, sum_1], Original ATen: [aten.clone, aten.sum]
# Source node to ATen node mapping:
#   mask_base_1 => clone_1
#   sum_1 => sum_1
# Graph fragment:
#   %clone_1 : [num_users=5] = call_function[target=torch.ops.aten.clone.default](args = (%unsqueeze_11,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%clone_1, [-1]), kwargs = {})
triton_per_fused_clone_sum_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_sum_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (129*x0)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr1 + (r1 + (128*x0)), tmp0, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/7o/c7o2yms2jvg7x4tvi2tmf72sc3djbqu7wg2adotd6syf2fl6yhag.py
# Topologically Sorted Source Nodes: [lt, _need_diag, maximum, copy_], Original ATen: [aten.lt, aten._to_copy, aten.maximum, aten.copy]
# Source node to ATen node mapping:
#   _need_diag => convert_element_type_1
#   copy_ => copy
#   lt => lt
#   maximum => maximum
# Graph fragment:
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%sum_1, 1), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt, torch.float32), kwargs = {})
#   %maximum : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%diagonal, %convert_element_type_1), kwargs = {})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%diagonal, %maximum), kwargs = {})
#   %copy__default : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%diagonal_default, %copy), kwargs = {})
triton_poi_fused__to_copy_copy_lt_maximum_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_copy_lt_maximum_8', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((130*x0) + (16512*x1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = 1.0
    tmp3 = tmp1 < tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = triton_helpers.maximum(tmp0, tmp4)
    tl.store(out_ptr0 + ((129*x0) + (16384*x1)), tmp5, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/jj/cjj436jc4cqbcykyt4z2vphywwyzxf77twa4qmhnuciujengrqvl.py
# Topologically Sorted Source Nodes: [bias_physics], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   bias_physics => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {})
triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (516*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/nr/cnrjdxs6455zg5z4saggotuto7ev6hthm32vzyef4s6xmi5ehli2.py
# Topologically Sorted Source Nodes: [bias_physics, setitem], Original ATen: [aten.clone, aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   bias_physics => clone
#   setitem => full_default_4, index_put_4
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_4 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%clone, [None, None, %iota, %iota], %full_default_4), kwargs = {})
triton_poi_fused_clone_index_put_lift_fresh_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_index_put_lift_fresh_10', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 4
    x1 = (xindex // 4) % 128
    x2 = (xindex // 512)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (516*x1) + (65536*x2)), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/if/cifkjvqjwklm7pebh2imi2due4zmsyl6cghn4dvat64ob4f5kxr6.py
# Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_1 => full_default_5, index_put_5
# Graph fragment:
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_5 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_5, [None, None, %iota, %iota], %full_default_5), kwargs = {})
triton_poi_fused_index_put_lift_fresh_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_11', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + (4*x0)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/r3/cr3n4ouxih3bmmygpeplckwroio6trcmcezkp3my7zasi5xbuelc.py
# Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_1 => full_default_5, index_put_5
# Graph fragment:
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_5 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_5, [None, None, %iota, %iota], %full_default_5), kwargs = {})
triton_poi_fused_index_put_lift_fresh_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_12', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = 1.0
    tl.store(out_ptr0 + ((129*x0) + (16384*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/3l/c3lgtgumvss5d2t4x6b5bdqwwbnetnsupamygdla67xular4zzeh.py
# Topologically Sorted Source Nodes: [q_f, k_f, v_f, eq_1, logit_bias_1, logit_bias, x_soft], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   eq_1 => eq_1
#   k_f => clone_3
#   logit_bias => clone_5
#   logit_bias_1 => full_default_7, where_1
#   q_f => clone_2
#   v_f => clone_4
#   x_soft => _scaled_dot_product_efficient_attention
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_17,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_18,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_19,), kwargs = {memory_format: torch.contiguous_format})
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%expand_8, 0), kwargs = {})
#   %full_default_7 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_6,), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_1, %full_default_7, %clone_5), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_2, %clone_3, %clone_4, %expand_9, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_13', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16) % 128
    x2 = (xindex // 2048) % 8
    x3 = (xindex // 16384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*x2) + (384*x1) + (49152*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ih/cihrikdm2pzsfp3td7jxeyt7gaser3wzy45k4pwfchj7fmxftdgf.py
# Topologically Sorted Source Nodes: [q_f, k_f, v_f, eq_1, logit_bias_1, logit_bias, x_soft], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   eq_1 => eq_1
#   k_f => clone_3
#   logit_bias => clone_5
#   logit_bias_1 => full_default_7, where_1
#   q_f => clone_2
#   v_f => clone_4
#   x_soft => _scaled_dot_product_efficient_attention
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_17,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_18,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_19,), kwargs = {memory_format: torch.contiguous_format})
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%expand_8, 0), kwargs = {})
#   %full_default_7 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_6,), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_1, %full_default_7, %clone_5), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_2, %clone_3, %clone_4, %expand_9, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_14', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16) % 128
    x2 = (xindex // 2048) % 8
    x3 = (xindex // 16384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0 + (16*x2) + (384*x1) + (49152*x3)), None)
    tmp1 = tl.load(in_ptr1 + (128 + x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ec/cecdxtyd4ziwdiyoofonebeu4gly64zholmzwx64dwpap5hvpoqw.py
# Topologically Sorted Source Nodes: [q_f, k_f, v_f, eq_1, logit_bias_1, logit_bias, x_soft, einsum], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   einsum => clone_7
#   eq_1 => eq_1
#   k_f => clone_3
#   logit_bias => clone_5
#   logit_bias_1 => full_default_7, where_1
#   q_f => clone_2
#   v_f => clone_4
#   x_soft => _scaled_dot_product_efficient_attention
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_17,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_18,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_19,), kwargs = {memory_format: torch.contiguous_format})
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%expand_8, 0), kwargs = {})
#   %full_default_7 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_6,), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_1, %full_default_7, %clone_5), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_2, %clone_3, %clone_4, %expand_9, False), kwargs = {})
#   %clone_7 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_24,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_15', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16) % 128
    x2 = (xindex // 2048) % 8
    x3 = (xindex // 16384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + (16*x2) + (384*x1) + (49152*x3)), None)
    tmp1 = tl.load(in_ptr1 + (256 + x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
    tl.store(out_ptr1 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/sw/cswayud6ehlbggddoisvflbdnpwugea2ilpbtjmx6xn6fdbvjtxp.py
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
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_16', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 4
    x1 = (xindex // 4)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2), None)
    tmp0 = x0
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/co/ccohjeabph7vny3zzseggj6skfy6bskyo4jgaazr5qgt3ccmc5nj.py
# Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_11 => add_10, erf_3, mul_15, mul_16, mul_17
# Graph fragment:
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_21, 0.5), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_21, 0.7071067811865476), kwargs = {})
#   %erf_3 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_16,), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_3, 1), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %add_10), kwargs = {})
triton_poi_fused_gelu_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_17', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 16
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/w6/cw6hso23ud2x4ptayxr56khs63pbawt5ndhmvqweryzxvdazrwcy.py
# Topologically Sorted Source Nodes: [q_f, k_f, v_f, eq_1, logit_bias_1, logit_bias, x_soft, einsum], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   einsum => clone_6
#   eq_1 => eq_1
#   k_f => clone_3
#   logit_bias => clone_5
#   logit_bias_1 => full_default_7, where_1
#   q_f => clone_2
#   v_f => clone_4
#   x_soft => _scaled_dot_product_efficient_attention
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_17,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_18,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_19,), kwargs = {memory_format: torch.contiguous_format})
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%expand_8, 0), kwargs = {})
#   %full_default_7 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_6,), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_1, %full_default_7, %clone_5), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_2, %clone_3, %clone_4, %expand_9, False), kwargs = {})
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_23,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_18', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y1 = (yindex // 8)
    y3 = yindex
    y0 = yindex % 8
    tmp0 = tl.load(in_ptr0 + (x2 + (16384*y1)), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2 + (16384*y1)), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (3 + (4*x2) + (65536*y1)), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0 + (8*x2) + (131072*y1)), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1, 1], 3, tl.int32)
    tmp4 = tmp3 == tmp3
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = float("-inf")
    tmp9 = tl.where(tmp2, tmp8, tmp7)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.where(tmp2, tmp1, tmp12)
    tl.store(out_ptr0 + (x2 + (16384*y3)), tmp9, ymask)
    tl.store(out_ptr1 + (x2 + (16384*y3)), tmp13, ymask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/xt/cxtrl6j3ejejcleeenobihtv2rthkewufjfcecmecae6ucgsyczf.py
# Topologically Sorted Source Nodes: [diag_prep_block], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   diag_prep_block => mean
# Graph fragment:
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%view_8, [2], True), kwargs = {})
triton_red_fused_mean_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_19', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = 128.0
    tmp5 = tmp2 / tmp4
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/me/cmewtoj37vgpfcclstkx6b4ppdx33pczlrdi2mbfarvzrqqaecme.py
# Topologically Sorted Source Nodes: [x_mid, mul_2, x_mid_1], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#   mul_2 => mul_18
#   x_mid => add_11
#   x_mid_1 => add_12
# Graph fragment:
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_31, %view_35), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_16, %view_38), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %mul_18), kwargs = {})
triton_poi_fused_add_mul_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_20', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x0 = xindex % 16
    x1 = (xindex // 16) % 8
    x2 = (xindex // 128) % 128
    x3 = (xindex // 16384)
    x5 = (xindex // 16)
    x6 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (16*x2) + (2048*x1) + (16384*x3)), None)
    tmp3 = tl.load(in_ptr1 + (x5), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (256 + x6 + (384*x3)), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (256 + x6), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = tmp2 + tmp10
    tl.store(in_out_ptr0 + (x4), tmp11, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ji/cjiony5h3quwzg7bbwh7f4yij5cmnzshwhcrcyv23li4jz6wrivi.py
# Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_14 => add_16, erf_4, mul_21, mul_22, mul_23
# Graph fragment:
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_46, 0.5), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_46, 0.7071067811865476), kwargs = {})
#   %erf_4 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_22,), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_4, 1), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %add_16), kwargs = {})
triton_poi_fused_gelu_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_21', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/2v/c2visghvyy74de6qxxgywrx5jhh2svmeaa3a32ecgebnkojwsefu.py
# Topologically Sorted Source Nodes: [x, x_1, layer_norm_5], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_5 => add_29, add_30, mul_35, mul_36, rsqrt_5, sub_5, var_mean_5
#   x => add_13
#   x_1 => add_17
# Graph fragment:
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %view_44), kwargs = {})
#   %add_17 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_48), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_17, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_17, %getitem_19), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_18, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_29,), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_5), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_35, %arg72_1), kwargs = {})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_36, %arg73_1), kwargs = {})
triton_per_fused_add_native_layer_norm_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_22', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_out_ptr0 + (r1 + (128*x0)), xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 128.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp8, xmask)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp35, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ma/cmajhjxmhf7jlbej632hbok3eef4swwt2q5fuygwzehckw5lmimn.py
# Topologically Sorted Source Nodes: [row_sum_1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   row_sum_1 => full_default_8
# Graph fragment:
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([6, 1, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_23', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/fp/cfpnuo2diyssovwghf4emvwmcybkqrwul362wucc3noots7j6c3w.py
# Topologically Sorted Source Nodes: [row_sum_1, getitem_17, index_add__4, col_sum_1, getitem_18, index_add__5, row_sum, getitem_8, index_add__2, col_sum, getitem_9, index_add__3], Original ATen: [aten.zeros, aten.index, aten.index_add]
# Source node to ATen node mapping:
#   col_sum => full_default_3
#   col_sum_1 => full_default_9
#   getitem_17 => index_4
#   getitem_18 => index_5
#   getitem_8 => index_2
#   getitem_9 => index_3
#   index_add__2 => index_put_2
#   index_add__3 => index_put_3
#   index_add__4 => index_put_6
#   index_add__5 => index_put_7
#   row_sum => full_default_2
#   row_sum_1 => full_default_8
# Graph fragment:
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([6, 1, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_4 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%permute_31, [%arg32_1]), kwargs = {})
#   %index_put_6 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_8, [%arg34_1], %index_4, True), kwargs = {})
#   %full_default_9 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([6, 1, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_5 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%permute_31, [%arg33_1]), kwargs = {})
#   %index_put_7 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_9, [%arg34_1], %index_5, True), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([6, 1, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_2 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%permute_11, [%arg32_1]), kwargs = {})
#   %index_put_2 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_2, [%arg34_1], %index_2, True), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([6, 1, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_3 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%permute_11, [%arg33_1]), kwargs = {})
#   %index_put_3 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_3, [%arg34_1], %index_3, True), kwargs = {})
triton_poi_fused_index_index_add_zeros_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_index_add_zeros_24', 'mutated_arg_names': ['out_ptr0', 'out_ptr1', 'out_ptr2', 'out_ptr3'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 16384)
    x0 = xindex % 16384
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 6, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 6), "index out of bounds: 0 <= tmp4 < 6")
    tmp7 = tl.full([XBLOCK], 4, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert((0 <= tmp10) & (tmp10 < 4), "index out of bounds: 0 <= tmp10 < 4")
    tmp12 = tl.load(in_ptr2 + (x0 + (16384*tmp10)), None)
    tmp14 = tmp13 + tmp7
    tmp15 = tmp13 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp13)
    tl.device_assert((0 <= tmp16) & (tmp16 < 4), "index out of bounds: 0 <= tmp16 < 4")
    tmp18 = tl.load(in_ptr2 + (x0 + (16384*tmp16)), None)
    tmp19 = tl.load(in_ptr4 + (x0 + (16384*tmp10)), None)
    tmp20 = tl.load(in_ptr4 + (x0 + (16384*tmp16)), None)
    tl.atomic_add(out_ptr0 + (x0 + (16384*tmp4)), tmp12, None, sem='relaxed')
    tl.atomic_add(out_ptr1 + (x0 + (16384*tmp4)), tmp18, None, sem='relaxed')
    tl.atomic_add(out_ptr2 + (x0 + (16384*tmp4)), tmp19, None, sem='relaxed')
    tl.atomic_add(out_ptr3 + (x0 + (16384*tmp4)), tmp20, None, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/jx/cjx4mtgvsdb4vixbjppmbvvemrl2tiv5dpyryojjgugj6ztjib66.py
# Topologically Sorted Source Nodes: [layer_norm_3, off_prep_block], Original ATen: [aten.native_layer_norm, aten.mean]
# Source node to ATen node mapping:
#   layer_norm_3 => add_19, add_20, mul_24, mul_25, rsqrt_3, sub_3, var_mean_3
#   off_prep_block => mean_2
# Graph fragment:
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_51, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_51, %getitem_11), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, %arg54_1), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_25, %arg55_1), kwargs = {})
#   %mean_2 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%view_12, [2], True), kwargs = {})
triton_red_fused_mean_native_layer_norm_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_native_layer_norm_25', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 3, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = (xindex // 128)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x0 = xindex % 128
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr3 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr4 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 / tmp2
        tmp5 = tmp4 / tmp2
        tmp6 = tmp3 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
        tmp12 = tmp11 / tmp2
        tmp14 = tmp13 / tmp2
        tmp15 = tmp12 + tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp19 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.load(in_ptr2 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp32 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp34 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp1.to(tl.float32)
        tmp21 = tmp19 / tmp20
        tmp23 = tmp22 / tmp20
        tmp24 = tmp21 + tmp23
        tmp25 = tmp24 - tmp8
        tmp26 = 128.0
        tmp27 = tmp9 / tmp26
        tmp28 = 1e-05
        tmp29 = tmp27 + tmp28
        tmp30 = libdevice.rsqrt(tmp29)
        tmp31 = tmp25 * tmp30
        tmp33 = tmp31 * tmp32
        tmp35 = tmp33 + tmp34
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp35, rmask & xmask)
    tmp36 = 128.0
    tmp37 = tmp17 / tmp36
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp37, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ja/cjaojzv5ptgwhffxryvvbij66c6af5w2sw5r65wl76zfkms5p3yy.py
# Topologically Sorted Source Nodes: [mask_base_2], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   mask_base_2 => full_default_10
# Graph fragment:
#   %full_default_10 : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([6, 1, 128, 128], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_clone_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_26', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/rx/crxk53xz3rma2ugk4rnkjn3nkngdripuqmump6nofaekcehlewyi.py
# Topologically Sorted Source Nodes: [mask_base_2, sum_2, lt_1, _need_diag_1, maximum_1, copy__1], Original ATen: [aten.clone, aten.sum, aten.lt, aten._to_copy, aten.maximum, aten.copy]
# Source node to ATen node mapping:
#   _need_diag_1 => convert_element_type_3
#   copy__1 => copy_1
#   lt_1 => lt_1
#   mask_base_2 => full_default_10
#   maximum_1 => maximum_1
#   sum_2 => sum_2
# Graph fragment:
#   %full_default_10 : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([6, 1, 128, 128], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%full_default_10, [-1]), kwargs = {})
#   %lt_1 : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%sum_2, 1), kwargs = {})
#   %convert_element_type_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt_1, torch.float32), kwargs = {})
#   %maximum_1 : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%diagonal_2, %convert_element_type_3), kwargs = {})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%diagonal_2, %maximum_1), kwargs = {})
#   %copy__default_1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%diagonal_default_1, %copy_1), kwargs = {})
triton_per_fused__to_copy_clone_copy_lt_maximum_sum_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_clone_copy_lt_maximum_sum_27', 'mutated_arg_names': ['out_ptr1'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex
    x2 = xindex % 128
    x3 = (xindex // 128)
    tmp0 = 1.0
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = tmp4 < tmp0
    tmp6 = tmp5.to(tl.float32)
    tmp7 = triton_helpers.maximum(tmp0, tmp6)
    tl.store(out_ptr1 + ((129*x2) + (16384*x3)), tmp7, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/et/cetywfrec4l36fkutbl2g2sgfmler43q2u66cb4kis4ejexb4ary.py
# Topologically Sorted Source Nodes: [bias_physics_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   bias_physics_1 => clone_8
# Graph fragment:
#   %clone_8 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_32,), kwargs = {})
triton_poi_fused_clone_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_28', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (516*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/tj/ctjmjybzll3rybl3t6slggwzab6ovqi6ztcba433izqjnsaed5r3.py
# Topologically Sorted Source Nodes: [bias_physics_1, setitem_2], Original ATen: [aten.clone, aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   bias_physics_1 => clone_8
#   setitem_2 => full_default_11, index_put_8
# Graph fragment:
#   %clone_8 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_32,), kwargs = {})
#   %full_default_11 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_8 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%clone_8, [None, None, %iota_1, %iota_1], %full_default_11), kwargs = {})
triton_poi_fused_clone_index_put_lift_fresh_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_index_put_lift_fresh_29', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = (xindex // 4) % 128
    x2 = (xindex // 512)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (516*x1) + (65536*x2)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ti/ctiocvzhea5slbzyjdj76f4yh2pacp4pwdugb2gegfri2xkmtedf.py
# Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_3 => full_default_12, index_put_9
# Graph fragment:
#   %full_default_12 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_9 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_10, [None, None, %iota_1, %iota_1], %full_default_12), kwargs = {})
triton_poi_fused_index_put_lift_fresh_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_30', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + (4*x0)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/tu/cturbe54665o7twzt7xhf2wnfrn2gohybnhjlghkq5iitn52hdhv.py
# Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_3 => full_default_12, index_put_9
# Graph fragment:
#   %full_default_12 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_9 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_10, [None, None, %iota_1, %iota_1], %full_default_12), kwargs = {})
triton_poi_fused_index_put_lift_fresh_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_31', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = 1.0
    tl.store(out_ptr0 + ((129*x0) + (16384*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/q4/cq4q6t4ncwkmr7dynn42wbr3c2m7bonfs7x534lcveh6wnomcrcy.py
# Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, eq_3, logit_bias_3, logit_bias_2, x_soft_2], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   eq_3 => eq_3
#   k_f_1 => clone_11
#   logit_bias_2 => clone_13
#   logit_bias_3 => full_default_14, where_3
#   q_f_1 => clone_10
#   v_f_1 => clone_12
#   x_soft_2 => _scaled_dot_product_efficient_attention_1
# Graph fragment:
#   %clone_10 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_37,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_38,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_12 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_39,), kwargs = {memory_format: torch.contiguous_format})
#   %eq_3 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%expand_13, 0), kwargs = {})
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clone_13 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_11,), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_3, %full_default_14, %clone_13), kwargs = {})
#   %_scaled_dot_product_efficient_attention_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_10, %clone_11, %clone_12, %expand_14, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_32', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16) % 128
    x2 = (xindex // 2048) % 8
    x3 = (xindex // 16384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*x2) + (384*x1) + (49152*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/sr/csrmynanaonoibfjjzvzoiovn3lnl7ns5ig2ytsfsy66x526taav.py
# Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, eq_3, logit_bias_3, logit_bias_2, x_soft_2], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   eq_3 => eq_3
#   k_f_1 => clone_11
#   logit_bias_2 => clone_13
#   logit_bias_3 => full_default_14, where_3
#   q_f_1 => clone_10
#   v_f_1 => clone_12
#   x_soft_2 => _scaled_dot_product_efficient_attention_1
# Graph fragment:
#   %clone_10 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_37,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_38,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_12 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_39,), kwargs = {memory_format: torch.contiguous_format})
#   %eq_3 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%expand_13, 0), kwargs = {})
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clone_13 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_11,), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_3, %full_default_14, %clone_13), kwargs = {})
#   %_scaled_dot_product_efficient_attention_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_10, %clone_11, %clone_12, %expand_14, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_33', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16) % 128
    x2 = (xindex // 2048) % 8
    x3 = (xindex // 16384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0 + (16*x2) + (384*x1) + (49152*x3)), None)
    tmp1 = tl.load(in_ptr1 + (128 + x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/qq/cqqnzinu3oqcibjvlysn7fy33udhdpjresh4zxcqnngqj4thloph.py
# Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, eq_3, logit_bias_3, logit_bias_2, x_soft_2, einsum_1], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   einsum_1 => clone_15
#   eq_3 => eq_3
#   k_f_1 => clone_11
#   logit_bias_2 => clone_13
#   logit_bias_3 => full_default_14, where_3
#   q_f_1 => clone_10
#   v_f_1 => clone_12
#   x_soft_2 => _scaled_dot_product_efficient_attention_1
# Graph fragment:
#   %clone_10 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_37,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_38,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_12 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_39,), kwargs = {memory_format: torch.contiguous_format})
#   %eq_3 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%expand_13, 0), kwargs = {})
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clone_13 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_11,), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_3, %full_default_14, %clone_13), kwargs = {})
#   %_scaled_dot_product_efficient_attention_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_10, %clone_11, %clone_12, %expand_14, False), kwargs = {})
#   %clone_15 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_44,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_34', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16) % 128
    x2 = (xindex // 2048) % 8
    x3 = (xindex // 16384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + (16*x2) + (384*x1) + (49152*x3)), None)
    tmp1 = tl.load(in_ptr1 + (256 + x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
    tl.store(out_ptr1 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/pn/cpnmmwts34oonkvqcwlexwm46u4zlovdpx5qrqeoxlxkanf7ywjb.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %select_scatter_default_1 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%index_put_8, %index_put_9, 4, 3), kwargs = {})
triton_poi_fused_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_35', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 4
    x1 = (xindex // 4)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2), None)
    tmp0 = x0
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/qo/cqovkmrktkdimqiteialdglab5y6w6zzzgfn3477krcdtza4fviv.py
# Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_17 => add_21, erf_5, mul_26, mul_27, mul_28
# Graph fragment:
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_60, 0.5), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_60, 0.7071067811865476), kwargs = {})
#   %erf_5 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_27,), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_5, 1), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, %add_21), kwargs = {})
triton_poi_fused_gelu_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_36', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 16
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/6r/c6rsq2awxx4jauwuiaxvlcmfmlkq3x2bldgly3s5bxh44ljriapp.py
# Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, eq_3, logit_bias_3, logit_bias_2, x_soft_2, einsum_1], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   einsum_1 => clone_14
#   eq_3 => eq_3
#   k_f_1 => clone_11
#   logit_bias_2 => clone_13
#   logit_bias_3 => full_default_14, where_3
#   q_f_1 => clone_10
#   v_f_1 => clone_12
#   x_soft_2 => _scaled_dot_product_efficient_attention_1
# Graph fragment:
#   %clone_10 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_37,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_38,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_12 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_39,), kwargs = {memory_format: torch.contiguous_format})
#   %eq_3 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%expand_13, 0), kwargs = {})
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clone_13 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_11,), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_3, %full_default_14, %clone_13), kwargs = {})
#   %_scaled_dot_product_efficient_attention_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_10, %clone_11, %clone_12, %expand_14, False), kwargs = {})
#   %clone_14 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_43,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[64, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_37', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 48
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y1 = (yindex // 8)
    y3 = yindex
    y0 = yindex % 8
    tmp0 = tl.load(in_ptr0 + (x2 + (16384*y1)), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2 + (16384*y1)), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (3 + (4*x2) + (65536*y1)), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0 + (8*x2) + (131072*y1)), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1, 1], 3, tl.int32)
    tmp4 = tmp3 == tmp3
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = float("-inf")
    tmp9 = tl.where(tmp2, tmp8, tmp7)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.where(tmp2, tmp1, tmp12)
    tl.store(out_ptr0 + (x2 + (16384*y3)), tmp9, ymask)
    tl.store(out_ptr1 + (x2 + (16384*y3)), tmp13, ymask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/eu/ceui7zu7qwrf7stzgvbk6d76ferggh2jdtdo266s537v6lxon76i.py
# Topologically Sorted Source Nodes: [x_mid_2, mul_3, x_mid_3], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#   mul_3 => mul_29
#   x_mid_2 => add_22
#   x_mid_3 => add_23
# Graph fragment:
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_70, %view_74), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_21, %view_77), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_22, %mul_29), kwargs = {})
triton_poi_fused_add_mul_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_38', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x0 = xindex % 16
    x1 = (xindex // 16) % 8
    x2 = (xindex // 128) % 128
    x3 = (xindex // 16384)
    x5 = (xindex // 16)
    x6 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (16*x2) + (2048*x1) + (16384*x3)), None)
    tmp3 = tl.load(in_ptr1 + (x5), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (256 + x6 + (384*x3)), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (256 + x6), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = tmp2 + tmp10
    tl.store(in_out_ptr0 + (x4), tmp11, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/lz/clz6uooqliiblbwtme7k4cjvn4cvjjxifb56ja5lv265cwhglof3.py
# Topologically Sorted Source Nodes: [x_2, layer_norm_4], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_4 => add_25, add_26, mul_30, mul_31, rsqrt_4, sub_4, var_mean_4
#   x_2 => add_24
# Graph fragment:
#   %add_24 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_51, %view_83), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_24, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, %getitem_17), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_25,), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, %arg66_1), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %arg67_1), kwargs = {})
triton_per_fused_add_native_layer_norm_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_39', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
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
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (r2 + (128*x3)), xmask, other=0.0)
    tmp7 = tl.load(in_out_ptr0 + (r2 + (128*x3)), xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 / tmp2
    tmp5 = tmp4 / tmp2
    tmp6 = tmp3 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = tmp10 - tmp20
    tmp28 = 128.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-05
    tmp31 = tmp29 + tmp30
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tl.store(in_out_ptr0 + (r2 + (128*x3)), tmp10, xmask)
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp37, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/wi/cwivespjmm3i4xrwzkbl37icqahp2cmaaxqxy7dyni2j6tk4pgjn.py
# Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_4 => add_34
#   x_5 => add_38
# Graph fragment:
#   %add_34 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %view_119), kwargs = {})
#   %add_38 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_34, %view_123), kwargs = {})
triton_poi_fused_add_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_40', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/pv/cpvtcsczfllyttvtaztv4xv4xw32py2uuo2feuubljxohhu3rkvy.py
# Topologically Sorted Source Nodes: [row_sum_2, getitem_33, index_add__6, col_sum_2, getitem_34, index_add__7], Original ATen: [aten.zeros, aten.index, aten.index_add]
# Source node to ATen node mapping:
#   col_sum_2 => full_default_20
#   getitem_33 => index_6
#   getitem_34 => index_7
#   index_add__6 => index_put_12
#   index_add__7 => index_put_13
#   row_sum_2 => full_default_19
# Graph fragment:
#   %full_default_19 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([6, 1, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_6 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%permute_68, [%arg32_1]), kwargs = {})
#   %index_put_12 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_19, [%arg34_1], %index_6, True), kwargs = {})
#   %full_default_20 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([6, 1, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_7 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%permute_68, [%arg33_1]), kwargs = {})
#   %index_put_13 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_20, [%arg34_1], %index_7, True), kwargs = {})
triton_poi_fused_index_index_add_zeros_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_index_add_zeros_41', 'mutated_arg_names': ['out_ptr0', 'out_ptr1'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 16384)
    x0 = xindex % 16384
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 6, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 6), "index out of bounds: 0 <= tmp4 < 6")
    tmp7 = tl.full([XBLOCK], 4, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert((0 <= tmp10) & (tmp10 < 4), "index out of bounds: 0 <= tmp10 < 4")
    tmp12 = tl.load(in_ptr2 + (x0 + (16384*tmp10)), None)
    tmp14 = tmp13 + tmp7
    tmp15 = tmp13 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp13)
    tl.device_assert((0 <= tmp16) & (tmp16 < 4), "index out of bounds: 0 <= tmp16 < 4")
    tmp18 = tl.load(in_ptr2 + (x0 + (16384*tmp16)), None)
    tl.atomic_add(out_ptr0 + (x0 + (16384*tmp4)), tmp12, None, sem='relaxed')
    tl.atomic_add(out_ptr1 + (x0 + (16384*tmp4)), tmp18, None, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/6c/c6cwai3wwt6mmggzhqwmyo4lww522gx3wwlgg6dz2kfjska7lrmr.py
# Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_20 => add_27, erf_6, mul_32, mul_33, mul_34
# Graph fragment:
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_85, 0.5), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_85, 0.7071067811865476), kwargs = {})
#   %erf_6 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_33,), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_6, 1), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %add_27), kwargs = {})
triton_poi_fused_gelu_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_42', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/g5/cg5ivhfj2kkyy4fqy525afu4phfad5oyp4ept2x4j6ciyw7rcqfq.py
# Topologically Sorted Source Nodes: [x_3, off_in, layer_norm_7], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_7 => add_41, add_42, mul_46, mul_47, rsqrt_7, sub_7, var_mean_7
#   off_in => add_40
#   x_3 => add_28
# Graph fragment:
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %view_87), kwargs = {})
#   %add_40 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_126, %add_28), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_40, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_40, %getitem_27), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-05), kwargs = {})
#   %rsqrt_7 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_41,), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_7), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %arg90_1), kwargs = {})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %arg91_1), kwargs = {})
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_43', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
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
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (r2 + (128*x3)), xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r2 + (128*x3)), xmask, other=0.0)
    tmp8 = tl.load(in_out_ptr0 + (r2 + (128*x3)), xmask, other=0.0)
    tmp9 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr6 + (r2), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 / tmp2
    tmp5 = tmp4 / tmp2
    tmp6 = tmp3 + tmp5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp18 = tl.where(xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.where(xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None]
    tmp29 = tmp12 - tmp22
    tmp30 = 128.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tl.store(in_out_ptr0 + (r2 + (128*x3)), tmp12, xmask)
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp39, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/v6/cv6zwjsdqlnuai7jncpcaol7da3pvef4amtvdi7g7mspr7rgzfb3.py
# Topologically Sorted Source Nodes: [x_6, layer_norm_8], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_8 => add_47, add_48, mul_52, mul_53, rsqrt_8, sub_8, var_mean_8
#   x_6 => add_46
# Graph fragment:
#   %add_46 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_40, %view_158), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_46, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_46, %getitem_33), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_32, 1e-05), kwargs = {})
#   %rsqrt_8 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_47,), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_8), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %arg102_1), kwargs = {})
#   %add_48 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %arg103_1), kwargs = {})
triton_per_fused_add_native_layer_norm_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_44', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/x2/cx2d64utka46leb2jizsq6352zexl7t27cfyxu3gh55vtidjrdn3.py
# Topologically Sorted Source Nodes: [x_6, x_7], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_6 => add_46
#   x_7 => add_50
# Graph fragment:
#   %add_46 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_40, %view_158), kwargs = {})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_46, %view_162), kwargs = {})
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
    xnumel = 98304
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/g4/cg42ztmbi3pap5gcwg4wzow22v2xw2s6gwcw46vzsnx67k2r7hfk.py
# Topologically Sorted Source Nodes: [sigmoid_4], Original ATen: [aten.sigmoid]
# Source node to ATen node mapping:
#   sigmoid_4 => sigmoid_4
# Graph fragment:
#   %sigmoid_4 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_182,), kwargs = {})
triton_poi_fused_sigmoid_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_46', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/f6/cf6gkf66v6mxfnuxo256kusslhb5hfa34q67u6bnerbcf5hduaqk.py
# Topologically Sorted Source Nodes: [packed_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   packed_1 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_3, %view_183], 1), kwargs = {})
triton_poi_fused_cat_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_47', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 295424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 294912, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 65536, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.load(in_ptr0 + (x0), tmp7 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp0 >= tmp5
    tmp10 = tl.full([1], 163840, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tmp12 & tmp4
    tmp14 = tl.load(in_ptr1 + ((-65536) + x0), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp0 >= tmp10
    tmp16 = tl.full([1], 229376, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr2 + ((-163840) + x0), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp0 >= tmp16
    tmp22 = tmp21 & tmp4
    tmp23 = tl.load(in_ptr3 + ((-229376) + x0), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp18, tmp20, tmp23)
    tmp25 = tl.where(tmp12, tmp14, tmp24)
    tmp26 = tl.where(tmp6, tmp8, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp4, tmp26, tmp27)
    tmp29 = tmp0 >= tmp3
    tmp30 = tl.full([1], 295424, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = tl.load(in_ptr4 + ((-294912) + x0), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.where(tmp4, tmp28, tmp32)
    tl.store(out_ptr0 + (x0), tmp33, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1 = args
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
    assert_size_stride(arg28_1, (4, 128, 129, 4), (66048, 516, 4, 1))
    assert_size_stride(arg29_1, (4, 128, 129), (16512, 129, 1))
    assert_size_stride(arg30_1, (6, 128, 129, 4), (66048, 516, 4, 1))
    assert_size_stride(arg31_1, (6, 128, 129), (16512, 129, 1))
    assert_size_stride(arg32_1, (6, ), (1, ))
    assert_size_stride(arg33_1, (6, ), (1, ))
    assert_size_stride(arg34_1, (6, ), (1, ))
    assert_size_stride(arg35_1, (6, ), (1, ))
    assert_size_stride(arg36_1, (128, ), (1, ))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (384, 128), (128, 1))
    assert_size_stride(arg39_1, (384, ), (1, ))
    assert_size_stride(arg40_1, (16, 4), (4, 1))
    assert_size_stride(arg41_1, (16, ), (1, ))
    assert_size_stride(arg42_1, (8, 16), (16, 1))
    assert_size_stride(arg43_1, (8, ), (1, ))
    assert_size_stride(arg44_1, (8, 128), (128, 1))
    assert_size_stride(arg45_1, (8, ), (1, ))
    assert_size_stride(arg46_1, (128, 128), (128, 1))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (128, ), (1, ))
    assert_size_stride(arg49_1, (128, ), (1, ))
    assert_size_stride(arg50_1, (512, 128), (128, 1))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (128, 512), (512, 1))
    assert_size_stride(arg53_1, (128, ), (1, ))
    assert_size_stride(arg54_1, (128, ), (1, ))
    assert_size_stride(arg55_1, (128, ), (1, ))
    assert_size_stride(arg56_1, (384, 128), (128, 1))
    assert_size_stride(arg57_1, (384, ), (1, ))
    assert_size_stride(arg58_1, (16, 4), (4, 1))
    assert_size_stride(arg59_1, (16, ), (1, ))
    assert_size_stride(arg60_1, (8, 16), (16, 1))
    assert_size_stride(arg61_1, (8, ), (1, ))
    assert_size_stride(arg62_1, (8, 128), (128, 1))
    assert_size_stride(arg63_1, (8, ), (1, ))
    assert_size_stride(arg64_1, (128, 128), (128, 1))
    assert_size_stride(arg65_1, (128, ), (1, ))
    assert_size_stride(arg66_1, (128, ), (1, ))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (512, 128), (128, 1))
    assert_size_stride(arg69_1, (512, ), (1, ))
    assert_size_stride(arg70_1, (128, 512), (512, 1))
    assert_size_stride(arg71_1, (128, ), (1, ))
    assert_size_stride(arg72_1, (128, ), (1, ))
    assert_size_stride(arg73_1, (128, ), (1, ))
    assert_size_stride(arg74_1, (384, 128), (128, 1))
    assert_size_stride(arg75_1, (384, ), (1, ))
    assert_size_stride(arg76_1, (16, 4), (4, 1))
    assert_size_stride(arg77_1, (16, ), (1, ))
    assert_size_stride(arg78_1, (8, 16), (16, 1))
    assert_size_stride(arg79_1, (8, ), (1, ))
    assert_size_stride(arg80_1, (8, 128), (128, 1))
    assert_size_stride(arg81_1, (8, ), (1, ))
    assert_size_stride(arg82_1, (128, 128), (128, 1))
    assert_size_stride(arg83_1, (128, ), (1, ))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (128, ), (1, ))
    assert_size_stride(arg86_1, (512, 128), (128, 1))
    assert_size_stride(arg87_1, (512, ), (1, ))
    assert_size_stride(arg88_1, (128, 512), (512, 1))
    assert_size_stride(arg89_1, (128, ), (1, ))
    assert_size_stride(arg90_1, (128, ), (1, ))
    assert_size_stride(arg91_1, (128, ), (1, ))
    assert_size_stride(arg92_1, (384, 128), (128, 1))
    assert_size_stride(arg93_1, (384, ), (1, ))
    assert_size_stride(arg94_1, (16, 4), (4, 1))
    assert_size_stride(arg95_1, (16, ), (1, ))
    assert_size_stride(arg96_1, (8, 16), (16, 1))
    assert_size_stride(arg97_1, (8, ), (1, ))
    assert_size_stride(arg98_1, (8, 128), (128, 1))
    assert_size_stride(arg99_1, (8, ), (1, ))
    assert_size_stride(arg100_1, (128, 128), (128, 1))
    assert_size_stride(arg101_1, (128, ), (1, ))
    assert_size_stride(arg102_1, (128, ), (1, ))
    assert_size_stride(arg103_1, (128, ), (1, ))
    assert_size_stride(arg104_1, (512, 128), (128, 1))
    assert_size_stride(arg105_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (128, 512), (512, 1))
    assert_size_stride(arg107_1, (128, ), (1, ))
    assert_size_stride(arg108_1, (128, 128), (128, 1))
    assert_size_stride(arg109_1, (128, ), (1, ))
    assert_size_stride(arg110_1, (128, 128), (128, 1))
    assert_size_stride(arg111_1, (128, ), (1, ))
    assert_size_stride(arg112_1, (128, 128), (128, 1))
    assert_size_stride(arg113_1, (128, ), (1, ))
    assert_size_stride(arg114_1, (1, 128), (128, 1))
    assert_size_stride(arg115_1, (1, ), (1, ))
    assert_size_stride(arg116_1, (128, 128), (128, 1))
    assert_size_stride(arg117_1, (128, ), (1, ))
    assert_size_stride(arg118_1, (128, 128), (128, 1))
    assert_size_stride(arg119_1, (128, ), (1, ))
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
        buf29 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_6.run(buf25, arg36_1, arg37_1, buf29, 512, 128, grid=grid(512), stream=stream0)
        del arg36_1
        del arg37_1
        buf30 = empty_strided_cuda((512, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (512, 128), (128, 1), 0), reinterpret_tensor(arg38_1, (128, 384), (1, 128), 0), out=buf30)
        buf31 = empty_strided_cuda((1, 4, 128), (512, 128, 1), torch.float32)
        buf32 = reinterpret_tensor(buf12, (1, 4, 128, 128), (65536, 16384, 128, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [mask_base_1, sum_1], Original ATen: [aten.clone, aten.sum]
        triton_per_fused_clone_sum_7.run(arg29_1, buf31, buf32, 512, 128, grid=grid(512), stream=stream0)
        buf33 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [lt, _need_diag, maximum, copy_], Original ATen: [aten.lt, aten._to_copy, aten.maximum, aten.copy]
        triton_poi_fused__to_copy_copy_lt_maximum_8.run(buf33, arg29_1, buf32, 512, grid=grid(512), stream=stream0)
        buf35 = empty_strided_cuda((1, 4, 128, 128, 4), (262144, 65536, 512, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bias_physics], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(arg28_1, buf35, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [bias_physics, setitem], Original ATen: [aten.clone, aten.lift_fresh, aten.index_put]
        triton_poi_fused_clone_index_put_lift_fresh_10.run(buf35, 2048, grid=grid(2048), stream=stream0)
        buf37 = empty_strided_cuda((1, 4, 128, 128), (65536, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_11.run(buf35, buf37, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_12.run(buf37, 512, grid=grid(512), stream=stream0)
        buf39 = empty_strided_cuda((4, 8, 128, 16), (16384, 2048, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f, k_f, v_f, eq_1, logit_bias_1, logit_bias, x_soft], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_13.run(buf30, arg39_1, buf39, 65536, grid=grid(65536), stream=stream0)
        buf40 = empty_strided_cuda((4, 8, 128, 16), (16384, 2048, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f, k_f, v_f, eq_1, logit_bias_1, logit_bias, x_soft], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_14.run(buf30, arg39_1, buf40, 65536, grid=grid(65536), stream=stream0)
        buf41 = empty_strided_cuda((4, 8, 128, 16), (16384, 2048, 16, 1), torch.float32)
        buf53 = empty_strided_cuda((4, 8, 128, 1, 16, 1), (16384, 2048, 16, 16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f, k_f, v_f, eq_1, logit_bias_1, logit_bias, x_soft, einsum], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_15.run(buf30, arg39_1, buf41, buf53, 65536, grid=grid(65536), stream=stream0)
        buf48 = empty_strided_cuda((1, 4, 128, 128, 4), (262144, 65536, 512, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(buf37, buf35, buf48, 262144, grid=grid(262144), stream=stream0)
        buf49 = empty_strided_cuda((65536, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf48, (65536, 4), (4, 1), 0), reinterpret_tensor(arg40_1, (4, 16), (1, 4), 0), out=buf49)
        del arg40_1
        buf50 = reinterpret_tensor(buf49, (1, 4, 128, 128, 16), (1048576, 262144, 2048, 16, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_17.run(buf50, arg41_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg41_1
        buf51 = empty_strided_cuda((65536, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf50, (65536, 16), (16, 1), 0), reinterpret_tensor(arg42_1, (16, 8), (1, 16), 0), out=buf51)
        del arg42_1
        buf42 = empty_strided_cuda((4, 8, 128, 128), (131072, 16384, 128, 1), torch.float32)
        buf52 = empty_strided_cuda((4, 8, 128, 128, 1, 1), (131072, 16384, 128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f, k_f, v_f, eq_1, logit_bias_1, logit_bias, x_soft, einsum], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_18.run(buf32, buf37, buf35, buf51, arg43_1, buf42, buf52, 32, 16384, grid=grid(32, 16384), stream=stream0)
        del arg43_1
        del buf32
        # Topologically Sorted Source Nodes: [q_f, k_f, v_f, eq_1, logit_bias_1, logit_bias, x_soft], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        buf43 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf39, buf40, buf41, buf42, False)
        buf44 = buf43[0]
        del buf43
        buf54 = reinterpret_tensor(buf41, (32, 128, 16), (2048, 16, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf52, (32, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf53, (32, 128, 16), (2048, 16, 1), 0), out=buf54)
        buf55 = empty_strided_cuda((512, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (512, 128), (128, 1), 0), reinterpret_tensor(arg44_1, (128, 8), (1, 128), 0), out=buf55)
        del arg44_1
        buf56 = reinterpret_tensor(buf33, (1, 4, 1, 128), (512, 128, 512, 1), 0); del buf33  # reuse
        buf57 = reinterpret_tensor(buf56, (1, 4, 1, 128), (512, 128, 128, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [diag_prep_block], Original ATen: [aten.mean]
        triton_red_fused_mean_19.run(buf57, buf25, 512, 128, grid=grid(512), stream=stream0)
        buf58 = empty_strided_cuda((4, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (4, 128), (128, 1), 0), reinterpret_tensor(arg38_1, (128, 384), (1, 128), 0), out=buf58)
        del arg38_1
        buf59 = reinterpret_tensor(buf44, (1, 4, 128, 8, 16), (65536, 16384, 128, 16, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [x_mid, mul_2, x_mid_1], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_20.run(buf59, buf54, buf55, arg45_1, buf58, arg39_1, 65536, grid=grid(65536), stream=stream0)
        del arg39_1
        del arg45_1
        buf60 = reinterpret_tensor(buf54, (512, 128), (128, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf59, (512, 128), (128, 1), 0), reinterpret_tensor(arg46_1, (128, 128), (1, 128), 0), out=buf60)
        del arg46_1
        buf64 = reinterpret_tensor(buf59, (1, 512, 128), (65536, 128, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [x, layer_norm_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_5.run(buf25, buf60, arg47_1, arg48_1, arg49_1, buf64, 512, 128, grid=grid(512), stream=stream0)
        del arg48_1
        del arg49_1
        buf65 = reinterpret_tensor(buf35, (512, 512), (512, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf64, (512, 128), (128, 1), 0), reinterpret_tensor(arg50_1, (128, 512), (1, 128), 0), out=buf65)
        del arg50_1
        buf66 = reinterpret_tensor(buf65, (1, 512, 512), (262144, 512, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf66, arg51_1, 262144, grid=grid(262144), stream=stream0)
        del arg51_1
        buf67 = reinterpret_tensor(buf64, (512, 128), (128, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf66, (512, 512), (512, 1), 0), reinterpret_tensor(arg52_1, (512, 128), (1, 512), 0), out=buf67)
        del arg52_1
        buf68 = reinterpret_tensor(buf67, (1, 512, 128), (65536, 128, 1), 0); del buf67  # reuse
        buf118 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [x, x_1, layer_norm_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf68, buf25, buf60, arg47_1, arg53_1, arg72_1, arg73_1, buf118, 512, 128, grid=grid(512), stream=stream0)
        del arg47_1
        del arg53_1
        del arg72_1
        del arg73_1
        buf69 = empty_strided_cuda((6, 1, 128, 128), (16384, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [row_sum_1], Original ATen: [aten.zeros]
        triton_poi_fused_zeros_23.run(buf69, 98304, grid=grid(98304), stream=stream0)
        buf102 = empty_strided_cuda((6, 1, 128, 128), (16384, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [row_sum], Original ATen: [aten.zeros]
        triton_poi_fused_zeros_23.run(buf102, 98304, grid=grid(98304), stream=stream0)
        buf104 = empty_strided_cuda((6, 1, 128, 128), (16384, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [col_sum], Original ATen: [aten.zeros]
        triton_poi_fused_zeros_23.run(buf104, 98304, grid=grid(98304), stream=stream0)
        buf71 = empty_strided_cuda((6, 1, 128, 128), (16384, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [col_sum_1], Original ATen: [aten.zeros]
        triton_poi_fused_zeros_23.run(buf71, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [row_sum_1, getitem_17, index_add__4, col_sum_1, getitem_18, index_add__5, row_sum, getitem_8, index_add__2, col_sum, getitem_9, index_add__3], Original ATen: [aten.zeros, aten.index, aten.index_add]
        triton_poi_fused_index_index_add_zeros_24.run(arg34_1, arg32_1, buf68, arg33_1, buf25, buf69, buf71, buf102, buf104, 98304, grid=grid(98304), stream=stream0)
        buf106 = empty_strided_cuda((6, 1, 1, 128), (128, 768, 768, 1), torch.float32)
        buf76 = empty_strided_cuda((6, 128, 128), (16384, 128, 1), torch.float32)
        buf107 = reinterpret_tensor(buf106, (6, 1, 1, 128), (128, 1, 768, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_3, off_prep_block], Original ATen: [aten.native_layer_norm, aten.mean]
        triton_red_fused_mean_native_layer_norm_25.run(buf107, buf69, arg35_1, buf71, buf102, buf104, arg54_1, arg55_1, buf76, 768, 128, grid=grid(768), stream=stream0)
        del arg54_1
        del arg55_1
        buf77 = empty_strided_cuda((768, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (768, 128), (128, 1), 0), reinterpret_tensor(arg56_1, (128, 384), (1, 128), 0), out=buf77)
        buf79 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [mask_base_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf79, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [mask_base_2, sum_2, lt_1, _need_diag_1, maximum_1, copy__1], Original ATen: [aten.clone, aten.sum, aten.lt, aten._to_copy, aten.maximum, aten.copy]
        triton_per_fused__to_copy_clone_copy_lt_maximum_sum_27.run(buf79, 768, 128, grid=grid(768), stream=stream0)
        buf81 = empty_strided_cuda((6, 1, 128, 128, 4), (65536, 393216, 512, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bias_physics_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(arg30_1, buf81, 393216, grid=grid(393216), stream=stream0)
        # Topologically Sorted Source Nodes: [bias_physics_1, setitem_2], Original ATen: [aten.clone, aten.lift_fresh, aten.index_put]
        triton_poi_fused_clone_index_put_lift_fresh_29.run(buf81, 3072, grid=grid(3072), stream=stream0)
        buf83 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_30.run(buf81, buf83, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_31.run(buf83, 768, grid=grid(768), stream=stream0)
        buf85 = empty_strided_cuda((6, 8, 128, 16), (16384, 2048, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, eq_3, logit_bias_3, logit_bias_2, x_soft_2], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_32.run(buf77, arg57_1, buf85, 98304, grid=grid(98304), stream=stream0)
        buf86 = empty_strided_cuda((6, 8, 128, 16), (16384, 2048, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, eq_3, logit_bias_3, logit_bias_2, x_soft_2], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_33.run(buf77, arg57_1, buf86, 98304, grid=grid(98304), stream=stream0)
        buf87 = empty_strided_cuda((6, 8, 128, 16), (16384, 2048, 16, 1), torch.float32)
        buf99 = empty_strided_cuda((6, 8, 128, 1, 16, 1), (16384, 2048, 16, 16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, eq_3, logit_bias_3, logit_bias_2, x_soft_2, einsum_1], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_34.run(buf77, arg57_1, buf87, buf99, 98304, grid=grid(98304), stream=stream0)
        buf94 = empty_strided_cuda((6, 1, 128, 128, 4), (65536, 1, 512, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_35.run(buf83, buf81, buf94, 393216, grid=grid(393216), stream=stream0)
        buf95 = empty_strided_cuda((98304, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (98304, 4), (4, 1), 0), reinterpret_tensor(arg58_1, (4, 16), (1, 4), 0), out=buf95)
        del arg58_1
        buf96 = reinterpret_tensor(buf95, (6, 1, 128, 128, 16), (262144, 1, 2048, 16, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_36.run(buf96, arg59_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg59_1
        buf97 = empty_strided_cuda((98304, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf96, (98304, 16), (16, 1), 0), reinterpret_tensor(arg60_1, (16, 8), (1, 16), 0), out=buf97)
        del arg60_1
        buf88 = empty_strided_cuda((6, 8, 128, 128), (131072, 16384, 128, 1), torch.float32)
        buf98 = empty_strided_cuda((6, 8, 128, 128, 1, 1), (131072, 16384, 128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, eq_3, logit_bias_3, logit_bias_2, x_soft_2, einsum_1], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_37.run(buf79, buf83, buf81, buf97, arg61_1, buf88, buf98, 48, 16384, grid=grid(48, 16384), stream=stream0)
        del arg61_1
        del buf79
        del buf83
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, eq_3, logit_bias_3, logit_bias_2, x_soft_2], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        buf89 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf85, buf86, buf87, buf88, False)
        buf90 = buf89[0]
        del buf89
        buf100 = reinterpret_tensor(buf87, (48, 128, 16), (2048, 16, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [einsum_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf98, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf99, (48, 128, 16), (2048, 16, 1), 0), out=buf100)
        buf101 = empty_strided_cuda((768, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (768, 128), (128, 1), 0), reinterpret_tensor(arg62_1, (128, 8), (1, 128), 0), out=buf101)
        del arg62_1
        buf108 = empty_strided_cuda((6, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf107, (6, 128), (128, 1), 0), reinterpret_tensor(arg56_1, (128, 384), (1, 128), 0), out=buf108)
        del arg56_1
        buf109 = reinterpret_tensor(buf90, (6, 1, 128, 8, 16), (16384, 1, 128, 16, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [x_mid_2, mul_3, x_mid_3], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_38.run(buf109, buf100, buf101, arg63_1, buf108, arg57_1, 98304, grid=grid(98304), stream=stream0)
        del arg57_1
        del arg63_1
        buf110 = reinterpret_tensor(buf100, (768, 128), (128, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf109, (768, 128), (128, 1), 0), reinterpret_tensor(arg64_1, (128, 128), (1, 128), 0), out=buf110)
        del arg64_1
        buf111 = reinterpret_tensor(buf110, (6, 128, 128), (16384, 128, 1), 0); del buf110  # reuse
        buf160 = reinterpret_tensor(buf109, (6, 128, 128), (16384, 128, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_2, layer_norm_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_39.run(buf111, buf69, arg35_1, buf71, arg65_1, arg66_1, arg67_1, buf160, 768, 128, grid=grid(768), stream=stream0)
        del arg65_1
        del arg66_1
        del arg67_1
        buf119 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (512, 128), (128, 1), 0), reinterpret_tensor(arg74_1, (128, 384), (1, 128), 0), out=buf119)
        buf120 = empty_strided_cuda((1, 4, 128), (512, 128, 1), torch.float32)
        buf121 = reinterpret_tensor(buf25, (1, 4, 128, 128), (65536, 16384, 128, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [mask_base_4, sum_3], Original ATen: [aten.clone, aten.sum]
        triton_per_fused_clone_sum_7.run(arg29_1, buf120, buf121, 512, 128, grid=grid(512), stream=stream0)
        buf122 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [lt_2, _need_diag_2, maximum_2, copy__2], Original ATen: [aten.lt, aten._to_copy, aten.maximum, aten.copy]
        triton_poi_fused__to_copy_copy_lt_maximum_8.run(buf122, arg29_1, buf121, 512, grid=grid(512), stream=stream0)
        del arg29_1
        del buf122
        buf124 = reinterpret_tensor(buf66, (1, 4, 128, 128, 4), (262144, 65536, 512, 4, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [bias_physics_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(arg28_1, buf124, 262144, grid=grid(262144), stream=stream0)
        del arg28_1
        # Topologically Sorted Source Nodes: [bias_physics_2, setitem_4], Original ATen: [aten.clone, aten.lift_fresh, aten.index_put]
        triton_poi_fused_clone_index_put_lift_fresh_10.run(buf124, 2048, grid=grid(2048), stream=stream0)
        buf126 = reinterpret_tensor(buf60, (1, 4, 128, 128), (65536, 16384, 128, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [setitem_5], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_11.run(buf124, buf126, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_5], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_12.run(buf126, 512, grid=grid(512), stream=stream0)
        buf128 = reinterpret_tensor(buf53, (4, 8, 128, 16), (16384, 2048, 16, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [q_f_2, k_f_2, v_f_2, eq_5, logit_bias_5, logit_bias_4, x_soft_4], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_13.run(buf119, arg75_1, buf128, 65536, grid=grid(65536), stream=stream0)
        buf129 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [q_f_2, k_f_2, v_f_2, eq_5, logit_bias_5, logit_bias_4, x_soft_4], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_14.run(buf119, arg75_1, buf129, 65536, grid=grid(65536), stream=stream0)
        buf130 = buf39; del buf39  # reuse
        buf142 = reinterpret_tensor(buf37, (4, 8, 128, 1, 16, 1), (16384, 2048, 16, 16, 1, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [q_f_2, k_f_2, v_f_2, eq_5, logit_bias_5, logit_bias_4, x_soft_4, einsum_2], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_15.run(buf119, arg75_1, buf130, buf142, 65536, grid=grid(65536), stream=stream0)
        del buf119
        buf137 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(buf126, buf124, buf137, 262144, grid=grid(262144), stream=stream0)
        buf138 = reinterpret_tensor(buf50, (65536, 16), (16, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf137, (65536, 4), (4, 1), 0), reinterpret_tensor(arg76_1, (4, 16), (1, 4), 0), out=buf138)
        del arg76_1
        del buf137
        buf139 = reinterpret_tensor(buf138, (1, 4, 128, 128, 16), (1048576, 262144, 2048, 16, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_17.run(buf139, arg77_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg77_1
        buf140 = reinterpret_tensor(buf52, (65536, 8), (8, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf139, (65536, 16), (16, 1), 0), reinterpret_tensor(arg78_1, (16, 8), (1, 16), 0), out=buf140)
        del arg78_1
        del buf139
        buf131 = buf42; del buf42  # reuse
        buf141 = reinterpret_tensor(buf51, (4, 8, 128, 128, 1, 1), (131072, 16384, 128, 1, 1, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [q_f_2, k_f_2, v_f_2, eq_5, logit_bias_5, logit_bias_4, x_soft_4, einsum_2], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_18.run(buf121, buf126, buf124, buf140, arg79_1, buf131, buf141, 32, 16384, grid=grid(32, 16384), stream=stream0)
        del arg79_1
        del buf121
        del buf126
        del buf140
        # Topologically Sorted Source Nodes: [q_f_2, k_f_2, v_f_2, eq_5, logit_bias_5, logit_bias_4, x_soft_4], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        buf132 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf128, buf129, buf130, buf131, False)
        del buf128
        del buf129
        del buf131
        buf133 = buf132[0]
        del buf132
        buf143 = reinterpret_tensor(buf130, (32, 128, 16), (2048, 16, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf141, (32, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf142, (32, 128, 16), (2048, 16, 1), 0), out=buf143)
        del buf141
        del buf142
        buf144 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (512, 128), (128, 1), 0), reinterpret_tensor(arg80_1, (128, 8), (1, 128), 0), out=buf144)
        del arg80_1
        buf145 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (4, 128), (128, 1), 0), reinterpret_tensor(arg74_1, (128, 384), (1, 128), 0), out=buf145)
        del arg74_1
        buf146 = reinterpret_tensor(buf133, (1, 4, 128, 8, 16), (65536, 16384, 128, 16, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [x_mid_4, mul_4, x_mid_5], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_20.run(buf146, buf143, buf144, arg81_1, buf145, arg75_1, 65536, grid=grid(65536), stream=stream0)
        del arg75_1
        del arg81_1
        del buf144
        del buf145
        buf147 = reinterpret_tensor(buf143, (512, 128), (128, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf146, (512, 128), (128, 1), 0), reinterpret_tensor(arg82_1, (128, 128), (1, 128), 0), out=buf147)
        del arg82_1
        buf151 = reinterpret_tensor(buf146, (1, 512, 128), (65536, 128, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [x_4, layer_norm_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_5.run(buf68, buf147, arg83_1, arg84_1, arg85_1, buf151, 512, 128, grid=grid(512), stream=stream0)
        del arg84_1
        del arg85_1
        buf152 = reinterpret_tensor(buf124, (512, 512), (512, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (512, 128), (128, 1), 0), reinterpret_tensor(arg86_1, (128, 512), (1, 128), 0), out=buf152)
        del arg86_1
        buf153 = reinterpret_tensor(buf152, (1, 512, 512), (262144, 512, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf153, arg87_1, 262144, grid=grid(262144), stream=stream0)
        del arg87_1
        buf154 = reinterpret_tensor(buf151, (512, 128), (128, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (512, 512), (512, 1), 0), reinterpret_tensor(arg88_1, (512, 128), (1, 512), 0), out=buf154)
        del arg88_1
        del buf153
        buf155 = reinterpret_tensor(buf154, (1, 512, 128), (65536, 128, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten.add]
        triton_poi_fused_add_40.run(buf155, buf68, buf147, arg83_1, arg89_1, 65536, grid=grid(65536), stream=stream0)
        del arg83_1
        del arg89_1
        buf156 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [row_sum_2], Original ATen: [aten.zeros]
        triton_poi_fused_zeros_23.run(buf156, 98304, grid=grid(98304), stream=stream0)
        buf158 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [col_sum_2], Original ATen: [aten.zeros]
        triton_poi_fused_zeros_23.run(buf158, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [row_sum_2, getitem_33, index_add__6, col_sum_2, getitem_34, index_add__7], Original ATen: [aten.zeros, aten.index, aten.index_add]
        triton_poi_fused_index_index_add_zeros_41.run(arg34_1, arg32_1, buf155, arg33_1, buf156, buf158, 98304, grid=grid(98304), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        buf161 = reinterpret_tensor(buf81, (768, 512), (512, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf160, (768, 128), (128, 1), 0), reinterpret_tensor(arg68_1, (128, 512), (1, 128), 0), out=buf161)
        del arg68_1
        buf162 = reinterpret_tensor(buf161, (6, 128, 512), (65536, 512, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_42.run(buf162, arg69_1, 393216, grid=grid(393216), stream=stream0)
        del arg69_1
        buf163 = reinterpret_tensor(buf160, (768, 128), (128, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf162, (768, 512), (512, 1), 0), reinterpret_tensor(arg70_1, (512, 128), (1, 512), 0), out=buf163)
        del arg70_1
        buf164 = reinterpret_tensor(buf163, (6, 128, 128), (16384, 128, 1), 0); del buf163  # reuse
        buf168 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [x_3, off_in, layer_norm_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_43.run(buf164, buf156, arg35_1, buf158, buf111, arg71_1, arg90_1, arg91_1, buf168, 768, 128, grid=grid(768), stream=stream0)
        del arg35_1
        del arg71_1
        del arg90_1
        del arg91_1
        buf169 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf168, (768, 128), (128, 1), 0), reinterpret_tensor(arg92_1, (128, 384), (1, 128), 0), out=buf169)
        buf171 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [mask_base_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf171, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [mask_base_5, sum_4, lt_3, _need_diag_3, maximum_3, copy__3], Original ATen: [aten.clone, aten.sum, aten.lt, aten._to_copy, aten.maximum, aten.copy]
        triton_per_fused__to_copy_clone_copy_lt_maximum_sum_27.run(buf171, 768, 128, grid=grid(768), stream=stream0)
        buf173 = reinterpret_tensor(buf162, (6, 1, 128, 128, 4), (65536, 393216, 512, 4, 1), 0); del buf162  # reuse
        # Topologically Sorted Source Nodes: [bias_physics_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(arg30_1, buf173, 393216, grid=grid(393216), stream=stream0)
        del arg30_1
        # Topologically Sorted Source Nodes: [bias_physics_3, setitem_6], Original ATen: [aten.clone, aten.lift_fresh, aten.index_put]
        triton_poi_fused_clone_index_put_lift_fresh_29.run(buf173, 3072, grid=grid(3072), stream=stream0)
        buf175 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [setitem_7], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_30.run(buf173, buf175, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_7], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_31.run(buf175, 768, grid=grid(768), stream=stream0)
        buf177 = reinterpret_tensor(buf111, (6, 8, 128, 16), (16384, 2048, 16, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [q_f_3, k_f_3, v_f_3, eq_7, logit_bias_7, logit_bias_6, x_soft_6], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_32.run(buf169, arg93_1, buf177, 98304, grid=grid(98304), stream=stream0)
        buf178 = reinterpret_tensor(buf99, (6, 8, 128, 16), (16384, 2048, 16, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [q_f_3, k_f_3, v_f_3, eq_7, logit_bias_7, logit_bias_6, x_soft_6], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_33.run(buf169, arg93_1, buf178, 98304, grid=grid(98304), stream=stream0)
        buf179 = buf86; del buf86  # reuse
        buf191 = reinterpret_tensor(buf85, (6, 8, 128, 1, 16, 1), (16384, 2048, 16, 16, 1, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [q_f_3, k_f_3, v_f_3, eq_7, logit_bias_7, logit_bias_6, x_soft_6, einsum_3], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_34.run(buf169, arg93_1, buf179, buf191, 98304, grid=grid(98304), stream=stream0)
        del buf169
        buf186 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_35.run(buf175, buf173, buf186, 393216, grid=grid(393216), stream=stream0)
        buf187 = reinterpret_tensor(buf96, (98304, 16), (16, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf186, (98304, 4), (4, 1), 0), reinterpret_tensor(arg94_1, (4, 16), (1, 4), 0), out=buf187)
        del arg94_1
        del buf186
        buf188 = reinterpret_tensor(buf187, (6, 1, 128, 128, 16), (262144, 1, 2048, 16, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_36.run(buf188, arg95_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg95_1
        buf189 = reinterpret_tensor(buf98, (98304, 8), (8, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf188, (98304, 16), (16, 1), 0), reinterpret_tensor(arg96_1, (16, 8), (1, 16), 0), out=buf189)
        del arg96_1
        del buf188
        buf180 = buf88; del buf88  # reuse
        buf190 = reinterpret_tensor(buf97, (6, 8, 128, 128, 1, 1), (131072, 16384, 128, 1, 1, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [q_f_3, k_f_3, v_f_3, eq_7, logit_bias_7, logit_bias_6, x_soft_6, einsum_3], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_37.run(buf171, buf175, buf173, buf189, arg97_1, buf180, buf190, 48, 16384, grid=grid(48, 16384), stream=stream0)
        del arg97_1
        del buf171
        del buf175
        del buf189
        # Topologically Sorted Source Nodes: [q_f_3, k_f_3, v_f_3, eq_7, logit_bias_7, logit_bias_6, x_soft_6], Original ATen: [aten.clone, aten.eq, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
        buf181 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf177, buf178, buf179, buf180, False)
        del buf177
        del buf178
        del buf180
        buf182 = buf181[0]
        del buf181
        buf192 = reinterpret_tensor(buf179, (48, 128, 16), (2048, 16, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [einsum_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf190, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf191, (48, 128, 16), (2048, 16, 1), 0), out=buf192)
        del buf190
        del buf191
        buf193 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf168, (768, 128), (128, 1), 0), reinterpret_tensor(arg98_1, (128, 8), (1, 128), 0), out=buf193)
        del arg98_1
        del buf168
        buf194 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf107, (6, 128), (128, 1), 0), reinterpret_tensor(arg92_1, (128, 384), (1, 128), 0), out=buf194)
        del arg92_1
        del buf107
        buf195 = reinterpret_tensor(buf182, (6, 1, 128, 8, 16), (16384, 1, 128, 16, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [x_mid_6, mul_5, x_mid_7], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_38.run(buf195, buf192, buf193, arg99_1, buf194, arg93_1, 98304, grid=grid(98304), stream=stream0)
        del arg93_1
        del arg99_1
        del buf193
        del buf194
        buf196 = reinterpret_tensor(buf192, (768, 128), (128, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf195, (768, 128), (128, 1), 0), reinterpret_tensor(arg100_1, (128, 128), (1, 128), 0), out=buf196)
        del arg100_1
        buf202 = reinterpret_tensor(buf195, (6, 128, 128), (16384, 128, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [x_6, layer_norm_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_44.run(buf164, buf196, arg101_1, arg102_1, arg103_1, buf202, 768, 128, grid=grid(768), stream=stream0)
        del arg102_1
        del arg103_1
        buf200 = reinterpret_tensor(buf68, (512, 128), (128, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [u_leaf], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg109_1, reinterpret_tensor(buf155, (512, 128), (128, 1), 0), reinterpret_tensor(arg108_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf200)
        del arg108_1
        del arg109_1
        buf201 = reinterpret_tensor(buf147, (4, 128, 128), (16384, 128, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf200, (4, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf200, (4, 128, 128), (16384, 1, 128), 0), out=buf201)
        buf203 = reinterpret_tensor(buf173, (768, 512), (512, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf202, (768, 128), (128, 1), 0), reinterpret_tensor(arg104_1, (128, 512), (1, 128), 0), out=buf203)
        del arg104_1
        buf204 = reinterpret_tensor(buf203, (6, 128, 512), (65536, 512, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_42.run(buf204, arg105_1, 393216, grid=grid(393216), stream=stream0)
        del arg105_1
        buf205 = reinterpret_tensor(buf202, (768, 128), (128, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf204, (768, 512), (512, 1), 0), reinterpret_tensor(arg106_1, (512, 128), (1, 512), 0), out=buf205)
        del arg106_1
        del buf204
        buf206 = reinterpret_tensor(buf205, (6, 128, 128), (16384, 128, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [x_6, x_7], Original ATen: [aten.add]
        triton_poi_fused_add_45.run(buf206, buf164, buf196, arg101_1, arg107_1, 98304, grid=grid(98304), stream=stream0)
        del arg101_1
        del arg107_1
        buf207 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [u_leaf_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg111_1, reinterpret_tensor(buf206, (768, 128), (128, 1), 0), reinterpret_tensor(arg110_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf207)
        del arg110_1
        del arg111_1
        buf208 = reinterpret_tensor(buf164, (768, 128), (128, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [v_leaf], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg113_1, reinterpret_tensor(buf206, (768, 128), (128, 1), 0), reinterpret_tensor(arg112_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf208)
        del arg112_1
        del arg113_1
        buf209 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf207, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf208, (6, 128, 128), (16384, 1, 128), 0), out=buf209)
        del buf207
        del buf208
        buf210 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [node_U], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg117_1, reinterpret_tensor(buf155, (512, 128), (128, 1), 0), reinterpret_tensor(arg116_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf210)
        del arg116_1
        del arg117_1
        buf211 = reinterpret_tensor(buf118, (512, 128), (128, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [node_V], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg119_1, reinterpret_tensor(buf155, (512, 128), (128, 1), 0), reinterpret_tensor(arg118_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf211)
        del arg118_1
        del arg119_1
        buf212 = reinterpret_tensor(buf57, (512, 1), (1, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf155, (512, 128), (128, 1), 0), reinterpret_tensor(arg114_1, (128, 1), (1, 128), 0), out=buf212)
        del arg114_1
        del buf155
        buf213 = reinterpret_tensor(buf212, (1, 4, 128, 1), (512, 128, 1, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_4], Original ATen: [aten.sigmoid]
        triton_poi_fused_sigmoid_46.run(buf213, arg115_1, 512, grid=grid(512), stream=stream0)
        del arg115_1
        buf214 = empty_strided_cuda((1, 295424), (295424, 1), torch.float32)
        # Topologically Sorted Source Nodes: [packed_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_47.run(buf201, buf209, buf210, buf211, buf213, buf214, 295424, grid=grid(295424), stream=stream0)
        del buf201
        del buf209
        del buf210
        del buf211
    return (buf214, reinterpret_tensor(buf213, (1, 4, 128), (512, 128, 1), 0), )


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
    arg28_1 = rand_strided((4, 128, 129, 4), (66048, 516, 4, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((4, 128, 129), (16512, 129, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((6, 128, 129, 4), (66048, 516, 4, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((6, 128, 129), (16512, 129, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg33_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg34_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg35_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg36_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
