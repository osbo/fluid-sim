# AOT ID: ['0_inference']
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ey/ceyi62lve7sfhuld5ynavsvzgilsllgia3ogcpfnf5bf2vswpchw.py
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
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 18
    x1 = (xindex // 18)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 6, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (3 + (9*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 18, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((-6) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/uu/cuuufj4qzn6iwbopza2cnutdz475v6farujdffngpl2uqkxypc7w.py
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
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_1', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/uu/cuuwt6vpun2ig7aj7ajyjmdflaglcqkllcqpouwolr4uejlzc2ea.py
# Topologically Sorted Source Nodes: [aggr], Original ATen: [aten.zeros_like]
# Source node to ATen node mapping:
#   aggr => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1024, 512], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_like_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_like_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (1024*x1)), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ja/cjaiuoescax6npczorsotaijz7afnv42pdbzzkxxpx4nhxiducec.py
# Topologically Sorted Source Nodes: [aggr, getitem_4, messages, index_add_], Original ATen: [aten.zeros_like, aten.index, aten.mul, aten.index_add]
# Source node to ATen node mapping:
#   aggr => full_default
#   getitem_4 => index
#   index_add_ => index_put
#   messages => mul_3
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1024, 512], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_index_add_mul_zeros_like_3', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3449344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 512)
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (6737 + x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 1024, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 1024)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 1024")
    tmp7 = tmp6 + tmp1
    tmp8 = tmp6 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp6)
    tl.device_assert(((0 <= tmp9) & (tmp9 < 1024)) | ~(xmask), "index out of bounds: 0 <= tmp9 < 1024")
    tmp11 = tl.load(in_ptr1 + (x0 + (512*tmp9)), xmask)
    tmp13 = tmp11 * tmp12
    tl.atomic_add(out_ptr0 + (x0 + (1024*tmp4)), tmp13, xmask, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ue/cue3p2bhgb5x3sw47gftjmqgc7zur7ugyqyhcipz3ubjzt5nip5v.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   out => add_2
# Graph fragment:
#   %add_tensor_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_23, %arg15_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze, %add_tensor_23), kwargs = {})
triton_poi_fused_add_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_4', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/i2/ci2klfek7asqi3ny366sq3bvdohvhk2tzmi34m6usthjde7tbaih.py
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
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_5', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.full([1], 512, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp4 - tmp12
    tmp19 = 512.0
    tmp20 = tmp17 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp28, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/3v/c3v25if6gtsnva5uvx5a2bkkrwkgmbprrnacjvprlgqwxxy7tfko.py
# Topologically Sorted Source Nodes: [layer_norm_1], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_1 => var_mean_1
# Graph fragment:
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_5, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_layer_norm_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_6', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 512, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/63/c63oebklp4czwjfhqx5eip3latxvsbrqpi6zwmcadj7ndrl34de3.py
# Topologically Sorted Source Nodes: [sub_1], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   sub_1 => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_8, [2, 4]), kwargs = {})
triton_per_fused_mean_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex % 8
    r4 = (rindex // 8)
    x0 = xindex % 4
    x1 = (xindex // 4) % 32
    x2 = (xindex // 128)
    x6 = xindex
    x5 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x0 + (4*r3) + (32*x1) + (1028*r4) + (8224*x2)), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = 64.0
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr1 + (x5 + (132*x2)), tmp5, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/cu/ccuonkgfbgozyexasberd5p5tqgqhrpnyowd74azvncwaickkiyc.py
# Topologically Sorted Source Nodes: [c_1, out_3], Original ATen: [aten.mean, aten.cat]
# Source node to ATen node mapping:
#   c_1 => mean_1
#   out_3 => cat_4
# Graph fragment:
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_9, [2]), kwargs = {})
#   %cat_4 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mean, %unsqueeze_6], 2), kwargs = {})
triton_per_fused_cat_mean_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[512, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_mean_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 4
    x1 = (xindex // 4)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0 + (1028*r2) + (8224*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 8.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x0 + (132*x1)), tmp6, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/yx/cyxvk7ih3ehxdhnr6qqg26ccptuyvwu7zap6nq76t5763hh5e5z4.py
# Topologically Sorted Source Nodes: [sub], Original ATen: [aten.amax]
# Source node to ATen node mapping:
#   sub => amax
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_6, [2, 4]), kwargs = {})
triton_per_fused_amax_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_amax_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex % 8
    r3 = (rindex // 8)
    x0 = xindex % 32
    x1 = (xindex // 32)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (8*x0) + (257*r3) + (2056*x1)), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = triton_helpers.max2(tmp1, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp3, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/x6/cx6b7syb2vt6z6vhga3ag3xbbsvvpcapddxnlizba7oqyo7e3vgk.py
# Topologically Sorted Source Nodes: [c], Original ATen: [aten.amax]
# Source node to ATen node mapping:
#   c => amax_1
# Graph fragment:
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_7, [2]), kwargs = {})
triton_per_fused_amax_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_amax_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + (257*r1) + (2056*x0)), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/vm/cvmxyqosnrhkgp5nx6m7lsnucj7ardtub3kxxzaqaxb6bx2pfslt.py
# Topologically Sorted Source Nodes: [x_attn], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_attn => mean_4
# Graph fragment:
#   %mean_4 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%view_17, [3]), kwargs = {})
triton_per_fused_mean_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_11', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x5 = xindex
    x3 = xindex % 16384
    x4 = (xindex // 16384)
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (4096*x1)), None)
    tmp1 = tl.load(in_ptr1 + (r2 + (8*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r2 + (8*x1)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None]
    tmp17 = 8.0
    tmp18 = tmp16 / tmp17
    tl.store(out_ptr1 + (x3 + (16896*x4)), tmp18, None)
    tl.store(out_ptr0 + (x5), tmp16, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/so/csoeqlmkcqzu3nya7pov6urnjxhjxjojdk4rjrae6gfpxnmnwn5h.py
# Topologically Sorted Source Nodes: [x_attn, block_node], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   block_node => mean_5
#   x_attn => mean_4
# Graph fragment:
#   %mean_4 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%view_17, [3]), kwargs = {})
#   %mean_5 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mean_4, [2], True), kwargs = {})
triton_per_fused_mean_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_12', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (16384*x1)), None)
    tmp1 = 8.0
    tmp2 = tmp0 / tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 32.0
    tmp7 = tmp5 / tmp6
    tl.store(out_ptr1 + (x0 + (16896*x1)), tmp7, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ec/cec5zqzl5zfn5q4hppwfugaxiqrj547epun7nfa7zahpeglw6w7x.py
# Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   einsum => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_14,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_13', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 32
    x2 = (xindex // 2048) % 8
    x3 = (xindex // 16384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1536*x1) + (50688*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/x7/cx7f66jvd6h2t3xm73g5asnnfddubt2kz5hj3pgb4onoaqtaxlsu.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_7 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_24, %full_default_33], 2), kwargs = {})
triton_poi_fused_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_14', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 36
    x1 = (xindex // 36) % 64
    x2 = (xindex // 2304)
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 33, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (512 + x1 + (64*(x2 % 8)) + (1536*x0) + (50688*(x2 // 8))), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (512 + x1 + (64*(x2 % 8))), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 36, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = 0.0
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp9, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/cr/ccrgjawbmhiiwdx352cusy5b3tgzdnxh6y2j3ontr5xabygppyfk.py
# Topologically Sorted Source Nodes: [setitem, setitem_1, setitem_4, setitem_5], Original ATen: [aten.lift_fresh, aten.fill]
# Source node to ATen node mapping:
#   setitem => copy, full_default_2
#   setitem_1 => copy_1, full_default_3
#   setitem_4 => copy_2, full_default_8
#   setitem_5 => copy_3, full_default_9
# Graph fragment:
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_8, %full_default_2), kwargs = {})
#   %select_scatter_default : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%expand_3, %copy, 3, 32), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_14, %full_default_3), kwargs = {})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int, %copy_1, 3, 3), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default, %select_scatter_default_1, 3, 32), kwargs = {})
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_2 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_21, %full_default_8), kwargs = {})
#   %select_scatter_default_4 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%expand_5, %copy_2, 3, 32), kwargs = {})
#   %full_default_9 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_3 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_27, %full_default_9), kwargs = {})
#   %select_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_1, %copy_3, 3, 3), kwargs = {})
#   %select_scatter_default_6 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_4, %select_scatter_default_5, 3, 32), kwargs = {})
triton_poi_fused_fill_lift_fresh_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_fill_lift_fresh_15', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 4) % 33
    x0 = xindex % 4
    x2 = (xindex // 132)
    x3 = xindex
    tmp7 = tl.load(in_ptr0 + (128 + x0 + (132*x2)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (x3), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 32, tl.int32)
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
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/rm/crmdzuhiwlaumzrpsjrld56y3yg4now2ahbg3lg75kki53fj2asu.py
# Topologically Sorted Source Nodes: [setitem, setitem_1, setitem_2], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
# Source node to ATen node mapping:
#   setitem => copy, full_default_2
#   setitem_1 => copy_1, full_default_3
#   setitem_2 => full_default_4, index_put_2
# Graph fragment:
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_8, %full_default_2), kwargs = {})
#   %select_scatter_default : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%expand_3, %copy, 3, 32), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_14, %full_default_3), kwargs = {})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int, %copy_1, 3, 3), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default, %select_scatter_default_1, 3, 32), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_2 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%select_scatter_default_2, [None, None, %iota, %iota], %full_default_4), kwargs = {})
triton_poi_fused_fill_index_put_lift_fresh_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_fill_index_put_lift_fresh_16', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = (xindex // 4) % 32
    x2 = (xindex // 128)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (136*x1) + (4224*x2)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/vf/cvfiqhc53diyaikrunotog5dlgcpirnqqsow3nhcsbhnt5pc2tmg.py
# Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_3 => full_default_5, index_put_3
# Graph fragment:
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_3 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_19, [None, None, %iota, %iota], %full_default_5), kwargs = {})
triton_poi_fused_index_put_lift_fresh_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_17', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + (4*x0)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/k4/ck4nswdrmuhcsjenj46wad3pqtbz2h26iye26okf22klz7jnuggv.py
# Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_3 => full_default_5, index_put_3
# Graph fragment:
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_3 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_19, [None, None, %iota, %iota], %full_default_5), kwargs = {})
triton_poi_fused_index_put_lift_fresh_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_18', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp0 = 1.0
    tl.store(out_ptr0 + ((34*x0) + (1056*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/qt/cqtkwxqqjfcoh24rbnritj32xhpu2mlnjnlfilg3ggqxxpobf3lf.py
# Topologically Sorted Source Nodes: [eq, scores_2, scores, scores_1, attn_probs], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_probs => amax_4, exp, sub_2, sum_1
#   eq => eq
#   scores => mul_15
#   scores_1 => add_9
#   scores_2 => clone_3, full_default_6, where
# Graph fragment:
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%unsqueeze_15, 0), kwargs = {})
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_26, 0.125), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %slice_65), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default_6, %add_9), kwargs = {})
#   %clone_3 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%where,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_4 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_3, [3], True), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_3, %amax_4), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [3], True), kwargs = {})
triton_per_fused__softmax_add_eq_masked_fill_mul_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_eq_masked_fill_mul_19', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 33
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 32
    x2 = (xindex // 256)
    x4 = xindex
    tmp13 = tl.load(in_ptr2 + (r3 + (36*x4)), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr3 + (r3 + (33*x0) + (1056*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_ptr4 + (3 + (4*r3) + (132*x0) + (4224*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp0 = r3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((32*x0) + (1024*x2) + r3), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 33, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (tl.broadcast_to(x0 + (32*x2), [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp11 = 0.0
    tmp12 = tmp10 == tmp11
    tmp14 = 0.125
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full([1, 1], 3, tl.int32)
    tmp17 = tmp16 == tmp16
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp15 + tmp20
    tmp22 = float("-inf")
    tmp23 = tl.where(tmp12, tmp22, tmp21)
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, float("-inf"))
    tmp27 = triton_helpers.max2(tmp26, 1)[:, None]
    tmp28 = tmp23 - tmp27
    tmp29 = tl_math.exp(tmp28)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp30, 0)
    tmp33 = tl.sum(tmp32, 1)[:, None]
    tl.store(out_ptr0 + (r3 + (33*x4)), tmp23, rmask & xmask)
    tl.store(out_ptr1 + (x4), tmp27, xmask)
    tl.store(out_ptr2 + (x4), tmp33, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/hs/chsbzyot2xopacixeplnyzurqtzb6qsegy4zrw6bpcma6hwslc3u.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %select_scatter_default_3 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%index_put_2, %index_put_3, 4, 3), kwargs = {})
triton_poi_fused_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_20', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16896
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/vk/cvkewd22esrp4skmoxnu4redocctlphv6tmuysscxr4jr2ohhs7p.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_5 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_30, %full_default_31], 2), kwargs = {})
triton_poi_fused_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_21', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 1056
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex
    x2 = (xindex // 33)
    x1 = xindex % 33
    tmp0 = tl.load(in_ptr0 + (x3 + (1056*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (32*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (32*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + ((8*x3) + (8448*(y0 // 8)) + (y0 % 8)), xmask & ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (y0 % 8), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tmp6 = x1
    tmp7 = tl.full([1, 1], 0, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tl.full([1, 1], 32, tl.int64)
    tmp10 = tmp6 < tmp9
    tmp11 = tl.load(in_ptr3 + ((32*x2) + (1024*(y0 // 8)) + x1), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp6 >= tmp9
    tmp13 = tl.full([1, 1], 33, tl.int64)
    tmp14 = tmp6 < tmp13
    tmp15 = tl.load(in_ptr4 + (x2 + (32*(y0 // 8))), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.where(tmp10, tmp11, tmp15)
    tmp17 = 0.0
    tmp18 = tmp16 == tmp17
    tmp21 = tmp19 + tmp20
    tmp22 = tl.where(tmp18, tmp17, tmp21)
    tmp23 = tmp5 + tmp22
    tl.store(out_ptr0 + (x1 + (36*x2) + (1152*y0)), tmp23, xmask & ymask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/w3/cw37naetxzn5gbiudy37tiuoli5vzt7fdaoqwyog6y72hm3mtu7a.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_31 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 32, 3], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(1,), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_22', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x1 = (xindex // 3)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (36*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/hi/chircyrt7dbxr5i4zxv6vejoaecz2rtz7wekco2xjorzt6ncqeam.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_6 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_31, %full_default_32], 1), kwargs = {})
triton_poi_fused_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_23', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 64) % 36
    x0 = xindex % 64
    x2 = (xindex // 2304)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 33, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (1024 + x0 + (64*(x2 % 8)) + (1536*x1) + (50688*(x2 // 8))), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (1024 + x0 + (64*(x2 % 8))), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 36, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = 0.0
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp9, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/eu/ceuojuanrnytajo4y6oz2dmudvebap3gubwcaieodmknkb63asm3.py
# Topologically Sorted Source Nodes: [x_out_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_out_1 => clone_6
# Graph fragment:
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_33,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_24', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 8
    x2 = (xindex // 512) % 32
    x3 = (xindex // 16384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (2048*x1) + (16384*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/gc/cgc4m3ha4shxucchnc3ccfismbs26yyt5kicr7wjo66zh5jbh3o2.py
# Topologically Sorted Source Nodes: [x, layer_norm_2], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_2 => add_12, add_13, mul_16, mul_17, rsqrt_2, sub_3, var_mean_2
#   x => add_11
# Graph fragment:
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %view_38), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_11, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_11, %getitem_5), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %arg40_1), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %arg41_1), kwargs = {})
triton_per_fused_add_native_layer_norm_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_25', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*((x0 % 256) // 8)) + (16384*(x0 // 256))), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.full([1], 512, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp4 - tmp12
    tmp19 = 512.0
    tmp20 = tmp17 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp28, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/c6/cc6223ccbmq76tz6kscmlnb2qxaw5seo3rneuhh52pxprhsgjnlq.py
# Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_11 => add_14, erf_3, mul_18, mul_19, mul_20
# Graph fragment:
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_40, 0.5), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_40, 0.7071067811865476), kwargs = {})
#   %erf_3 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_19,), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_3, 1), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %add_14), kwargs = {})
triton_poi_fused_gelu_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_26', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 2048
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ux/cuxxscfy6rdpailfdmbagemps5xrr36gouemlaorxbqjq55qogt7.py
# Topologically Sorted Source Nodes: [x, x_1, layer_norm_3], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_3 => var_mean_3
#   x => add_11
#   x_1 => add_15
# Graph fragment:
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %view_38), kwargs = {})
#   %add_15 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_42), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_15, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_27', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*((x0 % 256) // 8)) + (16384*(x0 // 256))), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (r1 + (512*x0)), None)
    tmp6 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tl.full([1], 512, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp9 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, None)
    tl.store(out_ptr0 + (x0), tmp16, None)
    tl.store(out_ptr1 + (x0), tmp21, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/mb/cmbeiavzff55gruojnhbewrpzryi25fpsi5ze2eeeoyywgo2jrdu.py
# Topologically Sorted Source Nodes: [x_2, x_3, h_diag], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   h_diag => add_25, add_26, mul_29, mul_30, rsqrt_5, sub_7, var_mean_5
#   x_2 => add_20
#   x_3 => add_24
# Graph fragment:
#   %add_20 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %view_65), kwargs = {})
#   %add_24 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %view_69), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_24, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, %getitem_11), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_25,), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_5), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %arg60_1), kwargs = {})
#   %add_26 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %arg61_1), kwargs = {})
triton_per_fused_add_native_layer_norm_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_28', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*((x0 % 256) // 8)) + (16384*(x0 // 256))), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (r1 + (512*x0)), None)
    tmp6 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tl.full([1], 512, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp9 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = tmp8 - tmp16
    tmp23 = 512.0
    tmp24 = tmp21 / tmp23
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, None)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp32, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/i2/ci2uzvejdlnzy4wxvhlatkwy2tq6uyxn4wdx3atmmi6skfggawv4.py
# Topologically Sorted Source Nodes: [layer_norm_6], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_6 => var_mean_6
# Graph fragment:
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_86, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_layer_norm_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_29', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1536
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tl.full([1], 512, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tl.store(out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr1 + (x0), tmp15, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/63/c635oxc4eejm225azs7f7evvzqfjwmwjtliiuuykqfp5zo2cashp.py
# Topologically Sorted Source Nodes: [sub_3], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   sub_3 => mean_2
# Graph fragment:
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_12, [2, 4]), kwargs = {})
triton_per_fused_mean_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_30', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex % 8
    r4 = (rindex // 8)
    x0 = xindex % 4
    x1 = (xindex // 4) % 32
    x2 = (xindex // 128)
    x6 = xindex
    x5 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x0 + (4*r3) + (32*x1) + (1028*r4) + (8224*x2)), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = 64.0
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr1 + (x5 + (132*x2)), tmp5, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/4x/c4xqu5ms7lpge3ertzos3qnw2a2y34ca577qvf5o3d2ylrkz2jid.py
# Topologically Sorted Source Nodes: [c_3, out_5], Original ATen: [aten.mean, aten.cat]
# Source node to ATen node mapping:
#   c_3 => mean_3
#   out_5 => cat_6
# Graph fragment:
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_13, [2]), kwargs = {})
#   %cat_6 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%mean_2, %unsqueeze_8], 2), kwargs = {})
triton_per_fused_cat_mean_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_mean_31', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 4
    x1 = (xindex // 4)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0 + (1028*r2) + (8224*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 8.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x0 + (132*x1)), tmp6, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/r2/cr2ucnmen6yrqmqw74cpw4iv4eeeibil7qai6ntnje6rfe2qe4rr.py
# Topologically Sorted Source Nodes: [sub_2], Original ATen: [aten.amax]
# Source node to ATen node mapping:
#   sub_2 => amax_2
# Graph fragment:
#   %amax_2 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_10, [2, 4]), kwargs = {})
triton_per_fused_amax_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_amax_32', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex % 8
    r3 = (rindex // 8)
    x0 = xindex % 32
    x1 = (xindex // 32)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (8*x0) + (257*r3) + (2056*x1)), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = triton_helpers.max2(tmp1, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp3, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ga/cgaq5b4rosfkoidphcitfwviassd2yyarxh65esnwndcxj3tzgql.py
# Topologically Sorted Source Nodes: [c_2], Original ATen: [aten.amax]
# Source node to ATen node mapping:
#   c_2 => amax_3
# Graph fragment:
#   %amax_3 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_11, [2]), kwargs = {})
triton_per_fused_amax_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[256, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_amax_33', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + (257*r1) + (2056*x0)), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/z7/cz7y2asrlokcijwvyqjifczud6mj5vtiq2kp5gjrbi5fqnhh42tf.py
# Topologically Sorted Source Nodes: [x_attn_2], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_attn_2 => mean_9
# Graph fragment:
#   %mean_9 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%view_88, [3]), kwargs = {})
triton_per_fused_mean_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[131072, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_34', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x5 = xindex
    x3 = xindex % 16384
    x4 = (xindex // 16384)
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (4096*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (4096*x1)), None)
    tmp3 = tl.load(in_ptr2 + (r2 + (8*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (r2 + (8*x1)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 512.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.sum(tmp16, 1)[:, None]
    tmp19 = 8.0
    tmp20 = tmp18 / tmp19
    tl.store(out_ptr1 + (x3 + (16896*x4)), tmp20, None)
    tl.store(out_ptr0 + (x5), tmp18, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/sp/cspwhvqs7pv3pgt55zc3wbmcj6h2zihpacmalhsxm7rqmi7cq6dt.py
# Topologically Sorted Source Nodes: [x_attn_2, block_node_2], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   block_node_2 => mean_10
#   x_attn_2 => mean_9
# Graph fragment:
#   %mean_9 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%view_88, [3]), kwargs = {})
#   %mean_10 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mean_9, [2], True), kwargs = {})
triton_per_fused_mean_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_35', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (16384*x1)), xmask, other=0.0)
    tmp1 = 8.0
    tmp2 = tmp0 / tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 32.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr1 + (x0 + (16896*x1)), tmp8, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/zy/czywwuvsikavmxqt472wlxllrqakrqqea537k2ppblfkffxh46pm.py
# Topologically Sorted Source Nodes: [einsum_6], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   einsum_6 => clone_16
# Graph fragment:
#   %clone_16 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_56,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_36', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 32
    x2 = (xindex // 2048) % 8
    x3 = (xindex // 16384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1536*x1) + (50688*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/g2/cg2pqmonxuqpwmkrwfom4ngshiwak3se6w5r2d7r7cqp2kvlw7zw.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_95, %full_default_27], 2), kwargs = {})
triton_poi_fused_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_37', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 110592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 36
    x1 = (xindex // 36) % 64
    x2 = (xindex // 2304)
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 33, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (512 + x1 + (64*(x2 % 8)) + (1536*x0) + (50688*(x2 // 8))), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (512 + x1 + (64*(x2 % 8))), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 36, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = 0.0
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp9, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/6v/c6vf5dozw2xflno3sazzbvnrxvrltik2l7vom2ohywwwzpqe2zre.py
# Topologically Sorted Source Nodes: [setitem_8, setitem_9, setitem_12, setitem_13], Original ATen: [aten.lift_fresh, aten.fill]
# Source node to ATen node mapping:
#   setitem_12 => copy_6, full_default_20
#   setitem_13 => copy_7, full_default_21
#   setitem_8 => copy_4, full_default_14
#   setitem_9 => copy_5, full_default_15
# Graph fragment:
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_4 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_34, %full_default_14), kwargs = {})
#   %select_scatter_default_8 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%view_15, %copy_4, 3, 32), kwargs = {})
#   %full_default_15 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_5 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_40, %full_default_15), kwargs = {})
#   %select_scatter_default_9 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_2, %copy_5, 3, 3), kwargs = {})
#   %select_scatter_default_10 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_8, %select_scatter_default_9, 3, 32), kwargs = {})
#   %full_default_20 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_6 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_47, %full_default_20), kwargs = {})
#   %select_scatter_default_12 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%view_15, %copy_6, 3, 32), kwargs = {})
#   %full_default_21 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_7 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_53, %full_default_21), kwargs = {})
#   %select_scatter_default_13 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_3, %copy_7, 3, 3), kwargs = {})
#   %select_scatter_default_14 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_12, %select_scatter_default_13, 3, 32), kwargs = {})
triton_poi_fused_fill_lift_fresh_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_fill_lift_fresh_38', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 4) % 33
    x0 = xindex % 4
    x2 = (xindex // 132)
    x3 = xindex
    tmp7 = tl.load(in_ptr0 + (128 + x0 + (132*x2)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (x3), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 32, tl.int32)
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
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/hk/chkypybdhzr4pjnaebvcn6l6o447ew3smrra3wgetkrwnvi6vklp.py
# Topologically Sorted Source Nodes: [setitem_8, setitem_9, setitem_10], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
# Source node to ATen node mapping:
#   setitem_10 => full_default_16, index_put_6
#   setitem_8 => copy_4, full_default_14
#   setitem_9 => copy_5, full_default_15
# Graph fragment:
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_4 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_34, %full_default_14), kwargs = {})
#   %select_scatter_default_8 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%view_15, %copy_4, 3, 32), kwargs = {})
#   %full_default_15 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_5 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_40, %full_default_15), kwargs = {})
#   %select_scatter_default_9 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_2, %copy_5, 3, 3), kwargs = {})
#   %select_scatter_default_10 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_8, %select_scatter_default_9, 3, 32), kwargs = {})
#   %full_default_16 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_6 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%select_scatter_default_10, [None, None, %iota_2, %iota_2], %full_default_16), kwargs = {})
triton_poi_fused_fill_index_put_lift_fresh_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_fill_index_put_lift_fresh_39', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = (xindex // 4) % 32
    x2 = (xindex // 128)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (136*x1) + (4224*x2)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ja/cjaygodhe2tqepuqgcnobsoai6xerqkjken2bzfuso76zrwawvnc.py
# Topologically Sorted Source Nodes: [setitem_11], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_11 => full_default_17, index_put_7
# Graph fragment:
#   %full_default_17 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_7 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_45, [None, None, %iota_2, %iota_2], %full_default_17), kwargs = {})
triton_poi_fused_index_put_lift_fresh_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_40', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + (4*x0)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/vk/cvka4c3kazqncpvcfeiokwkohcu4nmjwneoxngycojr4yp3tneio.py
# Topologically Sorted Source Nodes: [setitem_11], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_11 => full_default_17, index_put_7
# Graph fragment:
#   %full_default_17 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_7 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_45, [None, None, %iota_2, %iota_2], %full_default_17), kwargs = {})
triton_poi_fused_index_put_lift_fresh_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_41', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp0 = 1.0
    tl.store(out_ptr0 + ((34*x0) + (1056*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/my/cmyohd2etweydzarith2vs4wykckxadxj4l4aykhd24whcnhl5gl.py
# Topologically Sorted Source Nodes: [eq_4, scores_8, scores_6, scores_7, attn_probs_2], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_probs_2 => amax_6, exp_2, sub_9, sum_3
#   eq_4 => eq_4
#   scores_6 => mul_33
#   scores_7 => add_30
#   scores_8 => clone_19, full_default_18, where_4
# Graph fragment:
#   %eq_4 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%unsqueeze_37, 0), kwargs = {})
#   %full_default_18 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_97, 0.125), kwargs = {})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %slice_161), kwargs = {})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_4, %full_default_18, %add_30), kwargs = {})
#   %clone_19 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%where_4,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_6 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_19, [3], True), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_19, %amax_6), kwargs = {})
#   %exp_2 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_9,), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_2, [3], True), kwargs = {})
triton_per_fused__softmax_add_eq_masked_fill_mul_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_eq_masked_fill_mul_42', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 33
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 32
    x2 = (xindex // 256)
    x4 = xindex
    tmp13 = tl.load(in_ptr2 + (r3 + (36*x4)), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr3 + (r3 + (33*x0) + (1056*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_ptr4 + (3 + (4*r3) + (132*x0) + (4224*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp0 = r3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((32*x0) + (1024*x2) + r3), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 33, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (tl.broadcast_to(x0 + (32*x2), [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp11 = 0.0
    tmp12 = tmp10 == tmp11
    tmp14 = 0.125
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full([1, 1], 3, tl.int32)
    tmp17 = tmp16 == tmp16
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp15 + tmp20
    tmp22 = float("-inf")
    tmp23 = tl.where(tmp12, tmp22, tmp21)
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, float("-inf"))
    tmp27 = triton_helpers.max2(tmp26, 1)[:, None]
    tmp28 = tmp23 - tmp27
    tmp29 = tl_math.exp(tmp28)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp30, 0)
    tmp33 = tl.sum(tmp32, 1)[:, None]
    tl.store(out_ptr0 + (r3 + (33*x4)), tmp23, rmask & xmask)
    tl.store(out_ptr1 + (x4), tmp27, xmask)
    tl.store(out_ptr2 + (x4), tmp33, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/fh/cfh3imwuvzydm6hi2anuaym43kes4ioaslsprnltkbl6vwcsv6zr.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %select_scatter_default_11 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%index_put_6, %index_put_7, 4, 3), kwargs = {})
triton_poi_fused_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_43', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25344
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/2g/c2gtns6zsda7elqb45tlgziwlu6epgf6du4mamt6eah3djmkd2zt.py
# Topologically Sorted Source Nodes: [x_out_10], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_out_10 => clone_20
# Graph fragment:
#   %clone_20 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_62,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[64, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_44', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 48
    xnumel = 1056
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
    tmp0 = tl.load(in_out_ptr0 + (x5 + (1056*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x3 + (32*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x3 + (32*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (y0 + (8*x5) + (8448*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tmp6 = x2
    tmp7 = tl.full([1, 1], 0, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tl.full([1, 1], 32, tl.int64)
    tmp10 = tmp6 < tmp9
    tmp11 = tl.load(in_ptr2 + ((32*x3) + (1024*y1) + x2), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp6 >= tmp9
    tmp13 = tl.full([1, 1], 33, tl.int64)
    tmp14 = tmp6 < tmp13
    tmp15 = tl.load(in_ptr3 + (x3 + (32*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.where(tmp10, tmp11, tmp15)
    tmp17 = 0.0
    tmp18 = tmp16 == tmp17
    tmp21 = tmp19 + tmp20
    tmp22 = tl.where(tmp18, tmp17, tmp21)
    tmp23 = tmp5 + tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5 + (1056*y4)), tmp23, xmask & ymask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/x6/cx6477roqdcyb4jq7mc3ya3dmb64sc4ccuhzcrogg744re4hvn2v.py
# Topologically Sorted Source Nodes: [x_out_10], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_out_10 => clone_21
# Graph fragment:
#   %clone_21 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_63,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_45', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 101376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 33
    x2 = (xindex // 2112) % 8
    x3 = (xindex // 16896)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0 + (64*x2) + (1536*x1) + (50688*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (1024 + x0 + (64*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/u5/cu5a6fzl4iy5grwbngeyemav5zkfzcxj64b4kzh7lb53rojbnh36.py
# Topologically Sorted Source Nodes: [x_out_11], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_out_11 => clone_22
# Graph fragment:
#   %clone_22 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_104,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_46', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 8
    x2 = (xindex // 512) % 32
    x3 = (xindex // 16384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (2048*x1) + (16384*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/su/csuvxkqvbzb7lwr574xoo5lushrmhudmbw2kbps763lvakkdoa25.py
# Topologically Sorted Source Nodes: [x_4, layer_norm_7], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_7 => add_33, add_34, mul_34, mul_35, rsqrt_7, sub_10, var_mean_7
#   x_4 => add_32
# Graph fragment:
#   %add_32 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_86, %view_109), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_32, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_32, %getitem_15), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-05), kwargs = {})
#   %rsqrt_7 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_33,), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %rsqrt_7), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %arg74_1), kwargs = {})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %arg75_1), kwargs = {})
triton_per_fused_add_native_layer_norm_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_47', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1536
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), None)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), None)
    tmp3 = tl.load(in_ptr2 + (r2 + (512*(x0 // 8)) + (16384*x1)), None)
    tmp4 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tl.full([1], 512, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp7 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = tmp6 - tmp14
    tmp21 = 512.0
    tmp22 = tmp19 / tmp21
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp30, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/6n/c6nrs7k4246prbpxozxl2cf7hkfretujiwbbiosbxr77jppyb3ut.py
# Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_17 => add_35, erf_5, mul_36, mul_37, mul_38
# Graph fragment:
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_111, 0.5), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_111, 0.7071067811865476), kwargs = {})
#   %erf_5 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_37,), kwargs = {})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_5, 1), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_36, %add_35), kwargs = {})
triton_poi_fused_gelu_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_48', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 2048
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/lu/clunfkls6eu6zwalkdgyji3xz5k3go2slywbzpy4vx3c275jpdhr.py
# Topologically Sorted Source Nodes: [x_4, x_5, layer_norm_8], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_8 => var_mean_8
#   x_4 => add_32
#   x_5 => add_36
# Graph fragment:
#   %add_32 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_86, %view_109), kwargs = {})
#   %add_36 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_32, %view_113), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_36, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_49', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1536
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), None)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), None)
    tmp3 = tl.load(in_ptr2 + (r2 + (512*(x0 // 8)) + (16384*x1)), None)
    tmp4 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (r2 + (512*x3)), None)
    tmp8 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tl.full([1], 512, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp11 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp10, None)
    tl.store(out_ptr0 + (x3), tmp18, None)
    tl.store(out_ptr1 + (x3), tmp23, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/wz/cwz7khdcfcqgldkhtnoboyykbljlwpthyusmp25i625hnftkeiom.py
# Topologically Sorted Source Nodes: [x_attn_3], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_attn_3 => mean_11
# Graph fragment:
#   %mean_11 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%view_115, [3]), kwargs = {})
triton_per_fused_mean_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[131072, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_50', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x5 = xindex
    x3 = xindex % 16384
    x4 = (xindex // 16384)
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (4096*x1)), None)
    tmp1 = tl.load(in_ptr1 + (r2 + (8*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r2 + (8*x1)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None]
    tmp17 = 8.0
    tmp18 = tmp16 / tmp17
    tl.store(out_ptr1 + (x3 + (16896*x4)), tmp18, None)
    tl.store(out_ptr0 + (x5), tmp16, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/bz/cbzmcbnamigydm2rtjjvoag57utbfkkhzsyntx5twvezvlf6xeny.py
# Topologically Sorted Source Nodes: [x_6, layer_norm_9], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_9 => add_42, add_43, mul_42, mul_43, rsqrt_9, sub_13, var_mean_9
#   x_6 => add_41
# Graph fragment:
#   %add_41 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_36, %view_136), kwargs = {})
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_41, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_41, %getitem_19), kwargs = {})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_18, 1e-05), kwargs = {})
#   %rsqrt_9 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_42,), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %rsqrt_9), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_42, %arg88_1), kwargs = {})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %arg89_1), kwargs = {})
triton_per_fused_add_native_layer_norm_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_51', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1536
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), None)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(x0 // 8)) + (16384*x1)), None)
    tmp2 = tl.load(in_ptr2 + (r2), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.full([1], 512, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp4 - tmp12
    tmp19 = 512.0
    tmp20 = tmp17 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp28, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/no/cnowlyfs4uqapbr7ym7ye26vjm2ouw4sxa3tnx5rqfpa43lgmv3r.py
# Topologically Sorted Source Nodes: [h_a], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   h_a => mean_8
# Graph fragment:
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_71, [3]), kwargs = {})
triton_per_fused_mean_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_52', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (4096*x1)), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = 8.0
    tmp5 = tmp3 / tmp4
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ix/cixm5pedynghxyrbl2y6educ7udunynxinwebwokrpzzve37tndo.py
# Topologically Sorted Source Nodes: [h_a_1], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   h_a_1 => mean_13
# Graph fragment:
#   %mean_13 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%view_144, [3]), kwargs = {})
triton_per_fused_mean_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[131072, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_53', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (4096*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0 + (512*r2) + (4096*x1)), None)
    tmp6 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None]
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/gt/cgtyf72u44mxnvg2yaaf7bfkhlpjndjzw7zc5taacaw2safeu7k2.py
# Topologically Sorted Source Nodes: [sigmoid], Original ATen: [aten.sigmoid]
# Source node to ATen node mapping:
#   sigmoid => sigmoid
# Graph fragment:
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_155,), kwargs = {})
triton_poi_fused_sigmoid_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_54', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/lg/clg2abkgsdmvdqnb6y2g7o6m3mwbfi7y6hgqbqya2o4uymyinbx2.py
# Topologically Sorted Source Nodes: [packed_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   packed_1 => cat_12
# Graph fragment:
#   %cat_12 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_11, %view_156], 1), kwargs = {})
triton_poi_fused_cat_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_55', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 76800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 75776, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 4096, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.load(in_ptr0 + (x0), tmp7 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp0 >= tmp5
    tmp10 = tl.full([1], 10240, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tmp12 & tmp4
    tmp14 = tl.load(in_ptr1 + ((-4096) + x0), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp0 >= tmp10
    tmp16 = tl.full([1], 43008, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr2 + ((-10240) + x0), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp0 >= tmp16
    tmp22 = tmp21 & tmp4
    tmp23 = tl.load(in_ptr3 + ((-43008) + x0), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp18, tmp20, tmp23)
    tmp25 = tl.where(tmp12, tmp14, tmp24)
    tmp26 = tl.where(tmp6, tmp8, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp4, tmp26, tmp27)
    tmp29 = tmp0 >= tmp3
    tmp30 = tl.full([1], 76800, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = tl.load(in_ptr4 + ((-75776) + x0), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.where(tmp4, tmp28, tmp32)
    tl.store(out_ptr0 + (x0), tmp33, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1024, 9), (67716, 9, 1))
    assert_size_stride(arg1_1, (1, 12), (12, 1))
    assert_size_stride(arg2_1, (512, 18), (18, 1))
    assert_size_stride(arg3_1, (512, ), (1, ))
    assert_size_stride(arg4_1, (512, 512), (512, 1))
    assert_size_stride(arg5_1, (512, ), (1, ))
    assert_size_stride(arg6_1, (2, 6737), (6737, 1))
    assert_size_stride(arg7_1, (6737, ), (1, ))
    assert_size_stride(arg8_1, (512, 512), (512, 1))
    assert_size_stride(arg9_1, (512, ), (1, ))
    assert_size_stride(arg10_1, (512, 512), (512, 1))
    assert_size_stride(arg11_1, (512, ), (1, ))
    assert_size_stride(arg12_1, (512, 1024), (1024, 1))
    assert_size_stride(arg13_1, (512, ), (1, ))
    assert_size_stride(arg14_1, (512, 512), (512, 1))
    assert_size_stride(arg15_1, (512, ), (1, ))
    assert_size_stride(arg16_1, (512, 512), (512, 1))
    assert_size_stride(arg17_1, (512, ), (1, ))
    assert_size_stride(arg18_1, (512, 512), (512, 1))
    assert_size_stride(arg19_1, (512, ), (1, ))
    assert_size_stride(arg20_1, (512, 1024), (1024, 1))
    assert_size_stride(arg21_1, (512, ), (1, ))
    assert_size_stride(arg22_1, (512, 512), (512, 1))
    assert_size_stride(arg23_1, (512, ), (1, ))
    assert_size_stride(arg24_1, (512, ), (1, ))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (512, 512), (512, 1))
    assert_size_stride(arg27_1, (512, ), (1, ))
    assert_size_stride(arg28_1, (4, 256, 257, 4), (263168, 1028, 4, 1))
    assert_size_stride(arg29_1, (4, 256, 257), (65792, 257, 1))
    assert_size_stride(arg30_1, (6, 256, 257, 4), (263168, 1028, 4, 1))
    assert_size_stride(arg31_1, (6, 256, 257), (65792, 257, 1))
    assert_size_stride(arg32_1, (512, ), (1, ))
    assert_size_stride(arg33_1, (512, ), (1, ))
    assert_size_stride(arg34_1, (1536, 512), (512, 1))
    assert_size_stride(arg35_1, (1536, ), (1, ))
    assert_size_stride(arg36_1, (8, 4), (4, 1))
    assert_size_stride(arg37_1, (8, ), (1, ))
    assert_size_stride(arg38_1, (512, 512), (512, 1))
    assert_size_stride(arg39_1, (512, ), (1, ))
    assert_size_stride(arg40_1, (512, ), (1, ))
    assert_size_stride(arg41_1, (512, ), (1, ))
    assert_size_stride(arg42_1, (2048, 512), (512, 1))
    assert_size_stride(arg43_1, (2048, ), (1, ))
    assert_size_stride(arg44_1, (512, 2048), (2048, 1))
    assert_size_stride(arg45_1, (512, ), (1, ))
    assert_size_stride(arg46_1, (512, ), (1, ))
    assert_size_stride(arg47_1, (512, ), (1, ))
    assert_size_stride(arg48_1, (1536, 512), (512, 1))
    assert_size_stride(arg49_1, (1536, ), (1, ))
    assert_size_stride(arg50_1, (8, 4), (4, 1))
    assert_size_stride(arg51_1, (8, ), (1, ))
    assert_size_stride(arg52_1, (512, 512), (512, 1))
    assert_size_stride(arg53_1, (512, ), (1, ))
    assert_size_stride(arg54_1, (512, ), (1, ))
    assert_size_stride(arg55_1, (512, ), (1, ))
    assert_size_stride(arg56_1, (2048, 512), (512, 1))
    assert_size_stride(arg57_1, (2048, ), (1, ))
    assert_size_stride(arg58_1, (512, 2048), (2048, 1))
    assert_size_stride(arg59_1, (512, ), (1, ))
    assert_size_stride(arg60_1, (512, ), (1, ))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (32, 512), (512, 1))
    assert_size_stride(arg63_1, (32, ), (1, ))
    assert_size_stride(arg64_1, (6, 4), (4, 1))
    assert_size_stride(arg65_1, (6, 4), (4, 1))
    assert_size_stride(arg66_1, (512, ), (1, ))
    assert_size_stride(arg67_1, (512, ), (1, ))
    assert_size_stride(arg68_1, (1536, 512), (512, 1))
    assert_size_stride(arg69_1, (1536, ), (1, ))
    assert_size_stride(arg70_1, (8, 4), (4, 1))
    assert_size_stride(arg71_1, (8, ), (1, ))
    assert_size_stride(arg72_1, (512, 512), (512, 1))
    assert_size_stride(arg73_1, (512, ), (1, ))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, ), (1, ))
    assert_size_stride(arg76_1, (2048, 512), (512, 1))
    assert_size_stride(arg77_1, (2048, ), (1, ))
    assert_size_stride(arg78_1, (512, 2048), (2048, 1))
    assert_size_stride(arg79_1, (512, ), (1, ))
    assert_size_stride(arg80_1, (512, ), (1, ))
    assert_size_stride(arg81_1, (512, ), (1, ))
    assert_size_stride(arg82_1, (1536, 512), (512, 1))
    assert_size_stride(arg83_1, (1536, ), (1, ))
    assert_size_stride(arg84_1, (8, 4), (4, 1))
    assert_size_stride(arg85_1, (8, ), (1, ))
    assert_size_stride(arg86_1, (512, 512), (512, 1))
    assert_size_stride(arg87_1, (512, ), (1, ))
    assert_size_stride(arg88_1, (512, ), (1, ))
    assert_size_stride(arg89_1, (512, ), (1, ))
    assert_size_stride(arg90_1, (2048, 512), (512, 1))
    assert_size_stride(arg91_1, (2048, ), (1, ))
    assert_size_stride(arg92_1, (512, 2048), (2048, 1))
    assert_size_stride(arg93_1, (512, ), (1, ))
    assert_size_stride(arg94_1, (32, 512), (512, 1))
    assert_size_stride(arg95_1, (32, ), (1, ))
    assert_size_stride(arg96_1, (32, 512), (512, 1))
    assert_size_stride(arg97_1, (32, ), (1, ))
    assert_size_stride(arg98_1, (1, 512), (512, 1))
    assert_size_stride(arg99_1, (1, ), (1, ))
    assert_size_stride(arg100_1, (32, 512), (512, 1))
    assert_size_stride(arg101_1, (32, ), (1, ))
    assert_size_stride(arg102_1, (32, 512), (512, 1))
    assert_size_stride(arg103_1, (32, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1024, 18), (18432, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [node_feats_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(arg0_1, arg1_1, buf0, 18432, grid=grid(18432), stream=stream0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((1024, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf0, (1024, 18), (18, 1), 0), reinterpret_tensor(arg2_1, (18, 512), (1, 18), 0), out=buf1)
        del arg2_1
        del buf0
        buf2 = reinterpret_tensor(buf1, (1, 1024, 512), (524288, 512, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_1.run(buf2, arg3_1, 524288, grid=grid(524288), stream=stream0)
        del arg3_1
        buf3 = empty_strided_cuda((1024, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg5_1, reinterpret_tensor(buf2, (1024, 512), (512, 1), 0), reinterpret_tensor(arg4_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf3)
        del arg4_1
        del arg5_1
        buf8 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        buf4 = reinterpret_tensor(buf8, (1024, 512), (1024, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [self_features], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, buf3, reinterpret_tensor(arg10_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf4)
        del arg10_1
        del arg11_1
        buf5 = reinterpret_tensor(buf2, (1024, 512), (512, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [neighbor_features], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg9_1, buf3, reinterpret_tensor(arg8_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf5)
        del arg8_1
        del arg9_1
        buf6 = reinterpret_tensor(buf8, (1024, 512), (1024, 1), 512)  # alias
        # Topologically Sorted Source Nodes: [aggr], Original ATen: [aten.zeros_like]
        triton_poi_fused_zeros_like_2.run(buf6, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [aggr, getitem_4, messages, index_add_], Original ATen: [aten.zeros_like, aten.index, aten.mul, aten.index_add]
        triton_poi_fused_index_index_add_mul_zeros_like_3.run(arg6_1, buf5, arg7_1, buf6, 3449344, grid=grid(3449344), stream=stream0)
        del buf4
        del buf6
        buf9 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf8, reinterpret_tensor(arg12_1, (1024, 512), (1, 1024), 0), out=buf9)
        del arg12_1
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_1.run(buf10, arg13_1, 524288, grid=grid(524288), stream=stream0)
        del arg13_1
        buf11 = empty_strided_cuda((1024, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.gelu]
        extern_kernels.mm(buf10, reinterpret_tensor(arg14_1, (512, 512), (1, 512), 0), out=buf11)
        del arg14_1
        buf12 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.add]
        triton_poi_fused_add_4.run(buf12, buf3, arg15_1, 524288, grid=grid(524288), stream=stream0)
        del arg15_1
        buf17 = buf8; del buf8  # reuse
        buf13 = reinterpret_tensor(buf17, (1024, 512), (1024, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [self_features_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg19_1, buf12, reinterpret_tensor(arg18_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf13)
        del arg18_1
        del arg19_1
        buf14 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [neighbor_features_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg17_1, buf12, reinterpret_tensor(arg16_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf14)
        del arg16_1
        del arg17_1
        buf15 = reinterpret_tensor(buf17, (1024, 512), (1024, 1), 512)  # alias
        # Topologically Sorted Source Nodes: [aggr_1], Original ATen: [aten.zeros_like]
        triton_poi_fused_zeros_like_2.run(buf15, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [aggr_1, getitem_7, messages_1, index_add__1], Original ATen: [aten.zeros_like, aten.index, aten.mul, aten.index_add]
        triton_poi_fused_index_index_add_mul_zeros_like_3.run(arg6_1, buf14, arg7_1, buf15, 3449344, grid=grid(3449344), stream=stream0)
        del arg6_1
        del arg7_1
        del buf13
        del buf15
        buf18 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf17, reinterpret_tensor(arg20_1, (1024, 512), (1, 1024), 0), out=buf18)
        del arg20_1
        del buf17
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_1.run(buf19, arg21_1, 524288, grid=grid(524288), stream=stream0)
        del arg21_1
        buf20 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.gelu]
        extern_kernels.mm(buf19, reinterpret_tensor(arg22_1, (512, 512), (1, 512), 0), out=buf20)
        del arg22_1
        buf24 = reinterpret_tensor(buf19, (1, 1024, 512), (524288, 512, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [h_2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_5.run(buf12, buf20, arg23_1, arg24_1, arg25_1, buf24, 1024, 512, grid=grid(1024), stream=stream0)
        del arg23_1
        del arg24_1
        del arg25_1
        buf25 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg27_1, reinterpret_tensor(buf24, (1024, 512), (512, 1), 0), reinterpret_tensor(arg26_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf25)
        del arg26_1
        del arg27_1
        buf26 = empty_strided_cuda((1, 1024, 1), (1024, 1, 1024), torch.float32)
        buf27 = empty_strided_cuda((1, 1024, 1), (1024, 1, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_6.run(buf25, buf26, buf27, 1024, 512, grid=grid(1024), stream=stream0)
        buf33 = empty_strided_cuda((4, 32, 33, 4), (4224, 132, 4, 1), torch.float32)
        buf31 = reinterpret_tensor(buf33, (4, 32, 32, 4), (4224, 132, 4, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sub_1], Original ATen: [aten.mean]
        triton_per_fused_mean_7.run(arg28_1, buf31, 16384, 64, grid=grid(16384), stream=stream0)
        buf32 = reinterpret_tensor(buf33, (4, 32, 1, 4), (4224, 132, 4, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [c_1, out_3], Original ATen: [aten.mean, aten.cat]
        triton_per_fused_cat_mean_8.run(arg28_1, buf32, 512, 8, grid=grid(512), stream=stream0)
        del arg28_1
        buf34 = empty_strided_cuda((4, 32, 32), (1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sub], Original ATen: [aten.amax]
        triton_per_fused_amax_9.run(arg29_1, buf34, 4096, 64, grid=grid(4096), stream=stream0)
        del buf31
        del buf32
        buf35 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [c], Original ATen: [aten.amax]
        triton_per_fused_amax_10.run(arg29_1, buf35, 128, 8, grid=grid(128), stream=stream0)
        del arg29_1
        buf36 = empty_strided_cuda((1, 4, 32, 512), (65536, 16384, 512, 1), torch.float32)
        buf40 = empty_strided_cuda((1, 4, 33, 512), (67584, 16896, 512, 1), torch.float32)
        buf38 = reinterpret_tensor(buf40, (1, 4, 32, 512), (67584, 16896, 512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_attn], Original ATen: [aten.mean]
        triton_per_fused_mean_11.run(buf25, buf26, buf27, arg32_1, arg33_1, buf36, buf38, 65536, 8, grid=grid(65536), stream=stream0)
        del arg32_1
        del arg33_1
        buf39 = reinterpret_tensor(buf40, (1, 4, 1, 512), (67584, 16896, 512, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [x_attn, block_node], Original ATen: [aten.mean]
        triton_per_fused_mean_12.run(buf36, buf39, 2048, 32, grid=grid(2048), stream=stream0)
        del buf38
        del buf39
        buf41 = empty_strided_cuda((132, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (132, 512), (512, 1), 0), reinterpret_tensor(arg34_1, (512, 1536), (1, 512), 0), out=buf41)
        del arg34_1
        buf42 = reinterpret_tensor(buf36, (4, 8, 32, 64, 1, 1), (16384, 2048, 64, 1, 1, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf41, arg35_1, buf42, 65536, grid=grid(65536), stream=stream0)
        buf43 = empty_strided_cuda((32, 64, 36), (2304, 36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(buf41, arg35_1, buf43, 73728, grid=grid(73728), stream=stream0)
        buf44 = empty_strided_cuda((32, 32, 36), (1152, 36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf42, (32, 32, 64), (2048, 64, 1), 0), buf43, out=buf44)
        buf45 = empty_strided_cuda((1, 4, 32, 33, 4), (16896, 4224, 132, 4, 1), torch.float32)
        buf81 = empty_strided_cuda((1, 4, 32, 33, 4), (16896, 4224, 132, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem, setitem_1, setitem_4, setitem_5], Original ATen: [aten.lift_fresh, aten.fill]
        triton_poi_fused_fill_lift_fresh_15.run(buf33, buf45, buf81, 16896, grid=grid(16896), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem, setitem_1, setitem_2], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
        triton_poi_fused_fill_index_put_lift_fresh_16.run(buf45, 512, grid=grid(512), stream=stream0)
        buf47 = empty_strided_cuda((1, 4, 32, 33), (4224, 1056, 33, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_17.run(buf45, buf47, 4224, grid=grid(4224), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_18.run(buf47, 128, grid=grid(128), stream=stream0)
        buf49 = empty_strided_cuda((1, 4, 32, 33, 8), (33792, 8448, 33, 1, 1056), torch.float32)
        buf50 = reinterpret_tensor(buf27, (1, 4, 32, 1, 8), (1024, 256, 1, 1024, 32), 0); del buf27  # reuse
        buf51 = reinterpret_tensor(buf26, (1, 4, 32, 1, 8), (1024, 256, 1, 1024, 32), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [eq, scores_2, scores, scores_1, attn_probs], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_eq_masked_fill_mul_19.run(buf34, buf35, buf44, buf47, buf45, buf49, buf50, buf51, 1024, 33, grid=grid(1024), stream=stream0)
        buf52 = reinterpret_tensor(buf33, (1, 4, 32, 33, 4), (16896, 4224, 132, 4, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_20.run(buf47, buf45, buf52, 16896, grid=grid(16896), stream=stream0)
        del buf45
        buf53 = empty_strided_cuda((4224, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf52, (4224, 4), (4, 1), 0), reinterpret_tensor(arg36_1, (4, 8), (1, 4), 0), out=buf53)
        del arg36_1
        buf56 = buf44; del buf44  # reuse
        buf54 = reinterpret_tensor(buf56, (32, 32, 33), (1152, 36, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_21.run(buf49, buf50, buf51, buf34, buf35, buf53, arg37_1, buf54, 32, 1056, grid=grid(32, 1056), stream=stream0)
        del arg37_1
        buf55 = reinterpret_tensor(buf56, (32, 32, 3), (1152, 36, 1), 33)  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_22.run(buf55, 3072, grid=grid(3072), stream=stream0)
        buf57 = reinterpret_tensor(buf43, (32, 36, 64), (2304, 64, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_23.run(buf41, arg35_1, buf57, 73728, grid=grid(73728), stream=stream0)
        del arg35_1
        del buf54
        del buf55
        buf58 = reinterpret_tensor(buf42, (32, 32, 64), (2048, 64, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(buf56, buf57, out=buf58)
        buf59 = empty_strided_cuda((1, 4, 32, 8, 64), (65536, 16384, 512, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_out_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf58, buf59, 65536, grid=grid(65536), stream=stream0)
        buf60 = reinterpret_tensor(buf58, (128, 512), (512, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf59, (128, 512), (512, 1), 0), reinterpret_tensor(arg38_1, (512, 512), (1, 512), 0), out=buf60)
        del arg38_1
        buf64 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [x, layer_norm_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_25.run(buf25, buf60, arg39_1, arg40_1, arg41_1, buf64, 1024, 512, grid=grid(1024), stream=stream0)
        del arg40_1
        del arg41_1
        buf65 = empty_strided_cuda((1024, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf64, (1024, 512), (512, 1), 0), reinterpret_tensor(arg42_1, (512, 2048), (1, 512), 0), out=buf65)
        del arg42_1
        buf66 = reinterpret_tensor(buf65, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_26.run(buf66, arg43_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg43_1
        buf67 = reinterpret_tensor(buf64, (1024, 512), (512, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf66, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg44_1, (2048, 512), (1, 2048), 0), out=buf67)
        del arg44_1
        buf68 = reinterpret_tensor(buf67, (1, 1024, 512), (524288, 512, 1), 0); del buf67  # reuse
        buf69 = reinterpret_tensor(buf51, (1, 1024, 1), (1024, 1, 1024), 0); del buf51  # reuse
        buf70 = reinterpret_tensor(buf50, (1, 1024, 1), (1024, 1, 1024), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [x, x_1, layer_norm_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf68, buf25, buf60, arg39_1, arg45_1, buf69, buf70, 1024, 512, grid=grid(1024), stream=stream0)
        del arg39_1
        del arg45_1
        buf72 = reinterpret_tensor(buf60, (1, 4, 32, 512), (65536, 16384, 512, 1), 0); del buf60  # reuse
        buf76 = buf40; del buf40  # reuse
        buf74 = reinterpret_tensor(buf76, (1, 4, 32, 512), (67584, 16896, 512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_attn_1], Original ATen: [aten.mean]
        triton_per_fused_mean_11.run(buf68, buf69, buf70, arg46_1, arg47_1, buf72, buf74, 65536, 8, grid=grid(65536), stream=stream0)
        del arg46_1
        del arg47_1
        buf75 = reinterpret_tensor(buf76, (1, 4, 1, 512), (67584, 16896, 512, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [x_attn_1, block_node_1], Original ATen: [aten.mean]
        triton_per_fused_mean_12.run(buf72, buf75, 2048, 32, grid=grid(2048), stream=stream0)
        del buf74
        del buf75
        buf77 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (132, 512), (512, 1), 0), reinterpret_tensor(arg48_1, (512, 1536), (1, 512), 0), out=buf77)
        del arg48_1
        del buf76
        buf78 = reinterpret_tensor(buf72, (4, 8, 32, 64, 1, 1), (16384, 2048, 64, 1, 1, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf77, arg49_1, buf78, 65536, grid=grid(65536), stream=stream0)
        buf79 = reinterpret_tensor(buf57, (32, 64, 36), (2304, 36, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(buf77, arg49_1, buf79, 73728, grid=grid(73728), stream=stream0)
        buf80 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf78, (32, 32, 64), (2048, 64, 1), 0), buf79, out=buf80)
        # Topologically Sorted Source Nodes: [setitem_4, setitem_5, setitem_6], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
        triton_poi_fused_fill_index_put_lift_fresh_16.run(buf81, 512, grid=grid(512), stream=stream0)
        buf83 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [setitem_7], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_17.run(buf81, buf83, 4224, grid=grid(4224), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_7], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_18.run(buf83, 128, grid=grid(128), stream=stream0)
        buf85 = reinterpret_tensor(buf53, (1, 4, 32, 33, 8), (33792, 8448, 33, 1, 1056), 0); del buf53  # reuse
        buf86 = reinterpret_tensor(buf70, (1, 4, 32, 1, 8), (1024, 256, 1, 1024, 32), 0); del buf70  # reuse
        buf87 = reinterpret_tensor(buf69, (1, 4, 32, 1, 8), (1024, 256, 1, 1024, 32), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [eq_2, scores_5, scores_3, scores_4, attn_probs_1], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_eq_masked_fill_mul_19.run(buf34, buf35, buf80, buf83, buf81, buf85, buf86, buf87, 1024, 33, grid=grid(1024), stream=stream0)
        buf88 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_20.run(buf83, buf81, buf88, 16896, grid=grid(16896), stream=stream0)
        del buf81
        del buf83
        buf89 = reinterpret_tensor(buf49, (4224, 8), (8, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf88, (4224, 4), (4, 1), 0), reinterpret_tensor(arg50_1, (4, 8), (1, 4), 0), out=buf89)
        del arg50_1
        del buf88
        buf92 = buf80; del buf80  # reuse
        buf90 = reinterpret_tensor(buf92, (32, 32, 33), (1152, 36, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_21.run(buf85, buf86, buf87, buf34, buf35, buf89, arg51_1, buf90, 32, 1056, grid=grid(32, 1056), stream=stream0)
        del arg51_1
        del buf35
        del buf85
        del buf86
        del buf89
        buf91 = reinterpret_tensor(buf92, (32, 32, 3), (1152, 36, 1), 33)  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_22.run(buf91, 3072, grid=grid(3072), stream=stream0)
        buf93 = reinterpret_tensor(buf79, (32, 36, 64), (2304, 64, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_23.run(buf77, arg49_1, buf93, 73728, grid=grid(73728), stream=stream0)
        del arg49_1
        del buf77
        del buf90
        del buf91
        buf94 = reinterpret_tensor(buf78, (32, 32, 64), (2048, 64, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(buf92, buf93, out=buf94)
        del buf92
        del buf93
        buf95 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [x_out_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf94, buf95, 65536, grid=grid(65536), stream=stream0)
        buf96 = reinterpret_tensor(buf94, (128, 512), (512, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (128, 512), (512, 1), 0), reinterpret_tensor(arg52_1, (512, 512), (1, 512), 0), out=buf96)
        del arg52_1
        del buf95
        buf100 = reinterpret_tensor(buf12, (1, 1024, 512), (524288, 512, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_2, layer_norm_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_25.run(buf68, buf96, arg53_1, arg54_1, arg55_1, buf100, 1024, 512, grid=grid(1024), stream=stream0)
        del arg54_1
        del arg55_1
        buf101 = reinterpret_tensor(buf66, (1024, 2048), (2048, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf100, (1024, 512), (512, 1), 0), reinterpret_tensor(arg56_1, (512, 2048), (1, 512), 0), out=buf101)
        del arg56_1
        buf102 = reinterpret_tensor(buf101, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_26.run(buf102, arg57_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg57_1
        buf103 = reinterpret_tensor(buf100, (1024, 512), (512, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf102, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg58_1, (2048, 512), (1, 2048), 0), out=buf103)
        del arg58_1
        del buf102
        buf104 = reinterpret_tensor(buf103, (1, 1024, 512), (524288, 512, 1), 0); del buf103  # reuse
        buf180 = empty_strided_cuda((1, 1024, 512), (524288, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_3, h_diag], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_28.run(buf104, buf68, buf96, arg53_1, arg59_1, arg60_1, arg61_1, buf180, 1024, 512, grid=grid(1024), stream=stream0)
        del arg53_1
        del arg59_1
        del arg60_1
        del arg61_1
        del buf104
        del buf68
        buf108 = empty_strided_cuda((1, 6, 131072), (786432, 131072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [row_p], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg64_1, (1, 6, 4), (24, 4, 1), 0), reinterpret_tensor(buf25, (1, 4, 131072), (524288, 131072, 1), 0), out=buf108)
        del arg64_1
        buf109 = empty_strided_cuda((1, 6, 131072), (786432, 131072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [col_p], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg65_1, (1, 6, 4), (24, 4, 1), 0), reinterpret_tensor(buf25, (1, 4, 131072), (524288, 131072, 1), 0), out=buf109)
        del arg65_1
        del buf25
        buf110 = empty_strided_cuda((6, 256, 1), (256, 1, 1536), torch.float32)
        buf111 = empty_strided_cuda((6, 256, 1), (256, 1, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_6], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_29.run(buf108, buf109, buf110, buf111, 1536, 512, grid=grid(1536), stream=stream0)
        buf117 = empty_strided_cuda((6, 32, 33, 4), (4224, 132, 4, 1), torch.float32)
        buf115 = reinterpret_tensor(buf117, (6, 32, 32, 4), (4224, 132, 4, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sub_3], Original ATen: [aten.mean]
        triton_per_fused_mean_30.run(arg30_1, buf115, 24576, 64, grid=grid(24576), stream=stream0)
        buf116 = reinterpret_tensor(buf117, (6, 32, 1, 4), (4224, 132, 4, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [c_3, out_5], Original ATen: [aten.mean, aten.cat]
        triton_per_fused_cat_mean_31.run(arg30_1, buf116, 768, 8, grid=grid(768), stream=stream0)
        del arg30_1
        buf118 = empty_strided_cuda((6, 32, 32), (1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sub_2], Original ATen: [aten.amax]
        triton_per_fused_amax_32.run(arg31_1, buf118, 6144, 64, grid=grid(6144), stream=stream0)
        del buf115
        del buf116
        buf119 = empty_strided_cuda((6, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [c_2], Original ATen: [aten.amax]
        triton_per_fused_amax_33.run(arg31_1, buf119, 192, 8, grid=grid(192), stream=stream0)
        del arg31_1
        buf120 = empty_strided_cuda((6, 1, 32, 512), (16384, 98304, 512, 1), torch.float32)
        buf124 = empty_strided_cuda((6, 1, 33, 512), (16896, 1, 512, 1), torch.float32)
        buf122 = reinterpret_tensor(buf124, (6, 1, 32, 512), (16896, 1, 512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_attn_2], Original ATen: [aten.mean]
        triton_per_fused_mean_34.run(buf108, buf109, buf110, buf111, arg66_1, arg67_1, buf120, buf122, 98304, 8, grid=grid(98304), stream=stream0)
        del arg66_1
        del arg67_1
        buf123 = reinterpret_tensor(buf124, (6, 1, 1, 512), (16896, 1, 512, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [x_attn_2, block_node_2], Original ATen: [aten.mean]
        triton_per_fused_mean_35.run(buf120, buf123, 3072, 32, grid=grid(3072), stream=stream0)
        del buf122
        del buf123
        buf125 = empty_strided_cuda((198, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (198, 512), (512, 1), 0), reinterpret_tensor(arg68_1, (512, 1536), (1, 512), 0), out=buf125)
        del arg68_1
        buf126 = reinterpret_tensor(buf120, (6, 8, 32, 64, 1, 1), (16384, 2048, 64, 1, 1, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [einsum_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_36.run(buf125, arg69_1, buf126, 98304, grid=grid(98304), stream=stream0)
        buf127 = empty_strided_cuda((48, 64, 36), (2304, 36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_37.run(buf125, arg69_1, buf127, 110592, grid=grid(110592), stream=stream0)
        buf128 = empty_strided_cuda((48, 32, 36), (1152, 36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf126, (48, 32, 64), (2048, 64, 1), 0), buf127, out=buf128)
        buf129 = empty_strided_cuda((6, 1, 32, 33, 4), (4224, 25344, 132, 4, 1), torch.float32)
        buf163 = empty_strided_cuda((6, 1, 32, 33, 4), (4224, 25344, 132, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_8, setitem_9, setitem_12, setitem_13], Original ATen: [aten.lift_fresh, aten.fill]
        triton_poi_fused_fill_lift_fresh_38.run(buf117, buf129, buf163, 25344, grid=grid(25344), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_8, setitem_9, setitem_10], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
        triton_poi_fused_fill_index_put_lift_fresh_39.run(buf129, 768, grid=grid(768), stream=stream0)
        buf131 = empty_strided_cuda((6, 1, 32, 33), (1056, 1056, 33, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_11], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_40.run(buf129, buf131, 6336, grid=grid(6336), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_11], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_41.run(buf131, 192, grid=grid(192), stream=stream0)
        buf133 = empty_strided_cuda((6, 1, 32, 33, 8), (8448, 50688, 33, 1, 1056), torch.float32)
        buf134 = reinterpret_tensor(buf111, (6, 1, 32, 1, 8), (256, 1536, 1, 1536, 32), 0); del buf111  # reuse
        buf135 = reinterpret_tensor(buf110, (6, 1, 32, 1, 8), (256, 1536, 1, 1536, 32), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [eq_4, scores_8, scores_6, scores_7, attn_probs_2], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_eq_masked_fill_mul_42.run(buf118, buf119, buf128, buf131, buf129, buf133, buf134, buf135, 1536, 33, grid=grid(1536), stream=stream0)
        buf136 = reinterpret_tensor(buf117, (6, 1, 32, 33, 4), (4224, 1, 132, 4, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_43.run(buf131, buf129, buf136, 25344, grid=grid(25344), stream=stream0)
        del buf129
        buf137 = empty_strided_cuda((6336, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf136, (6336, 4), (4, 1), 0), reinterpret_tensor(arg70_1, (4, 8), (1, 4), 0), out=buf137)
        del arg70_1
        buf138 = reinterpret_tensor(buf133, (6, 8, 32, 33, 1, 1), (8448, 1056, 33, 1, 1, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [x_out_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf138, buf134, buf135, buf118, buf119, buf137, arg71_1, 48, 1056, grid=grid(48, 1056), stream=stream0)
        del arg71_1
        buf139 = reinterpret_tensor(buf124, (6, 8, 33, 1, 64, 1), (16896, 2112, 64, 64, 1, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [x_out_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_45.run(buf125, arg69_1, buf139, 101376, grid=grid(101376), stream=stream0)
        del arg69_1
        buf140 = reinterpret_tensor(buf126, (48, 32, 64), (2048, 64, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [x_out_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf138, (48, 32, 33), (1056, 33, 1), 0), reinterpret_tensor(buf139, (48, 33, 64), (2112, 64, 1), 0), out=buf140)
        buf141 = empty_strided_cuda((6, 1, 32, 8, 64), (16384, 1, 512, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_out_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_46.run(buf140, buf141, 98304, grid=grid(98304), stream=stream0)
        buf142 = reinterpret_tensor(buf140, (192, 512), (512, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf141, (192, 512), (512, 1), 0), reinterpret_tensor(arg72_1, (512, 512), (1, 512), 0), out=buf142)
        del arg72_1
        buf146 = empty_strided_cuda((6, 256, 512), (131072, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, layer_norm_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf108, buf109, buf142, arg73_1, arg74_1, arg75_1, buf146, 1536, 512, grid=grid(1536), stream=stream0)
        del arg74_1
        del arg75_1
        buf147 = empty_strided_cuda((1536, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf146, (1536, 512), (512, 1), 0), reinterpret_tensor(arg76_1, (512, 2048), (1, 512), 0), out=buf147)
        del arg76_1
        buf148 = reinterpret_tensor(buf147, (6, 256, 2048), (524288, 2048, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf148, arg77_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg77_1
        buf149 = reinterpret_tensor(buf146, (1536, 512), (512, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf148, (1536, 2048), (2048, 1), 0), reinterpret_tensor(arg78_1, (2048, 512), (1, 2048), 0), out=buf149)
        del arg78_1
        buf150 = reinterpret_tensor(buf149, (6, 256, 512), (131072, 512, 1), 0); del buf149  # reuse
        buf151 = reinterpret_tensor(buf135, (6, 256, 1), (256, 1, 1536), 0); del buf135  # reuse
        buf152 = reinterpret_tensor(buf134, (6, 256, 1), (256, 1, 1536), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [x_4, x_5, layer_norm_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_49.run(buf150, buf108, buf109, buf142, arg73_1, arg79_1, buf151, buf152, 1536, 512, grid=grid(1536), stream=stream0)
        del arg73_1
        del arg79_1
        del buf108
        buf154 = reinterpret_tensor(buf142, (6, 1, 32, 512), (16384, 98304, 512, 1), 0); del buf142  # reuse
        buf158 = reinterpret_tensor(buf139, (6, 1, 33, 512), (16896, 1, 512, 1), 0); del buf139  # reuse
        buf156 = reinterpret_tensor(buf158, (6, 1, 32, 512), (16896, 1, 512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_attn_3], Original ATen: [aten.mean]
        triton_per_fused_mean_50.run(buf150, buf151, buf152, arg80_1, arg81_1, buf154, buf156, 98304, 8, grid=grid(98304), stream=stream0)
        del arg80_1
        del arg81_1
        buf157 = reinterpret_tensor(buf158, (6, 1, 1, 512), (16896, 1, 512, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [x_attn_3, block_node_3], Original ATen: [aten.mean]
        triton_per_fused_mean_35.run(buf154, buf157, 3072, 32, grid=grid(3072), stream=stream0)
        del buf156
        del buf157
        buf159 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf158, (198, 512), (512, 1), 0), reinterpret_tensor(arg82_1, (512, 1536), (1, 512), 0), out=buf159)
        del arg82_1
        buf160 = reinterpret_tensor(buf154, (6, 8, 32, 64, 1, 1), (16384, 2048, 64, 1, 1, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [einsum_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_36.run(buf159, arg83_1, buf160, 98304, grid=grid(98304), stream=stream0)
        buf161 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_37.run(buf159, arg83_1, buf161, 110592, grid=grid(110592), stream=stream0)
        buf162 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf160, (48, 32, 64), (2048, 64, 1), 0), buf161, out=buf162)
        del buf161
        # Topologically Sorted Source Nodes: [setitem_12, setitem_13, setitem_14], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
        triton_poi_fused_fill_index_put_lift_fresh_39.run(buf163, 768, grid=grid(768), stream=stream0)
        buf165 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [setitem_15], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_40.run(buf163, buf165, 6336, grid=grid(6336), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_15], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_41.run(buf165, 192, grid=grid(192), stream=stream0)
        buf167 = reinterpret_tensor(buf138, (6, 1, 32, 33, 8), (8448, 50688, 33, 1, 1056), 0); del buf138  # reuse
        buf168 = reinterpret_tensor(buf152, (6, 1, 32, 1, 8), (256, 1536, 1, 1536, 32), 0); del buf152  # reuse
        buf169 = reinterpret_tensor(buf151, (6, 1, 32, 1, 8), (256, 1536, 1, 1536, 32), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [eq_6, scores_11, scores_9, scores_10, attn_probs_3], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_eq_masked_fill_mul_42.run(buf118, buf119, buf162, buf165, buf163, buf167, buf168, buf169, 1536, 33, grid=grid(1536), stream=stream0)
        del buf162
        buf170 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_43.run(buf165, buf163, buf170, 25344, grid=grid(25344), stream=stream0)
        del buf163
        del buf165
        buf171 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf170, (6336, 4), (4, 1), 0), reinterpret_tensor(arg84_1, (4, 8), (1, 4), 0), out=buf171)
        del arg84_1
        del buf170
        buf172 = reinterpret_tensor(buf167, (6, 8, 32, 33, 1, 1), (8448, 1056, 33, 1, 1, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [x_out_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf172, buf168, buf169, buf118, buf119, buf171, arg85_1, 48, 1056, grid=grid(48, 1056), stream=stream0)
        del arg85_1
        del buf119
        del buf168
        del buf169
        del buf171
        buf173 = reinterpret_tensor(buf158, (6, 8, 33, 1, 64, 1), (16896, 2112, 64, 64, 1, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [x_out_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_45.run(buf159, arg83_1, buf173, 101376, grid=grid(101376), stream=stream0)
        del arg83_1
        del buf159
        buf174 = reinterpret_tensor(buf160, (48, 32, 64), (2048, 64, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [x_out_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf172, (48, 32, 33), (1056, 33, 1), 0), reinterpret_tensor(buf173, (48, 33, 64), (2112, 64, 1), 0), out=buf174)
        del buf172
        del buf173
        buf175 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [x_out_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_46.run(buf174, buf175, 98304, grid=grid(98304), stream=stream0)
        buf176 = reinterpret_tensor(buf174, (192, 512), (512, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf175, (192, 512), (512, 1), 0), reinterpret_tensor(arg86_1, (512, 512), (1, 512), 0), out=buf176)
        del arg86_1
        buf185 = reinterpret_tensor(buf109, (6, 256, 512), (131072, 512, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_6, layer_norm_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_51.run(buf150, buf176, arg87_1, arg88_1, arg89_1, buf185, 1536, 512, grid=grid(1536), stream=stream0)
        del arg88_1
        del arg89_1
        buf181 = reinterpret_tensor(buf96, (1, 4, 32, 512), (65536, 16384, 512, 1), 0); del buf96  # reuse
        buf182 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [h_a], Original ATen: [aten.mean]
        triton_per_fused_mean_52.run(buf182, buf180, 65536, 8, grid=grid(65536), stream=stream0)
        buf183 = reinterpret_tensor(buf34, (128, 32), (32, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [u_leaf], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg63_1, reinterpret_tensor(buf182, (128, 512), (512, 1), 0), reinterpret_tensor(arg62_1, (512, 32), (1, 512), 0), alpha=1, beta=1, out=buf183)
        del arg62_1
        del arg63_1
        del buf182
        buf184 = empty_strided_cuda((4, 32, 32), (1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf183, (4, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf183, (4, 32, 32), (1024, 1, 32), 0), out=buf184)
        del buf183
        buf186 = reinterpret_tensor(buf148, (1536, 2048), (2048, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf185, (1536, 512), (512, 1), 0), reinterpret_tensor(arg90_1, (512, 2048), (1, 512), 0), out=buf186)
        del arg90_1
        buf187 = reinterpret_tensor(buf186, (6, 256, 2048), (524288, 2048, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf187, arg91_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg91_1
        buf188 = reinterpret_tensor(buf185, (1536, 512), (512, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (1536, 2048), (2048, 1), 0), reinterpret_tensor(arg92_1, (2048, 512), (1, 2048), 0), out=buf188)
        del arg92_1
        del buf187
        buf189 = reinterpret_tensor(buf175, (6, 1, 32, 512), (16384, 98304, 512, 1), 0); del buf175  # reuse
        buf190 = reinterpret_tensor(buf189, (6, 1, 32, 512), (16384, 1, 512, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [h_a_1], Original ATen: [aten.mean]
        triton_per_fused_mean_53.run(buf190, buf150, buf176, arg87_1, buf188, arg93_1, 98304, 8, grid=grid(98304), stream=stream0)
        del arg87_1
        del arg93_1
        del buf150
        del buf176
        del buf188
        buf191 = reinterpret_tensor(buf118, (192, 32), (32, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [u_leaf_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg95_1, reinterpret_tensor(buf190, (192, 512), (512, 1), 0), reinterpret_tensor(arg94_1, (512, 32), (1, 512), 0), alpha=1, beta=1, out=buf191)
        del arg94_1
        del arg95_1
        buf192 = empty_strided_cuda((192, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_leaf], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg97_1, reinterpret_tensor(buf190, (192, 512), (512, 1), 0), reinterpret_tensor(arg96_1, (512, 32), (1, 512), 0), alpha=1, beta=1, out=buf192)
        del arg96_1
        del arg97_1
        del buf190
        buf193 = empty_strided_cuda((6, 32, 32), (1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf191, (6, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf192, (6, 32, 32), (1024, 1, 32), 0), out=buf193)
        del buf191
        del buf192
        buf194 = empty_strided_cuda((1024, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [node_U], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg101_1, reinterpret_tensor(buf180, (1024, 512), (512, 1), 0), reinterpret_tensor(arg100_1, (512, 32), (1, 512), 0), alpha=1, beta=1, out=buf194)
        del arg100_1
        del arg101_1
        buf195 = empty_strided_cuda((1024, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [node_V], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg103_1, reinterpret_tensor(buf180, (1024, 512), (512, 1), 0), reinterpret_tensor(arg102_1, (512, 32), (1, 512), 0), alpha=1, beta=1, out=buf195)
        del arg102_1
        del arg103_1
        buf196 = reinterpret_tensor(buf87, (1024, 1), (1, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf180, (1024, 512), (512, 1), 0), reinterpret_tensor(arg98_1, (512, 1), (1, 512), 0), out=buf196)
        del arg98_1
        del buf180
        buf197 = reinterpret_tensor(buf196, (1, 4, 256, 1), (1024, 256, 1, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [sigmoid], Original ATen: [aten.sigmoid]
        triton_poi_fused_sigmoid_54.run(buf197, arg99_1, 1024, grid=grid(1024), stream=stream0)
        del arg99_1
        buf198 = empty_strided_cuda((1, 76800), (76800, 1), torch.float32)
        # Topologically Sorted Source Nodes: [packed_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_55.run(buf184, buf193, buf194, buf195, buf197, buf198, 76800, grid=grid(76800), stream=stream0)
        del buf184
        del buf193
        del buf194
        del buf195
    return (buf198, reinterpret_tensor(buf197, (1, 4, 256), (1024, 256, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1024, 9), (67716, 9, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, 18), (18, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((2, 6737), (6737, 1), device='cuda:0', dtype=torch.int64)
    arg7_1 = rand_strided((6737, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((4, 256, 257, 4), (263168, 1028, 4, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((4, 256, 257), (65792, 257, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((6, 256, 257, 4), (263168, 1028, 4, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((6, 256, 257), (65792, 257, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((8, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((8, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((6, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((6, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((8, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((8, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
