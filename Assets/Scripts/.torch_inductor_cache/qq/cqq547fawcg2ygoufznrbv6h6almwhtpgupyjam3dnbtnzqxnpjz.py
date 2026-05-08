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
#   %add_tensor_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_19, %arg15_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze, %add_tensor_19), kwargs = {})
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ni/cnignjbmoquup42u5rshzu4p5qg6hyrr6kshtcmbg62zzcwzm4ju.py
# Topologically Sorted Source Nodes: [layer_norm_1, kv], Original ATen: [aten.native_layer_norm, aten.cat]
# Source node to ATen node mapping:
#   kv => cat_3
#   layer_norm_1 => add_7, add_8, mul_13, mul_14, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %getitem_3), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %arg32_1), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %arg33_1), kwargs = {})
#   %cat_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_8, %mean], 2), kwargs = {})
triton_per_fused_cat_native_layer_norm_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_6', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, rnumel):
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
    x2 = xindex % 32
    x3 = (xindex // 32)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), None)
    tmp21 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
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
    tmp14 = tmp0 - tmp8
    tmp15 = 512.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp24, None)
    tl.store(out_ptr3 + (r1 + (512*x2) + (16896*x3)), tmp24, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/kw/ckwhnpzpkzyqptbchgzy37lmsg5badsmk6kfj3l3fb5go7yavfxc.py
# Topologically Sorted Source Nodes: [block_node], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   block_node => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_8, [2], True), kwargs = {})
triton_per_fused_mean_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = 32.0
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr1 + (x0 + (16896*x1)), tmp5, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/wx/cwxp5beecg44du7oahu7xrzojr7vp3verkurj55qsryesstoeayd.py
# Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   einsum => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_15,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 32
    x2 = (xindex // 2048) % 8
    x3 = (xindex // 16384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1536*x1) + (49152*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/fz/cfzhpjnexmyugu2kivncgh3wxr5ohxkqmt7x63e4c2544aaf5bii.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_11 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_17, %full_default_37], 2), kwargs = {})
triton_poi_fused_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/lo/clove3fj4tqrmh4hz44gihjtqkfjpm3qd54nrsny6dm2ioxytzj6.py
# Topologically Sorted Source Nodes: [setitem, setitem_1, setitem_4, setitem_5], Original ATen: [aten.lift_fresh, aten.fill]
# Source node to ATen node mapping:
#   setitem => copy, full_default_2
#   setitem_1 => copy_1, full_default_3
#   setitem_4 => copy_2, full_default_8
#   setitem_5 => copy_3, full_default_9
# Graph fragment:
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_4, %full_default_2), kwargs = {})
#   %select_scatter_default : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%expand_3, %copy, 3, 32), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_10, %full_default_3), kwargs = {})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int, %copy_1, 3, 3), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default, %select_scatter_default_1, 3, 32), kwargs = {})
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_2 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_17, %full_default_8), kwargs = {})
#   %select_scatter_default_8 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%expand_4, %copy_2, 3, 32), kwargs = {})
#   %full_default_9 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_3 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_23, %full_default_9), kwargs = {})
#   %select_scatter_default_9 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_2, %copy_3, 3, 3), kwargs = {})
#   %select_scatter_default_10 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_8, %select_scatter_default_9, 3, 32), kwargs = {})
triton_poi_fused_fill_lift_fresh_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_fill_lift_fresh_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 135168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 4) % 33
    x0 = xindex % 4
    x2 = (xindex // 132)
    x3 = xindex
    tmp7 = tl.load(in_ptr0 + (128 + x0 + (132*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (x3), None)
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
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp14, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/2d/c2dzjkgnhyow6afooy6c2mh5oytxzc3dqffeyqrciveeebs6qjji.py
# Topologically Sorted Source Nodes: [setitem, setitem_1, setitem_2], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
# Source node to ATen node mapping:
#   setitem => copy, full_default_2
#   setitem_1 => copy_1, full_default_3
#   setitem_2 => full_default_4, index_put_2
# Graph fragment:
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_4, %full_default_2), kwargs = {})
#   %select_scatter_default : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%expand_3, %copy, 3, 32), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_10, %full_default_3), kwargs = {})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int, %copy_1, 3, 3), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default, %select_scatter_default_1, 3, 32), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_2 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%select_scatter_default_2, [None, None, %iota, %iota], %full_default_4), kwargs = {})
triton_poi_fused_fill_index_put_lift_fresh_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_fill_index_put_lift_fresh_11', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 4
    x1 = (xindex // 4) % 32
    x2 = (xindex // 128)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (136*x1) + (4224*x2)), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/55/c55baufcmiitnaa4jqone2nub4fyvlafscqohrqlcblkjjyisaq4.py
# Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_3 => full_default_5, index_put_3
# Graph fragment:
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_3 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_15, [None, None, %iota, %iota], %full_default_5), kwargs = {})
triton_poi_fused_index_put_lift_fresh_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_12', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + (4*x0)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/27/c27xw3rkddty7bbdtq6gppjzkda2ycplrpoblunhqi7lqd5pq54t.py
# Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_3 => full_default_5, index_put_3
# Graph fragment:
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_3 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_15, [None, None, %iota, %iota], %full_default_5), kwargs = {})
triton_poi_fused_index_put_lift_fresh_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_13', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp0 = 1.0
    tl.store(out_ptr0 + ((34*x0) + (1056*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/7u/c7u4cnudc7vbp43xss2clof4sxahxyloj3gx2gs4oct6wk2wzhgq.py
# Topologically Sorted Source Nodes: [eq, scores_2, scores, scores_1, attn_probs], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_probs => amax, exp, sub_2, sum_1
#   eq => eq
#   scores => mul_15
#   scores_1 => add_9
#   scores_2 => clone_3, full_default_6, where
# Graph fragment:
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%unsqueeze_11, 0), kwargs = {})
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, 0.125), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %slice_48), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default_6, %add_9), kwargs = {})
#   %clone_3 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%where,), kwargs = {memory_format: torch.contiguous_format})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_3, [3], True), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_3, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [3], True), kwargs = {})
triton_per_fused__softmax_add_eq_masked_fill_mul_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_eq_masked_fill_mul_14', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 33
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 32
    x2 = (xindex // 256)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (33*x0) + (1056*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (r3 + (36*x4)), rmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r3 + (33*x0) + (1056*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (3 + (4*r3) + (132*x0) + (4224*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tmp4 = 0.125
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
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tl.store(out_ptr1 + (r3 + (33*x4)), tmp19, rmask)
    tl.store(out_ptr2 + (x4), tmp23, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/lw/clwzml2akdy7cmr674mrc6f3jme2mo2b4cunbr35njwuw3rs6uu7.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %select_scatter_default_3 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%index_put_2, %index_put_3, 4, 3), kwargs = {})
triton_poi_fused_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_15', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 135168
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/7w/c7waxdyc2gndyqeamwbhubx7nyivl67bgelxvskewy7uuma5of5l.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_9 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_23, %full_default_35], 2), kwargs = {})
triton_poi_fused_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[256, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_16', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
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
    tmp3 = tl.load(in_ptr2 + (x3 + (1056*(y0 // 8))), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + ((8*x3) + (8448*(y0 // 8)) + (y0 % 8)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0 % 8), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 / tmp1
    tmp4 = 0.0
    tmp5 = tmp3 == tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = tl.where(tmp5, tmp4, tmp8)
    tmp10 = tmp2 + tmp9
    tl.store(out_ptr0 + (x1 + (36*x2) + (1152*y0)), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/xx/cxxbtp5qn77io5xta745jqbqwvdmxs3qig6wkvtlq7zu6j3e35wd.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_35 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([256, 32, 3], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(1,), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_17', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 3
    x1 = (xindex // 3)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (36*x1)), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/gy/cgycuegtqb7wpmmb4qxoir6zihzcv5xzvfmer3y6np7etvi6xx65.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_10 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_24, %full_default_36], 1), kwargs = {})
triton_poi_fused_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_18', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/2b/c2bmrvyxh7b7kykzev5f3qhzrmntsynao6lblcujqpfwi6toy4y4.py
# Topologically Sorted Source Nodes: [x_out_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_out_1 => clone_6
# Graph fragment:
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_26,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_19', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/gg/cgg3lzxbt426qqd3udnhji7vbs5xdkoy5aal2wgttuyvgpbybmtb.py
# Topologically Sorted Source Nodes: [x, layer_norm_2, kv_1], Original ATen: [aten.add, aten.native_layer_norm, aten.cat]
# Source node to ATen node mapping:
#   kv_1 => cat_4
#   layer_norm_2 => add_12, add_13, mul_16, mul_17, rsqrt_2, sub_3, var_mean_2
#   x => add_11
# Graph fragment:
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %view_30), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_11, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_11, %getitem_5), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %arg40_1), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %arg41_1), kwargs = {})
#   %cat_4 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_31, %mean_1], 2), kwargs = {})
triton_per_fused_add_cat_native_layer_norm_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_20', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
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
    x2 = xindex % 32
    x3 = (xindex // 32)
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
    tl.store(out_ptr3 + (r1 + (512*x2) + (16896*x3)), tmp28, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/pk/cpkmlu7k6nzb7h7k724tss6ghc7o4c55qhs46yzrc3fuq3vu2ppm.py
# Topologically Sorted Source Nodes: [layer_norm_3, kv_2], Original ATen: [aten.native_layer_norm, aten.cat]
# Source node to ATen node mapping:
#   kv_2 => cat_5
#   layer_norm_3 => add_18, add_19, mul_19, mul_20, rsqrt_3, sub_5, var_mean_3
# Graph fragment:
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_69, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_69, %getitem_7), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_3), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %arg52_1), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %arg53_1), kwargs = {})
#   %cat_5 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_70, %mean_2], 2), kwargs = {})
triton_per_fused_cat_native_layer_norm_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_21', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 3488
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
    x2 = xindex % 32
    x3 = (xindex // 32)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), None)
    tmp23 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
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
    tmp16 = tmp2 - tmp10
    tmp17 = 512.0
    tmp18 = tmp15 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp16 * tmp21
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp26, None)
    tl.store(out_ptr3 + (r1 + (512*x2) + (16896*x3)), tmp26, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/4o/c4oaur7qh6zx2zpstv7zp25s72v466jyldsuw5g7r427lzggo42a.py
# Topologically Sorted Source Nodes: [block_node_2], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   block_node_2 => mean_2
# Graph fragment:
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_70, [2], True), kwargs = {})
triton_per_fused_mean_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_22', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 55808
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 32.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x0 + (16896*x1)), tmp6, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/r4/cr4wittykziydwjwndy5ia7cebpynqhavzqjxi7dw3iwjd2fgzmi.py
# Topologically Sorted Source Nodes: [einsum_6], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   einsum_6 => clone_14
# Graph fragment:
#   %clone_14 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_55,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_23', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1785856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 32
    x2 = (xindex // 2048) % 8
    x3 = (xindex // 16384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1536*x1) + (49152*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ll/cllwsmrp4gpa3gfnoz27m3wrnbgzhhu6wdhlaqc5f5da7d7eyoj7.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_5 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_79, %full_default_31], 2), kwargs = {})
triton_poi_fused_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_24', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2009088
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/bf/cbfcyk24wqko774hcb5ta6fsd7hxvgr5njzl3crmxdazyh72mmut.py
# Topologically Sorted Source Nodes: [setitem_8, setitem_9, setitem_12, setitem_13], Original ATen: [aten.lift_fresh, aten.fill]
# Source node to ATen node mapping:
#   setitem_12 => copy_6, full_default_20
#   setitem_13 => copy_7, full_default_21
#   setitem_8 => copy_4, full_default_14
#   setitem_9 => copy_5, full_default_15
# Graph fragment:
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_4 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_30, %full_default_14), kwargs = {})
#   %select_scatter_default_4 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%view_7, %copy_4, 3, 32), kwargs = {})
#   %full_default_15 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_5 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_36, %full_default_15), kwargs = {})
#   %select_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_1, %copy_5, 3, 3), kwargs = {})
#   %select_scatter_default_6 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_4, %select_scatter_default_5, 3, 32), kwargs = {})
#   %full_default_20 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_6 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_43, %full_default_20), kwargs = {})
#   %select_scatter_default_12 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%view_7, %copy_6, 3, 32), kwargs = {})
#   %full_default_21 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_7 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_49, %full_default_21), kwargs = {})
#   %select_scatter_default_13 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_3, %copy_7, 3, 3), kwargs = {})
#   %select_scatter_default_14 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_12, %select_scatter_default_13, 3, 32), kwargs = {})
triton_poi_fused_fill_lift_fresh_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_fill_lift_fresh_25', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 460416
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/uu/cuu3zgfrzjvf4pdmswbulwlfisr6lkuadhqumjnsyy2hrrwvk6vr.py
# Topologically Sorted Source Nodes: [setitem_8, setitem_9, setitem_10], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
# Source node to ATen node mapping:
#   setitem_10 => full_default_16, index_put_6
#   setitem_8 => copy_4, full_default_14
#   setitem_9 => copy_5, full_default_15
# Graph fragment:
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_4 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_30, %full_default_14), kwargs = {})
#   %select_scatter_default_4 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%view_7, %copy_4, 3, 32), kwargs = {})
#   %full_default_15 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_5 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_36, %full_default_15), kwargs = {})
#   %select_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_1, %copy_5, 3, 3), kwargs = {})
#   %select_scatter_default_6 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_4, %select_scatter_default_5, 3, 32), kwargs = {})
#   %full_default_16 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_6 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%select_scatter_default_6, [None, None, %iota_2, %iota_2], %full_default_16), kwargs = {})
triton_poi_fused_fill_index_put_lift_fresh_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_fill_index_put_lift_fresh_26', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = (xindex // 4) % 32
    x2 = (xindex // 128)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (136*x1) + (4224*x2)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/3h/c3hof7atit5tlrwuymecpx46elxcfukrj4gyahx6oltjreqeygm3.py
# Topologically Sorted Source Nodes: [setitem_11], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_11 => full_default_17, index_put_7
# Graph fragment:
#   %full_default_17 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_7 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_41, [None, None, %iota_2, %iota_2], %full_default_17), kwargs = {})
triton_poi_fused_index_put_lift_fresh_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_27', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 115104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + (4*x0)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/td/ctdbjn6lx6kgvhl6fwxncs2froiimugnsgk4l7ejikfcvptrzkpr.py
# Topologically Sorted Source Nodes: [setitem_11], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_11 => full_default_17, index_put_7
# Graph fragment:
#   %full_default_17 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_7 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_41, [None, None, %iota_2, %iota_2], %full_default_17), kwargs = {})
triton_poi_fused_index_put_lift_fresh_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_28', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp0 = 1.0
    tl.store(out_ptr0 + ((34*x0) + (1056*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/xj/cxjorlsqhpv3e6eoxwf2mydgrft25j323wretqjnkm3nkrbqddef.py
# Topologically Sorted Source Nodes: [eq_4, scores_8, scores_6, scores_7, attn_probs_2], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_probs_2 => amax_2, exp_2, sub_6
#   eq_4 => eq_4
#   scores_6 => mul_21
#   scores_7 => add_20
#   scores_8 => clone_17, full_default_18, where_4
# Graph fragment:
#   %eq_4 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%unsqueeze_31, 0), kwargs = {})
#   %full_default_18 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_81, 0.125), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %slice_142), kwargs = {})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_4, %full_default_18, %add_20), kwargs = {})
#   %clone_17 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%where_4,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_2 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_17, [3], True), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_17, %amax_2), kwargs = {})
#   %exp_2 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_6,), kwargs = {})
triton_per_fused__softmax_add_eq_masked_fill_mul_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_eq_masked_fill_mul_29', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 27904
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
    tmp0 = tl.load(in_ptr0 + (r3 + (33*x0) + (1056*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (r3 + (36*x4)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r3 + (33*x0) + (1056*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (3 + (4*r3) + (132*x0) + (4224*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full([1, 1], 3, tl.int32)
    tmp7 = tmp6 == tmp6
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp5 + tmp10
    tmp12 = float("-inf")
    tmp13 = tl.where(tmp2, tmp12, tmp11)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, float("-inf"))
    tmp17 = triton_helpers.max2(tmp16, 1)[:, None]
    tmp18 = tmp13 - tmp17
    tmp19 = tl_math.exp(tmp18)
    tl.store(out_ptr1 + (r3 + (33*x4)), tmp19, rmask & xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/pl/cpld6giwte62nv3zhepmvkans5aytqp2psyo2xicei6qywgnlzcp.py
# Topologically Sorted Source Nodes: [attn_probs_2], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_probs_2 => sum_3
# Graph fragment:
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_2, [3], True), kwargs = {})
triton_per_fused__softmax_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_30', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 27904
    rnumel = 33
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 8
    x1 = (xindex // 8) % 32
    x2 = (xindex // 256)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (33*x1) + (1056*x0) + (8448*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/cz/cczvpmvfhgnqstkeiux5zpm2pycc5px7jhbqawlocna5ihspq2ty.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %select_scatter_default_7 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%index_put_6, %index_put_7, 4, 3), kwargs = {})
triton_poi_fused_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_31', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 460416
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/2w/c2w2gygdxninvlay2vmwvf7av3qbj3k7tmnxwzldzrkc7vzhlhhf.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_85, %full_default_29], 2), kwargs = {})
triton_poi_fused_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1024, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_32', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 872
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
    tmp1 = tl.load(in_ptr1 + ((8*x2) + (256*(y0 // 8)) + (y0 % 8)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3 + (1056*(y0 // 8))), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + ((8*x3) + (8448*(y0 // 8)) + (y0 % 8)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0 % 8), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 / tmp1
    tmp4 = 0.0
    tmp5 = tmp3 == tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = tl.where(tmp5, tmp4, tmp8)
    tmp10 = tmp2 + tmp9
    tl.store(out_ptr0 + (x1 + (36*x2) + (1152*y0)), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/y3/cy3ruiciwkn2fwm4swxjcac7sijvf5gy4lktdfziqhww5h53fhb2.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_29 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([872, 32, 3], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(1,), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_33', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 83712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x1 = (xindex // 3)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (36*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/g7/cg7v5a5udce5isrxkgaf7bxar64g4ddqcyzu47hn5u5dmy4x7pxy.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_4 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_86, %full_default_30], 1), kwargs = {})
triton_poi_fused_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_34', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2009088
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/uk/cukii7lxuqnb4goly4fhqlyjbxqaeaydfcxcbybgbltxew45rwpo.py
# Topologically Sorted Source Nodes: [x_out_9], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_out_9 => clone_20
# Graph fragment:
#   %clone_20 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_88,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_35', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1785856
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/7x/c7xnfgoflnhblpaygjmos4rke2y2c7vfog35phsnvcjjqgynhmge.py
# Topologically Sorted Source Nodes: [x_2, layer_norm_4, kv_3], Original ATen: [aten.add, aten.native_layer_norm, aten.cat]
# Source node to ATen node mapping:
#   kv_3 => cat_6
#   layer_norm_4 => add_23, add_24, mul_22, mul_23, rsqrt_4, sub_7, var_mean_4
#   x_2 => add_22
# Graph fragment:
#   %add_22 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_69, %view_92), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_22, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_22, %getitem_9), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_23,), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_4), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %arg60_1), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %arg61_1), kwargs = {})
#   %cat_6 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_93, %mean_3], 2), kwargs = {})
triton_per_fused_add_cat_native_layer_norm_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_36', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 3488
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
    x2 = xindex % 32
    x3 = (xindex // 32)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), None)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), None)
    tmp4 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp30, None)
    tl.store(out_ptr3 + (r1 + (512*x2) + (16896*x3)), tmp30, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/4t/c4tywv3hbs4xilvlvyz4odb3rrdkodr657myfsstwii434abc7ei.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x => add_11
#   x_1 => add_16
# Graph fragment:
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %view_30), kwargs = {})
#   %add_16 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_53), kwargs = {})
triton_poi_fused_add_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_37', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/xd/cxdwvxxnhwvvzokid32w4cka2sjyoa36oov3qh7fvwqx42sqfue4.py
# Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_2 => add_22
#   x_3 => add_27
# Graph fragment:
#   %add_22 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_69, %view_92), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_22, %view_115), kwargs = {})
triton_poi_fused_add_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_38', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1785856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (x2), None)
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/gk/cgkhc7m76cnmslj4q4mzjw2da2lisleaqnrxqncdcyk77vrlp5e7.py
# Topologically Sorted Source Nodes: [sigmoid], Original ATen: [aten.sigmoid]
# Source node to ATen node mapping:
#   sigmoid => sigmoid
# Graph fragment:
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_129,), kwargs = {})
triton_poi_fused_sigmoid_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_39', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/f7/cf7g527g4b6nrnurw43u2gnyrumh7df2twtpfmmegjg5tjxm2dse.py
# Topologically Sorted Source Nodes: [packed_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   packed_1 => cat_8
# Graph fragment:
#   %cat_8 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_7, %view_130], 1), kwargs = {})
triton_poi_fused_cat_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_40', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 145408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 144384, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 32768, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.load(in_ptr0 + (x0), tmp7, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp0 >= tmp5
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr1 + ((-32768) + x0), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.where(tmp6, tmp8, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1], 145408, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-144384) + x0), tmp15, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.where(tmp4, tmp14, tmp18)
    tl.store(out_ptr0 + (x0), tmp19, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1 = args
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
    assert_size_stride(arg28_1, (32, 32, 33, 4), (4224, 132, 4, 1))
    assert_size_stride(arg29_1, (32, 32, 33), (1056, 33, 1))
    assert_size_stride(arg30_1, (109, 32, 33, 4), (4224, 132, 4, 1))
    assert_size_stride(arg31_1, (109, 32, 33), (1056, 33, 1))
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
    assert_size_stride(arg42_1, (1536, 512), (512, 1))
    assert_size_stride(arg43_1, (1536, ), (1, ))
    assert_size_stride(arg44_1, (8, 4), (4, 1))
    assert_size_stride(arg45_1, (8, ), (1, ))
    assert_size_stride(arg46_1, (512, 512), (512, 1))
    assert_size_stride(arg47_1, (512, ), (1, ))
    assert_size_stride(arg48_1, (32, 512), (512, 1))
    assert_size_stride(arg49_1, (32, ), (1, ))
    assert_size_stride(arg50_1, (109, 32), (32, 1))
    assert_size_stride(arg51_1, (109, 32), (32, 1))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (512, ), (1, ))
    assert_size_stride(arg54_1, (1536, 512), (512, 1))
    assert_size_stride(arg55_1, (1536, ), (1, ))
    assert_size_stride(arg56_1, (8, 4), (4, 1))
    assert_size_stride(arg57_1, (8, ), (1, ))
    assert_size_stride(arg58_1, (512, 512), (512, 1))
    assert_size_stride(arg59_1, (512, ), (1, ))
    assert_size_stride(arg60_1, (512, ), (1, ))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (1536, 512), (512, 1))
    assert_size_stride(arg63_1, (1536, ), (1, ))
    assert_size_stride(arg64_1, (8, 4), (4, 1))
    assert_size_stride(arg65_1, (8, ), (1, ))
    assert_size_stride(arg66_1, (512, 512), (512, 1))
    assert_size_stride(arg67_1, (512, ), (1, ))
    assert_size_stride(arg68_1, (32, 512), (512, 1))
    assert_size_stride(arg69_1, (32, ), (1, ))
    assert_size_stride(arg70_1, (32, 512), (512, 1))
    assert_size_stride(arg71_1, (32, ), (1, ))
    assert_size_stride(arg72_1, (1, 512), (512, 1))
    assert_size_stride(arg73_1, (1, ), (1, ))
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
        buf29 = buf24; del buf24  # reuse
        buf34 = empty_strided_cuda((1, 32, 33, 512), (540672, 16896, 512, 1), torch.float32)
        buf32 = reinterpret_tensor(buf34, (1, 32, 32, 512), (540672, 16896, 512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [layer_norm_1, kv], Original ATen: [aten.native_layer_norm, aten.cat]
        triton_per_fused_cat_native_layer_norm_6.run(buf25, arg32_1, arg33_1, buf29, buf32, 1024, 512, grid=grid(1024), stream=stream0)
        del arg32_1
        del arg33_1
        buf30 = empty_strided_cuda((1024, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (1024, 512), (512, 1), 0), reinterpret_tensor(arg34_1, (512, 1536), (1, 512), 0), out=buf30)
        buf33 = reinterpret_tensor(buf34, (1, 32, 1, 512), (540672, 16896, 512, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [block_node], Original ATen: [aten.mean]
        triton_per_fused_mean_7.run(buf29, buf33, 16384, 32, grid=grid(16384), stream=stream0)
        del buf32
        del buf33
        buf35 = empty_strided_cuda((1056, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf34, (1056, 512), (512, 1), 0), reinterpret_tensor(arg34_1, (512, 1536), (1, 512), 0), out=buf35)
        del arg34_1
        buf36 = reinterpret_tensor(buf29, (32, 8, 32, 64, 1, 1), (16384, 2048, 64, 1, 1, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf30, arg35_1, buf36, 524288, grid=grid(524288), stream=stream0)
        buf37 = empty_strided_cuda((256, 64, 36), (2304, 36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf35, arg35_1, buf37, 589824, grid=grid(589824), stream=stream0)
        buf38 = empty_strided_cuda((256, 32, 36), (1152, 36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf36, (256, 32, 64), (2048, 64, 1), 0), buf37, out=buf38)
        buf39 = empty_strided_cuda((1, 32, 32, 33, 4), (135168, 4224, 132, 4, 1), torch.float32)
        buf102 = empty_strided_cuda((1, 32, 32, 33, 4), (135168, 4224, 132, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem, setitem_1, setitem_4, setitem_5], Original ATen: [aten.lift_fresh, aten.fill]
        triton_poi_fused_fill_lift_fresh_10.run(arg28_1, buf39, buf102, 135168, grid=grid(135168), stream=stream0)
        del arg28_1
        # Topologically Sorted Source Nodes: [setitem, setitem_1, setitem_2], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
        triton_poi_fused_fill_index_put_lift_fresh_11.run(buf39, 4096, grid=grid(4096), stream=stream0)
        buf41 = empty_strided_cuda((1, 32, 32, 33), (33792, 1056, 33, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_12.run(buf39, buf41, 33792, grid=grid(33792), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_13.run(buf41, 1024, grid=grid(1024), stream=stream0)
        buf44 = empty_strided_cuda((1, 32, 32, 33, 8), (270336, 8448, 33, 1, 1056), torch.float32)
        buf45 = empty_strided_cuda((1, 32, 32, 1, 8), (8192, 256, 1, 8192, 32), torch.float32)
        # Topologically Sorted Source Nodes: [eq, scores_2, scores, scores_1, attn_probs], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_eq_masked_fill_mul_14.run(arg29_1, buf38, buf41, buf39, buf44, buf45, 8192, 33, grid=grid(8192), stream=stream0)
        buf46 = empty_strided_cuda((1, 32, 32, 33, 4), (135168, 4224, 132, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_15.run(buf41, buf39, buf46, 135168, grid=grid(135168), stream=stream0)
        del buf39
        buf47 = empty_strided_cuda((33792, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (33792, 4), (4, 1), 0), reinterpret_tensor(arg36_1, (4, 8), (1, 4), 0), out=buf47)
        del arg36_1
        buf50 = buf38; del buf38  # reuse
        buf48 = reinterpret_tensor(buf50, (256, 32, 33), (1152, 36, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(buf44, buf45, arg29_1, buf47, arg37_1, buf48, 256, 1056, grid=grid(256, 1056), stream=stream0)
        del arg37_1
        buf49 = reinterpret_tensor(buf50, (256, 32, 3), (1152, 36, 1), 33)  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_17.run(buf49, 24576, grid=grid(24576), stream=stream0)
        buf51 = reinterpret_tensor(buf37, (256, 36, 64), (2304, 64, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_18.run(buf35, arg35_1, buf51, 589824, grid=grid(589824), stream=stream0)
        del arg35_1
        del buf48
        del buf49
        buf52 = reinterpret_tensor(buf36, (256, 32, 64), (2048, 64, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(buf50, buf51, out=buf52)
        buf53 = reinterpret_tensor(buf12, (1, 32, 32, 8, 64), (524288, 16384, 512, 64, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_out_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf52, buf53, 524288, grid=grid(524288), stream=stream0)
        buf54 = reinterpret_tensor(buf52, (1024, 512), (512, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (1024, 512), (512, 1), 0), reinterpret_tensor(arg38_1, (512, 512), (1, 512), 0), out=buf54)
        del arg38_1
        buf92 = reinterpret_tensor(buf53, (1, 1024, 512), (524288, 512, 1), 0); del buf53  # reuse
        buf97 = buf34; del buf34  # reuse
        buf95 = reinterpret_tensor(buf97, (1, 32, 32, 512), (540672, 16896, 512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x, layer_norm_2, kv_1], Original ATen: [aten.add, aten.native_layer_norm, aten.cat]
        triton_per_fused_add_cat_native_layer_norm_20.run(buf25, buf54, arg39_1, arg40_1, arg41_1, buf92, buf95, 1024, 512, grid=grid(1024), stream=stream0)
        del arg40_1
        del arg41_1
        buf58 = empty_strided_cuda((1, 109, 16384), (1785856, 16384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [row_p], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg50_1, (1, 109, 32), (3488, 32, 1), 0), reinterpret_tensor(buf25, (1, 32, 16384), (524288, 16384, 1), 0), out=buf58)
        del arg50_1
        buf59 = empty_strided_cuda((1, 109, 16384), (1785856, 16384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [col_p], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg51_1, (1, 109, 32), (3488, 32, 1), 0), reinterpret_tensor(buf25, (1, 32, 16384), (524288, 16384, 1), 0), out=buf59)
        del arg51_1
        buf63 = empty_strided_cuda((109, 32, 512), (16384, 512, 1), torch.float32)
        buf68 = empty_strided_cuda((109, 1, 33, 512), (16896, 1, 512, 1), torch.float32)
        buf66 = reinterpret_tensor(buf68, (109, 1, 32, 512), (16896, 1, 512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [layer_norm_3, kv_2], Original ATen: [aten.native_layer_norm, aten.cat]
        triton_per_fused_cat_native_layer_norm_21.run(buf58, buf59, arg52_1, arg53_1, buf63, buf66, 3488, 512, grid=grid(3488), stream=stream0)
        del arg52_1
        del arg53_1
        buf64 = empty_strided_cuda((3488, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (3488, 512), (512, 1), 0), reinterpret_tensor(arg54_1, (512, 1536), (1, 512), 0), out=buf64)
        buf67 = reinterpret_tensor(buf68, (109, 1, 1, 512), (16896, 1, 512, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [block_node_2], Original ATen: [aten.mean]
        triton_per_fused_mean_22.run(buf63, buf67, 55808, 32, grid=grid(55808), stream=stream0)
        del buf66
        del buf67
        buf69 = empty_strided_cuda((3597, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (3597, 512), (512, 1), 0), reinterpret_tensor(arg54_1, (512, 1536), (1, 512), 0), out=buf69)
        del arg54_1
        buf70 = reinterpret_tensor(buf63, (109, 8, 32, 64, 1, 1), (16384, 2048, 64, 1, 1, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [einsum_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf64, arg55_1, buf70, 1785856, grid=grid(1785856), stream=stream0)
        buf71 = empty_strided_cuda((872, 64, 36), (2304, 36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_24.run(buf69, arg55_1, buf71, 2009088, grid=grid(2009088), stream=stream0)
        buf72 = empty_strided_cuda((872, 32, 36), (1152, 36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf70, (872, 32, 64), (2048, 64, 1), 0), buf71, out=buf72)
        buf73 = empty_strided_cuda((109, 1, 32, 33, 4), (4224, 460416, 132, 4, 1), torch.float32)
        buf131 = empty_strided_cuda((109, 1, 32, 33, 4), (4224, 460416, 132, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_8, setitem_9, setitem_12, setitem_13], Original ATen: [aten.lift_fresh, aten.fill]
        triton_poi_fused_fill_lift_fresh_25.run(arg30_1, buf73, buf131, 460416, grid=grid(460416), stream=stream0)
        del arg30_1
        # Topologically Sorted Source Nodes: [setitem_8, setitem_9, setitem_10], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
        triton_poi_fused_fill_index_put_lift_fresh_26.run(buf73, 13952, grid=grid(13952), stream=stream0)
        buf75 = empty_strided_cuda((109, 1, 32, 33), (1056, 1056, 33, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_11], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_27.run(buf73, buf75, 115104, grid=grid(115104), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_11], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_28.run(buf75, 3488, grid=grid(3488), stream=stream0)
        buf78 = empty_strided_cuda((109, 1, 32, 33, 8), (8448, 920832, 33, 1, 1056), torch.float32)
        # Topologically Sorted Source Nodes: [eq_4, scores_8, scores_6, scores_7, attn_probs_2], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_eq_masked_fill_mul_29.run(arg31_1, buf72, buf75, buf73, buf78, 27904, 33, grid=grid(27904), stream=stream0)
        buf79 = empty_strided_cuda((109, 1, 32, 1, 8), (256, 27904, 8, 27904, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_probs_2], Original ATen: [aten._softmax]
        triton_per_fused__softmax_30.run(buf78, buf79, 27904, 33, grid=grid(27904), stream=stream0)
        buf80 = empty_strided_cuda((109, 1, 32, 33, 4), (4224, 1, 132, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_31.run(buf75, buf73, buf80, 460416, grid=grid(460416), stream=stream0)
        del buf73
        buf81 = empty_strided_cuda((115104, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf80, (115104, 4), (4, 1), 0), reinterpret_tensor(arg56_1, (4, 8), (1, 4), 0), out=buf81)
        del arg56_1
        buf84 = buf72; del buf72  # reuse
        buf82 = reinterpret_tensor(buf84, (872, 32, 33), (1152, 36, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_32.run(buf78, buf79, arg31_1, buf81, arg57_1, buf82, 872, 1056, grid=grid(872, 1056), stream=stream0)
        del arg57_1
        buf83 = reinterpret_tensor(buf84, (872, 32, 3), (1152, 36, 1), 33)  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_33.run(buf83, 83712, grid=grid(83712), stream=stream0)
        buf85 = reinterpret_tensor(buf71, (872, 36, 64), (2304, 64, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_34.run(buf69, arg55_1, buf85, 2009088, grid=grid(2009088), stream=stream0)
        del arg55_1
        del buf82
        del buf83
        buf86 = reinterpret_tensor(buf70, (872, 32, 64), (2048, 64, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(buf84, buf85, out=buf86)
        buf87 = empty_strided_cuda((109, 1, 32, 8, 64), (16384, 1, 512, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_out_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf86, buf87, 1785856, grid=grid(1785856), stream=stream0)
        buf88 = reinterpret_tensor(buf86, (3488, 512), (512, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (3488, 512), (512, 1), 0), reinterpret_tensor(arg58_1, (512, 512), (1, 512), 0), out=buf88)
        del arg58_1
        buf121 = reinterpret_tensor(buf87, (109, 32, 512), (16384, 512, 1), 0); del buf87  # reuse
        buf126 = buf68; del buf68  # reuse
        buf124 = reinterpret_tensor(buf126, (109, 1, 32, 512), (16896, 1, 512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_2, layer_norm_4, kv_3], Original ATen: [aten.add, aten.native_layer_norm, aten.cat]
        triton_per_fused_add_cat_native_layer_norm_36.run(buf58, buf59, buf88, arg59_1, arg60_1, arg61_1, buf121, buf124, 3488, 512, grid=grid(3488), stream=stream0)
        del arg60_1
        del arg61_1
        buf93 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (1024, 512), (512, 1), 0), reinterpret_tensor(arg42_1, (512, 1536), (1, 512), 0), out=buf93)
        buf96 = reinterpret_tensor(buf97, (1, 32, 1, 512), (540672, 16896, 512, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [block_node_1], Original ATen: [aten.mean]
        triton_per_fused_mean_7.run(buf92, buf96, 16384, 32, grid=grid(16384), stream=stream0)
        del buf95
        del buf96
        buf98 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (1056, 512), (512, 1), 0), reinterpret_tensor(arg42_1, (512, 1536), (1, 512), 0), out=buf98)
        del arg42_1
        del buf97
        buf99 = reinterpret_tensor(buf92, (32, 8, 32, 64, 1, 1), (16384, 2048, 64, 1, 1, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf93, arg43_1, buf99, 524288, grid=grid(524288), stream=stream0)
        del buf93
        buf100 = reinterpret_tensor(buf51, (256, 64, 36), (2304, 36, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf98, arg43_1, buf100, 589824, grid=grid(589824), stream=stream0)
        buf101 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf99, (256, 32, 64), (2048, 64, 1), 0), buf100, out=buf101)
        # Topologically Sorted Source Nodes: [setitem_4, setitem_5, setitem_6], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
        triton_poi_fused_fill_index_put_lift_fresh_11.run(buf102, 4096, grid=grid(4096), stream=stream0)
        buf104 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [setitem_7], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_12.run(buf102, buf104, 33792, grid=grid(33792), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_7], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_13.run(buf104, 1024, grid=grid(1024), stream=stream0)
        buf107 = reinterpret_tensor(buf47, (1, 32, 32, 33, 8), (270336, 8448, 33, 1, 1056), 0); del buf47  # reuse
        buf108 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [eq_2, scores_5, scores_3, scores_4, attn_probs_1], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_eq_masked_fill_mul_14.run(arg29_1, buf101, buf104, buf102, buf107, buf108, 8192, 33, grid=grid(8192), stream=stream0)
        buf109 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_15.run(buf104, buf102, buf109, 135168, grid=grid(135168), stream=stream0)
        del buf102
        del buf104
        buf110 = reinterpret_tensor(buf44, (33792, 8), (8, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf109, (33792, 4), (4, 1), 0), reinterpret_tensor(arg44_1, (4, 8), (1, 4), 0), out=buf110)
        del arg44_1
        del buf109
        buf113 = buf101; del buf101  # reuse
        buf111 = reinterpret_tensor(buf113, (256, 32, 33), (1152, 36, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(buf107, buf108, arg29_1, buf110, arg45_1, buf111, 256, 1056, grid=grid(256, 1056), stream=stream0)
        del arg29_1
        del arg45_1
        del buf107
        del buf108
        del buf110
        buf112 = reinterpret_tensor(buf113, (256, 32, 3), (1152, 36, 1), 33)  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_17.run(buf112, 24576, grid=grid(24576), stream=stream0)
        buf114 = reinterpret_tensor(buf100, (256, 36, 64), (2304, 64, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_18.run(buf98, arg43_1, buf114, 589824, grid=grid(589824), stream=stream0)
        del arg43_1
        del buf111
        del buf112
        del buf98
        buf115 = reinterpret_tensor(buf99, (256, 32, 64), (2048, 64, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(buf113, buf114, out=buf115)
        del buf113
        del buf114
        buf116 = empty_strided_cuda((1, 32, 32, 8, 64), (524288, 16384, 512, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_out_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf115, buf116, 524288, grid=grid(524288), stream=stream0)
        buf117 = reinterpret_tensor(buf115, (1024, 512), (512, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (1024, 512), (512, 1), 0), reinterpret_tensor(arg46_1, (512, 512), (1, 512), 0), out=buf117)
        del arg46_1
        del buf116
        buf118 = reinterpret_tensor(buf117, (1, 1024, 512), (524288, 512, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.add]
        triton_poi_fused_add_37.run(buf118, buf25, buf54, arg39_1, arg47_1, 524288, grid=grid(524288), stream=stream0)
        del arg39_1
        del arg47_1
        del buf25
        del buf54
        buf119 = empty_strided_cuda((1024, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_leaf], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg49_1, reinterpret_tensor(buf118, (1024, 512), (512, 1), 0), reinterpret_tensor(arg48_1, (512, 32), (1, 512), 0), alpha=1, beta=1, out=buf119)
        del arg48_1
        del arg49_1
        buf120 = empty_strided_cuda((32, 32, 32), (1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf119, (32, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf119, (32, 32, 32), (1024, 1, 32), 0), out=buf120)
        del buf119
        buf122 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf121, (3488, 512), (512, 1), 0), reinterpret_tensor(arg62_1, (512, 1536), (1, 512), 0), out=buf122)
        buf125 = reinterpret_tensor(buf126, (109, 1, 1, 512), (16896, 1, 512, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [block_node_3], Original ATen: [aten.mean]
        triton_per_fused_mean_22.run(buf121, buf125, 55808, 32, grid=grid(55808), stream=stream0)
        del buf124
        del buf125
        buf127 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf126, (3597, 512), (512, 1), 0), reinterpret_tensor(arg62_1, (512, 1536), (1, 512), 0), out=buf127)
        del arg62_1
        del buf126
        buf128 = reinterpret_tensor(buf121, (109, 8, 32, 64, 1, 1), (16384, 2048, 64, 1, 1, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [einsum_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf122, arg63_1, buf128, 1785856, grid=grid(1785856), stream=stream0)
        del buf122
        buf129 = reinterpret_tensor(buf85, (872, 64, 36), (2304, 36, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_24.run(buf127, arg63_1, buf129, 2009088, grid=grid(2009088), stream=stream0)
        buf130 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf128, (872, 32, 64), (2048, 64, 1), 0), buf129, out=buf130)
        # Topologically Sorted Source Nodes: [setitem_12, setitem_13, setitem_14], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
        triton_poi_fused_fill_index_put_lift_fresh_26.run(buf131, 13952, grid=grid(13952), stream=stream0)
        buf133 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [setitem_15], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_27.run(buf131, buf133, 115104, grid=grid(115104), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_15], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_28.run(buf133, 3488, grid=grid(3488), stream=stream0)
        buf136 = reinterpret_tensor(buf81, (109, 1, 32, 33, 8), (8448, 920832, 33, 1, 1056), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [eq_6, scores_11, scores_9, scores_10, attn_probs_3], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_eq_masked_fill_mul_29.run(arg31_1, buf130, buf133, buf131, buf136, 27904, 33, grid=grid(27904), stream=stream0)
        buf137 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [attn_probs_3], Original ATen: [aten._softmax]
        triton_per_fused__softmax_30.run(buf136, buf137, 27904, 33, grid=grid(27904), stream=stream0)
        buf138 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_31.run(buf133, buf131, buf138, 460416, grid=grid(460416), stream=stream0)
        del buf131
        del buf133
        buf139 = reinterpret_tensor(buf78, (115104, 8), (8, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf138, (115104, 4), (4, 1), 0), reinterpret_tensor(arg64_1, (4, 8), (1, 4), 0), out=buf139)
        del arg64_1
        del buf138
        buf142 = buf130; del buf130  # reuse
        buf140 = reinterpret_tensor(buf142, (872, 32, 33), (1152, 36, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_32.run(buf136, buf137, arg31_1, buf139, arg65_1, buf140, 872, 1056, grid=grid(872, 1056), stream=stream0)
        del arg31_1
        del arg65_1
        del buf136
        del buf137
        del buf139
        buf141 = reinterpret_tensor(buf142, (872, 32, 3), (1152, 36, 1), 33)  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_33.run(buf141, 83712, grid=grid(83712), stream=stream0)
        buf143 = reinterpret_tensor(buf129, (872, 36, 64), (2304, 64, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_34.run(buf127, arg63_1, buf143, 2009088, grid=grid(2009088), stream=stream0)
        del arg63_1
        del buf127
        del buf140
        del buf141
        buf144 = reinterpret_tensor(buf128, (872, 32, 64), (2048, 64, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(buf142, buf143, out=buf144)
        del buf142
        del buf143
        buf145 = empty_strided_cuda((109, 1, 32, 8, 64), (16384, 1, 512, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_out_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf144, buf145, 1785856, grid=grid(1785856), stream=stream0)
        buf146 = reinterpret_tensor(buf144, (3488, 512), (512, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf145, (3488, 512), (512, 1), 0), reinterpret_tensor(arg66_1, (512, 512), (1, 512), 0), out=buf146)
        del arg66_1
        del buf145
        buf147 = reinterpret_tensor(buf146, (109, 32, 512), (16384, 512, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.add]
        triton_poi_fused_add_38.run(buf147, buf58, buf59, buf88, arg59_1, arg67_1, 1785856, grid=grid(1785856), stream=stream0)
        del arg59_1
        del arg67_1
        del buf58
        del buf59
        del buf88
        buf148 = empty_strided_cuda((3488, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_leaf_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg69_1, reinterpret_tensor(buf147, (3488, 512), (512, 1), 0), reinterpret_tensor(arg68_1, (512, 32), (1, 512), 0), alpha=1, beta=1, out=buf148)
        del arg68_1
        del arg69_1
        buf149 = empty_strided_cuda((3488, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_leaf], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg71_1, reinterpret_tensor(buf147, (3488, 512), (512, 1), 0), reinterpret_tensor(arg70_1, (512, 32), (1, 512), 0), alpha=1, beta=1, out=buf149)
        del arg70_1
        del arg71_1
        del buf147
        buf150 = empty_strided_cuda((109, 32, 32), (1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf148, (109, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf149, (109, 32, 32), (1024, 1, 32), 0), out=buf150)
        del buf148
        del buf149
        buf151 = empty_strided_cuda((1024, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (1024, 512), (512, 1), 0), reinterpret_tensor(arg72_1, (512, 1), (1, 512), 0), out=buf151)
        del arg72_1
        del buf118
        buf152 = reinterpret_tensor(buf151, (1, 32, 32, 1), (1024, 32, 1, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [sigmoid], Original ATen: [aten.sigmoid]
        triton_poi_fused_sigmoid_39.run(buf152, arg73_1, 1024, grid=grid(1024), stream=stream0)
        del arg73_1
        buf153 = empty_strided_cuda((1, 145408), (145408, 1), torch.float32)
        # Topologically Sorted Source Nodes: [packed_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_40.run(buf120, buf150, buf152, buf153, 145408, grid=grid(145408), stream=stream0)
        del buf120
        del buf150
    return (buf153, reinterpret_tensor(buf152, (1, 32, 32), (1024, 32, 1), 0), )


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
    arg28_1 = rand_strided((32, 32, 33, 4), (4224, 132, 4, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((32, 32, 33), (1056, 33, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((109, 32, 33, 4), (4224, 132, 4, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((109, 32, 33), (1056, 33, 1), device='cuda:0', dtype=torch.float32)
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
    arg42_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((8, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((109, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((109, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((8, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((8, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
