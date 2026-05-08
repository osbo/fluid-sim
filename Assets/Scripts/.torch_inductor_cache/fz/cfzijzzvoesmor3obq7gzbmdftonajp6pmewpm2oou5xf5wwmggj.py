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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/6m/c6mkroeer47lof7sdw6o4blm5kjfqels53pgvp2kv3vco4zb45wv.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view, %full_default_44], 1), kwargs = {})
triton_poi_fused_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 20
    x1 = (xindex // 20)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 18, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 6, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.load(in_ptr0 + (3 + (9*x1) + x0), tmp7, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp0 >= tmp5
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr1 + ((-6) + x0), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.where(tmp6, tmp8, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1], 20, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = 0.0
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp4, tmp14, tmp20)
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/td/ctdmq5ehloqbiwsprnyzmv7qvxtxo5cmffzckpaghybxqah2n43p.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute, %full_default_45],), kwargs = {})
triton_poi_fused_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256)
    x0 = xindex % 256
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 18, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((18*x0) + x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 20, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = 0.0
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.where(tmp4, tmp5, tmp11)
    tl.store(out_ptr0 + (x2), tmp12, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/d7/cd7pbcz54k5rckxqhfp7bcbug437p7g6inwfqep2zctphjhywpz5.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_2 => add, erf, mul, mul_1, mul_2
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.5), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_1,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %add), kwargs = {})
triton_poi_fused_gelu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/hk/chk6krdzdzi64wv6ndbshjd2j4skwn2bbjf4pat6sdmnmbngxhwz.py
# Topologically Sorted Source Nodes: [aggr], Original ATen: [aten.zeros_like]
# Source node to ATen node mapping:
#   aggr => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8192, 256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_like_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_like_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (512*x1)), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/3y/c3y5c3x2xhiojxxecwb3dat2s4vo4jtrh6jmqxj3oionqir4ntml.py
# Topologically Sorted Source Nodes: [aggr, getitem_4, messages, index_add_], Original ATen: [aten.zeros_like, aten.index, aten.mul, aten.index_add]
# Source node to ATen node mapping:
#   aggr => full_default
#   getitem_4 => index
#   index_add_ => index_put
#   messages => mul_3
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8192, 256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%addmm_2, [%select_1]), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index, %unsqueeze_1), kwargs = {})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default, [%select], %mul_3, True), kwargs = {})
triton_poi_fused_index_index_add_mul_zeros_like_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_index_add_mul_zeros_like_4', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10392576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256)
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (40596 + x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 8192, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 8192)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 8192")
    tmp7 = tmp6 + tmp1
    tmp8 = tmp6 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp6)
    tl.device_assert(((0 <= tmp9) & (tmp9 < 8192)) | ~(xmask), "index out of bounds: 0 <= tmp9 < 8192")
    tmp11 = tl.load(in_ptr1 + (x0 + (256*tmp9)), xmask)
    tmp13 = tmp11 * tmp12
    tl.atomic_add(out_ptr0 + (x0 + (512*tmp4)), tmp13, xmask, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/cd/ccdvltxj7qtatogamnw6q5phnnogo3t7srwr75narrc63smnqryu.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   out => add_2
# Graph fragment:
#   %add_tensor_66 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_66, %arg15_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze, %add_tensor_66), kwargs = {})
triton_poi_fused_add_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/uk/cukmokjyxal5q5txlsvbrrftc6euudju7rbnftuio7oab2mkzocp.py
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
triton_per_fused_native_layer_norm_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_6', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 8192
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.full([1], 256, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp4 - tmp12
    tmp19 = 256.0
    tmp20 = tmp17 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp28, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/xp/cxpv5d6uvd74go27ysv7rszj7pjgr5gcogyuzhddx3dgsmwrlvop.py
# Topologically Sorted Source Nodes: [layer_norm_1], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_1 => add_8, add_9, mul_13, mul_14, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_9, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_9, %getitem_3), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %arg32_1), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %arg33_1), kwargs = {})
triton_per_fused_native_layer_norm_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_7', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
    xnumel = 8192
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), None)
    tmp21 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 256, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp0 - tmp8
    tmp15 = 256.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp24, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/kf/ckfmcdsgt6qsndb5ztlt7blik7d7p35oytw3c4cy7wq4vzvv4lul.py
# Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem => full_default_2, index_put_2
# Graph fragment:
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_2 : [num_users=2] = call_function[target=torch.ops.aten.index_put.default](args = (%expand_3, [None, None, %iota, %iota], %full_default_2), kwargs = {})
triton_poi_fused_index_put_lift_fresh_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/pb/cpbw347mfmckcwr6wda6isyhkjrqxqahmdm5pudviomoghhd52oi.py
# Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem => full_default_2, index_put_2
# Graph fragment:
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_2 : [num_users=2] = call_function[target=torch.ops.aten.index_put.default](args = (%expand_3, [None, None, %iota, %iota], %full_default_2), kwargs = {})
triton_poi_fused_index_put_lift_fresh_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_9', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 4
    x1 = (xindex // 4) % 128
    x2 = (xindex // 512)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (516*x1) + (65536*x2)), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/zu/czu7istaobmgcmkeb4zi42ahserrki72mtgm62ltoqg47xjyvr5l.py
# Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_1 => full_default_3, index_put_3
# Graph fragment:
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_3 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_5, [None, None, %iota, %iota], %full_default_3), kwargs = {})
triton_poi_fused_index_put_lift_fresh_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + (4*x0)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/pn/cpn5tpx4j2e5jgjsdgjbwl5anbpem24mlqkrt3wbljoylletrbvj.py
# Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_1 => full_default_3, index_put_3
# Graph fragment:
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_3 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_5, [None, None, %iota, %iota], %full_default_3), kwargs = {})
triton_poi_fused_index_put_lift_fresh_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_11', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = 1.0
    tl.store(out_ptr0 + ((129*x0) + (16384*x1)), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/bd/cbd6r2tszroxq3vqluabsa2mg4uk4urqlex6as7bgixs3q3gl6o2.py
# Topologically Sorted Source Nodes: [q_f, k_f, v_f, logit_bias, x_soft], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   k_f => clone_2
#   logit_bias => clone_4
#   q_f => clone_1
#   v_f => clone_3
#   x_soft => _scaled_dot_product_efficient_attention
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_22,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_23,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_24,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_1, %clone_2, %clone_3, %expand_6, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_12', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 128
    x2 = (xindex // 4096) % 8
    x3 = (xindex // 32768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (768*x1) + (98304*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/md/cmdepaacd556k5oqax6vsw7tz4ii6sfaqfflfswwfm75sfgdn3pn.py
# Topologically Sorted Source Nodes: [q_f, k_f, v_f, logit_bias, x_soft], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   k_f => clone_2
#   logit_bias => clone_4
#   q_f => clone_1
#   v_f => clone_3
#   x_soft => _scaled_dot_product_efficient_attention
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_22,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_23,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_24,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_1, %clone_2, %clone_3, %expand_6, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_13', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 128
    x2 = (xindex // 4096) % 8
    x3 = (xindex // 32768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + (32*x2) + (768*x1) + (98304*x3)), None)
    tmp1 = tl.load(in_ptr1 + (256 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/uv/cuvudt5jihflbdxzqw7xtk427nesaj6kbulp6ec4sh26kkimvjoq.py
# Topologically Sorted Source Nodes: [q_f, k_f, v_f, logit_bias, x_soft, einsum_2], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   einsum_2 => clone_6
#   k_f => clone_2
#   logit_bias => clone_4
#   q_f => clone_1
#   v_f => clone_3
#   x_soft => _scaled_dot_product_efficient_attention
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_22,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_23,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_24,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_1, %clone_2, %clone_3, %expand_6, False), kwargs = {})
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_31,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_14', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 128
    x2 = (xindex // 4096) % 8
    x3 = (xindex // 32768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + (32*x2) + (768*x1) + (98304*x3)), None)
    tmp1 = tl.load(in_ptr1 + (512 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
    tl.store(out_ptr1 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/5o/c5ocyb7wmwhyv6fqfffly7dum6gmfigbutufcynrlxjtczxjeih3.py
# Topologically Sorted Source Nodes: [q_f, k_f, v_f, logit_bias, x_soft], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   k_f => clone_2
#   logit_bias => clone_4
#   q_f => clone_1
#   v_f => clone_3
#   x_soft => _scaled_dot_product_efficient_attention
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_22,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_23,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_24,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_1, %clone_2, %clone_3, %expand_6, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_15', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16384
    x2 = (xindex // 131072)
    x3 = xindex
    tmp2 = tl.load(in_ptr0 + (x0 + (16384*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (3 + (4*x0) + (65536*x2)), None, eviction_policy='evict_last')
    tmp0 = tl.full([1], 3, tl.int32)
    tmp1 = tmp0 == tmp0
    tmp4 = tl.where(tmp1, tmp2, tmp3)
    tl.store(out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/mw/cmwhjx4la6vclmka4djre4o7keghlbdfmfz52t4wikockw6pmjgx.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %select_scatter_default : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%index_put_2, %index_put_3, 4, 3), kwargs = {})
triton_poi_fused_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_16', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/mr/cmrzbxdcvxos2jqesicagwgajgy3jpiygvaar5cc2osouwiwrb5f.py
# Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_11 => add_10, erf_3, mul_15, mul_16, mul_17
# Graph fragment:
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_35, 0.5), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_35, 0.7071067811865476), kwargs = {})
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
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_17', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/y4/cy4jnqzuutuhtatxsz5mhmibj4pbbtfusdjm4brlcl22ijmwd3uq.py
# Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   einsum_2 => clone_5
# Graph fragment:
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_30,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[512, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_18', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = yindex % 8
    y1 = (yindex // 8)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (8*x2) + (131072*y1)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (16384*y3)), tmp2, ymask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/kf/ckfm5uenw4cjwa6dbcj6yw2m4hldw2qgth34crf2cislxebjow35.py
# Topologically Sorted Source Nodes: [x_mid], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_mid => add_11
# Graph fragment:
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_38, %view_42), kwargs = {})
triton_poi_fused_add_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_19', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x0 = xindex % 32
    x1 = (xindex // 32) % 8
    x2 = (xindex // 256) % 128
    x3 = (xindex // 32768)
    tmp0 = tl.load(in_out_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (32*x2) + (4096*x1) + (32768*x3)), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/tw/ctwtmnpebjcmhpetiv42gx4xdfl4qm5iw5zxoxcwy3w3zx34qsay.py
# Topologically Sorted Source Nodes: [row_highway], Original ATen: [aten.new_zeros]
# Source node to ATen node mapping:
#   row_highway => full_default_4
# Graph fragment:
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 64, 128, 256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_new_zeros_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_20', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ds/cdsckalghr67a6av7wxabhb5z2mce2nsceyuvdpyt3jmazzxmlq6.py
# Topologically Sorted Source Nodes: [x_mid_d, z, row_highway, index_add__2, col_highway, index_add__3], Original ATen: [aten.add, aten.native_layer_norm, aten.new_zeros, aten.index_add]
# Source node to ATen node mapping:
#   col_highway => full_default_5
#   index_add__2 => index_put_4
#   index_add__3 => index_put_5
#   row_highway => full_default_4
#   x_mid_d => add_12
#   z => add_13, add_14, mul_18, mul_19, rsqrt_2, sub_2, var_mean_2
# Graph fragment:
#   %add_12 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_9, %view_46), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_12, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_12, %getitem_9), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %arg46_1), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %arg47_1), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 64, 128, 256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_4 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_4, [None, %slice_31], %view_47, True), kwargs = {})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 64, 128, 256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_5 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_5, [None, %slice_31], %view_47, True), kwargs = {})
triton_per_fused_add_index_add_native_layer_norm_new_zeros_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_index_add_native_layer_norm_new_zeros_21', 'mutated_arg_names': ['out_ptr2', 'out_ptr3'], 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 8192
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x3 = (xindex // 128)
    x2 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.full([1], 256, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp19 = tl.full([RBLOCK], 64, tl.int32)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp18)
    tl.device_assert((0 <= tmp22) & (tmp22 < 64), "index out of bounds: 0 <= tmp22 < 64")
    tmp24 = tmp4 - tmp12
    tmp25 = 256.0
    tmp26 = tmp17 / tmp25
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp24 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tl.atomic_add(out_ptr2 + (r1 + (256*x2) + (32768*tmp22)), tmp4, None, sem='relaxed')
    tl.atomic_add(out_ptr3 + (r1 + (256*x2) + (32768*tmp22)), tmp4, None, sem='relaxed')
    tl.store(out_ptr4 + (r1 + (1024*x0)), tmp34, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/on/conau535pie45iozjgprngqviqqx3j5vkmjsn2rp3o3jfusrub7o.py
# Topologically Sorted Source Nodes: [x_mid_d, g], Original ATen: [aten.add, aten.mean]
# Source node to ATen node mapping:
#   g => mean_4
#   x_mid_d => add_12
# Graph fragment:
#   %add_12 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_9, %view_46), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_12, [1], True), kwargs = {})
triton_red_fused_add_mean_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mean_22', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/g4/cg4yjkzifzsn322o34xqf67u665l3bmsfnvxswtjkhvhuocky42z.py
# Topologically Sorted Source Nodes: [x_mid_d, g], Original ATen: [aten.add, aten.mean]
# Source node to ATen node mapping:
#   g => mean_4
#   x_mid_d => add_12
# Graph fragment:
#   %add_12 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_9, %view_46), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_12, [1], True), kwargs = {})
triton_per_fused_add_mean_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[256, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_23', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/nr/cnrjisjubrqi4o6jbofiwlnloz6a6j2b4bud67ih4dgev6aqksdn.py
# Topologically Sorted Source Nodes: [z_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   z_1 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_14, %view_56, %view_59, %expand_7], -1), kwargs = {})
triton_poi_fused_cat_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_24', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (1024*x1)), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ng/cngo7fhamt3y4t345msa2rq455z6n54jarwtkcqw6hd2bctrye7j.py
# Topologically Sorted Source Nodes: [z_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   z_1 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_14, %view_56, %view_59, %expand_7], -1), kwargs = {})
triton_poi_fused_cat_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_25', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = 8192.0
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + (x0 + (1024*x1)), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/s2/cs2zs2kw6q2fwz4w3hpafvnmamgf6zdq35de2setltmyr7ysr7yo.py
# Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_14 => add_15, erf_4, mul_20, mul_21, mul_22
# Graph fragment:
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_61, 0.5), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_61, 0.7071067811865476), kwargs = {})
#   %erf_4 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_21,), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_4, 1), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %add_15), kwargs = {})
triton_poi_fused_gelu_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_26', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1024
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/fn/cfnx2oxt5i6fkffgub45e3pkwr4qqcwgz2kuyv4r6wl6rgaudx5z.py
# Topologically Sorted Source Nodes: [x_mid_d, h_diag_1, layer_norm_5], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   h_diag_1 => add_16
#   layer_norm_5 => add_27, add_28, mul_33, mul_34, rsqrt_5, sub_5, var_mean_5
#   x_mid_d => add_12
# Graph fragment:
#   %add_12 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_9, %view_46), kwargs = {})
#   %add_16 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %view_63), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_16, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_16, %getitem_19), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_18, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_27,), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_5), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_33, %arg68_1), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, %arg69_1), kwargs = {})
triton_per_fused_add_native_layer_norm_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_27', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 8192
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (r1 + (256*x0)), None)
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
    tmp14 = tl.full([1], 256, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp9 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = tmp8 - tmp16
    tmp23 = 256.0
    tmp24 = tmp21 / tmp23
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp8, None)
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp32, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/43/c43mflz7cbmdrgrwrzhsnw5bxs6clyuuizx7sgj6hazsja745dbt.py
# Topologically Sorted Source Nodes: [h_off_3, layer_norm_3], Original ATen: [aten.mean, aten.native_layer_norm]
# Source node to ATen node mapping:
#   h_off_3 => mean_5
#   layer_norm_3 => add_18, add_19, mul_23, mul_24, rsqrt_3, sub_3, var_mean_3
# Graph fragment:
#   %mean_5 : [num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%view_74, [2]), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%mean_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean_5, %getitem_11), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %arg52_1), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %arg53_1), kwargs = {})
triton_per_fused_mean_native_layer_norm_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_layer_norm_28', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 10, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 7488
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), None)
    tmp3 = tl.load(in_ptr0 + (256 + r1 + (1024*x0)), None)
    tmp4 = tl.load(in_ptr1 + (256 + r1 + (1024*x0)), None)
    tmp7 = tl.load(in_ptr0 + (512 + r1 + (1024*x0)), None)
    tmp8 = tl.load(in_ptr1 + (512 + r1 + (1024*x0)), None)
    tmp11 = tl.load(in_ptr0 + (768 + r1 + (1024*x0)), None)
    tmp12 = tl.load(in_ptr1 + (768 + r1 + (1024*x0)), None)
    tmp37 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = tl.full([1], 256, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 / tmp23
    tmp25 = tmp17 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp30 = tmp16 - tmp24
    tmp31 = 256.0
    tmp32 = tmp29 / tmp31
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tl.store(out_ptr0 + (r1 + (256*x0)), tmp16, None)
    tl.store(out_ptr3 + (r1 + (256*x0)), tmp40, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/qc/cqce5qruqymspobuctiq6xbcbc4ih7sftnlwghuubj4gqphoe4vl.py
# Topologically Sorted Source Nodes: [sub, setitem_2, setitem_6, setitem_10, setitem_14], Original ATen: [aten.mean, aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_10 => full_default_22, index_put_30
#   setitem_14 => full_default_30, index_put_42
#   setitem_2 => full_default_6, index_put_6
#   setitem_6 => full_default_14, index_put_18
#   sub => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_6, [2, 4]), kwargs = {})
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_6 : [num_users=2] = call_function[target=torch.ops.aten.index_put.default](args = (%view_7, [None, None, %iota_1, %iota_1], %full_default_6), kwargs = {})
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_18 : [num_users=2] = call_function[target=torch.ops.aten.index_put.default](args = (%view_7, [None, None, %iota_3, %iota_3], %full_default_14), kwargs = {})
#   %full_default_22 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_30 : [num_users=2] = call_function[target=torch.ops.aten.index_put.default](args = (%view_7, [None, None, %iota_5, %iota_5], %full_default_22), kwargs = {})
#   %full_default_30 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_42 : [num_users=2] = call_function[target=torch.ops.aten.index_put.default](args = (%view_7, [None, None, %iota_7, %iota_7], %full_default_30), kwargs = {})
triton_per_fused_index_put_lift_fresh_mean_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1048576, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_index_put_lift_fresh_mean_29', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 958464
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex % 4
    r4 = (rindex // 4)
    x0 = xindex % 4
    x1 = (xindex // 4) % 32
    x2 = (xindex // 128)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (4*r3) + (16*x1) + (512*r4) + (2048*x2)), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = 16.0
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr1 + (x5), tmp5, None)
    tl.store(out_ptr2 + (x5), tmp5, None)
    tl.store(out_ptr3 + (x5), tmp5, None)
    tl.store(out_ptr4 + (x5), tmp5, None)
    tl.store(out_ptr5 + (x5), tmp5, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/nf/cnfmf3feeo6ggotdqjqhaa4wfvuygkuktbgurt27ldjlekovy4n6.py
# Topologically Sorted Source Nodes: [setitem_2], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_2 => full_default_6, index_put_6
# Graph fragment:
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_6 : [num_users=2] = call_function[target=torch.ops.aten.index_put.default](args = (%view_7, [None, None, %iota_1, %iota_1], %full_default_6), kwargs = {})
triton_poi_fused_index_put_lift_fresh_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_30', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 29952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = (xindex // 4) % 32
    x2 = (xindex // 128)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (132*x1) + (4096*x2)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/dm/cdmub7g473aj4vgajh4ycg6m5m3hwytow6ns6qxrelddchqozras.py
# Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_3 => full_default_7, index_put_7
# Graph fragment:
#   %full_default_7 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_7 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_10, [None, None, %iota_1, %iota_1], %full_default_7), kwargs = {})
triton_poi_fused_index_put_lift_fresh_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_31', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 239616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + (4*x0)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/hv/chvz2au5a5okgl6mvpant3qmg7yaswotolgluebi32ujr3ehwmlj.py
# Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_3 => full_default_7, index_put_7
# Graph fragment:
#   %full_default_7 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_7 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_10, [None, None, %iota_1, %iota_1], %full_default_7), kwargs = {})
triton_poi_fused_index_put_lift_fresh_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_32', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp0 = 1.0
    tl.store(out_ptr0 + ((33*x0) + (1024*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/d6/cd6jdldkt4wo2ni5gurtvdvh5uy2itroif25us4nweemv4usxn22.py
# Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, logit_bias_1, x_soft_2], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   k_f_1 => clone_9
#   logit_bias_1 => clone_11
#   q_f_1 => clone_8
#   v_f_1 => clone_10
#   x_soft_2 => _scaled_dot_product_efficient_attention_1
# Graph fragment:
#   %clone_8 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_47,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_9 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_48,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_10 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_49,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_9,), kwargs = {})
#   %_scaled_dot_product_efficient_attention_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_8, %clone_9, %clone_10, %expand_10, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_33', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1916928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 32
    x2 = (xindex // 1024) % 8
    x3 = (xindex // 8192)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (768*x1) + (24576*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/3h/c3hddfu5egoqcavtgxitm4coc3mkhp5xag2kepai6tfuq5ovpkgr.py
# Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, logit_bias_1, x_soft_2], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   k_f_1 => clone_9
#   logit_bias_1 => clone_11
#   q_f_1 => clone_8
#   v_f_1 => clone_10
#   x_soft_2 => _scaled_dot_product_efficient_attention_1
# Graph fragment:
#   %clone_8 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_47,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_9 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_48,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_10 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_49,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_9,), kwargs = {})
#   %_scaled_dot_product_efficient_attention_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_8, %clone_9, %clone_10, %expand_10, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_34', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1916928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 32
    x2 = (xindex // 1024) % 8
    x3 = (xindex // 8192)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + (32*x2) + (768*x1) + (24576*x3)), None)
    tmp1 = tl.load(in_ptr1 + (256 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/47/c475owssbhlg6rfelggk7bckw4vfzlppwzuw3tevs7nb3w24kkd4.py
# Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, logit_bias_1, x_soft_2, einsum_5], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   einsum_5 => clone_13
#   k_f_1 => clone_9
#   logit_bias_1 => clone_11
#   q_f_1 => clone_8
#   v_f_1 => clone_10
#   x_soft_2 => _scaled_dot_product_efficient_attention_1
# Graph fragment:
#   %clone_8 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_47,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_9 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_48,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_10 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_49,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_9,), kwargs = {})
#   %_scaled_dot_product_efficient_attention_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_8, %clone_9, %clone_10, %expand_10, False), kwargs = {})
#   %clone_13 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_56,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_35', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1916928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 32
    x2 = (xindex // 1024) % 8
    x3 = (xindex // 8192)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + (32*x2) + (768*x1) + (24576*x3)), None)
    tmp1 = tl.load(in_ptr1 + (512 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
    tl.store(out_ptr1 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ui/cuihfmkmzloryeqif2gv52mvgn36wn32hngt3tqgnqozchsz6uq3.py
# Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, logit_bias_1, x_soft_2], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   k_f_1 => clone_9
#   logit_bias_1 => clone_11
#   q_f_1 => clone_8
#   v_f_1 => clone_10
#   x_soft_2 => _scaled_dot_product_efficient_attention_1
# Graph fragment:
#   %clone_8 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_47,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_9 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_48,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_10 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_49,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_9,), kwargs = {})
#   %_scaled_dot_product_efficient_attention_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_8, %clone_9, %clone_10, %expand_10, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_36', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1916928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 1024
    x2 = (xindex // 8192)
    x3 = xindex
    tmp2 = tl.load(in_ptr0 + (x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (3 + (4*x0) + (4096*x2)), None, eviction_policy='evict_last')
    tmp0 = tl.full([1], 3, tl.int32)
    tmp1 = tmp0 == tmp0
    tmp4 = tl.where(tmp1, tmp2, tmp3)
    tl.store(out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/d6/cd6fo4z7ndvdohipcgohtsmw6laisnhqvd5vucnktvyalgi6qp7v.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %select_scatter_default_1 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%index_put_6, %index_put_7, 4, 3), kwargs = {})
triton_poi_fused_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_37', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 958464
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/mc/cmcd46tcrjcfrk4jxfnckldlx3llsccxgsupfjoesjjnushrfoki.py
# Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_17 => add_20, erf_5, mul_25, mul_26, mul_27
# Graph fragment:
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_88, 0.5), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_88, 0.7071067811865476), kwargs = {})
#   %erf_5 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_26,), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_5, 1), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %add_20), kwargs = {})
triton_poi_fused_gelu_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_38', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3833856
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/4i/c4iljtg4tfylscu3vi6irwwvowcxttzlejozoldrygkem63mtfyw.py
# Topologically Sorted Source Nodes: [einsum_5], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   einsum_5 => clone_12
# Graph fragment:
#   %clone_12 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_55,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_39', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1872
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 8
    y1 = (yindex // 8)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (8*x2) + (8192*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/u4/cu4uupjqhzyzgmtumq2waqvm5qwcruvhmteqjh7ruvsg5wtmxcur.py
# Topologically Sorted Source Nodes: [x_mid_1], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_mid_1 => add_21
# Graph fragment:
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_91, %view_95), kwargs = {})
triton_poi_fused_add_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_40', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1916928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x0 = xindex % 32
    x1 = (xindex // 32) % 8
    x2 = (xindex // 256) % 32
    x3 = (xindex // 8192)
    tmp0 = tl.load(in_out_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (32*x2) + (1024*x1) + (8192*x3)), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/a6/ca62hcoelzwacjtawqamp2ufjeprae6jmsot2jip7p3kppglv73x.py
# Topologically Sorted Source Nodes: [x_mid_o, z_2], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_mid_o => add_22
#   z_2 => add_23, add_24, mul_28, mul_29, rsqrt_4, sub_4, var_mean_4
# Graph fragment:
#   %add_22 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_5, %view_99), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_22, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_22, %getitem_17), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_23,), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %arg62_1), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %arg63_1), kwargs = {})
triton_per_fused_add_native_layer_norm_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_41', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 7488
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.full([1], 256, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp4 - tmp12
    tmp19 = 256.0
    tmp20 = tmp17 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp28, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/px/cpxznyey2rsazi2r7zjypnnc56kvv4k5r4ubxohmsiqdlc7bqlce.py
# Topologically Sorted Source Nodes: [x_mid_d_1, z_4, row_highway_2, index_add__8, row_highway_1, index_add__4, col_highway_1, index_add__5, col_highway_2, index_add__9], Original ATen: [aten.add, aten.native_layer_norm, aten.new_zeros, aten.index_add]
# Source node to ATen node mapping:
#   col_highway_1 => full_default_9
#   col_highway_2 => full_default_13
#   index_add__4 => index_put_8
#   index_add__5 => index_put_9
#   index_add__8 => index_put_14
#   index_add__9 => index_put_15
#   row_highway_1 => full_default_8
#   row_highway_2 => full_default_12
#   x_mid_d_1 => add_31
#   z_4 => add_32, add_33, mul_38, mul_39, rsqrt_6, sub_6, var_mean_6
# Graph fragment:
#   %add_31 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %view_154), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_31, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_31, %getitem_25), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_24, 1e-05), kwargs = {})
#   %rsqrt_6 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_32,), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_6), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_38, %arg78_1), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_39, %arg79_1), kwargs = {})
#   %full_default_12 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 64, 128, 256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_14 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_12, [None, %slice_31], %view_156, True), kwargs = {})
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 64, 128, 256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_8 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_8, [None, %slice_31], %view_101, True), kwargs = {})
#   %full_default_9 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 64, 128, 256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_9 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_9, [None, %slice_31], %view_101, True), kwargs = {})
#   %full_default_13 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 64, 128, 256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_15 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_13, [None, %slice_31], %view_156, True), kwargs = {})
triton_per_fused_add_index_add_native_layer_norm_new_zeros_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_index_add_native_layer_norm_new_zeros_42', 'mutated_arg_names': ['out_ptr2', 'out_ptr3', 'out_ptr5', 'out_ptr6'], 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel):
    xnumel = 8192
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x3 = (xindex // 128)
    x2 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.full([1], 256, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp19 = tl.full([RBLOCK], 64, tl.int32)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp18)
    tl.device_assert((0 <= tmp22) & (tmp22 < 64), "index out of bounds: 0 <= tmp22 < 64")
    tmp24 = tmp4 - tmp12
    tmp25 = 256.0
    tmp26 = tmp17 / tmp25
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp24 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tl.atomic_add(out_ptr2 + (r1 + (256*x2) + (32768*tmp22)), tmp4, None, sem='relaxed')
    tl.atomic_add(out_ptr3 + (r1 + (256*x2) + (32768*tmp22)), tmp4, None, sem='relaxed')
    tl.store(out_ptr4 + (r1 + (1024*x0)), tmp34, None)
    tl.atomic_add(out_ptr5 + (r1 + (256*x2) + (32768*tmp22)), tmp0, None, sem='relaxed')
    tl.atomic_add(out_ptr6 + (r1 + (256*x2) + (32768*tmp22)), tmp0, None, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ra/craf37kzmjguijcoiehyiek5i46parsu4onguk2uupznhgywymcb.py
# Topologically Sorted Source Nodes: [toks_1, index_add__6, index_add__7], Original ATen: [aten.index_select, aten.index_add]
# Source node to ATen node mapping:
#   index_add__6 => index_put_10
#   index_add__7 => index_put_11
#   toks_1 => index_2
# Graph fragment:
#   %index_2 : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_103, [None, %device_put_2]), kwargs = {})
#   %index_put_10 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_8, [None, %device_put], %index_2, True), kwargs = {})
#   %index_put_11 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_9, [None, %device_put_1], %index_2, True), kwargs = {})
triton_poi_fused_index_add_index_select_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_add_index_select_43', 'mutated_arg_names': ['out_ptr0', 'out_ptr1'], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14745600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 32768)
    x0 = xindex % 256
    x1 = (xindex // 256) % 128
    x3 = xindex % 32768
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 64, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 64), "index out of bounds: 0 <= tmp4 < 64")
    tmp7 = tl.full([XBLOCK], 234, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert((0 <= tmp10) & (tmp10 < 234), "index out of bounds: 0 <= tmp10 < 234")
    tmp12 = tl.load(in_ptr2 + (x0 + (256*(x1 // 4)) + (8192*tmp10)), None)
    tmp13 = tl.load(in_ptr3 + (x0 + (256*(x1 // 4)) + (8192*tmp10)), None)
    tmp15 = tmp13 + tmp14
    tmp16 = tmp12 + tmp15
    tmp18 = tmp17 + tmp1
    tmp19 = tmp17 < 0
    tmp20 = tl.where(tmp19, tmp18, tmp17)
    tl.device_assert((0 <= tmp20) & (tmp20 < 64), "index out of bounds: 0 <= tmp20 < 64")
    tl.atomic_add(out_ptr0 + (x3 + (32768*tmp4)), tmp16, None, sem='relaxed')
    tl.atomic_add(out_ptr1 + (x3 + (32768*tmp20)), tmp16, None, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/nm/cnmkmtf5qwffughdhpjrytrvzxfrmaer6pd3lowftnxdcyetxcqu.py
# Topologically Sorted Source Nodes: [x_mid_o, g_1], Original ATen: [aten.add, aten.mean]
# Source node to ATen node mapping:
#   g_1 => mean_8
#   x_mid_o => add_22
# Graph fragment:
#   %add_22 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_5, %view_99), kwargs = {})
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_22, [1], True), kwargs = {})
triton_per_fused_add_mean_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_44', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 59904
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (8192*x1)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*r2) + (8192*x1)), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/kx/ckxwfukdsrxy23tnkbmqb6i4mizrdm6bhbj3pulfy5hb7ulegx5t.py
# Topologically Sorted Source Nodes: [z_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   z_3 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_24, %view_124, %view_125, %expand_12], -1), kwargs = {})
triton_poi_fused_cat_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_45', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1916928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + (1024*x1)), None)
    tmp3 = tl.load(in_ptr0 + (512 + x0 + (1024*x1)), None)
    tmp5 = tl.load(in_ptr0 + (768 + x0 + (1024*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr0 + (x0 + (1024*x1)), tmp8, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/h6/ch6be5tsernebp7yaxphx5jrmq63rkfjt773ywaymhxlrqcxuxdv.py
# Topologically Sorted Source Nodes: [z_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   z_3 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_24, %view_124, %view_125, %expand_12], -1), kwargs = {})
triton_poi_fused_cat_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_46', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1916928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 256
    x2 = (xindex // 8192)
    x3 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp1 = 32.0
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + (x0 + (1024*x3)), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/at/cat74bcrctlspr6f7qyteaiv7sx6sk7xiyo2bvxk73vldw6f25mp.py
# Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_20 => add_25, erf_6, mul_30, mul_31, mul_32
# Graph fragment:
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_127, 0.5), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_127, 0.7071067811865476), kwargs = {})
#   %erf_6 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_31,), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_6, 1), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, %add_25), kwargs = {})
triton_poi_fused_gelu_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_47', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7667712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1024
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/3s/c3sqqe5el2rfafdehg4dqlyntx3m74g6blbxzseusvr36es5ac3z.py
# Topologically Sorted Source Nodes: [x_mid_o, off_stream], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   off_stream => add_26
#   x_mid_o => add_22
# Graph fragment:
#   %add_22 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_5, %view_99), kwargs = {})
#   %add_26 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_22, %view_129), kwargs = {})
triton_poi_fused_add_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_48', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1916928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/eg/cegkem46tcgi4exz4jwkcv7puvs6xloiklzgzocrix4pvx5ioyri.py
# Topologically Sorted Source Nodes: [toks_3, index_add__10, index_add__11], Original ATen: [aten.index_select, aten.index_add]
# Source node to ATen node mapping:
#   index_add__10 => index_put_16
#   index_add__11 => index_put_17
#   toks_3 => index_3
# Graph fragment:
#   %index_3 : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_158, [None, %device_put_2]), kwargs = {})
#   %index_put_16 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_14, [None, %device_put], %index_3, True), kwargs = {})
#   %index_put_17 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_15, [None, %device_put_1], %index_3, True), kwargs = {})
triton_poi_fused_index_add_index_select_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_add_index_select_49', 'mutated_arg_names': ['out_ptr0', 'out_ptr1'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14745600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 32768)
    x0 = xindex % 256
    x1 = (xindex // 256) % 128
    x3 = xindex % 32768
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 64, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 64), "index out of bounds: 0 <= tmp4 < 64")
    tmp7 = tl.full([XBLOCK], 234, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert((0 <= tmp10) & (tmp10 < 234), "index out of bounds: 0 <= tmp10 < 234")
    tmp12 = tl.load(in_ptr2 + (x0 + (256*(x1 // 4)) + (8192*tmp10)), None)
    tmp14 = tmp13 + tmp1
    tmp15 = tmp13 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp13)
    tl.device_assert((0 <= tmp16) & (tmp16 < 64), "index out of bounds: 0 <= tmp16 < 64")
    tl.atomic_add(out_ptr0 + (x3 + (32768*tmp4)), tmp12, None, sem='relaxed')
    tl.atomic_add(out_ptr1 + (x3 + (32768*tmp16)), tmp12, None, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ka/ckaw4bhusjuao4ikrb3lmbkz3ouw4mjuderhxu4ot67dfc236trp.py
# Topologically Sorted Source Nodes: [h_off_5, off_in, layer_norm_7], Original ATen: [aten.mean, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   h_off_5 => mean_10
#   layer_norm_7 => add_38, add_39, mul_43, mul_44, rsqrt_7, sub_7, var_mean_7
#   off_in => add_37
# Graph fragment:
#   %mean_10 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_185, [2]), kwargs = {})
#   %add_37 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_10, %add_26), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_37, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_37, %getitem_27), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-05), kwargs = {})
#   %rsqrt_7 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_38,), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_7), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %arg84_1), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %arg85_1), kwargs = {})
triton_per_fused_add_mean_native_layer_norm_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_native_layer_norm_50', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 11, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 7488
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), None)
    tmp3 = tl.load(in_ptr0 + (256 + r1 + (1024*x0)), None)
    tmp4 = tl.load(in_ptr1 + (256 + r1 + (1024*x0)), None)
    tmp7 = tl.load(in_ptr0 + (512 + r1 + (1024*x0)), None)
    tmp8 = tl.load(in_ptr1 + (512 + r1 + (1024*x0)), None)
    tmp11 = tl.load(in_ptr0 + (768 + r1 + (1024*x0)), None)
    tmp12 = tl.load(in_ptr1 + (768 + r1 + (1024*x0)), None)
    tmp17 = tl.load(in_out_ptr0 + (r1 + (256*x0)), None)
    tmp39 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = tl.full([1], 256, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp19 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp32 = tmp18 - tmp26
    tmp33 = 256.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp18, None)
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp42, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/a6/ca6kido2ji5t6gvh2cful2rkypdglww7bmiru34c5wi72wsohseq.py
# Topologically Sorted Source Nodes: [x_mid_d_4, h_diag_5, row_highway_9, index_add__36, col_highway_9, index_add__37, col_highway_10, index_add__41, row_highway_10, index_add__40], Original ATen: [aten.add, aten.new_zeros, aten.index_add]
# Source node to ATen node mapping:
#   col_highway_10 => full_default_43
#   col_highway_9 => full_default_41
#   h_diag_5 => add_95
#   index_add__36 => index_put_56
#   index_add__37 => index_put_57
#   index_add__40 => index_put_60
#   index_add__41 => index_put_61
#   row_highway_10 => full_default_42
#   row_highway_9 => full_default_40
#   x_mid_d_4 => add_91
# Graph fragment:
#   %add_91 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_75, %view_487), kwargs = {})
#   %add_95 : [num_users=7] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_91, %view_507), kwargs = {})
#   %full_default_40 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 64, 128, 256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_56 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_40, [None, %slice_31], %view_545, True), kwargs = {})
#   %full_default_41 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 64, 128, 256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_57 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_41, [None, %slice_31], %view_545, True), kwargs = {})
#   %full_default_43 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 64, 128, 256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_61 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_43, [None, %slice_31], %view_575, True), kwargs = {})
#   %full_default_42 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 64, 128, 256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_60 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_42, [None, %slice_31], %view_575, True), kwargs = {})
triton_poi_fused_add_index_add_new_zeros_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_index_add_new_zeros_51', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr0', 'out_ptr1', 'out_ptr2', 'out_ptr3'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x0 = xindex % 256
    x3 = (xindex // 32768)
    x2 = xindex % 32768
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x4), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x4), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp10 = tl.full([XBLOCK], 64, tl.int32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tl.device_assert((0 <= tmp13) & (tmp13 < 64), "index out of bounds: 0 <= tmp13 < 64")
    tl.store(in_out_ptr0 + (x4), tmp8, None)
    tl.atomic_add(out_ptr0 + (x2 + (32768*tmp13)), tmp8, None, sem='relaxed')
    tl.atomic_add(out_ptr1 + (x2 + (32768*tmp13)), tmp8, None, sem='relaxed')
    tl.atomic_add(out_ptr2 + (x2 + (32768*tmp13)), tmp8, None, sem='relaxed')
    tl.atomic_add(out_ptr3 + (x2 + (32768*tmp13)), tmp8, None, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/pb/cpbcucxlx7pdxcl352rgbov6c7dqdnlw44djltsov4yrjicmyldy.py
# Topologically Sorted Source Nodes: [input_74], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_74 => add_108, erf_24, mul_116, mul_117, mul_118
# Graph fragment:
#   %mul_116 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_592, 0.5), kwargs = {})
#   %mul_117 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_592, 0.7071067811865476), kwargs = {})
#   %erf_24 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_117,), kwargs = {})
#   %add_108 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_24, 1), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_116, %add_108), kwargs = {})
triton_poi_fused_gelu_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_52', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1916928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/tl/ctljesnectea34coub7ldjxenzmowf7kjofgirphjtavdkdiqqis.py
# Topologically Sorted Source Nodes: [sigmoid], Original ATen: [aten.sigmoid]
# Source node to ATen node mapping:
#   sigmoid => sigmoid
# Graph fragment:
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_605,), kwargs = {})
triton_poi_fused_sigmoid_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_53', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ee/ceefmmenp76hlwuqlr55fjt56ycvnp54ousff3e53rs4f7xt7bir.py
# Topologically Sorted Source Nodes: [packed_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   packed_1 => cat_14
# Graph fragment:
#   %cat_14 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_13, %view_606], 1), kwargs = {})
triton_poi_fused_cat_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_54', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1820672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1812480, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 1048576, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.load(in_ptr0 + (x0), tmp7, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp0 >= tmp5
    tmp10 = tl.full([1], 1288192, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tmp12 & tmp4
    tmp14 = tl.load(in_ptr1 + ((-1048576) + x0), tmp13, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp0 >= tmp10
    tmp16 = tl.full([1], 1550336, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr2 + ((-1288192) + x0), tmp19, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp0 >= tmp16
    tmp22 = tmp21 & tmp4
    tmp23 = tl.load(in_ptr3 + ((-1550336) + x0), tmp22, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp18, tmp20, tmp23)
    tmp25 = tl.where(tmp12, tmp14, tmp24)
    tmp26 = tl.where(tmp6, tmp8, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp4, tmp26, tmp27)
    tmp29 = tmp0 >= tmp3
    tmp30 = tl.full([1], 1820672, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = tl.load(in_ptr4 + ((-1812480) + x0), tmp29, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.where(tmp4, tmp28, tmp32)
    tl.store(out_ptr0 + (x0), tmp33, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 8192, 9), (73728, 9, 1))
    assert_size_stride(arg1_1, (1, 12), (12, 1))
    assert_size_stride(arg2_1, (256, 18), (18, 1))
    assert_size_stride(arg3_1, (256, ), (1, ))
    assert_size_stride(arg4_1, (256, 256), (256, 1))
    assert_size_stride(arg5_1, (256, ), (1, ))
    assert_size_stride(arg6_1, (2, 40596), (40596, 1))
    assert_size_stride(arg7_1, (40596, ), (1, ))
    assert_size_stride(arg8_1, (256, 256), (256, 1))
    assert_size_stride(arg9_1, (256, ), (1, ))
    assert_size_stride(arg10_1, (256, 256), (256, 1))
    assert_size_stride(arg11_1, (256, ), (1, ))
    assert_size_stride(arg12_1, (256, 512), (512, 1))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (256, 256), (256, 1))
    assert_size_stride(arg15_1, (256, ), (1, ))
    assert_size_stride(arg16_1, (256, 256), (256, 1))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (256, 256), (256, 1))
    assert_size_stride(arg19_1, (256, ), (1, ))
    assert_size_stride(arg20_1, (256, 512), (512, 1))
    assert_size_stride(arg21_1, (256, ), (1, ))
    assert_size_stride(arg22_1, (256, 256), (256, 1))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (256, ), (1, ))
    assert_size_stride(arg25_1, (256, ), (1, ))
    assert_size_stride(arg26_1, (256, 256), (256, 1))
    assert_size_stride(arg27_1, (256, ), (1, ))
    assert_size_stride(arg28_1, (64, 128, 128, 4), (65536, 512, 4, 1))
    assert_size_stride(arg29_1, (234, 128, 128, 4), (65536, 512, 4, 1))
    assert_size_stride(arg30_1, (234, 64), (64, 1))
    assert_size_stride(arg31_1, (234, 64), (64, 1))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (768, 256), (256, 1))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (16, 4), (4, 1))
    assert_size_stride(arg37_1, (16, ), (1, ))
    assert_size_stride(arg38_1, (8, 16), (16, 1))
    assert_size_stride(arg39_1, (8, ), (1, ))
    assert_size_stride(arg40_1, (256, 256), (256, 1))
    assert_size_stride(arg41_1, (256, ), (1, ))
    assert_size_stride(arg42_1, (64, ), (1, ))
    assert_size_stride(arg43_1, (450, ), (1, ))
    assert_size_stride(arg44_1, (450, ), (1, ))
    assert_size_stride(arg45_1, (450, ), (1, ))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (256, ), (1, ))
    assert_size_stride(arg48_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg49_1, (1024, ), (1, ))
    assert_size_stride(arg50_1, (256, 1024), (1024, 1))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (768, 256), (256, 1))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (16, 4), (4, 1))
    assert_size_stride(arg57_1, (16, ), (1, ))
    assert_size_stride(arg58_1, (8, 16), (16, 1))
    assert_size_stride(arg59_1, (8, ), (1, ))
    assert_size_stride(arg60_1, (256, 256), (256, 1))
    assert_size_stride(arg61_1, (256, ), (1, ))
    assert_size_stride(arg62_1, (256, ), (1, ))
    assert_size_stride(arg63_1, (256, ), (1, ))
    assert_size_stride(arg64_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg65_1, (1024, ), (1, ))
    assert_size_stride(arg66_1, (256, 1024), (1024, 1))
    assert_size_stride(arg67_1, (256, ), (1, ))
    assert_size_stride(arg68_1, (256, ), (1, ))
    assert_size_stride(arg69_1, (256, ), (1, ))
    assert_size_stride(arg70_1, (768, 256), (256, 1))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (16, 4), (4, 1))
    assert_size_stride(arg73_1, (16, ), (1, ))
    assert_size_stride(arg74_1, (8, 16), (16, 1))
    assert_size_stride(arg75_1, (8, ), (1, ))
    assert_size_stride(arg76_1, (256, 256), (256, 1))
    assert_size_stride(arg77_1, (256, ), (1, ))
    assert_size_stride(arg78_1, (256, ), (1, ))
    assert_size_stride(arg79_1, (256, ), (1, ))
    assert_size_stride(arg80_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg81_1, (1024, ), (1, ))
    assert_size_stride(arg82_1, (256, 1024), (1024, 1))
    assert_size_stride(arg83_1, (256, ), (1, ))
    assert_size_stride(arg84_1, (256, ), (1, ))
    assert_size_stride(arg85_1, (256, ), (1, ))
    assert_size_stride(arg86_1, (768, 256), (256, 1))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (16, 4), (4, 1))
    assert_size_stride(arg89_1, (16, ), (1, ))
    assert_size_stride(arg90_1, (8, 16), (16, 1))
    assert_size_stride(arg91_1, (8, ), (1, ))
    assert_size_stride(arg92_1, (256, 256), (256, 1))
    assert_size_stride(arg93_1, (256, ), (1, ))
    assert_size_stride(arg94_1, (256, ), (1, ))
    assert_size_stride(arg95_1, (256, ), (1, ))
    assert_size_stride(arg96_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg97_1, (1024, ), (1, ))
    assert_size_stride(arg98_1, (256, 1024), (1024, 1))
    assert_size_stride(arg99_1, (256, ), (1, ))
    assert_size_stride(arg100_1, (256, ), (1, ))
    assert_size_stride(arg101_1, (256, ), (1, ))
    assert_size_stride(arg102_1, (768, 256), (256, 1))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (16, 4), (4, 1))
    assert_size_stride(arg105_1, (16, ), (1, ))
    assert_size_stride(arg106_1, (8, 16), (16, 1))
    assert_size_stride(arg107_1, (8, ), (1, ))
    assert_size_stride(arg108_1, (256, 256), (256, 1))
    assert_size_stride(arg109_1, (256, ), (1, ))
    assert_size_stride(arg110_1, (256, ), (1, ))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg113_1, (1024, ), (1, ))
    assert_size_stride(arg114_1, (256, 1024), (1024, 1))
    assert_size_stride(arg115_1, (256, ), (1, ))
    assert_size_stride(arg116_1, (256, ), (1, ))
    assert_size_stride(arg117_1, (256, ), (1, ))
    assert_size_stride(arg118_1, (768, 256), (256, 1))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (16, 4), (4, 1))
    assert_size_stride(arg121_1, (16, ), (1, ))
    assert_size_stride(arg122_1, (8, 16), (16, 1))
    assert_size_stride(arg123_1, (8, ), (1, ))
    assert_size_stride(arg124_1, (256, 256), (256, 1))
    assert_size_stride(arg125_1, (256, ), (1, ))
    assert_size_stride(arg126_1, (256, ), (1, ))
    assert_size_stride(arg127_1, (256, ), (1, ))
    assert_size_stride(arg128_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg129_1, (1024, ), (1, ))
    assert_size_stride(arg130_1, (256, 1024), (1024, 1))
    assert_size_stride(arg131_1, (256, ), (1, ))
    assert_size_stride(arg132_1, (256, ), (1, ))
    assert_size_stride(arg133_1, (256, ), (1, ))
    assert_size_stride(arg134_1, (768, 256), (256, 1))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (16, 4), (4, 1))
    assert_size_stride(arg137_1, (16, ), (1, ))
    assert_size_stride(arg138_1, (8, 16), (16, 1))
    assert_size_stride(arg139_1, (8, ), (1, ))
    assert_size_stride(arg140_1, (256, 256), (256, 1))
    assert_size_stride(arg141_1, (256, ), (1, ))
    assert_size_stride(arg142_1, (256, ), (1, ))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg145_1, (1024, ), (1, ))
    assert_size_stride(arg146_1, (256, 1024), (1024, 1))
    assert_size_stride(arg147_1, (256, ), (1, ))
    assert_size_stride(arg148_1, (256, ), (1, ))
    assert_size_stride(arg149_1, (256, ), (1, ))
    assert_size_stride(arg150_1, (768, 256), (256, 1))
    assert_size_stride(arg151_1, (768, ), (1, ))
    assert_size_stride(arg152_1, (16, 4), (4, 1))
    assert_size_stride(arg153_1, (16, ), (1, ))
    assert_size_stride(arg154_1, (8, 16), (16, 1))
    assert_size_stride(arg155_1, (8, ), (1, ))
    assert_size_stride(arg156_1, (256, 256), (256, 1))
    assert_size_stride(arg157_1, (256, ), (1, ))
    assert_size_stride(arg158_1, (256, ), (1, ))
    assert_size_stride(arg159_1, (256, ), (1, ))
    assert_size_stride(arg160_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg161_1, (1024, ), (1, ))
    assert_size_stride(arg162_1, (256, 1024), (1024, 1))
    assert_size_stride(arg163_1, (256, ), (1, ))
    assert_size_stride(arg164_1, (256, ), (1, ))
    assert_size_stride(arg165_1, (256, ), (1, ))
    assert_size_stride(arg166_1, (768, 256), (256, 1))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (16, 4), (4, 1))
    assert_size_stride(arg169_1, (16, ), (1, ))
    assert_size_stride(arg170_1, (8, 16), (16, 1))
    assert_size_stride(arg171_1, (8, ), (1, ))
    assert_size_stride(arg172_1, (256, 256), (256, 1))
    assert_size_stride(arg173_1, (256, ), (1, ))
    assert_size_stride(arg174_1, (256, ), (1, ))
    assert_size_stride(arg175_1, (256, ), (1, ))
    assert_size_stride(arg176_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg177_1, (1024, ), (1, ))
    assert_size_stride(arg178_1, (256, 1024), (1024, 1))
    assert_size_stride(arg179_1, (256, ), (1, ))
    assert_size_stride(arg180_1, (256, ), (1, ))
    assert_size_stride(arg181_1, (256, ), (1, ))
    assert_size_stride(arg182_1, (768, 256), (256, 1))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (16, 4), (4, 1))
    assert_size_stride(arg185_1, (16, ), (1, ))
    assert_size_stride(arg186_1, (8, 16), (16, 1))
    assert_size_stride(arg187_1, (8, ), (1, ))
    assert_size_stride(arg188_1, (256, 256), (256, 1))
    assert_size_stride(arg189_1, (256, ), (1, ))
    assert_size_stride(arg190_1, (256, ), (1, ))
    assert_size_stride(arg191_1, (256, ), (1, ))
    assert_size_stride(arg192_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg193_1, (1024, ), (1, ))
    assert_size_stride(arg194_1, (256, 1024), (1024, 1))
    assert_size_stride(arg195_1, (256, ), (1, ))
    assert_size_stride(arg196_1, (256, 256), (256, 1))
    assert_size_stride(arg197_1, (256, ), (1, ))
    assert_size_stride(arg198_1, (128, 256), (256, 1))
    assert_size_stride(arg199_1, (128, ), (1, ))
    assert_size_stride(arg200_1, (256, 256), (256, 1))
    assert_size_stride(arg201_1, (256, ), (1, ))
    assert_size_stride(arg202_1, (32, 256), (256, 1))
    assert_size_stride(arg203_1, (32, ), (1, ))
    assert_size_stride(arg204_1, (256, 256), (256, 1))
    assert_size_stride(arg205_1, (256, ), (1, ))
    assert_size_stride(arg206_1, (32, 256), (256, 1))
    assert_size_stride(arg207_1, (32, ), (1, ))
    assert_size_stride(arg208_1, (1, 256), (256, 1))
    assert_size_stride(arg209_1, (1, ), (1, ))
    assert_size_stride(arg210_1, (32, 256), (256, 1))
    assert_size_stride(arg211_1, (32, ), (1, ))
    assert_size_stride(arg212_1, (32, 256), (256, 1))
    assert_size_stride(arg213_1, (32, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8192, 20), (20, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(arg0_1, arg1_1, buf0, 163840, grid=grid(163840), stream=stream0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((20, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(arg2_1, buf1, 5120, grid=grid(5120), stream=stream0)
        del arg2_1
        buf2 = empty_strided_cuda((8192, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf0, buf1, out=buf2)
        del buf0
        del buf1
        buf3 = reinterpret_tensor(buf2, (1, 8192, 256), (2097152, 256, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf3, arg3_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg3_1
        buf4 = empty_strided_cuda((8192, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg5_1, reinterpret_tensor(buf3, (8192, 256), (256, 1), 0), reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf4)
        del arg4_1
        del arg5_1
        buf9 = empty_strided_cuda((8192, 512), (512, 1), torch.float32)
        buf5 = reinterpret_tensor(buf9, (8192, 256), (512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [self_features], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, buf4, reinterpret_tensor(arg10_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf5)
        del arg10_1
        del arg11_1
        buf6 = reinterpret_tensor(buf3, (8192, 256), (256, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [neighbor_features], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg9_1, buf4, reinterpret_tensor(arg8_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf6)
        del arg8_1
        del arg9_1
        buf7 = reinterpret_tensor(buf9, (8192, 256), (512, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [aggr], Original ATen: [aten.zeros_like]
        triton_poi_fused_zeros_like_3.run(buf7, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [aggr, getitem_4, messages, index_add_], Original ATen: [aten.zeros_like, aten.index, aten.mul, aten.index_add]
        triton_poi_fused_index_index_add_mul_zeros_like_4.run(arg6_1, buf6, arg7_1, buf7, 10392576, grid=grid(10392576), stream=stream0)
        del buf5
        del buf7
        buf10 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf9, reinterpret_tensor(arg12_1, (512, 256), (1, 512), 0), out=buf10)
        del arg12_1
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf11, arg13_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg13_1
        buf12 = empty_strided_cuda((8192, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.gelu]
        extern_kernels.mm(buf11, reinterpret_tensor(arg14_1, (256, 256), (1, 256), 0), out=buf12)
        del arg14_1
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.add]
        triton_poi_fused_add_5.run(buf13, buf4, arg15_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg15_1
        buf18 = buf9; del buf9  # reuse
        buf14 = reinterpret_tensor(buf18, (8192, 256), (512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [self_features_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg19_1, buf13, reinterpret_tensor(arg18_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf14)
        del arg18_1
        del arg19_1
        buf15 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [neighbor_features_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg17_1, buf13, reinterpret_tensor(arg16_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf15)
        del arg16_1
        del arg17_1
        buf16 = reinterpret_tensor(buf18, (8192, 256), (512, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [aggr_1], Original ATen: [aten.zeros_like]
        triton_poi_fused_zeros_like_3.run(buf16, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [aggr_1, getitem_7, messages_1, index_add__1], Original ATen: [aten.zeros_like, aten.index, aten.mul, aten.index_add]
        triton_poi_fused_index_index_add_mul_zeros_like_4.run(arg6_1, buf15, arg7_1, buf16, 10392576, grid=grid(10392576), stream=stream0)
        del arg6_1
        del arg7_1
        del buf14
        del buf16
        buf19 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf18, reinterpret_tensor(arg20_1, (512, 256), (1, 512), 0), out=buf19)
        del arg20_1
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf20, arg21_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg21_1
        buf21 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.gelu]
        extern_kernels.mm(buf20, reinterpret_tensor(arg22_1, (256, 256), (1, 256), 0), out=buf21)
        del arg22_1
        buf25 = reinterpret_tensor(buf20, (1, 8192, 256), (2097152, 256, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [h_2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_6.run(buf13, buf21, arg23_1, arg24_1, arg25_1, buf25, 8192, 256, grid=grid(8192), stream=stream0)
        del arg23_1
        del arg24_1
        del arg25_1
        buf26 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg27_1, reinterpret_tensor(buf25, (8192, 256), (256, 1), 0), reinterpret_tensor(arg26_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf26)
        del arg26_1
        del arg27_1
        buf30 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_7.run(buf26, arg32_1, arg33_1, buf30, 8192, 256, grid=grid(8192), stream=stream0)
        del arg32_1
        del arg33_1
        buf31 = empty_strided_cuda((8192, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (8192, 256), (256, 1), 0), reinterpret_tensor(arg34_1, (256, 768), (1, 256), 0), out=buf31)
        del arg34_1
        buf32 = reinterpret_tensor(buf18, (1, 64, 128, 128, 4), (4194304, 65536, 512, 4, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_8.run(arg28_1, buf32, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_9.run(buf32, 32768, grid=grid(32768), stream=stream0)
        buf34 = empty_strided_cuda((1, 64, 128, 128), (1048576, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_10.run(buf32, buf34, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_11.run(buf34, 8192, grid=grid(8192), stream=stream0)
        buf36 = reinterpret_tensor(buf30, (64, 8, 128, 32), (32768, 4096, 32, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [q_f, k_f, v_f, logit_bias, x_soft], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_12.run(buf31, arg35_1, buf36, 2097152, grid=grid(2097152), stream=stream0)
        buf37 = reinterpret_tensor(buf13, (64, 8, 128, 32), (32768, 4096, 32, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [q_f, k_f, v_f, logit_bias, x_soft], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_13.run(buf31, arg35_1, buf37, 2097152, grid=grid(2097152), stream=stream0)
        buf38 = empty_strided_cuda((64, 8, 128, 32), (32768, 4096, 32, 1), torch.float32)
        buf50 = empty_strided_cuda((64, 8, 128, 1, 32, 1), (32768, 4096, 32, 32, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f, k_f, v_f, logit_bias, x_soft, einsum_2], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_14.run(buf31, arg35_1, buf38, buf50, 2097152, grid=grid(2097152), stream=stream0)
        del arg35_1
        buf39 = empty_strided_cuda((64, 8, 128, 128), (131072, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f, k_f, v_f, logit_bias, x_soft], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_15.run(buf34, buf32, buf39, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [q_f, k_f, v_f, logit_bias, x_soft], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf40 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf36, buf37, buf38, buf39, False)
        del buf36
        buf41 = buf40[0]
        del buf40
        buf45 = empty_strided_cuda((1, 64, 128, 128, 4), (4194304, 65536, 512, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(buf34, buf32, buf45, 4194304, grid=grid(4194304), stream=stream0)
        buf46 = empty_strided_cuda((1048576, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf45, (1048576, 4), (4, 1), 0), reinterpret_tensor(arg36_1, (4, 16), (1, 4), 0), out=buf46)
        del arg36_1
        buf47 = reinterpret_tensor(buf46, (1, 64, 128, 128, 16), (16777216, 262144, 2048, 16, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_17.run(buf47, arg37_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg37_1
        buf48 = reinterpret_tensor(buf39, (1048576, 8), (8, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf47, (1048576, 16), (16, 1), 0), reinterpret_tensor(arg38_1, (16, 8), (1, 16), 0), out=buf48)
        del arg38_1
        buf49 = empty_strided_cuda((64, 8, 128, 128, 1, 1), (131072, 16384, 128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf48, arg39_1, buf49, 512, 16384, grid=grid(512, 16384), stream=stream0)
        del arg39_1
        buf51 = reinterpret_tensor(buf38, (512, 128, 32), (4096, 32, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf49, (512, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf50, (512, 128, 32), (4096, 32, 1), 0), out=buf51)
        buf52 = reinterpret_tensor(buf41, (1, 64, 128, 8, 32), (2097152, 32768, 256, 32, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_mid], Original ATen: [aten.add]
        triton_poi_fused_add_19.run(buf52, buf51, 2097152, grid=grid(2097152), stream=stream0)
        buf53 = reinterpret_tensor(buf51, (8192, 256), (256, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf52, (8192, 256), (256, 1), 0), reinterpret_tensor(arg40_1, (256, 256), (1, 256), 0), out=buf53)
        del arg40_1
        buf57 = reinterpret_tensor(buf52, (1, 64, 128, 256), (2097152, 32768, 256, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [row_highway], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf57, 2097152, grid=grid(2097152), stream=stream0)
        buf59 = reinterpret_tensor(buf50, (1, 64, 128, 256), (2097152, 32768, 256, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [col_highway], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf59, 2097152, grid=grid(2097152), stream=stream0)
        buf67 = reinterpret_tensor(buf49, (1, 8192, 1024), (8388608, 1024, 1), 0); del buf49  # reuse
        buf63 = reinterpret_tensor(buf67, (1, 8192, 256), (8388608, 1024, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_mid_d, z, row_highway, index_add__2, col_highway, index_add__3], Original ATen: [aten.add, aten.native_layer_norm, aten.new_zeros, aten.index_add]
        triton_per_fused_add_index_add_native_layer_norm_new_zeros_21.run(buf26, buf53, arg41_1, arg42_1, arg46_1, arg47_1, buf57, buf59, buf63, 8192, 256, grid=grid(8192), stream=stream0)
        del arg46_1
        del arg47_1
        buf61 = empty_strided_cuda((1, 1, 256, 64), (16384, 16384, 1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_mid_d, g], Original ATen: [aten.add, aten.mean]
        triton_red_fused_add_mean_22.run(buf26, buf53, arg41_1, buf61, 16384, 128, grid=grid(16384), stream=stream0)
        buf62 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_mid_d, g], Original ATen: [aten.add, aten.mean]
        triton_per_fused_add_mean_23.run(buf61, buf62, 256, 64, grid=grid(256), stream=stream0)
        buf64 = reinterpret_tensor(buf67, (1, 8192, 256), (8388608, 1024, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [z_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_24.run(buf57, buf64, 2097152, grid=grid(2097152), stream=stream0)
        buf65 = reinterpret_tensor(buf67, (1, 8192, 256), (8388608, 1024, 1), 512)  # alias
        # Topologically Sorted Source Nodes: [z_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_24.run(buf59, buf65, 2097152, grid=grid(2097152), stream=stream0)
        buf66 = reinterpret_tensor(buf67, (1, 8192, 256), (8388608, 1024, 1), 768)  # alias
        # Topologically Sorted Source Nodes: [z_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_25.run(buf62, buf66, 2097152, grid=grid(2097152), stream=stream0)
        del buf63
        del buf64
        del buf65
        del buf66
        buf68 = reinterpret_tensor(buf48, (8192, 1024), (1024, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf67, (8192, 1024), (1024, 1), 0), reinterpret_tensor(arg48_1, (1024, 1024), (1, 1024), 0), out=buf68)
        del arg48_1
        buf69 = reinterpret_tensor(buf68, (1, 8192, 1024), (8388608, 1024, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_26.run(buf69, arg49_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg49_1
        buf70 = reinterpret_tensor(buf59, (8192, 256), (256, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf69, (8192, 1024), (1024, 1), 0), reinterpret_tensor(arg50_1, (1024, 256), (1, 1024), 0), out=buf70)
        del arg50_1
        buf71 = reinterpret_tensor(buf70, (1, 8192, 256), (2097152, 256, 1), 0); del buf70  # reuse
        buf109 = reinterpret_tensor(buf57, (1, 8192, 256), (2097152, 256, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d, h_diag_1, layer_norm_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf71, buf26, buf53, arg41_1, arg51_1, arg68_1, arg69_1, buf109, 8192, 256, grid=grid(8192), stream=stream0)
        del arg41_1
        del arg51_1
        del arg68_1
        del arg69_1
        buf72 = empty_strided_cuda((1, 234, 32768), (7667712, 32768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [row_p_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg30_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf71, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf72)
        buf73 = empty_strided_cuda((1, 234, 32768), (7667712, 32768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [col_p_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg31_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf71, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf73)
        buf74 = empty_strided_cuda((234, 32, 256), (8192, 256, 1), torch.float32)
        buf78 = empty_strided_cuda((234, 32, 256), (8192, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_off_3, layer_norm_3], Original ATen: [aten.mean, aten.native_layer_norm]
        triton_per_fused_mean_native_layer_norm_28.run(buf72, buf73, arg52_1, arg53_1, buf74, buf78, 7488, 256, grid=grid(7488), stream=stream0)
        del arg52_1
        del arg53_1
        buf79 = empty_strided_cuda((7488, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (7488, 256), (256, 1), 0), reinterpret_tensor(arg54_1, (256, 768), (1, 256), 0), out=buf79)
        del arg54_1
        buf81 = empty_strided_cuda((234, 1, 32, 32, 4), (4096, 958464, 128, 4, 1), torch.float32)
        buf182 = empty_strided_cuda((234, 1, 32, 32, 4), (4096, 958464, 128, 4, 1), torch.float32)
        buf280 = empty_strided_cuda((234, 1, 32, 32, 4), (4096, 958464, 128, 4, 1), torch.float32)
        buf378 = empty_strided_cuda((234, 1, 32, 32, 4), (4096, 958464, 128, 4, 1), torch.float32)
        buf476 = empty_strided_cuda((234, 32, 32, 4), (4096, 128, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sub, setitem_2, setitem_6, setitem_10, setitem_14], Original ATen: [aten.mean, aten.lift_fresh, aten.index_put]
        triton_per_fused_index_put_lift_fresh_mean_29.run(arg29_1, buf81, buf182, buf280, buf378, buf476, 958464, 16, grid=grid(958464), stream=stream0)
        del arg29_1
        # Topologically Sorted Source Nodes: [setitem_2], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_30.run(buf81, 29952, grid=grid(29952), stream=stream0)
        buf83 = empty_strided_cuda((234, 1, 32, 32), (1024, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_31.run(buf81, buf83, 239616, grid=grid(239616), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_32.run(buf83, 7488, grid=grid(7488), stream=stream0)
        buf85 = reinterpret_tensor(buf78, (234, 8, 32, 32), (8192, 1024, 32, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, logit_bias_1, x_soft_2], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_33.run(buf79, arg55_1, buf85, 1916928, grid=grid(1916928), stream=stream0)
        buf86 = empty_strided_cuda((234, 8, 32, 32), (8192, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, logit_bias_1, x_soft_2], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_34.run(buf79, arg55_1, buf86, 1916928, grid=grid(1916928), stream=stream0)
        buf87 = empty_strided_cuda((234, 8, 32, 32), (8192, 1024, 32, 1), torch.float32)
        buf99 = empty_strided_cuda((234, 8, 32, 1, 32, 1), (8192, 1024, 32, 32, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, logit_bias_1, x_soft_2, einsum_5], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_35.run(buf79, arg55_1, buf87, buf99, 1916928, grid=grid(1916928), stream=stream0)
        del arg55_1
        buf88 = empty_strided_cuda((234, 8, 32, 32), (8192, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, logit_bias_1, x_soft_2], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_36.run(buf83, buf81, buf88, 1916928, grid=grid(1916928), stream=stream0)
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, logit_bias_1, x_soft_2], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf89 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf85, buf86, buf87, buf88, False)
        del buf85
        buf90 = buf89[0]
        del buf89
        buf94 = empty_strided_cuda((234, 1, 32, 32, 4), (4096, 1, 128, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_37.run(buf83, buf81, buf94, 958464, grid=grid(958464), stream=stream0)
        del buf81
        buf95 = empty_strided_cuda((239616, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (239616, 4), (4, 1), 0), reinterpret_tensor(arg56_1, (4, 16), (1, 4), 0), out=buf95)
        del arg56_1
        buf96 = reinterpret_tensor(buf95, (234, 1, 32, 32, 16), (16384, 1, 512, 16, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_38.run(buf96, arg57_1, 3833856, grid=grid(3833856), stream=stream0)
        del arg57_1
        buf97 = reinterpret_tensor(buf88, (239616, 8), (8, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf96, (239616, 16), (16, 1), 0), reinterpret_tensor(arg58_1, (16, 8), (1, 16), 0), out=buf97)
        del arg58_1
        buf98 = reinterpret_tensor(buf87, (234, 8, 32, 32, 1, 1), (8192, 1024, 32, 1, 1, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [einsum_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf97, arg59_1, buf98, 1872, 1024, grid=grid(1872, 1024), stream=stream0)
        del arg59_1
        buf100 = reinterpret_tensor(buf97, (1872, 32, 32), (1024, 32, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [einsum_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf98, (1872, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf99, (1872, 32, 32), (1024, 32, 1), 0), out=buf100)
        buf101 = reinterpret_tensor(buf90, (234, 1, 32, 8, 32), (8192, 1, 256, 32, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [x_mid_1], Original ATen: [aten.add]
        triton_poi_fused_add_40.run(buf101, buf100, 1916928, grid=grid(1916928), stream=stream0)
        buf102 = reinterpret_tensor(buf100, (7488, 256), (256, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (7488, 256), (256, 1), 0), reinterpret_tensor(arg60_1, (256, 256), (1, 256), 0), out=buf102)
        del arg60_1
        buf154 = reinterpret_tensor(buf73, (234, 32, 1024), (32768, 1024, 1), 0); del buf73  # reuse
        buf150 = reinterpret_tensor(buf154, (234, 32, 256), (32768, 1024, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_mid_o, z_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_41.run(buf74, buf102, arg61_1, arg62_1, arg63_1, buf150, 7488, 256, grid=grid(7488), stream=stream0)
        del arg62_1
        del arg63_1
        buf110 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf109, (8192, 256), (256, 1), 0), reinterpret_tensor(arg70_1, (256, 768), (1, 256), 0), out=buf110)
        del arg70_1
        buf111 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [setitem_4], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_8.run(arg28_1, buf111, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_4], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_9.run(buf111, 32768, grid=grid(32768), stream=stream0)
        buf113 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [setitem_5], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_10.run(buf111, buf113, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_5], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_11.run(buf113, 8192, grid=grid(8192), stream=stream0)
        buf115 = reinterpret_tensor(buf109, (64, 8, 128, 32), (32768, 4096, 32, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [q_f_2, k_f_2, v_f_2, logit_bias_2, x_soft_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_12.run(buf110, arg71_1, buf115, 2097152, grid=grid(2097152), stream=stream0)
        buf116 = reinterpret_tensor(buf53, (64, 8, 128, 32), (32768, 4096, 32, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [q_f_2, k_f_2, v_f_2, logit_bias_2, x_soft_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_13.run(buf110, arg71_1, buf116, 2097152, grid=grid(2097152), stream=stream0)
        buf117 = reinterpret_tensor(buf26, (64, 8, 128, 32), (32768, 4096, 32, 1), 0); del buf26  # reuse
        buf129 = reinterpret_tensor(buf37, (64, 8, 128, 1, 32, 1), (32768, 4096, 32, 32, 1, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [q_f_2, k_f_2, v_f_2, logit_bias_2, x_soft_4, einsum_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_14.run(buf110, arg71_1, buf117, buf129, 2097152, grid=grid(2097152), stream=stream0)
        del arg71_1
        buf118 = reinterpret_tensor(buf69, (64, 8, 128, 128), (131072, 16384, 128, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [q_f_2, k_f_2, v_f_2, logit_bias_2, x_soft_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_15.run(buf113, buf111, buf118, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [q_f_2, k_f_2, v_f_2, logit_bias_2, x_soft_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf119 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf115, buf116, buf117, buf118, False)
        buf120 = buf119[0]
        del buf119
        buf124 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(buf113, buf111, buf124, 4194304, grid=grid(4194304), stream=stream0)
        buf125 = reinterpret_tensor(buf47, (1048576, 16), (16, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (1048576, 4), (4, 1), 0), reinterpret_tensor(arg72_1, (4, 16), (1, 4), 0), out=buf125)
        del arg72_1
        buf126 = reinterpret_tensor(buf125, (1, 64, 128, 128, 16), (16777216, 262144, 2048, 16, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_17.run(buf126, arg73_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg73_1
        buf127 = reinterpret_tensor(buf118, (1048576, 8), (8, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf126, (1048576, 16), (16, 1), 0), reinterpret_tensor(arg74_1, (16, 8), (1, 16), 0), out=buf127)
        del arg74_1
        buf128 = reinterpret_tensor(buf67, (64, 8, 128, 128, 1, 1), (131072, 16384, 128, 1, 1, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [einsum_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf127, arg75_1, buf128, 512, 16384, grid=grid(512, 16384), stream=stream0)
        del arg75_1
        buf130 = reinterpret_tensor(buf117, (512, 128, 32), (4096, 32, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [einsum_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf128, (512, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf129, (512, 128, 32), (4096, 32, 1), 0), out=buf130)
        buf131 = reinterpret_tensor(buf120, (1, 64, 128, 8, 32), (2097152, 32768, 256, 32, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [x_mid_2], Original ATen: [aten.add]
        triton_poi_fused_add_19.run(buf131, buf130, 2097152, grid=grid(2097152), stream=stream0)
        buf132 = reinterpret_tensor(buf130, (8192, 256), (256, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf131, (8192, 256), (256, 1), 0), reinterpret_tensor(arg76_1, (256, 256), (1, 256), 0), out=buf132)
        del arg76_1
        buf136 = reinterpret_tensor(buf131, (1, 64, 128, 256), (2097152, 32768, 256, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [row_highway_2], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf136, 2097152, grid=grid(2097152), stream=stream0)
        buf138 = reinterpret_tensor(buf129, (1, 64, 128, 256), (2097152, 32768, 256, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [row_highway_1], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf138, 2097152, grid=grid(2097152), stream=stream0)
        buf144 = reinterpret_tensor(buf116, (1, 64, 128, 256), (2097152, 32768, 256, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [col_highway_1], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf144, 2097152, grid=grid(2097152), stream=stream0)
        buf160 = reinterpret_tensor(buf115, (1, 64, 128, 256), (2097152, 32768, 256, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [col_highway_2], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf160, 2097152, grid=grid(2097152), stream=stream0)
        buf169 = reinterpret_tensor(buf128, (1, 8192, 1024), (8388608, 1024, 1), 0); del buf128  # reuse
        buf165 = reinterpret_tensor(buf169, (1, 8192, 256), (8388608, 1024, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_mid_d_1, z_4, row_highway_2, index_add__8, row_highway_1, index_add__4, col_highway_1, index_add__5, col_highway_2, index_add__9], Original ATen: [aten.add, aten.native_layer_norm, aten.new_zeros, aten.index_add]
        triton_per_fused_add_index_add_native_layer_norm_new_zeros_42.run(buf71, buf132, arg77_1, arg42_1, arg78_1, arg79_1, buf136, buf160, buf165, buf138, buf144, 8192, 256, grid=grid(8192), stream=stream0)
        del arg78_1
        del arg79_1
        buf140 = empty_strided_cuda((450, ), (1, ), torch.int64)
        buf140.copy_(arg43_1)
        del arg43_1
        buf141 = empty_strided_cuda((450, ), (1, ), torch.int64)
        buf141.copy_(arg45_1)
        del arg45_1
        buf146 = empty_strided_cuda((450, ), (1, ), torch.int64)
        buf146.copy_(arg44_1)
        del arg44_1
        # Topologically Sorted Source Nodes: [toks_1, index_add__6, index_add__7], Original ATen: [aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_43.run(buf140, buf141, buf74, buf102, arg61_1, buf146, buf138, buf144, 14745600, grid=grid(14745600), stream=stream0)
        buf143 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [row_t], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg30_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf138, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf143)
        del buf138
        buf148 = empty_strided_cuda((1, 234, 32768), (7667712, 32768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [col_t], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg31_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf144, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf148)
        buf149 = empty_strided_cuda((234, 1, 256), (256, 59904, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_mid_o, g_1], Original ATen: [aten.add, aten.mean]
        triton_per_fused_add_mean_44.run(buf74, buf102, arg61_1, buf149, 59904, 32, grid=grid(59904), stream=stream0)
        buf151 = reinterpret_tensor(buf154, (234, 32, 256), (32768, 1024, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [z_3], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf143, buf151, 1916928, grid=grid(1916928), stream=stream0)
        buf152 = reinterpret_tensor(buf154, (234, 32, 256), (32768, 1024, 1), 512)  # alias
        # Topologically Sorted Source Nodes: [z_3], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf148, buf152, 1916928, grid=grid(1916928), stream=stream0)
        buf153 = reinterpret_tensor(buf154, (234, 32, 256), (32768, 1024, 1), 768)  # alias
        # Topologically Sorted Source Nodes: [z_3], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf149, buf153, 1916928, grid=grid(1916928), stream=stream0)
        del buf150
        del buf151
        del buf152
        del buf153
        buf155 = reinterpret_tensor(buf148, (7488, 1024), (1024, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf154, (7488, 1024), (1024, 1), 0), reinterpret_tensor(arg64_1, (1024, 1024), (1, 1024), 0), out=buf155)
        del arg64_1
        buf156 = reinterpret_tensor(buf155, (234, 32, 1024), (32768, 1024, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_47.run(buf156, arg65_1, 7667712, grid=grid(7667712), stream=stream0)
        del arg65_1
        buf157 = reinterpret_tensor(buf101, (7488, 256), (256, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf156, (7488, 1024), (1024, 1), 0), reinterpret_tensor(arg66_1, (1024, 256), (1, 1024), 0), out=buf157)
        del arg66_1
        buf158 = reinterpret_tensor(buf157, (234, 32, 256), (8192, 256, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [x_mid_o, off_stream], Original ATen: [aten.add]
        triton_poi_fused_add_48.run(buf158, buf74, buf102, arg61_1, arg67_1, 1916928, grid=grid(1916928), stream=stream0)
        del arg61_1
        del arg67_1
        # Topologically Sorted Source Nodes: [toks_3, index_add__10, index_add__11], Original ATen: [aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_49.run(buf140, buf141, buf158, buf146, buf136, buf160, 14745600, grid=grid(14745600), stream=stream0)
        buf163 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d_1, g_2], Original ATen: [aten.add, aten.mean]
        triton_red_fused_add_mean_22.run(buf71, buf132, arg77_1, buf163, 16384, 128, grid=grid(16384), stream=stream0)
        buf164 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d_1, g_2], Original ATen: [aten.add, aten.mean]
        triton_per_fused_add_mean_23.run(buf163, buf164, 256, 64, grid=grid(256), stream=stream0)
        buf166 = reinterpret_tensor(buf169, (1, 8192, 256), (8388608, 1024, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [z_5], Original ATen: [aten.cat]
        triton_poi_fused_cat_24.run(buf136, buf166, 2097152, grid=grid(2097152), stream=stream0)
        buf167 = reinterpret_tensor(buf169, (1, 8192, 256), (8388608, 1024, 1), 512)  # alias
        # Topologically Sorted Source Nodes: [z_5], Original ATen: [aten.cat]
        triton_poi_fused_cat_24.run(buf160, buf167, 2097152, grid=grid(2097152), stream=stream0)
        buf168 = reinterpret_tensor(buf169, (1, 8192, 256), (8388608, 1024, 1), 768)  # alias
        # Topologically Sorted Source Nodes: [z_5], Original ATen: [aten.cat]
        triton_poi_fused_cat_25.run(buf164, buf168, 2097152, grid=grid(2097152), stream=stream0)
        del buf165
        del buf166
        del buf167
        del buf168
        buf170 = reinterpret_tensor(buf127, (8192, 1024), (1024, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (8192, 1024), (1024, 1), 0), reinterpret_tensor(arg80_1, (1024, 1024), (1, 1024), 0), out=buf170)
        del arg80_1
        buf171 = reinterpret_tensor(buf170, (1, 8192, 1024), (8388608, 1024, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_26.run(buf171, arg81_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg81_1
        buf172 = reinterpret_tensor(buf160, (8192, 256), (256, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (8192, 1024), (1024, 1), 0), reinterpret_tensor(arg82_1, (1024, 256), (1, 1024), 0), out=buf172)
        del arg82_1
        buf173 = reinterpret_tensor(buf172, (1, 8192, 256), (2097152, 256, 1), 0); del buf172  # reuse
        buf210 = reinterpret_tensor(buf136, (1, 8192, 256), (2097152, 256, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d_1, h_diag_2, layer_norm_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf173, buf71, buf132, arg77_1, arg83_1, arg100_1, arg101_1, buf210, 8192, 256, grid=grid(8192), stream=stream0)
        del arg100_1
        del arg101_1
        del arg77_1
        del arg83_1
        buf174 = reinterpret_tensor(buf156, (1, 234, 32768), (7667712, 32768, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [row_p_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg30_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf173, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf174)
        buf175 = reinterpret_tensor(buf154, (1, 234, 32768), (7667712, 32768, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [col_p_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg31_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf173, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf175)
        buf176 = buf158; del buf158  # reuse
        buf180 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [h_off_5, off_in, layer_norm_7], Original ATen: [aten.mean, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mean_native_layer_norm_50.run(buf176, buf174, buf175, arg84_1, arg85_1, buf180, 7488, 256, grid=grid(7488), stream=stream0)
        del arg84_1
        del arg85_1
        buf181 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf180, (7488, 256), (256, 1), 0), reinterpret_tensor(arg86_1, (256, 768), (1, 256), 0), out=buf181)
        del arg86_1
        # Topologically Sorted Source Nodes: [setitem_6], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_30.run(buf182, 29952, grid=grid(29952), stream=stream0)
        buf184 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [setitem_7], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_31.run(buf182, buf184, 239616, grid=grid(239616), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_7], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_32.run(buf184, 7488, grid=grid(7488), stream=stream0)
        buf186 = reinterpret_tensor(buf180, (234, 8, 32, 32), (8192, 1024, 32, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [q_f_3, k_f_3, v_f_3, logit_bias_3, x_soft_6], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_33.run(buf181, arg87_1, buf186, 1916928, grid=grid(1916928), stream=stream0)
        buf187 = reinterpret_tensor(buf102, (234, 8, 32, 32), (8192, 1024, 32, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [q_f_3, k_f_3, v_f_3, logit_bias_3, x_soft_6], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_34.run(buf181, arg87_1, buf187, 1916928, grid=grid(1916928), stream=stream0)
        buf188 = reinterpret_tensor(buf99, (234, 8, 32, 32), (8192, 1024, 32, 1), 0); del buf99  # reuse
        buf200 = reinterpret_tensor(buf98, (234, 8, 32, 1, 32, 1), (8192, 1024, 32, 32, 1, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [q_f_3, k_f_3, v_f_3, logit_bias_3, x_soft_6, einsum_11], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_35.run(buf181, arg87_1, buf188, buf200, 1916928, grid=grid(1916928), stream=stream0)
        del arg87_1
        buf189 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [q_f_3, k_f_3, v_f_3, logit_bias_3, x_soft_6], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_36.run(buf184, buf182, buf189, 1916928, grid=grid(1916928), stream=stream0)
        # Topologically Sorted Source Nodes: [q_f_3, k_f_3, v_f_3, logit_bias_3, x_soft_6], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf190 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf186, buf187, buf188, buf189, False)
        del buf186
        buf191 = buf190[0]
        del buf190
        buf195 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_37.run(buf184, buf182, buf195, 958464, grid=grid(958464), stream=stream0)
        del buf182
        buf196 = reinterpret_tensor(buf96, (239616, 16), (16, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf195, (239616, 4), (4, 1), 0), reinterpret_tensor(arg88_1, (4, 16), (1, 4), 0), out=buf196)
        del arg88_1
        buf197 = reinterpret_tensor(buf196, (234, 1, 32, 32, 16), (16384, 1, 512, 16, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_38.run(buf197, arg89_1, 3833856, grid=grid(3833856), stream=stream0)
        del arg89_1
        buf198 = reinterpret_tensor(buf189, (239616, 8), (8, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf197, (239616, 16), (16, 1), 0), reinterpret_tensor(arg90_1, (16, 8), (1, 16), 0), out=buf198)
        del arg90_1
        buf199 = reinterpret_tensor(buf188, (234, 8, 32, 32, 1, 1), (8192, 1024, 32, 1, 1, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [einsum_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf198, arg91_1, buf199, 1872, 1024, grid=grid(1872, 1024), stream=stream0)
        del arg91_1
        buf201 = reinterpret_tensor(buf198, (1872, 32, 32), (1024, 32, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [einsum_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf199, (1872, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf200, (1872, 32, 32), (1024, 32, 1), 0), out=buf201)
        buf202 = reinterpret_tensor(buf191, (234, 1, 32, 8, 32), (8192, 1, 256, 32, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [x_mid_3], Original ATen: [aten.add]
        triton_poi_fused_add_40.run(buf202, buf201, 1916928, grid=grid(1916928), stream=stream0)
        buf203 = reinterpret_tensor(buf201, (7488, 256), (256, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf202, (7488, 256), (256, 1), 0), reinterpret_tensor(arg92_1, (256, 256), (1, 256), 0), out=buf203)
        del arg92_1
        buf252 = reinterpret_tensor(buf175, (234, 32, 1024), (32768, 1024, 1), 0); del buf175  # reuse
        buf248 = reinterpret_tensor(buf252, (234, 32, 256), (32768, 1024, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_mid_o_1, z_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_41.run(buf176, buf203, arg93_1, arg94_1, arg95_1, buf248, 7488, 256, grid=grid(7488), stream=stream0)
        del arg94_1
        del arg95_1
        buf211 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (8192, 256), (256, 1), 0), reinterpret_tensor(arg102_1, (256, 768), (1, 256), 0), out=buf211)
        del arg102_1
        buf212 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [setitem_8], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_8.run(arg28_1, buf212, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_8], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_9.run(buf212, 32768, grid=grid(32768), stream=stream0)
        buf214 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [setitem_9], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_10.run(buf212, buf214, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_9], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_11.run(buf214, 8192, grid=grid(8192), stream=stream0)
        buf216 = reinterpret_tensor(buf210, (64, 8, 128, 32), (32768, 4096, 32, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [q_f_4, k_f_4, v_f_4, logit_bias_4, x_soft_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_12.run(buf211, arg103_1, buf216, 2097152, grid=grid(2097152), stream=stream0)
        buf217 = reinterpret_tensor(buf71, (64, 8, 128, 32), (32768, 4096, 32, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [q_f_4, k_f_4, v_f_4, logit_bias_4, x_soft_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_13.run(buf211, arg103_1, buf217, 2097152, grid=grid(2097152), stream=stream0)
        buf218 = reinterpret_tensor(buf132, (64, 8, 128, 32), (32768, 4096, 32, 1), 0); del buf132  # reuse
        buf230 = reinterpret_tensor(buf144, (64, 8, 128, 1, 32, 1), (32768, 4096, 32, 32, 1, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [q_f_4, k_f_4, v_f_4, logit_bias_4, x_soft_8, einsum_14], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_14.run(buf211, arg103_1, buf218, buf230, 2097152, grid=grid(2097152), stream=stream0)
        del arg103_1
        buf219 = reinterpret_tensor(buf171, (64, 8, 128, 128), (131072, 16384, 128, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [q_f_4, k_f_4, v_f_4, logit_bias_4, x_soft_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_15.run(buf214, buf212, buf219, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [q_f_4, k_f_4, v_f_4, logit_bias_4, x_soft_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf220 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf216, buf217, buf218, buf219, False)
        buf221 = buf220[0]
        del buf220
        buf225 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(buf214, buf212, buf225, 4194304, grid=grid(4194304), stream=stream0)
        buf226 = reinterpret_tensor(buf126, (1048576, 16), (16, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf225, (1048576, 4), (4, 1), 0), reinterpret_tensor(arg104_1, (4, 16), (1, 4), 0), out=buf226)
        del arg104_1
        buf227 = reinterpret_tensor(buf226, (1, 64, 128, 128, 16), (16777216, 262144, 2048, 16, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_17.run(buf227, arg105_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg105_1
        buf228 = reinterpret_tensor(buf219, (1048576, 8), (8, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf227, (1048576, 16), (16, 1), 0), reinterpret_tensor(arg106_1, (16, 8), (1, 16), 0), out=buf228)
        del arg106_1
        buf229 = reinterpret_tensor(buf169, (64, 8, 128, 128, 1, 1), (131072, 16384, 128, 1, 1, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [einsum_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf228, arg107_1, buf229, 512, 16384, grid=grid(512, 16384), stream=stream0)
        del arg107_1
        buf231 = reinterpret_tensor(buf218, (512, 128, 32), (4096, 32, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [einsum_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf229, (512, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf230, (512, 128, 32), (4096, 32, 1), 0), out=buf231)
        buf232 = reinterpret_tensor(buf221, (1, 64, 128, 8, 32), (2097152, 32768, 256, 32, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [x_mid_4], Original ATen: [aten.add]
        triton_poi_fused_add_19.run(buf232, buf231, 2097152, grid=grid(2097152), stream=stream0)
        buf233 = reinterpret_tensor(buf231, (8192, 256), (256, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (8192, 256), (256, 1), 0), reinterpret_tensor(arg108_1, (256, 256), (1, 256), 0), out=buf233)
        del arg108_1
        buf237 = reinterpret_tensor(buf232, (1, 64, 128, 256), (2097152, 32768, 256, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [row_highway_4], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf237, 2097152, grid=grid(2097152), stream=stream0)
        buf239 = reinterpret_tensor(buf230, (1, 64, 128, 256), (2097152, 32768, 256, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [row_highway_3], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf239, 2097152, grid=grid(2097152), stream=stream0)
        buf243 = reinterpret_tensor(buf217, (1, 64, 128, 256), (2097152, 32768, 256, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [col_highway_3], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf243, 2097152, grid=grid(2097152), stream=stream0)
        buf258 = reinterpret_tensor(buf216, (1, 64, 128, 256), (2097152, 32768, 256, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [col_highway_4], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf258, 2097152, grid=grid(2097152), stream=stream0)
        buf267 = reinterpret_tensor(buf229, (1, 8192, 1024), (8388608, 1024, 1), 0); del buf229  # reuse
        buf263 = reinterpret_tensor(buf267, (1, 8192, 256), (8388608, 1024, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_mid_d_2, z_8, row_highway_4, index_add__16, row_highway_3, index_add__12, col_highway_3, index_add__13, col_highway_4, index_add__17], Original ATen: [aten.add, aten.native_layer_norm, aten.new_zeros, aten.index_add]
        triton_per_fused_add_index_add_native_layer_norm_new_zeros_42.run(buf173, buf233, arg109_1, arg42_1, arg110_1, arg111_1, buf237, buf258, buf263, buf239, buf243, 8192, 256, grid=grid(8192), stream=stream0)
        del arg110_1
        del arg111_1
        # Topologically Sorted Source Nodes: [toks_5, index_add__14, index_add__15], Original ATen: [aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_43.run(buf140, buf141, buf176, buf203, arg93_1, buf146, buf239, buf243, 14745600, grid=grid(14745600), stream=stream0)
        buf242 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [row_t_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg30_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf239, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf242)
        del buf239
        buf246 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [col_t_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg31_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf243, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf246)
        buf247 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [x_mid_o_1, g_3], Original ATen: [aten.add, aten.mean]
        triton_per_fused_add_mean_44.run(buf176, buf203, arg93_1, buf247, 59904, 32, grid=grid(59904), stream=stream0)
        buf249 = reinterpret_tensor(buf252, (234, 32, 256), (32768, 1024, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [z_7], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf242, buf249, 1916928, grid=grid(1916928), stream=stream0)
        buf250 = reinterpret_tensor(buf252, (234, 32, 256), (32768, 1024, 1), 512)  # alias
        # Topologically Sorted Source Nodes: [z_7], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf246, buf250, 1916928, grid=grid(1916928), stream=stream0)
        buf251 = reinterpret_tensor(buf252, (234, 32, 256), (32768, 1024, 1), 768)  # alias
        # Topologically Sorted Source Nodes: [z_7], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf247, buf251, 1916928, grid=grid(1916928), stream=stream0)
        del buf248
        del buf249
        del buf250
        del buf251
        buf253 = reinterpret_tensor(buf246, (7488, 1024), (1024, 1), 0); del buf246  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf252, (7488, 1024), (1024, 1), 0), reinterpret_tensor(arg96_1, (1024, 1024), (1, 1024), 0), out=buf253)
        del arg96_1
        buf254 = reinterpret_tensor(buf253, (234, 32, 1024), (32768, 1024, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_47.run(buf254, arg97_1, 7667712, grid=grid(7667712), stream=stream0)
        del arg97_1
        buf255 = reinterpret_tensor(buf202, (7488, 256), (256, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf254, (7488, 1024), (1024, 1), 0), reinterpret_tensor(arg98_1, (1024, 256), (1, 1024), 0), out=buf255)
        del arg98_1
        buf256 = reinterpret_tensor(buf255, (234, 32, 256), (8192, 256, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [x_mid_o_1, off_stream_1], Original ATen: [aten.add]
        triton_poi_fused_add_48.run(buf256, buf176, buf203, arg93_1, arg99_1, 1916928, grid=grid(1916928), stream=stream0)
        del arg93_1
        del arg99_1
        # Topologically Sorted Source Nodes: [toks_7, index_add__18, index_add__19], Original ATen: [aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_49.run(buf140, buf141, buf256, buf146, buf237, buf258, 14745600, grid=grid(14745600), stream=stream0)
        buf261 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d_2, g_4], Original ATen: [aten.add, aten.mean]
        triton_red_fused_add_mean_22.run(buf173, buf233, arg109_1, buf261, 16384, 128, grid=grid(16384), stream=stream0)
        buf262 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d_2, g_4], Original ATen: [aten.add, aten.mean]
        triton_per_fused_add_mean_23.run(buf261, buf262, 256, 64, grid=grid(256), stream=stream0)
        buf264 = reinterpret_tensor(buf267, (1, 8192, 256), (8388608, 1024, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [z_9], Original ATen: [aten.cat]
        triton_poi_fused_cat_24.run(buf237, buf264, 2097152, grid=grid(2097152), stream=stream0)
        buf265 = reinterpret_tensor(buf267, (1, 8192, 256), (8388608, 1024, 1), 512)  # alias
        # Topologically Sorted Source Nodes: [z_9], Original ATen: [aten.cat]
        triton_poi_fused_cat_24.run(buf258, buf265, 2097152, grid=grid(2097152), stream=stream0)
        buf266 = reinterpret_tensor(buf267, (1, 8192, 256), (8388608, 1024, 1), 768)  # alias
        # Topologically Sorted Source Nodes: [z_9], Original ATen: [aten.cat]
        triton_poi_fused_cat_25.run(buf262, buf266, 2097152, grid=grid(2097152), stream=stream0)
        del buf263
        del buf264
        del buf265
        del buf266
        buf268 = reinterpret_tensor(buf228, (8192, 1024), (1024, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (8192, 1024), (1024, 1), 0), reinterpret_tensor(arg112_1, (1024, 1024), (1, 1024), 0), out=buf268)
        del arg112_1
        buf269 = reinterpret_tensor(buf268, (1, 8192, 1024), (8388608, 1024, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_26.run(buf269, arg113_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg113_1
        buf270 = reinterpret_tensor(buf258, (8192, 256), (256, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf269, (8192, 1024), (1024, 1), 0), reinterpret_tensor(arg114_1, (1024, 256), (1, 1024), 0), out=buf270)
        del arg114_1
        buf271 = reinterpret_tensor(buf270, (1, 8192, 256), (2097152, 256, 1), 0); del buf270  # reuse
        buf308 = reinterpret_tensor(buf237, (1, 8192, 256), (2097152, 256, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d_2, h_diag_3, layer_norm_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf271, buf173, buf233, arg109_1, arg115_1, arg132_1, arg133_1, buf308, 8192, 256, grid=grid(8192), stream=stream0)
        del arg109_1
        del arg115_1
        del arg132_1
        del arg133_1
        buf272 = reinterpret_tensor(buf254, (1, 234, 32768), (7667712, 32768, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [row_p_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg30_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf271, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf272)
        buf273 = reinterpret_tensor(buf252, (1, 234, 32768), (7667712, 32768, 1), 0); del buf252  # reuse
        # Topologically Sorted Source Nodes: [col_p_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg31_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf271, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf273)
        buf274 = buf256; del buf256  # reuse
        buf278 = reinterpret_tensor(buf203, (234, 32, 256), (8192, 256, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [h_off_7, off_in_1, layer_norm_11], Original ATen: [aten.mean, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mean_native_layer_norm_50.run(buf274, buf272, buf273, arg116_1, arg117_1, buf278, 7488, 256, grid=grid(7488), stream=stream0)
        del arg116_1
        del arg117_1
        buf279 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf278, (7488, 256), (256, 1), 0), reinterpret_tensor(arg118_1, (256, 768), (1, 256), 0), out=buf279)
        del arg118_1
        # Topologically Sorted Source Nodes: [setitem_10], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_30.run(buf280, 29952, grid=grid(29952), stream=stream0)
        buf282 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [setitem_11], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_31.run(buf280, buf282, 239616, grid=grid(239616), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_11], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_32.run(buf282, 7488, grid=grid(7488), stream=stream0)
        buf284 = reinterpret_tensor(buf278, (234, 8, 32, 32), (8192, 1024, 32, 1), 0); del buf278  # reuse
        # Topologically Sorted Source Nodes: [q_f_5, k_f_5, v_f_5, logit_bias_5, x_soft_10], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_33.run(buf279, arg119_1, buf284, 1916928, grid=grid(1916928), stream=stream0)
        buf285 = reinterpret_tensor(buf176, (234, 8, 32, 32), (8192, 1024, 32, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [q_f_5, k_f_5, v_f_5, logit_bias_5, x_soft_10], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_34.run(buf279, arg119_1, buf285, 1916928, grid=grid(1916928), stream=stream0)
        buf286 = reinterpret_tensor(buf200, (234, 8, 32, 32), (8192, 1024, 32, 1), 0); del buf200  # reuse
        buf298 = reinterpret_tensor(buf199, (234, 8, 32, 1, 32, 1), (8192, 1024, 32, 32, 1, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [q_f_5, k_f_5, v_f_5, logit_bias_5, x_soft_10, einsum_17], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_35.run(buf279, arg119_1, buf286, buf298, 1916928, grid=grid(1916928), stream=stream0)
        del arg119_1
        buf287 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [q_f_5, k_f_5, v_f_5, logit_bias_5, x_soft_10], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_36.run(buf282, buf280, buf287, 1916928, grid=grid(1916928), stream=stream0)
        # Topologically Sorted Source Nodes: [q_f_5, k_f_5, v_f_5, logit_bias_5, x_soft_10], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf288 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf284, buf285, buf286, buf287, False)
        del buf284
        buf289 = buf288[0]
        del buf288
        buf293 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_37.run(buf282, buf280, buf293, 958464, grid=grid(958464), stream=stream0)
        del buf280
        buf294 = reinterpret_tensor(buf197, (239616, 16), (16, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf293, (239616, 4), (4, 1), 0), reinterpret_tensor(arg120_1, (4, 16), (1, 4), 0), out=buf294)
        del arg120_1
        buf295 = reinterpret_tensor(buf294, (234, 1, 32, 32, 16), (16384, 1, 512, 16, 1), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_38.run(buf295, arg121_1, 3833856, grid=grid(3833856), stream=stream0)
        del arg121_1
        buf296 = reinterpret_tensor(buf287, (239616, 8), (8, 1), 0); del buf287  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf295, (239616, 16), (16, 1), 0), reinterpret_tensor(arg122_1, (16, 8), (1, 16), 0), out=buf296)
        del arg122_1
        buf297 = reinterpret_tensor(buf286, (234, 8, 32, 32, 1, 1), (8192, 1024, 32, 1, 1, 1), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [einsum_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf296, arg123_1, buf297, 1872, 1024, grid=grid(1872, 1024), stream=stream0)
        del arg123_1
        buf299 = reinterpret_tensor(buf296, (1872, 32, 32), (1024, 32, 1), 0); del buf296  # reuse
        # Topologically Sorted Source Nodes: [einsum_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf297, (1872, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf298, (1872, 32, 32), (1024, 32, 1), 0), out=buf299)
        buf300 = reinterpret_tensor(buf289, (234, 1, 32, 8, 32), (8192, 1, 256, 32, 1), 0); del buf289  # reuse
        # Topologically Sorted Source Nodes: [x_mid_5], Original ATen: [aten.add]
        triton_poi_fused_add_40.run(buf300, buf299, 1916928, grid=grid(1916928), stream=stream0)
        buf301 = reinterpret_tensor(buf299, (7488, 256), (256, 1), 0); del buf299  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf300, (7488, 256), (256, 1), 0), reinterpret_tensor(arg124_1, (256, 256), (1, 256), 0), out=buf301)
        del arg124_1
        buf350 = reinterpret_tensor(buf273, (234, 32, 1024), (32768, 1024, 1), 0); del buf273  # reuse
        buf346 = reinterpret_tensor(buf350, (234, 32, 256), (32768, 1024, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_mid_o_2, z_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_41.run(buf274, buf301, arg125_1, arg126_1, arg127_1, buf346, 7488, 256, grid=grid(7488), stream=stream0)
        del arg126_1
        del arg127_1
        buf309 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf308, (8192, 256), (256, 1), 0), reinterpret_tensor(arg134_1, (256, 768), (1, 256), 0), out=buf309)
        del arg134_1
        buf310 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [setitem_12], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_8.run(arg28_1, buf310, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_12], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_9.run(buf310, 32768, grid=grid(32768), stream=stream0)
        buf312 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [setitem_13], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_10.run(buf310, buf312, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_13], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_11.run(buf312, 8192, grid=grid(8192), stream=stream0)
        buf314 = reinterpret_tensor(buf308, (64, 8, 128, 32), (32768, 4096, 32, 1), 0); del buf308  # reuse
        # Topologically Sorted Source Nodes: [q_f_6, k_f_6, v_f_6, logit_bias_6, x_soft_12], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_12.run(buf309, arg135_1, buf314, 2097152, grid=grid(2097152), stream=stream0)
        buf315 = reinterpret_tensor(buf233, (64, 8, 128, 32), (32768, 4096, 32, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [q_f_6, k_f_6, v_f_6, logit_bias_6, x_soft_12], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_13.run(buf309, arg135_1, buf315, 2097152, grid=grid(2097152), stream=stream0)
        buf316 = reinterpret_tensor(buf173, (64, 8, 128, 32), (32768, 4096, 32, 1), 0); del buf173  # reuse
        buf328 = reinterpret_tensor(buf243, (64, 8, 128, 1, 32, 1), (32768, 4096, 32, 32, 1, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [q_f_6, k_f_6, v_f_6, logit_bias_6, x_soft_12, einsum_20], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_14.run(buf309, arg135_1, buf316, buf328, 2097152, grid=grid(2097152), stream=stream0)
        del arg135_1
        buf317 = reinterpret_tensor(buf269, (64, 8, 128, 128), (131072, 16384, 128, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [q_f_6, k_f_6, v_f_6, logit_bias_6, x_soft_12], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_15.run(buf312, buf310, buf317, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [q_f_6, k_f_6, v_f_6, logit_bias_6, x_soft_12], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf318 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf314, buf315, buf316, buf317, False)
        buf319 = buf318[0]
        del buf318
        buf323 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(buf312, buf310, buf323, 4194304, grid=grid(4194304), stream=stream0)
        buf324 = reinterpret_tensor(buf227, (1048576, 16), (16, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (1048576, 4), (4, 1), 0), reinterpret_tensor(arg136_1, (4, 16), (1, 4), 0), out=buf324)
        del arg136_1
        buf325 = reinterpret_tensor(buf324, (1, 64, 128, 128, 16), (16777216, 262144, 2048, 16, 1), 0); del buf324  # reuse
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_17.run(buf325, arg137_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg137_1
        buf326 = reinterpret_tensor(buf317, (1048576, 8), (8, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf325, (1048576, 16), (16, 1), 0), reinterpret_tensor(arg138_1, (16, 8), (1, 16), 0), out=buf326)
        del arg138_1
        buf327 = reinterpret_tensor(buf267, (64, 8, 128, 128, 1, 1), (131072, 16384, 128, 1, 1, 1), 0); del buf267  # reuse
        # Topologically Sorted Source Nodes: [einsum_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf326, arg139_1, buf327, 512, 16384, grid=grid(512, 16384), stream=stream0)
        del arg139_1
        buf329 = reinterpret_tensor(buf316, (512, 128, 32), (4096, 32, 1), 0); del buf316  # reuse
        # Topologically Sorted Source Nodes: [einsum_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf327, (512, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf328, (512, 128, 32), (4096, 32, 1), 0), out=buf329)
        buf330 = reinterpret_tensor(buf319, (1, 64, 128, 8, 32), (2097152, 32768, 256, 32, 1), 0); del buf319  # reuse
        # Topologically Sorted Source Nodes: [x_mid_6], Original ATen: [aten.add]
        triton_poi_fused_add_19.run(buf330, buf329, 2097152, grid=grid(2097152), stream=stream0)
        buf331 = reinterpret_tensor(buf329, (8192, 256), (256, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf330, (8192, 256), (256, 1), 0), reinterpret_tensor(arg140_1, (256, 256), (1, 256), 0), out=buf331)
        del arg140_1
        buf335 = reinterpret_tensor(buf330, (1, 64, 128, 256), (2097152, 32768, 256, 1), 0); del buf330  # reuse
        # Topologically Sorted Source Nodes: [row_highway_6], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf335, 2097152, grid=grid(2097152), stream=stream0)
        buf337 = reinterpret_tensor(buf328, (1, 64, 128, 256), (2097152, 32768, 256, 1), 0); del buf328  # reuse
        # Topologically Sorted Source Nodes: [row_highway_5], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf337, 2097152, grid=grid(2097152), stream=stream0)
        buf341 = reinterpret_tensor(buf315, (1, 64, 128, 256), (2097152, 32768, 256, 1), 0); del buf315  # reuse
        # Topologically Sorted Source Nodes: [col_highway_5], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf341, 2097152, grid=grid(2097152), stream=stream0)
        buf356 = reinterpret_tensor(buf314, (1, 64, 128, 256), (2097152, 32768, 256, 1), 0); del buf314  # reuse
        # Topologically Sorted Source Nodes: [col_highway_6], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf356, 2097152, grid=grid(2097152), stream=stream0)
        buf365 = reinterpret_tensor(buf327, (1, 8192, 1024), (8388608, 1024, 1), 0); del buf327  # reuse
        buf361 = reinterpret_tensor(buf365, (1, 8192, 256), (8388608, 1024, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_mid_d_3, z_12, row_highway_6, index_add__24, row_highway_5, index_add__20, col_highway_5, index_add__21, col_highway_6, index_add__25], Original ATen: [aten.add, aten.native_layer_norm, aten.new_zeros, aten.index_add]
        triton_per_fused_add_index_add_native_layer_norm_new_zeros_42.run(buf271, buf331, arg141_1, arg42_1, arg142_1, arg143_1, buf335, buf356, buf361, buf337, buf341, 8192, 256, grid=grid(8192), stream=stream0)
        del arg142_1
        del arg143_1
        # Topologically Sorted Source Nodes: [toks_9, index_add__22, index_add__23], Original ATen: [aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_43.run(buf140, buf141, buf274, buf301, arg125_1, buf146, buf337, buf341, 14745600, grid=grid(14745600), stream=stream0)
        buf340 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [row_t_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg30_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf337, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf340)
        buf344 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [col_t_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg31_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf341, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf344)
        buf345 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [x_mid_o_2, g_5], Original ATen: [aten.add, aten.mean]
        triton_per_fused_add_mean_44.run(buf274, buf301, arg125_1, buf345, 59904, 32, grid=grid(59904), stream=stream0)
        buf347 = reinterpret_tensor(buf350, (234, 32, 256), (32768, 1024, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [z_11], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf340, buf347, 1916928, grid=grid(1916928), stream=stream0)
        buf348 = reinterpret_tensor(buf350, (234, 32, 256), (32768, 1024, 1), 512)  # alias
        # Topologically Sorted Source Nodes: [z_11], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf344, buf348, 1916928, grid=grid(1916928), stream=stream0)
        buf349 = reinterpret_tensor(buf350, (234, 32, 256), (32768, 1024, 1), 768)  # alias
        # Topologically Sorted Source Nodes: [z_11], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf345, buf349, 1916928, grid=grid(1916928), stream=stream0)
        del buf346
        del buf347
        del buf348
        del buf349
        buf351 = reinterpret_tensor(buf344, (7488, 1024), (1024, 1), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf350, (7488, 1024), (1024, 1), 0), reinterpret_tensor(arg128_1, (1024, 1024), (1, 1024), 0), out=buf351)
        del arg128_1
        buf352 = reinterpret_tensor(buf351, (234, 32, 1024), (32768, 1024, 1), 0); del buf351  # reuse
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_47.run(buf352, arg129_1, 7667712, grid=grid(7667712), stream=stream0)
        del arg129_1
        buf353 = reinterpret_tensor(buf300, (7488, 256), (256, 1), 0); del buf300  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf352, (7488, 1024), (1024, 1), 0), reinterpret_tensor(arg130_1, (1024, 256), (1, 1024), 0), out=buf353)
        del arg130_1
        buf354 = reinterpret_tensor(buf353, (234, 32, 256), (8192, 256, 1), 0); del buf353  # reuse
        # Topologically Sorted Source Nodes: [x_mid_o_2, off_stream_2], Original ATen: [aten.add]
        triton_poi_fused_add_48.run(buf354, buf274, buf301, arg125_1, arg131_1, 1916928, grid=grid(1916928), stream=stream0)
        del arg125_1
        del arg131_1
        # Topologically Sorted Source Nodes: [toks_11, index_add__26, index_add__27], Original ATen: [aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_49.run(buf140, buf141, buf354, buf146, buf335, buf356, 14745600, grid=grid(14745600), stream=stream0)
        buf359 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d_3, g_6], Original ATen: [aten.add, aten.mean]
        triton_red_fused_add_mean_22.run(buf271, buf331, arg141_1, buf359, 16384, 128, grid=grid(16384), stream=stream0)
        buf360 = buf262; del buf262  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d_3, g_6], Original ATen: [aten.add, aten.mean]
        triton_per_fused_add_mean_23.run(buf359, buf360, 256, 64, grid=grid(256), stream=stream0)
        buf362 = reinterpret_tensor(buf365, (1, 8192, 256), (8388608, 1024, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [z_13], Original ATen: [aten.cat]
        triton_poi_fused_cat_24.run(buf335, buf362, 2097152, grid=grid(2097152), stream=stream0)
        buf363 = reinterpret_tensor(buf365, (1, 8192, 256), (8388608, 1024, 1), 512)  # alias
        # Topologically Sorted Source Nodes: [z_13], Original ATen: [aten.cat]
        triton_poi_fused_cat_24.run(buf356, buf363, 2097152, grid=grid(2097152), stream=stream0)
        buf364 = reinterpret_tensor(buf365, (1, 8192, 256), (8388608, 1024, 1), 768)  # alias
        # Topologically Sorted Source Nodes: [z_13], Original ATen: [aten.cat]
        triton_poi_fused_cat_25.run(buf360, buf364, 2097152, grid=grid(2097152), stream=stream0)
        del buf361
        del buf362
        del buf363
        del buf364
        buf366 = reinterpret_tensor(buf326, (8192, 1024), (1024, 1), 0); del buf326  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf365, (8192, 1024), (1024, 1), 0), reinterpret_tensor(arg144_1, (1024, 1024), (1, 1024), 0), out=buf366)
        del arg144_1
        buf367 = reinterpret_tensor(buf366, (1, 8192, 1024), (8388608, 1024, 1), 0); del buf366  # reuse
        # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_26.run(buf367, arg145_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg145_1
        buf368 = reinterpret_tensor(buf356, (8192, 256), (256, 1), 0); del buf356  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf367, (8192, 1024), (1024, 1), 0), reinterpret_tensor(arg146_1, (1024, 256), (1, 1024), 0), out=buf368)
        del arg146_1
        buf369 = reinterpret_tensor(buf368, (1, 8192, 256), (2097152, 256, 1), 0); del buf368  # reuse
        buf406 = reinterpret_tensor(buf335, (1, 8192, 256), (2097152, 256, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d_3, h_diag_4, layer_norm_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf369, buf271, buf331, arg141_1, arg147_1, arg164_1, arg165_1, buf406, 8192, 256, grid=grid(8192), stream=stream0)
        del arg141_1
        del arg147_1
        del arg164_1
        del arg165_1
        buf370 = reinterpret_tensor(buf352, (1, 234, 32768), (7667712, 32768, 1), 0); del buf352  # reuse
        # Topologically Sorted Source Nodes: [row_p_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg30_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf369, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf370)
        buf371 = reinterpret_tensor(buf350, (1, 234, 32768), (7667712, 32768, 1), 0); del buf350  # reuse
        # Topologically Sorted Source Nodes: [col_p_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg31_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf369, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf371)
        buf372 = buf354; del buf354  # reuse
        buf376 = reinterpret_tensor(buf301, (234, 32, 256), (8192, 256, 1), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [h_off_9, off_in_2, layer_norm_15], Original ATen: [aten.mean, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mean_native_layer_norm_50.run(buf372, buf370, buf371, arg148_1, arg149_1, buf376, 7488, 256, grid=grid(7488), stream=stream0)
        del arg148_1
        del arg149_1
        buf377 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf376, (7488, 256), (256, 1), 0), reinterpret_tensor(arg150_1, (256, 768), (1, 256), 0), out=buf377)
        del arg150_1
        # Topologically Sorted Source Nodes: [setitem_14], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_30.run(buf378, 29952, grid=grid(29952), stream=stream0)
        buf380 = buf282; del buf282  # reuse
        # Topologically Sorted Source Nodes: [setitem_15], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_31.run(buf378, buf380, 239616, grid=grid(239616), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_15], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_32.run(buf380, 7488, grid=grid(7488), stream=stream0)
        buf382 = reinterpret_tensor(buf376, (234, 8, 32, 32), (8192, 1024, 32, 1), 0); del buf376  # reuse
        # Topologically Sorted Source Nodes: [q_f_7, k_f_7, v_f_7, logit_bias_7, x_soft_14], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_33.run(buf377, arg151_1, buf382, 1916928, grid=grid(1916928), stream=stream0)
        buf383 = reinterpret_tensor(buf274, (234, 8, 32, 32), (8192, 1024, 32, 1), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [q_f_7, k_f_7, v_f_7, logit_bias_7, x_soft_14], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_34.run(buf377, arg151_1, buf383, 1916928, grid=grid(1916928), stream=stream0)
        buf384 = reinterpret_tensor(buf298, (234, 8, 32, 32), (8192, 1024, 32, 1), 0); del buf298  # reuse
        buf396 = reinterpret_tensor(buf297, (234, 8, 32, 1, 32, 1), (8192, 1024, 32, 32, 1, 1), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [q_f_7, k_f_7, v_f_7, logit_bias_7, x_soft_14, einsum_23], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_35.run(buf377, arg151_1, buf384, buf396, 1916928, grid=grid(1916928), stream=stream0)
        del arg151_1
        buf385 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [q_f_7, k_f_7, v_f_7, logit_bias_7, x_soft_14], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_36.run(buf380, buf378, buf385, 1916928, grid=grid(1916928), stream=stream0)
        # Topologically Sorted Source Nodes: [q_f_7, k_f_7, v_f_7, logit_bias_7, x_soft_14], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf386 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf382, buf383, buf384, buf385, False)
        del buf382
        buf387 = buf386[0]
        del buf386
        buf391 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_37.run(buf380, buf378, buf391, 958464, grid=grid(958464), stream=stream0)
        del buf378
        buf392 = reinterpret_tensor(buf295, (239616, 16), (16, 1), 0); del buf295  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf391, (239616, 4), (4, 1), 0), reinterpret_tensor(arg152_1, (4, 16), (1, 4), 0), out=buf392)
        del arg152_1
        buf393 = reinterpret_tensor(buf392, (234, 1, 32, 32, 16), (16384, 1, 512, 16, 1), 0); del buf392  # reuse
        # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_38.run(buf393, arg153_1, 3833856, grid=grid(3833856), stream=stream0)
        del arg153_1
        buf394 = reinterpret_tensor(buf385, (239616, 8), (8, 1), 0); del buf385  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf393, (239616, 16), (16, 1), 0), reinterpret_tensor(arg154_1, (16, 8), (1, 16), 0), out=buf394)
        del arg154_1
        buf395 = reinterpret_tensor(buf384, (234, 8, 32, 32, 1, 1), (8192, 1024, 32, 1, 1, 1), 0); del buf384  # reuse
        # Topologically Sorted Source Nodes: [einsum_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf394, arg155_1, buf395, 1872, 1024, grid=grid(1872, 1024), stream=stream0)
        del arg155_1
        buf397 = reinterpret_tensor(buf394, (1872, 32, 32), (1024, 32, 1), 0); del buf394  # reuse
        # Topologically Sorted Source Nodes: [einsum_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf395, (1872, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf396, (1872, 32, 32), (1024, 32, 1), 0), out=buf397)
        buf398 = reinterpret_tensor(buf387, (234, 1, 32, 8, 32), (8192, 1, 256, 32, 1), 0); del buf387  # reuse
        # Topologically Sorted Source Nodes: [x_mid_7], Original ATen: [aten.add]
        triton_poi_fused_add_40.run(buf398, buf397, 1916928, grid=grid(1916928), stream=stream0)
        buf399 = reinterpret_tensor(buf397, (7488, 256), (256, 1), 0); del buf397  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf398, (7488, 256), (256, 1), 0), reinterpret_tensor(arg156_1, (256, 256), (1, 256), 0), out=buf399)
        del arg156_1
        buf448 = reinterpret_tensor(buf371, (234, 32, 1024), (32768, 1024, 1), 0); del buf371  # reuse
        buf444 = reinterpret_tensor(buf448, (234, 32, 256), (32768, 1024, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_mid_o_3, z_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_41.run(buf372, buf399, arg157_1, arg158_1, arg159_1, buf444, 7488, 256, grid=grid(7488), stream=stream0)
        del arg158_1
        del arg159_1
        buf407 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf406, (8192, 256), (256, 1), 0), reinterpret_tensor(arg166_1, (256, 768), (1, 256), 0), out=buf407)
        del arg166_1
        buf408 = buf323; del buf323  # reuse
        # Topologically Sorted Source Nodes: [setitem_16], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_8.run(arg28_1, buf408, 4194304, grid=grid(4194304), stream=stream0)
        del arg28_1
        # Topologically Sorted Source Nodes: [setitem_16], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_9.run(buf408, 32768, grid=grid(32768), stream=stream0)
        buf410 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [setitem_17], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_10.run(buf408, buf410, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_17], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_11.run(buf410, 8192, grid=grid(8192), stream=stream0)
        buf412 = reinterpret_tensor(buf406, (64, 8, 128, 32), (32768, 4096, 32, 1), 0); del buf406  # reuse
        # Topologically Sorted Source Nodes: [q_f_8, k_f_8, v_f_8, logit_bias_8, x_soft_16], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_12.run(buf407, arg167_1, buf412, 2097152, grid=grid(2097152), stream=stream0)
        buf413 = reinterpret_tensor(buf331, (64, 8, 128, 32), (32768, 4096, 32, 1), 0); del buf331  # reuse
        # Topologically Sorted Source Nodes: [q_f_8, k_f_8, v_f_8, logit_bias_8, x_soft_16], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_13.run(buf407, arg167_1, buf413, 2097152, grid=grid(2097152), stream=stream0)
        buf414 = reinterpret_tensor(buf271, (64, 8, 128, 32), (32768, 4096, 32, 1), 0); del buf271  # reuse
        buf426 = reinterpret_tensor(buf341, (64, 8, 128, 1, 32, 1), (32768, 4096, 32, 32, 1, 1), 0); del buf341  # reuse
        # Topologically Sorted Source Nodes: [q_f_8, k_f_8, v_f_8, logit_bias_8, x_soft_16, einsum_26], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_14.run(buf407, arg167_1, buf414, buf426, 2097152, grid=grid(2097152), stream=stream0)
        del arg167_1
        del buf407
        buf415 = reinterpret_tensor(buf367, (64, 8, 128, 128), (131072, 16384, 128, 1), 0); del buf367  # reuse
        # Topologically Sorted Source Nodes: [q_f_8, k_f_8, v_f_8, logit_bias_8, x_soft_16], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_15.run(buf410, buf408, buf415, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [q_f_8, k_f_8, v_f_8, logit_bias_8, x_soft_16], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf416 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf412, buf413, buf414, buf415, False)
        buf417 = buf416[0]
        del buf416
        buf421 = buf310; del buf310  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(buf410, buf408, buf421, 4194304, grid=grid(4194304), stream=stream0)
        del buf408
        buf422 = reinterpret_tensor(buf325, (1048576, 16), (16, 1), 0); del buf325  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf421, (1048576, 4), (4, 1), 0), reinterpret_tensor(arg168_1, (4, 16), (1, 4), 0), out=buf422)
        del arg168_1
        del buf421
        buf423 = reinterpret_tensor(buf422, (1, 64, 128, 128, 16), (16777216, 262144, 2048, 16, 1), 0); del buf422  # reuse
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_17.run(buf423, arg169_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg169_1
        buf424 = reinterpret_tensor(buf415, (1048576, 8), (8, 1), 0); del buf415  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf423, (1048576, 16), (16, 1), 0), reinterpret_tensor(arg170_1, (16, 8), (1, 16), 0), out=buf424)
        del arg170_1
        del buf423
        buf425 = reinterpret_tensor(buf365, (64, 8, 128, 128, 1, 1), (131072, 16384, 128, 1, 1, 1), 0); del buf365  # reuse
        # Topologically Sorted Source Nodes: [einsum_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf424, arg171_1, buf425, 512, 16384, grid=grid(512, 16384), stream=stream0)
        del arg171_1
        buf427 = reinterpret_tensor(buf414, (512, 128, 32), (4096, 32, 1), 0); del buf414  # reuse
        # Topologically Sorted Source Nodes: [einsum_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf425, (512, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf426, (512, 128, 32), (4096, 32, 1), 0), out=buf427)
        buf428 = reinterpret_tensor(buf417, (1, 64, 128, 8, 32), (2097152, 32768, 256, 32, 1), 0); del buf417  # reuse
        # Topologically Sorted Source Nodes: [x_mid_8], Original ATen: [aten.add]
        triton_poi_fused_add_19.run(buf428, buf427, 2097152, grid=grid(2097152), stream=stream0)
        buf429 = reinterpret_tensor(buf427, (8192, 256), (256, 1), 0); del buf427  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf428, (8192, 256), (256, 1), 0), reinterpret_tensor(arg172_1, (256, 256), (1, 256), 0), out=buf429)
        del arg172_1
        buf433 = reinterpret_tensor(buf428, (1, 64, 128, 256), (2097152, 32768, 256, 1), 0); del buf428  # reuse
        # Topologically Sorted Source Nodes: [row_highway_8], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf433, 2097152, grid=grid(2097152), stream=stream0)
        buf435 = reinterpret_tensor(buf426, (1, 64, 128, 256), (2097152, 32768, 256, 1), 0); del buf426  # reuse
        # Topologically Sorted Source Nodes: [row_highway_7], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf435, 2097152, grid=grid(2097152), stream=stream0)
        buf439 = reinterpret_tensor(buf413, (1, 64, 128, 256), (2097152, 32768, 256, 1), 0); del buf413  # reuse
        # Topologically Sorted Source Nodes: [col_highway_7], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf439, 2097152, grid=grid(2097152), stream=stream0)
        buf454 = reinterpret_tensor(buf412, (1, 64, 128, 256), (2097152, 32768, 256, 1), 0); del buf412  # reuse
        # Topologically Sorted Source Nodes: [col_highway_8], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf454, 2097152, grid=grid(2097152), stream=stream0)
        buf463 = reinterpret_tensor(buf425, (1, 8192, 1024), (8388608, 1024, 1), 0); del buf425  # reuse
        buf459 = reinterpret_tensor(buf463, (1, 8192, 256), (8388608, 1024, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_mid_d_4, z_16, row_highway_8, index_add__32, row_highway_7, index_add__28, col_highway_7, index_add__29, col_highway_8, index_add__33], Original ATen: [aten.add, aten.native_layer_norm, aten.new_zeros, aten.index_add]
        triton_per_fused_add_index_add_native_layer_norm_new_zeros_42.run(buf369, buf429, arg173_1, arg42_1, arg174_1, arg175_1, buf433, buf454, buf459, buf435, buf439, 8192, 256, grid=grid(8192), stream=stream0)
        del arg174_1
        del arg175_1
        # Topologically Sorted Source Nodes: [toks_13, index_add__30, index_add__31], Original ATen: [aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_43.run(buf140, buf141, buf372, buf399, arg157_1, buf146, buf435, buf439, 14745600, grid=grid(14745600), stream=stream0)
        buf438 = buf370; del buf370  # reuse
        # Topologically Sorted Source Nodes: [row_t_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg30_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf435, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf438)
        buf442 = buf340; del buf340  # reuse
        # Topologically Sorted Source Nodes: [col_t_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg31_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf439, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf442)
        buf443 = buf345; del buf345  # reuse
        # Topologically Sorted Source Nodes: [x_mid_o_3, g_7], Original ATen: [aten.add, aten.mean]
        triton_per_fused_add_mean_44.run(buf372, buf399, arg157_1, buf443, 59904, 32, grid=grid(59904), stream=stream0)
        buf445 = reinterpret_tensor(buf448, (234, 32, 256), (32768, 1024, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [z_15], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf438, buf445, 1916928, grid=grid(1916928), stream=stream0)
        buf446 = reinterpret_tensor(buf448, (234, 32, 256), (32768, 1024, 1), 512)  # alias
        # Topologically Sorted Source Nodes: [z_15], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf442, buf446, 1916928, grid=grid(1916928), stream=stream0)
        buf447 = reinterpret_tensor(buf448, (234, 32, 256), (32768, 1024, 1), 768)  # alias
        # Topologically Sorted Source Nodes: [z_15], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf443, buf447, 1916928, grid=grid(1916928), stream=stream0)
        del buf444
        del buf445
        del buf446
        del buf447
        buf449 = reinterpret_tensor(buf442, (7488, 1024), (1024, 1), 0); del buf442  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf448, (7488, 1024), (1024, 1), 0), reinterpret_tensor(arg160_1, (1024, 1024), (1, 1024), 0), out=buf449)
        del arg160_1
        buf450 = reinterpret_tensor(buf449, (234, 32, 1024), (32768, 1024, 1), 0); del buf449  # reuse
        # Topologically Sorted Source Nodes: [input_56], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_47.run(buf450, arg161_1, 7667712, grid=grid(7667712), stream=stream0)
        del arg161_1
        buf451 = reinterpret_tensor(buf398, (7488, 256), (256, 1), 0); del buf398  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf450, (7488, 1024), (1024, 1), 0), reinterpret_tensor(arg162_1, (1024, 256), (1, 1024), 0), out=buf451)
        del arg162_1
        buf452 = reinterpret_tensor(buf451, (234, 32, 256), (8192, 256, 1), 0); del buf451  # reuse
        # Topologically Sorted Source Nodes: [x_mid_o_3, off_stream_3], Original ATen: [aten.add]
        triton_poi_fused_add_48.run(buf452, buf372, buf399, arg157_1, arg163_1, 1916928, grid=grid(1916928), stream=stream0)
        del arg157_1
        del arg163_1
        # Topologically Sorted Source Nodes: [toks_15, index_add__34, index_add__35], Original ATen: [aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_49.run(buf140, buf141, buf452, buf146, buf433, buf454, 14745600, grid=grid(14745600), stream=stream0)
        buf457 = buf359; del buf359  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d_4, g_8], Original ATen: [aten.add, aten.mean]
        triton_red_fused_add_mean_22.run(buf369, buf429, arg173_1, buf457, 16384, 128, grid=grid(16384), stream=stream0)
        buf458 = buf360; del buf360  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d_4, g_8], Original ATen: [aten.add, aten.mean]
        triton_per_fused_add_mean_23.run(buf457, buf458, 256, 64, grid=grid(256), stream=stream0)
        del buf457
        buf460 = reinterpret_tensor(buf463, (1, 8192, 256), (8388608, 1024, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [z_17], Original ATen: [aten.cat]
        triton_poi_fused_cat_24.run(buf433, buf460, 2097152, grid=grid(2097152), stream=stream0)
        buf461 = reinterpret_tensor(buf463, (1, 8192, 256), (8388608, 1024, 1), 512)  # alias
        # Topologically Sorted Source Nodes: [z_17], Original ATen: [aten.cat]
        triton_poi_fused_cat_24.run(buf454, buf461, 2097152, grid=grid(2097152), stream=stream0)
        buf462 = reinterpret_tensor(buf463, (1, 8192, 256), (8388608, 1024, 1), 768)  # alias
        # Topologically Sorted Source Nodes: [z_17], Original ATen: [aten.cat]
        triton_poi_fused_cat_25.run(buf458, buf462, 2097152, grid=grid(2097152), stream=stream0)
        del buf458
        del buf459
        del buf460
        del buf461
        del buf462
        buf464 = reinterpret_tensor(buf424, (8192, 1024), (1024, 1), 0); del buf424  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf463, (8192, 1024), (1024, 1), 0), reinterpret_tensor(arg176_1, (1024, 1024), (1, 1024), 0), out=buf464)
        del arg176_1
        del buf463
        buf465 = reinterpret_tensor(buf464, (1, 8192, 1024), (8388608, 1024, 1), 0); del buf464  # reuse
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_26.run(buf465, arg177_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg177_1
        buf466 = reinterpret_tensor(buf454, (8192, 256), (256, 1), 0); del buf454  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf465, (8192, 1024), (1024, 1), 0), reinterpret_tensor(arg178_1, (1024, 256), (1, 1024), 0), out=buf466)
        del arg178_1
        del buf465
        buf505 = buf433; del buf433  # reuse
        # Topologically Sorted Source Nodes: [row_highway_9], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf505, 2097152, grid=grid(2097152), stream=stream0)
        buf509 = buf439; del buf439  # reuse
        # Topologically Sorted Source Nodes: [col_highway_9], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf509, 2097152, grid=grid(2097152), stream=stream0)
        buf535 = buf435; del buf435  # reuse
        # Topologically Sorted Source Nodes: [col_highway_10], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf535, 2097152, grid=grid(2097152), stream=stream0)
        buf538 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [row_highway_10], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf538, 2097152, grid=grid(2097152), stream=stream0)
        buf467 = reinterpret_tensor(buf466, (1, 8192, 256), (2097152, 256, 1), 0); del buf466  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d_4, h_diag_5, row_highway_9, index_add__36, col_highway_9, index_add__37, col_highway_10, index_add__41, row_highway_10, index_add__40], Original ATen: [aten.add, aten.new_zeros, aten.index_add]
        triton_poi_fused_add_index_add_new_zeros_51.run(buf467, buf369, buf429, arg173_1, arg179_1, arg42_1, buf505, buf509, buf535, buf538, 2097152, grid=grid(2097152), stream=stream0)
        del arg173_1
        del arg179_1
        del buf369
        buf468 = reinterpret_tensor(buf450, (1, 234, 32768), (7667712, 32768, 1), 0); del buf450  # reuse
        # Topologically Sorted Source Nodes: [row_p_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg30_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf467, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf468)
        buf469 = reinterpret_tensor(buf448, (1, 234, 32768), (7667712, 32768, 1), 0); del buf448  # reuse
        # Topologically Sorted Source Nodes: [col_p_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg31_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf467, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf469)
        buf470 = buf452; del buf452  # reuse
        buf474 = reinterpret_tensor(buf399, (234, 32, 256), (8192, 256, 1), 0); del buf399  # reuse
        # Topologically Sorted Source Nodes: [h_off_11, off_in_3, layer_norm_19], Original ATen: [aten.mean, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mean_native_layer_norm_50.run(buf470, buf468, buf469, arg180_1, arg181_1, buf474, 7488, 256, grid=grid(7488), stream=stream0)
        del arg180_1
        del arg181_1
        buf475 = buf377; del buf377  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf474, (7488, 256), (256, 1), 0), reinterpret_tensor(arg182_1, (256, 768), (1, 256), 0), out=buf475)
        del arg182_1
        # Topologically Sorted Source Nodes: [setitem_18], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_30.run(buf476, 29952, grid=grid(29952), stream=stream0)
        buf478 = buf380; del buf380  # reuse
        # Topologically Sorted Source Nodes: [setitem_19], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_31.run(buf476, buf478, 239616, grid=grid(239616), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_19], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_32.run(buf478, 7488, grid=grid(7488), stream=stream0)
        buf480 = reinterpret_tensor(buf474, (234, 8, 32, 32), (8192, 1024, 32, 1), 0); del buf474  # reuse
        # Topologically Sorted Source Nodes: [q_f_9, k_f_9, v_f_9, logit_bias_9, x_soft_18], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_33.run(buf475, arg183_1, buf480, 1916928, grid=grid(1916928), stream=stream0)
        buf481 = reinterpret_tensor(buf372, (234, 8, 32, 32), (8192, 1024, 32, 1), 0); del buf372  # reuse
        # Topologically Sorted Source Nodes: [q_f_9, k_f_9, v_f_9, logit_bias_9, x_soft_18], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_34.run(buf475, arg183_1, buf481, 1916928, grid=grid(1916928), stream=stream0)
        buf482 = reinterpret_tensor(buf396, (234, 8, 32, 32), (8192, 1024, 32, 1), 0); del buf396  # reuse
        buf494 = reinterpret_tensor(buf395, (234, 8, 32, 1, 32, 1), (8192, 1024, 32, 32, 1, 1), 0); del buf395  # reuse
        # Topologically Sorted Source Nodes: [q_f_9, k_f_9, v_f_9, logit_bias_9, x_soft_18, einsum_29], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_35.run(buf475, arg183_1, buf482, buf494, 1916928, grid=grid(1916928), stream=stream0)
        del arg183_1
        del buf475
        buf483 = buf383; del buf383  # reuse
        # Topologically Sorted Source Nodes: [q_f_9, k_f_9, v_f_9, logit_bias_9, x_soft_18], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_36.run(buf478, buf476, buf483, 1916928, grid=grid(1916928), stream=stream0)
        # Topologically Sorted Source Nodes: [q_f_9, k_f_9, v_f_9, logit_bias_9, x_soft_18], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf484 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf480, buf481, buf482, buf483, False)
        del buf480
        del buf481
        buf485 = buf484[0]
        del buf484
        buf489 = buf391; del buf391  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_37.run(buf478, buf476, buf489, 958464, grid=grid(958464), stream=stream0)
        del buf476
        buf490 = reinterpret_tensor(buf393, (239616, 16), (16, 1), 0); del buf393  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf489, (239616, 4), (4, 1), 0), reinterpret_tensor(arg184_1, (4, 16), (1, 4), 0), out=buf490)
        del arg184_1
        del buf489
        buf491 = reinterpret_tensor(buf490, (234, 1, 32, 32, 16), (16384, 1, 512, 16, 1), 0); del buf490  # reuse
        # Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_38.run(buf491, arg185_1, 3833856, grid=grid(3833856), stream=stream0)
        del arg185_1
        buf492 = reinterpret_tensor(buf483, (239616, 8), (8, 1), 0); del buf483  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf491, (239616, 16), (16, 1), 0), reinterpret_tensor(arg186_1, (16, 8), (1, 16), 0), out=buf492)
        del arg186_1
        del buf491
        buf493 = reinterpret_tensor(buf482, (234, 8, 32, 32, 1, 1), (8192, 1024, 32, 1, 1, 1), 0); del buf482  # reuse
        # Topologically Sorted Source Nodes: [einsum_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf492, arg187_1, buf493, 1872, 1024, grid=grid(1872, 1024), stream=stream0)
        del arg187_1
        buf495 = reinterpret_tensor(buf492, (1872, 32, 32), (1024, 32, 1), 0); del buf492  # reuse
        # Topologically Sorted Source Nodes: [einsum_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf493, (1872, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf494, (1872, 32, 32), (1024, 32, 1), 0), out=buf495)
        del buf493
        del buf494
        buf496 = reinterpret_tensor(buf485, (234, 1, 32, 8, 32), (8192, 1, 256, 32, 1), 0); del buf485  # reuse
        # Topologically Sorted Source Nodes: [x_mid_9], Original ATen: [aten.add]
        triton_poi_fused_add_40.run(buf496, buf495, 1916928, grid=grid(1916928), stream=stream0)
        buf497 = reinterpret_tensor(buf495, (7488, 256), (256, 1), 0); del buf495  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf496, (7488, 256), (256, 1), 0), reinterpret_tensor(arg188_1, (256, 256), (1, 256), 0), out=buf497)
        del arg188_1
        buf518 = reinterpret_tensor(buf469, (234, 32, 1024), (32768, 1024, 1), 0); del buf469  # reuse
        buf514 = reinterpret_tensor(buf518, (234, 32, 256), (32768, 1024, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_mid_o_4, z_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_41.run(buf470, buf497, arg189_1, arg190_1, arg191_1, buf514, 7488, 256, grid=grid(7488), stream=stream0)
        del arg190_1
        del arg191_1
        buf501 = buf429; del buf429  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf467, (8192, 256), (256, 1), 0), reinterpret_tensor(arg196_1, (256, 256), (1, 256), 0), out=buf501)
        del arg196_1
        buf502 = reinterpret_tensor(buf501, (1, 64, 128, 256), (2097152, 32768, 256, 1), 0); del buf501  # reuse
        # Topologically Sorted Source Nodes: [input_71], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf502, arg197_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg197_1
        buf503 = reinterpret_tensor(buf410, (8192, 128), (128, 1), 0); del buf410  # reuse
        # Topologically Sorted Source Nodes: [input_72], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg199_1, reinterpret_tensor(buf502, (8192, 256), (256, 1), 0), reinterpret_tensor(arg198_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf503)
        del arg198_1
        del arg199_1
        del buf502
        buf504 = empty_strided_cuda((64, 128, 128), (16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf503, (64, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf503, (64, 128, 128), (16384, 1, 128), 0), out=buf504)
        del buf503
        # Topologically Sorted Source Nodes: [toks_17, index_add__38, index_add__39], Original ATen: [aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_43.run(buf140, buf141, buf470, buf497, arg189_1, buf146, buf505, buf509, 14745600, grid=grid(14745600), stream=stream0)
        buf508 = buf468; del buf468  # reuse
        # Topologically Sorted Source Nodes: [row_t_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg30_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf505, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf508)
        del arg30_1
        del buf505
        buf512 = buf438; del buf438  # reuse
        # Topologically Sorted Source Nodes: [col_t_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg31_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf509, (1, 64, 32768), (2097152, 32768, 1), 0), out=buf512)
        del arg31_1
        del buf509
        buf513 = buf443; del buf443  # reuse
        # Topologically Sorted Source Nodes: [x_mid_o_4, g_9], Original ATen: [aten.add, aten.mean]
        triton_per_fused_add_mean_44.run(buf470, buf497, arg189_1, buf513, 59904, 32, grid=grid(59904), stream=stream0)
        buf515 = reinterpret_tensor(buf518, (234, 32, 256), (32768, 1024, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [z_19], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf508, buf515, 1916928, grid=grid(1916928), stream=stream0)
        del buf508
        buf516 = reinterpret_tensor(buf518, (234, 32, 256), (32768, 1024, 1), 512)  # alias
        # Topologically Sorted Source Nodes: [z_19], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf512, buf516, 1916928, grid=grid(1916928), stream=stream0)
        buf517 = reinterpret_tensor(buf518, (234, 32, 256), (32768, 1024, 1), 768)  # alias
        # Topologically Sorted Source Nodes: [z_19], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf513, buf517, 1916928, grid=grid(1916928), stream=stream0)
        del buf513
        del buf514
        del buf515
        del buf516
        del buf517
        buf519 = reinterpret_tensor(buf512, (7488, 1024), (1024, 1), 0); del buf512  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf518, (7488, 1024), (1024, 1), 0), reinterpret_tensor(arg192_1, (1024, 1024), (1, 1024), 0), out=buf519)
        del arg192_1
        del buf518
        buf520 = reinterpret_tensor(buf519, (234, 32, 1024), (32768, 1024, 1), 0); del buf519  # reuse
        # Topologically Sorted Source Nodes: [input_68], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_47.run(buf520, arg193_1, 7667712, grid=grid(7667712), stream=stream0)
        del arg193_1
        buf521 = reinterpret_tensor(buf496, (7488, 256), (256, 1), 0); del buf496  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf520, (7488, 1024), (1024, 1), 0), reinterpret_tensor(arg194_1, (1024, 256), (1, 1024), 0), out=buf521)
        del arg194_1
        del buf520
        buf522 = reinterpret_tensor(buf521, (234, 32, 256), (8192, 256, 1), 0); del buf521  # reuse
        # Topologically Sorted Source Nodes: [x_mid_o_4, off_stream_4], Original ATen: [aten.add]
        triton_poi_fused_add_48.run(buf522, buf470, buf497, arg189_1, arg195_1, 1916928, grid=grid(1916928), stream=stream0)
        del arg189_1
        del arg195_1
        del buf470
        buf523 = buf497; del buf497  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf522, (7488, 256), (256, 1), 0), reinterpret_tensor(arg200_1, (256, 256), (1, 256), 0), out=buf523)
        del arg200_1
        buf524 = reinterpret_tensor(buf523, (234, 1, 32, 256), (8192, 1, 256, 1), 0); del buf523  # reuse
        # Topologically Sorted Source Nodes: [input_74], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_52.run(buf524, arg201_1, 1916928, grid=grid(1916928), stream=stream0)
        del arg201_1
        buf525 = reinterpret_tensor(buf478, (7488, 32), (32, 1), 0); del buf478  # reuse
        # Topologically Sorted Source Nodes: [input_75], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg203_1, reinterpret_tensor(buf524, (7488, 256), (256, 1), 0), reinterpret_tensor(arg202_1, (256, 32), (1, 256), 0), alpha=1, beta=1, out=buf525)
        del arg202_1
        del arg203_1
        buf526 = reinterpret_tensor(buf524, (7488, 256), (256, 1), 0); del buf524  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf522, (7488, 256), (256, 1), 0), reinterpret_tensor(arg204_1, (256, 256), (1, 256), 0), out=buf526)
        del arg204_1
        buf527 = reinterpret_tensor(buf526, (234, 1, 32, 256), (8192, 1, 256, 1), 0); del buf526  # reuse
        # Topologically Sorted Source Nodes: [input_77], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_52.run(buf527, arg205_1, 1916928, grid=grid(1916928), stream=stream0)
        del arg205_1
        buf528 = empty_strided_cuda((7488, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_78], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg207_1, reinterpret_tensor(buf527, (7488, 256), (256, 1), 0), reinterpret_tensor(arg206_1, (256, 32), (1, 256), 0), alpha=1, beta=1, out=buf528)
        del arg206_1
        del arg207_1
        del buf527
        buf529 = empty_strided_cuda((234, 32, 32), (1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf525, (234, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf528, (234, 32, 32), (1024, 1, 32), 0), out=buf529)
        del buf525
        del buf528
        buf530 = empty_strided_cuda((8192, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [node_U], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg211_1, reinterpret_tensor(buf467, (8192, 256), (256, 1), 0), reinterpret_tensor(arg210_1, (256, 32), (1, 256), 0), alpha=1, beta=1, out=buf530)
        del arg210_1
        del arg211_1
        buf531 = empty_strided_cuda((8192, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [node_V], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg213_1, reinterpret_tensor(buf467, (8192, 256), (256, 1), 0), reinterpret_tensor(arg212_1, (256, 32), (1, 256), 0), alpha=1, beta=1, out=buf531)
        del arg212_1
        del arg213_1
        buf532 = empty_strided_cuda((8192, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf467, (8192, 256), (256, 1), 0), reinterpret_tensor(arg208_1, (256, 1), (1, 256), 0), out=buf532)
        del arg208_1
        del buf467
        buf533 = reinterpret_tensor(buf532, (1, 64, 128, 1), (8192, 128, 1, 1), 0); del buf532  # reuse
        # Topologically Sorted Source Nodes: [sigmoid], Original ATen: [aten.sigmoid]
        triton_poi_fused_sigmoid_53.run(buf533, arg209_1, 8192, grid=grid(8192), stream=stream0)
        del arg209_1
        buf534 = empty_strided_cuda((1, 1820672), (1820672, 1), torch.float32)
        # Topologically Sorted Source Nodes: [packed_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_54.run(buf504, buf529, buf530, buf531, buf533, buf534, 1820672, grid=grid(1820672), stream=stream0)
        del buf504
        del buf529
        del buf530
        del buf531
        # Topologically Sorted Source Nodes: [toks_19, index_add__43, index_add__42], Original ATen: [aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_49.run(buf146, buf141, buf522, buf140, buf535, buf538, 14745600, grid=grid(14745600), stream=stream0)
        del buf522
    return (buf534, reinterpret_tensor(buf533, (1, 64, 128), (8192, 128, 1), 0), reinterpret_tensor(buf535, (1, 8192, 256), (2097152, 256, 1), 0), reinterpret_tensor(buf538, (1, 8192, 256), (2097152, 256, 1), 0), arg42_1, buf140, buf146, buf141, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 8192, 9), (73728, 9, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((256, 18), (18, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((2, 40596), (40596, 1), device='cuda:0', dtype=torch.int64)
    arg7_1 = rand_strided((40596, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, 128, 128, 4), (65536, 512, 4, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((234, 128, 128, 4), (65536, 512, 4, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((234, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((234, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg43_1 = rand_strided((450, ), (1, ), device='cpu', dtype=torch.int64)
    arg44_1 = rand_strided((450, ), (1, ), device='cpu', dtype=torch.int64)
    arg45_1 = rand_strided((450, ), (1, ), device='cpu', dtype=torch.int64)
    arg46_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((32, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((32, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((32, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((32, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
