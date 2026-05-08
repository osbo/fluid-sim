# AOT ID: ['3_inference']
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ar/car64ymdldd2xtelq2bhgysj327ebe4aobrmyghz5c7baezor3px.py
# Topologically Sorted Source Nodes: [off_diag_blocks_1], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   off_diag_blocks_1 => mul
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, %arg3_1), kwargs = {})
triton_poi_fused_mul_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 239616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (950272 + x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/mk/cmko3ki7dxvlsqysaa65l2i564i3xpcnfvzpwbvplgq2xoj2cjfy.py
# Topologically Sorted Source Nodes: [zero__3], Original ATen: [aten.zero]
# Source node to ATen node mapping:
#   zero__3 => full_default_1
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([234, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zero_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zero_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/6q/c6qjuy34ypanfqlcdnplioenb2lpbk2pentiwn6sbr3zezx7zwqz.py
# Topologically Sorted Source Nodes: [zero__1, zero__3, index_select_1, index_add__1], Original ATen: [aten.zero, aten.index_select, aten.index_add]
# Source node to ATen node mapping:
#   index_add__1 => index_put_1
#   index_select_1 => index_1
#   zero__1 => full_1
#   zero__3 => full_default_1
# Graph fragment:
#   %full_1 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([64, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([234, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_1, %view_23, 0, 0, 58), kwargs = {})
#   %index_1 : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%slice_scatter_default, [%arg7_1]), kwargs = {})
#   %index_put_1 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_1, [%arg8_1], %index_1, True), kwargs = {})
triton_poi_fused_index_add_index_select_zero_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_add_index_select_zero_2', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 32)
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 234, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 234)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 234")
    tmp7 = tl.full([XBLOCK], 64, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert(((0 <= tmp10) & (tmp10 < 64)) | ~(xmask), "index out of bounds: 0 <= tmp10 < 64")
    tmp12 = tmp10
    tmp13 = tl.full([1], 58, tl.int64)
    tmp14 = tmp12 < tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + (32*tmp10)), tmp14 & xmask, other=0.0)
    tmp16 = 0.0
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tl.atomic_add(out_ptr0 + (x0 + (32*tmp4)), tmp17, xmask, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/2x/c2x5urqaoaxmsthe4jgrxwfa5yvkoz6ikqph2ako657eoeppulv3.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %copy__default : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%select_int, %bmm_3), kwargs = {})
triton_poi_fused_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/bv/cbvse3wz3tx6higwp5yuwh2urztpixtsydmkugwuru3pdiahgcyy.py
# Topologically Sorted Source Nodes: [zero__4], Original ATen: [aten.zero]
# Source node to ATen node mapping:
#   zero__4 => full_default_2
# Graph fragment:
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 64, 32, 1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zero_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zero_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/oe/coeofonrbqxiz5fwscugarwphzxqqacdbx2mb7l3cu5au4ijvtpi.py
# Topologically Sorted Source Nodes: [zero_, zero__4, index_select_2, index_add__2, zero__2, index_select, index_add_], Original ATen: [aten.zero, aten.index_select, aten.index_add]
# Source node to ATen node mapping:
#   index_add_ => index_put
#   index_add__2 => index_put_2
#   index_select => index
#   index_select_2 => index_2
#   zero_ => full
#   zero__2 => full_default
#   zero__4 => full_default_2
# Graph fragment:
#   %full : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([64, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 64, 32, 1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_2 : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg13_1, [None, %arg8_1]), kwargs = {})
#   %index_put_2 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_2, [None, %arg6_1], %index_2, True), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([234, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default_1 : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full, %view_22, 0, 0, 58), kwargs = {})
#   %index : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%slice_scatter_default_1, [%arg6_1]), kwargs = {})
#   %index_put : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default, [%arg8_1], %index, True), kwargs = {})
#   %copy__3 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg9_1, %index), kwargs = {})
triton_poi_fused_index_add_index_select_zero_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_add_index_select_zero_5', 'mutated_arg_names': ['out_ptr0', 'out_ptr1', 'out_ptr3'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 32)
    x0 = xindex % 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 64, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 64)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 64")
    tmp7 = tl.full([XBLOCK], 234, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert(((0 <= tmp10) & (tmp10 < 234)) | ~(xmask), "index out of bounds: 0 <= tmp10 < 234")
    tmp12 = tl.load(in_ptr2 + (x0 + (32*tmp10)), xmask)
    tmp13 = tmp4
    tmp14 = tl.full([1], 58, tl.int64)
    tmp15 = tmp13 < tmp14
    tmp16 = tl.load(in_ptr3 + (x0 + (32*tmp4)), tmp15 & xmask, other=0.0)
    tmp17 = 0.0
    tmp18 = tl.where(tmp15, tmp16, tmp17)
    tl.atomic_add(out_ptr0 + (x0 + (32*tmp4)), tmp12, xmask, sem='relaxed')
    tl.atomic_add(out_ptr1 + (x0 + (32*tmp10)), tmp18, xmask, sem='relaxed')
    tl.store(out_ptr3 + (x2), tmp18, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/cp/ccpbfcfuesl3w4rmunkxy7qhd5pp3ow7ebqwqkmupwlt7eiqejat.py
# Topologically Sorted Source Nodes: [zero__1, index_select_1, index_select_2, zero__5, index_select_3, index_add__3], Original ATen: [aten.zero, aten.index_select, aten.index_add]
# Source node to ATen node mapping:
#   index_add__3 => index_put_3
#   index_select_1 => index_1
#   index_select_2 => index_2
#   index_select_3 => index_3
#   zero__1 => full_1
#   zero__5 => full_default_3
# Graph fragment:
#   %full_1 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([64, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_1, %view_23, 0, 0, 58), kwargs = {})
#   %index_1 : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%slice_scatter_default, [%arg7_1]), kwargs = {})
#   %index_2 : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg13_1, [None, %arg8_1]), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 64, 32, 1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_3 : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg14_1, [None, %arg8_1]), kwargs = {})
#   %index_put_3 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_3, [None, %arg7_1], %index_3, True), kwargs = {})
#   %copy__5 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg11_1, %index_1), kwargs = {})
#   %copy__9 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg15_1, %index_2), kwargs = {})
#   %copy__10 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg16_1, %index_3), kwargs = {})
triton_poi_fused_index_add_index_select_zero_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_add_index_select_zero_6', 'mutated_arg_names': ['out_ptr0', 'out_ptr2', 'out_ptr4', 'out_ptr6'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr4, out_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 32)
    x0 = xindex % 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 64, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 64)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 64")
    tmp7 = tl.full([XBLOCK], 234, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert(((0 <= tmp10) & (tmp10 < 234)) | ~(xmask), "index out of bounds: 0 <= tmp10 < 234")
    tmp12 = tl.load(in_ptr2 + (x0 + (32*tmp10)), xmask)
    tmp13 = tmp4
    tmp14 = tl.full([1], 58, tl.int64)
    tmp15 = tmp13 < tmp14
    tmp16 = tl.load(in_ptr3 + (x0 + (32*tmp4)), tmp15 & xmask, other=0.0)
    tmp17 = 0.0
    tmp18 = tl.where(tmp15, tmp16, tmp17)
    tmp19 = tl.load(in_ptr4 + (x0 + (32*tmp10)), xmask)
    tl.atomic_add(out_ptr0 + (x0 + (32*tmp4)), tmp12, xmask, sem='relaxed')
    tl.store(out_ptr2 + (x2), tmp18, xmask)
    tl.store(out_ptr4 + (x2), tmp19, xmask)
    tl.store(out_ptr6 + (x2), tmp12, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/27/c27ws2h32pjduwfqywap2fhciyeibmmnkmy45gmvrvmtxed3atjy.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %copy_ : [num_users=1] = call_function[target=torch.ops.aten.copy_.default](args = (%arg1_1, %view_42), kwargs = {})
triton_poi_fused_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    tmp8 = tl.load(in_ptr3 + (1665024 + x0), xmask)
    tmp9 = tl.load(in_ptr4 + (x0), xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = 1.0
    tmp7 = tmp5 * tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp4 + tmp11
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/qx/cqx6aufcsiocunrs6dylheougys74e6tdnu3es3fczogbpgq6w2r.py
# Topologically Sorted Source Nodes: [zero_], Original ATen: [aten.zero]
# Source node to ATen node mapping:
#   zero_ => full
# Graph fragment:
#   %full : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([64, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default_1 : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full, %view_22, 0, 0, 58), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg4_1, %slice_scatter_default_1), kwargs = {})
triton_poi_fused_zero_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zero_8', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 32)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 58, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x2), tmp2, other=0.0)
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/nt/cntkvrzcdjs2nejwb5twsizf4dq5t2ma53rls4klnaawxw526tie.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %copy__11 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg17_1, %index_put_2), kwargs = {})
triton_poi_fused_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1 = args
    args.clear()
    assert_size_stride(arg0_1, (7424, 1), (1, 1))
    assert_size_stride(arg1_1, (7424, 1), (1, 1))
    assert_size_stride(arg2_1, (1, 1672448), (1672448, 1))
    assert_size_stride(arg3_1, (1, 234, 1, 1), (234, 1, 1, 1))
    assert_size_stride(arg4_1, (64, 32), (32, 1))
    assert_size_stride(arg5_1, (64, 32), (32, 1))
    assert_size_stride(arg6_1, (450, ), (1, ))
    assert_size_stride(arg7_1, (450, ), (1, ))
    assert_size_stride(arg8_1, (450, ), (1, ))
    assert_size_stride(arg9_1, (450, 32), (32, 1))
    assert_size_stride(arg10_1, (234, 32), (32, 1))
    assert_size_stride(arg11_1, (450, 32), (32, 1))
    assert_size_stride(arg12_1, (234, 32), (32, 1))
    assert_size_stride(arg13_1, (1, 234, 32, 1), (7488, 32, 1, 1))
    assert_size_stride(arg14_1, (1, 234, 32, 1), (7488, 32, 1, 1))
    assert_size_stride(arg15_1, (1, 450, 32, 1), (14400, 32, 1, 1))
    assert_size_stride(arg16_1, (1, 450, 32, 1), (14400, 32, 1, 1))
    assert_size_stride(arg17_1, (1, 64, 32, 1), (2048, 32, 1, 1))
    assert_size_stride(arg18_1, (1, 64, 32, 1), (2048, 32, 1, 1))
    assert_size_stride(arg19_1, (1, 7424), (7424, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((58, 128, 1), (128, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg2_1, (58, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(arg0_1, (58, 128, 1), (128, 1, 1), 0), out=buf0)
        buf1 = empty_strided_cuda((1, 234, 32, 32), (239616, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [off_diag_blocks_1], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(arg2_1, arg3_1, buf1, 239616, grid=grid(239616), stream=stream0)
        del arg3_1
        buf2 = empty_strided_cuda((58, 32, 1), (32, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_proj_V], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg2_1, (58, 32, 128), (4096, 1, 32), 1427456), reinterpret_tensor(arg0_1, (58, 128, 1), (128, 1, 1), 0), out=buf2)
        buf3 = empty_strided_cuda((234, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [zero__3], Original ATen: [aten.zero]
        triton_poi_fused_zero_1.run(buf3, 7488, grid=grid(7488), stream=stream0)
        # Topologically Sorted Source Nodes: [zero__1, zero__3, index_select_1, index_add__1], Original ATen: [aten.zero, aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_zero_2.run(arg8_1, arg7_1, buf2, buf3, 14400, grid=grid(14400), stream=stream0)
        buf5 = empty_strided_cuda((234, 32, 1), (32, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1, (234, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf3, (234, 32, 1), (32, 1, 1), 0), out=buf5)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf5, arg13_1, 7488, grid=grid(7488), stream=stream0)
        buf7 = empty_strided_cuda((1, 64, 32, 1), (2048, 32, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [zero__4], Original ATen: [aten.zero]
        triton_poi_fused_zero_4.run(buf7, 2048, grid=grid(2048), stream=stream0)
        buf10 = empty_strided_cuda((58, 32, 1), (32, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_proj_U], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg2_1, (58, 32, 128), (4096, 1, 32), 1189888), reinterpret_tensor(arg0_1, (58, 128, 1), (128, 1, 1), 0), out=buf10)
        buf11 = reinterpret_tensor(buf5, (234, 32), (32, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [zero__2], Original ATen: [aten.zero]
        triton_poi_fused_zero_1.run(buf11, 7488, grid=grid(7488), stream=stream0)
        # Topologically Sorted Source Nodes: [zero_, zero__4, index_select_2, index_add__2, zero__2, index_select, index_add_], Original ATen: [aten.zero, aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_zero_5.run(arg6_1, arg8_1, arg13_1, buf10, buf7, buf11, arg9_1, 14400, grid=grid(14400), stream=stream0)
        del arg6_1
        del arg9_1
        buf9 = empty_strided_cuda((58, 128, 1), (128, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_r_final], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg2_1, (58, 128, 32), (4096, 32, 1), 1189888), reinterpret_tensor(buf7, (58, 32, 1), (32, 1, 1), 0), out=buf9)
        buf13 = empty_strided_cuda((234, 32, 1), (32, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1, (234, 32, 32), (1024, 1, 32), 0), reinterpret_tensor(buf11, (234, 32, 1), (32, 1, 1), 0), out=buf13)
        del buf1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf13, arg14_1, 7488, grid=grid(7488), stream=stream0)
        del buf13
        buf15 = empty_strided_cuda((1, 64, 32, 1), (2048, 32, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [zero__5], Original ATen: [aten.zero]
        triton_poi_fused_zero_4.run(buf15, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [zero__1, index_select_1, index_select_2, zero__5, index_select_3, index_add__3], Original ATen: [aten.zero, aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_zero_6.run(arg7_1, arg8_1, arg14_1, buf2, arg13_1, buf15, arg11_1, arg15_1, arg16_1, 14400, grid=grid(14400), stream=stream0)
        del arg11_1
        del arg13_1
        del arg14_1
        del arg15_1
        del arg16_1
        del arg7_1
        del arg8_1
        buf17 = empty_strided_cuda((58, 128, 1), (128, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_c_final], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg2_1, (58, 128, 32), (4096, 32, 1), 1427456), reinterpret_tensor(buf15, (58, 32, 1), (32, 1, 1), 0), out=buf17)
        buf18 = reinterpret_tensor(buf0, (7424, 1), (1, 7424), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf18, buf9, buf17, arg0_1, arg2_1, arg19_1, arg1_1, 7424, grid=grid(7424), stream=stream0)
        del arg0_1
        del arg19_1
        del arg2_1
        del buf17
        del buf18
        del buf9
        # Topologically Sorted Source Nodes: [zero_], Original ATen: [aten.zero]
        triton_poi_fused_zero_8.run(buf10, arg4_1, 2048, grid=grid(2048), stream=stream0)
        del arg4_1
        del buf10
        # Topologically Sorted Source Nodes: [zero__1], Original ATen: [aten.zero]
        triton_poi_fused_zero_8.run(buf2, arg5_1, 2048, grid=grid(2048), stream=stream0)
        del arg5_1
        del buf2
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf11, arg10_1, 7488, grid=grid(7488), stream=stream0)
        del arg10_1
        del buf11
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf3, arg12_1, 7488, grid=grid(7488), stream=stream0)
        del arg12_1
        del buf3
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf7, arg17_1, 2048, grid=grid(2048), stream=stream0)
        del arg17_1
        del buf7
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(buf15, arg18_1, 2048, grid=grid(2048), stream=stream0)
        del arg18_1
        del buf15
    return (arg1_1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((7424, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((7424, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 1672448), (1672448, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 234, 1, 1), (234, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((450, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg7_1 = rand_strided((450, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg8_1 = rand_strided((450, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg9_1 = rand_strided((450, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((234, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((450, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((234, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((1, 234, 32, 1), (7488, 32, 1, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((1, 234, 32, 1), (7488, 32, 1, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((1, 450, 32, 1), (14400, 32, 1, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((1, 450, 32, 1), (14400, 32, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((1, 64, 32, 1), (2048, 32, 1, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((1, 64, 32, 1), (2048, 32, 1, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((1, 7424), (7424, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
