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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/6m/c6mkroeer47lof7sdw6o4blm5kjfqels53pgvp2kv3vco4zb45wv.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view, %full_default_28], 1), kwargs = {})
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ev/ceveqsjtrb52ihrsk7ubsdret6zecokr7zd67zrvc4gmua344wj5.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute, %full_default_29],), kwargs = {})
triton_poi_fused_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128)
    x0 = xindex % 128
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/id/cid2mdqg2cvq2nqitmccoskp42tn26hzqpuzfeydmhg4edsnzpwv.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/mc/cmczfa3odlh637eqyugsedmnwas2vrit6wprfswynn746bbqaiqy.py
# Topologically Sorted Source Nodes: [aggr], Original ATen: [aten.zeros_like]
# Source node to ATen node mapping:
#   aggr => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8192, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_like_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_like_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (256*x1)), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/t2/ct2mu3lknuysbduzk7gjrllbljlz4xm7d43qdmijxekg4ug6rden.py
# Topologically Sorted Source Nodes: [aggr, getitem_4, messages, index_add_], Original ATen: [aten.zeros_like, aten.index, aten.mul, aten.index_add]
# Source node to ATen node mapping:
#   aggr => full_default
#   getitem_4 => index
#   index_add_ => index_put
#   messages => mul_3
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8192, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_index_add_mul_zeros_like_4', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5196288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128)
    x0 = xindex % 128
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
    tmp11 = tl.load(in_ptr1 + (x0 + (128*tmp9)), xmask)
    tmp13 = tmp11 * tmp12
    tl.atomic_add(out_ptr0 + (x0 + (256*tmp4)), tmp13, xmask, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/nv/cnvc7uf2q2lc3oxci32nvstkdvbp3ourem6vbl4jskurbmjfjpsi.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   out => add_2
# Graph fragment:
#   %add_tensor_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_42, %arg15_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze, %add_tensor_42), kwargs = {})
triton_poi_fused_add_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/bs/cbsxwvytjzqkkjjx3wni4jp2wsoyb3gik7gea4kz2ivdbyjvmnmd.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.sum(tmp15, 1)[:, None]
    tmp18 = tmp4 - tmp12
    tmp19 = 128.0
    tmp20 = tmp17 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp28, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/uw/cuwt46nqcqygkfw5efya4rlout6ekyti2z65ow6ot2d6g63epmjo.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), None)
    tmp21 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.sum(tmp11, 1)[:, None]
    tmp14 = tmp0 - tmp8
    tmp15 = 128.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp24, None)
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/6r/c6ro3uljs7fssvkc4clnzz35krxankgs46agyvooq3vcfu2dbej3.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_12', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/6o/c6otbptcamhlvp6hxhihzh6i4kg75ctsf5onlmvjoefwbuyduqsx.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_13', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/4u/c4u6v3hsxur6k6nmc3lgnem3krktnjqihwcnueaslobjvehl4zm5.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_14', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/3b/c3bszioqysnu4ls2vnk7ykeywtrhtxk73yrjbcctfhtcrtrywgz3.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_19', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x0 = xindex % 16
    x1 = (xindex // 16) % 8
    x2 = (xindex // 128) % 128
    x3 = (xindex // 16384)
    tmp0 = tl.load(in_out_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (16*x2) + (2048*x1) + (16384*x3)), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/tw/ctwtmnpebjcmhpetiv42gx4xdfl4qm5iw5zxoxcwy3w3zx34qsay.py
# Topologically Sorted Source Nodes: [row_highway], Original ATen: [aten.new_zeros]
# Source node to ATen node mapping:
#   row_highway => full_default_4
# Graph fragment:
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 128, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/jw/cjwmee6jhccnbg2w3zvmexohyr5kcgt6elm2tchxrogwouphvinu.py
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
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %arg43_1), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %arg44_1), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 128, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_4 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_4, [None, %arg42_1], %view_47, True), kwargs = {})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 128, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_5 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_5, [None, %arg42_1], %view_47, True), kwargs = {})
triton_per_fused_add_index_add_native_layer_norm_new_zeros_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_index_add_native_layer_norm_new_zeros_21', 'mutated_arg_names': ['out_ptr2', 'out_ptr3'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x3 = (xindex // 128)
    x2 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.sum(tmp15, 1)[:, None]
    tmp19 = tl.full([XBLOCK, RBLOCK], 128, tl.int32)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp18)
    tl.device_assert((0 <= tmp22) & (tmp22 < 128), "index out of bounds: 0 <= tmp22 < 128")
    tmp24 = tmp4 - tmp12
    tmp25 = 128.0
    tmp26 = tmp17 / tmp25
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp24 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tl.atomic_add(out_ptr2 + (r1 + (128*x2) + (16384*tmp22)), tmp4, None, sem='relaxed')
    tl.atomic_add(out_ptr3 + (r1 + (128*x2) + (16384*tmp22)), tmp4, None, sem='relaxed')
    tl.store(out_ptr4 + (r1 + (512*x0)), tmp34, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/zo/czozcmlm64torzr2qq4q5syurnzuds2vdx2sbpd6go5trlspepdn.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mean_22', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/3i/c3ithsijxlelsr5pv5j6vssaorgelng7wdqfrj252wdudz4nldeo.py
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
    size_hints=[128, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_23', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/hs/chs2ei7pza7o3lpducu2nhtkh2qq6bpsrcxlsiuqcih5uccqh4qn.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_24', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (512*x1)), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/gf/cgf74gebpdaxi6hxqzjdx6hxglnm376ndq2jbamey5uersc5ty7i.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_25', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = 8192.0
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + (x0 + (512*x1)), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ua/cuah625inlsqltgu6bhhvuhoozznjnvirxmq5jls56njk24vvukr.py
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
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_26', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/pt/cptdbga4uq5f4vw2evswgj2cxcqbs6w6wipiibyyms6nug2pelyk.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_27', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (r1 + (128*x0)), None)
    tmp6 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp13 = tl.sum(tmp11, 1)[:, None]
    tmp14 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp9 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.sum(tmp19, 1)[:, None]
    tmp22 = tmp8 - tmp16
    tmp23 = 128.0
    tmp24 = tmp21 / tmp23
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp8, None)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp32, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/k6/ck6tw7mm7apfyuxmko545ry5e5yhzyafd2cawgrf7ywpnpa5nqgc.py
# Topologically Sorted Source Nodes: [h_k_3], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   h_k_3 => constant_pad_nd_1
# Graph fragment:
#   %constant_pad_nd_1 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_64, [0, 0, 0, 0, 0, 64], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_28', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 16384)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 64, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x2), tmp2, other=0.0)
    tl.store(out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/gn/cgncbh2z3df7uu4sjq3v25luxkn3hsveh5suptgpnnnepqxisas7.py
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
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %arg49_1), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %arg50_1), kwargs = {})
triton_per_fused_mean_native_layer_norm_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_layer_norm_29', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 15584
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), xmask, other=0.0)
    tmp3 = tl.load(in_ptr0 + (128 + r1 + (512*x0)), xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (128 + r1 + (512*x0)), xmask, other=0.0)
    tmp7 = tl.load(in_ptr0 + (256 + r1 + (512*x0)), xmask, other=0.0)
    tmp8 = tl.load(in_ptr1 + (256 + r1 + (512*x0)), xmask, other=0.0)
    tmp11 = tl.load(in_ptr0 + (384 + r1 + (512*x0)), xmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + (384 + r1 + (512*x0)), xmask, other=0.0)
    tmp40 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp22 = tl.where(xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp33 = tmp16 - tmp26
    tmp34 = 128.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-05
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tl.store(out_ptr0 + (r1 + (128*x0)), tmp16, xmask)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp43, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/st/cstphxt7wjm3bru3vx7qzgrjvv2scnfrbu3elcjwkwn47ookpf7b.py
# Topologically Sorted Source Nodes: [sub, setitem_2, setitem_6], Original ATen: [aten.mean, aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_2 => full_default_6, index_put_6
#   setitem_6 => full_default_14, index_put_18
#   sub => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_6, [2, 4]), kwargs = {})
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_6 : [num_users=2] = call_function[target=torch.ops.aten.index_put.default](args = (%view_7, [None, None, %iota_1, %iota_1], %full_default_6), kwargs = {})
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_18 : [num_users=2] = call_function[target=torch.ops.aten.index_put.default](args = (%view_7, [None, None, %iota_3, %iota_3], %full_default_14), kwargs = {})
triton_per_fused_index_put_lift_fresh_mean_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2097152, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_index_put_lift_fresh_mean_30', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1994752
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
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/7b/c7bktiyrwl5rlqscm35pvnd5oawmet7nuxgrghdr5aelz33lad2j.py
# Topologically Sorted Source Nodes: [setitem_2], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_2 => full_default_6, index_put_6
# Graph fragment:
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_6 : [num_users=2] = call_function[target=torch.ops.aten.index_put.default](args = (%view_7, [None, None, %iota_1, %iota_1], %full_default_6), kwargs = {})
triton_poi_fused_index_put_lift_fresh_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_31', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = (xindex // 4) % 32
    x2 = (xindex // 128)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (132*x1) + (4096*x2)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/g7/cg7qhxfbauqjkwokei6nodw52speqbjbwktti4jfgfpsb6hgfzaq.py
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
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_32', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 498688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + (4*x0)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/tu/ctum74ee3c6msogafgvnd6s3nmuvlk6kmw7f5acpiqdniue3zxkr.py
# Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_3 => full_default_7, index_put_7
# Graph fragment:
#   %full_default_7 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_7 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_10, [None, None, %iota_1, %iota_1], %full_default_7), kwargs = {})
triton_poi_fused_index_put_lift_fresh_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_33', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp0 = 1.0
    tl.store(out_ptr0 + ((33*x0) + (1024*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/4x/c4xicuyqwuloc7phmj6oamnc2gy36aiiuscrdyz7lythvg2ejvae.py
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
    xnumel = 1994752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16) % 32
    x2 = (xindex // 512) % 8
    x3 = (xindex // 4096)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*x2) + (384*x1) + (12288*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/cf/ccfkqzw76nvv637wf6owjmewcjxzrruyhqwcuyjio4ye37dbdghf.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_35', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1994752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16) % 32
    x2 = (xindex // 512) % 8
    x3 = (xindex // 4096)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0 + (16*x2) + (384*x1) + (12288*x3)), None)
    tmp1 = tl.load(in_ptr1 + (128 + x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/bs/cbsg3weecn6tl2tkeenfykd2fz655fptksp3eet4dc6gj6guwxwr.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_36', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1994752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16) % 32
    x2 = (xindex // 512) % 8
    x3 = (xindex // 4096)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + (16*x2) + (384*x1) + (12288*x3)), None)
    tmp1 = tl.load(in_ptr1 + (256 + x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
    tl.store(out_ptr1 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/5u/c5ungkdypoovpmi5afokhfnar6k5hy4eff3nh7ogqftwm4sluolc.py
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
triton_poi_fused__scaled_dot_product_efficient_attention_clone_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_37', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3989504
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/2h/c2h67ymhjbcpifeh3mur62ukvqpjg2l5ohc5x23biv7lxafejd4z.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %select_scatter_default_1 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%index_put_6, %index_put_7, 4, 3), kwargs = {})
triton_poi_fused_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_38', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1994752
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/pg/cpg72vlgijwtdate52hugs7isnbhysfdcihi3zyboeisqzhixxnz.py
# Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_17 => add_20, erf_5, mul_25, mul_26, mul_27
# Graph fragment:
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_88, 0.5), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_88, 0.7071067811865476), kwargs = {})
#   %erf_5 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_26,), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_5, 1), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %add_20), kwargs = {})
triton_poi_fused_gelu_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_39', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7979008
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/n4/cn4ipyhl7myh4pxb6roz5q7cwxs3jqcobcearg3tmhutooxd2k7p.py
# Topologically Sorted Source Nodes: [einsum_5], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   einsum_5 => clone_12
# Graph fragment:
#   %clone_12 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_55,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4096, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_40', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3896
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/gh/cghptuqm2csdmdsn67hee7whrusowxbb4n2g5cxysmbp6uudnhdu.py
# Topologically Sorted Source Nodes: [x_mid_1], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_mid_1 => add_21
# Graph fragment:
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_91, %view_95), kwargs = {})
triton_poi_fused_add_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_41', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1994752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x0 = xindex % 16
    x1 = (xindex // 16) % 8
    x2 = (xindex // 128) % 32
    x3 = (xindex // 4096)
    tmp0 = tl.load(in_out_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (16*x2) + (512*x1) + (4096*x3)), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/2u/c2undnt6bt76jkt4qsfvywi442psof665d33p5xvqsxmd52ywmxd.py
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
triton_per_fused_add_native_layer_norm_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_42', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 15584
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
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp31, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ru/crus2p3heuyesfvldevs6cfcerfbu37htcfxwb6hdrqtbt47w4fg.py
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
#   %add_31 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %view_152), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_31, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_31, %getitem_25), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_24, 1e-05), kwargs = {})
#   %rsqrt_6 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_32,), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_6), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_38, %arg78_1), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_39, %arg79_1), kwargs = {})
#   %full_default_12 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 128, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_14 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_12, [None, %arg42_1], %view_154, True), kwargs = {})
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 128, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_8 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_8, [None, %arg42_1], %view_101, True), kwargs = {})
#   %full_default_9 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 128, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_9 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_9, [None, %arg42_1], %view_101, True), kwargs = {})
#   %full_default_13 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 128, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_15 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_13, [None, %arg42_1], %view_154, True), kwargs = {})
triton_per_fused_add_index_add_native_layer_norm_new_zeros_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_index_add_native_layer_norm_new_zeros_43', 'mutated_arg_names': ['out_ptr2', 'out_ptr3', 'out_ptr5', 'out_ptr6'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x3 = (xindex // 128)
    x2 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.sum(tmp15, 1)[:, None]
    tmp19 = tl.full([XBLOCK, RBLOCK], 128, tl.int32)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp18)
    tl.device_assert((0 <= tmp22) & (tmp22 < 128), "index out of bounds: 0 <= tmp22 < 128")
    tmp24 = tmp4 - tmp12
    tmp25 = 128.0
    tmp26 = tmp17 / tmp25
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp24 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tl.atomic_add(out_ptr2 + (r1 + (128*x2) + (16384*tmp22)), tmp4, None, sem='relaxed')
    tl.atomic_add(out_ptr3 + (r1 + (128*x2) + (16384*tmp22)), tmp4, None, sem='relaxed')
    tl.store(out_ptr4 + (r1 + (512*x0)), tmp34, None)
    tl.atomic_add(out_ptr5 + (r1 + (128*x2) + (16384*tmp22)), tmp0, None, sem='relaxed')
    tl.atomic_add(out_ptr6 + (r1 + (128*x2) + (16384*tmp22)), tmp0, None, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/r2/cr2fsr6hlhbqproyl2bydmyijumwl6dm6cnazvp43ik2crtrwnzc.py
# Topologically Sorted Source Nodes: [toks_1, index_add__6, index_add__7], Original ATen: [aten.index_select, aten.index_add]
# Source node to ATen node mapping:
#   index_add__6 => index_put_10
#   index_add__7 => index_put_11
#   toks_1 => index_2
# Graph fragment:
#   %index_2 : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_103, [None, %arg59_1]), kwargs = {})
#   %index_put_10 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_8, [None, %arg60_1], %index_2, True), kwargs = {})
#   %index_put_11 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_9, [None, %arg61_1], %index_2, True), kwargs = {})
triton_poi_fused_index_add_index_select_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_add_index_select_44', 'mutated_arg_names': ['out_ptr0', 'out_ptr1'], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17858560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 16384)
    x0 = xindex % 128
    x1 = (xindex // 128) % 128
    x3 = xindex % 16384
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 128, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 128), "index out of bounds: 0 <= tmp4 < 128")
    tmp7 = tl.full([XBLOCK], 487, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert((0 <= tmp10) & (tmp10 < 487), "index out of bounds: 0 <= tmp10 < 487")
    tmp12 = tl.load(in_ptr2 + (x0 + (128*(x1 // 4)) + (4096*tmp10)), None)
    tmp13 = tl.load(in_ptr3 + (x0 + (128*(x1 // 4)) + (4096*tmp10)), None)
    tmp15 = tmp13 + tmp14
    tmp16 = tmp12 + tmp15
    tmp18 = tmp17 + tmp1
    tmp19 = tmp17 < 0
    tmp20 = tl.where(tmp19, tmp18, tmp17)
    tl.device_assert((0 <= tmp20) & (tmp20 < 128), "index out of bounds: 0 <= tmp20 < 128")
    tl.atomic_add(out_ptr0 + (x3 + (16384*tmp4)), tmp16, None, sem='relaxed')
    tl.atomic_add(out_ptr1 + (x3 + (16384*tmp20)), tmp16, None, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/2e/c2edo2pi2ruw6dwbtq4lkolqwpujxtnc57fdsa7o4trgnc7gtcum.py
# Topologically Sorted Source Nodes: [x_mid_o, g_1], Original ATen: [aten.add, aten.mean]
# Source node to ATen node mapping:
#   g_1 => mean_8
#   x_mid_o => add_22
# Graph fragment:
#   %add_22 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_5, %view_99), kwargs = {})
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_22, [1], True), kwargs = {})
triton_per_fused_add_mean_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_45', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 62336
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (4096*x1)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*r2) + (4096*x1)), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/3x/c3xpennrvk3cpojtfvwwe5mrecovyos56vjxuxyce42zxfr5fwra.py
# Topologically Sorted Source Nodes: [z_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   z_3 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_24, %view_122, %view_123, %expand_12], -1), kwargs = {})
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_46', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1994752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x1)), None)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + (512*x1)), None)
    tmp3 = tl.load(in_ptr0 + (256 + x0 + (512*x1)), None)
    tmp5 = tl.load(in_ptr0 + (384 + x0 + (512*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr0 + (x0 + (512*x1)), tmp8, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/kx/ckxss6yuociwayjntku4aolzlg3ef3g4phpfmivd6yta2qfrk6rm.py
# Topologically Sorted Source Nodes: [z_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   z_3 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_24, %view_122, %view_123, %expand_12], -1), kwargs = {})
triton_poi_fused_cat_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_47', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1994752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x2 = (xindex // 4096)
    x3 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x2)), None, eviction_policy='evict_last')
    tmp1 = 32.0
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + (x0 + (512*x3)), tmp2, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/qo/cqo5jojujknqqnx5dh5janbn72vhahds5wegex6mvrfmqluupyjj.py
# Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_20 => add_25, erf_6, mul_30, mul_31, mul_32
# Graph fragment:
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_125, 0.5), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_125, 0.7071067811865476), kwargs = {})
#   %erf_6 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_31,), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_6, 1), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, %add_25), kwargs = {})
triton_poi_fused_gelu_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_48', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7979008
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/7x/c7xky42jwbhsoq2wr4bumn5js3dmbtzwqdsxjitnpuodp7jn324i.py
# Topologically Sorted Source Nodes: [x_mid_o, off_stream], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   off_stream => add_26
#   x_mid_o => add_22
# Graph fragment:
#   %add_22 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_5, %view_99), kwargs = {})
#   %add_26 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_22, %view_127), kwargs = {})
triton_poi_fused_add_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_49', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1994752
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/jh/cjhwnsnuica7w4heckmesjb7i2opm7iak7vnfxvn7gs4d5p4x6ma.py
# Topologically Sorted Source Nodes: [toks_3, index_add__10, index_add__11], Original ATen: [aten.index_select, aten.index_add]
# Source node to ATen node mapping:
#   index_add__10 => index_put_16
#   index_add__11 => index_put_17
#   toks_3 => index_3
# Graph fragment:
#   %index_3 : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_156, [None, %arg59_1]), kwargs = {})
#   %index_put_16 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_14, [None, %arg60_1], %index_3, True), kwargs = {})
#   %index_put_17 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_15, [None, %arg61_1], %index_3, True), kwargs = {})
triton_poi_fused_index_add_index_select_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_add_index_select_50', 'mutated_arg_names': ['out_ptr0', 'out_ptr1'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17858560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 16384)
    x0 = xindex % 128
    x1 = (xindex // 128) % 128
    x3 = xindex % 16384
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 128, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 128), "index out of bounds: 0 <= tmp4 < 128")
    tmp7 = tl.full([XBLOCK], 487, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert((0 <= tmp10) & (tmp10 < 487), "index out of bounds: 0 <= tmp10 < 487")
    tmp12 = tl.load(in_ptr2 + (x0 + (128*(x1 // 4)) + (4096*tmp10)), None)
    tmp14 = tmp13 + tmp1
    tmp15 = tmp13 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp13)
    tl.device_assert((0 <= tmp16) & (tmp16 < 128), "index out of bounds: 0 <= tmp16 < 128")
    tl.atomic_add(out_ptr0 + (x3 + (16384*tmp4)), tmp12, None, sem='relaxed')
    tl.atomic_add(out_ptr1 + (x3 + (16384*tmp16)), tmp12, None, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ag/cagq6ksdhqd2ryaoyrvlqdof3f44capd5q3g2oqg2xwuf3u2twyz.py
# Topologically Sorted Source Nodes: [h_off_5, off_in, layer_norm_7], Original ATen: [aten.mean, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   h_off_5 => mean_10
#   layer_norm_7 => add_38, add_39, mul_43, mul_44, rsqrt_7, sub_7, var_mean_7
#   off_in => add_37
# Graph fragment:
#   %mean_10 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_183, [2]), kwargs = {})
#   %add_37 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_10, %add_26), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_37, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_37, %getitem_27), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-05), kwargs = {})
#   %rsqrt_7 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_38,), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_7), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %arg84_1), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %arg85_1), kwargs = {})
triton_per_fused_add_mean_native_layer_norm_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_native_layer_norm_51', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 11, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 15584
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), xmask, other=0.0)
    tmp3 = tl.load(in_ptr0 + (128 + r1 + (512*x0)), xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (128 + r1 + (512*x0)), xmask, other=0.0)
    tmp7 = tl.load(in_ptr0 + (256 + r1 + (512*x0)), xmask, other=0.0)
    tmp8 = tl.load(in_ptr1 + (256 + r1 + (512*x0)), xmask, other=0.0)
    tmp11 = tl.load(in_ptr0 + (384 + r1 + (512*x0)), xmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + (384 + r1 + (512*x0)), xmask, other=0.0)
    tmp17 = tl.load(in_out_ptr0 + (r1 + (128*x0)), xmask, other=0.0)
    tmp42 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
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
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp24 = tl.where(xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(xmask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None]
    tmp35 = tmp18 - tmp28
    tmp36 = 128.0
    tmp37 = tmp34 / tmp36
    tmp38 = 1e-05
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.rsqrt(tmp39)
    tmp41 = tmp35 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp18, xmask)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp45, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/sn/csnhzft2h47hk3uucqivomzz64p2wx65m5xhhs7pt2sx3aeuskx5.py
# Topologically Sorted Source Nodes: [x_mid_d_2, h_diag_3, row_highway_5, index_add__20, col_highway_5, index_add__21, col_highway_6, index_add__25, row_highway_6, index_add__24], Original ATen: [aten.add, aten.new_zeros, aten.index_add]
# Source node to ATen node mapping:
#   col_highway_5 => full_default_25
#   col_highway_6 => full_default_27
#   h_diag_3 => add_55
#   index_add__20 => index_put_32
#   index_add__21 => index_put_33
#   index_add__24 => index_put_36
#   index_add__25 => index_put_37
#   row_highway_5 => full_default_24
#   row_highway_6 => full_default_26
#   x_mid_d_2 => add_51
# Graph fragment:
#   %add_51 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_35, %view_261), kwargs = {})
#   %add_55 : [num_users=7] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_51, %view_281), kwargs = {})
#   %full_default_24 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 128, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_32 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_24, [None, %arg42_1], %view_319, True), kwargs = {})
#   %full_default_25 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 128, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_33 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_25, [None, %arg42_1], %view_319, True), kwargs = {})
#   %full_default_27 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 128, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_37 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_27, [None, %arg42_1], %view_347, True), kwargs = {})
#   %full_default_26 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 128, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_36 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_26, [None, %arg42_1], %view_347, True), kwargs = {})
triton_poi_fused_add_index_add_new_zeros_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_index_add_new_zeros_52', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr0', 'out_ptr1', 'out_ptr2', 'out_ptr3'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x0 = xindex % 128
    x3 = (xindex // 16384)
    x2 = xindex % 16384
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
    tmp10 = tl.full([XBLOCK], 128, tl.int32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tl.device_assert((0 <= tmp13) & (tmp13 < 128), "index out of bounds: 0 <= tmp13 < 128")
    tl.store(in_out_ptr0 + (x4), tmp8, None)
    tl.atomic_add(out_ptr0 + (x2 + (16384*tmp13)), tmp8, None, sem='relaxed')
    tl.atomic_add(out_ptr1 + (x2 + (16384*tmp13)), tmp8, None, sem='relaxed')
    tl.atomic_add(out_ptr2 + (x2 + (16384*tmp13)), tmp8, None, sem='relaxed')
    tl.atomic_add(out_ptr3 + (x2 + (16384*tmp13)), tmp8, None, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/xq/cxqktolufcowzepnbicjwlx7ja4at5xs7yoxrrxvchyk2xsokln5.py
# Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_50 => add_68, erf_16, mul_76, mul_77, mul_78
# Graph fragment:
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_364, 0.5), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_364, 0.7071067811865476), kwargs = {})
#   %erf_16 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_77,), kwargs = {})
#   %add_68 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_16, 1), kwargs = {})
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, %add_68), kwargs = {})
triton_poi_fused_gelu_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_53', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1994752
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/7j/c7jiofenbtwry25nnwp2hq6qkqpmypnvuoyjb6bhy7vwz24udags.py
# Topologically Sorted Source Nodes: [sigmoid], Original ATen: [aten.sigmoid]
# Source node to ATen node mapping:
#   sigmoid => sigmoid
# Graph fragment:
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_377,), kwargs = {})
triton_poi_fused_sigmoid_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_54', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/2x/c2xodthij6rwpeqiz56462sj4ddddxpxxcjwfbqkyai6gkkub7je.py
# Topologically Sorted Source Nodes: [packed_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   packed_1 => cat_10
# Graph fragment:
#   %cat_10 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_9, %view_378], 1), kwargs = {})
triton_poi_fused_cat_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_55', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2079744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2071552, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 1048576, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.load(in_ptr0 + (x0), tmp7 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp0 >= tmp5
    tmp10 = tl.full([1], 1547264, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tmp12 & tmp4
    tmp14 = tl.load(in_ptr1 + ((-1048576) + x0), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp0 >= tmp10
    tmp16 = tl.full([1], 1809408, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr2 + ((-1547264) + x0), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp0 >= tmp16
    tmp22 = tmp21 & tmp4
    tmp23 = tl.load(in_ptr3 + ((-1809408) + x0), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp18, tmp20, tmp23)
    tmp25 = tl.where(tmp12, tmp14, tmp24)
    tmp26 = tl.where(tmp6, tmp8, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp4, tmp26, tmp27)
    tmp29 = tmp0 >= tmp3
    tmp30 = tl.full([1], 2079744, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = tl.load(in_ptr4 + ((-2071552) + x0), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.where(tmp4, tmp28, tmp32)
    tl.store(out_ptr0 + (x0), tmp33, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 8192, 9), (73728, 9, 1))
    assert_size_stride(arg1_1, (1, 12), (12, 1))
    assert_size_stride(arg2_1, (128, 18), (18, 1))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, 128), (128, 1))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (2, 40596), (40596, 1))
    assert_size_stride(arg7_1, (40596, ), (1, ))
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
    assert_size_stride(arg28_1, (64, 128, 128, 4), (65536, 512, 4, 1))
    assert_size_stride(arg29_1, (487, 128, 128, 4), (65536, 512, 4, 1))
    assert_size_stride(arg30_1, (487, 128), (128, 1))
    assert_size_stride(arg31_1, (487, 128), (128, 1))
    assert_size_stride(arg32_1, (128, ), (1, ))
    assert_size_stride(arg33_1, (128, ), (1, ))
    assert_size_stride(arg34_1, (384, 128), (128, 1))
    assert_size_stride(arg35_1, (384, ), (1, ))
    assert_size_stride(arg36_1, (16, 4), (4, 1))
    assert_size_stride(arg37_1, (16, ), (1, ))
    assert_size_stride(arg38_1, (8, 16), (16, 1))
    assert_size_stride(arg39_1, (8, ), (1, ))
    assert_size_stride(arg40_1, (128, 128), (128, 1))
    assert_size_stride(arg41_1, (128, ), (1, ))
    assert_size_stride(arg42_1, (64, ), (1, ))
    assert_size_stride(arg43_1, (128, ), (1, ))
    assert_size_stride(arg44_1, (128, ), (1, ))
    assert_size_stride(arg45_1, (512, 512), (512, 1))
    assert_size_stride(arg46_1, (512, ), (1, ))
    assert_size_stride(arg47_1, (128, 512), (512, 1))
    assert_size_stride(arg48_1, (128, ), (1, ))
    assert_size_stride(arg49_1, (128, ), (1, ))
    assert_size_stride(arg50_1, (128, ), (1, ))
    assert_size_stride(arg51_1, (384, 128), (128, 1))
    assert_size_stride(arg52_1, (384, ), (1, ))
    assert_size_stride(arg53_1, (16, 4), (4, 1))
    assert_size_stride(arg54_1, (16, ), (1, ))
    assert_size_stride(arg55_1, (8, 16), (16, 1))
    assert_size_stride(arg56_1, (8, ), (1, ))
    assert_size_stride(arg57_1, (128, 128), (128, 1))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (1090, ), (1, ))
    assert_size_stride(arg60_1, (1090, ), (1, ))
    assert_size_stride(arg61_1, (1090, ), (1, ))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (512, 512), (512, 1))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (128, 512), (512, 1))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (128, ), (1, ))
    assert_size_stride(arg69_1, (128, ), (1, ))
    assert_size_stride(arg70_1, (384, 128), (128, 1))
    assert_size_stride(arg71_1, (384, ), (1, ))
    assert_size_stride(arg72_1, (16, 4), (4, 1))
    assert_size_stride(arg73_1, (16, ), (1, ))
    assert_size_stride(arg74_1, (8, 16), (16, 1))
    assert_size_stride(arg75_1, (8, ), (1, ))
    assert_size_stride(arg76_1, (128, 128), (128, 1))
    assert_size_stride(arg77_1, (128, ), (1, ))
    assert_size_stride(arg78_1, (128, ), (1, ))
    assert_size_stride(arg79_1, (128, ), (1, ))
    assert_size_stride(arg80_1, (512, 512), (512, 1))
    assert_size_stride(arg81_1, (512, ), (1, ))
    assert_size_stride(arg82_1, (128, 512), (512, 1))
    assert_size_stride(arg83_1, (128, ), (1, ))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (128, ), (1, ))
    assert_size_stride(arg86_1, (384, 128), (128, 1))
    assert_size_stride(arg87_1, (384, ), (1, ))
    assert_size_stride(arg88_1, (16, 4), (4, 1))
    assert_size_stride(arg89_1, (16, ), (1, ))
    assert_size_stride(arg90_1, (8, 16), (16, 1))
    assert_size_stride(arg91_1, (8, ), (1, ))
    assert_size_stride(arg92_1, (128, 128), (128, 1))
    assert_size_stride(arg93_1, (128, ), (1, ))
    assert_size_stride(arg94_1, (128, ), (1, ))
    assert_size_stride(arg95_1, (128, ), (1, ))
    assert_size_stride(arg96_1, (512, 512), (512, 1))
    assert_size_stride(arg97_1, (512, ), (1, ))
    assert_size_stride(arg98_1, (128, 512), (512, 1))
    assert_size_stride(arg99_1, (128, ), (1, ))
    assert_size_stride(arg100_1, (128, ), (1, ))
    assert_size_stride(arg101_1, (128, ), (1, ))
    assert_size_stride(arg102_1, (384, 128), (128, 1))
    assert_size_stride(arg103_1, (384, ), (1, ))
    assert_size_stride(arg104_1, (16, 4), (4, 1))
    assert_size_stride(arg105_1, (16, ), (1, ))
    assert_size_stride(arg106_1, (8, 16), (16, 1))
    assert_size_stride(arg107_1, (8, ), (1, ))
    assert_size_stride(arg108_1, (128, 128), (128, 1))
    assert_size_stride(arg109_1, (128, ), (1, ))
    assert_size_stride(arg110_1, (128, ), (1, ))
    assert_size_stride(arg111_1, (128, ), (1, ))
    assert_size_stride(arg112_1, (512, 512), (512, 1))
    assert_size_stride(arg113_1, (512, ), (1, ))
    assert_size_stride(arg114_1, (128, 512), (512, 1))
    assert_size_stride(arg115_1, (128, ), (1, ))
    assert_size_stride(arg116_1, (128, ), (1, ))
    assert_size_stride(arg117_1, (128, ), (1, ))
    assert_size_stride(arg118_1, (384, 128), (128, 1))
    assert_size_stride(arg119_1, (384, ), (1, ))
    assert_size_stride(arg120_1, (16, 4), (4, 1))
    assert_size_stride(arg121_1, (16, ), (1, ))
    assert_size_stride(arg122_1, (8, 16), (16, 1))
    assert_size_stride(arg123_1, (8, ), (1, ))
    assert_size_stride(arg124_1, (128, 128), (128, 1))
    assert_size_stride(arg125_1, (128, ), (1, ))
    assert_size_stride(arg126_1, (128, ), (1, ))
    assert_size_stride(arg127_1, (128, ), (1, ))
    assert_size_stride(arg128_1, (512, 512), (512, 1))
    assert_size_stride(arg129_1, (512, ), (1, ))
    assert_size_stride(arg130_1, (128, 512), (512, 1))
    assert_size_stride(arg131_1, (128, ), (1, ))
    assert_size_stride(arg132_1, (128, 128), (128, 1))
    assert_size_stride(arg133_1, (128, ), (1, ))
    assert_size_stride(arg134_1, (128, 128), (128, 1))
    assert_size_stride(arg135_1, (128, ), (1, ))
    assert_size_stride(arg136_1, (128, 128), (128, 1))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (32, 128), (128, 1))
    assert_size_stride(arg139_1, (32, ), (1, ))
    assert_size_stride(arg140_1, (128, 128), (128, 1))
    assert_size_stride(arg141_1, (128, ), (1, ))
    assert_size_stride(arg142_1, (32, 128), (128, 1))
    assert_size_stride(arg143_1, (32, ), (1, ))
    assert_size_stride(arg144_1, (1, 128), (128, 1))
    assert_size_stride(arg145_1, (1, ), (1, ))
    assert_size_stride(arg146_1, (32, 128), (128, 1))
    assert_size_stride(arg147_1, (32, ), (1, ))
    assert_size_stride(arg148_1, (32, 128), (128, 1))
    assert_size_stride(arg149_1, (32, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8192, 20), (20, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(arg0_1, arg1_1, buf0, 163840, grid=grid(163840), stream=stream0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((20, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(arg2_1, buf1, 2560, grid=grid(2560), stream=stream0)
        del arg2_1
        buf2 = empty_strided_cuda((8192, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf0, buf1, out=buf2)
        del buf0
        del buf1
        buf3 = reinterpret_tensor(buf2, (1, 8192, 128), (1048576, 128, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf3, arg3_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg3_1
        buf4 = empty_strided_cuda((8192, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg5_1, reinterpret_tensor(buf3, (8192, 128), (128, 1), 0), reinterpret_tensor(arg4_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf4)
        del arg4_1
        del arg5_1
        buf9 = empty_strided_cuda((8192, 256), (256, 1), torch.float32)
        buf5 = reinterpret_tensor(buf9, (8192, 128), (256, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [self_features], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, buf4, reinterpret_tensor(arg10_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf5)
        del arg10_1
        del arg11_1
        buf6 = reinterpret_tensor(buf3, (8192, 128), (128, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [neighbor_features], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg9_1, buf4, reinterpret_tensor(arg8_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf6)
        del arg8_1
        del arg9_1
        buf7 = reinterpret_tensor(buf9, (8192, 128), (256, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [aggr], Original ATen: [aten.zeros_like]
        triton_poi_fused_zeros_like_3.run(buf7, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [aggr, getitem_4, messages, index_add_], Original ATen: [aten.zeros_like, aten.index, aten.mul, aten.index_add]
        triton_poi_fused_index_index_add_mul_zeros_like_4.run(arg6_1, buf6, arg7_1, buf7, 5196288, grid=grid(5196288), stream=stream0)
        del buf5
        del buf7
        buf10 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf9, reinterpret_tensor(arg12_1, (256, 128), (1, 256), 0), out=buf10)
        del arg12_1
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf11, arg13_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg13_1
        buf12 = empty_strided_cuda((8192, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.gelu]
        extern_kernels.mm(buf11, reinterpret_tensor(arg14_1, (128, 128), (1, 128), 0), out=buf12)
        del arg14_1
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.add]
        triton_poi_fused_add_5.run(buf13, buf4, arg15_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg15_1
        buf18 = buf9; del buf9  # reuse
        buf14 = reinterpret_tensor(buf18, (8192, 128), (256, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [self_features_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg19_1, buf13, reinterpret_tensor(arg18_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf14)
        del arg18_1
        del arg19_1
        buf15 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [neighbor_features_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg17_1, buf13, reinterpret_tensor(arg16_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf15)
        del arg16_1
        del arg17_1
        buf16 = reinterpret_tensor(buf18, (8192, 128), (256, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [aggr_1], Original ATen: [aten.zeros_like]
        triton_poi_fused_zeros_like_3.run(buf16, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [aggr_1, getitem_7, messages_1, index_add__1], Original ATen: [aten.zeros_like, aten.index, aten.mul, aten.index_add]
        triton_poi_fused_index_index_add_mul_zeros_like_4.run(arg6_1, buf15, arg7_1, buf16, 5196288, grid=grid(5196288), stream=stream0)
        del arg6_1
        del arg7_1
        del buf14
        del buf16
        buf19 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf18, reinterpret_tensor(arg20_1, (256, 128), (1, 256), 0), out=buf19)
        del arg20_1
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf20, arg21_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg21_1
        buf21 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.gelu]
        extern_kernels.mm(buf20, reinterpret_tensor(arg22_1, (128, 128), (1, 128), 0), out=buf21)
        del arg22_1
        buf25 = reinterpret_tensor(buf20, (1, 8192, 128), (1048576, 128, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [h_2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_6.run(buf13, buf21, arg23_1, arg24_1, arg25_1, buf25, 8192, 128, grid=grid(8192), stream=stream0)
        del arg23_1
        del arg24_1
        del arg25_1
        buf26 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg27_1, reinterpret_tensor(buf25, (8192, 128), (128, 1), 0), reinterpret_tensor(arg26_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf26)
        del arg26_1
        del arg27_1
        buf30 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_7.run(buf26, arg32_1, arg33_1, buf30, 8192, 128, grid=grid(8192), stream=stream0)
        del arg32_1
        del arg33_1
        buf31 = empty_strided_cuda((8192, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (8192, 128), (128, 1), 0), reinterpret_tensor(arg34_1, (128, 384), (1, 128), 0), out=buf31)
        del arg34_1
        buf32 = empty_strided_cuda((1, 64, 128, 128, 4), (4194304, 65536, 512, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_8.run(arg28_1, buf32, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_9.run(buf32, 32768, grid=grid(32768), stream=stream0)
        buf34 = reinterpret_tensor(buf30, (1, 64, 128, 128), (1048576, 16384, 128, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_10.run(buf32, buf34, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_11.run(buf34, 8192, grid=grid(8192), stream=stream0)
        buf36 = reinterpret_tensor(buf13, (64, 8, 128, 16), (16384, 2048, 16, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [q_f, k_f, v_f, logit_bias, x_soft], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_12.run(buf31, arg35_1, buf36, 1048576, grid=grid(1048576), stream=stream0)
        buf37 = empty_strided_cuda((64, 8, 128, 16), (16384, 2048, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f, k_f, v_f, logit_bias, x_soft], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_13.run(buf31, arg35_1, buf37, 1048576, grid=grid(1048576), stream=stream0)
        buf38 = empty_strided_cuda((64, 8, 128, 16), (16384, 2048, 16, 1), torch.float32)
        buf50 = empty_strided_cuda((64, 8, 128, 1, 16, 1), (16384, 2048, 16, 16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f, k_f, v_f, logit_bias, x_soft, einsum_2], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_14.run(buf31, arg35_1, buf38, buf50, 1048576, grid=grid(1048576), stream=stream0)
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
        buf51 = reinterpret_tensor(buf34, (512, 128, 16), (2048, 16, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf49, (512, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf50, (512, 128, 16), (2048, 16, 1), 0), out=buf51)
        buf52 = reinterpret_tensor(buf41, (1, 64, 128, 8, 16), (1048576, 16384, 128, 16, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_mid], Original ATen: [aten.add]
        triton_poi_fused_add_19.run(buf52, buf51, 1048576, grid=grid(1048576), stream=stream0)
        buf53 = reinterpret_tensor(buf51, (8192, 128), (128, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf52, (8192, 128), (128, 1), 0), reinterpret_tensor(arg40_1, (128, 128), (1, 128), 0), out=buf53)
        del arg40_1
        buf57 = reinterpret_tensor(buf18, (1, 128, 128, 128), (2097152, 16384, 128, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [row_highway], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf57, 2097152, grid=grid(2097152), stream=stream0)
        buf59 = empty_strided_cuda((1, 128, 128, 128), (2097152, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [col_highway], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf59, 2097152, grid=grid(2097152), stream=stream0)
        buf67 = reinterpret_tensor(buf45, (1, 8192, 512), (4194304, 512, 1), 0); del buf45  # reuse
        buf63 = reinterpret_tensor(buf67, (1, 8192, 128), (4194304, 512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_mid_d, z, row_highway, index_add__2, col_highway, index_add__3], Original ATen: [aten.add, aten.native_layer_norm, aten.new_zeros, aten.index_add]
        triton_per_fused_add_index_add_native_layer_norm_new_zeros_21.run(buf26, buf53, arg41_1, arg42_1, arg43_1, arg44_1, buf57, buf59, buf63, 8192, 128, grid=grid(8192), stream=stream0)
        del arg43_1
        del arg44_1
        buf61 = empty_strided_cuda((1, 1, 128, 64), (8192, 8192, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_mid_d, g], Original ATen: [aten.add, aten.mean]
        triton_red_fused_add_mean_22.run(buf26, buf53, arg41_1, buf61, 8192, 128, grid=grid(8192), stream=stream0)
        buf62 = empty_strided_cuda((1, 1, 128), (128, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_mid_d, g], Original ATen: [aten.add, aten.mean]
        triton_per_fused_add_mean_23.run(buf61, buf62, 128, 64, grid=grid(128), stream=stream0)
        buf64 = reinterpret_tensor(buf67, (1, 8192, 128), (4194304, 512, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [z_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_24.run(buf57, buf64, 1048576, grid=grid(1048576), stream=stream0)
        buf65 = reinterpret_tensor(buf67, (1, 8192, 128), (4194304, 512, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [z_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_24.run(buf59, buf65, 1048576, grid=grid(1048576), stream=stream0)
        buf66 = reinterpret_tensor(buf67, (1, 8192, 128), (4194304, 512, 1), 384)  # alias
        # Topologically Sorted Source Nodes: [z_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_25.run(buf62, buf66, 1048576, grid=grid(1048576), stream=stream0)
        del buf63
        del buf64
        del buf65
        del buf66
        buf68 = reinterpret_tensor(buf32, (8192, 512), (512, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf67, (8192, 512), (512, 1), 0), reinterpret_tensor(arg45_1, (512, 512), (1, 512), 0), out=buf68)
        del arg45_1
        buf69 = reinterpret_tensor(buf68, (1, 8192, 512), (4194304, 512, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_26.run(buf69, arg46_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg46_1
        buf70 = reinterpret_tensor(buf52, (8192, 128), (128, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf69, (8192, 512), (512, 1), 0), reinterpret_tensor(arg47_1, (512, 128), (1, 512), 0), out=buf70)
        del arg47_1
        buf71 = reinterpret_tensor(buf70, (1, 8192, 128), (1048576, 128, 1), 0); del buf70  # reuse
        buf110 = reinterpret_tensor(buf50, (1, 8192, 128), (1048576, 128, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d, h_diag_1, layer_norm_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf71, buf26, buf53, arg41_1, arg48_1, arg68_1, arg69_1, buf110, 8192, 128, grid=grid(8192), stream=stream0)
        del arg41_1
        del arg48_1
        del arg68_1
        del arg69_1
        buf72 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [h_k_3], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_28.run(buf71, buf72, 2097152, grid=grid(2097152), stream=stream0)
        buf73 = empty_strided_cuda((1, 487, 16384), (7979008, 16384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [row_p_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg30_1, (1, 487, 128), (62336, 128, 1), 0), reinterpret_tensor(buf72, (1, 128, 16384), (0, 16384, 1), 0), out=buf73)
        buf74 = empty_strided_cuda((1, 487, 16384), (7979008, 16384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [col_p_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg31_1, (1, 487, 128), (62336, 128, 1), 0), reinterpret_tensor(buf72, (1, 128, 16384), (2097152, 16384, 1), 0), out=buf74)
        buf75 = empty_strided_cuda((487, 32, 128), (4096, 128, 1), torch.float32)
        buf79 = empty_strided_cuda((487, 32, 128), (4096, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_off_3, layer_norm_3], Original ATen: [aten.mean, aten.native_layer_norm]
        triton_per_fused_mean_native_layer_norm_29.run(buf73, buf74, arg49_1, arg50_1, buf75, buf79, 15584, 128, grid=grid(15584), stream=stream0)
        del arg49_1
        del arg50_1
        buf80 = empty_strided_cuda((15584, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf79, (15584, 128), (128, 1), 0), reinterpret_tensor(arg51_1, (128, 384), (1, 128), 0), out=buf80)
        del arg51_1
        buf82 = reinterpret_tensor(buf79, (487, 1, 32, 32, 4), (4096, 1994752, 128, 4, 1), 0); del buf79  # reuse
        buf183 = empty_strided_cuda((487, 1, 32, 32, 4), (4096, 1994752, 128, 4, 1), torch.float32)
        buf284 = empty_strided_cuda((487, 32, 32, 4), (4096, 128, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sub, setitem_2, setitem_6], Original ATen: [aten.mean, aten.lift_fresh, aten.index_put]
        triton_per_fused_index_put_lift_fresh_mean_30.run(arg29_1, buf82, buf183, buf284, 1994752, 16, grid=grid(1994752), stream=stream0)
        del arg29_1
        # Topologically Sorted Source Nodes: [setitem_2], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_31.run(buf82, 62336, grid=grid(62336), stream=stream0)
        buf84 = empty_strided_cuda((487, 1, 32, 32), (1024, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_32.run(buf82, buf84, 498688, grid=grid(498688), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_33.run(buf84, 15584, grid=grid(15584), stream=stream0)
        buf86 = empty_strided_cuda((487, 8, 32, 16), (4096, 512, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, logit_bias_1, x_soft_2], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_34.run(buf80, arg52_1, buf86, 1994752, grid=grid(1994752), stream=stream0)
        buf87 = empty_strided_cuda((487, 8, 32, 16), (4096, 512, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, logit_bias_1, x_soft_2], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_35.run(buf80, arg52_1, buf87, 1994752, grid=grid(1994752), stream=stream0)
        buf88 = empty_strided_cuda((487, 8, 32, 16), (4096, 512, 16, 1), torch.float32)
        buf100 = empty_strided_cuda((487, 8, 32, 1, 16, 1), (4096, 512, 16, 16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, logit_bias_1, x_soft_2, einsum_5], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_36.run(buf80, arg52_1, buf88, buf100, 1994752, grid=grid(1994752), stream=stream0)
        del arg52_1
        buf89 = empty_strided_cuda((487, 8, 32, 32), (8192, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, logit_bias_1, x_soft_2], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_37.run(buf84, buf82, buf89, 3989504, grid=grid(3989504), stream=stream0)
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, logit_bias_1, x_soft_2], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf90 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf86, buf87, buf88, buf89, False)
        del buf86
        del buf87
        buf91 = buf90[0]
        del buf90
        buf95 = reinterpret_tensor(buf88, (487, 1, 32, 32, 4), (4096, 1, 128, 4, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_38.run(buf84, buf82, buf95, 1994752, grid=grid(1994752), stream=stream0)
        buf96 = reinterpret_tensor(buf74, (498688, 16), (16, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (498688, 4), (4, 1), 0), reinterpret_tensor(arg53_1, (4, 16), (1, 4), 0), out=buf96)
        del arg53_1
        buf97 = reinterpret_tensor(buf96, (487, 1, 32, 32, 16), (16384, 1, 512, 16, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_39.run(buf97, arg54_1, 7979008, grid=grid(7979008), stream=stream0)
        del arg54_1
        buf98 = reinterpret_tensor(buf89, (498688, 8), (8, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (498688, 16), (16, 1), 0), reinterpret_tensor(arg55_1, (16, 8), (1, 16), 0), out=buf98)
        del arg55_1
        buf99 = empty_strided_cuda((487, 8, 32, 32, 1, 1), (8192, 1024, 32, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf98, arg56_1, buf99, 3896, 1024, grid=grid(3896, 1024), stream=stream0)
        del arg56_1
        buf101 = reinterpret_tensor(buf95, (3896, 32, 16), (512, 16, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [einsum_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf99, (3896, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf100, (3896, 32, 16), (512, 16, 1), 0), out=buf101)
        buf102 = reinterpret_tensor(buf91, (487, 1, 32, 8, 16), (4096, 1, 128, 16, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [x_mid_1], Original ATen: [aten.add]
        triton_poi_fused_add_41.run(buf102, buf101, 1994752, grid=grid(1994752), stream=stream0)
        buf103 = reinterpret_tensor(buf101, (15584, 128), (128, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf102, (15584, 128), (128, 1), 0), reinterpret_tensor(arg57_1, (128, 128), (1, 128), 0), out=buf103)
        del arg57_1
        buf154 = reinterpret_tensor(buf97, (487, 32, 512), (16384, 512, 1), 0); del buf97  # reuse
        buf150 = reinterpret_tensor(buf154, (487, 32, 128), (16384, 512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_mid_o, z_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_42.run(buf75, buf103, arg58_1, arg62_1, arg63_1, buf150, 15584, 128, grid=grid(15584), stream=stream0)
        del arg62_1
        del arg63_1
        buf111 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf110, (8192, 128), (128, 1), 0), reinterpret_tensor(arg70_1, (128, 384), (1, 128), 0), out=buf111)
        del arg70_1
        buf112 = reinterpret_tensor(buf69, (1, 64, 128, 128, 4), (4194304, 65536, 512, 4, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [setitem_4], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_8.run(arg28_1, buf112, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_4], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_9.run(buf112, 32768, grid=grid(32768), stream=stream0)
        buf114 = reinterpret_tensor(buf110, (1, 64, 128, 128), (1048576, 16384, 128, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [setitem_5], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_10.run(buf112, buf114, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_5], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_11.run(buf114, 8192, grid=grid(8192), stream=stream0)
        buf116 = reinterpret_tensor(buf53, (64, 8, 128, 16), (16384, 2048, 16, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [q_f_2, k_f_2, v_f_2, logit_bias_2, x_soft_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_12.run(buf111, arg71_1, buf116, 1048576, grid=grid(1048576), stream=stream0)
        buf117 = reinterpret_tensor(buf26, (64, 8, 128, 16), (16384, 2048, 16, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [q_f_2, k_f_2, v_f_2, logit_bias_2, x_soft_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_13.run(buf111, arg71_1, buf117, 1048576, grid=grid(1048576), stream=stream0)
        buf118 = buf38; del buf38  # reuse
        buf130 = reinterpret_tensor(buf37, (64, 8, 128, 1, 16, 1), (16384, 2048, 16, 16, 1, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [q_f_2, k_f_2, v_f_2, logit_bias_2, x_soft_4, einsum_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_14.run(buf111, arg71_1, buf118, buf130, 1048576, grid=grid(1048576), stream=stream0)
        del arg71_1
        buf119 = reinterpret_tensor(buf49, (64, 8, 128, 128), (131072, 16384, 128, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [q_f_2, k_f_2, v_f_2, logit_bias_2, x_soft_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_15.run(buf114, buf112, buf119, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [q_f_2, k_f_2, v_f_2, logit_bias_2, x_soft_4], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf120 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf116, buf117, buf118, buf119, False)
        del buf116
        buf121 = buf120[0]
        del buf120
        buf125 = reinterpret_tensor(buf67, (1, 64, 128, 128, 4), (4194304, 65536, 512, 4, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(buf114, buf112, buf125, 4194304, grid=grid(4194304), stream=stream0)
        buf126 = reinterpret_tensor(buf47, (1048576, 16), (16, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf125, (1048576, 4), (4, 1), 0), reinterpret_tensor(arg72_1, (4, 16), (1, 4), 0), out=buf126)
        del arg72_1
        buf127 = reinterpret_tensor(buf126, (1, 64, 128, 128, 16), (16777216, 262144, 2048, 16, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_17.run(buf127, arg73_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg73_1
        buf128 = reinterpret_tensor(buf119, (1048576, 8), (8, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf127, (1048576, 16), (16, 1), 0), reinterpret_tensor(arg74_1, (16, 8), (1, 16), 0), out=buf128)
        del arg74_1
        buf129 = reinterpret_tensor(buf48, (64, 8, 128, 128, 1, 1), (131072, 16384, 128, 1, 1, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [einsum_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf128, arg75_1, buf129, 512, 16384, grid=grid(512, 16384), stream=stream0)
        del arg75_1
        buf131 = reinterpret_tensor(buf114, (512, 128, 16), (2048, 16, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [einsum_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf129, (512, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf130, (512, 128, 16), (2048, 16, 1), 0), out=buf131)
        buf132 = reinterpret_tensor(buf121, (1, 64, 128, 8, 16), (1048576, 16384, 128, 16, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [x_mid_2], Original ATen: [aten.add]
        triton_poi_fused_add_19.run(buf132, buf131, 1048576, grid=grid(1048576), stream=stream0)
        buf133 = reinterpret_tensor(buf131, (8192, 128), (128, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf132, (8192, 128), (128, 1), 0), reinterpret_tensor(arg76_1, (128, 128), (1, 128), 0), out=buf133)
        del arg76_1
        buf137 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [row_highway_2], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf137, 2097152, grid=grid(2097152), stream=stream0)
        buf139 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [row_highway_1], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf139, 2097152, grid=grid(2097152), stream=stream0)
        buf144 = empty_strided_cuda((1, 128, 128, 128), (2097152, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [col_highway_1], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf144, 2097152, grid=grid(2097152), stream=stream0)
        buf160 = empty_strided_cuda((1, 128, 128, 128), (2097152, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [col_highway_2], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf160, 2097152, grid=grid(2097152), stream=stream0)
        buf169 = reinterpret_tensor(buf125, (1, 8192, 512), (4194304, 512, 1), 0); del buf125  # reuse
        buf165 = reinterpret_tensor(buf169, (1, 8192, 128), (4194304, 512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_mid_d_1, z_4, row_highway_2, index_add__8, row_highway_1, index_add__4, col_highway_1, index_add__5, col_highway_2, index_add__9], Original ATen: [aten.add, aten.native_layer_norm, aten.new_zeros, aten.index_add]
        triton_per_fused_add_index_add_native_layer_norm_new_zeros_43.run(buf71, buf133, arg77_1, arg42_1, arg78_1, arg79_1, buf137, buf160, buf165, buf139, buf144, 8192, 128, grid=grid(8192), stream=stream0)
        del arg78_1
        del arg79_1
        # Topologically Sorted Source Nodes: [toks_1, index_add__6, index_add__7], Original ATen: [aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_44.run(arg60_1, arg59_1, buf75, buf103, arg58_1, arg61_1, buf139, buf144, 17858560, grid=grid(17858560), stream=stream0)
        buf142 = empty_strided_cuda((1, 128, 128, 128), (2097152, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [row_k_2], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_28.run(buf139, buf142, 2097152, grid=grid(2097152), stream=stream0)
        buf143 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [row_t], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg30_1, (1, 487, 128), (62336, 128, 1), 0), reinterpret_tensor(buf142, (1, 128, 16384), (0, 16384, 1), 0), out=buf143)
        buf147 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [col_k_2], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_28.run(buf144, buf147, 2097152, grid=grid(2097152), stream=stream0)
        buf148 = empty_strided_cuda((1, 487, 16384), (7979008, 16384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [col_t], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg31_1, (1, 487, 128), (62336, 128, 1), 0), reinterpret_tensor(buf147, (1, 128, 16384), (0, 16384, 1), 0), out=buf148)
        buf149 = empty_strided_cuda((487, 1, 128), (128, 62336, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_mid_o, g_1], Original ATen: [aten.add, aten.mean]
        triton_per_fused_add_mean_45.run(buf75, buf103, arg58_1, buf149, 62336, 32, grid=grid(62336), stream=stream0)
        buf151 = reinterpret_tensor(buf154, (487, 32, 128), (16384, 512, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [z_3], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf143, buf151, 1994752, grid=grid(1994752), stream=stream0)
        buf152 = reinterpret_tensor(buf154, (487, 32, 128), (16384, 512, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [z_3], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf148, buf152, 1994752, grid=grid(1994752), stream=stream0)
        buf153 = reinterpret_tensor(buf154, (487, 32, 128), (16384, 512, 1), 384)  # alias
        # Topologically Sorted Source Nodes: [z_3], Original ATen: [aten.cat]
        triton_poi_fused_cat_47.run(buf149, buf153, 1994752, grid=grid(1994752), stream=stream0)
        del buf150
        del buf151
        del buf152
        del buf153
        buf155 = reinterpret_tensor(buf148, (15584, 512), (512, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf154, (15584, 512), (512, 1), 0), reinterpret_tensor(arg64_1, (512, 512), (1, 512), 0), out=buf155)
        del arg64_1
        buf156 = reinterpret_tensor(buf155, (487, 32, 512), (16384, 512, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf156, arg65_1, 7979008, grid=grid(7979008), stream=stream0)
        del arg65_1
        buf157 = reinterpret_tensor(buf102, (15584, 128), (128, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf156, (15584, 512), (512, 1), 0), reinterpret_tensor(arg66_1, (512, 128), (1, 512), 0), out=buf157)
        del arg66_1
        buf158 = reinterpret_tensor(buf157, (487, 32, 128), (4096, 128, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [x_mid_o, off_stream], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(buf158, buf75, buf103, arg58_1, arg67_1, 1994752, grid=grid(1994752), stream=stream0)
        del arg58_1
        del arg67_1
        # Topologically Sorted Source Nodes: [toks_3, index_add__10, index_add__11], Original ATen: [aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_50.run(arg60_1, arg59_1, buf158, arg61_1, buf137, buf160, 17858560, grid=grid(17858560), stream=stream0)
        buf163 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d_1, g_2], Original ATen: [aten.add, aten.mean]
        triton_red_fused_add_mean_22.run(buf71, buf133, arg77_1, buf163, 8192, 128, grid=grid(8192), stream=stream0)
        buf164 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d_1, g_2], Original ATen: [aten.add, aten.mean]
        triton_per_fused_add_mean_23.run(buf163, buf164, 128, 64, grid=grid(128), stream=stream0)
        buf166 = reinterpret_tensor(buf169, (1, 8192, 128), (4194304, 512, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [z_5], Original ATen: [aten.cat]
        triton_poi_fused_cat_24.run(buf137, buf166, 1048576, grid=grid(1048576), stream=stream0)
        buf167 = reinterpret_tensor(buf169, (1, 8192, 128), (4194304, 512, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [z_5], Original ATen: [aten.cat]
        triton_poi_fused_cat_24.run(buf160, buf167, 1048576, grid=grid(1048576), stream=stream0)
        buf168 = reinterpret_tensor(buf169, (1, 8192, 128), (4194304, 512, 1), 384)  # alias
        # Topologically Sorted Source Nodes: [z_5], Original ATen: [aten.cat]
        triton_poi_fused_cat_25.run(buf164, buf168, 1048576, grid=grid(1048576), stream=stream0)
        del buf165
        del buf166
        del buf167
        del buf168
        buf170 = reinterpret_tensor(buf112, (8192, 512), (512, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (8192, 512), (512, 1), 0), reinterpret_tensor(arg80_1, (512, 512), (1, 512), 0), out=buf170)
        del arg80_1
        buf171 = reinterpret_tensor(buf170, (1, 8192, 512), (4194304, 512, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_26.run(buf171, arg81_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg81_1
        buf172 = reinterpret_tensor(buf132, (8192, 128), (128, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (8192, 512), (512, 1), 0), reinterpret_tensor(arg82_1, (512, 128), (1, 512), 0), out=buf172)
        del arg82_1
        buf173 = reinterpret_tensor(buf172, (1, 8192, 128), (1048576, 128, 1), 0); del buf172  # reuse
        buf211 = reinterpret_tensor(buf130, (1, 8192, 128), (1048576, 128, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d_1, h_diag_2, layer_norm_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf173, buf71, buf133, arg77_1, arg83_1, arg100_1, arg101_1, buf211, 8192, 128, grid=grid(8192), stream=stream0)
        del arg100_1
        del arg101_1
        del arg77_1
        del arg83_1
        buf174 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [h_k_5], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_28.run(buf173, buf174, 2097152, grid=grid(2097152), stream=stream0)
        buf175 = reinterpret_tensor(buf156, (1, 487, 16384), (7979008, 16384, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [row_p_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg30_1, (1, 487, 128), (62336, 128, 1), 0), reinterpret_tensor(buf174, (1, 128, 16384), (0, 16384, 1), 0), out=buf175)
        buf176 = reinterpret_tensor(buf154, (1, 487, 16384), (7979008, 16384, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [col_p_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg31_1, (1, 487, 128), (62336, 128, 1), 0), reinterpret_tensor(buf174, (1, 128, 16384), (2097152, 16384, 1), 0), out=buf176)
        buf177 = buf158; del buf158  # reuse
        buf181 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [h_off_5, off_in, layer_norm_7], Original ATen: [aten.mean, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mean_native_layer_norm_51.run(buf177, buf175, buf176, arg84_1, arg85_1, buf181, 15584, 128, grid=grid(15584), stream=stream0)
        del arg84_1
        del arg85_1
        buf182 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf181, (15584, 128), (128, 1), 0), reinterpret_tensor(arg86_1, (128, 384), (1, 128), 0), out=buf182)
        del arg86_1
        # Topologically Sorted Source Nodes: [setitem_6], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_31.run(buf183, 62336, grid=grid(62336), stream=stream0)
        buf185 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [setitem_7], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_32.run(buf183, buf185, 498688, grid=grid(498688), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_7], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_33.run(buf185, 15584, grid=grid(15584), stream=stream0)
        buf187 = reinterpret_tensor(buf181, (487, 8, 32, 16), (4096, 512, 16, 1), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [q_f_3, k_f_3, v_f_3, logit_bias_3, x_soft_6], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_34.run(buf182, arg87_1, buf187, 1994752, grid=grid(1994752), stream=stream0)
        buf188 = reinterpret_tensor(buf103, (487, 8, 32, 16), (4096, 512, 16, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [q_f_3, k_f_3, v_f_3, logit_bias_3, x_soft_6], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_35.run(buf182, arg87_1, buf188, 1994752, grid=grid(1994752), stream=stream0)
        buf189 = reinterpret_tensor(buf100, (487, 8, 32, 16), (4096, 512, 16, 1), 0); del buf100  # reuse
        buf201 = reinterpret_tensor(buf82, (487, 8, 32, 1, 16, 1), (4096, 512, 16, 16, 1, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [q_f_3, k_f_3, v_f_3, logit_bias_3, x_soft_6, einsum_11], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_36.run(buf182, arg87_1, buf189, buf201, 1994752, grid=grid(1994752), stream=stream0)
        del arg87_1
        buf190 = reinterpret_tensor(buf99, (487, 8, 32, 32), (8192, 1024, 32, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [q_f_3, k_f_3, v_f_3, logit_bias_3, x_soft_6], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_37.run(buf185, buf183, buf190, 3989504, grid=grid(3989504), stream=stream0)
        # Topologically Sorted Source Nodes: [q_f_3, k_f_3, v_f_3, logit_bias_3, x_soft_6], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf191 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf187, buf188, buf189, buf190, False)
        del buf187
        del buf188
        buf192 = buf191[0]
        del buf191
        buf196 = reinterpret_tensor(buf189, (487, 1, 32, 32, 4), (4096, 1, 128, 4, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_38.run(buf185, buf183, buf196, 1994752, grid=grid(1994752), stream=stream0)
        buf197 = reinterpret_tensor(buf176, (498688, 16), (16, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (498688, 4), (4, 1), 0), reinterpret_tensor(arg88_1, (4, 16), (1, 4), 0), out=buf197)
        del arg88_1
        buf198 = reinterpret_tensor(buf197, (487, 1, 32, 32, 16), (16384, 1, 512, 16, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_39.run(buf198, arg89_1, 7979008, grid=grid(7979008), stream=stream0)
        del arg89_1
        buf199 = reinterpret_tensor(buf190, (498688, 8), (8, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (498688, 16), (16, 1), 0), reinterpret_tensor(arg90_1, (16, 8), (1, 16), 0), out=buf199)
        del arg90_1
        buf200 = reinterpret_tensor(buf98, (487, 8, 32, 32, 1, 1), (8192, 1024, 32, 1, 1, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [einsum_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf199, arg91_1, buf200, 3896, 1024, grid=grid(3896, 1024), stream=stream0)
        del arg91_1
        buf202 = reinterpret_tensor(buf196, (3896, 32, 16), (512, 16, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [einsum_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf200, (3896, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf201, (3896, 32, 16), (512, 16, 1), 0), out=buf202)
        buf203 = reinterpret_tensor(buf192, (487, 1, 32, 8, 16), (4096, 1, 128, 16, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [x_mid_3], Original ATen: [aten.add]
        triton_poi_fused_add_41.run(buf203, buf202, 1994752, grid=grid(1994752), stream=stream0)
        buf204 = reinterpret_tensor(buf202, (15584, 128), (128, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (15584, 128), (128, 1), 0), reinterpret_tensor(arg92_1, (128, 128), (1, 128), 0), out=buf204)
        del arg92_1
        buf255 = reinterpret_tensor(buf198, (487, 32, 512), (16384, 512, 1), 0); del buf198  # reuse
        buf251 = reinterpret_tensor(buf255, (487, 32, 128), (16384, 512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_mid_o_1, z_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_42.run(buf177, buf204, arg93_1, arg94_1, arg95_1, buf251, 15584, 128, grid=grid(15584), stream=stream0)
        del arg94_1
        del arg95_1
        buf212 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf211, (8192, 128), (128, 1), 0), reinterpret_tensor(arg102_1, (128, 384), (1, 128), 0), out=buf212)
        del arg102_1
        buf213 = reinterpret_tensor(buf171, (1, 64, 128, 128, 4), (4194304, 65536, 512, 4, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [setitem_8], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_8.run(arg28_1, buf213, 4194304, grid=grid(4194304), stream=stream0)
        del arg28_1
        # Topologically Sorted Source Nodes: [setitem_8], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_9.run(buf213, 32768, grid=grid(32768), stream=stream0)
        buf215 = reinterpret_tensor(buf211, (1, 64, 128, 128), (1048576, 16384, 128, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [setitem_9], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_10.run(buf213, buf215, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_9], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_11.run(buf215, 8192, grid=grid(8192), stream=stream0)
        buf217 = reinterpret_tensor(buf71, (64, 8, 128, 16), (16384, 2048, 16, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [q_f_4, k_f_4, v_f_4, logit_bias_4, x_soft_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_12.run(buf212, arg103_1, buf217, 1048576, grid=grid(1048576), stream=stream0)
        buf218 = reinterpret_tensor(buf133, (64, 8, 128, 16), (16384, 2048, 16, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [q_f_4, k_f_4, v_f_4, logit_bias_4, x_soft_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_13.run(buf212, arg103_1, buf218, 1048576, grid=grid(1048576), stream=stream0)
        buf219 = buf118; del buf118  # reuse
        buf231 = reinterpret_tensor(buf117, (64, 8, 128, 1, 16, 1), (16384, 2048, 16, 16, 1, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [q_f_4, k_f_4, v_f_4, logit_bias_4, x_soft_8, einsum_14], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_14.run(buf212, arg103_1, buf219, buf231, 1048576, grid=grid(1048576), stream=stream0)
        del arg103_1
        del buf212
        buf220 = reinterpret_tensor(buf129, (64, 8, 128, 128), (131072, 16384, 128, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [q_f_4, k_f_4, v_f_4, logit_bias_4, x_soft_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_15.run(buf215, buf213, buf220, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [q_f_4, k_f_4, v_f_4, logit_bias_4, x_soft_8], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf221 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf217, buf218, buf219, buf220, False)
        del buf217
        del buf218
        del buf219
        buf222 = buf221[0]
        del buf221
        buf226 = reinterpret_tensor(buf169, (1, 64, 128, 128, 4), (4194304, 65536, 512, 4, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(buf215, buf213, buf226, 4194304, grid=grid(4194304), stream=stream0)
        buf227 = reinterpret_tensor(buf127, (1048576, 16), (16, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf226, (1048576, 4), (4, 1), 0), reinterpret_tensor(arg104_1, (4, 16), (1, 4), 0), out=buf227)
        del arg104_1
        buf228 = reinterpret_tensor(buf227, (1, 64, 128, 128, 16), (16777216, 262144, 2048, 16, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_17.run(buf228, arg105_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg105_1
        buf229 = reinterpret_tensor(buf220, (1048576, 8), (8, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (1048576, 16), (16, 1), 0), reinterpret_tensor(arg106_1, (16, 8), (1, 16), 0), out=buf229)
        del arg106_1
        del buf228
        buf230 = reinterpret_tensor(buf128, (64, 8, 128, 128, 1, 1), (131072, 16384, 128, 1, 1, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [einsum_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf229, arg107_1, buf230, 512, 16384, grid=grid(512, 16384), stream=stream0)
        del arg107_1
        del buf229
        buf232 = reinterpret_tensor(buf215, (512, 128, 16), (2048, 16, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [einsum_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf230, (512, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf231, (512, 128, 16), (2048, 16, 1), 0), out=buf232)
        del buf230
        del buf231
        buf233 = reinterpret_tensor(buf222, (1, 64, 128, 8, 16), (1048576, 16384, 128, 16, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [x_mid_4], Original ATen: [aten.add]
        triton_poi_fused_add_19.run(buf233, buf232, 1048576, grid=grid(1048576), stream=stream0)
        buf234 = reinterpret_tensor(buf232, (8192, 128), (128, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf233, (8192, 128), (128, 1), 0), reinterpret_tensor(arg108_1, (128, 128), (1, 128), 0), out=buf234)
        del arg108_1
        buf238 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [row_highway_4], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf238, 2097152, grid=grid(2097152), stream=stream0)
        buf240 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [row_highway_3], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf240, 2097152, grid=grid(2097152), stream=stream0)
        buf245 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [col_highway_3], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf245, 2097152, grid=grid(2097152), stream=stream0)
        buf261 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [col_highway_4], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf261, 2097152, grid=grid(2097152), stream=stream0)
        buf270 = reinterpret_tensor(buf226, (1, 8192, 512), (4194304, 512, 1), 0); del buf226  # reuse
        buf266 = reinterpret_tensor(buf270, (1, 8192, 128), (4194304, 512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_mid_d_2, z_8, row_highway_4, index_add__16, row_highway_3, index_add__12, col_highway_3, index_add__13, col_highway_4, index_add__17], Original ATen: [aten.add, aten.native_layer_norm, aten.new_zeros, aten.index_add]
        triton_per_fused_add_index_add_native_layer_norm_new_zeros_43.run(buf173, buf234, arg109_1, arg42_1, arg110_1, arg111_1, buf238, buf261, buf266, buf240, buf245, 8192, 128, grid=grid(8192), stream=stream0)
        del arg110_1
        del arg111_1
        # Topologically Sorted Source Nodes: [toks_5, index_add__14, index_add__15], Original ATen: [aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_44.run(arg60_1, arg59_1, buf177, buf204, arg93_1, arg61_1, buf240, buf245, 17858560, grid=grid(17858560), stream=stream0)
        buf243 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [row_k_5], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_28.run(buf240, buf243, 2097152, grid=grid(2097152), stream=stream0)
        buf244 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [row_t_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg30_1, (1, 487, 128), (62336, 128, 1), 0), reinterpret_tensor(buf243, (1, 128, 16384), (0, 16384, 1), 0), out=buf244)
        buf248 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [col_k_5], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_28.run(buf245, buf248, 2097152, grid=grid(2097152), stream=stream0)
        buf249 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [col_t_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg31_1, (1, 487, 128), (62336, 128, 1), 0), reinterpret_tensor(buf248, (1, 128, 16384), (0, 16384, 1), 0), out=buf249)
        buf250 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [x_mid_o_1, g_3], Original ATen: [aten.add, aten.mean]
        triton_per_fused_add_mean_45.run(buf177, buf204, arg93_1, buf250, 62336, 32, grid=grid(62336), stream=stream0)
        buf252 = reinterpret_tensor(buf255, (487, 32, 128), (16384, 512, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [z_7], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf244, buf252, 1994752, grid=grid(1994752), stream=stream0)
        buf253 = reinterpret_tensor(buf255, (487, 32, 128), (16384, 512, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [z_7], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf249, buf253, 1994752, grid=grid(1994752), stream=stream0)
        buf254 = reinterpret_tensor(buf255, (487, 32, 128), (16384, 512, 1), 384)  # alias
        # Topologically Sorted Source Nodes: [z_7], Original ATen: [aten.cat]
        triton_poi_fused_cat_47.run(buf250, buf254, 1994752, grid=grid(1994752), stream=stream0)
        del buf251
        del buf252
        del buf253
        del buf254
        buf256 = reinterpret_tensor(buf249, (15584, 512), (512, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf255, (15584, 512), (512, 1), 0), reinterpret_tensor(arg96_1, (512, 512), (1, 512), 0), out=buf256)
        del arg96_1
        buf257 = reinterpret_tensor(buf256, (487, 32, 512), (16384, 512, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf257, arg97_1, 7979008, grid=grid(7979008), stream=stream0)
        del arg97_1
        buf258 = reinterpret_tensor(buf203, (15584, 128), (128, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (15584, 512), (512, 1), 0), reinterpret_tensor(arg98_1, (512, 128), (1, 512), 0), out=buf258)
        del arg98_1
        buf259 = reinterpret_tensor(buf258, (487, 32, 128), (4096, 128, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [x_mid_o_1, off_stream_1], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(buf259, buf177, buf204, arg93_1, arg99_1, 1994752, grid=grid(1994752), stream=stream0)
        del arg93_1
        del arg99_1
        # Topologically Sorted Source Nodes: [toks_7, index_add__18, index_add__19], Original ATen: [aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_50.run(arg60_1, arg59_1, buf259, arg61_1, buf238, buf261, 17858560, grid=grid(17858560), stream=stream0)
        buf264 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d_2, g_4], Original ATen: [aten.add, aten.mean]
        triton_red_fused_add_mean_22.run(buf173, buf234, arg109_1, buf264, 8192, 128, grid=grid(8192), stream=stream0)
        buf265 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d_2, g_4], Original ATen: [aten.add, aten.mean]
        triton_per_fused_add_mean_23.run(buf264, buf265, 128, 64, grid=grid(128), stream=stream0)
        buf267 = reinterpret_tensor(buf270, (1, 8192, 128), (4194304, 512, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [z_9], Original ATen: [aten.cat]
        triton_poi_fused_cat_24.run(buf238, buf267, 1048576, grid=grid(1048576), stream=stream0)
        buf268 = reinterpret_tensor(buf270, (1, 8192, 128), (4194304, 512, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [z_9], Original ATen: [aten.cat]
        triton_poi_fused_cat_24.run(buf261, buf268, 1048576, grid=grid(1048576), stream=stream0)
        buf269 = reinterpret_tensor(buf270, (1, 8192, 128), (4194304, 512, 1), 384)  # alias
        # Topologically Sorted Source Nodes: [z_9], Original ATen: [aten.cat]
        triton_poi_fused_cat_25.run(buf265, buf269, 1048576, grid=grid(1048576), stream=stream0)
        del buf265
        del buf266
        del buf267
        del buf268
        del buf269
        buf271 = reinterpret_tensor(buf213, (8192, 512), (512, 1), 0); del buf213  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf270, (8192, 512), (512, 1), 0), reinterpret_tensor(arg112_1, (512, 512), (1, 512), 0), out=buf271)
        del arg112_1
        del buf270
        buf272 = reinterpret_tensor(buf271, (1, 8192, 512), (4194304, 512, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_26.run(buf272, arg113_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg113_1
        buf273 = reinterpret_tensor(buf233, (8192, 128), (128, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf272, (8192, 512), (512, 1), 0), reinterpret_tensor(arg114_1, (512, 128), (1, 512), 0), out=buf273)
        del arg114_1
        del buf272
        buf313 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [row_highway_5], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf313, 2097152, grid=grid(2097152), stream=stream0)
        buf318 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [col_highway_5], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf318, 2097152, grid=grid(2097152), stream=stream0)
        buf345 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [col_highway_6], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf345, 2097152, grid=grid(2097152), stream=stream0)
        buf348 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [row_highway_6], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_20.run(buf348, 2097152, grid=grid(2097152), stream=stream0)
        buf274 = reinterpret_tensor(buf273, (1, 8192, 128), (1048576, 128, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d_2, h_diag_3, row_highway_5, index_add__20, col_highway_5, index_add__21, col_highway_6, index_add__25, row_highway_6, index_add__24], Original ATen: [aten.add, aten.new_zeros, aten.index_add]
        triton_poi_fused_add_index_add_new_zeros_52.run(buf274, buf173, buf234, arg109_1, arg115_1, arg42_1, buf313, buf318, buf345, buf348, 1048576, grid=grid(1048576), stream=stream0)
        del arg109_1
        del arg115_1
        del arg42_1
        buf275 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [h_k_7], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_28.run(buf274, buf275, 2097152, grid=grid(2097152), stream=stream0)
        buf276 = reinterpret_tensor(buf257, (1, 487, 16384), (7979008, 16384, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [row_p_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg30_1, (1, 487, 128), (62336, 128, 1), 0), reinterpret_tensor(buf275, (1, 128, 16384), (0, 16384, 1), 0), out=buf276)
        buf277 = reinterpret_tensor(buf255, (1, 487, 16384), (7979008, 16384, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [col_p_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg31_1, (1, 487, 128), (62336, 128, 1), 0), reinterpret_tensor(buf275, (1, 128, 16384), (2097152, 16384, 1), 0), out=buf277)
        buf278 = buf259; del buf259  # reuse
        buf282 = reinterpret_tensor(buf204, (487, 32, 128), (4096, 128, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [h_off_7, off_in_1, layer_norm_11], Original ATen: [aten.mean, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mean_native_layer_norm_51.run(buf278, buf276, buf277, arg116_1, arg117_1, buf282, 15584, 128, grid=grid(15584), stream=stream0)
        del arg116_1
        del arg117_1
        buf283 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf282, (15584, 128), (128, 1), 0), reinterpret_tensor(arg118_1, (128, 384), (1, 128), 0), out=buf283)
        del arg118_1
        # Topologically Sorted Source Nodes: [setitem_10], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_31.run(buf284, 62336, grid=grid(62336), stream=stream0)
        buf286 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [setitem_11], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_32.run(buf284, buf286, 498688, grid=grid(498688), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_11], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_33.run(buf286, 15584, grid=grid(15584), stream=stream0)
        buf288 = reinterpret_tensor(buf282, (487, 8, 32, 16), (4096, 512, 16, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [q_f_5, k_f_5, v_f_5, logit_bias_5, x_soft_10], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_34.run(buf283, arg119_1, buf288, 1994752, grid=grid(1994752), stream=stream0)
        buf289 = reinterpret_tensor(buf177, (487, 8, 32, 16), (4096, 512, 16, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [q_f_5, k_f_5, v_f_5, logit_bias_5, x_soft_10], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_35.run(buf283, arg119_1, buf289, 1994752, grid=grid(1994752), stream=stream0)
        buf290 = reinterpret_tensor(buf201, (487, 8, 32, 16), (4096, 512, 16, 1), 0); del buf201  # reuse
        buf302 = reinterpret_tensor(buf183, (487, 8, 32, 1, 16, 1), (4096, 512, 16, 16, 1, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [q_f_5, k_f_5, v_f_5, logit_bias_5, x_soft_10, einsum_17], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_36.run(buf283, arg119_1, buf290, buf302, 1994752, grid=grid(1994752), stream=stream0)
        del arg119_1
        del buf283
        buf291 = reinterpret_tensor(buf200, (487, 8, 32, 32), (8192, 1024, 32, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [q_f_5, k_f_5, v_f_5, logit_bias_5, x_soft_10], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_37.run(buf286, buf284, buf291, 3989504, grid=grid(3989504), stream=stream0)
        # Topologically Sorted Source Nodes: [q_f_5, k_f_5, v_f_5, logit_bias_5, x_soft_10], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf292 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf288, buf289, buf290, buf291, False)
        del buf288
        del buf289
        buf293 = buf292[0]
        del buf292
        buf297 = reinterpret_tensor(buf290, (487, 1, 32, 32, 4), (4096, 1, 128, 4, 1), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_38.run(buf286, buf284, buf297, 1994752, grid=grid(1994752), stream=stream0)
        del buf284
        buf298 = reinterpret_tensor(buf277, (498688, 16), (16, 1), 0); del buf277  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf297, (498688, 4), (4, 1), 0), reinterpret_tensor(arg120_1, (4, 16), (1, 4), 0), out=buf298)
        del arg120_1
        buf299 = reinterpret_tensor(buf298, (487, 1, 32, 32, 16), (16384, 1, 512, 16, 1), 0); del buf298  # reuse
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_39.run(buf299, arg121_1, 7979008, grid=grid(7979008), stream=stream0)
        del arg121_1
        buf300 = reinterpret_tensor(buf291, (498688, 8), (8, 1), 0); del buf291  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf299, (498688, 16), (16, 1), 0), reinterpret_tensor(arg122_1, (16, 8), (1, 16), 0), out=buf300)
        del arg122_1
        buf301 = reinterpret_tensor(buf199, (487, 8, 32, 32, 1, 1), (8192, 1024, 32, 1, 1, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [einsum_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf300, arg123_1, buf301, 3896, 1024, grid=grid(3896, 1024), stream=stream0)
        del arg123_1
        del buf300
        buf303 = reinterpret_tensor(buf297, (3896, 32, 16), (512, 16, 1), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [einsum_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf301, (3896, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf302, (3896, 32, 16), (512, 16, 1), 0), out=buf303)
        del buf301
        del buf302
        buf304 = reinterpret_tensor(buf293, (487, 1, 32, 8, 16), (4096, 1, 128, 16, 1), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [x_mid_5], Original ATen: [aten.add]
        triton_poi_fused_add_41.run(buf304, buf303, 1994752, grid=grid(1994752), stream=stream0)
        buf305 = reinterpret_tensor(buf303, (15584, 128), (128, 1), 0); del buf303  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf304, (15584, 128), (128, 1), 0), reinterpret_tensor(arg124_1, (128, 128), (1, 128), 0), out=buf305)
        del arg124_1
        buf328 = reinterpret_tensor(buf299, (487, 32, 512), (16384, 512, 1), 0); del buf299  # reuse
        buf324 = reinterpret_tensor(buf328, (487, 32, 128), (16384, 512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_mid_o_2, z_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_42.run(buf278, buf305, arg125_1, arg126_1, arg127_1, buf324, 15584, 128, grid=grid(15584), stream=stream0)
        del arg126_1
        del arg127_1
        buf309 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (8192, 128), (128, 1), 0), reinterpret_tensor(arg132_1, (128, 128), (1, 128), 0), out=buf309)
        del arg132_1
        buf310 = reinterpret_tensor(buf309, (1, 64, 128, 128), (1048576, 16384, 128, 1), 0); del buf309  # reuse
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf310, arg133_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg133_1
        buf311 = reinterpret_tensor(buf173, (8192, 128), (128, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg135_1, reinterpret_tensor(buf310, (8192, 128), (128, 1), 0), reinterpret_tensor(arg134_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf311)
        del arg134_1
        del arg135_1
        buf312 = reinterpret_tensor(buf310, (64, 128, 128), (16384, 128, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf311, (64, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf311, (64, 128, 128), (16384, 1, 128), 0), out=buf312)
        del buf311
        # Topologically Sorted Source Nodes: [toks_9, index_add__22, index_add__23], Original ATen: [aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_44.run(arg60_1, arg59_1, buf278, buf305, arg125_1, arg61_1, buf313, buf318, 17858560, grid=grid(17858560), stream=stream0)
        buf316 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [row_k_8], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_28.run(buf313, buf316, 2097152, grid=grid(2097152), stream=stream0)
        del buf313
        buf317 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [row_t_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg30_1, (1, 487, 128), (62336, 128, 1), 0), reinterpret_tensor(buf316, (1, 128, 16384), (0, 16384, 1), 0), out=buf317)
        del arg30_1
        buf321 = buf316; del buf316  # reuse
        # Topologically Sorted Source Nodes: [col_k_8], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_28.run(buf318, buf321, 2097152, grid=grid(2097152), stream=stream0)
        del buf318
        buf322 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [col_t_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg31_1, (1, 487, 128), (62336, 128, 1), 0), reinterpret_tensor(buf321, (1, 128, 16384), (0, 16384, 1), 0), out=buf322)
        del arg31_1
        del buf321
        buf323 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [x_mid_o_2, g_5], Original ATen: [aten.add, aten.mean]
        triton_per_fused_add_mean_45.run(buf278, buf305, arg125_1, buf323, 62336, 32, grid=grid(62336), stream=stream0)
        buf325 = reinterpret_tensor(buf328, (487, 32, 128), (16384, 512, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [z_11], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf317, buf325, 1994752, grid=grid(1994752), stream=stream0)
        del buf317
        buf326 = reinterpret_tensor(buf328, (487, 32, 128), (16384, 512, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [z_11], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf322, buf326, 1994752, grid=grid(1994752), stream=stream0)
        buf327 = reinterpret_tensor(buf328, (487, 32, 128), (16384, 512, 1), 384)  # alias
        # Topologically Sorted Source Nodes: [z_11], Original ATen: [aten.cat]
        triton_poi_fused_cat_47.run(buf323, buf327, 1994752, grid=grid(1994752), stream=stream0)
        del buf323
        del buf324
        del buf325
        del buf326
        del buf327
        buf329 = reinterpret_tensor(buf322, (15584, 512), (512, 1), 0); del buf322  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf328, (15584, 512), (512, 1), 0), reinterpret_tensor(arg128_1, (512, 512), (1, 512), 0), out=buf329)
        del arg128_1
        del buf328
        buf330 = reinterpret_tensor(buf329, (487, 32, 512), (16384, 512, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_48.run(buf330, arg129_1, 7979008, grid=grid(7979008), stream=stream0)
        del arg129_1
        buf331 = reinterpret_tensor(buf304, (15584, 128), (128, 1), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf330, (15584, 512), (512, 1), 0), reinterpret_tensor(arg130_1, (512, 128), (1, 512), 0), out=buf331)
        del arg130_1
        del buf330
        buf332 = reinterpret_tensor(buf331, (487, 32, 128), (4096, 128, 1), 0); del buf331  # reuse
        # Topologically Sorted Source Nodes: [x_mid_o_2, off_stream_2], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(buf332, buf278, buf305, arg125_1, arg131_1, 1994752, grid=grid(1994752), stream=stream0)
        del arg125_1
        del arg131_1
        del buf278
        buf333 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (15584, 128), (128, 1), 0), reinterpret_tensor(arg136_1, (128, 128), (1, 128), 0), out=buf333)
        del arg136_1
        buf334 = reinterpret_tensor(buf333, (487, 1, 32, 128), (4096, 1, 128, 1), 0); del buf333  # reuse
        # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_53.run(buf334, arg137_1, 1994752, grid=grid(1994752), stream=stream0)
        del arg137_1
        buf335 = reinterpret_tensor(buf286, (15584, 32), (32, 1), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg139_1, reinterpret_tensor(buf334, (15584, 128), (128, 1), 0), reinterpret_tensor(arg138_1, (128, 32), (1, 128), 0), alpha=1, beta=1, out=buf335)
        del arg138_1
        del arg139_1
        buf336 = reinterpret_tensor(buf334, (15584, 128), (128, 1), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (15584, 128), (128, 1), 0), reinterpret_tensor(arg140_1, (128, 128), (1, 128), 0), out=buf336)
        del arg140_1
        buf337 = reinterpret_tensor(buf336, (487, 1, 32, 128), (4096, 1, 128, 1), 0); del buf336  # reuse
        # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_53.run(buf337, arg141_1, 1994752, grid=grid(1994752), stream=stream0)
        del arg141_1
        buf338 = empty_strided_cuda((15584, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg143_1, reinterpret_tensor(buf337, (15584, 128), (128, 1), 0), reinterpret_tensor(arg142_1, (128, 32), (1, 128), 0), alpha=1, beta=1, out=buf338)
        del arg142_1
        del arg143_1
        del buf337
        buf339 = empty_strided_cuda((487, 32, 32), (1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf335, (487, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf338, (487, 32, 32), (1024, 1, 32), 0), out=buf339)
        del buf335
        del buf338
        buf340 = empty_strided_cuda((8192, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [node_U], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg147_1, reinterpret_tensor(buf274, (8192, 128), (128, 1), 0), reinterpret_tensor(arg146_1, (128, 32), (1, 128), 0), alpha=1, beta=1, out=buf340)
        del arg146_1
        del arg147_1
        buf341 = empty_strided_cuda((8192, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [node_V], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg149_1, reinterpret_tensor(buf274, (8192, 128), (128, 1), 0), reinterpret_tensor(arg148_1, (128, 32), (1, 128), 0), alpha=1, beta=1, out=buf341)
        del arg148_1
        del arg149_1
        buf342 = reinterpret_tensor(buf264, (8192, 1), (1, 1), 0); del buf264  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (8192, 128), (128, 1), 0), reinterpret_tensor(arg144_1, (128, 1), (1, 128), 0), out=buf342)
        del arg144_1
        del buf274
        buf343 = reinterpret_tensor(buf342, (1, 64, 128, 1), (8192, 128, 1, 1), 0); del buf342  # reuse
        # Topologically Sorted Source Nodes: [sigmoid], Original ATen: [aten.sigmoid]
        triton_poi_fused_sigmoid_54.run(buf343, arg145_1, 8192, grid=grid(8192), stream=stream0)
        del arg145_1
        buf344 = empty_strided_cuda((1, 2079744), (2079744, 1), torch.float32)
        # Topologically Sorted Source Nodes: [packed_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_55.run(buf312, buf339, buf340, buf341, buf343, buf344, 2079744, grid=grid(2079744), stream=stream0)
        del buf312
        del buf339
        del buf340
        del buf341
        # Topologically Sorted Source Nodes: [toks_11, index_add__27, index_add__26], Original ATen: [aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_50.run(arg61_1, arg59_1, buf332, arg60_1, buf345, buf348, 17858560, grid=grid(17858560), stream=stream0)
        del arg59_1
        del arg60_1
        del arg61_1
        del buf332
    return (buf344, reinterpret_tensor(buf343, (1, 64, 128), (8192, 128, 1), 0), reinterpret_tensor(buf345, (1, 8192, 128), (2097152, 128, 1), 0), reinterpret_tensor(buf348, (1, 8192, 128), (2097152, 128, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 8192, 9), (73728, 9, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 18), (18, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((2, 40596), (40596, 1), device='cuda:0', dtype=torch.int64)
    arg7_1 = rand_strided((40596, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    arg28_1 = rand_strided((64, 128, 128, 4), (65536, 512, 4, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((487, 128, 128, 4), (65536, 512, 4, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((487, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((487, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg43_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1090, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg60_1 = rand_strided((1090, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg61_1 = rand_strided((1090, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((32, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((32, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((32, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((32, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
