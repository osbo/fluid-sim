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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ux/cuxas5smet5irzbutm5mlikntnfb7p2cnw37ue4dj3uszieztsgd.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view, %full_default_6], 1), kwargs = {})
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
    xnumel = 148480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
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
    tmp8 = tl.load(in_ptr0 + (3 + (9*x1) + x0), tmp7 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp0 >= tmp5
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr1 + ((-6) + x0), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ev/ceveqsjtrb52ihrsk7ubsdret6zecokr7zd67zrvc4gmua344wj5.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute, %full_default_7],), kwargs = {})
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/wi/cwitdy2norlk43khhli2cmbcvyq67h5t4awmrbhpazmdybxlkg4h.py
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
    xnumel = 950272
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/7v/c7vvztgrxcn2iawfljt4t2yf2ownv6q2hymcdlbivodzv6w3xwlw.py
# Topologically Sorted Source Nodes: [aggr], Original ATen: [aten.zeros_like]
# Source node to ATen node mapping:
#   aggr => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([7424, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
    xnumel = 950272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (256*x1)), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ht/chtqzw6ojtwrnoqgyh6vg4375dphl3gs5jq7zcui3wofdfcravr5.py
# Topologically Sorted Source Nodes: [aggr, getitem_4, messages, index_add_], Original ATen: [aten.zeros_like, aten.index, aten.mul, aten.index_add]
# Source node to ATen node mapping:
#   aggr => full_default
#   getitem_4 => index
#   index_add_ => index_put
#   messages => mul_3
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([7424, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
    xnumel = 6323328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128)
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (49401 + x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 7424, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 7424)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 7424")
    tmp7 = tmp6 + tmp1
    tmp8 = tmp6 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp6)
    tl.device_assert(((0 <= tmp9) & (tmp9 < 7424)) | ~(xmask), "index out of bounds: 0 <= tmp9 < 7424")
    tmp11 = tl.load(in_ptr1 + (x0 + (128*tmp9)), xmask)
    tmp13 = tmp11 * tmp12
    tl.atomic_add(out_ptr0 + (x0 + (256*tmp4)), tmp13, xmask, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/yp/cypi5wkeitlamzobaihuffxfmzcnk5qctxm3gdx444dogunhxdnt.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   out => add_2
# Graph fragment:
#   %add_tensor_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_18, %arg15_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze, %add_tensor_18), kwargs = {})
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
    xnumel = 950272
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/oh/cohb3hdgqlfolw6swt7cq7awdbndqzeafo5zervq3kpvdohq7upv.py
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
    xnumel = 7424
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/l6/cl6nqxxwa24cetyiic3z75xmzhbepyw4yc46rqky2gtrcrbn5rmd.py
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
    xnumel = 7424
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/w2/cw2rlnszlwjp2vo3pbb3am45zu3e6ltknn66ehfrzbhhk3nseevf.py
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
    xnumel = 3801088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ot/cotjvh7qrbmz6wkzeyfpiz2lfq3l5mejtfzmkp7enuoiwzxpotqk.py
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
    xnumel = 29696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = (xindex // 4) % 128
    x2 = (xindex // 512)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (516*x1) + (65536*x2)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/w4/cw4cnnytpgt5t5nxco7r4dauab4bxiwlx5hcuhm3oygwx7va55xv.py
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
    xnumel = 950272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + (4*x0)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/4b/c4bsenfilvi3zg6hymmrhm56nvrqjiinojo7t6bbmo5q754f25ie.py
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
    xnumel = 7424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = 1.0
    tl.store(out_ptr0 + ((129*x0) + (16384*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/so/cso6onpcfvoz33ftva2trkuh66rtam26oqx5wmjdmjlk5l3msogd.py
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
    xnumel = 950272
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/y7/cy7dugxmbvb2hdh3bqav2ivmigxlivbgvdl34bdgtubybxau7hb2.py
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
    xnumel = 950272
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/az/cazpjvk4b3hmzq2xo5wqowy5n6g343bpmjwt3xo4bd27qqabi7v3.py
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
    xnumel = 950272
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/7z/c7zbqhkxk4eophmaiicqke6bbenc3snm4ope2jtuqau7xnw3omwb.py
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
    xnumel = 7602176
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/kj/ckjcrktzktagfowjddmcgplcfv7a7mm7zbbqomvlfb4uppcbieuq.py
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
    xnumel = 3801088
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/pi/cpijqclwmn23jwjqt3fvak6xq4r5vpj5h35ubkbwbd2klwkybdxd.py
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
    xnumel = 15204352
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/hs/chstxqccuiynohy3bs7l6qlzpssaxjmf4k727xyupk3ohctqrafi.py
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
    ynumel = 464
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/om/com5vun7ktkjigf4ntygdk75hwti36trqkznmgjzgivqc3rek6bc.py
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
    xnumel = 950272
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/if/cify4hklcroanqtqsix45bekvnbmg4oqtxbbb6ewt42rmg3kigpq.py
# Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_14 => add_15, erf_4, mul_20, mul_21, mul_22
# Graph fragment:
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_48, 0.5), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_48, 0.7071067811865476), kwargs = {})
#   %erf_4 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_21,), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_4, 1), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %add_15), kwargs = {})
triton_poi_fused_gelu_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_20', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3801088
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/zr/czrpke75t2myd2b25udotrams2cakuzjhfczj2eslgi2cirp7xfz.py
# Topologically Sorted Source Nodes: [x_mid_d, h_diag_1], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   h_diag_1 => add_16
#   x_mid_d => add_12
# Graph fragment:
#   %add_12 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_9, %view_46), kwargs = {})
#   %add_16 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %view_50), kwargs = {})
triton_poi_fused_add_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_21', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 950272
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/tj/ctjdjjx552jkauwlpyrsrtbhxuz4vrtrgynusydb7lo4lskqyah2.py
# Topologically Sorted Source Nodes: [h_k_3], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   h_k_3 => constant_pad_nd_1
# Graph fragment:
#   %constant_pad_nd_1 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_51, [0, 0, 0, 0, 0, 6], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_22', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 16384)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 58, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x2), tmp2, other=0.0)
    tl.store(out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/7a/c7alae23mlqyanrnlomtuxmub2okiuv56tycff3zwyss4x7k5r2t.py
# Topologically Sorted Source Nodes: [h_off_3, layer_norm_3], Original ATen: [aten.mean, aten.native_layer_norm]
# Source node to ATen node mapping:
#   h_off_3 => mean_4
#   layer_norm_3 => add_18, add_19, mul_23, mul_24, rsqrt_3, sub_3, var_mean_3
# Graph fragment:
#   %mean_4 : [num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%view_61, [2]), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%mean_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean_4, %getitem_11), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %arg48_1), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %arg49_1), kwargs = {})
triton_per_fused_mean_native_layer_norm_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_layer_norm_23', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7488
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/r2/cr2bw5xzjagyor3gdod3expszfurzg4ajv5qsk76bnld2axqbdue.py
# Topologically Sorted Source Nodes: [sub], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   sub => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_6, [2, 4]), kwargs = {})
triton_per_fused_mean_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_24', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5), tmp5, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/wi/cwiv4vyjaaztb7nebnr63haeyablgvfjopu7bqnofluawb4wmgmn.py
# Topologically Sorted Source Nodes: [setitem_2], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_2 => full_default_4, index_put_4
# Graph fragment:
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_4 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%view_7, [None, None, %iota_1, %iota_1], %full_default_4), kwargs = {})
triton_poi_fused_index_put_lift_fresh_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_25', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/v7/cv76weylz66z2xdicrju65ma3dnpt5hkynfxqud3foxwcmxugss3.py
# Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_3 => full_default_5, index_put_5
# Graph fragment:
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_5 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_10, [None, None, %iota_1, %iota_1], %full_default_5), kwargs = {})
triton_poi_fused_index_put_lift_fresh_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_26', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/vl/cvlrntf3qjv6blukwqtsvvjn6awyd5codzc2c26ydlotwq75njap.py
# Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_3 => full_default_5, index_put_5
# Graph fragment:
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_5 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_10, [None, None, %iota_1, %iota_1], %full_default_5), kwargs = {})
triton_poi_fused_index_put_lift_fresh_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_27', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/zb/czbz6mh7szwz52u4wbjdulnjkaxejhe5qawabb4m3i7panb44ud7.py
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
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_8,), kwargs = {})
#   %_scaled_dot_product_efficient_attention_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_8, %clone_9, %clone_10, %expand_9, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_28', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 958464
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/tj/ctjquu3dfiyelffy7dtcf6wbjcivwy2uxzxu6tsdnfuxd5na3ymt.py
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
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_8,), kwargs = {})
#   %_scaled_dot_product_efficient_attention_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_8, %clone_9, %clone_10, %expand_9, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_29', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 958464
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/7u/c7uylpmwh3te4k73mvkvwts2ka7puzdzgdpn7q2if3cbk4asj5h2.py
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
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_8,), kwargs = {})
#   %_scaled_dot_product_efficient_attention_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_8, %clone_9, %clone_10, %expand_9, False), kwargs = {})
#   %clone_13 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_56,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_30', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 958464
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/fp/cfp34es3sgepujcnth7vhwromjidy2klab43ylvohwwkdcozutgf.py
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
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_8,), kwargs = {})
#   %_scaled_dot_product_efficient_attention_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_8, %clone_9, %clone_10, %expand_9, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention_clone_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_31', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/3l/c3lthhgxxm7fnwe56g6oxlklevv6zhdsnsxhmnsehdwkg7dlwses.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %select_scatter_default_1 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%index_put_4, %index_put_5, 4, 3), kwargs = {})
triton_poi_fused_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_32', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/5u/c5unz2vtscm3xdsuqneqjol4h43clkwiaprxnoe4mttonnmjowk3.py
# Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_17 => add_20, erf_5, mul_25, mul_26, mul_27
# Graph fragment:
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_75, 0.5), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_75, 0.7071067811865476), kwargs = {})
#   %erf_5 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_26,), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_5, 1), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %add_20), kwargs = {})
triton_poi_fused_gelu_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_33', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/bi/cbimjydunip7dokjcpdacrnplnsno4w32bvroezno7sdlrqjr4ox.py
# Topologically Sorted Source Nodes: [einsum_5], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   einsum_5 => clone_12
# Graph fragment:
#   %clone_12 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_55,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_34', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ks/cks4zzm44uhgjol6nrzl62wlialsqar6wlghga3yjda4v3s3gknq.py
# Topologically Sorted Source Nodes: [x_mid_1], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_mid_1 => add_21
# Graph fragment:
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_78, %view_82), kwargs = {})
triton_poi_fused_add_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_35', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 958464
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/cx/ccx3objsygkvp7vbjczugvjy57ctpc2vj6lusgssqwtivbv4ki2m.py
# Topologically Sorted Source Nodes: [x_mid_o, z_1], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_mid_o => add_22
#   z_1 => add_23, add_24, mul_28, mul_29, rsqrt_4, sub_4, var_mean_4
# Graph fragment:
#   %add_22 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_4, %view_86), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_22, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_22, %getitem_17), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_23,), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %arg58_1), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %arg59_1), kwargs = {})
triton_per_fused_add_native_layer_norm_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_36', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7488
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/2h/c2hna3nodrfp3sc5i7b2p42kxukhcmuudzyadezqzsem6ppneltz.py
# Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_20 => add_25, erf_6, mul_30, mul_31, mul_32
# Graph fragment:
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_88, 0.5), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_88, 0.7071067811865476), kwargs = {})
#   %erf_6 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_31,), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_6, 1), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, %add_25), kwargs = {})
triton_poi_fused_gelu_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_37', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3833856
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/bz/cbzf6b7ra5dvdxf3p6tud4rsvh2chmp3nw5pjlayzaln7cfc77wy.py
# Topologically Sorted Source Nodes: [x_mid_o, off_stream], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   off_stream => add_26
#   x_mid_o => add_22
# Graph fragment:
#   %add_22 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_4, %view_86), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_22, %view_90), kwargs = {})
triton_poi_fused_add_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_38', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 958464
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/jq/cjq6rmbnmfvhowyuvsvbf5r6kmw5mkhli4jexandzdkqybf7kmld.py
# Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_26 => add_28, erf_8, mul_36, mul_37, mul_38
# Graph fragment:
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_103, 0.5), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_103, 0.7071067811865476), kwargs = {})
#   %erf_8 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_37,), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_8, 1), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_36, %add_28), kwargs = {})
triton_poi_fused_gelu_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_39', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 958464
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/5y/c5yjcq5vuyr6enlajdr7ohqvjtqe6f7x55xy6cki2zph2bhv2fym.py
# Topologically Sorted Source Nodes: [sigmoid], Original ATen: [aten.sigmoid]
# Source node to ATen node mapping:
#   sigmoid => sigmoid
# Graph fragment:
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_116,), kwargs = {})
triton_poi_fused_sigmoid_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_40', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7424
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/zo/czoiwtxdfdvyksxz5nviwgj6kunfs7yoqua6zdfnk565hnwopsyo.py
# Topologically Sorted Source Nodes: [packed_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   packed_1 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_3, %view_117], 1), kwargs = {})
triton_poi_fused_cat_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_41', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1672448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1665024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 950272, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.load(in_ptr0 + (x0), tmp7 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp0 >= tmp5
    tmp10 = tl.full([1], 1189888, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tmp12 & tmp4
    tmp14 = tl.load(in_ptr1 + ((-950272) + x0), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp0 >= tmp10
    tmp16 = tl.full([1], 1427456, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr2 + ((-1189888) + x0), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp0 >= tmp16
    tmp22 = tmp21 & tmp4
    tmp23 = tl.load(in_ptr3 + ((-1427456) + x0), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp18, tmp20, tmp23)
    tmp25 = tl.where(tmp12, tmp14, tmp24)
    tmp26 = tl.where(tmp6, tmp8, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp4, tmp26, tmp27)
    tmp29 = tmp0 >= tmp3
    tmp30 = tl.full([1], 1672448, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = tl.load(in_ptr4 + ((-1665024) + x0), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.where(tmp4, tmp28, tmp32)
    tl.store(out_ptr0 + (x0), tmp33, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 7424, 9), (67716, 9, 1))
    assert_size_stride(arg1_1, (1, 12), (12, 1))
    assert_size_stride(arg2_1, (128, 18), (18, 1))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, 128), (128, 1))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (2, 49401), (49401, 1))
    assert_size_stride(arg7_1, (49401, ), (1, ))
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
    assert_size_stride(arg28_1, (58, 128, 128, 4), (65536, 512, 4, 1))
    assert_size_stride(arg29_1, (234, 128, 128, 4), (65536, 512, 4, 1))
    assert_size_stride(arg30_1, (234, 64), (64, 1))
    assert_size_stride(arg31_1, (234, 64), (64, 1))
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
    assert_size_stride(arg42_1, (128, ), (1, ))
    assert_size_stride(arg43_1, (128, ), (1, ))
    assert_size_stride(arg44_1, (512, 128), (128, 1))
    assert_size_stride(arg45_1, (512, ), (1, ))
    assert_size_stride(arg46_1, (128, 512), (512, 1))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (128, ), (1, ))
    assert_size_stride(arg49_1, (128, ), (1, ))
    assert_size_stride(arg50_1, (384, 128), (128, 1))
    assert_size_stride(arg51_1, (384, ), (1, ))
    assert_size_stride(arg52_1, (16, 4), (4, 1))
    assert_size_stride(arg53_1, (16, ), (1, ))
    assert_size_stride(arg54_1, (8, 16), (16, 1))
    assert_size_stride(arg55_1, (8, ), (1, ))
    assert_size_stride(arg56_1, (128, 128), (128, 1))
    assert_size_stride(arg57_1, (128, ), (1, ))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (128, ), (1, ))
    assert_size_stride(arg60_1, (512, 128), (128, 1))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (128, 512), (512, 1))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (128, 128), (128, 1))
    assert_size_stride(arg65_1, (128, ), (1, ))
    assert_size_stride(arg66_1, (128, 128), (128, 1))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (128, 128), (128, 1))
    assert_size_stride(arg69_1, (128, ), (1, ))
    assert_size_stride(arg70_1, (32, 128), (128, 1))
    assert_size_stride(arg71_1, (32, ), (1, ))
    assert_size_stride(arg72_1, (128, 128), (128, 1))
    assert_size_stride(arg73_1, (128, ), (1, ))
    assert_size_stride(arg74_1, (32, 128), (128, 1))
    assert_size_stride(arg75_1, (32, ), (1, ))
    assert_size_stride(arg76_1, (1, 128), (128, 1))
    assert_size_stride(arg77_1, (1, ), (1, ))
    assert_size_stride(arg78_1, (32, 128), (128, 1))
    assert_size_stride(arg79_1, (32, ), (1, ))
    assert_size_stride(arg80_1, (32, 128), (128, 1))
    assert_size_stride(arg81_1, (32, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((7424, 20), (20, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(arg0_1, arg1_1, buf0, 148480, grid=grid(148480), stream=stream0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((20, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(arg2_1, buf1, 2560, grid=grid(2560), stream=stream0)
        del arg2_1
        buf2 = empty_strided_cuda((7424, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf0, buf1, out=buf2)
        del buf0
        del buf1
        buf3 = reinterpret_tensor(buf2, (1, 7424, 128), (950272, 128, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf3, arg3_1, 950272, grid=grid(950272), stream=stream0)
        del arg3_1
        buf4 = empty_strided_cuda((7424, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg5_1, reinterpret_tensor(buf3, (7424, 128), (128, 1), 0), reinterpret_tensor(arg4_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf4)
        del arg4_1
        del arg5_1
        buf9 = empty_strided_cuda((7424, 256), (256, 1), torch.float32)
        buf5 = reinterpret_tensor(buf9, (7424, 128), (256, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [self_features], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, buf4, reinterpret_tensor(arg10_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf5)
        del arg10_1
        del arg11_1
        buf6 = reinterpret_tensor(buf3, (7424, 128), (128, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [neighbor_features], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg9_1, buf4, reinterpret_tensor(arg8_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf6)
        del arg8_1
        del arg9_1
        buf7 = reinterpret_tensor(buf9, (7424, 128), (256, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [aggr], Original ATen: [aten.zeros_like]
        triton_poi_fused_zeros_like_3.run(buf7, 950272, grid=grid(950272), stream=stream0)
        # Topologically Sorted Source Nodes: [aggr, getitem_4, messages, index_add_], Original ATen: [aten.zeros_like, aten.index, aten.mul, aten.index_add]
        triton_poi_fused_index_index_add_mul_zeros_like_4.run(arg6_1, buf6, arg7_1, buf7, 6323328, grid=grid(6323328), stream=stream0)
        del buf5
        del buf7
        buf10 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf9, reinterpret_tensor(arg12_1, (256, 128), (1, 256), 0), out=buf10)
        del arg12_1
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf11, arg13_1, 950272, grid=grid(950272), stream=stream0)
        del arg13_1
        buf12 = empty_strided_cuda((7424, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.gelu]
        extern_kernels.mm(buf11, reinterpret_tensor(arg14_1, (128, 128), (1, 128), 0), out=buf12)
        del arg14_1
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.add]
        triton_poi_fused_add_5.run(buf13, buf4, arg15_1, 950272, grid=grid(950272), stream=stream0)
        del arg15_1
        buf18 = buf9; del buf9  # reuse
        buf14 = reinterpret_tensor(buf18, (7424, 128), (256, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [self_features_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg19_1, buf13, reinterpret_tensor(arg18_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf14)
        del arg18_1
        del arg19_1
        buf15 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [neighbor_features_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg17_1, buf13, reinterpret_tensor(arg16_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf15)
        del arg16_1
        del arg17_1
        buf16 = reinterpret_tensor(buf18, (7424, 128), (256, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [aggr_1], Original ATen: [aten.zeros_like]
        triton_poi_fused_zeros_like_3.run(buf16, 950272, grid=grid(950272), stream=stream0)
        # Topologically Sorted Source Nodes: [aggr_1, getitem_7, messages_1, index_add__1], Original ATen: [aten.zeros_like, aten.index, aten.mul, aten.index_add]
        triton_poi_fused_index_index_add_mul_zeros_like_4.run(arg6_1, buf15, arg7_1, buf16, 6323328, grid=grid(6323328), stream=stream0)
        del arg6_1
        del arg7_1
        del buf14
        del buf16
        buf19 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf18, reinterpret_tensor(arg20_1, (256, 128), (1, 256), 0), out=buf19)
        del arg20_1
        del buf18
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf20, arg21_1, 950272, grid=grid(950272), stream=stream0)
        del arg21_1
        buf21 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.gelu]
        extern_kernels.mm(buf20, reinterpret_tensor(arg22_1, (128, 128), (1, 128), 0), out=buf21)
        del arg22_1
        buf25 = reinterpret_tensor(buf20, (1, 7424, 128), (950272, 128, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [h_2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_6.run(buf13, buf21, arg23_1, arg24_1, arg25_1, buf25, 7424, 128, grid=grid(7424), stream=stream0)
        del arg23_1
        del arg24_1
        del arg25_1
        buf26 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg27_1, reinterpret_tensor(buf25, (7424, 128), (128, 1), 0), reinterpret_tensor(arg26_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf26)
        del arg26_1
        del arg27_1
        buf30 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_7.run(buf26, arg32_1, arg33_1, buf30, 7424, 128, grid=grid(7424), stream=stream0)
        del arg32_1
        del arg33_1
        buf31 = empty_strided_cuda((7424, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (7424, 128), (128, 1), 0), reinterpret_tensor(arg34_1, (128, 384), (1, 128), 0), out=buf31)
        del arg34_1
        buf32 = empty_strided_cuda((1, 58, 128, 128, 4), (3801088, 65536, 512, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_8.run(arg28_1, buf32, 3801088, grid=grid(3801088), stream=stream0)
        del arg28_1
        # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_9.run(buf32, 29696, grid=grid(29696), stream=stream0)
        buf34 = reinterpret_tensor(buf30, (1, 58, 128, 128), (950272, 16384, 128, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_10.run(buf32, buf34, 950272, grid=grid(950272), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_11.run(buf34, 7424, grid=grid(7424), stream=stream0)
        buf36 = reinterpret_tensor(buf13, (58, 8, 128, 16), (16384, 2048, 16, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [q_f, k_f, v_f, logit_bias, x_soft], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_12.run(buf31, arg35_1, buf36, 950272, grid=grid(950272), stream=stream0)
        buf37 = empty_strided_cuda((58, 8, 128, 16), (16384, 2048, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f, k_f, v_f, logit_bias, x_soft], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_13.run(buf31, arg35_1, buf37, 950272, grid=grid(950272), stream=stream0)
        buf38 = empty_strided_cuda((58, 8, 128, 16), (16384, 2048, 16, 1), torch.float32)
        buf50 = empty_strided_cuda((58, 8, 128, 1, 16, 1), (16384, 2048, 16, 16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f, k_f, v_f, logit_bias, x_soft, einsum_2], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_14.run(buf31, arg35_1, buf38, buf50, 950272, grid=grid(950272), stream=stream0)
        del arg35_1
        del buf31
        buf39 = empty_strided_cuda((58, 8, 128, 128), (131072, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f, k_f, v_f, logit_bias, x_soft], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_15.run(buf34, buf32, buf39, 7602176, grid=grid(7602176), stream=stream0)
        # Topologically Sorted Source Nodes: [q_f, k_f, v_f, logit_bias, x_soft], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf40 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf36, buf37, buf38, buf39, False)
        del buf36
        del buf37
        del buf38
        buf41 = buf40[0]
        del buf40
        buf45 = empty_strided_cuda((1, 58, 128, 128, 4), (3801088, 65536, 512, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_16.run(buf34, buf32, buf45, 3801088, grid=grid(3801088), stream=stream0)
        del buf32
        buf46 = empty_strided_cuda((950272, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf45, (950272, 4), (4, 1), 0), reinterpret_tensor(arg36_1, (4, 16), (1, 4), 0), out=buf46)
        del arg36_1
        buf47 = reinterpret_tensor(buf46, (1, 58, 128, 128, 16), (15204352, 262144, 2048, 16, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_17.run(buf47, arg37_1, 15204352, grid=grid(15204352), stream=stream0)
        del arg37_1
        buf48 = reinterpret_tensor(buf39, (950272, 8), (8, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf47, (950272, 16), (16, 1), 0), reinterpret_tensor(arg38_1, (16, 8), (1, 16), 0), out=buf48)
        del arg38_1
        del buf47
        buf49 = empty_strided_cuda((58, 8, 128, 128, 1, 1), (131072, 16384, 128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf48, arg39_1, buf49, 464, 16384, grid=grid(464, 16384), stream=stream0)
        del arg39_1
        del buf48
        buf51 = reinterpret_tensor(buf34, (464, 128, 16), (2048, 16, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf49, (464, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf50, (464, 128, 16), (2048, 16, 1), 0), out=buf51)
        del buf49
        del buf50
        buf52 = reinterpret_tensor(buf41, (1, 58, 128, 8, 16), (950272, 16384, 128, 16, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_mid], Original ATen: [aten.add]
        triton_poi_fused_add_19.run(buf52, buf51, 950272, grid=grid(950272), stream=stream0)
        buf53 = reinterpret_tensor(buf51, (7424, 128), (128, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf52, (7424, 128), (128, 1), 0), reinterpret_tensor(arg40_1, (128, 128), (1, 128), 0), out=buf53)
        del arg40_1
        buf57 = reinterpret_tensor(buf52, (1, 7424, 128), (950272, 128, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d, z], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_6.run(buf26, buf53, arg41_1, arg42_1, arg43_1, buf57, 7424, 128, grid=grid(7424), stream=stream0)
        del arg42_1
        del arg43_1
        buf58 = reinterpret_tensor(buf45, (7424, 512), (512, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (7424, 128), (128, 1), 0), reinterpret_tensor(arg44_1, (128, 512), (1, 128), 0), out=buf58)
        del arg44_1
        buf59 = reinterpret_tensor(buf58, (1, 7424, 512), (3801088, 512, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_20.run(buf59, arg45_1, 3801088, grid=grid(3801088), stream=stream0)
        del arg45_1
        buf60 = reinterpret_tensor(buf57, (7424, 128), (128, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf59, (7424, 512), (512, 1), 0), reinterpret_tensor(arg46_1, (512, 128), (1, 512), 0), out=buf60)
        del arg46_1
        del buf59
        buf61 = reinterpret_tensor(buf60, (1, 7424, 128), (950272, 128, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [x_mid_d, h_diag_1], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf61, buf26, buf53, arg41_1, arg47_1, 950272, grid=grid(950272), stream=stream0)
        del arg41_1
        del arg47_1
        buf62 = empty_strided_cuda((1, 64, 128, 128), (1048576, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_k_3], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_22.run(buf61, buf62, 1048576, grid=grid(1048576), stream=stream0)
        buf63 = empty_strided_cuda((1, 234, 16384), (3833856, 16384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [row_p_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg30_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf62, (1, 64, 16384), (0, 16384, 1), 0), out=buf63)
        del arg30_1
        buf64 = empty_strided_cuda((1, 234, 16384), (3833856, 16384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [col_p_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg31_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf62, (1, 64, 16384), (1048576, 16384, 1), 0), out=buf64)
        del arg31_1
        del buf62
        buf65 = empty_strided_cuda((234, 32, 128), (4096, 128, 1), torch.float32)
        buf69 = empty_strided_cuda((234, 32, 128), (4096, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_off_3, layer_norm_3], Original ATen: [aten.mean, aten.native_layer_norm]
        triton_per_fused_mean_native_layer_norm_23.run(buf63, buf64, arg48_1, arg49_1, buf65, buf69, 7488, 128, grid=grid(7488), stream=stream0)
        del arg48_1
        del arg49_1
        del buf63
        buf70 = empty_strided_cuda((7488, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf69, (7488, 128), (128, 1), 0), reinterpret_tensor(arg50_1, (128, 384), (1, 128), 0), out=buf70)
        del arg50_1
        buf71 = reinterpret_tensor(buf69, (234, 32, 32, 4), (4096, 128, 4, 1), 0); del buf69  # reuse
        buf72 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [sub], Original ATen: [aten.mean]
        triton_per_fused_mean_24.run(buf72, arg29_1, 958464, 16, grid=grid(958464), stream=stream0)
        del arg29_1
        # Topologically Sorted Source Nodes: [setitem_2], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_25.run(buf72, 29952, grid=grid(29952), stream=stream0)
        buf74 = empty_strided_cuda((234, 1, 32, 32), (1024, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_26.run(buf72, buf74, 239616, grid=grid(239616), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_27.run(buf74, 7488, grid=grid(7488), stream=stream0)
        buf76 = empty_strided_cuda((234, 8, 32, 16), (4096, 512, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, logit_bias_1, x_soft_2], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_28.run(buf70, arg51_1, buf76, 958464, grid=grid(958464), stream=stream0)
        buf77 = empty_strided_cuda((234, 8, 32, 16), (4096, 512, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, logit_bias_1, x_soft_2], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_29.run(buf70, arg51_1, buf77, 958464, grid=grid(958464), stream=stream0)
        buf78 = empty_strided_cuda((234, 8, 32, 16), (4096, 512, 16, 1), torch.float32)
        buf90 = empty_strided_cuda((234, 8, 32, 1, 16, 1), (4096, 512, 16, 16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, logit_bias_1, x_soft_2, einsum_5], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_30.run(buf70, arg51_1, buf78, buf90, 958464, grid=grid(958464), stream=stream0)
        del arg51_1
        del buf70
        buf79 = empty_strided_cuda((234, 8, 32, 32), (8192, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, logit_bias_1, x_soft_2], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention_clone_31.run(buf74, buf72, buf79, 1916928, grid=grid(1916928), stream=stream0)
        # Topologically Sorted Source Nodes: [q_f_1, k_f_1, v_f_1, logit_bias_1, x_soft_2], Original ATen: [aten.clone, aten._scaled_dot_product_efficient_attention]
        buf80 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf76, buf77, buf78, buf79, False)
        del buf76
        del buf77
        buf81 = buf80[0]
        del buf80
        buf85 = reinterpret_tensor(buf78, (234, 1, 32, 32, 4), (4096, 1, 128, 4, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_32.run(buf74, buf72, buf85, 958464, grid=grid(958464), stream=stream0)
        del buf72
        buf86 = reinterpret_tensor(buf64, (239616, 16), (16, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf85, (239616, 4), (4, 1), 0), reinterpret_tensor(arg52_1, (4, 16), (1, 4), 0), out=buf86)
        del arg52_1
        buf87 = reinterpret_tensor(buf86, (234, 1, 32, 32, 16), (16384, 1, 512, 16, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_33.run(buf87, arg53_1, 3833856, grid=grid(3833856), stream=stream0)
        del arg53_1
        buf88 = reinterpret_tensor(buf79, (239616, 8), (8, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (239616, 16), (16, 1), 0), reinterpret_tensor(arg54_1, (16, 8), (1, 16), 0), out=buf88)
        del arg54_1
        buf89 = empty_strided_cuda((234, 8, 32, 32, 1, 1), (8192, 1024, 32, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf88, arg55_1, buf89, 1872, 1024, grid=grid(1872, 1024), stream=stream0)
        del arg55_1
        del buf88
        buf91 = reinterpret_tensor(buf85, (1872, 32, 16), (512, 16, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [einsum_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf89, (1872, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf90, (1872, 32, 16), (512, 16, 1), 0), out=buf91)
        del buf89
        del buf90
        buf92 = reinterpret_tensor(buf81, (234, 1, 32, 8, 16), (4096, 1, 128, 16, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [x_mid_1], Original ATen: [aten.add]
        triton_poi_fused_add_35.run(buf92, buf91, 958464, grid=grid(958464), stream=stream0)
        buf93 = reinterpret_tensor(buf91, (7488, 128), (128, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (7488, 128), (128, 1), 0), reinterpret_tensor(arg56_1, (128, 128), (1, 128), 0), out=buf93)
        del arg56_1
        buf101 = reinterpret_tensor(buf92, (234, 32, 128), (4096, 128, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [x_mid_o, z_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_36.run(buf65, buf93, arg57_1, arg58_1, arg59_1, buf101, 7488, 128, grid=grid(7488), stream=stream0)
        del arg58_1
        del arg59_1
        buf97 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (7424, 128), (128, 1), 0), reinterpret_tensor(arg64_1, (128, 128), (1, 128), 0), out=buf97)
        del arg64_1
        buf98 = reinterpret_tensor(buf97, (1, 58, 128, 128), (950272, 16384, 128, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf98, arg65_1, 950272, grid=grid(950272), stream=stream0)
        del arg65_1
        buf99 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg67_1, reinterpret_tensor(buf98, (7424, 128), (128, 1), 0), reinterpret_tensor(arg66_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf99)
        del arg66_1
        del arg67_1
        buf100 = reinterpret_tensor(buf98, (58, 128, 128), (16384, 128, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf99, (58, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf99, (58, 128, 128), (16384, 1, 128), 0), out=buf100)
        del buf99
        buf102 = reinterpret_tensor(buf87, (7488, 512), (512, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (7488, 128), (128, 1), 0), reinterpret_tensor(arg60_1, (128, 512), (1, 128), 0), out=buf102)
        del arg60_1
        buf103 = reinterpret_tensor(buf102, (234, 32, 512), (16384, 512, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf103, arg61_1, 3833856, grid=grid(3833856), stream=stream0)
        del arg61_1
        buf104 = reinterpret_tensor(buf101, (7488, 128), (128, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (7488, 512), (512, 1), 0), reinterpret_tensor(arg62_1, (512, 128), (1, 512), 0), out=buf104)
        del arg62_1
        del buf103
        buf105 = reinterpret_tensor(buf104, (234, 32, 128), (4096, 128, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [x_mid_o, off_stream], Original ATen: [aten.add]
        triton_poi_fused_add_38.run(buf105, buf65, buf93, arg57_1, arg63_1, 958464, grid=grid(958464), stream=stream0)
        del arg57_1
        del arg63_1
        del buf65
        buf106 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf105, (7488, 128), (128, 1), 0), reinterpret_tensor(arg68_1, (128, 128), (1, 128), 0), out=buf106)
        del arg68_1
        buf107 = reinterpret_tensor(buf106, (234, 1, 32, 128), (4096, 1, 128, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_39.run(buf107, arg69_1, 958464, grid=grid(958464), stream=stream0)
        del arg69_1
        buf108 = reinterpret_tensor(buf74, (7488, 32), (32, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg71_1, reinterpret_tensor(buf107, (7488, 128), (128, 1), 0), reinterpret_tensor(arg70_1, (128, 32), (1, 128), 0), alpha=1, beta=1, out=buf108)
        del arg70_1
        del arg71_1
        buf109 = reinterpret_tensor(buf107, (7488, 128), (128, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf105, (7488, 128), (128, 1), 0), reinterpret_tensor(arg72_1, (128, 128), (1, 128), 0), out=buf109)
        del arg72_1
        del buf105
        buf110 = reinterpret_tensor(buf109, (234, 1, 32, 128), (4096, 1, 128, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_39.run(buf110, arg73_1, 958464, grid=grid(958464), stream=stream0)
        del arg73_1
        buf111 = empty_strided_cuda((7488, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg75_1, reinterpret_tensor(buf110, (7488, 128), (128, 1), 0), reinterpret_tensor(arg74_1, (128, 32), (1, 128), 0), alpha=1, beta=1, out=buf111)
        del arg74_1
        del arg75_1
        del buf110
        buf112 = empty_strided_cuda((234, 32, 32), (1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf108, (234, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf111, (234, 32, 32), (1024, 1, 32), 0), out=buf112)
        del buf108
        del buf111
        buf113 = empty_strided_cuda((7424, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [node_U], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg79_1, reinterpret_tensor(buf61, (7424, 128), (128, 1), 0), reinterpret_tensor(arg78_1, (128, 32), (1, 128), 0), alpha=1, beta=1, out=buf113)
        del arg78_1
        del arg79_1
        buf114 = empty_strided_cuda((7424, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [node_V], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg81_1, reinterpret_tensor(buf61, (7424, 128), (128, 1), 0), reinterpret_tensor(arg80_1, (128, 32), (1, 128), 0), alpha=1, beta=1, out=buf114)
        del arg80_1
        del arg81_1
        buf115 = empty_strided_cuda((7424, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (7424, 128), (128, 1), 0), reinterpret_tensor(arg76_1, (128, 1), (1, 128), 0), out=buf115)
        del arg76_1
        del buf61
        buf116 = reinterpret_tensor(buf115, (1, 58, 128, 1), (7424, 128, 1, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [sigmoid], Original ATen: [aten.sigmoid]
        triton_poi_fused_sigmoid_40.run(buf116, arg77_1, 7424, grid=grid(7424), stream=stream0)
        del arg77_1
        buf117 = empty_strided_cuda((1, 1672448), (1672448, 1), torch.float32)
        # Topologically Sorted Source Nodes: [packed_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_41.run(buf100, buf112, buf113, buf114, buf116, buf117, 1672448, grid=grid(1672448), stream=stream0)
        del buf100
        del buf112
        del buf113
        del buf114
    return (buf117, reinterpret_tensor(buf116, (1, 58, 128), (7424, 128, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 7424, 9), (67716, 9, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 18), (18, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((2, 49401), (49401, 1), device='cuda:0', dtype=torch.int64)
    arg7_1 = rand_strided((49401, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    arg28_1 = rand_strided((58, 128, 128, 4), (65536, 512, 4, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((234, 128, 128, 4), (65536, 512, 4, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((234, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((234, 64), (64, 1), device='cuda:0', dtype=torch.float32)
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
    arg42_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((32, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((32, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((32, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((32, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
