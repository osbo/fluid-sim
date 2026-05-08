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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/wo/cwofmn5tx5wxqog4off6k5kzninp7goa3nxpziotitzvemmhxo7e.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_6 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view, %full_default_32], 1), kwargs = {})
triton_poi_fused_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/rb/crbacxkps2mvvfnsmfa4x3hr2jicroriikz3prnju6nydatm763l.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_7 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute, %full_default_33],), kwargs = {})
triton_poi_fused_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 512)
    x0 = xindex % 512
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 18, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((18*x0) + x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 20, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = 0.0
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.where(tmp4, tmp5, tmp11)
    tl.store(out_ptr0 + (x2), tmp12, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/gn/cgng6rlb5oz3624gvmb357ge6wxb2pcudrukuksgwq2ldg6w2fgx.py
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/xu/cxukeoedvkd4q54xbdugzokpb7rmrwmtrep2mlvrig2e5kuzjn77.py
# Topologically Sorted Source Nodes: [aggr], Original ATen: [aten.zeros_like]
# Source node to ATen node mapping:
#   aggr => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2048, 512], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (1024*x1)), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/kt/cktvljiy5amr3trumsuyitkntyhkvg4vyzhsip6c4vjrps7p2uvc.py
# Topologically Sorted Source Nodes: [aggr, getitem_4, messages, index_add_], Original ATen: [aten.zeros_like, aten.index, aten.mul, aten.index_add]
# Source node to ATen node mapping:
#   aggr => full_default
#   getitem_4 => index
#   index_add_ => index_put
#   messages => mul_3
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2048, 512], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
    xnumel = 6958080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 512)
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (13590 + x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 2048, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 2048)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 2048")
    tmp7 = tmp6 + tmp1
    tmp8 = tmp6 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp6)
    tl.device_assert(((0 <= tmp9) & (tmp9 < 2048)) | ~(xmask), "index out of bounds: 0 <= tmp9 < 2048")
    tmp11 = tl.load(in_ptr1 + (x0 + (512*tmp9)), xmask)
    tmp13 = tmp11 * tmp12
    tl.atomic_add(out_ptr0 + (x0 + (1024*tmp4)), tmp13, xmask, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/xm/cxmt4hrloprnfw46e7az2nx3ga3qjyks3svgckn5knkbj4bice52.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   out => add_2
# Graph fragment:
#   %add_tensor_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_27, %arg15_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze, %add_tensor_27), kwargs = {})
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
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/pt/cptq5pywaxnys2jx2v6fbut6slatayn7peehlby5pzh7xbntxh56.py
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
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_6', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 2048
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/xb/cxbguhfhzm3ak2vog5zoxq3kssmauapwntdifewm64am4tkedrf6.py
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
triton_per_fused_cat_native_layer_norm_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_7', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 2048
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/em/cemhhrjbq2lfezz3sfcbi5uamd5xwqgaerutijwj7adnzbxqbjzv.py
# Topologically Sorted Source Nodes: [block_node], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   block_node => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_8, [2], True), kwargs = {})
triton_per_fused_mean_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/dk/cdkjf6d7mpsmsgfxhgb55yag6ien2iguhw7bdnl5liwwu6x22ni6.py
# Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   einsum => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_15,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/uv/cuv6diwcjckqq4nixohsmr2yftssdxtip5lgti5dy54lkvbgo4xa.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_5 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_17, %full_default_31], 2), kwargs = {})
triton_poi_fused_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/fm/cfmds5jwty25hg6o2c3gw7x5svk7rfdnfiitwkrtb533xxz63ngf.py
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
#   %select_scatter_default_4 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%expand_4, %copy_2, 3, 32), kwargs = {})
#   %full_default_9 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_3 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_23, %full_default_9), kwargs = {})
#   %select_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_1, %copy_3, 3, 3), kwargs = {})
#   %select_scatter_default_6 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_4, %select_scatter_default_5, 3, 32), kwargs = {})
triton_poi_fused_fill_lift_fresh_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_fill_lift_fresh_11', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 270336
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/j6/cj6db4kijtxywzk26e3dbfus2a7wmqndtbvgnrd6xo3nyui4tynh.py
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
triton_poi_fused_fill_index_put_lift_fresh_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_fill_index_put_lift_fresh_12', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 4
    x1 = (xindex // 4) % 32
    x2 = (xindex // 128)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (136*x1) + (4224*x2)), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/lu/clu7o5mcpra5g3ok3f5if4wl5okofh46pfbk7pnz2jqhcdzvexyh.py
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
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_13', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + (4*x0)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/tu/ctuapit6ojwodwgmfmji5ymsjxhm2tcmxamy5usgckzjhxenkfty.py
# Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_3 => full_default_5, index_put_3
# Graph fragment:
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_3 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_15, [None, None, %iota, %iota], %full_default_5), kwargs = {})
triton_poi_fused_index_put_lift_fresh_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_14', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp0 = 1.0
    tl.store(out_ptr0 + ((34*x0) + (1056*x1)), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/4v/c4v36nsvcn4m443xvt7vt3erytg43pboiesy7ixejjnqun4aevl3.py
# Topologically Sorted Source Nodes: [eq, scores_2, scores, scores_1, attn_probs], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_probs => amax, exp, sub_2
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
triton_per_fused__softmax_add_eq_masked_fill_mul_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_eq_masked_fill_mul_15', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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
    tl.store(out_ptr1 + (r3 + (33*x4)), tmp19, rmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/to/ctosteog274jjmh3np2c3lpaay3zlqimrgb7pmppwelap6towabc.py
# Topologically Sorted Source Nodes: [attn_probs], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_probs => sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [3], True), kwargs = {})
triton_per_fused__softmax_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_16', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 33
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 8
    x1 = (xindex // 8) % 32
    x2 = (xindex // 256)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (33*x1) + (1056*x0) + (8448*x2)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/j5/cj5r6ykqxnlf6545tyrtdjrlfi3clovksx4g5tx2xobvsjr4uelf.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %select_scatter_default_3 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%index_put_2, %index_put_3, 4, 3), kwargs = {})
triton_poi_fused_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_17', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 270336
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/yr/cyrn3hzdznpucc6nffe2wf4siwhs7mpvv5cl7wsrl277cauwugwu.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_23, %full_default_29], 2), kwargs = {})
triton_poi_fused_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[512, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_18', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/gj/cgjfecjc653nnwhhffkjjqmm3pxeyp7kctoeevdsh6wnrdghvwll.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_29 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([512, 32, 3], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(1,), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_19', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 3
    x1 = (xindex // 3)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (36*x1)), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/cs/ccszeqb74evanbfzxilceozaretxh5omwmmcuixg5zh7tshysjwb.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default_4 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_24, %full_default_30], 1), kwargs = {})
triton_poi_fused_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_20', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/xr/cxrnou7rxwmmd4yepu2lcs2zmyoo3i54ifqxzfcrwyf3pht7ika2.py
# Topologically Sorted Source Nodes: [x_out_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_out_1 => clone_6
# Graph fragment:
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_26,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_21', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/7u/c7upsy2emy6tu36pvccbavfuovvivr6k644dx7l7ebg2tok2d465.py
# Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_11 => add_14, erf_3, mul_18, mul_19, mul_20
# Graph fragment:
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_32, 0.5), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_32, 0.7071067811865476), kwargs = {})
#   %erf_3 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_19,), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_3, 1), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %add_14), kwargs = {})
triton_poi_fused_gelu_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_22', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/b5/cb53vq5utpxx2e46x33b5upi37e6c3gxyudfubasnwbj77ye6mec.py
# Topologically Sorted Source Nodes: [x, x_1, layer_norm_3, kv_1], Original ATen: [aten.add, aten.native_layer_norm, aten.cat]
# Source node to ATen node mapping:
#   kv_1 => cat_4
#   layer_norm_3 => add_16, add_17, mul_21, mul_22, rsqrt_3, sub_4, var_mean_3
#   x => add_11
#   x_1 => add_15
# Graph fragment:
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %view_30), kwargs = {})
#   %add_15 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_34), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_15, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_15, %getitem_7), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_16,), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %arg46_1), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %arg47_1), kwargs = {})
#   %cat_4 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_35, %mean_1], 2), kwargs = {})
triton_per_fused_add_cat_native_layer_norm_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_23', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 2048
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
    tl.store(out_ptr3 + (r1 + (512*x2) + (16896*x3)), tmp32, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/wa/cwa7dmhjhmaeogbshqsk3olo6jkcn3j7mryv6n52k6b4roqxzuxf.py
# Topologically Sorted Source Nodes: [layer_norm_5, kv_2], Original ATen: [aten.native_layer_norm, aten.cat]
# Source node to ATen node mapping:
#   kv_2 => cat_5
#   layer_norm_5 => add_26, add_27, mul_29, mul_30, rsqrt_5, sub_7, var_mean_5
# Graph fragment:
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_77, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_77, %getitem_11), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_26,), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_5), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %arg64_1), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %arg65_1), kwargs = {})
#   %cat_5 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_78, %mean_2], 2), kwargs = {})
triton_per_fused_cat_native_layer_norm_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_24', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 7488
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/4b/c4b3mb4di66ozr5kf7heyapjwdygloozzmqpjcfi4ckel7s2qjk4.py
# Topologically Sorted Source Nodes: [block_node_2], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   block_node_2 => mean_2
# Graph fragment:
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_78, [2], True), kwargs = {})
triton_per_fused_mean_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[131072, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_25', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 119808
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ty/cty4uqqmfm464c2lblmn7vypfeworni42zf7xnbb2wfzbhghg4g7.py
# Topologically Sorted Source Nodes: [einsum_6], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   einsum_6 => clone_14
# Graph fragment:
#   %clone_14 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_59,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_26', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3833856
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/x2/cx2gb67grwy6z2yram5ywr2rj5ukfbng7fxufpb75jaditl2ebyq.py
# Topologically Sorted Source Nodes: [einsum_6], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   einsum_6 => clone_15
# Graph fragment:
#   %clone_15 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_60,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_27', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 119808
    xnumel = 33
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (512 + y0 + (1536*x2) + (50688*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (512 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (33*y3)), tmp2, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/vi/cvi2cqmwm4gs63dzd67wbfdyd3qfjiax3lf6a72ydddtbbazo7wn.py
# Topologically Sorted Source Nodes: [setitem_8, setitem_9, setitem_12, setitem_13], Original ATen: [aten.lift_fresh, aten.fill]
# Source node to ATen node mapping:
#   setitem_12 => copy_6, full_default_20
#   setitem_13 => copy_7, full_default_21
#   setitem_8 => copy_4, full_default_14
#   setitem_9 => copy_5, full_default_15
# Graph fragment:
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_4 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_30, %full_default_14), kwargs = {})
#   %select_scatter_default_8 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%view_7, %copy_4, 3, 32), kwargs = {})
#   %full_default_15 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_5 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_36, %full_default_15), kwargs = {})
#   %select_scatter_default_9 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_2, %copy_5, 3, 3), kwargs = {})
#   %select_scatter_default_10 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_8, %select_scatter_default_9, 3, 32), kwargs = {})
#   %full_default_20 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_6 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_43, %full_default_20), kwargs = {})
#   %select_scatter_default_12 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%view_7, %copy_6, 3, 32), kwargs = {})
#   %full_default_21 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_7 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_49, %full_default_21), kwargs = {})
#   %select_scatter_default_13 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_3, %copy_7, 3, 3), kwargs = {})
#   %select_scatter_default_14 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_12, %select_scatter_default_13, 3, 32), kwargs = {})
triton_poi_fused_fill_lift_fresh_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_fill_lift_fresh_28', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 988416
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/sa/csauhhfcln65l4hjocyp2iqhqutrwjoustt4rqaqxmc6nncnskzf.py
# Topologically Sorted Source Nodes: [setitem_8, setitem_9, setitem_10], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
# Source node to ATen node mapping:
#   setitem_10 => full_default_16, index_put_6
#   setitem_8 => copy_4, full_default_14
#   setitem_9 => copy_5, full_default_15
# Graph fragment:
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_4 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_30, %full_default_14), kwargs = {})
#   %select_scatter_default_8 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%view_7, %copy_4, 3, 32), kwargs = {})
#   %full_default_15 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_5 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_36, %full_default_15), kwargs = {})
#   %select_scatter_default_9 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_2, %copy_5, 3, 3), kwargs = {})
#   %select_scatter_default_10 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_8, %select_scatter_default_9, 3, 32), kwargs = {})
#   %full_default_16 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_6 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%select_scatter_default_10, [None, None, %iota_2, %iota_2], %full_default_16), kwargs = {})
triton_poi_fused_fill_index_put_lift_fresh_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_fill_index_put_lift_fresh_29', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tl.store(out_ptr0 + (x0 + (136*x1) + (4224*x2)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/wc/cwccwfm3z5o6bmytyehavh7lhq7zudcedkzhxfsokgvfy4va4mcn.py
# Topologically Sorted Source Nodes: [setitem_11], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_11 => full_default_17, index_put_7
# Graph fragment:
#   %full_default_17 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_7 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_41, [None, None, %iota_2, %iota_2], %full_default_17), kwargs = {})
triton_poi_fused_index_put_lift_fresh_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_30', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 247104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + (4*x0)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/sf/csfzdtgt3oj2dbfl3h2patmqmrbxoonnnmb5fotlr2onhlha7njz.py
# Topologically Sorted Source Nodes: [setitem_11], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_11 => full_default_17, index_put_7
# Graph fragment:
#   %full_default_17 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_7 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_41, [None, None, %iota_2, %iota_2], %full_default_17), kwargs = {})
triton_poi_fused_index_put_lift_fresh_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_31', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tl.store(out_ptr0 + ((34*x0) + (1056*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/yx/cyxx6g4oixaj2qf32wzw54heqiaeg3z2qcei64zn4j6azoupt47r.py
# Topologically Sorted Source Nodes: [eq_4, scores_8, scores_6, scores_7, attn_probs_2], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_probs_2 => amax_2, exp_2, sub_8
#   eq_4 => eq_4
#   scores_6 => mul_31
#   scores_7 => add_28
#   scores_8 => clone_17, full_default_18, where_4
# Graph fragment:
#   %eq_4 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%unsqueeze_31, 0), kwargs = {})
#   %full_default_18 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_89, 0.125), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %slice_142), kwargs = {})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_4, %full_default_18, %add_28), kwargs = {})
#   %clone_17 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%where_4,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_2 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_17, [3], True), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_17, %amax_2), kwargs = {})
#   %exp_2 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_8,), kwargs = {})
triton_per_fused__softmax_add_eq_masked_fill_mul_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_eq_masked_fill_mul_32', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 59904
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
    tmp3 = tl.load(in_ptr1 + (r3 + (33*x4)), rmask & xmask, other=0.0)
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/5m/c5m77x2f3x3cueraet4f32eupua2yp446mposha7wmtiqlfqlmps.py
# Topologically Sorted Source Nodes: [attn_probs_2], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_probs_2 => sum_3
# Graph fragment:
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_2, [3], True), kwargs = {})
triton_per_fused__softmax_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_33', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 59904
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/kd/ckdfbx3g6xmc3t4xqoaqmczgt37zdjr24vqucqtq3urpk2y7bep3.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %select_scatter_default_11 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%index_put_6, %index_put_7, 4, 3), kwargs = {})
triton_poi_fused_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_34', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 988416
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/mi/cmibekuv62kic6yf6f6y5yky5theqowim27p2duffpxsfhe5iquo.py
# Topologically Sorted Source Nodes: [x_out_8], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_out_8 => clone_18
# Graph fragment:
#   %clone_18 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_65,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_35', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1872
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
    y0 = yindex % 8
    y1 = (yindex // 8)
    tmp0 = tl.load(in_out_ptr0 + (x5 + (1056*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (8*x3) + (256*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x5 + (1056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (y0 + (8*x5) + (8448*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 / tmp1
    tmp4 = 0.0
    tmp5 = tmp3 == tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = tl.where(tmp5, tmp4, tmp8)
    tmp10 = tmp2 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5 + (1056*y4)), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ze/czezdjlaa6rez6eyizfrflu3sbxhbvzidpqi2kyaqslmn6ilrwvy.py
# Topologically Sorted Source Nodes: [x_out_8], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_out_8 => clone_19
# Graph fragment:
#   %clone_19 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_66,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_36', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3953664
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/s4/cs43h5gf7b3l666qcnhy4mea7m2ugi2h2pzp7db5tdzu2unxujks.py
# Topologically Sorted Source Nodes: [x_out_9], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_out_9 => clone_20
# Graph fragment:
#   %clone_20 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_96,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_37', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3833856
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ht/cht6iq56b2q6ajycphzrqnlbcefjbobnziebed5exig3djjxalkx.py
# Topologically Sorted Source Nodes: [x_4, layer_norm_6], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_6 => add_31, add_32, mul_32, mul_33, rsqrt_6, sub_9, var_mean_6
#   x_4 => add_30
# Graph fragment:
#   %add_30 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_77, %view_100), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_30, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_30, %getitem_13), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-05), kwargs = {})
#   %rsqrt_6 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_31,), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %rsqrt_6), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %arg72_1), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %arg73_1), kwargs = {})
triton_per_fused_add_native_layer_norm_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_38', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 7488
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
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/6h/c6h5v2334znczrwgypg5jyva5hjleqfouplrnggcizm2l4oyfbkt.py
# Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_17 => add_33, erf_5, mul_34, mul_35, mul_36
# Graph fragment:
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_102, 0.5), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_102, 0.7071067811865476), kwargs = {})
#   %erf_5 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_35,), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_5, 1), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %add_33), kwargs = {})
triton_poi_fused_gelu_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_39', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15335424
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ro/croow7h76laqbxtezq4dou2ay6ztrnb5sligrieqhcnxe4ts3pzh.py
# Topologically Sorted Source Nodes: [x_4, x_5, layer_norm_7, kv_3], Original ATen: [aten.add, aten.native_layer_norm, aten.cat]
# Source node to ATen node mapping:
#   kv_3 => cat_6
#   layer_norm_7 => add_35, add_36, mul_37, mul_38, rsqrt_7, sub_10, var_mean_7
#   x_4 => add_30
#   x_5 => add_34
# Graph fragment:
#   %add_30 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_77, %view_100), kwargs = {})
#   %add_34 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_30, %view_104), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_34, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_34, %getitem_15), kwargs = {})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-05), kwargs = {})
#   %rsqrt_7 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_35,), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %rsqrt_7), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %arg78_1), kwargs = {})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %arg79_1), kwargs = {})
#   %cat_6 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_105, %mean_3], 2), kwargs = {})
triton_per_fused_add_cat_native_layer_norm_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_40', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 7488
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
    tmp7 = tl.load(in_out_ptr0 + (r1 + (512*x0)), None)
    tmp8 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr6 + (r1), None, eviction_policy='evict_last')
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
    tmp24 = tmp10 - tmp18
    tmp25 = 512.0
    tmp26 = tmp23 / tmp25
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp24 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp10, None)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp34, None)
    tl.store(out_ptr3 + (r1 + (512*x2) + (16896*x3)), tmp34, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/bw/cbwtnrggi5jjk7ntbh5fyqkpostq63v2bn6nbhxdfi6yoyhyeciu.py
# Topologically Sorted Source Nodes: [x_6, layer_norm_8], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_8 => add_40, add_41, mul_40, mul_41, rsqrt_8, sub_12, var_mean_8
#   x_6 => add_39
# Graph fragment:
#   %add_39 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_34, %view_127), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_39, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_39, %getitem_17), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_8 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_40,), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_8), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %arg86_1), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %arg87_1), kwargs = {})
triton_per_fused_add_native_layer_norm_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_41', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 7488
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/qr/cqrorzj3tpfkl36iluelubkqwd6riiksmboc3zvwkv2emcgop5pg.py
# Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_2 => add_20
#   x_3 => add_24
# Graph fragment:
#   %add_20 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %view_57), kwargs = {})
#   %add_24 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %view_61), kwargs = {})
triton_poi_fused_add_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_42', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/2q/c2qg6byyoqprd2j7oq3prlql2qixbmaksa2kiyx5z6vb75glnyym.py
# Topologically Sorted Source Nodes: [x_6, x_7], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_6 => add_39
#   x_7 => add_43
# Graph fragment:
#   %add_39 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_34, %view_127), kwargs = {})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_39, %view_131), kwargs = {})
triton_poi_fused_add_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_43', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3833856
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/mj/cmj7ylozss4hez2xzk7o7f4vsl3xloosqztlnxyjd2pwy7ekmxvm.py
# Topologically Sorted Source Nodes: [sigmoid], Original ATen: [aten.sigmoid]
# Source node to ATen node mapping:
#   sigmoid => sigmoid
# Graph fragment:
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_145,), kwargs = {})
triton_poi_fused_sigmoid_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_44', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/42/c42kzwdvgrlmdarbjp7cagqx727yh6tj7lydj3wlhb2lvrid5yto.py
# Topologically Sorted Source Nodes: [packed_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   packed_1 => cat_8
# Graph fragment:
#   %cat_8 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_7, %view_146], 1), kwargs = {})
triton_poi_fused_cat_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_45', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 438272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 436224, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 65536, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.load(in_ptr0 + (x0), tmp7, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp0 >= tmp5
    tmp10 = tl.full([1], 305152, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tmp12 & tmp4
    tmp14 = tl.load(in_ptr1 + ((-65536) + x0), tmp13, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp0 >= tmp10
    tmp16 = tl.full([1], 370688, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr2 + ((-305152) + x0), tmp19, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp0 >= tmp16
    tmp22 = tmp21 & tmp4
    tmp23 = tl.load(in_ptr3 + ((-370688) + x0), tmp22, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp18, tmp20, tmp23)
    tmp25 = tl.where(tmp12, tmp14, tmp24)
    tmp26 = tl.where(tmp6, tmp8, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp4, tmp26, tmp27)
    tmp29 = tmp0 >= tmp3
    tmp30 = tl.full([1], 438272, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = tl.load(in_ptr4 + ((-436224) + x0), tmp29, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.where(tmp4, tmp28, tmp32)
    tl.store(out_ptr0 + (x0), tmp33, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 2048, 9), (67716, 9, 1))
    assert_size_stride(arg1_1, (1, 12), (12, 1))
    assert_size_stride(arg2_1, (512, 18), (18, 1))
    assert_size_stride(arg3_1, (512, ), (1, ))
    assert_size_stride(arg4_1, (512, 512), (512, 1))
    assert_size_stride(arg5_1, (512, ), (1, ))
    assert_size_stride(arg6_1, (2, 13590), (13590, 1))
    assert_size_stride(arg7_1, (13590, ), (1, ))
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
    assert_size_stride(arg28_1, (64, 32, 33, 4), (4224, 132, 4, 1))
    assert_size_stride(arg29_1, (64, 32, 33), (1056, 33, 1))
    assert_size_stride(arg30_1, (234, 32, 33, 4), (4224, 132, 4, 1))
    assert_size_stride(arg31_1, (234, 32, 33), (1056, 33, 1))
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
    assert_size_stride(arg60_1, (32, 512), (512, 1))
    assert_size_stride(arg61_1, (32, ), (1, ))
    assert_size_stride(arg62_1, (234, 64), (64, 1))
    assert_size_stride(arg63_1, (234, 64), (64, 1))
    assert_size_stride(arg64_1, (512, ), (1, ))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (1536, 512), (512, 1))
    assert_size_stride(arg67_1, (1536, ), (1, ))
    assert_size_stride(arg68_1, (8, 4), (4, 1))
    assert_size_stride(arg69_1, (8, ), (1, ))
    assert_size_stride(arg70_1, (512, 512), (512, 1))
    assert_size_stride(arg71_1, (512, ), (1, ))
    assert_size_stride(arg72_1, (512, ), (1, ))
    assert_size_stride(arg73_1, (512, ), (1, ))
    assert_size_stride(arg74_1, (2048, 512), (512, 1))
    assert_size_stride(arg75_1, (2048, ), (1, ))
    assert_size_stride(arg76_1, (512, 2048), (2048, 1))
    assert_size_stride(arg77_1, (512, ), (1, ))
    assert_size_stride(arg78_1, (512, ), (1, ))
    assert_size_stride(arg79_1, (512, ), (1, ))
    assert_size_stride(arg80_1, (1536, 512), (512, 1))
    assert_size_stride(arg81_1, (1536, ), (1, ))
    assert_size_stride(arg82_1, (8, 4), (4, 1))
    assert_size_stride(arg83_1, (8, ), (1, ))
    assert_size_stride(arg84_1, (512, 512), (512, 1))
    assert_size_stride(arg85_1, (512, ), (1, ))
    assert_size_stride(arg86_1, (512, ), (1, ))
    assert_size_stride(arg87_1, (512, ), (1, ))
    assert_size_stride(arg88_1, (2048, 512), (512, 1))
    assert_size_stride(arg89_1, (2048, ), (1, ))
    assert_size_stride(arg90_1, (512, 2048), (2048, 1))
    assert_size_stride(arg91_1, (512, ), (1, ))
    assert_size_stride(arg92_1, (32, 512), (512, 1))
    assert_size_stride(arg93_1, (32, ), (1, ))
    assert_size_stride(arg94_1, (32, 512), (512, 1))
    assert_size_stride(arg95_1, (32, ), (1, ))
    assert_size_stride(arg96_1, (1, 512), (512, 1))
    assert_size_stride(arg97_1, (1, ), (1, ))
    assert_size_stride(arg98_1, (32, 512), (512, 1))
    assert_size_stride(arg99_1, (32, ), (1, ))
    assert_size_stride(arg100_1, (32, 512), (512, 1))
    assert_size_stride(arg101_1, (32, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2048, 20), (20, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(arg0_1, arg1_1, buf0, 40960, grid=grid(40960), stream=stream0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((20, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(arg2_1, buf1, 10240, grid=grid(10240), stream=stream0)
        del arg2_1
        buf2 = empty_strided_cuda((2048, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf0, buf1, out=buf2)
        del buf0
        del buf1
        buf3 = reinterpret_tensor(buf2, (1, 2048, 512), (1048576, 512, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf3, arg3_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg3_1
        buf4 = empty_strided_cuda((2048, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg5_1, reinterpret_tensor(buf3, (2048, 512), (512, 1), 0), reinterpret_tensor(arg4_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf4)
        del arg4_1
        del arg5_1
        buf9 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float32)
        buf5 = reinterpret_tensor(buf9, (2048, 512), (1024, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [self_features], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, buf4, reinterpret_tensor(arg10_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf5)
        del arg10_1
        del arg11_1
        buf6 = reinterpret_tensor(buf3, (2048, 512), (512, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [neighbor_features], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg9_1, buf4, reinterpret_tensor(arg8_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf6)
        del arg8_1
        del arg9_1
        buf7 = reinterpret_tensor(buf9, (2048, 512), (1024, 1), 512)  # alias
        # Topologically Sorted Source Nodes: [aggr], Original ATen: [aten.zeros_like]
        triton_poi_fused_zeros_like_3.run(buf7, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [aggr, getitem_4, messages, index_add_], Original ATen: [aten.zeros_like, aten.index, aten.mul, aten.index_add]
        triton_poi_fused_index_index_add_mul_zeros_like_4.run(arg6_1, buf6, arg7_1, buf7, 6958080, grid=grid(6958080), stream=stream0)
        del buf5
        del buf7
        buf10 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf9, reinterpret_tensor(arg12_1, (1024, 512), (1, 1024), 0), out=buf10)
        del arg12_1
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf11, arg13_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg13_1
        buf12 = empty_strided_cuda((2048, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.gelu]
        extern_kernels.mm(buf11, reinterpret_tensor(arg14_1, (512, 512), (1, 512), 0), out=buf12)
        del arg14_1
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.add]
        triton_poi_fused_add_5.run(buf13, buf4, arg15_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg15_1
        buf18 = buf9; del buf9  # reuse
        buf14 = reinterpret_tensor(buf18, (2048, 512), (1024, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [self_features_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg19_1, buf13, reinterpret_tensor(arg18_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf14)
        del arg18_1
        del arg19_1
        buf15 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [neighbor_features_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg17_1, buf13, reinterpret_tensor(arg16_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf15)
        del arg16_1
        del arg17_1
        buf16 = reinterpret_tensor(buf18, (2048, 512), (1024, 1), 512)  # alias
        # Topologically Sorted Source Nodes: [aggr_1], Original ATen: [aten.zeros_like]
        triton_poi_fused_zeros_like_3.run(buf16, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [aggr_1, getitem_7, messages_1, index_add__1], Original ATen: [aten.zeros_like, aten.index, aten.mul, aten.index_add]
        triton_poi_fused_index_index_add_mul_zeros_like_4.run(arg6_1, buf15, arg7_1, buf16, 6958080, grid=grid(6958080), stream=stream0)
        del arg6_1
        del arg7_1
        del buf14
        del buf16
        buf19 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf18, reinterpret_tensor(arg20_1, (1024, 512), (1, 1024), 0), out=buf19)
        del arg20_1
        del buf18
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf20, arg21_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg21_1
        buf21 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.gelu]
        extern_kernels.mm(buf20, reinterpret_tensor(arg22_1, (512, 512), (1, 512), 0), out=buf21)
        del arg22_1
        buf25 = reinterpret_tensor(buf20, (1, 2048, 512), (1048576, 512, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [h_2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_6.run(buf13, buf21, arg23_1, arg24_1, arg25_1, buf25, 2048, 512, grid=grid(2048), stream=stream0)
        del arg23_1
        del arg24_1
        del arg25_1
        buf26 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [h_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg27_1, reinterpret_tensor(buf25, (2048, 512), (512, 1), 0), reinterpret_tensor(arg26_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf26)
        del arg26_1
        del arg27_1
        buf30 = buf25; del buf25  # reuse
        buf35 = empty_strided_cuda((1, 64, 33, 512), (1081344, 16896, 512, 1), torch.float32)
        buf33 = reinterpret_tensor(buf35, (1, 64, 32, 512), (1081344, 16896, 512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [layer_norm_1, kv], Original ATen: [aten.native_layer_norm, aten.cat]
        triton_per_fused_cat_native_layer_norm_7.run(buf26, arg32_1, arg33_1, buf30, buf33, 2048, 512, grid=grid(2048), stream=stream0)
        del arg32_1
        del arg33_1
        buf31 = empty_strided_cuda((2048, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (2048, 512), (512, 1), 0), reinterpret_tensor(arg34_1, (512, 1536), (1, 512), 0), out=buf31)
        buf34 = reinterpret_tensor(buf35, (1, 64, 1, 512), (1081344, 16896, 512, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [block_node], Original ATen: [aten.mean]
        triton_per_fused_mean_8.run(buf30, buf34, 32768, 32, grid=grid(32768), stream=stream0)
        del buf33
        del buf34
        buf36 = empty_strided_cuda((2112, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf35, (2112, 512), (512, 1), 0), reinterpret_tensor(arg34_1, (512, 1536), (1, 512), 0), out=buf36)
        del arg34_1
        buf37 = reinterpret_tensor(buf30, (64, 8, 32, 64, 1, 1), (16384, 2048, 64, 1, 1, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf31, arg35_1, buf37, 1048576, grid=grid(1048576), stream=stream0)
        buf38 = empty_strided_cuda((512, 64, 36), (2304, 36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(buf36, arg35_1, buf38, 1179648, grid=grid(1179648), stream=stream0)
        buf39 = empty_strided_cuda((512, 32, 36), (1152, 36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf37, (512, 32, 64), (2048, 64, 1), 0), buf38, out=buf39)
        buf40 = empty_strided_cuda((1, 64, 32, 33, 4), (270336, 4224, 132, 4, 1), torch.float32)
        buf77 = empty_strided_cuda((1, 64, 32, 33, 4), (270336, 4224, 132, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem, setitem_1, setitem_4, setitem_5], Original ATen: [aten.lift_fresh, aten.fill]
        triton_poi_fused_fill_lift_fresh_11.run(arg28_1, buf40, buf77, 270336, grid=grid(270336), stream=stream0)
        del arg28_1
        # Topologically Sorted Source Nodes: [setitem, setitem_1, setitem_2], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
        triton_poi_fused_fill_index_put_lift_fresh_12.run(buf40, 8192, grid=grid(8192), stream=stream0)
        buf42 = empty_strided_cuda((1, 64, 32, 33), (67584, 1056, 33, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_13.run(buf40, buf42, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_14.run(buf42, 2048, grid=grid(2048), stream=stream0)
        buf45 = empty_strided_cuda((1, 64, 32, 33, 8), (540672, 8448, 33, 1, 1056), torch.float32)
        # Topologically Sorted Source Nodes: [eq, scores_2, scores, scores_1, attn_probs], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_eq_masked_fill_mul_15.run(arg29_1, buf39, buf42, buf40, buf45, 16384, 33, grid=grid(16384), stream=stream0)
        buf46 = empty_strided_cuda((1, 64, 32, 1, 8), (16384, 256, 8, 16384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_probs], Original ATen: [aten._softmax]
        triton_per_fused__softmax_16.run(buf45, buf46, 16384, 33, grid=grid(16384), stream=stream0)
        buf47 = empty_strided_cuda((1, 64, 32, 33, 4), (270336, 4224, 132, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_17.run(buf42, buf40, buf47, 270336, grid=grid(270336), stream=stream0)
        del buf40
        buf48 = empty_strided_cuda((67584, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf47, (67584, 4), (4, 1), 0), reinterpret_tensor(arg36_1, (4, 8), (1, 4), 0), out=buf48)
        del arg36_1
        buf51 = buf39; del buf39  # reuse
        buf49 = reinterpret_tensor(buf51, (512, 32, 33), (1152, 36, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_18.run(buf45, buf46, arg29_1, buf48, arg37_1, buf49, 512, 1056, grid=grid(512, 1056), stream=stream0)
        del arg37_1
        buf50 = reinterpret_tensor(buf51, (512, 32, 3), (1152, 36, 1), 33)  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_19.run(buf50, 49152, grid=grid(49152), stream=stream0)
        buf52 = reinterpret_tensor(buf38, (512, 36, 64), (2304, 64, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_20.run(buf36, arg35_1, buf52, 1179648, grid=grid(1179648), stream=stream0)
        del arg35_1
        del buf49
        del buf50
        buf53 = reinterpret_tensor(buf37, (512, 32, 64), (2048, 64, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(buf51, buf52, out=buf53)
        buf54 = reinterpret_tensor(buf13, (1, 64, 32, 8, 64), (1048576, 16384, 512, 64, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [x_out_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf53, buf54, 1048576, grid=grid(1048576), stream=stream0)
        buf55 = reinterpret_tensor(buf53, (2048, 512), (512, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf54, (2048, 512), (512, 1), 0), reinterpret_tensor(arg38_1, (512, 512), (1, 512), 0), out=buf55)
        del arg38_1
        buf59 = reinterpret_tensor(buf54, (1, 2048, 512), (1048576, 512, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [x, layer_norm_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_6.run(buf26, buf55, arg39_1, arg40_1, arg41_1, buf59, 2048, 512, grid=grid(2048), stream=stream0)
        del arg40_1
        del arg41_1
        buf60 = empty_strided_cuda((2048, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf59, (2048, 512), (512, 1), 0), reinterpret_tensor(arg42_1, (512, 2048), (1, 512), 0), out=buf60)
        del arg42_1
        buf61 = reinterpret_tensor(buf60, (1, 2048, 2048), (4194304, 2048, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_22.run(buf61, arg43_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg43_1
        buf62 = reinterpret_tensor(buf59, (2048, 512), (512, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (2048, 2048), (2048, 1), 0), reinterpret_tensor(arg44_1, (2048, 512), (1, 2048), 0), out=buf62)
        del arg44_1
        buf63 = reinterpret_tensor(buf62, (1, 2048, 512), (1048576, 512, 1), 0); del buf62  # reuse
        buf67 = empty_strided_cuda((1, 2048, 512), (1048576, 512, 1), torch.float32)
        buf72 = buf35; del buf35  # reuse
        buf70 = reinterpret_tensor(buf72, (1, 64, 32, 512), (1081344, 16896, 512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x, x_1, layer_norm_3, kv_1], Original ATen: [aten.add, aten.native_layer_norm, aten.cat]
        triton_per_fused_add_cat_native_layer_norm_23.run(buf63, buf26, buf55, arg39_1, arg45_1, arg46_1, arg47_1, buf67, buf70, 2048, 512, grid=grid(2048), stream=stream0)
        del arg39_1
        del arg45_1
        del arg46_1
        del arg47_1
        buf68 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf67, (2048, 512), (512, 1), 0), reinterpret_tensor(arg48_1, (512, 1536), (1, 512), 0), out=buf68)
        buf71 = reinterpret_tensor(buf72, (1, 64, 1, 512), (1081344, 16896, 512, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [block_node_1], Original ATen: [aten.mean]
        triton_per_fused_mean_8.run(buf67, buf71, 32768, 32, grid=grid(32768), stream=stream0)
        del buf70
        del buf71
        buf73 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (2112, 512), (512, 1), 0), reinterpret_tensor(arg48_1, (512, 1536), (1, 512), 0), out=buf73)
        del arg48_1
        del buf72
        buf74 = reinterpret_tensor(buf67, (64, 8, 32, 64, 1, 1), (16384, 2048, 64, 1, 1, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf68, arg49_1, buf74, 1048576, grid=grid(1048576), stream=stream0)
        del buf68
        buf75 = reinterpret_tensor(buf52, (512, 64, 36), (2304, 36, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(buf73, arg49_1, buf75, 1179648, grid=grid(1179648), stream=stream0)
        buf76 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf74, (512, 32, 64), (2048, 64, 1), 0), buf75, out=buf76)
        # Topologically Sorted Source Nodes: [setitem_4, setitem_5, setitem_6], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
        triton_poi_fused_fill_index_put_lift_fresh_12.run(buf77, 8192, grid=grid(8192), stream=stream0)
        buf79 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [setitem_7], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_13.run(buf77, buf79, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_7], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_14.run(buf79, 2048, grid=grid(2048), stream=stream0)
        buf82 = reinterpret_tensor(buf48, (1, 64, 32, 33, 8), (540672, 8448, 33, 1, 1056), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [eq_2, scores_5, scores_3, scores_4, attn_probs_1], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_eq_masked_fill_mul_15.run(arg29_1, buf76, buf79, buf77, buf82, 16384, 33, grid=grid(16384), stream=stream0)
        buf83 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [attn_probs_1], Original ATen: [aten._softmax]
        triton_per_fused__softmax_16.run(buf82, buf83, 16384, 33, grid=grid(16384), stream=stream0)
        buf84 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_17.run(buf79, buf77, buf84, 270336, grid=grid(270336), stream=stream0)
        del buf77
        del buf79
        buf85 = reinterpret_tensor(buf45, (67584, 8), (8, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf84, (67584, 4), (4, 1), 0), reinterpret_tensor(arg50_1, (4, 8), (1, 4), 0), out=buf85)
        del arg50_1
        del buf84
        buf88 = buf76; del buf76  # reuse
        buf86 = reinterpret_tensor(buf88, (512, 32, 33), (1152, 36, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_18.run(buf82, buf83, arg29_1, buf85, arg51_1, buf86, 512, 1056, grid=grid(512, 1056), stream=stream0)
        del arg29_1
        del arg51_1
        del buf82
        del buf83
        del buf85
        buf87 = reinterpret_tensor(buf88, (512, 32, 3), (1152, 36, 1), 33)  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_19.run(buf87, 49152, grid=grid(49152), stream=stream0)
        buf89 = reinterpret_tensor(buf75, (512, 36, 64), (2304, 64, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_20.run(buf73, arg49_1, buf89, 1179648, grid=grid(1179648), stream=stream0)
        del arg49_1
        del buf73
        del buf86
        del buf87
        buf90 = reinterpret_tensor(buf74, (512, 32, 64), (2048, 64, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(buf88, buf89, out=buf90)
        del buf88
        del buf89
        buf91 = reinterpret_tensor(buf55, (1, 64, 32, 8, 64), (1048576, 16384, 512, 64, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [x_out_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf90, buf91, 1048576, grid=grid(1048576), stream=stream0)
        buf92 = reinterpret_tensor(buf90, (2048, 512), (512, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf91, (2048, 512), (512, 1), 0), reinterpret_tensor(arg52_1, (512, 512), (1, 512), 0), out=buf92)
        del arg52_1
        buf163 = reinterpret_tensor(buf91, (1, 2048, 512), (1048576, 512, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [x_2, layer_norm_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_6.run(buf63, buf92, arg53_1, arg54_1, arg55_1, buf163, 2048, 512, grid=grid(2048), stream=stream0)
        del arg54_1
        del arg55_1
        buf96 = empty_strided_cuda((1, 234, 16384), (3833856, 16384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [row_p], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg62_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf26, (1, 64, 16384), (1048576, 16384, 1), 0), out=buf96)
        del arg62_1
        buf97 = empty_strided_cuda((1, 234, 16384), (3833856, 16384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [col_p], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg63_1, (1, 234, 64), (14976, 64, 1), 0), reinterpret_tensor(buf26, (1, 64, 16384), (1048576, 16384, 1), 0), out=buf97)
        del arg63_1
        del buf26
        buf101 = empty_strided_cuda((234, 32, 512), (16384, 512, 1), torch.float32)
        buf106 = empty_strided_cuda((234, 1, 33, 512), (16896, 1, 512, 1), torch.float32)
        buf104 = reinterpret_tensor(buf106, (234, 1, 32, 512), (16896, 1, 512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [layer_norm_5, kv_2], Original ATen: [aten.native_layer_norm, aten.cat]
        triton_per_fused_cat_native_layer_norm_24.run(buf96, buf97, arg64_1, arg65_1, buf101, buf104, 7488, 512, grid=grid(7488), stream=stream0)
        del arg64_1
        del arg65_1
        buf102 = empty_strided_cuda((7488, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (7488, 512), (512, 1), 0), reinterpret_tensor(arg66_1, (512, 1536), (1, 512), 0), out=buf102)
        buf105 = reinterpret_tensor(buf106, (234, 1, 1, 512), (16896, 1, 512, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [block_node_2], Original ATen: [aten.mean]
        triton_per_fused_mean_25.run(buf101, buf105, 119808, 32, grid=grid(119808), stream=stream0)
        del buf104
        del buf105
        buf107 = empty_strided_cuda((7722, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf106, (7722, 512), (512, 1), 0), reinterpret_tensor(arg66_1, (512, 1536), (1, 512), 0), out=buf107)
        del arg66_1
        buf108 = reinterpret_tensor(buf101, (234, 8, 32, 64, 1, 1), (16384, 2048, 64, 1, 1, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [einsum_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf102, arg67_1, buf108, 3833856, grid=grid(3833856), stream=stream0)
        buf109 = reinterpret_tensor(buf106, (234, 8, 64, 1, 33, 1), (16896, 2112, 33, 33, 1, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [einsum_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf107, arg67_1, buf109, 119808, 33, grid=grid(119808, 33), stream=stream0)
        buf110 = empty_strided_cuda((1872, 32, 33), (1056, 33, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf108, (1872, 32, 64), (2048, 64, 1), 0), reinterpret_tensor(buf109, (1872, 64, 33), (2112, 33, 1), 0), out=buf110)
        buf111 = empty_strided_cuda((234, 1, 32, 33, 4), (4224, 988416, 132, 4, 1), torch.float32)
        buf146 = empty_strided_cuda((234, 1, 32, 33, 4), (4224, 988416, 132, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_8, setitem_9, setitem_12, setitem_13], Original ATen: [aten.lift_fresh, aten.fill]
        triton_poi_fused_fill_lift_fresh_28.run(arg30_1, buf111, buf146, 988416, grid=grid(988416), stream=stream0)
        del arg30_1
        # Topologically Sorted Source Nodes: [setitem_8, setitem_9, setitem_10], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
        triton_poi_fused_fill_index_put_lift_fresh_29.run(buf111, 29952, grid=grid(29952), stream=stream0)
        buf113 = empty_strided_cuda((234, 1, 32, 33), (1056, 1056, 33, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_11], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_30.run(buf111, buf113, 247104, grid=grid(247104), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_11], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_31.run(buf113, 7488, grid=grid(7488), stream=stream0)
        buf116 = empty_strided_cuda((234, 1, 32, 33, 8), (8448, 1976832, 33, 1, 1056), torch.float32)
        # Topologically Sorted Source Nodes: [eq_4, scores_8, scores_6, scores_7, attn_probs_2], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_eq_masked_fill_mul_32.run(arg31_1, buf110, buf113, buf111, buf116, 59904, 33, grid=grid(59904), stream=stream0)
        buf117 = empty_strided_cuda((234, 1, 32, 1, 8), (256, 59904, 8, 59904, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_probs_2], Original ATen: [aten._softmax]
        triton_per_fused__softmax_33.run(buf116, buf117, 59904, 33, grid=grid(59904), stream=stream0)
        buf118 = empty_strided_cuda((234, 1, 32, 33, 4), (4224, 1, 132, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_34.run(buf113, buf111, buf118, 988416, grid=grid(988416), stream=stream0)
        del buf111
        buf119 = reinterpret_tensor(buf110, (247104, 8), (8, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (247104, 4), (4, 1), 0), reinterpret_tensor(arg68_1, (4, 8), (1, 4), 0), out=buf119)
        del arg68_1
        buf120 = reinterpret_tensor(buf116, (234, 8, 32, 33, 1, 1), (8448, 1056, 33, 1, 1, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [x_out_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf120, buf117, arg31_1, buf119, arg69_1, 1872, 1056, grid=grid(1872, 1056), stream=stream0)
        del arg69_1
        buf121 = reinterpret_tensor(buf109, (234, 8, 33, 1, 64, 1), (16896, 2112, 64, 64, 1, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_out_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_36.run(buf107, arg67_1, buf121, 3953664, grid=grid(3953664), stream=stream0)
        del arg67_1
        buf122 = reinterpret_tensor(buf108, (1872, 32, 64), (2048, 64, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [x_out_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf120, (1872, 32, 33), (1056, 33, 1), 0), reinterpret_tensor(buf121, (1872, 33, 64), (2112, 64, 1), 0), out=buf122)
        buf123 = empty_strided_cuda((234, 1, 32, 8, 64), (16384, 1, 512, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_out_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf122, buf123, 3833856, grid=grid(3833856), stream=stream0)
        buf124 = reinterpret_tensor(buf122, (7488, 512), (512, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (7488, 512), (512, 1), 0), reinterpret_tensor(arg70_1, (512, 512), (1, 512), 0), out=buf124)
        del arg70_1
        buf128 = reinterpret_tensor(buf123, (234, 32, 512), (16384, 512, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [x_4, layer_norm_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_38.run(buf96, buf97, buf124, arg71_1, arg72_1, arg73_1, buf128, 7488, 512, grid=grid(7488), stream=stream0)
        del arg72_1
        del arg73_1
        buf129 = empty_strided_cuda((7488, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (7488, 512), (512, 1), 0), reinterpret_tensor(arg74_1, (512, 2048), (1, 512), 0), out=buf129)
        del arg74_1
        buf130 = reinterpret_tensor(buf129, (234, 32, 2048), (65536, 2048, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_39.run(buf130, arg75_1, 15335424, grid=grid(15335424), stream=stream0)
        del arg75_1
        buf131 = reinterpret_tensor(buf128, (7488, 512), (512, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf130, (7488, 2048), (2048, 1), 0), reinterpret_tensor(arg76_1, (2048, 512), (1, 2048), 0), out=buf131)
        del arg76_1
        buf132 = reinterpret_tensor(buf131, (234, 32, 512), (16384, 512, 1), 0); del buf131  # reuse
        buf136 = empty_strided_cuda((234, 32, 512), (16384, 512, 1), torch.float32)
        buf141 = reinterpret_tensor(buf121, (234, 1, 33, 512), (16896, 1, 512, 1), 0); del buf121  # reuse
        buf139 = reinterpret_tensor(buf141, (234, 1, 32, 512), (16896, 1, 512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_4, x_5, layer_norm_7, kv_3], Original ATen: [aten.add, aten.native_layer_norm, aten.cat]
        triton_per_fused_add_cat_native_layer_norm_40.run(buf132, buf96, buf97, buf124, arg71_1, arg77_1, arg78_1, arg79_1, buf136, buf139, 7488, 512, grid=grid(7488), stream=stream0)
        del arg71_1
        del arg77_1
        del arg78_1
        del arg79_1
        del buf124
        del buf96
        buf137 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf136, (7488, 512), (512, 1), 0), reinterpret_tensor(arg80_1, (512, 1536), (1, 512), 0), out=buf137)
        buf140 = reinterpret_tensor(buf141, (234, 1, 1, 512), (16896, 1, 512, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [block_node_3], Original ATen: [aten.mean]
        triton_per_fused_mean_25.run(buf136, buf140, 119808, 32, grid=grid(119808), stream=stream0)
        del buf139
        del buf140
        buf142 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf141, (7722, 512), (512, 1), 0), reinterpret_tensor(arg80_1, (512, 1536), (1, 512), 0), out=buf142)
        del arg80_1
        buf143 = reinterpret_tensor(buf136, (234, 8, 32, 64, 1, 1), (16384, 2048, 64, 1, 1, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [einsum_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf137, arg81_1, buf143, 3833856, grid=grid(3833856), stream=stream0)
        del buf137
        buf144 = reinterpret_tensor(buf141, (234, 8, 64, 1, 33, 1), (16896, 2112, 33, 33, 1, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [einsum_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf142, arg81_1, buf144, 119808, 33, grid=grid(119808, 33), stream=stream0)
        buf145 = reinterpret_tensor(buf120, (1872, 32, 33), (1056, 33, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [einsum_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf143, (1872, 32, 64), (2048, 64, 1), 0), reinterpret_tensor(buf144, (1872, 64, 33), (2112, 33, 1), 0), out=buf145)
        # Topologically Sorted Source Nodes: [setitem_12, setitem_13, setitem_14], Original ATen: [aten.lift_fresh, aten.fill, aten.index_put]
        triton_poi_fused_fill_index_put_lift_fresh_29.run(buf146, 29952, grid=grid(29952), stream=stream0)
        buf148 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [setitem_15], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_30.run(buf146, buf148, 247104, grid=grid(247104), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_15], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_31.run(buf148, 7488, grid=grid(7488), stream=stream0)
        buf151 = reinterpret_tensor(buf119, (234, 1, 32, 33, 8), (8448, 1976832, 33, 1, 1056), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [eq_6, scores_11, scores_9, scores_10, attn_probs_3], Original ATen: [aten.eq, aten.masked_fill, aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_eq_masked_fill_mul_32.run(arg31_1, buf145, buf148, buf146, buf151, 59904, 33, grid=grid(59904), stream=stream0)
        buf152 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [attn_probs_3], Original ATen: [aten._softmax]
        triton_per_fused__softmax_33.run(buf151, buf152, 59904, 33, grid=grid(59904), stream=stream0)
        buf153 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_34.run(buf148, buf146, buf153, 988416, grid=grid(988416), stream=stream0)
        del buf146
        del buf148
        buf154 = reinterpret_tensor(buf145, (247104, 8), (8, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (247104, 4), (4, 1), 0), reinterpret_tensor(arg82_1, (4, 8), (1, 4), 0), out=buf154)
        del arg82_1
        del buf153
        buf155 = reinterpret_tensor(buf151, (234, 8, 32, 33, 1, 1), (8448, 1056, 33, 1, 1, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [x_out_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf155, buf152, arg31_1, buf154, arg83_1, 1872, 1056, grid=grid(1872, 1056), stream=stream0)
        del arg31_1
        del arg83_1
        del buf152
        del buf154
        buf156 = reinterpret_tensor(buf144, (234, 8, 33, 1, 64, 1), (16896, 2112, 64, 64, 1, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [x_out_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_36.run(buf142, arg81_1, buf156, 3953664, grid=grid(3953664), stream=stream0)
        del arg81_1
        del buf142
        buf157 = reinterpret_tensor(buf143, (1872, 32, 64), (2048, 64, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [x_out_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf155, (1872, 32, 33), (1056, 33, 1), 0), reinterpret_tensor(buf156, (1872, 33, 64), (2112, 64, 1), 0), out=buf157)
        del buf155
        del buf156
        buf158 = reinterpret_tensor(buf97, (234, 1, 32, 8, 64), (16384, 1, 512, 64, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [x_out_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf157, buf158, 3833856, grid=grid(3833856), stream=stream0)
        buf159 = reinterpret_tensor(buf157, (7488, 512), (512, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf158, (7488, 512), (512, 1), 0), reinterpret_tensor(arg84_1, (512, 512), (1, 512), 0), out=buf159)
        del arg84_1
        buf170 = reinterpret_tensor(buf158, (234, 32, 512), (16384, 512, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [x_6, layer_norm_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_41.run(buf132, buf159, arg85_1, arg86_1, arg87_1, buf170, 7488, 512, grid=grid(7488), stream=stream0)
        del arg86_1
        del arg87_1
        buf164 = reinterpret_tensor(buf61, (2048, 2048), (2048, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf163, (2048, 512), (512, 1), 0), reinterpret_tensor(arg56_1, (512, 2048), (1, 512), 0), out=buf164)
        del arg56_1
        buf165 = reinterpret_tensor(buf164, (1, 2048, 2048), (4194304, 2048, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_22.run(buf165, arg57_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg57_1
        buf166 = reinterpret_tensor(buf163, (2048, 512), (512, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf165, (2048, 2048), (2048, 1), 0), reinterpret_tensor(arg58_1, (2048, 512), (1, 2048), 0), out=buf166)
        del arg58_1
        del buf165
        buf167 = reinterpret_tensor(buf166, (1, 2048, 512), (1048576, 512, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.add]
        triton_poi_fused_add_42.run(buf167, buf63, buf92, arg53_1, arg59_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg53_1
        del arg59_1
        del buf63
        del buf92
        buf168 = empty_strided_cuda((2048, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_leaf], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg61_1, reinterpret_tensor(buf167, (2048, 512), (512, 1), 0), reinterpret_tensor(arg60_1, (512, 32), (1, 512), 0), alpha=1, beta=1, out=buf168)
        del arg60_1
        del arg61_1
        buf169 = empty_strided_cuda((64, 32, 32), (1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf168, (64, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf168, (64, 32, 32), (1024, 1, 32), 0), out=buf169)
        buf171 = reinterpret_tensor(buf130, (7488, 2048), (2048, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf170, (7488, 512), (512, 1), 0), reinterpret_tensor(arg88_1, (512, 2048), (1, 512), 0), out=buf171)
        del arg88_1
        buf172 = reinterpret_tensor(buf171, (234, 32, 2048), (65536, 2048, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_39.run(buf172, arg89_1, 15335424, grid=grid(15335424), stream=stream0)
        del arg89_1
        buf173 = reinterpret_tensor(buf170, (7488, 512), (512, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf172, (7488, 2048), (2048, 1), 0), reinterpret_tensor(arg90_1, (2048, 512), (1, 2048), 0), out=buf173)
        del arg90_1
        del buf172
        buf174 = reinterpret_tensor(buf173, (234, 32, 512), (16384, 512, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [x_6, x_7], Original ATen: [aten.add]
        triton_poi_fused_add_43.run(buf174, buf132, buf159, arg85_1, arg91_1, 3833856, grid=grid(3833856), stream=stream0)
        del arg85_1
        del arg91_1
        del buf132
        del buf159
        buf175 = empty_strided_cuda((7488, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_leaf_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg93_1, reinterpret_tensor(buf174, (7488, 512), (512, 1), 0), reinterpret_tensor(arg92_1, (512, 32), (1, 512), 0), alpha=1, beta=1, out=buf175)
        del arg92_1
        del arg93_1
        buf176 = empty_strided_cuda((7488, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_leaf], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg95_1, reinterpret_tensor(buf174, (7488, 512), (512, 1), 0), reinterpret_tensor(arg94_1, (512, 32), (1, 512), 0), alpha=1, beta=1, out=buf176)
        del arg94_1
        del arg95_1
        del buf174
        buf177 = empty_strided_cuda((234, 32, 32), (1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf175, (234, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf176, (234, 32, 32), (1024, 1, 32), 0), out=buf177)
        del buf175
        del buf176
        buf178 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [node_U], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg99_1, reinterpret_tensor(buf167, (2048, 512), (512, 1), 0), reinterpret_tensor(arg98_1, (512, 32), (1, 512), 0), alpha=1, beta=1, out=buf178)
        del arg98_1
        del arg99_1
        buf179 = empty_strided_cuda((2048, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [node_V], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg101_1, reinterpret_tensor(buf167, (2048, 512), (512, 1), 0), reinterpret_tensor(arg100_1, (512, 32), (1, 512), 0), alpha=1, beta=1, out=buf179)
        del arg100_1
        del arg101_1
        buf180 = empty_strided_cuda((2048, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf167, (2048, 512), (512, 1), 0), reinterpret_tensor(arg96_1, (512, 1), (1, 512), 0), out=buf180)
        del arg96_1
        del buf167
        buf181 = reinterpret_tensor(buf180, (1, 64, 32, 1), (2048, 32, 1, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [sigmoid], Original ATen: [aten.sigmoid]
        triton_poi_fused_sigmoid_44.run(buf181, arg97_1, 2048, grid=grid(2048), stream=stream0)
        del arg97_1
        buf182 = empty_strided_cuda((1, 438272), (438272, 1), torch.float32)
        # Topologically Sorted Source Nodes: [packed_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf169, buf177, buf178, buf179, buf181, buf182, 438272, grid=grid(438272), stream=stream0)
        del buf169
        del buf177
        del buf178
        del buf179
    return (buf182, reinterpret_tensor(buf181, (1, 64, 32), (2048, 32, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 2048, 9), (67716, 9, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, 18), (18, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((2, 13590), (13590, 1), device='cuda:0', dtype=torch.int64)
    arg7_1 = rand_strided((13590, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    arg28_1 = rand_strided((64, 32, 33, 4), (4224, 132, 4, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, 32, 33), (1056, 33, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((234, 32, 33, 4), (4224, 132, 4, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((234, 32, 33), (1056, 33, 1), device='cuda:0', dtype=torch.float32)
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
    arg60_1 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((234, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((234, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((8, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((8, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
