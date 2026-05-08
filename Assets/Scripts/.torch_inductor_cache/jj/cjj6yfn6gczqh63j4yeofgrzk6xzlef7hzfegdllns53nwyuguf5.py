# AOT ID: ['11_inference']
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/ru/crutytjghgnquqlng53ky3lqa4kjz26olrrokyzajpgk2jnmey37.py
# Topologically Sorted Source Nodes: [row_highway], Original ATen: [aten.new_zeros]
# Source node to ATen node mapping:
#   row_highway => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 32, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_new_zeros_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/p7/cp7cujbqj5mydoi4jgr6uvqdqopt4lxxnz55tuftcgewrpyrrhlw.py
# Topologically Sorted Source Nodes: [row_highway, index_add_, col_highway, index_add__1], Original ATen: [aten.new_zeros, aten.index_add]
# Source node to ATen node mapping:
#   col_highway => full_default_1
#   index_add_ => index_put
#   index_add__1 => index_put_1
#   row_highway => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 32, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default, [None, %arg2_1], %view_1, True), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 32, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_1 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_1, [None, %arg2_1], %view_1, True), kwargs = {})
triton_poi_fused_index_add_new_zeros_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_add_new_zeros_1', 'mutated_arg_names': ['out_ptr0', 'out_ptr1'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 16384)
    x2 = xindex
    x0 = xindex % 16384
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2), None)
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 32), "index out of bounds: 0 <= tmp4 < 32")
    tl.atomic_add(out_ptr0 + (x0 + (16384*tmp4)), tmp6, None, sem='relaxed')
    tl.atomic_add(out_ptr1 + (x0 + (16384*tmp4)), tmp6, None, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/gq/cgq7worum5mfzne5fehafxi7mdje63iwrmvcrpme5ynv6luxi7au.py
# Topologically Sorted Source Nodes: [toks_1, index_add__2, index_add__3], Original ATen: [aten.index_select, aten.index_add]
# Source node to ATen node mapping:
#   index_add__2 => index_put_2
#   index_add__3 => index_put_3
#   toks_1 => index
# Graph fragment:
#   %index : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_3, [None, %arg3_1]), kwargs = {})
#   %index_put_2 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put, [None, %arg4_1], %index, True), kwargs = {})
#   %index_put_3 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_1, [None, %arg5_1], %index, True), kwargs = {})
triton_poi_fused_index_add_index_select_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_add_index_select_2', 'mutated_arg_names': ['out_ptr0', 'out_ptr1'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2916352
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
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 32), "index out of bounds: 0 <= tmp4 < 32")
    tmp7 = tl.full([XBLOCK], 109, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert((0 <= tmp10) & (tmp10 < 109), "index out of bounds: 0 <= tmp10 < 109")
    tmp12 = tl.load(in_ptr2 + (x0 + (128*(x1 // 4)) + (4096*tmp10)), None)
    tmp14 = tmp13 + tmp1
    tmp15 = tmp13 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp13)
    tl.device_assert((0 <= tmp16) & (tmp16 < 32), "index out of bounds: 0 <= tmp16 < 32")
    tl.atomic_add(out_ptr0 + (x3 + (16384*tmp4)), tmp12, None, sem='relaxed')
    tl.atomic_add(out_ptr1 + (x3 + (16384*tmp16)), tmp12, None, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/2u/c2unolhdjdxol6rvasnppbckma2qkml5s6ogg5atjeko5a36ny5a.py
# Topologically Sorted Source Nodes: [row_t_1], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   row_t_1 => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_22, [3]), kwargs = {})
triton_poi_fused_mean_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 446464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x1)), None)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + (512*x1)), None)
    tmp3 = tl.load(in_ptr0 + (256 + x0 + (512*x1)), None)
    tmp5 = tl.load(in_ptr0 + (384 + x0 + (512*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
    args.clear()
    assert_size_stride(arg0_1, (109, 32, 128), (4096, 128, 1))
    assert_size_stride(arg1_1, (1, 4096, 128), (524288, 128, 1))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (178, ), (1, ))
    assert_size_stride(arg4_1, (178, ), (1, ))
    assert_size_stride(arg5_1, (178, ), (1, ))
    assert_size_stride(arg6_1, (109, 32), (32, 1))
    assert_size_stride(arg7_1, (109, 32), (32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((109, 32), (32, 1), torch.float32)
        buf0.copy_(arg6_1)
        del arg6_1
        buf1 = empty_strided_cuda((1, 32, 128, 128), (524288, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [row_highway], Original ATen: [aten.new_zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_new_zeros_0.run(buf1, 524288, grid=grid(524288), stream=stream0)
        buf7 = empty_strided_cuda((1, 32, 128, 128), (524288, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [col_highway], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_0.run(buf7, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [row_highway, index_add_, col_highway, index_add__1], Original ATen: [aten.new_zeros, aten.index_add]
        triton_poi_fused_index_add_new_zeros_1.run(arg2_1, arg1_1, buf1, buf7, 524288, grid=grid(524288), stream=stream0)
        del arg1_1
        del arg2_1
        # Topologically Sorted Source Nodes: [toks_1, index_add__2, index_add__3], Original ATen: [aten.index_select, aten.index_add]
        triton_poi_fused_index_add_index_select_2.run(arg4_1, arg3_1, arg0_1, arg5_1, buf1, buf7, 2916352, grid=grid(2916352), stream=stream0)
        del arg0_1
        del arg3_1
        del arg4_1
        del arg5_1
        buf4 = empty_strided_cuda((1, 109, 16384), (1785856, 16384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [row_t], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (1, 109, 32), (3488, 32, 1), 0), reinterpret_tensor(buf1, (1, 32, 16384), (524288, 16384, 1), 0), out=buf4)
        del buf1
        buf5 = empty_strided_cuda((1, 109, 32, 128), (446464, 4096, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [row_t_1], Original ATen: [aten.mean]
        triton_poi_fused_mean_3.run(buf4, buf5, 446464, grid=grid(446464), stream=stream0)
        buf6 = buf0; del buf0  # reuse
        buf6.copy_(arg7_1)
        del arg7_1
        buf10 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [col_t], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 109, 32), (3488, 32, 1), 0), reinterpret_tensor(buf7, (1, 32, 16384), (524288, 16384, 1), 0), out=buf10)
        del buf6
        del buf7
        buf11 = empty_strided_cuda((1, 109, 32, 128), (446464, 4096, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [col_t_1], Original ATen: [aten.mean]
        triton_poi_fused_mean_3.run(buf10, buf11, 446464, grid=grid(446464), stream=stream0)
        del buf10
    return (reinterpret_tensor(buf5, (109, 32, 128), (4096, 128, 1), 0), reinterpret_tensor(buf11, (109, 32, 128), (4096, 128, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((109, 32, 128), (4096, 128, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 4096, 128), (524288, 128, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((178, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg4_1 = rand_strided((178, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg5_1 = rand_strided((178, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg6_1 = rand_strided((109, 32), (32, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((109, 32), (32, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
