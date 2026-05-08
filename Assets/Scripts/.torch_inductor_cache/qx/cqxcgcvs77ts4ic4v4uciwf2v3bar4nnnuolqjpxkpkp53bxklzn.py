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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/g4/cg43izmi3c5qowo366z4raypprzmv3ms5i6yxtqt5mt66cbas5nv.py
# Topologically Sorted Source Nodes: [r_batched_1], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   r_batched_1 => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%unsqueeze, [0, 0, 0, 672], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 7520, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0), tmp2, other=0.0)
    tl.store(out_ptr0 + (x0), tmp3, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/kh/ckh5wp2sdv34go4sog7v6dzlhtsbvpkqu4ja4vh6i4dgmxgftp4f.py
# Topologically Sorted Source Nodes: [getitem_2, index_add_, getitem_3, index_add__1], Original ATen: [aten.index, aten.index_add]
# Source node to ATen node mapping:
#   getitem_2 => index
#   getitem_3 => index_1
#   index_add_ => index_put
#   index_add__1 => index_put_1
# Graph fragment:
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_17, [None, %device_put_4]), kwargs = {})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%view_4, [None, %device_put_2], %index, True), kwargs = {})
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_18, [None, %device_put_4]), kwargs = {})
#   %index_put_1 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%view_20, [None, %device_put_3], %index_1, True), kwargs = {})
triton_poi_fused_index_index_add_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_index_add_1', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 32)
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 256, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 256)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 256")
    tmp7 = tl.full([XBLOCK], 996, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert(((0 <= tmp10) & (tmp10 < 996)) | ~(xmask), "index out of bounds: 0 <= tmp10 < 996")
    tmp12 = tl.load(in_ptr2 + (x0 + (32*tmp10)), xmask)
    tmp14 = tmp13 + tmp1
    tmp15 = tmp13 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp13)
    tl.device_assert(((0 <= tmp16) & (tmp16 < 256)) | ~(xmask), "index out of bounds: 0 <= tmp16 < 256")
    tmp18 = tl.load(in_ptr4 + (x0 + (32*tmp10)), xmask)
    tl.atomic_add(out_ptr0 + (x0 + (32*tmp4)), tmp12, xmask, sem='relaxed')
    tl.atomic_add(out_ptr0 + (x0 + (32*tmp16)), tmp18, xmask, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/q7/cq7rcyzdiug25rigwzakflys236tcgna2nnomrdijcmyp3wzowuq.py
# Topologically Sorted Source Nodes: [mul_1, y_1], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_1 => mul_1
#   y_1 => add
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_9, %constant_pad_nd), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_25, %mul_1), kwargs = {})
triton_poi_fused_add_mul_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (1282048 + x0), None)
    tmp2 = tl.load(in_ptr2 + (x0), None)
    tmp4 = tl.load(in_out_ptr0 + (x0), None)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
    args.clear()
    assert_size_stride(arg0_1, (7520, 1), (1, 1))
    assert_size_stride(arg1_1, (1, 1290240), (1290240, 1))
    assert_size_stride(arg2_1, (996, 256), (256, 1))
    assert_size_stride(arg3_1, (996, 256), (256, 1))
    assert_size_stride(arg4_1, (2562, ), (1, ))
    assert_size_stride(arg5_1, (2562, ), (1, ))
    assert_size_stride(arg6_1, (2562, ), (1, ))
    assert_size_stride(arg7_1, (1, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 8192, 1), (8192, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [r_batched_1], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0.run(arg0_1, buf0, 8192, grid=grid(8192), stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((256, 32, 1), (32, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_pool], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg1_1, (256, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf0, (256, 32, 1), (32, 1, 0), 0), out=buf1)
        buf2 = empty_strided_cuda((2562, ), (1, ), torch.int64)
        buf2.copy_(arg4_1)
        del arg4_1
        buf3 = empty_strided_cuda((996, 256), (256, 1), torch.float32)
        buf3.copy_(arg3_1)
        del arg3_1
        buf4 = empty_strided_cuda((1, 996, 32), (31872, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_col_strips], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf3, (1, 996, 256), (254976, 256, 1), 0), reinterpret_tensor(buf0, (1, 256, 32), (0, 32, 1), 0), out=buf4)
        buf5 = empty_strided_cuda((996, 32, 1), (32, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_r], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg1_1, (996, 32, 32), (1024, 32, 1), 262144), reinterpret_tensor(buf4, (996, 32, 1), (32, 1, 1), 0), out=buf5)
        buf6 = empty_strided_cuda((2562, ), (1, ), torch.int64)
        buf6.copy_(arg6_1)
        del arg6_1
        buf9 = buf3; del buf3  # reuse
        buf9.copy_(arg2_1)
        del arg2_1
        buf10 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_row_strips], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 996, 256), (254976, 256, 1), 0), reinterpret_tensor(buf0, (1, 256, 32), (0, 32, 1), 0), out=buf10)
        del buf9
        buf11 = empty_strided_cuda((996, 32, 1), (32, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_c], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg1_1, (996, 32, 32), (1024, 1, 32), 262144), reinterpret_tensor(buf10, (996, 32, 1), (32, 1, 1), 0), out=buf11)
        del buf10
        buf8 = empty_strided_cuda((2562, ), (1, ), torch.int64)
        buf8.copy_(arg5_1)
        del arg5_1
        # Topologically Sorted Source Nodes: [getitem_2, index_add_, getitem_3, index_add__1], Original ATen: [aten.index, aten.index_add]
        triton_poi_fused_index_index_add_1.run(buf2, buf6, buf5, buf8, buf11, buf1, 81984, grid=grid(81984), stream=stream0)
        del buf11
        del buf5
        buf13 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [mul_1, y_1], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf13, buf1, arg1_1, arg7_1, 8192, grid=grid(8192), stream=stream0)
        del arg1_1
        del arg7_1
        del buf1
    return (reinterpret_tensor(buf13, (7520, 1), (1, 1), 0), buf2, buf8, buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((7520, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1290240), (1290240, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((996, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((996, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((2562, ), (1, ), device='cpu', dtype=torch.int64)
    arg5_1 = rand_strided((2562, ), (1, ), device='cpu', dtype=torch.int64)
    arg6_1 = rand_strided((2562, ), (1, ), device='cpu', dtype=torch.int64)
    arg7_1 = rand_strided((1, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
