# AOT ID: ['2_inference']
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


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/zt/czt5k52gtyncttoiic2wg6gr3sjxyspcnmwia2oepndzh3i2tvm7.py
# Topologically Sorted Source Nodes: [h_off_1], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   h_off_1 => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_10, [2]), kwargs = {})
triton_poi_fused_mean_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 446464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*x1)), None)
    tmp3 = tl.load(in_ptr0 + (128 + x0 + (512*x1)), None)
    tmp4 = tl.load(in_ptr1 + (128 + x0 + (512*x1)), None)
    tmp7 = tl.load(in_ptr0 + (256 + x0 + (512*x1)), None)
    tmp8 = tl.load(in_ptr1 + (256 + x0 + (512*x1)), None)
    tmp11 = tl.load(in_ptr0 + (384 + x0 + (512*x1)), None)
    tmp12 = tl.load(in_ptr1 + (384 + x0 + (512*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tl.store(out_ptr0 + (x2), tmp16, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 4096, 128), (524288, 128, 1))
    assert_size_stride(arg1_1, (109, 32), (32, 1))
    assert_size_stride(arg2_1, (109, 32), (32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((109, 32), (32, 1), torch.float32)
        buf0.copy_(arg1_1)
        del arg1_1
        buf1 = empty_strided_cuda((1, 109, 16384), (1785856, 16384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [row_p], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (1, 109, 32), (3488, 32, 1), 0), reinterpret_tensor(arg0_1, (1, 32, 16384), (524288, 16384, 1), 0), out=buf1)
        buf2 = buf0; del buf0  # reuse
        buf2.copy_(arg2_1)
        del arg2_1
        buf3 = empty_strided_cuda((1, 109, 16384), (1785856, 16384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [col_p], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf2, (1, 109, 32), (3488, 32, 1), 0), reinterpret_tensor(arg0_1, (1, 32, 16384), (524288, 16384, 1), 0), out=buf3)
        del arg0_1
        del buf2
        buf4 = empty_strided_cuda((109, 32, 128), (4096, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_off_1], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_0.run(buf1, buf3, buf4, 446464, grid=grid(446464), stream=stream0)
        del buf1
        del buf3
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 4096, 128), (524288, 128, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((109, 32), (32, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((109, 32), (32, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
