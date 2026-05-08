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


cpp_fused_clone_0 = async_compile.cpp_pybinding(['const int64_t*', 'const int64_t*', 'const int64_t*', 'const float*', 'const float*', 'const float*', 'const float*', 'int64_t*', 'int64_t*', 'int64_t*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       const int64_t* in_ptr1,
                       const int64_t* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       int64_t* out_ptr0,
                       int64_t* out_ptr1,
                       int64_t* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(96L); x0+=static_cast<int64_t>(16L))
        {
            auto tmp0 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            tmp0.store(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
        }
        #pragma omp simd simdlen(8) 
        for(int64_t x0=static_cast<int64_t>(96L); x0<static_cast<int64_t>(109L); x0+=static_cast<int64_t>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
            out_ptr0[static_cast<int64_t>(x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(96L); x0+=static_cast<int64_t>(16L))
        {
            auto tmp0 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            tmp0.store(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
        }
        #pragma omp simd simdlen(8) 
        for(int64_t x0=static_cast<int64_t>(96L); x0<static_cast<int64_t>(109L); x0+=static_cast<int64_t>(1L))
        {
            auto tmp0 = in_ptr1[static_cast<int64_t>(x0)];
            out_ptr1[static_cast<int64_t>(x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(96L); x0+=static_cast<int64_t>(16L))
        {
            auto tmp0 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            tmp0.store(out_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
        }
        #pragma omp simd simdlen(8) 
        for(int64_t x0=static_cast<int64_t>(96L); x0<static_cast<int64_t>(109L); x0+=static_cast<int64_t>(1L))
        {
            auto tmp0 = in_ptr2[static_cast<int64_t>(x0)];
            out_ptr2[static_cast<int64_t>(x0)] = tmp0;
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3488L); x0+=static_cast<int64_t>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            tmp0.store(out_ptr3 + static_cast<int64_t>(x0));
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3488L); x0+=static_cast<int64_t>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            tmp0.store(out_ptr4 + static_cast<int64_t>(x0));
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3488L); x0+=static_cast<int64_t>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            tmp0.store(out_ptr5 + static_cast<int64_t>(x0));
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3488L); x0+=static_cast<int64_t>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            tmp0.store(out_ptr6 + static_cast<int64_t>(x0));
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
    args.clear()
    assert_size_stride(arg0_1, (109, ), (1, ))
    assert_size_stride(arg1_1, (109, ), (1, ))
    assert_size_stride(arg2_1, (109, ), (1, ))
    assert_size_stride(arg3_1, (109, 32), (32, 1))
    assert_size_stride(arg4_1, (109, 32), (32, 1))
    assert_size_stride(arg5_1, (109, 32), (32, 1))
    assert_size_stride(arg6_1, (109, 32), (32, 1))
    buf0 = empty_strided_cpu((109, ), (1, ), torch.int64)
    buf1 = empty_strided_cpu((109, ), (1, ), torch.int64)
    buf2 = empty_strided_cpu((109, ), (1, ), torch.int64)
    buf3 = empty_strided_cpu((109, 32), (32, 1), torch.float32)
    buf4 = empty_strided_cpu((109, 32), (32, 1), torch.float32)
    buf5 = empty_strided_cpu((109, 32), (32, 1), torch.float32)
    buf6 = empty_strided_cpu((109, 32), (32, 1), torch.float32)
    cpp_fused_clone_0(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, buf0, buf1, buf2, buf3, buf4, buf5, buf6)
    del arg0_1
    del arg1_1
    del arg2_1
    del arg3_1
    del arg4_1
    del arg5_1
    del arg6_1
    return (buf0, buf1, buf2, buf3, buf4, buf5, buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((109, ), (1, ), device='cpu', dtype=torch.int64)
    arg1_1 = rand_strided((109, ), (1, ), device='cpu', dtype=torch.int64)
    arg2_1 = rand_strided((109, ), (1, ), device='cpu', dtype=torch.int64)
    arg3_1 = rand_strided((109, 32), (32, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((109, 32), (32, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((109, 32), (32, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((109, 32), (32, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
