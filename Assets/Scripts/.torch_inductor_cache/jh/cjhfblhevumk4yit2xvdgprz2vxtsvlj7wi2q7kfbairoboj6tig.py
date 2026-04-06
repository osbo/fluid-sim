# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
from torch._inductor.runtime.runtime_utils import compile_mps_shader



# Topologically Sorted Source Nodes: [mul, pAp, alpha, mul_1, add_, mul_2, sub_, mul_3, rho_new, fill_, beta, mul_, add__1], Original ATen: [aten.mul, aten.sum, aten.div, aten.add, aten.sub, aten.fill, aten.copy_]
# Source node to ATen node mapping:
#   add_ => add
#   add__1 => add_1
#   alpha => div
#   beta => div_1
#   fill_ => copy_1
#   mul => mul
#   mul_ => mul_4
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   pAp => sum_1
#   rho_new => sum_2
#   sub_ => sub
# Graph fragment:
#   %copy_ : Tensor "f32[7424, 1][1, 1]mps:0" = PlaceHolder[target=copy_]
#   %mm : Tensor "f32[7424, 1][1, 1]mps:0" = PlaceHolder[target=mm]
#   %copy__3 : Tensor "f32[7424, 1][1, 1]mps:0" = PlaceHolder[target=copy__3]
#   %copy__1 : Tensor "f32[1][1]mps:0" = PlaceHolder[target=copy__1]
#   %sum_1 : Tensor "f32[][]mps:0" = PlaceHolder[target=sum_1]
#   %copy__2 : Tensor "f32[7424, 1][1, 1]mps:0" = PlaceHolder[target=copy__2]
#   %sum_2 : Tensor "f32[][]mps:0" = PlaceHolder[target=sum_2]
#   %add_1 : Tensor "f32[7424, 1][1, 7424]mps:0" = PlaceHolder[target=add_1]
#   %add : Tensor "f32[7424, 1][1, 7424]mps:0" = PlaceHolder[target=add]
#   %sub : Tensor "f32[7424, 1][1, 7424]mps:0" = PlaceHolder[target=sub]
#   %copy__4 : Tensor "f32[7424, 1][1, 1]mps:0" = PlaceHolder[target=copy__4]
#   %mul : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %mm), kwargs = {})
#   %sum_1 : Tensor "f32[][]mps:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul,), kwargs = {})
#   %div : Tensor "f32[1][1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg2_1, %sum_1), kwargs = {})
#   %mul_1 : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %div), kwargs = {})
#   %add : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mul_1), kwargs = {})
#   %mul_2 : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, %div), kwargs = {})
#   %sub : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg4_1, %mul_2), kwargs = {})
#   %mul_3 : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %sub), kwargs = {})
#   %sum_2 : Tensor "f32[][]mps:0"[num_users=2] = call_function[target=torch.ops.aten.sum.default](args = (%mul_3,), kwargs = {})
#   %copy_1 : Tensor "f32[1][1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%arg2_1, %sum_2), kwargs = {})
#   %div_1 : Tensor "f32[1][1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_2, %arg2_1), kwargs = {})
#   %mul_4 : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %div_1), kwargs = {})
#   %add_1 : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %sub), kwargs = {})
#   %copy_ : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg1_1, %add_1), kwargs = {})
#   %copy__1 : Tensor "f32[1][1]mps:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg2_1, %copy_1), kwargs = {})
#   %copy__2 : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg3_1, %add), kwargs = {})
#   %copy__3 : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg4_1, %sub), kwargs = {})
#   %copy__4 : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg5_1, %sub), kwargs = {})
#   return %sum_1,%sum_2,%add,%add_1,%sub,%buf8,%buf16,%buf17,%buf15,%buf14
mps_lib_0 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    [[max_total_threads_per_threadgroup(1024)]]
    kernel void generated_kernel(
        device float* out_ptr2,
        device float* out_ptr5,
        device float* out_ptr6,
        device float* out_ptr7,
        device float* out_ptr8,
        device float* out_ptr9,
        constant float* in_ptr0,
        constant float* in_ptr1,
        constant float* in_ptr2,
        constant float* in_ptr3,
        constant float* in_ptr4,
        uint2 thread_pos [[thread_position_in_grid]],
        uint2 group_pos [[thread_position_in_threadgroup]]
    ) {
        auto xindex = thread_pos.x;
        auto r0_index = thread_pos.y;
        threadgroup float tmp_acc_0[32];
        float tmp_acc_1 = 0;
        threadgroup float tmp_acc_2[32];
        float tmp_acc_3 = 0;
        for(auto r0_0_cnt = 0; r0_0_cnt < 8; ++r0_0_cnt) {
            int r0_0 = 8 * r0_index + r0_0_cnt;
            if (r0_0 >= 7424) break;
            auto tmp0 = in_ptr0[r0_0];
            auto tmp1 = in_ptr1[r0_0];
            auto tmp2 = tmp0 * tmp1;
            tmp_acc_1 += tmp2;
        }
        auto tmp3 = c10::metal::threadgroup_sum(tmp_acc_0, tmp_acc_1, r0_index * 1, 1024);
        for(auto r0_0_cnt = 0; r0_0_cnt < 8; ++r0_0_cnt) {
            int r0_0 = 8 * r0_index + r0_0_cnt;
            if (r0_0 >= 7424) break;
            auto tmp4 = in_ptr2[r0_0];
            auto tmp5 = in_ptr1[r0_0];
            auto tmp6 = in_ptr3[0];
            auto tmp12 = in_ptr4[r0_0];
            auto tmp13 = in_ptr0[r0_0];
            auto tmp7 = tmp6 / tmp3;
            auto tmp8 = tmp5 * tmp7;
            auto tmp9 = tmp4 - tmp8;
            auto tmp10 = tmp9 * tmp9;
            tmp_acc_3 += tmp10;
            auto tmp14 = tmp13 * tmp7;
            auto tmp15 = tmp12 + tmp14;
            out_ptr2[r0_0] = static_cast<float>(tmp15);
        }
        auto tmp11 = c10::metal::threadgroup_sum(tmp_acc_2, tmp_acc_3, r0_index * 1, 1024);
        for(auto r0_0_cnt = 0; r0_0_cnt < 8; ++r0_0_cnt) {
            int r0_0 = 8 * r0_index + r0_0_cnt;
            if (r0_0 >= 7424) break;
            auto tmp16 = in_ptr0[r0_0];
            auto tmp17 = in_ptr3[0];
            auto tmp20 = in_ptr2[r0_0];
            auto tmp21 = in_ptr1[r0_0];
            auto tmp26 = out_ptr2[r0_0];
            auto tmp18 = tmp11 / tmp17;
            auto tmp19 = tmp16 * tmp18;
            auto tmp22 = tmp17 / tmp3;
            auto tmp23 = tmp21 * tmp22;
            auto tmp24 = tmp20 - tmp23;
            auto tmp25 = tmp19 + tmp24;
            out_ptr5[r0_0] = static_cast<float>(tmp25);
            out_ptr6[r0_0] = static_cast<float>(tmp24);
            out_ptr7[r0_0] = static_cast<float>(tmp24);
            out_ptr8[r0_0] = static_cast<float>(tmp26);
        }
        out_ptr9[0] = static_cast<float>(tmp11);
    }
''')

def partition_0(args):
    arg0_1, arg1_1, arg4_1, arg2_1, arg3_1, arg5_1 = args
    args.clear()
    assert_size_stride(arg0_1, (7424, 7424), (7424, 1))
    assert_size_stride(arg1_1, (7424, 1), (1, 1))
    assert_size_stride(arg4_1, (7424, 1), (1, 1))
    assert_size_stride(arg2_1, (1, ), (1, ))
    assert_size_stride(arg3_1, (7424, 1), (1, 1))
    assert_size_stride(arg5_1, (7424, 1), (1, 1))
    buf0 = empty_strided((7424, 1), (1, 1), device='mps', dtype=torch.float32)
    # Topologically Sorted Source Nodes: [Ap], Original ATen: [aten.mm]
    extern_kernels.mm(arg0_1, arg1_1, out=buf0)
    del arg0_1
    buf5 = empty_strided((7424, 1), (1, 7424), device='mps', dtype=torch.float32)
    mps_lib_0.generated_kernel(buf5, arg1_1, arg4_1, arg5_1, arg3_1, arg2_1, arg1_1, buf0, arg4_1, arg2_1, arg3_1, threads=[1, 1024], group_size=[1, 1024])
    del arg1_1
    del arg2_1
    del arg3_1
    del arg4_1
    del arg5_1
    return ()


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
        args.clear()
        partition0_args = [arg0_1, arg1_1, arg4_1, arg2_1, arg3_1, arg5_1]
        del arg0_1, arg1_1, arg4_1, arg2_1, arg3_1, arg5_1
        () = self.partitions[0](partition0_args)
        del partition0_args
        return ()

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((7424, 7424), (7424, 1), device='mps:0', dtype=torch.float32)
    arg1_1 = rand_strided((7424, 1), (1, 1), device='mps:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, ), (1, ), device='mps:0', dtype=torch.float32)
    arg3_1 = rand_strided((7424, 1), (1, 1), device='mps:0', dtype=torch.float32)
    arg4_1 = rand_strided((7424, 1), (1, 1), device='mps:0', dtype=torch.float32)
    arg5_1 = rand_strided((7424, 1), (1, 1), device='mps:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
