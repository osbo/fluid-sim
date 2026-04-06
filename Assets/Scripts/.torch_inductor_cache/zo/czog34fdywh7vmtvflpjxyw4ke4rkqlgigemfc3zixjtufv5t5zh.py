# AOT ID: ['3_inference']
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



# Topologically Sorted Source Nodes: [getitem_1, off_diag_blocks, off_diag_blocks_1], Original ATen: [aten.slice, aten.view, aten.mul]
# Source node to ATen node mapping:
#   getitem_1 => slice_2
#   off_diag_blocks => view_16
#   off_diag_blocks_1 => mul_3
# Graph fragment:
#   %arg6_1 : Tensor "f32[1, 1672448][1672448, 1]mps:0" = PlaceHolder[target=arg6_1]
#   %arg7_1 : Tensor "f32[1, 234, 1, 1][234, 1, 1, 1]mps:0" = PlaceHolder[target=arg7_1]
#   %slice_2 : Tensor "f32[1, 239616][1672448, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg6_1, 1, 950272, 1189888), kwargs = {})
#   %view_16 : Tensor "f32[1, 234, 32, 32][1672448, 1024, 32, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%slice_2, [1, 234, 32, 32]), kwargs = {})
#   %mul_3 : Tensor "f32[1, 234, 32, 32][239616, 1024, 32, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_16, %arg7_1), kwargs = {})
#   return %mul_3
mps_lib_0 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        constant float* in_ptr0,
        constant float* in_ptr1,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x2 = xindex;
        int x1 = c10::metal::floor_divide(xindex, 1024);
        auto tmp0 = in_ptr0[950272 + x2];
        auto tmp1 = in_ptr1[x1];
        auto tmp2 = tmp0 * tmp1;
        out_ptr0[x2] = static_cast<float>(tmp2);
    }
''')


# Topologically Sorted Source Nodes: [zero__3], Original ATen: [aten.zero]
# Source node to ATen node mapping:
#   zero__3 => full_default_1
# Graph fragment:
#   %full_default_1 : Tensor "f32[234, 32][32, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([234, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: mps:0, pin_memory: False})
#   return %index_put_1
mps_lib_1 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x0 = xindex;
        auto tmp0 = 0.0;
        out_ptr0[x0] = static_cast<float>(tmp0);
    }
''')


# Topologically Sorted Source Nodes: [mul, pAp, alpha, mul_2, sub_], Original ATen: [aten.mul, aten.sum, aten.div, aten.sub]
# Source node to ATen node mapping:
#   alpha => div
#   mul => mul
#   mul_2 => mul_2
#   pAp => sum_1
#   sub_ => sub
# Graph fragment:
#   %copy_ : Tensor "f32[7424, 1][1, 1]mps:0" = PlaceHolder[target=copy_]
#   %mm : Tensor "f32[7424, 1][1, 1]mps:0" = PlaceHolder[target=mm]
#   %copy__3 : Tensor "f32[7424, 1][1, 1]mps:0" = PlaceHolder[target=copy__3]
#   %copy__1 : Tensor "f32[1][1]mps:0" = PlaceHolder[target=copy__1]
#   %sum_1 : Tensor "f32[][]mps:0" = PlaceHolder[target=sum_1]
#   %mul : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %mm), kwargs = {})
#   %sum_1 : Tensor "f32[][]mps:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul,), kwargs = {})
#   %div : Tensor "f32[1][1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg2_1, %sum_1), kwargs = {})
#   %mul_2 : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, %div), kwargs = {})
#   %sub : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg4_1, %mul_2), kwargs = {})
#   return %sum_1,%sub
mps_lib_2 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    [[max_total_threads_per_threadgroup(1024)]]
    kernel void generated_kernel(
        device float* out_ptr0,
        device float* out_ptr1,
        constant float* in_ptr0,
        constant float* in_ptr1,
        constant float* in_ptr2,
        constant float* in_ptr3,
        uint2 thread_pos [[thread_position_in_grid]],
        uint2 group_pos [[thread_position_in_threadgroup]]
    ) {
        auto xindex = thread_pos.x;
        auto r0_index = thread_pos.y;
        threadgroup float tmp_acc_0[32];
        float tmp_acc_1 = 0;
        for(auto r0_0_cnt = 0; r0_0_cnt < 8; ++r0_0_cnt) {
            int r0_0 = 8 * r0_index + r0_0_cnt;
            if (r0_0 >= 7424) break;
            auto tmp0 = in_ptr0[r0_0];
            auto tmp1 = in_ptr1[r0_0];
            auto tmp2 = tmp0 * tmp1;
            tmp_acc_1 += tmp2;
        }
        auto tmp3 = c10::metal::threadgroup_sum(tmp_acc_0, tmp_acc_1, r0_index * 1, 1024);
        if (r0_index == 0) out_ptr0[0] = static_cast<float>(tmp3);
        for(auto r0_0_cnt = 0; r0_0_cnt < 8; ++r0_0_cnt) {
            int r0_0 = 8 * r0_index + r0_0_cnt;
            if (r0_0 >= 7424) break;
            auto tmp4 = in_ptr2[r0_0];
            auto tmp5 = in_ptr1[r0_0];
            auto tmp6 = in_ptr3[0];
            auto tmp7 = tmp6 / tmp3;
            auto tmp8 = tmp5 * tmp7;
            auto tmp9 = tmp4 - tmp8;
            out_ptr1[r0_0] = static_cast<float>(tmp9);
        }
    }
''')


# Topologically Sorted Source Nodes: [zero__1, zero__3, x_proj_V, getitem_7, reshape_3, index_select_1, index_add__1], Original ATen: [aten.zero, aten.view, aten.select, aten.index_select, aten.index_add]
# Source node to ATen node mapping:
#   getitem_7 => select_1
#   index_add__1 => index_put_1
#   index_select_1 => index_1
#   reshape_3 => view_32
#   x_proj_V => view_30
#   zero__1 => full_1
#   zero__3 => full_default_1
# Graph fragment:
#   %arg13_1 : Tensor "i64[450][1]mps:0" = PlaceHolder[target=arg13_1]
#   %arg14_1 : Tensor "i64[450][1]mps:0" = PlaceHolder[target=arg14_1]
#   %bmm_2 : Tensor "f32[58, 32, 1][32, 1, 1]mps:0" = PlaceHolder[target=bmm_2]
#   %index_put_1 : Tensor "f32[234, 32][32, 1]mps:0" = PlaceHolder[target=index_put_1]
#   %full_1 : Tensor "f32[64, 32][32, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([64, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: mps:0, pin_memory: False})
#   %full_default_1 : Tensor "f32[234, 32][32, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([234, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: mps:0, pin_memory: False})
#   %view_30 : Tensor "f32[1, 58, 32, 1][1856, 32, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_2, [1, 58, 32, 1]), kwargs = {})
#   %select_1 : Tensor "f32[58, 32, 1][32, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_30, 0, 0), kwargs = {})
#   %view_32 : Tensor "f32[58, 32][32, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [58, 32]), kwargs = {})
#   %slice_scatter_default : Tensor "f32[64, 32][32, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_1, %view_32, 0, 0, 58), kwargs = {})
#   %index_1 : Tensor "f32[450, 32][32, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%slice_scatter_default, [%arg14_1]), kwargs = {})
#   %index_put_1 : Tensor "f32[234, 32][32, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_1, [%arg13_1], %index_1, True), kwargs = {})
#   return %buf7
mps_lib_3 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        constant long* in_ptr0,
        constant long* in_ptr1,
        constant float* in_ptr2,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x1 = c10::metal::floor_divide(xindex, 32);
        int x0 = (xindex) % (32);
        auto tmp0 = in_ptr0[x1];
        auto tmp7 = in_ptr1[x1];
        auto tmp1 = 234;
        auto tmp2 = static_cast<long>(tmp1);
        auto tmp3 = tmp0 + tmp2;
        auto tmp4 = tmp0 < 0;
        auto tmp5 = tmp4 ? tmp3 : tmp0;
        if ((tmp5 < 0) && (tmp5 > 234)) return;
        auto tmp8 = 64;
        auto tmp9 = static_cast<long>(tmp8);
        auto tmp10 = tmp7 + tmp9;
        auto tmp11 = tmp7 < 0;
        auto tmp12 = tmp11 ? tmp10 : tmp7;
        if ((tmp12 < 0) && (tmp12 > 64)) return;
        auto tmp14 = tmp12;
        auto tmp15 = static_cast<long>(tmp14);
        auto tmp16 = 58;
        auto tmp17 = tmp15 < tmp16;
        auto tmp18 = in_ptr2[x0 + 32*tmp12];
        auto tmp19 = tmp17 ? tmp18 : 0.0;
        auto tmp20 = 0.0;
        auto tmp21 = tmp17 ? tmp19 : tmp20;
        c10::metal::AtomicType<float>::atomic_add(reinterpret_cast<device c10::metal::AtomicType<float>::type *>(out_ptr0), x0 + 32*tmp5, static_cast<float>(tmp21));
    }
''')


# Topologically Sorted Source Nodes: [], Original ATen: [aten.copy_]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf7 : Tensor "f32[234, 32][32, 1]mps:0" = PlaceHolder[target=buf7]
#   %copy__10 : Tensor "f32[234, 32][32, 1]mps:0" = PlaceHolder[target=copy__10]
#   %copy__10 : Tensor "f32[234, 32][32, 1]mps:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg16_1, %index_put_1), kwargs = {})
#   return %buf43
mps_lib_4 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        constant float* in_ptr0,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x0 = xindex;
        auto tmp0 = in_ptr0[x0];
        out_ptr0[x0] = static_cast<float>(tmp0);
    }
''')


# Topologically Sorted Source Nodes: [zero__1, x_proj_V, getitem_7, reshape_3], Original ATen: [aten.zero, aten.view, aten.select, aten.copy_]
# Source node to ATen node mapping:
#   getitem_7 => select_1
#   reshape_3 => view_32
#   x_proj_V => view_30
#   zero__1 => full_1
# Graph fragment:
#   %bmm_2 : Tensor "f32[58, 32, 1][32, 1, 1]mps:0" = PlaceHolder[target=bmm_2]
#   %copy__6 : Tensor "f32[64, 32][32, 1]mps:0" = PlaceHolder[target=copy__6]
#   %full_1 : Tensor "f32[64, 32][32, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([64, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: mps:0, pin_memory: False})
#   %view_30 : Tensor "f32[1, 58, 32, 1][1856, 32, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_2, [1, 58, 32, 1]), kwargs = {})
#   %select_1 : Tensor "f32[58, 32, 1][32, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_30, 0, 0), kwargs = {})
#   %view_32 : Tensor "f32[58, 32][32, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [58, 32]), kwargs = {})
#   %slice_scatter_default : Tensor "f32[64, 32][32, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_1, %view_32, 0, 0, 58), kwargs = {})
#   %copy__6 : Tensor "f32[64, 32][32, 1]mps:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg9_1, %slice_scatter_default), kwargs = {})
#   return %buf37
mps_lib_5 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        constant float* in_ptr0,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x1 = c10::metal::floor_divide(xindex, 32);
        int x2 = xindex;
        auto tmp4 = in_ptr0[x2];
        auto tmp0 = x1;
        auto tmp1 = static_cast<long>(tmp0);
        auto tmp2 = 58;
        auto tmp3 = tmp1 < tmp2;
        auto tmp5 = tmp3 ? tmp4 : 0.0;
        auto tmp6 = 0.0;
        auto tmp7 = tmp3 ? tmp5 : tmp6;
        out_ptr0[x2] = static_cast<float>(tmp7);
    }
''')


# Topologically Sorted Source Nodes: [zero_, zero__4, index_select_2, index_add__2, zero__2, x_proj_U, getitem_5, reshape_2, index_select, index_add_], Original ATen: [aten.zero, aten.index_select, aten.index_add, aten.view, aten.select, aten.copy_]
# Source node to ATen node mapping:
#   getitem_5 => select
#   index_add_ => index_put
#   index_add__2 => index_put_2
#   index_select => index
#   index_select_2 => index_2
#   reshape_2 => view_31
#   x_proj_U => view_24
#   zero_ => full
#   zero__2 => full_default
#   zero__4 => full_default_2
# Graph fragment:
#   %arg10_1 : Tensor "i64[450][1]mps:0" = PlaceHolder[target=arg10_1]
#   %arg13_1 : Tensor "i64[450][1]mps:0" = PlaceHolder[target=arg13_1]
#   %buf9 : Tensor "f32[234, 32, 1][7488, 32, 1, 1]mps:0" = PlaceHolder[target=buf9]
#   %index_put_2 : Tensor "f32[1, 64, 32, 1][2048, 32, 1, 1]mps:0" = PlaceHolder[target=index_put_2]
#   %bmm_1 : Tensor "f32[58, 32, 1][32, 1, 1]mps:0" = PlaceHolder[target=bmm_1]
#   %index_put : Tensor "f32[234, 32][32, 1]mps:0" = PlaceHolder[target=index_put]
#   %index : Tensor "f32[450, 32][32, 1]mps:0" = PlaceHolder[target=index]
#   %copy__7 : Tensor "f32[450, 32][32, 1]mps:0" = PlaceHolder[target=copy__7]
#   %full : Tensor "f32[64, 32][32, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([64, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: mps:0, pin_memory: False})
#   %full_default_2 : Tensor "f32[1, 64, 32, 1][2048, 32, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 64, 32, 1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: mps:0, pin_memory: False})
#   %index_2 : Tensor "f32[1, 450, 32, 1][14400, 32, 1, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg17_1, [None, %arg13_1]), kwargs = {})
#   %index_put_2 : Tensor "f32[1, 64, 32, 1][2048, 32, 1, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_2, [None, %arg10_1], %index_2, True), kwargs = {})
#   %full_default : Tensor "f32[234, 32][32, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([234, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: mps:0, pin_memory: False})
#   %view_24 : Tensor "f32[1, 58, 32, 1][1856, 32, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [1, 58, 32, 1]), kwargs = {})
#   %select : Tensor "f32[58, 32, 1][32, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_24, 0, 0), kwargs = {})
#   %view_31 : Tensor "f32[58, 32][32, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select, [58, 32]), kwargs = {})
#   %slice_scatter_default_1 : Tensor "f32[64, 32][32, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full, %view_31, 0, 0, 58), kwargs = {})
#   %index : Tensor "f32[450, 32][32, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%slice_scatter_default_1, [%arg10_1]), kwargs = {})
#   %index_put : Tensor "f32[234, 32][32, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default, [%arg13_1], %index, True), kwargs = {})
#   %copy__7 : Tensor "f32[450, 32][32, 1]mps:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg11_1, %index), kwargs = {})
#   return %buf11,%buf15,%index,%buf39
mps_lib_6 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        device float* out_ptr1,
        device float* out_ptr3,
        constant long* in_ptr0,
        constant long* in_ptr1,
        constant float* in_ptr2,
        constant float* in_ptr3,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x1 = c10::metal::floor_divide(xindex, 32);
        int x0 = (xindex) % (32);
        int x2 = xindex;
        auto tmp0 = in_ptr0[x1];
        auto tmp7 = in_ptr1[x1];
        auto tmp1 = 64;
        auto tmp2 = static_cast<long>(tmp1);
        auto tmp3 = tmp0 + tmp2;
        auto tmp4 = tmp0 < 0;
        auto tmp5 = tmp4 ? tmp3 : tmp0;
        if ((tmp5 < 0) && (tmp5 > 64)) return;
        auto tmp8 = 234;
        auto tmp9 = static_cast<long>(tmp8);
        auto tmp10 = tmp7 + tmp9;
        auto tmp11 = tmp7 < 0;
        auto tmp12 = tmp11 ? tmp10 : tmp7;
        if ((tmp12 < 0) && (tmp12 > 234)) return;
        auto tmp14 = in_ptr2[x0 + 32*tmp12];
        auto tmp15 = tmp5;
        auto tmp16 = static_cast<long>(tmp15);
        auto tmp17 = 58;
        auto tmp18 = tmp16 < tmp17;
        auto tmp19 = in_ptr3[x0 + 32*tmp5];
        auto tmp20 = tmp18 ? tmp19 : 0.0;
        auto tmp21 = 0.0;
        auto tmp22 = tmp18 ? tmp20 : tmp21;
        c10::metal::AtomicType<float>::atomic_add(reinterpret_cast<device c10::metal::AtomicType<float>::type *>(out_ptr0), x0 + 32*tmp5, static_cast<float>(tmp14));
        c10::metal::AtomicType<float>::atomic_add(reinterpret_cast<device c10::metal::AtomicType<float>::type *>(out_ptr1), x0 + 32*tmp12, static_cast<float>(tmp22));
        out_ptr3[x2] = static_cast<float>(tmp22);
    }
''')


# Topologically Sorted Source Nodes: [zero__1, x_proj_V, getitem_7, reshape_3, index_select_1, index_select_2, zero__5, index_select_3, index_add__3], Original ATen: [aten.zero, aten.view, aten.select, aten.index_select, aten.index_add, aten.copy_]
# Source node to ATen node mapping:
#   getitem_7 => select_1
#   index_add__3 => index_put_3
#   index_select_1 => index_1
#   index_select_2 => index_2
#   index_select_3 => index_3
#   reshape_3 => view_32
#   x_proj_V => view_30
#   zero__1 => full_1
#   zero__5 => full_default_3
# Graph fragment:
#   %arg14_1 : Tensor "i64[450][1]mps:0" = PlaceHolder[target=arg14_1]
#   %arg13_1 : Tensor "i64[450][1]mps:0" = PlaceHolder[target=arg13_1]
#   %buf17 : Tensor "f32[234, 32, 1][7488, 32, 1, 1]mps:0" = PlaceHolder[target=buf17]
#   %index_put_3 : Tensor "f32[1, 64, 32, 1][2048, 32, 1, 1]mps:0" = PlaceHolder[target=index_put_3]
#   %bmm_2 : Tensor "f32[58, 32, 1][32, 1, 1]mps:0" = PlaceHolder[target=bmm_2]
#   %index_1 : Tensor "f32[450, 32][32, 1]mps:0" = PlaceHolder[target=index_1]
#   %copy__9 : Tensor "f32[450, 32][32, 1]mps:0" = PlaceHolder[target=copy__9]
#   %buf9 : Tensor "f32[234, 32, 1][7488, 32, 1, 1]mps:0" = PlaceHolder[target=buf9]
#   %index_2 : Tensor "f32[1, 450, 32, 1][14400, 32, 1, 14400]mps:0" = PlaceHolder[target=index_2]
#   %copy__13 : Tensor "f32[1, 450, 32, 1][14400, 32, 1, 1]mps:0" = PlaceHolder[target=copy__13]
#   %index_3 : Tensor "f32[1, 450, 32, 1][14400, 32, 1, 14400]mps:0" = PlaceHolder[target=index_3]
#   %copy__14 : Tensor "f32[1, 450, 32, 1][14400, 32, 1, 1]mps:0" = PlaceHolder[target=copy__14]
#   %full_1 : Tensor "f32[64, 32][32, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([64, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: mps:0, pin_memory: False})
#   %view_30 : Tensor "f32[1, 58, 32, 1][1856, 32, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_2, [1, 58, 32, 1]), kwargs = {})
#   %select_1 : Tensor "f32[58, 32, 1][32, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_30, 0, 0), kwargs = {})
#   %view_32 : Tensor "f32[58, 32][32, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [58, 32]), kwargs = {})
#   %slice_scatter_default : Tensor "f32[64, 32][32, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_1, %view_32, 0, 0, 58), kwargs = {})
#   %index_1 : Tensor "f32[450, 32][32, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%slice_scatter_default, [%arg14_1]), kwargs = {})
#   %index_2 : Tensor "f32[1, 450, 32, 1][14400, 32, 1, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg17_1, [None, %arg13_1]), kwargs = {})
#   %full_default_3 : Tensor "f32[1, 64, 32, 1][2048, 32, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 64, 32, 1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: mps:0, pin_memory: False})
#   %index_3 : Tensor "f32[1, 450, 32, 1][14400, 32, 1, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg18_1, [None, %arg13_1]), kwargs = {})
#   %index_put_3 : Tensor "f32[1, 64, 32, 1][2048, 32, 1, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_3, [None, %arg14_1], %index_3, True), kwargs = {})
#   %copy__9 : Tensor "f32[450, 32][32, 1]mps:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg15_1, %index_1), kwargs = {})
#   %copy__13 : Tensor "f32[1, 450, 32, 1][14400, 32, 1, 1]mps:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg19_1, %index_2), kwargs = {})
#   %copy__14 : Tensor "f32[1, 450, 32, 1][14400, 32, 1, 1]mps:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg20_1, %index_3), kwargs = {})
#   return %buf19,%index_1,%buf42,%index_2,%buf45,%index_3,%buf47
mps_lib_7 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        device float* out_ptr2,
        device float* out_ptr4,
        device float* out_ptr6,
        constant long* in_ptr0,
        constant long* in_ptr1,
        constant float* in_ptr2,
        constant float* in_ptr3,
        constant float* in_ptr4,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x1 = c10::metal::floor_divide(xindex, 32);
        int x0 = (xindex) % (32);
        int x2 = xindex;
        auto tmp0 = in_ptr0[x1];
        auto tmp7 = in_ptr1[x1];
        auto tmp1 = 64;
        auto tmp2 = static_cast<long>(tmp1);
        auto tmp3 = tmp0 + tmp2;
        auto tmp4 = tmp0 < 0;
        auto tmp5 = tmp4 ? tmp3 : tmp0;
        if ((tmp5 < 0) && (tmp5 > 64)) return;
        auto tmp8 = 234;
        auto tmp9 = static_cast<long>(tmp8);
        auto tmp10 = tmp7 + tmp9;
        auto tmp11 = tmp7 < 0;
        auto tmp12 = tmp11 ? tmp10 : tmp7;
        if ((tmp12 < 0) && (tmp12 > 234)) return;
        auto tmp14 = in_ptr2[x0 + 32*tmp12];
        auto tmp15 = tmp5;
        auto tmp16 = static_cast<long>(tmp15);
        auto tmp17 = 58;
        auto tmp18 = tmp16 < tmp17;
        auto tmp19 = in_ptr3[x0 + 32*tmp5];
        auto tmp20 = tmp18 ? tmp19 : 0.0;
        auto tmp21 = 0.0;
        auto tmp22 = tmp18 ? tmp20 : tmp21;
        auto tmp23 = in_ptr4[x0 + 32*tmp12];
        c10::metal::AtomicType<float>::atomic_add(reinterpret_cast<device c10::metal::AtomicType<float>::type *>(out_ptr0), x0 + 32*tmp5, static_cast<float>(tmp14));
        out_ptr2[x2] = static_cast<float>(tmp22);
        out_ptr4[x2] = static_cast<float>(tmp23);
        out_ptr6[x2] = static_cast<float>(tmp14);
    }
''')


# Topologically Sorted Source Nodes: [alpha, mul_1, add_, matmul_1, y_r_final, view_8, y_c_final, view_9, add, add__1, addcmul_, getitem_14, mul_4, unsqueeze, mul_5, rho_new, fill_, beta, mul_, add__2], Original ATen: [aten.div, aten.mul, aten.add, aten.view, aten.addcmul, aten.slice, aten.unsqueeze, aten.sum, aten.fill, aten.copy_]
# Source node to ATen node mapping:
#   add => add_1
#   add_ => add
#   add__1 => add_2
#   add__2 => add_4
#   addcmul_ => add_3, mul_5, mul_6, view_51, view_52
#   alpha => div
#   beta => div_1
#   fill_ => copy_3
#   getitem_14 => slice_15
#   matmul_1 => view_10, view_12
#   mul_ => mul_8
#   mul_1 => mul_1
#   mul_4 => mul_4
#   mul_5 => mul_7
#   rho_new => sum_2
#   unsqueeze => unsqueeze
#   view_8 => view_46
#   view_9 => view_47
#   y_c_final => view_45
#   y_r_final => view_41
# Graph fragment:
#   %bmm : Tensor "f32[58, 128, 1][128, 1, 1]mps:0" = PlaceHolder[target=bmm]
#   %bmm_5 : Tensor "f32[58, 128, 1][128, 1, 1]mps:0" = PlaceHolder[target=bmm_5]
#   %bmm_6 : Tensor "f32[58, 128, 1][128, 1, 1]mps:0" = PlaceHolder[target=bmm_6]
#   %sub : Tensor "f32[7424, 1][1, 1]mps:0" = PlaceHolder[target=sub]
#   %arg6_1 : Tensor "f32[1, 1672448][1672448, 1]mps:0" = PlaceHolder[target=arg6_1]
#   %arg23_1 : Tensor "f32[1, 7424][7424, 1]mps:0" = PlaceHolder[target=arg23_1]
#   %copy__3 : Tensor "f32[7424, 1][1, 1]mps:0" = PlaceHolder[target=copy__3]
#   %add_3 : Tensor "f32[1, 7424, 1][7424, 1, 7424]mps:0" = PlaceHolder[target=add_3]
#   %copy__2 : Tensor "f32[7424, 1][1, 1]mps:0" = PlaceHolder[target=copy__2]
#   %copy_ : Tensor "f32[7424, 1][1, 1]mps:0" = PlaceHolder[target=copy_]
#   %copy__1 : Tensor "f32[1][1]mps:0" = PlaceHolder[target=copy__1]
#   %sum_1 : Tensor "f32[][]mps:0" = PlaceHolder[target=sum_1]
#   %sum_2 : Tensor "f32[][]mps:0" = PlaceHolder[target=sum_2]
#   %add_4 : Tensor "f32[7424, 1][1, 7424]mps:0" = PlaceHolder[target=add_4]
#   %mm : Tensor "f32[7424, 1][1, 1]mps:0" = PlaceHolder[target=mm]
#   %add : Tensor "f32[7424, 1][1, 7424]mps:0" = PlaceHolder[target=add]
#   %copy__4 : Tensor "f32[7424, 1][1, 1]mps:0" = PlaceHolder[target=copy__4]
#   %div : Tensor "f32[1][1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg2_1, %sum_1), kwargs = {})
#   %mul_1 : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %div), kwargs = {})
#   %add : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mul_1), kwargs = {})
#   %view_10 : Tensor "f32[1, 58, 128, 1][7424, 128, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [1, 58, 128, 1]), kwargs = {})
#   %view_12 : Tensor "f32[1, 7424, 1][7424, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_10, [1, 7424, 1]), kwargs = {})
#   %view_41 : Tensor "f32[1, 58, 128, 1][7424, 128, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_5, [1, 58, 128, 1]), kwargs = {})
#   %view_46 : Tensor "f32[1, 7424, 1][7424, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_41, [1, 7424, 1]), kwargs = {})
#   %view_45 : Tensor "f32[1, 58, 128, 1][7424, 128, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_6, [1, 58, 128, 1]), kwargs = {})
#   %view_47 : Tensor "f32[1, 7424, 1][7424, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_45, [1, 7424, 1]), kwargs = {})
#   %add_1 : Tensor "f32[1, 7424, 1][7424, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_46, %view_47), kwargs = {})
#   %add_2 : Tensor "f32[1, 7424, 1][7424, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_12, %add_1), kwargs = {})
#   %view_51 : Tensor "f32[1, 7424, 1][7424, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub, [1, 7424, 1]), kwargs = {})
#   %mul_5 : Tensor "f32[1, 7424, 1][7424, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_51, 1), kwargs = {})
#   %slice_15 : Tensor "f32[1, 7424][1672448, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg6_1, 1, 1665024, 9223372036854775807), kwargs = {})
#   %mul_4 : Tensor "f32[1, 7424][7424, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_15, %arg23_1), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 7424, 1][7424, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_4, -1), kwargs = {})
#   %mul_6 : Tensor "f32[1, 7424, 1][7424, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %unsqueeze), kwargs = {})
#   %add_3 : Tensor "f32[1, 7424, 1][7424, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_6), kwargs = {})
#   %view_52 : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_3, [7424, 1]), kwargs = {})
#   %mul_7 : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %view_52), kwargs = {})
#   %sum_2 : Tensor "f32[][]mps:0"[num_users=2] = call_function[target=torch.ops.aten.sum.default](args = (%mul_7,), kwargs = {})
#   %copy_3 : Tensor "f32[1][1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%arg2_1, %sum_2), kwargs = {})
#   %div_1 : Tensor "f32[1][1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_2, %arg2_1), kwargs = {})
#   %mul_8 : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %div_1), kwargs = {})
#   %add_4 : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %view_52), kwargs = {})
#   %copy_ : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg1_1, %add_4), kwargs = {})
#   %copy__1 : Tensor "f32[1][1]mps:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg2_1, %copy_3), kwargs = {})
#   %copy__2 : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg3_1, %add), kwargs = {})
#   %copy__3 : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg4_1, %sub), kwargs = {})
#   %copy__4 : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg5_1, %view_52), kwargs = {})
#   return %add_3,%buf34,%sum_2,%add,%add_4,%buf28,%buf33,%buf35,%buf32
mps_lib_8 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    [[max_total_threads_per_threadgroup(1024)]]
    kernel void generated_kernel(
        device float* out_ptr0,
        device float* out_ptr1,
        device float* out_ptr3,
        device float* out_ptr5,
        device float* out_ptr6,
        device float* out_ptr7,
        device float* out_ptr8,
        constant float* in_ptr0,
        constant float* in_ptr1,
        constant float* in_ptr2,
        constant float* in_ptr3,
        constant float* in_ptr4,
        constant float* in_ptr5,
        constant float* in_ptr6,
        constant float* in_ptr7,
        constant float* in_ptr8,
        constant float* in_ptr9,
        uint2 thread_pos [[thread_position_in_grid]],
        uint2 group_pos [[thread_position_in_threadgroup]]
    ) {
        auto xindex = thread_pos.x;
        auto r0_index = thread_pos.y;
        threadgroup float tmp_acc_0[32];
        float tmp_acc_1 = 0;
        for(auto r0_0_cnt = 0; r0_0_cnt < 8; ++r0_0_cnt) {
            int r0_0 = 8 * r0_index + r0_0_cnt;
            if (r0_0 >= 7424) break;
            auto tmp0 = in_ptr0[r0_0];
            auto tmp1 = in_ptr1[r0_0];
            auto tmp2 = in_ptr2[r0_0];
            auto tmp5 = in_ptr3[r0_0];
            auto tmp8 = in_ptr4[1665024 + r0_0];
            auto tmp9 = in_ptr5[r0_0];
            auto tmp3 = tmp1 + tmp2;
            auto tmp4 = tmp0 + tmp3;
            auto tmp6 = 1.0;
            auto tmp7 = tmp5 * tmp6;
            auto tmp10 = tmp8 * tmp9;
            auto tmp11 = tmp7 * tmp10;
            auto tmp12 = tmp4 + tmp11;
            out_ptr0[r0_0] = static_cast<float>(tmp12);
            out_ptr1[r0_0] = static_cast<float>(tmp5);
            auto tmp13 = tmp5 * tmp12;
            tmp_acc_1 += tmp13;
        }
        auto tmp14 = c10::metal::threadgroup_sum(tmp_acc_0, tmp_acc_1, r0_index * 1, 1024);
        for(auto r0_0_cnt = 0; r0_0_cnt < 8; ++r0_0_cnt) {
            int r0_0 = 8 * r0_index + r0_0_cnt;
            if (r0_0 >= 7424) break;
            auto tmp15 = in_ptr6[r0_0];
            auto tmp16 = in_ptr7[r0_0];
            auto tmp17 = in_ptr8[0];
            auto tmp18 = in_ptr9[0];
            auto tmp19 = tmp17 / tmp18;
            auto tmp20 = tmp16 * tmp19;
            auto tmp21 = tmp15 + tmp20;
            out_ptr3[r0_0] = static_cast<float>(tmp21);
        }
        for(auto r0_0_cnt = 0; r0_0_cnt < 8; ++r0_0_cnt) {
            int r0_0 = 8 * r0_index + r0_0_cnt;
            if (r0_0 >= 7424) break;
            auto tmp22 = in_ptr7[r0_0];
            auto tmp23 = in_ptr8[0];
            auto tmp26 = out_ptr0[r0_0];
            auto tmp28 = out_ptr3[r0_0];
            auto tmp24 = tmp14 / tmp23;
            auto tmp25 = tmp22 * tmp24;
            auto tmp27 = tmp25 + tmp26;
            out_ptr5[r0_0] = static_cast<float>(tmp27);
            out_ptr6[r0_0] = static_cast<float>(tmp28);
            out_ptr7[r0_0] = static_cast<float>(tmp26);
        }
        out_ptr8[0] = static_cast<float>(tmp14);
    }
''')

def partition_0(args):
    arg0_1, arg1_1, arg6_1, arg7_1, arg4_1, arg2_1, arg13_1, arg14_1, arg16_1, arg9_1, arg17_1, arg10_1, arg11_1, arg21_1, arg12_1, arg8_1, arg18_1, arg15_1, arg19_1, arg20_1, arg23_1, arg3_1, arg5_1, arg22_1 = args
    args.clear()
    assert_size_stride(arg0_1, (7424, 7424), (7424, 1))
    assert_size_stride(arg1_1, (7424, 1), (1, 1))
    assert_size_stride(arg6_1, (1, 1672448), (1672448, 1))
    assert_size_stride(arg7_1, (1, 234, 1, 1), (234, 1, 1, 1))
    assert_size_stride(arg4_1, (7424, 1), (1, 1))
    assert_size_stride(arg2_1, (1, ), (1, ))
    assert_size_stride(arg13_1, (450, ), (1, ))
    assert_size_stride(arg14_1, (450, ), (1, ))
    assert_size_stride(arg16_1, (234, 32), (32, 1))
    assert_size_stride(arg9_1, (64, 32), (32, 1))
    assert_size_stride(arg17_1, (1, 234, 32, 1), (7488, 32, 1, 1))
    assert_size_stride(arg10_1, (450, ), (1, ))
    assert_size_stride(arg11_1, (450, 32), (32, 1))
    assert_size_stride(arg21_1, (1, 64, 32, 1), (2048, 32, 1, 1))
    assert_size_stride(arg12_1, (234, 32), (32, 1))
    assert_size_stride(arg8_1, (64, 32), (32, 1))
    assert_size_stride(arg18_1, (1, 234, 32, 1), (7488, 32, 1, 1))
    assert_size_stride(arg15_1, (450, 32), (32, 1))
    assert_size_stride(arg19_1, (1, 450, 32, 1), (14400, 32, 1, 1))
    assert_size_stride(arg20_1, (1, 450, 32, 1), (14400, 32, 1, 1))
    assert_size_stride(arg23_1, (1, 7424), (7424, 1))
    assert_size_stride(arg3_1, (7424, 1), (1, 1))
    assert_size_stride(arg5_1, (7424, 1), (1, 1))
    assert_size_stride(arg22_1, (1, 64, 32, 1), (2048, 32, 1, 1))
    buf0 = empty_strided((7424, 1), (1, 1), device='mps', dtype=torch.float32)
    # Topologically Sorted Source Nodes: [Ap], Original ATen: [aten.mm]
    extern_kernels.mm(arg0_1, arg1_1, out=buf0)
    del arg0_1
    buf4 = empty_strided((1, 234, 32, 32), (239616, 1024, 32, 1), device='mps', dtype=torch.float32)
    mps_lib_0.generated_kernel(buf4, arg6_1, arg7_1, threads=[239616])
    del arg7_1
    buf6 = empty_strided((234, 32), (32, 1), device='mps', dtype=torch.float32)
    mps_lib_1.generated_kernel(buf6, threads=[7488])
    buf10 = empty_strided((1, 64, 32, 1), (2048, 32, 1, 1), device='mps', dtype=torch.float32)
    mps_lib_1.generated_kernel(buf10, threads=[2048])
    buf14 = empty_strided((234, 32), (32, 1), device='mps', dtype=torch.float32)
    mps_lib_1.generated_kernel(buf14, threads=[7488])
    buf18 = empty_strided((1, 64, 32, 1), (2048, 32, 1, 1), device='mps', dtype=torch.float32)
    mps_lib_1.generated_kernel(buf18, threads=[2048])
    buf1 = empty_strided((), (), device='mps', dtype=torch.float32)
    buf2 = empty_strided((7424, 1), (1, 1), device='mps', dtype=torch.float32)
    mps_lib_2.generated_kernel(buf1, buf2, arg1_1, buf0, arg4_1, arg2_1, threads=[1, 1024], group_size=[1, 1024])
    buf3 = empty_strided((58, 128, 1), (128, 1, 1), device='mps', dtype=torch.float32)
    # Topologically Sorted Source Nodes: [alpha, getitem, diag_blocks, matmul_1, mul_2, sub_], Original ATen: [aten.div, aten.slice, aten.view, aten.mul, aten.sub, aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(arg6_1, (58, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf2, (58, 128, 1), (128, 1, 0), 0), out=buf3)
    buf5 = empty_strided((58, 32, 1), (32, 1, 1), device='mps', dtype=torch.float32)
    # Topologically Sorted Source Nodes: [getitem_3, node_V, transpose_1, x_proj_V], Original ATen: [aten.slice, aten.view, aten.transpose, aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(arg6_1, (58, 32, 128), (4096, 1, 32), 1427456), reinterpret_tensor(buf2, (58, 128, 1), (128, 1, 1), 0), out=buf5)
    mps_lib_3.generated_kernel(buf6, arg13_1, arg14_1, buf5, threads=[14400])
    buf8 = empty_strided((234, 32, 1), (32, 1, 1), device='mps', dtype=torch.float32)
    # Topologically Sorted Source Nodes: [getitem_1, off_diag_blocks, off_diag_blocks_1, Br, bmm], Original ATen: [aten.slice, aten.view, aten.mul, aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf4, (234, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf6, (234, 32, 1), (32, 1, 1), 0), out=buf8)
    mps_lib_4.generated_kernel(arg16_1, buf6, threads=[7488])
    del arg16_1
    del buf6
    buf13 = empty_strided((58, 32, 1), (32, 1, 1), device='mps', dtype=torch.float32)
    # Topologically Sorted Source Nodes: [getitem_2, node_U, transpose, x_proj_U], Original ATen: [aten.slice, aten.view, aten.transpose, aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(arg6_1, (58, 32, 128), (4096, 1, 32), 1189888), reinterpret_tensor(buf2, (58, 128, 1), (128, 1, 1), 0), out=buf13)
    mps_lib_5.generated_kernel(arg9_1, buf5, threads=[2048])
    del arg9_1
    mps_lib_4.generated_kernel(arg17_1, buf8, threads=[7488])
    del buf8
    mps_lib_6.generated_kernel(buf10, buf14, arg11_1, arg10_1, arg13_1, arg17_1, buf13, threads=[14400])
    del arg10_1
    del arg11_1
    buf16 = empty_strided((234, 32, 1), (32, 1, 1), device='mps', dtype=torch.float32)
    # Topologically Sorted Source Nodes: [getitem_1, off_diag_blocks, off_diag_blocks_1, Br, transpose_2, bmm_1], Original ATen: [aten.slice, aten.view, aten.mul, aten.transpose, aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf4, (234, 32, 32), (1024, 1, 32), 0), reinterpret_tensor(buf14, (234, 32, 1), (32, 1, 1), 0), out=buf16)
    del buf4
    buf12 = empty_strided((58, 128, 1), (128, 1, 1), device='mps', dtype=torch.float32)
    # Topologically Sorted Source Nodes: [getitem_2, node_U, y_r_final], Original ATen: [aten.slice, aten.view, aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(arg6_1, (58, 128, 32), (4096, 32, 1), 1189888), reinterpret_tensor(buf10, (58, 32, 1), (32, 1, 1), 0), out=buf12)
    mps_lib_4.generated_kernel(arg21_1, buf10, threads=[2048])
    del arg21_1
    del buf10
    mps_lib_4.generated_kernel(arg12_1, buf14, threads=[7488])
    del arg12_1
    del buf14
    mps_lib_5.generated_kernel(arg8_1, buf13, threads=[2048])
    del arg8_1
    del buf13
    mps_lib_4.generated_kernel(arg18_1, buf16, threads=[7488])
    del buf16
    mps_lib_7.generated_kernel(buf18, arg15_1, arg19_1, arg20_1, arg14_1, arg13_1, arg18_1, buf5, arg17_1, threads=[14400])
    del arg13_1
    del arg14_1
    del arg15_1
    del arg17_1
    del arg18_1
    del arg19_1
    del arg20_1
    del buf5
    buf20 = empty_strided((58, 128, 1), (128, 1, 1), device='mps', dtype=torch.float32)
    # Topologically Sorted Source Nodes: [getitem_3, node_V, y_c_final], Original ATen: [aten.slice, aten.view, aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(arg6_1, (58, 128, 32), (4096, 32, 1), 1427456), reinterpret_tensor(buf18, (58, 32, 1), (32, 1, 1), 0), out=buf20)
    buf21 = empty_strided((1, 7424, 1), (7424, 1, 7424), device='mps', dtype=torch.float32)
    buf25 = empty_strided((7424, 1), (1, 7424), device='mps', dtype=torch.float32)
    mps_lib_8.generated_kernel(buf21, arg4_1, buf25, arg1_1, arg3_1, arg5_1, arg2_1, buf3, buf12, buf20, buf2, arg6_1, arg23_1, arg3_1, arg1_1, arg2_1, buf1, threads=[1, 1024], group_size=[1, 1024])
    del arg1_1
    del arg23_1
    del arg2_1
    del arg3_1
    del arg4_1
    del arg5_1
    del arg6_1
    del buf0
    del buf1
    del buf12
    del buf2
    del buf20
    del buf21
    del buf25
    del buf3
    mps_lib_4.generated_kernel(arg22_1, buf18, threads=[2048])
    del arg22_1
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1 = args
        args.clear()
        partition0_args = [arg0_1, arg1_1, arg6_1, arg7_1, arg4_1, arg2_1, arg13_1, arg14_1, arg16_1, arg9_1, arg17_1, arg10_1, arg11_1, arg21_1, arg12_1, arg8_1, arg18_1, arg15_1, arg19_1, arg20_1, arg23_1, arg3_1, arg5_1, arg22_1]
        del arg0_1, arg1_1, arg6_1, arg7_1, arg4_1, arg2_1, arg13_1, arg14_1, arg16_1, arg9_1, arg17_1, arg10_1, arg11_1, arg21_1, arg12_1, arg8_1, arg18_1, arg15_1, arg19_1, arg20_1, arg23_1, arg3_1, arg5_1, arg22_1
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
    arg6_1 = rand_strided((1, 1672448), (1672448, 1), device='mps:0', dtype=torch.float32)
    arg7_1 = rand_strided((1, 234, 1, 1), (234, 1, 1, 1), device='mps:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, 32), (32, 1), device='mps:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, 32), (32, 1), device='mps:0', dtype=torch.float32)
    arg10_1 = rand_strided((450, ), (1, ), device='mps:0', dtype=torch.int64)
    arg11_1 = rand_strided((450, 32), (32, 1), device='mps:0', dtype=torch.float32)
    arg12_1 = rand_strided((234, 32), (32, 1), device='mps:0', dtype=torch.float32)
    arg13_1 = rand_strided((450, ), (1, ), device='mps:0', dtype=torch.int64)
    arg14_1 = rand_strided((450, ), (1, ), device='mps:0', dtype=torch.int64)
    arg15_1 = rand_strided((450, 32), (32, 1), device='mps:0', dtype=torch.float32)
    arg16_1 = rand_strided((234, 32), (32, 1), device='mps:0', dtype=torch.float32)
    arg17_1 = rand_strided((1, 234, 32, 1), (7488, 32, 1, 1), device='mps:0', dtype=torch.float32)
    arg18_1 = rand_strided((1, 234, 32, 1), (7488, 32, 1, 1), device='mps:0', dtype=torch.float32)
    arg19_1 = rand_strided((1, 450, 32, 1), (14400, 32, 1, 1), device='mps:0', dtype=torch.float32)
    arg20_1 = rand_strided((1, 450, 32, 1), (14400, 32, 1, 1), device='mps:0', dtype=torch.float32)
    arg21_1 = rand_strided((1, 64, 32, 1), (2048, 32, 1, 1), device='mps:0', dtype=torch.float32)
    arg22_1 = rand_strided((1, 64, 32, 1), (2048, 32, 1, 1), device='mps:0', dtype=torch.float32)
    arg23_1 = rand_strided((1, 7424), (7424, 1), device='mps:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
