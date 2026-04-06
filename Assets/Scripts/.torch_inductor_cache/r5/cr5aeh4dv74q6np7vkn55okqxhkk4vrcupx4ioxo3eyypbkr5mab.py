# AOT ID: ['0_inference']
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


# Topologically Sorted Source Nodes: [node_feats, unsqueeze, gf, node_feats_1], Original ATen: [aten.slice, aten.unsqueeze, aten.expand, aten.cat]
# Source node to ATen node mapping:
#   gf => expand
#   node_feats => slice_1
#   node_feats_1 => cat
#   unsqueeze => unsqueeze
# Graph fragment:
#   %arg0_1 : Tensor "f32[1, 7424, 9][67716, 9, 1]mps:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[1, 12][12, 1]mps:0" = PlaceHolder[target=arg1_1]
#   %slice_1 : Tensor "f32[1, 7424, 6][67716, 9, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg0_1, 2, 3, 9223372036854775807), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 1, 12][12, 12, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg1_1, 1), kwargs = {})
#   %expand : Tensor "f32[1, 7424, 12][12, 0, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [-1, 7424, -1]), kwargs = {})
#   %cat : Tensor "f32[1, 7424, 18][133632, 18, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_1, %expand], -1), kwargs = {})
#   return %cat
mps_lib_0 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        constant float* in_ptr0,
        constant float* in_ptr1,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x0 = (xindex) % (18);
        int x1 = c10::metal::floor_divide(xindex, 18);
        int x2 = xindex;
        auto tmp6 = in_ptr0[3 + 9*x1 + (x0)];
        auto tmp11 = in_ptr1[(-6) + x0];
        auto tmp0 = x0;
        auto tmp1 = static_cast<long>(tmp0);
        auto tmp2 = 0;
        auto tmp3 = tmp1 >= tmp2;
        auto tmp4 = 6;
        auto tmp5 = tmp1 < tmp4;
        auto tmp7 = tmp5 ? tmp6 : 0.0;
        auto tmp8 = tmp1 >= tmp4;
        auto tmp9 = 18;
        auto tmp10 = tmp1 < tmp9;
        auto tmp12 = tmp8 ? tmp11 : 0.0;
        auto tmp13 = tmp5 ? tmp7 : tmp12;
        out_ptr0[x2] = static_cast<float>(tmp13);
    }
''')


# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.addmm, aten.view, aten.gelu]
# Source node to ATen node mapping:
#   input_1 => add_tensor_20, view_1
#   input_2 => add, erf, mul, mul_1, mul_2
# Graph fragment:
#   %mm_default_22 : Tensor "f32[7424, 64][64, 1]mps:0" = PlaceHolder[target=mm_default_22]
#   %arg3_1 : Tensor "f32[64][1]mps:0" = PlaceHolder[target=arg3_1]
#   %add_tensor_20 : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_22, %arg3_1), kwargs = {})
#   %view_1 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_20, [1, 7424, 64]), kwargs = {})
#   %mul : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.5), kwargs = {})
#   %mul_1 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_1,), kwargs = {})
#   %add : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_2 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %add), kwargs = {})
#   return %mul_2
mps_lib_1 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        constant float* in_ptr0,
        constant float* in_ptr1,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x2 = xindex;
        int x0 = (xindex) % (64);
        auto tmp0 = in_ptr0[x2];
        auto tmp1 = in_ptr1[x0];
        auto tmp2 = tmp0 + tmp1;
        auto tmp3 = 0.5;
        auto tmp4 = tmp2 * tmp3;
        auto tmp5 = 0.7071067811865476;
        auto tmp6 = tmp2 * tmp5;
        auto tmp7 = c10::metal::erf(tmp6);
        auto tmp8 = 1.0;
        auto tmp9 = tmp7 + tmp8;
        auto tmp10 = tmp4 * tmp9;
        out_ptr0[x2] = static_cast<float>(tmp10);
    }
''')


# Topologically Sorted Source Nodes: [aggr], Original ATen: [aten.zeros_like]
# Source node to ATen node mapping:
#   aggr => full_default
# Graph fragment:
#   %full_default : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([7424, 64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: mps:0, pin_memory: False})
#   return %index_put
mps_lib_2 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x0 = (xindex) % (64);
        int x1 = c10::metal::floor_divide(xindex, 64);
        auto tmp0 = 0.0;
        out_ptr0[x0 + 128*x1] = static_cast<float>(tmp0);
    }
''')


# Topologically Sorted Source Nodes: [aggr, row, col, getitem_4, unsqueeze_1, messages, index_add_], Original ATen: [aten.zeros_like, aten.select, aten.index, aten.unsqueeze, aten.mul, aten.index_add]
# Source node to ATen node mapping:
#   aggr => full_default
#   col => select_1
#   getitem_4 => index
#   index_add_ => index_put
#   messages => mul_3
#   row => select
#   unsqueeze_1 => unsqueeze_1
# Graph fragment:
#   %arg6_1 : Tensor "i64[2, 49401][49401, 1]mps:0" = PlaceHolder[target=arg6_1]
#   %addmm_2 : Tensor "f32[7424, 64][64, 1]mps:0" = PlaceHolder[target=addmm_2]
#   %arg7_1 : Tensor "f32[49401][1]mps:0" = PlaceHolder[target=arg7_1]
#   %index_put : Tensor "f32[7424, 64][128, 1]mps:0" = PlaceHolder[target=index_put]
#   %full_default : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([7424, 64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: mps:0, pin_memory: False})
#   %select : Tensor "i64[49401][1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg6_1, 0, 0), kwargs = {})
#   %select_1 : Tensor "i64[49401][1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg6_1, 0, 1), kwargs = {})
#   %index : Tensor "f32[49401, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%addmm_2, [%select_1]), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[49401, 1][1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, -1), kwargs = {})
#   %mul_3 : Tensor "f32[49401, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index, %unsqueeze_1), kwargs = {})
#   %index_put : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default, [%select], %mul_3, True), kwargs = {})
#   return %buf7
mps_lib_3 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        constant long* in_ptr0,
        constant float* in_ptr1,
        constant float* in_ptr2,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x1 = c10::metal::floor_divide(xindex, 64);
        int x0 = (xindex) % (64);
        auto tmp0 = in_ptr0[x1];
        auto tmp7 = in_ptr0[49401 + x1];
        auto tmp13 = in_ptr2[x1];
        auto tmp1 = 7424;
        auto tmp2 = static_cast<long>(tmp1);
        auto tmp3 = tmp0 + tmp2;
        auto tmp4 = tmp0 < 0;
        auto tmp5 = tmp4 ? tmp3 : tmp0;
        if ((tmp5 < 0) && (tmp5 > 7424)) return;
        auto tmp8 = tmp7 + tmp2;
        auto tmp9 = tmp7 < 0;
        auto tmp10 = tmp9 ? tmp8 : tmp7;
        if ((tmp10 < 0) && (tmp10 > 7424)) return;
        auto tmp12 = in_ptr1[x0 + 64*tmp10];
        auto tmp14 = tmp12 * tmp13;
        c10::metal::AtomicType<float>::atomic_add(reinterpret_cast<device c10::metal::AtomicType<float>::type *>(out_ptr0), x0 + 128*tmp5, static_cast<float>(tmp14));
    }
''')


# Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.addmm, aten.gelu]
# Source node to ATen node mapping:
#   input_4 => add_tensor_19
#   input_5 => add_1, erf_1, mul_4, mul_5, mul_6
# Graph fragment:
#   %mm_default_21 : Tensor "f32[7424, 64][64, 1]mps:0" = PlaceHolder[target=mm_default_21]
#   %arg13_1 : Tensor "f32[64][1]mps:0" = PlaceHolder[target=arg13_1]
#   %add_tensor_19 : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_21, %arg13_1), kwargs = {})
#   %mul_4 : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_19, 0.5), kwargs = {})
#   %mul_5 : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_19, 0.7071067811865476), kwargs = {})
#   %erf_1 : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_5,), kwargs = {})
#   %add_1 : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_1, 1), kwargs = {})
#   %mul_6 : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_1), kwargs = {})
#   return %mul_6
mps_lib_4 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        constant float* in_ptr0,
        constant float* in_ptr1,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x2 = xindex;
        int x0 = (xindex) % (64);
        auto tmp0 = in_ptr0[x2];
        auto tmp1 = in_ptr1[x0];
        auto tmp2 = tmp0 + tmp1;
        auto tmp3 = 0.5;
        auto tmp4 = tmp2 * tmp3;
        auto tmp5 = 0.7071067811865476;
        auto tmp6 = tmp2 * tmp5;
        auto tmp7 = c10::metal::erf(tmp6);
        auto tmp8 = 1.0;
        auto tmp9 = tmp7 + tmp8;
        auto tmp10 = tmp4 * tmp9;
        out_ptr0[x2] = static_cast<float>(tmp10);
    }
''')


# Topologically Sorted Source Nodes: [input_3, x_flat, input_6, out], Original ATen: [aten.view, aten.squeeze, aten.addmm, aten.add]
# Source node to ATen node mapping:
#   input_3 => view_3
#   input_6 => add_tensor_18
#   out => add_2
#   x_flat => squeeze
# Graph fragment:
#   %addmm_1 : Tensor "f32[7424, 64][64, 1]mps:0" = PlaceHolder[target=addmm_1]
#   %mm_default_20 : Tensor "f32[7424, 64][64, 1]mps:0" = PlaceHolder[target=mm_default_20]
#   %arg15_1 : Tensor "f32[64][1]mps:0" = PlaceHolder[target=arg15_1]
#   %view_3 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [1, 7424, 64]), kwargs = {})
#   %squeeze : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=3] = call_function[target=torch.ops.aten.squeeze.dim](args = (%view_3, 0), kwargs = {})
#   %add_tensor_18 : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_20, %arg15_1), kwargs = {})
#   %add_2 : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze, %add_tensor_18), kwargs = {})
#   return %add_2
mps_lib_5 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        constant float* in_ptr0,
        constant float* in_ptr1,
        constant float* in_ptr2,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x2 = xindex;
        int x0 = (xindex) % (64);
        auto tmp0 = in_ptr0[x2];
        auto tmp1 = in_ptr1[x2];
        auto tmp2 = in_ptr2[x0];
        auto tmp3 = tmp1 + tmp2;
        auto tmp4 = tmp0 + tmp3;
        out_ptr0[x2] = static_cast<float>(tmp4);
    }
''')


# Topologically Sorted Source Nodes: [aggr_1], Original ATen: [aten.zeros_like]
# Source node to ATen node mapping:
#   aggr_1 => full_default_1
# Graph fragment:
#   %full_default_1 : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([7424, 64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: mps:0, pin_memory: False})
#   return %index_put_1
mps_lib_6 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x0 = (xindex) % (64);
        int x1 = c10::metal::floor_divide(xindex, 64);
        auto tmp0 = 0.0;
        out_ptr0[x0 + 128*x1] = static_cast<float>(tmp0);
    }
''')


# Topologically Sorted Source Nodes: [input_3, x_flat, input_6, out, h, x_flat_1, input_9, out_1, h_1, h_2], Original ATen: [aten.view, aten.squeeze, aten.addmm, aten.add, aten.unsqueeze, aten.native_layer_norm]
# Source node to ATen node mapping:
#   h => unsqueeze_2
#   h_1 => unsqueeze_4
#   h_2 => add_5, add_6, mul_11, mul_12, rsqrt, sub, var_mean
#   input_3 => view_3
#   input_6 => add_tensor_18
#   input_9 => add_tensor_16
#   out => add_2
#   out_1 => add_4
#   x_flat => squeeze
#   x_flat_1 => squeeze_1
# Graph fragment:
#   %add_2 : Tensor "f32[7424, 64][64, 1]mps:0" = PlaceHolder[target=add_2]
#   %mm_default_18 : Tensor "f32[7424, 64][64, 1]mps:0" = PlaceHolder[target=mm_default_18]
#   %arg23_1 : Tensor "f32[64][1]mps:0" = PlaceHolder[target=arg23_1]
#   %getitem_1 : Tensor "f32[1, 7424, 1][7424, 1, 7424]mps:0" = PlaceHolder[target=getitem_1]
#   %buf22 : Tensor "f32[1, 7424, 1][7424, 1, 7424]mps:0" = PlaceHolder[target=buf22]
#   %arg24_1 : Tensor "f32[64][1]mps:0" = PlaceHolder[target=arg24_1]
#   %arg25_1 : Tensor "f32[64][1]mps:0" = PlaceHolder[target=arg25_1]
#   %view_3 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [1, 7424, 64]), kwargs = {})
#   %squeeze : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=3] = call_function[target=torch.ops.aten.squeeze.dim](args = (%view_3, 0), kwargs = {})
#   %add_tensor_18 : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_20, %arg15_1), kwargs = {})
#   %add_2 : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze, %add_tensor_18), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_2, 0), kwargs = {})
#   %squeeze_1 : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=3] = call_function[target=torch.ops.aten.squeeze.dim](args = (%unsqueeze_2, 0), kwargs = {})
#   %add_tensor_16 : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_18, %arg23_1), kwargs = {})
#   %add_4 : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze_1, %add_tensor_16), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_4, 0), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%unsqueeze_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_4, %getitem_1), kwargs = {})
#   %add_5 : Tensor "f32[1, 7424, 1][7424, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[1, 7424, 1][7424, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_11 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_12 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %arg24_1), kwargs = {})
#   %add_6 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %arg25_1), kwargs = {})
#   return %getitem_1,%buf22,%add_6
mps_lib_7 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    [[max_total_threads_per_threadgroup(64)]]
    kernel void generated_kernel(
        device float* out_ptr2,
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
        int r0_1 = r0_index;
        int x0 = xindex;
        threadgroup float tmp_acc_0[64];
        auto tmp0 = in_ptr0[r0_1 + 64*x0];
        auto tmp1 = in_ptr1[r0_1 + 64*x0];
        auto tmp2 = in_ptr2[r0_1];
        auto tmp3 = tmp1 + tmp2;
        auto tmp4 = tmp0 + tmp3;
        tmp_acc_0[r0_index * 1] = tmp4;
        auto tmp5 = c10::metal::threadgroup_welford_reduce(tmp_acc_0, 64);
        auto tmp13 = in_ptr3[r0_1];
        auto tmp15 = in_ptr4[r0_1];
        auto tmp6 = tmp4 - tmp5.x;
        auto tmp7 = 64.0;
        auto tmp8 = tmp5.y / tmp7;
        auto tmp9 = 1e-05;
        auto tmp10 = tmp8 + tmp9;
        auto tmp11 = metal::rsqrt(tmp10);
        auto tmp12 = tmp6 * tmp11;
        auto tmp14 = tmp12 * tmp13;
        auto tmp16 = tmp14 + tmp15;
        out_ptr2[r0_1 + 64*x0] = static_cast<float>(tmp16);
    }
''')


# Topologically Sorted Source Nodes: [h_3, layer_norm_1], Original ATen: [aten.view, aten.native_layer_norm]
# Source node to ATen node mapping:
#   h_3 => view_5
#   layer_norm_1 => add_8, add_9, mul_13, mul_14, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %addmm_10 : Tensor "f32[7424, 64][64, 1]mps:0" = PlaceHolder[target=addmm_10]
#   %getitem_3 : Tensor "f32[1, 7424, 1][7424, 1, 7424]mps:0" = PlaceHolder[target=getitem_3]
#   %buf27 : Tensor "f32[1, 7424, 1][7424, 1, 7424]mps:0" = PlaceHolder[target=buf27]
#   %arg32_1 : Tensor "f32[64][1]mps:0" = PlaceHolder[target=arg32_1]
#   %arg33_1 : Tensor "f32[64][1]mps:0" = PlaceHolder[target=arg33_1]
#   %view_5 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_10, [1, 7424, 64]), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %getitem_3), kwargs = {})
#   %add_8 : Tensor "f32[1, 7424, 1][7424, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 7424, 1][7424, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %mul_13 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_14 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %arg32_1), kwargs = {})
#   %add_9 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %arg33_1), kwargs = {})
#   return %getitem_3,%buf27,%add_9
mps_lib_8 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    [[max_total_threads_per_threadgroup(64)]]
    kernel void generated_kernel(
        device float* out_ptr2,
        constant float* in_ptr0,
        constant float* in_ptr1,
        constant float* in_ptr2,
        uint2 thread_pos [[thread_position_in_grid]],
        uint2 group_pos [[thread_position_in_threadgroup]]
    ) {
        auto xindex = thread_pos.x;
        auto r0_index = thread_pos.y;
        int r0_1 = r0_index;
        int x0 = xindex;
        threadgroup float tmp_acc_0[64];
        auto tmp0 = in_ptr0[r0_1 + 64*x0];
        tmp_acc_0[r0_index * 1] = tmp0;
        auto tmp1 = c10::metal::threadgroup_welford_reduce(tmp_acc_0, 64);
        auto tmp9 = in_ptr1[r0_1];
        auto tmp11 = in_ptr2[r0_1];
        auto tmp2 = tmp0 - tmp1.x;
        auto tmp3 = 64.0;
        auto tmp4 = tmp1.y / tmp3;
        auto tmp5 = 1e-05;
        auto tmp6 = tmp4 + tmp5;
        auto tmp7 = metal::rsqrt(tmp6);
        auto tmp8 = tmp2 * tmp7;
        auto tmp10 = tmp8 * tmp9;
        auto tmp12 = tmp10 + tmp11;
        out_ptr2[r0_1 + 64*x0] = static_cast<float>(tmp12);
    }
''')


# Topologically Sorted Source Nodes: [qkv_x, getitem_10, q, reshape_4, transpose, q_f], Original ATen: [aten.addmm, aten.view, aten.slice, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   getitem_10 => slice_2
#   q => view_25
#   q_f => clone_1
#   qkv_x => add_tensor_15, view_24
#   reshape_4 => view_28
#   transpose => permute_22
# Graph fragment:
#   %mm_default_17 : Tensor "f32[7424, 192][192, 1]mps:0" = PlaceHolder[target=mm_default_17]
#   %arg35_1 : Tensor "f32[192][1]mps:0" = PlaceHolder[target=arg35_1]
#   %add_tensor_15 : Tensor "f32[7424, 192][192, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_17, %arg35_1), kwargs = {})
#   %view_24 : Tensor "f32[1, 58, 128, 192][1425408, 24576, 192, 1]mps:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_15, [1, 58, 128, 192]), kwargs = {})
#   %slice_2 : Tensor "f32[1, 58, 128, 64][1425408, 24576, 192, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_24, 3, 0, 64), kwargs = {})
#   %view_25 : Tensor "f32[1, 58, 128, 8, 8][1425408, 24576, 192, 8, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%slice_2, [1, 58, 128, 8, 8]), kwargs = {})
#   %view_28 : Tensor "f32[58, 128, 8, 8][24576, 192, 8, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_25, [58, 128, 8, 8]), kwargs = {})
#   %permute_22 : Tensor "f32[58, 8, 128, 8][24576, 8, 192, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [0, 2, 1, 3]), kwargs = {})
#   %clone_1 : Tensor "f32[58, 8, 128, 8][8192, 1024, 8, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_22,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
mps_lib_9 = compile_mps_shader('''
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
        int x0 = (xindex) % (64);
        int x1 = c10::metal::floor_divide(xindex, 64);
        int x2 = xindex;
        auto tmp0 = in_ptr0[x0 + 192*x1];
        auto tmp1 = in_ptr1[x0];
        auto tmp2 = tmp0 + tmp1;
        out_ptr0[x2] = static_cast<float>(tmp2);
    }
''')


# Topologically Sorted Source Nodes: [qkv_x, getitem_11, k, reshape_5, transpose_1, k_f], Original ATen: [aten.addmm, aten.view, aten.slice, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   getitem_11 => slice_3
#   k => view_26
#   k_f => clone_2
#   qkv_x => add_tensor_15, view_24
#   reshape_5 => view_29
#   transpose_1 => permute_23
# Graph fragment:
#   %mm_default_17 : Tensor "f32[7424, 192][192, 1]mps:0" = PlaceHolder[target=mm_default_17]
#   %arg35_1 : Tensor "f32[192][1]mps:0" = PlaceHolder[target=arg35_1]
#   %add_tensor_15 : Tensor "f32[7424, 192][192, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_17, %arg35_1), kwargs = {})
#   %view_24 : Tensor "f32[1, 58, 128, 192][1425408, 24576, 192, 1]mps:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_15, [1, 58, 128, 192]), kwargs = {})
#   %slice_3 : Tensor "f32[1, 58, 128, 64][1425408, 24576, 192, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_24, 3, 64, 128), kwargs = {})
#   %view_26 : Tensor "f32[1, 58, 128, 8, 8][1425408, 24576, 192, 8, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%slice_3, [1, 58, 128, 8, 8]), kwargs = {})
#   %view_29 : Tensor "f32[58, 128, 8, 8][24576, 192, 8, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_26, [58, 128, 8, 8]), kwargs = {})
#   %permute_23 : Tensor "f32[58, 8, 128, 8][24576, 8, 192, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_29, [0, 2, 1, 3]), kwargs = {})
#   %clone_2 : Tensor "f32[58, 8, 128, 8][8192, 1024, 8, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_23,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
mps_lib_10 = compile_mps_shader('''
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
        int x0 = (xindex) % (64);
        int x1 = c10::metal::floor_divide(xindex, 64);
        int x2 = xindex;
        auto tmp0 = in_ptr0[64 + x0 + 192*x1];
        auto tmp1 = in_ptr1[64 + x0];
        auto tmp2 = tmp0 + tmp1;
        out_ptr0[x2] = static_cast<float>(tmp2);
    }
''')


# Topologically Sorted Source Nodes: [qkv_x, getitem_12, v_spatial, reshape_6, transpose_2, v_f], Original ATen: [aten.addmm, aten.view, aten.slice, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   getitem_12 => slice_4
#   qkv_x => add_tensor_15, view_24
#   reshape_6 => view_30
#   transpose_2 => permute_24
#   v_f => clone_3
#   v_spatial => view_27
# Graph fragment:
#   %mm_default_17 : Tensor "f32[7424, 192][192, 1]mps:0" = PlaceHolder[target=mm_default_17]
#   %arg35_1 : Tensor "f32[192][1]mps:0" = PlaceHolder[target=arg35_1]
#   %add_tensor_15 : Tensor "f32[7424, 192][192, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_17, %arg35_1), kwargs = {})
#   %view_24 : Tensor "f32[1, 58, 128, 192][1425408, 24576, 192, 1]mps:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_15, [1, 58, 128, 192]), kwargs = {})
#   %slice_4 : Tensor "f32[1, 58, 128, 64][1425408, 24576, 192, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_24, 3, 128, 192), kwargs = {})
#   %view_27 : Tensor "f32[1, 58, 128, 8, 8][1425408, 24576, 192, 8, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%slice_4, [1, 58, 128, 8, 8]), kwargs = {})
#   %view_30 : Tensor "f32[58, 128, 8, 8][24576, 192, 8, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_27, [58, 128, 8, 8]), kwargs = {})
#   %permute_24 : Tensor "f32[58, 8, 128, 8][24576, 8, 192, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_30, [0, 2, 1, 3]), kwargs = {})
#   %clone_3 : Tensor "f32[58, 8, 128, 8][8192, 1024, 8, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_24,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
mps_lib_11 = compile_mps_shader('''
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
        int x0 = (xindex) % (64);
        int x1 = c10::metal::floor_divide(xindex, 64);
        int x2 = xindex;
        auto tmp0 = in_ptr0[128 + x0 + 192*x1];
        auto tmp1 = in_ptr1[128 + x0];
        auto tmp2 = tmp0 + tmp1;
        out_ptr0[x2] = static_cast<float>(tmp2);
    }
''')


# Topologically Sorted Source Nodes: [unsqueeze_9, expand_3, arange_s, setitem], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.scalar_tensor, aten.index_put]
# Source node to ATen node mapping:
#   arange_s => iota
#   expand_3 => expand_3
#   setitem => full_default_2, index_put_2
#   unsqueeze_9 => unsqueeze_17
# Graph fragment:
#   %arg31_1 : Tensor "f32[58, 128, 128, 4][65536, 512, 4, 1]mps:0" = PlaceHolder[target=arg31_1]
#   %unsqueeze_17 : Tensor "f32[1, 58, 128, 128, 4][3801088, 65536, 512, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg31_1, 0), kwargs = {})
#   %expand_3 : Tensor "f32[1, 58, 128, 128, 4][3801088, 65536, 512, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_17, [1, -1, -1, -1, -1]), kwargs = {})
#   %iota : Tensor "i64[128][1]mps:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: mps:0, requires_grad: False})
#   %full_default_2 : Tensor "f32[][]mps:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: mps:0, pin_memory: False})
#   %index_put_2 : Tensor "f32[1, 58, 128, 128, 4][3801088, 65536, 512, 4, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.index_put.default](args = (%expand_3, [None, None, %iota, %iota], %full_default_2), kwargs = {})
#   return %index_put_2
mps_lib_12 = compile_mps_shader('''
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


# Topologically Sorted Source Nodes: [unsqueeze_9, expand_3, arange_s, setitem], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.scalar_tensor, aten.index_put]
# Source node to ATen node mapping:
#   arange_s => iota
#   expand_3 => expand_3
#   setitem => full_default_2, index_put_2
#   unsqueeze_9 => unsqueeze_17
# Graph fragment:
#   %index_put_2 : Tensor "f32[1, 58, 128, 128, 4][3801088, 65536, 512, 4, 1]mps:0" = PlaceHolder[target=index_put_2]
#   %unsqueeze_17 : Tensor "f32[1, 58, 128, 128, 4][3801088, 65536, 512, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg31_1, 0), kwargs = {})
#   %expand_3 : Tensor "f32[1, 58, 128, 128, 4][3801088, 65536, 512, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_17, [1, -1, -1, -1, -1]), kwargs = {})
#   %iota : Tensor "i64[128][1]mps:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: mps:0, requires_grad: False})
#   %full_default_2 : Tensor "f32[][]mps:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: mps:0, pin_memory: False})
#   %index_put_2 : Tensor "f32[1, 58, 128, 128, 4][3801088, 65536, 512, 4, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.index_put.default](args = (%expand_3, [None, None, %iota, %iota], %full_default_2), kwargs = {})
#   return %buf32
mps_lib_13 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x0 = (xindex) % (4);
        int x1 = ((xindex) / (4)) % (128);
        int x2 = c10::metal::floor_divide(xindex, 512);
        auto tmp0 = 0.0;
        out_ptr0[x0 + 516*x1 + 65536*x2] = static_cast<float>(tmp0);
    }
''')


# Topologically Sorted Source Nodes: [arange_s, setitem_1], Original ATen: [aten.arange, aten.select, aten.scalar_tensor, aten.index_put]
# Source node to ATen node mapping:
#   arange_s => iota
#   setitem_1 => full_default_3, index_put_3, select_5
# Graph fragment:
#   %buf32 : Tensor "f32[1, 58, 128, 128, 4][3801088, 65536, 512, 4, 1]mps:0" = PlaceHolder[target=buf32]
#   %iota : Tensor "i64[128][1]mps:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: mps:0, requires_grad: False})
#   %select_5 : Tensor "f32[1, 58, 128, 128][3801088, 65536, 512, 4]mps:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%index_put_2, 4, 3), kwargs = {})
#   %full_default_3 : Tensor "f32[][]mps:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: mps:0, pin_memory: False})
#   %index_put_3 : Tensor "f32[1, 58, 128, 128][950272, 16384, 128, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_5, [None, None, %iota, %iota], %full_default_3), kwargs = {})
#   return %index_put_3
mps_lib_14 = compile_mps_shader('''
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
        auto tmp0 = in_ptr0[3 + 4*x0];
        out_ptr0[x0] = static_cast<float>(tmp0);
    }
''')


# Topologically Sorted Source Nodes: [arange_s, setitem_1], Original ATen: [aten.arange, aten.select, aten.scalar_tensor, aten.index_put]
# Source node to ATen node mapping:
#   arange_s => iota
#   setitem_1 => full_default_3, index_put_3, select_5
# Graph fragment:
#   %index_put_3 : Tensor "f32[1, 58, 128, 128][950272, 16384, 128, 1]mps:0" = PlaceHolder[target=index_put_3]
#   %iota : Tensor "i64[128][1]mps:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: mps:0, requires_grad: False})
#   %select_5 : Tensor "f32[1, 58, 128, 128][3801088, 65536, 512, 4]mps:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%index_put_2, 4, 3), kwargs = {})
#   %full_default_3 : Tensor "f32[][]mps:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: mps:0, pin_memory: False})
#   %index_put_3 : Tensor "f32[1, 58, 128, 128][950272, 16384, 128, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_5, [None, None, %iota, %iota], %full_default_3), kwargs = {})
#   return %buf34
mps_lib_15 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x0 = (xindex) % (128);
        int x1 = c10::metal::floor_divide(xindex, 128);
        auto tmp0 = 1.0;
        out_ptr0[129*x0 + 16384*x1] = static_cast<float>(tmp0);
    }
''')


# Topologically Sorted Source Nodes: [logit_bias], Original ATen: [aten.select, aten.view, aten.expand, aten.clone]
# Source node to ATen node mapping:
#   logit_bias => clone_4, expand_5, select_8, view_32
# Graph fragment:
#   %buf34 : Tensor "f32[1, 58, 128, 128][950272, 16384, 128, 1]mps:0" = PlaceHolder[target=buf34]
#   %buf32 : Tensor "f32[1, 58, 128, 128, 4][3801088, 65536, 512, 4, 1]mps:0" = PlaceHolder[target=buf32]
#   %select_scatter_default : Tensor "f32[1, 58, 128, 128, 4][3801088, 65536, 512, 4, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%index_put_2, %index_put_3, 4, 3), kwargs = {})
#   %select_8 : Tensor "f32[1, 58, 128, 128][3801088, 65536, 512, 4]mps:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_scatter_default, 4, 3), kwargs = {})
#   %view_32 : Tensor "f32[58, 1, 128, 128][65536, 65536, 512, 4]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_8, [58, 1, 128, 128]), kwargs = {})
#   %expand_5 : Tensor "f32[58, 8, 128, 128][65536, 0, 512, 4]mps:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_32, [58, 8, 128, 128]), kwargs = {})
#   %clone_4 : Tensor "f32[58, 8, 128, 128][131072, 16384, 128, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {})
#   return %clone_4
mps_lib_16 = compile_mps_shader('''
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
        int x0 = (xindex) % (16384);
        int x2 = c10::metal::floor_divide(xindex, 131072);
        int x3 = xindex;
        auto tmp2 = in_ptr0[x0 + 16384*x2];
        auto tmp3 = in_ptr1[3 + 4*x0 + 65536*x2];
        auto tmp0 = 3;
        auto tmp1 = tmp0 == tmp0;
        auto tmp4 = tmp1 ? tmp2 : tmp3;
        out_ptr0[x3] = static_cast<float>(tmp4);
    }
''')


# Topologically Sorted Source Nodes: [qkv_x, getitem_12, v_spatial, einsum_2], Original ATen: [aten.addmm, aten.view, aten.slice, aten.unsqueeze, aten.permute, aten.clone]
# Source node to ATen node mapping:
#   einsum_2 => clone_6, permute_29, permute_31, unsqueeze_19
#   getitem_12 => slice_4
#   qkv_x => add_tensor_15, view_24
#   v_spatial => view_27
# Graph fragment:
#   %mm_default_17 : Tensor "f32[7424, 192][192, 1]mps:0" = PlaceHolder[target=mm_default_17]
#   %arg35_1 : Tensor "f32[192][1]mps:0" = PlaceHolder[target=arg35_1]
#   %add_tensor_15 : Tensor "f32[7424, 192][192, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_17, %arg35_1), kwargs = {})
#   %view_24 : Tensor "f32[1, 58, 128, 192][1425408, 24576, 192, 1]mps:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_15, [1, 58, 128, 192]), kwargs = {})
#   %slice_4 : Tensor "f32[1, 58, 128, 64][1425408, 24576, 192, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_24, 3, 128, 192), kwargs = {})
#   %view_27 : Tensor "f32[1, 58, 128, 8, 8][1425408, 24576, 192, 8, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%slice_4, [1, 58, 128, 8, 8]), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 58, 128, 8, 8, 1][1425408, 24576, 192, 8, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_27, 5), kwargs = {})
#   %permute_29 : Tensor "f32[1, 58, 1, 8, 8, 128][1425408, 24576, 1, 8, 1, 192]mps:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_19, [0, 1, 5, 3, 4, 2]), kwargs = {})
#   %permute_31 : Tensor "f32[58, 8, 128, 1, 8, 1][24576, 8, 192, 1425408, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_29, [1, 3, 5, 0, 4, 2]), kwargs = {})
#   %clone_6 : Tensor "f32[58, 8, 128, 1, 8, 1][8192, 1024, 8, 8, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_31,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_6
mps_lib_17 = compile_mps_shader('''
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
        int x0 = (xindex) % (8);
        int x1 = ((xindex) / (8)) % (128);
        int x2 = ((xindex) / (1024)) % (8);
        int x3 = c10::metal::floor_divide(xindex, 8192);
        int x4 = xindex;
        auto tmp0 = in_ptr0[128 + x0 + 8*x2 + 192*x1 + 24576*x3];
        auto tmp1 = in_ptr1[128 + x0 + 8*x2];
        auto tmp2 = tmp0 + tmp1;
        out_ptr0[x4] = static_cast<float>(tmp2);
    }
''')


# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %buf34 : Tensor "f32[1, 58, 128, 128][950272, 16384, 128, 1]mps:0" = PlaceHolder[target=buf34]
#   %buf32 : Tensor "f32[1, 58, 128, 128, 4][3801088, 65536, 512, 4, 1]mps:0" = PlaceHolder[target=buf32]
#   %select_scatter_default : Tensor "f32[1, 58, 128, 128, 4][3801088, 65536, 512, 4, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%index_put_2, %index_put_3, 4, 3), kwargs = {})
#   return %select_scatter_default
mps_lib_18 = compile_mps_shader('''
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
        int x0 = (xindex) % (4);
        int x1 = c10::metal::floor_divide(xindex, 4);
        int x2 = xindex;
        auto tmp4 = in_ptr0[x1];
        auto tmp5 = in_ptr1[x2];
        auto tmp0 = x0;
        auto tmp1 = static_cast<int>(tmp0);
        auto tmp2 = 3;
        auto tmp3 = tmp1 == tmp2;
        auto tmp6 = tmp3 ? tmp4 : tmp5;
        out_ptr0[x2] = static_cast<float>(tmp6);
    }
''')


# Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten.addmm, aten.view, aten.gelu]
# Source node to ATen node mapping:
#   input_10 => add_tensor_14, view_35
#   input_11 => add_10, erf_3, mul_15, mul_16, mul_17
# Graph fragment:
#   %mm_default_16 : Tensor "f32[950272, 16][16, 1]mps:0" = PlaceHolder[target=mm_default_16]
#   %arg37_1 : Tensor "f32[16][1]mps:0" = PlaceHolder[target=arg37_1]
#   %add_tensor_14 : Tensor "f32[950272, 16][16, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_16, %arg37_1), kwargs = {})
#   %view_35 : Tensor "f32[1, 58, 128, 128, 16][15204352, 262144, 2048, 16, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_14, [1, 58, 128, 128, 16]), kwargs = {})
#   %mul_15 : Tensor "f32[1, 58, 128, 128, 16][15204352, 262144, 2048, 16, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_35, 0.5), kwargs = {})
#   %mul_16 : Tensor "f32[1, 58, 128, 128, 16][15204352, 262144, 2048, 16, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_35, 0.7071067811865476), kwargs = {})
#   %erf_3 : Tensor "f32[1, 58, 128, 128, 16][15204352, 262144, 2048, 16, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_16,), kwargs = {})
#   %add_10 : Tensor "f32[1, 58, 128, 128, 16][15204352, 262144, 2048, 16, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_3, 1), kwargs = {})
#   %mul_17 : Tensor "f32[1, 58, 128, 128, 16][15204352, 262144, 2048, 16, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %add_10), kwargs = {})
#   return %mul_17
mps_lib_19 = compile_mps_shader('''
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
        int x0 = (xindex) % (16);
        auto tmp0 = in_ptr0[x2];
        auto tmp1 = in_ptr1[x0];
        auto tmp2 = tmp0 + tmp1;
        auto tmp3 = 0.5;
        auto tmp4 = tmp2 * tmp3;
        auto tmp5 = 0.7071067811865476;
        auto tmp6 = tmp2 * tmp5;
        auto tmp7 = c10::metal::erf(tmp6);
        auto tmp8 = 1.0;
        auto tmp9 = tmp7 + tmp8;
        auto tmp10 = tmp4 * tmp9;
        out_ptr0[x2] = static_cast<float>(tmp10);
    }
''')


# Topologically Sorted Source Nodes: [input_12, einsum_2], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.permute, aten.clone]
# Source node to ATen node mapping:
#   einsum_2 => clone_5, permute_28, permute_30, unsqueeze_18
#   input_12 => add_tensor_13, view_37
# Graph fragment:
#   %mm_default_15 : Tensor "f32[950272, 8][8, 1]mps:0" = PlaceHolder[target=mm_default_15]
#   %arg39_1 : Tensor "f32[8][1]mps:0" = PlaceHolder[target=arg39_1]
#   %add_tensor_13 : Tensor "f32[950272, 8][8, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_15, %arg39_1), kwargs = {})
#   %view_37 : Tensor "f32[1, 58, 128, 128, 8][7602176, 131072, 1024, 8, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_13, [1, 58, 128, 128, 8]), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 58, 128, 128, 8, 1][7602176, 131072, 1024, 8, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_37, 5), kwargs = {})
#   %permute_28 : Tensor "f32[1, 58, 128, 8, 1, 128][7602176, 131072, 1024, 1, 1, 8]mps:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_18, [0, 1, 2, 4, 5, 3]), kwargs = {})
#   %permute_30 : Tensor "f32[58, 8, 128, 128, 1, 1][131072, 1, 1024, 8, 7602176, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_28, [1, 3, 2, 5, 0, 4]), kwargs = {})
#   %clone_5 : Tensor "f32[58, 8, 128, 128, 1, 1][131072, 16384, 128, 1, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_30,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_5
mps_lib_20 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        constant float* in_ptr0,
        constant float* in_ptr1,
        uint2 thread_pos [[thread_position_in_grid]]
    ) {
        auto yindex = thread_pos.x;
        auto xindex = thread_pos.y;
        int x2 = xindex;
        int y0 = (yindex) % (8);
        int y1 = c10::metal::floor_divide(yindex, 8);
        int y3 = yindex;
        auto tmp0 = in_ptr0[y0 + 8*x2 + 131072*y1];
        auto tmp1 = in_ptr1[y0];
        auto tmp2 = tmp0 + tmp1;
        out_ptr0[x2 + 16384*y3] = static_cast<float>(tmp2);
    }
''')


# Topologically Sorted Source Nodes: [transpose_3, x_soft_1, einsum_2, x_mid, x_out], Original ATen: [aten.transpose, aten.view, aten.permute, aten.add, aten.clone]
# Source node to ATen node mapping:
#   einsum_2 => permute_32, view_41, view_42
#   transpose_3 => permute_27
#   x_mid => add_11
#   x_out => clone_7
#   x_soft_1 => view_38
# Graph fragment:
#   %getitem_4 : Tensor "f32[58, 8, 128, 8][8192, 1024, 8, 1]mps:0" = PlaceHolder[target=getitem_4]
#   %bmm_2 : Tensor "f32[464, 128, 8][1024, 8, 1]mps:0" = PlaceHolder[target=bmm_2]
#   %permute_27 : Tensor "f32[58, 128, 8, 8][8192, 8, 1024, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_4, [0, 2, 1, 3]), kwargs = {})
#   %view_38 : Tensor "f32[1, 58, 128, 8, 8][475136, 8192, 8, 1024, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_27, [1, 58, 128, 8, 8]), kwargs = {})
#   %view_41 : Tensor "f32[58, 8, 128, 1, 1, 8][8192, 1024, 8, 8, 8, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_2, [58, 8, 128, 1, 1, 8]), kwargs = {})
#   %permute_32 : Tensor "f32[1, 58, 128, 8, 8, 1][8, 8192, 8, 1024, 1, 8]mps:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_41, [4, 0, 2, 1, 5, 3]), kwargs = {})
#   %view_42 : Tensor "f32[1, 58, 128, 8, 8][8, 8192, 8, 1024, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_32, [1, 58, 128, 8, 8]), kwargs = {})
#   %add_11 : Tensor "f32[1, 58, 128, 8, 8][475136, 8192, 8, 1024, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_38, %view_42), kwargs = {})
#   %clone_7 : Tensor "f32[1, 58, 128, 8, 8][475136, 8192, 64, 8, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_11,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_7
mps_lib_21 = compile_mps_shader('''
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
        int x4 = xindex;
        int x0 = (xindex) % (8);
        int x1 = ((xindex) / (8)) % (128);
        int x2 = ((xindex) / (1024)) % (8);
        int x3 = c10::metal::floor_divide(xindex, 8192);
        auto tmp0 = in_ptr0[x4];
        auto tmp1 = in_ptr1[x4];
        auto tmp2 = tmp0 + tmp1;
        out_ptr0[x0 + 8*x2 + 64*x1 + 8192*x3] = static_cast<float>(tmp2);
    }
''')


# Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten.addmm, aten.view, aten.gelu]
# Source node to ATen node mapping:
#   input_13 => add_tensor_11, view_48
#   input_14 => add_15, erf_4, mul_20, mul_21, mul_22
# Graph fragment:
#   %mm_default_13 : Tensor "f32[7424, 256][256, 1]mps:0" = PlaceHolder[target=mm_default_13]
#   %arg45_1 : Tensor "f32[256][1]mps:0" = PlaceHolder[target=arg45_1]
#   %add_tensor_11 : Tensor "f32[7424, 256][256, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_13, %arg45_1), kwargs = {})
#   %view_48 : Tensor "f32[1, 7424, 256][1900544, 256, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_11, [1, 7424, 256]), kwargs = {})
#   %mul_20 : Tensor "f32[1, 7424, 256][1900544, 256, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_48, 0.5), kwargs = {})
#   %mul_21 : Tensor "f32[1, 7424, 256][1900544, 256, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_48, 0.7071067811865476), kwargs = {})
#   %erf_4 : Tensor "f32[1, 7424, 256][1900544, 256, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_21,), kwargs = {})
#   %add_15 : Tensor "f32[1, 7424, 256][1900544, 256, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_4, 1), kwargs = {})
#   %mul_22 : Tensor "f32[1, 7424, 256][1900544, 256, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %add_15), kwargs = {})
#   return %mul_22
mps_lib_22 = compile_mps_shader('''
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
        int x0 = (xindex) % (256);
        auto tmp0 = in_ptr0[x2];
        auto tmp1 = in_ptr1[x0];
        auto tmp2 = tmp0 + tmp1;
        auto tmp3 = 0.5;
        auto tmp4 = tmp2 * tmp3;
        auto tmp5 = 0.7071067811865476;
        auto tmp6 = tmp2 * tmp5;
        auto tmp7 = c10::metal::erf(tmp6);
        auto tmp8 = 1.0;
        auto tmp9 = tmp7 + tmp8;
        auto tmp10 = tmp4 * tmp9;
        out_ptr0[x2] = static_cast<float>(tmp10);
    }
''')


# Topologically Sorted Source Nodes: [h_3, x_out_1, x_out_2, x_mid_d, input_15, h_diag_1], Original ATen: [aten.view, aten.addmm, aten.add]
# Source node to ATen node mapping:
#   h_3 => view_5
#   h_diag_1 => add_16
#   input_15 => add_tensor_10, view_50
#   x_mid_d => add_12
#   x_out_1 => add_tensor_12, view_45
#   x_out_2 => view_46
# Graph fragment:
#   %addmm_10 : Tensor "f32[7424, 64][64, 1]mps:0" = PlaceHolder[target=addmm_10]
#   %mm_default_14 : Tensor "f32[7424, 64][64, 1]mps:0" = PlaceHolder[target=mm_default_14]
#   %arg41_1 : Tensor "f32[64][1]mps:0" = PlaceHolder[target=arg41_1]
#   %mm_default_12 : Tensor "f32[7424, 64][64, 1]mps:0" = PlaceHolder[target=mm_default_12]
#   %arg47_1 : Tensor "f32[64][1]mps:0" = PlaceHolder[target=arg47_1]
#   %view_5 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_10, [1, 7424, 64]), kwargs = {})
#   %add_tensor_12 : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_14, %arg41_1), kwargs = {})
#   %view_45 : Tensor "f32[1, 58, 128, 64][475136, 8192, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_12, [1, 58, 128, 64]), kwargs = {})
#   %view_46 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_45, [1, 7424, 64]), kwargs = {})
#   %add_12 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %view_46), kwargs = {})
#   %add_tensor_10 : Tensor "f32[7424, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_12, %arg47_1), kwargs = {})
#   %view_50 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_10, [1, 7424, 64]), kwargs = {})
#   %add_16 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0"[num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %view_50), kwargs = {})
#   return %add_16
mps_lib_23 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        constant float* in_ptr0,
        constant float* in_ptr1,
        constant float* in_ptr2,
        constant float* in_ptr3,
        constant float* in_ptr4,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x2 = xindex;
        int x0 = (xindex) % (64);
        auto tmp0 = in_ptr0[x2];
        auto tmp1 = in_ptr1[x2];
        auto tmp2 = in_ptr2[x0];
        auto tmp5 = in_ptr3[x2];
        auto tmp6 = in_ptr4[x0];
        auto tmp3 = tmp1 + tmp2;
        auto tmp4 = tmp0 + tmp3;
        auto tmp7 = tmp5 + tmp6;
        auto tmp8 = tmp4 + tmp7;
        out_ptr0[x2] = static_cast<float>(tmp8);
    }
''')


# Topologically Sorted Source Nodes: [h_k_2, h_k_3], Original ATen: [aten.view, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   h_k_2 => view_51
#   h_k_3 => constant_pad_nd_1
# Graph fragment:
#   %add_16 : Tensor "f32[1, 7424, 64][475136, 64, 1]mps:0" = PlaceHolder[target=add_16]
#   %view_51 : Tensor "f32[1, 58, 128, 64][475136, 8192, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_16, [1, 58, 128, 64]), kwargs = {})
#   %constant_pad_nd_1 : Tensor "f32[1, 64, 128, 64][524288, 8192, 64, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_51, [0, 0, 0, 0, 0, 6], 0.0), kwargs = {})
#   return %constant_pad_nd_1
mps_lib_24 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        constant float* in_ptr0,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x1 = c10::metal::floor_divide(xindex, 8192);
        int x2 = xindex;
        auto tmp4 = in_ptr0[x2];
        auto tmp0 = x1;
        auto tmp1 = static_cast<long>(tmp0);
        auto tmp2 = 58;
        auto tmp3 = tmp1 < tmp2;
        auto tmp5 = tmp3 ? tmp4 : 0.0;
        out_ptr0[x2] = static_cast<float>(tmp5);
    }
''')


# Topologically Sorted Source Nodes: [row_p_1, col_p_1, add_6, h_off_2, view_9, h_off_3, layer_norm_3], Original ATen: [aten.bmm, aten.view, aten.permute, aten.add, aten.mean, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_6 => add_17
#   col_p_1 => permute_45, unsqueeze_default, view_58, view_59
#   h_off_2 => view_60
#   h_off_3 => mean_4
#   layer_norm_3 => add_18, add_19, mul_23, mul_24, rsqrt_3, sub_3, var_mean_3
#   row_p_1 => permute_40, unsqueeze_default_1, view_54, view_55
#   view_9 => view_61
# Graph fragment:
#   %mm_default_1 : Tensor "f32[234, 8192][8192, 1]mps:0" = PlaceHolder[target=mm_default_1]
#   %mm_default : Tensor "f32[234, 8192][8192, 1]mps:0" = PlaceHolder[target=mm_default]
#   %mean_4 : Tensor "f32[234, 32, 64][2048, 64, 1]mps:0" = PlaceHolder[target=mean_4]
#   %getitem_9 : Tensor "f32[234, 32, 1][32, 1, 7488]mps:0" = PlaceHolder[target=getitem_9]
#   %buf64 : Tensor "f32[234, 32, 1][32, 1, 7488]mps:0" = PlaceHolder[target=buf64]
#   %arg48_1 : Tensor "f32[64][1]mps:0" = PlaceHolder[target=arg48_1]
#   %arg49_1 : Tensor "f32[64][1]mps:0" = PlaceHolder[target=arg49_1]
#   %unsqueeze_default_1 : Tensor "f32[1, 234, 8192][1916928, 8192, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default_1, 0), kwargs = {})
#   %view_54 : Tensor "f32[234, 1, 1, 128, 64][8192, 8192, 8192, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%unsqueeze_default_1, [234, 1, 1, 128, 64]), kwargs = {})
#   %permute_40 : Tensor "f32[1, 234, 128, 64, 1][8192, 8192, 64, 1, 8192]mps:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_54, [2, 0, 3, 4, 1]), kwargs = {})
#   %view_55 : Tensor "f32[1, 234, 128, 64][8192, 8192, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_40, [1, 234, 128, 64]), kwargs = {})
#   %unsqueeze_default : Tensor "f32[1, 234, 8192][1916928, 8192, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default, 0), kwargs = {})
#   %view_58 : Tensor "f32[234, 1, 1, 128, 64][8192, 8192, 8192, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%unsqueeze_default, [234, 1, 1, 128, 64]), kwargs = {})
#   %permute_45 : Tensor "f32[1, 234, 128, 64, 1][8192, 8192, 64, 1, 8192]mps:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_58, [2, 0, 3, 4, 1]), kwargs = {})
#   %view_59 : Tensor "f32[1, 234, 128, 64][8192, 8192, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_45, [1, 234, 128, 64]), kwargs = {})
#   %add_17 : Tensor "f32[1, 234, 128, 64][1916928, 8192, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_55, %view_59), kwargs = {})
#   %view_60 : Tensor "f32[234, 128, 64][8192, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_17, [234, 128, 64]), kwargs = {})
#   %view_61 : Tensor "f32[234, 32, 4, 64][8192, 256, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_60, [234, 32, 4, 64]), kwargs = {})
#   %mean_4 : Tensor "f32[234, 32, 64][2048, 64, 1]mps:0"[num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%view_61, [2]), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%mean_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[234, 32, 64][2048, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean_4, %getitem_9), kwargs = {})
#   %add_18 : Tensor "f32[234, 32, 1][32, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[234, 32, 1][32, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
#   %mul_23 : Tensor "f32[234, 32, 64][2048, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_24 : Tensor "f32[234, 32, 64][2048, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %arg48_1), kwargs = {})
#   %add_19 : Tensor "f32[234, 32, 64][2048, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %arg49_1), kwargs = {})
#   return %mean_4,%getitem_9,%buf64,%add_19
mps_lib_25 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    [[max_total_threads_per_threadgroup(64)]]
    kernel void generated_kernel(
        device float* out_ptr0,
        device float* out_ptr3,
        constant float* in_ptr0,
        constant float* in_ptr1,
        constant float* in_ptr2,
        constant float* in_ptr3,
        uint2 thread_pos [[thread_position_in_grid]],
        uint2 group_pos [[thread_position_in_threadgroup]]
    ) {
        auto xindex = thread_pos.x;
        auto r0_index = thread_pos.y;
        int r0_1 = r0_index;
        int x0 = xindex;
        threadgroup float tmp_acc_0[64];
        auto tmp0 = in_ptr0[r0_1 + 256*x0];
        auto tmp1 = in_ptr1[r0_1 + 256*x0];
        auto tmp3 = in_ptr0[64 + r0_1 + 256*x0];
        auto tmp4 = in_ptr1[64 + r0_1 + 256*x0];
        auto tmp7 = in_ptr0[128 + r0_1 + 256*x0];
        auto tmp8 = in_ptr1[128 + r0_1 + 256*x0];
        auto tmp11 = in_ptr0[192 + r0_1 + 256*x0];
        auto tmp12 = in_ptr1[192 + r0_1 + 256*x0];
        auto tmp2 = tmp0 + tmp1;
        auto tmp5 = tmp3 + tmp4;
        auto tmp6 = tmp2 + tmp5;
        auto tmp9 = tmp7 + tmp8;
        auto tmp10 = tmp6 + tmp9;
        auto tmp13 = tmp11 + tmp12;
        auto tmp14 = tmp10 + tmp13;
        auto tmp15 = 4.0;
        auto tmp16 = tmp14 / tmp15;
        out_ptr0[r0_1 + 64*x0] = static_cast<float>(tmp16);
        tmp_acc_0[r0_index * 1] = tmp16;
        auto tmp17 = c10::metal::threadgroup_welford_reduce(tmp_acc_0, 64);
        auto tmp25 = in_ptr2[r0_1];
        auto tmp27 = in_ptr3[r0_1];
        auto tmp18 = tmp16 - tmp17.x;
        auto tmp19 = 64.0;
        auto tmp20 = tmp17.y / tmp19;
        auto tmp21 = 1e-05;
        auto tmp22 = tmp20 + tmp21;
        auto tmp23 = metal::rsqrt(tmp22);
        auto tmp24 = tmp18 * tmp23;
        auto tmp26 = tmp24 * tmp25;
        auto tmp28 = tmp26 + tmp27;
        out_ptr3[r0_1 + 64*x0] = static_cast<float>(tmp28);
    }
''')


# Topologically Sorted Source Nodes: [reshape, sub], Original ATen: [aten.view, aten.mean]
# Source node to ATen node mapping:
#   reshape => view_6
#   sub => mean
# Graph fragment:
#   %arg28_1 : Tensor "f32[234, 128, 128, 4][65536, 512, 4, 1]mps:0" = PlaceHolder[target=arg28_1]
#   %buf68 : Tensor "f32[234, 32, 32, 4][4096, 128, 4, 1]mps:0" = PlaceHolder[target=buf68]
#   %view_6 : Tensor "f32[234, 32, 4, 32, 4, 4][65536, 2048, 512, 16, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg28_1, [234, 32, 4, 32, 4, 4]), kwargs = {})
#   %mean : Tensor "f32[234, 32, 32, 4][4096, 128, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_6, [2, 4]), kwargs = {})
#   return %buf68,%mean
mps_lib_26 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    [[max_total_threads_per_threadgroup(16)]]
    kernel void generated_kernel(
        device float* out_ptr1,
        constant float* in_ptr0,
        uint2 thread_pos [[thread_position_in_grid]],
        uint2 group_pos [[thread_position_in_threadgroup]]
    ) {
        auto xindex = thread_pos.x;
        auto r0_index = thread_pos.y;
        int r0_3 = (r0_index) % (4);
        int r0_4 = c10::metal::floor_divide(r0_index, 4);
        int x0 = (xindex) % (4);
        int x1 = ((xindex) / (4)) % (32);
        int x2 = c10::metal::floor_divide(xindex, 128);
        threadgroup float tmp_acc_0[1];
        int x5 = xindex;
        auto tmp0 = in_ptr0[x0 + 4*r0_3 + 16*x1 + 512*r0_4 + 2048*x2];
        auto tmp1 = c10::metal::threadgroup_sum(tmp_acc_0, tmp0, r0_index * 1, 16);
        auto tmp2 = 16.0;
        auto tmp3 = tmp1 / tmp2;
        out_ptr1[x5] = static_cast<float>(tmp3);
    }
''')


# Topologically Sorted Source Nodes: [reshape, sub, unsqueeze_5, oe5, off_edge_feats, arange_s_1, setitem_2], Original ATen: [aten.view, aten.mean, aten.unsqueeze, aten.expand, aten.arange, aten.scalar_tensor, aten.index_put]
# Source node to ATen node mapping:
#   arange_s_1 => iota_1
#   oe5 => expand_1
#   off_edge_feats => view_7
#   reshape => view_6
#   setitem_2 => full_default_4, index_put_4
#   sub => mean
#   unsqueeze_5 => unsqueeze_5
# Graph fragment:
#   %mean : Tensor "f32[234, 32, 32, 4][4096, 128, 4, 1]mps:0" = PlaceHolder[target=mean]
#   %view_6 : Tensor "f32[234, 32, 4, 32, 4, 4][65536, 2048, 512, 16, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg28_1, [234, 32, 4, 32, 4, 4]), kwargs = {})
#   %mean : Tensor "f32[234, 32, 32, 4][4096, 128, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_6, [2, 4]), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[1, 234, 32, 32, 4][958464, 4096, 128, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mean, 0), kwargs = {})
#   %expand_1 : Tensor "f32[1, 234, 32, 32, 4][958464, 4096, 128, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_5, [1, 234, -1, -1, -1]), kwargs = {})
#   %view_7 : Tensor "f32[234, 1, 32, 32, 4][4096, 4096, 128, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%expand_1, [234, 1, 32, 32, 4]), kwargs = {})
#   %iota_1 : Tensor "i64[32][1]mps:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: mps:0, requires_grad: False})
#   %full_default_4 : Tensor "f32[][]mps:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: mps:0, pin_memory: False})
#   %index_put_4 : Tensor "f32[234, 1, 32, 32, 4][4096, 4096, 128, 4, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%view_7, [None, None, %iota_1, %iota_1], %full_default_4), kwargs = {})
#   return %buf70
mps_lib_27 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x0 = (xindex) % (4);
        int x1 = ((xindex) / (4)) % (32);
        int x2 = c10::metal::floor_divide(xindex, 128);
        auto tmp0 = 0.0;
        out_ptr0[x0 + 132*x1 + 4096*x2] = static_cast<float>(tmp0);
    }
''')


# Topologically Sorted Source Nodes: [reshape, sub, unsqueeze_5, oe5, off_edge_feats, arange_s_1, setitem_3], Original ATen: [aten.view, aten.mean, aten.unsqueeze, aten.expand, aten.arange, aten.select, aten.scalar_tensor, aten.index_put]
# Source node to ATen node mapping:
#   arange_s_1 => iota_1
#   oe5 => expand_1
#   off_edge_feats => view_7
#   reshape => view_6
#   setitem_3 => full_default_5, index_put_5, select_10
#   sub => mean
#   unsqueeze_5 => unsqueeze_5
# Graph fragment:
#   %index_put_5 : Tensor "f32[234, 1, 32, 32][1024, 1024, 32, 1]mps:0" = PlaceHolder[target=index_put_5]
#   %view_6 : Tensor "f32[234, 32, 4, 32, 4, 4][65536, 2048, 512, 16, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg28_1, [234, 32, 4, 32, 4, 4]), kwargs = {})
#   %mean : Tensor "f32[234, 32, 32, 4][4096, 128, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_6, [2, 4]), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[1, 234, 32, 32, 4][958464, 4096, 128, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mean, 0), kwargs = {})
#   %expand_1 : Tensor "f32[1, 234, 32, 32, 4][958464, 4096, 128, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_5, [1, 234, -1, -1, -1]), kwargs = {})
#   %view_7 : Tensor "f32[234, 1, 32, 32, 4][4096, 4096, 128, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%expand_1, [234, 1, 32, 32, 4]), kwargs = {})
#   %iota_1 : Tensor "i64[32][1]mps:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: mps:0, requires_grad: False})
#   %select_10 : Tensor "f32[234, 1, 32, 32][4096, 4096, 128, 4]mps:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%index_put_4, 4, 3), kwargs = {})
#   %full_default_5 : Tensor "f32[][]mps:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: mps:0, pin_memory: False})
#   %index_put_5 : Tensor "f32[234, 1, 32, 32][1024, 1024, 32, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%select_10, [None, None, %iota_1, %iota_1], %full_default_5), kwargs = {})
#   return %buf72
mps_lib_28 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x0 = (xindex) % (32);
        int x1 = c10::metal::floor_divide(xindex, 32);
        auto tmp0 = 1.0;
        out_ptr0[33*x0 + 1024*x1] = static_cast<float>(tmp0);
    }
''')


# Topologically Sorted Source Nodes: [reshape, sub, unsqueeze_5, oe5, off_edge_feats, logit_bias_1], Original ATen: [aten.view, aten.mean, aten.unsqueeze, aten.expand, aten.select, aten.clone]
# Source node to ATen node mapping:
#   logit_bias_1 => clone_12, expand_7, select_13
#   oe5 => expand_1
#   off_edge_feats => view_7
#   reshape => view_6
#   sub => mean
#   unsqueeze_5 => unsqueeze_5
# Graph fragment:
#   %buf72 : Tensor "f32[234, 1, 32, 32][1024, 1024, 32, 1]mps:0" = PlaceHolder[target=buf72]
#   %buf70 : Tensor "f32[234, 1, 32, 32, 4][4096, 128, 4, 1]mps:0" = PlaceHolder[target=buf70]
#   %view_6 : Tensor "f32[234, 32, 4, 32, 4, 4][65536, 2048, 512, 16, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg28_1, [234, 32, 4, 32, 4, 4]), kwargs = {})
#   %mean : Tensor "f32[234, 32, 32, 4][4096, 128, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_6, [2, 4]), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[1, 234, 32, 32, 4][958464, 4096, 128, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mean, 0), kwargs = {})
#   %expand_1 : Tensor "f32[1, 234, 32, 32, 4][958464, 4096, 128, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_5, [1, 234, -1, -1, -1]), kwargs = {})
#   %view_7 : Tensor "f32[234, 1, 32, 32, 4][4096, 4096, 128, 4, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%expand_1, [234, 1, 32, 32, 4]), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[234, 1, 32, 32, 4][4096, 4096, 128, 4, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%index_put_4, %index_put_5, 4, 3), kwargs = {})
#   %select_13 : Tensor "f32[234, 1, 32, 32][4096, 4096, 128, 4]mps:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_scatter_default_1, 4, 3), kwargs = {})
#   %expand_7 : Tensor "f32[234, 8, 32, 32][4096, 0, 128, 4]mps:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%select_13, [234, 8, 32, 32]), kwargs = {})
#   %clone_12 : Tensor "f32[234, 8, 32, 32][8192, 1024, 32, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_7,), kwargs = {})
#   return %clone_12
mps_lib_29 = compile_mps_shader('''
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
        int x0 = (xindex) % (1024);
        int x2 = c10::metal::floor_divide(xindex, 8192);
        int x3 = xindex;
        auto tmp2 = in_ptr0[x0 + 1024*x2];
        auto tmp3 = in_ptr1[3 + 4*x0 + 4096*x2];
        auto tmp0 = 3;
        auto tmp1 = tmp0 == tmp0;
        auto tmp4 = tmp1 ? tmp2 : tmp3;
        out_ptr0[x3] = static_cast<float>(tmp4);
    }
''')


# Topologically Sorted Source Nodes: [linear_29, sigmoid], Original ATen: [aten.addmm, aten.view, aten.sigmoid]
# Source node to ATen node mapping:
#   linear_29 => add_tensor, view_116
#   sigmoid => sigmoid
# Graph fragment:
#   %mm_default_2 : Tensor "f32[7424, 1][1, 1]mps:0" = PlaceHolder[target=mm_default_2]
#   %arg77_1 : Tensor "f32[1][1]mps:0" = PlaceHolder[target=arg77_1]
#   %add_tensor : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_2, %arg77_1), kwargs = {})
#   %view_116 : Tensor "f32[1, 58, 128, 1][7424, 128, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [1, 58, 128, 1]), kwargs = {})
#   %sigmoid : Tensor "f32[1, 58, 128, 1][7424, 128, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_116,), kwargs = {})
#   return %sigmoid
mps_lib_30 = compile_mps_shader('''
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
        int x0 = xindex;
        auto tmp0 = in_ptr0[x0];
        auto tmp1 = in_ptr1[0];
        auto tmp2 = tmp0 + tmp1;
        auto tmp3 = 1;
        auto tmp4 = static_cast<decltype(tmp2)>(-tmp2);
        auto tmp5 = metal::exp(tmp4);
        auto tmp6 = tmp3 + tmp5;
        auto tmp7 = tmp3 / tmp6;
        out_ptr0[x0] = static_cast<float>(tmp7);
    }
''')


# Topologically Sorted Source Nodes: [qkv_x_1, getitem_17, v_spatial_1, einsum_5], Original ATen: [aten.addmm, aten.view, aten.slice, aten.unsqueeze, aten.permute, aten.clone]
# Source node to ATen node mapping:
#   einsum_5 => clone_14, permute_54, permute_56, unsqueeze_29
#   getitem_17 => slice_7
#   qkv_x_1 => add_tensor_9, view_64
#   v_spatial_1 => view_67
# Graph fragment:
#   %mm_default_11 : Tensor "f32[7488, 192][192, 1]mps:0" = PlaceHolder[target=mm_default_11]
#   %arg51_1 : Tensor "f32[192][1]mps:0" = PlaceHolder[target=arg51_1]
#   %add_tensor_9 : Tensor "f32[7488, 192][192, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_11, %arg51_1), kwargs = {})
#   %view_64 : Tensor "f32[234, 1, 32, 192][6144, 6144, 192, 1]mps:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_9, [234, 1, 32, 192]), kwargs = {})
#   %slice_7 : Tensor "f32[234, 1, 32, 64][6144, 6144, 192, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_64, 3, 128, 192), kwargs = {})
#   %view_67 : Tensor "f32[234, 1, 32, 8, 8][6144, 6144, 192, 8, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%slice_7, [234, 1, 32, 8, 8]), kwargs = {})
#   %unsqueeze_29 : Tensor "f32[234, 1, 32, 8, 8, 1][6144, 6144, 192, 8, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_67, 5), kwargs = {})
#   %permute_54 : Tensor "f32[234, 1, 1, 8, 8, 32][6144, 6144, 1, 8, 1, 192]mps:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_29, [0, 1, 5, 3, 4, 2]), kwargs = {})
#   %permute_56 : Tensor "f32[234, 8, 32, 1, 8, 1][6144, 8, 192, 6144, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_54, [0, 3, 5, 1, 4, 2]), kwargs = {})
#   %clone_14 : Tensor "f32[234, 8, 32, 1, 8, 1][2048, 256, 8, 8, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_56,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_14
mps_lib_31 = compile_mps_shader('''
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
        int x0 = (xindex) % (8);
        int x1 = ((xindex) / (8)) % (32);
        int x2 = ((xindex) / (256)) % (8);
        int x3 = c10::metal::floor_divide(xindex, 2048);
        int x4 = xindex;
        auto tmp0 = in_ptr0[128 + x0 + 8*x2 + 192*x1 + 6144*x3];
        auto tmp1 = in_ptr1[128 + x0 + 8*x2];
        auto tmp2 = tmp0 + tmp1;
        out_ptr0[x4] = static_cast<float>(tmp2);
    }
''')


# Topologically Sorted Source Nodes: [input_18, einsum_5], Original ATen: [aten.addmm, aten.view, aten.unsqueeze, aten.permute, aten.clone]
# Source node to ATen node mapping:
#   einsum_5 => clone_13, permute_53, permute_55, unsqueeze_28
#   input_18 => add_tensor_7, view_77
# Graph fragment:
#   %mm_default_9 : Tensor "f32[239616, 8][8, 1]mps:0" = PlaceHolder[target=mm_default_9]
#   %arg55_1 : Tensor "f32[8][1]mps:0" = PlaceHolder[target=arg55_1]
#   %add_tensor_7 : Tensor "f32[239616, 8][8, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_9, %arg55_1), kwargs = {})
#   %view_77 : Tensor "f32[234, 1, 32, 32, 8][8192, 8192, 256, 8, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_7, [234, 1, 32, 32, 8]), kwargs = {})
#   %unsqueeze_28 : Tensor "f32[234, 1, 32, 32, 8, 1][8192, 8192, 256, 8, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_77, 5), kwargs = {})
#   %permute_53 : Tensor "f32[234, 1, 32, 8, 1, 32][8192, 8192, 256, 1, 1, 8]mps:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_28, [0, 1, 2, 4, 5, 3]), kwargs = {})
#   %permute_55 : Tensor "f32[234, 8, 32, 32, 1, 1][8192, 1, 256, 8, 8192, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_53, [0, 3, 2, 5, 1, 4]), kwargs = {})
#   %clone_13 : Tensor "f32[234, 8, 32, 32, 1, 1][8192, 1024, 32, 1, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_55,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_13
mps_lib_32 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        constant float* in_ptr0,
        constant float* in_ptr1,
        uint2 thread_pos [[thread_position_in_grid]]
    ) {
        auto yindex = thread_pos.x;
        auto xindex = thread_pos.y;
        int x2 = xindex;
        int y0 = (yindex) % (8);
        int y1 = c10::metal::floor_divide(yindex, 8);
        int y3 = yindex;
        auto tmp0 = in_ptr0[y0 + 8*x2 + 8192*y1];
        auto tmp1 = in_ptr1[y0];
        auto tmp2 = tmp0 + tmp1;
        out_ptr0[x2 + 1024*y3] = static_cast<float>(tmp2);
    }
''')


# Topologically Sorted Source Nodes: [transpose_7, x_soft_3, einsum_5, x_mid_1, x_out_3], Original ATen: [aten.transpose, aten.view, aten.permute, aten.add, aten.clone]
# Source node to ATen node mapping:
#   einsum_5 => permute_57, view_81, view_82
#   transpose_7 => permute_52
#   x_mid_1 => add_21
#   x_out_3 => clone_15
#   x_soft_3 => view_78
# Graph fragment:
#   %getitem_10 : Tensor "f32[234, 8, 32, 8][2048, 256, 8, 1]mps:0" = PlaceHolder[target=getitem_10]
#   %bmm_5 : Tensor "f32[1872, 32, 8][256, 8, 1]mps:0" = PlaceHolder[target=bmm_5]
#   %permute_52 : Tensor "f32[234, 32, 8, 8][2048, 8, 256, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_10, [0, 2, 1, 3]), kwargs = {})
#   %view_78 : Tensor "f32[234, 1, 32, 8, 8][2048, 256, 8, 256, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_52, [234, 1, 32, 8, 8]), kwargs = {})
#   %view_81 : Tensor "f32[234, 8, 32, 1, 1, 8][2048, 256, 8, 8, 8, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_5, [234, 8, 32, 1, 1, 8]), kwargs = {})
#   %permute_57 : Tensor "f32[234, 1, 32, 8, 8, 1][2048, 8, 8, 256, 1, 8]mps:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_81, [0, 4, 2, 1, 5, 3]), kwargs = {})
#   %view_82 : Tensor "f32[234, 1, 32, 8, 8][2048, 8, 8, 256, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_57, [234, 1, 32, 8, 8]), kwargs = {})
#   %add_21 : Tensor "f32[234, 1, 32, 8, 8][2048, 256, 8, 256, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_78, %view_82), kwargs = {})
#   %clone_15 : Tensor "f32[234, 1, 32, 8, 8][2048, 2048, 64, 8, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_21,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_15
mps_lib_33 = compile_mps_shader('''
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
        int x4 = xindex;
        int x0 = (xindex) % (8);
        int x1 = ((xindex) / (8)) % (32);
        int x2 = ((xindex) / (256)) % (8);
        int x3 = c10::metal::floor_divide(xindex, 2048);
        auto tmp0 = in_ptr0[x4];
        auto tmp1 = in_ptr1[x4];
        auto tmp2 = tmp0 + tmp1;
        out_ptr0[x0 + 8*x2 + 64*x1 + 2048*x3] = static_cast<float>(tmp2);
    }
''')


# Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten.addmm, aten.view, aten.gelu]
# Source node to ATen node mapping:
#   input_25 => add_tensor_2, view_103
#   input_26 => add_28, erf_8, mul_36, mul_37, mul_38
# Graph fragment:
#   %mm_default_4 : Tensor "f32[7488, 64][64, 1]mps:0" = PlaceHolder[target=mm_default_4]
#   %arg69_1 : Tensor "f32[64][1]mps:0" = PlaceHolder[target=arg69_1]
#   %add_tensor_2 : Tensor "f32[7488, 64][64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_4, %arg69_1), kwargs = {})
#   %view_103 : Tensor "f32[234, 1, 32, 64][2048, 2048, 64, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_2, [234, 1, 32, 64]), kwargs = {})
#   %mul_36 : Tensor "f32[234, 1, 32, 64][2048, 2048, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_103, 0.5), kwargs = {})
#   %mul_37 : Tensor "f32[234, 1, 32, 64][2048, 2048, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_103, 0.7071067811865476), kwargs = {})
#   %erf_8 : Tensor "f32[234, 1, 32, 64][2048, 2048, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_37,), kwargs = {})
#   %add_28 : Tensor "f32[234, 1, 32, 64][2048, 2048, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_8, 1), kwargs = {})
#   %mul_38 : Tensor "f32[234, 1, 32, 64][2048, 2048, 64, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_36, %add_28), kwargs = {})
#   return %mul_38
mps_lib_34 = compile_mps_shader('''
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
        int x0 = (xindex) % (64);
        auto tmp0 = in_ptr0[x2];
        auto tmp1 = in_ptr1[x0];
        auto tmp2 = tmp0 + tmp1;
        auto tmp3 = 0.5;
        auto tmp4 = tmp2 * tmp3;
        auto tmp5 = 0.7071067811865476;
        auto tmp6 = tmp2 * tmp5;
        auto tmp7 = c10::metal::erf(tmp6);
        auto tmp8 = 1.0;
        auto tmp9 = tmp7 + tmp8;
        auto tmp10 = tmp4 * tmp9;
        out_ptr0[x2] = static_cast<float>(tmp10);
    }
''')


# Topologically Sorted Source Nodes: [out_2, packed_diag, out_3, out_4, packed_off, node_U, packed_U, node_V, packed_V, packed, linear_29, sigmoid, j_gate, jacobi_scale, packed_1], Original ATen: [aten.view, aten.cat, aten.addmm, aten.sigmoid, aten.squeeze]
# Source node to ATen node mapping:
#   j_gate => squeeze_2
#   jacobi_scale => view_117
#   linear_29 => add_tensor, view_116
#   node_U => view_119
#   node_V => view_121
#   out_2 => view_98
#   out_3 => view_112
#   out_4 => view_113
#   packed => cat_3
#   packed_1 => cat_4
#   packed_U => view_124
#   packed_V => view_125
#   packed_diag => view_122
#   packed_off => view_123
#   sigmoid => sigmoid
# Graph fragment:
#   %bmm_6 : Tensor "f32[58, 128, 128][16384, 128, 1]mps:0" = PlaceHolder[target=bmm_6]
#   %bmm_7 : Tensor "f32[234, 32, 32][1024, 32, 1]mps:0" = PlaceHolder[target=bmm_7]
#   %addmm_30 : Tensor "f32[7424, 32][32, 1]mps:0" = PlaceHolder[target=addmm_30]
#   %addmm_31 : Tensor "f32[7424, 32][32, 1]mps:0" = PlaceHolder[target=addmm_31]
#   %sigmoid : Tensor "f32[1, 58, 128, 1][7424, 128, 1, 1]mps:0" = PlaceHolder[target=sigmoid]
#   %view_98 : Tensor "f32[1, 58, 128, 128][950272, 16384, 128, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_6, [1, 58, 128, 128]), kwargs = {})
#   %view_122 : Tensor "f32[1, 950272][950272, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_98, [1, -1]), kwargs = {})
#   %view_112 : Tensor "f32[234, 1, 32, 32][1024, 1024, 32, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_7, [234, 1, 32, 32]), kwargs = {})
#   %view_113 : Tensor "f32[1, 234, 32, 32][239616, 1024, 32, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_112, [1, 234, 32, 32]), kwargs = {})
#   %view_123 : Tensor "f32[1, 239616][239616, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_113, [1, -1]), kwargs = {})
#   %view_119 : Tensor "f32[1, 7424, 32][237568, 32, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_30, [1, 7424, 32]), kwargs = {})
#   %view_124 : Tensor "f32[1, 237568][237568, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_119, [1, -1]), kwargs = {})
#   %view_121 : Tensor "f32[1, 7424, 32][237568, 32, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_31, [1, 7424, 32]), kwargs = {})
#   %view_125 : Tensor "f32[1, 237568][237568, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_121, [1, -1]), kwargs = {})
#   %cat_3 : Tensor "f32[1, 1665024][1665024, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_122, %view_123, %view_124, %view_125], 1), kwargs = {})
#   %add_tensor : Tensor "f32[7424, 1][1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_2, %arg77_1), kwargs = {})
#   %view_116 : Tensor "f32[1, 58, 128, 1][7424, 128, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [1, 58, 128, 1]), kwargs = {})
#   %sigmoid : Tensor "f32[1, 58, 128, 1][7424, 128, 1, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_116,), kwargs = {})
#   %squeeze_2 : Tensor "f32[1, 58, 128][7424, 128, 1]mps:0"[num_users=2] = call_function[target=torch.ops.aten.squeeze.dim](args = (%sigmoid, -1), kwargs = {})
#   %view_117 : Tensor "f32[1, 7424][7424, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%squeeze_2, [1, 7424]), kwargs = {})
#   %cat_4 : Tensor "f32[1, 1672448][1672448, 1]mps:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_3, %view_117], 1), kwargs = {})
#   return %cat_4
mps_lib_35 = compile_mps_shader('''
    #include <c10/metal/utils.h>
    #include <c10/metal/special_math.h>
    #include <c10/metal/atomic.h>
    #include <c10/metal/reduction_utils.h>
    kernel void generated_kernel(
        device float* out_ptr0,
        constant float* in_ptr0,
        constant float* in_ptr1,
        constant float* in_ptr2,
        constant float* in_ptr3,
        constant float* in_ptr4,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x0 = xindex;
        auto tmp9 = in_ptr0[x0];
        auto tmp16 = in_ptr1[(-950272) + (x0)];
        auto tmp23 = in_ptr2[(-1189888) + (x0)];
        auto tmp27 = in_ptr3[(-1427456) + (x0)];
        auto tmp36 = in_ptr4[(-1665024) + x0];
        auto tmp0 = x0;
        auto tmp1 = static_cast<long>(tmp0);
        auto tmp2 = 0;
        auto tmp3 = tmp1 >= tmp2;
        auto tmp4 = 1665024;
        auto tmp5 = tmp1 < tmp4;
        auto tmp6 = 950272;
        auto tmp7 = tmp1 < tmp6;
        auto tmp8 = tmp7 && tmp5;
        auto tmp10 = tmp8 ? tmp9 : 0.0;
        auto tmp11 = tmp1 >= tmp6;
        auto tmp12 = 1189888;
        auto tmp13 = tmp1 < tmp12;
        auto tmp14 = tmp11 & tmp13;
        auto tmp15 = tmp14 && tmp5;
        auto tmp17 = tmp15 ? tmp16 : 0.0;
        auto tmp18 = tmp1 >= tmp12;
        auto tmp19 = 1427456;
        auto tmp20 = tmp1 < tmp19;
        auto tmp21 = tmp18 & tmp20;
        auto tmp22 = tmp21 && tmp5;
        auto tmp24 = tmp22 ? tmp23 : 0.0;
        auto tmp25 = tmp1 >= tmp19;
        auto tmp26 = tmp25 && tmp5;
        auto tmp28 = tmp26 ? tmp27 : 0.0;
        auto tmp29 = tmp21 ? tmp24 : tmp28;
        auto tmp30 = tmp14 ? tmp17 : tmp29;
        auto tmp31 = tmp7 ? tmp10 : tmp30;
        auto tmp32 = tmp5 ? tmp31 : 0.0;
        auto tmp33 = tmp1 >= tmp4;
        auto tmp34 = 1672448;
        auto tmp35 = tmp1 < tmp34;
        auto tmp37 = tmp33 ? tmp36 : 0.0;
        auto tmp38 = tmp5 ? tmp32 : tmp37;
        out_ptr0[x0] = static_cast<float>(tmp38);
    }
''')


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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 7424, 9), (67716, 9, 1))
        assert_size_stride(arg1_1, (1, 12), (12, 1))
        assert_size_stride(arg2_1, (64, 18), (18, 1))
        assert_size_stride(arg3_1, (64, ), (1, ))
        assert_size_stride(arg4_1, (64, 64), (64, 1))
        assert_size_stride(arg5_1, (64, ), (1, ))
        assert_size_stride(arg6_1, (2, 49401), (49401, 1))
        assert_size_stride(arg7_1, (49401, ), (1, ))
        assert_size_stride(arg8_1, (64, 64), (64, 1))
        assert_size_stride(arg9_1, (64, ), (1, ))
        assert_size_stride(arg10_1, (64, 64), (64, 1))
        assert_size_stride(arg11_1, (64, ), (1, ))
        assert_size_stride(arg12_1, (64, 128), (128, 1))
        assert_size_stride(arg13_1, (64, ), (1, ))
        assert_size_stride(arg14_1, (64, 64), (64, 1))
        assert_size_stride(arg15_1, (64, ), (1, ))
        assert_size_stride(arg16_1, (64, 64), (64, 1))
        assert_size_stride(arg17_1, (64, ), (1, ))
        assert_size_stride(arg18_1, (64, 64), (64, 1))
        assert_size_stride(arg19_1, (64, ), (1, ))
        assert_size_stride(arg20_1, (64, 128), (128, 1))
        assert_size_stride(arg21_1, (64, ), (1, ))
        assert_size_stride(arg22_1, (64, 64), (64, 1))
        assert_size_stride(arg23_1, (64, ), (1, ))
        assert_size_stride(arg24_1, (64, ), (1, ))
        assert_size_stride(arg25_1, (64, ), (1, ))
        assert_size_stride(arg26_1, (64, 64), (64, 1))
        assert_size_stride(arg27_1, (64, ), (1, ))
        assert_size_stride(arg28_1, (234, 128, 128, 4), (65536, 512, 4, 1))
        assert_size_stride(arg29_1, (234, 64), (64, 1))
        assert_size_stride(arg30_1, (234, 64), (64, 1))
        assert_size_stride(arg31_1, (58, 128, 128, 4), (65536, 512, 4, 1))
        assert_size_stride(arg32_1, (64, ), (1, ))
        assert_size_stride(arg33_1, (64, ), (1, ))
        assert_size_stride(arg34_1, (192, 64), (64, 1))
        assert_size_stride(arg35_1, (192, ), (1, ))
        assert_size_stride(arg36_1, (16, 4), (4, 1))
        assert_size_stride(arg37_1, (16, ), (1, ))
        assert_size_stride(arg38_1, (8, 16), (16, 1))
        assert_size_stride(arg39_1, (8, ), (1, ))
        assert_size_stride(arg40_1, (64, 64), (64, 1))
        assert_size_stride(arg41_1, (64, ), (1, ))
        assert_size_stride(arg42_1, (64, ), (1, ))
        assert_size_stride(arg43_1, (64, ), (1, ))
        assert_size_stride(arg44_1, (256, 64), (64, 1))
        assert_size_stride(arg45_1, (256, ), (1, ))
        assert_size_stride(arg46_1, (64, 256), (256, 1))
        assert_size_stride(arg47_1, (64, ), (1, ))
        assert_size_stride(arg48_1, (64, ), (1, ))
        assert_size_stride(arg49_1, (64, ), (1, ))
        assert_size_stride(arg50_1, (192, 64), (64, 1))
        assert_size_stride(arg51_1, (192, ), (1, ))
        assert_size_stride(arg52_1, (16, 4), (4, 1))
        assert_size_stride(arg53_1, (16, ), (1, ))
        assert_size_stride(arg54_1, (8, 16), (16, 1))
        assert_size_stride(arg55_1, (8, ), (1, ))
        assert_size_stride(arg56_1, (64, 64), (64, 1))
        assert_size_stride(arg57_1, (64, ), (1, ))
        assert_size_stride(arg58_1, (64, ), (1, ))
        assert_size_stride(arg59_1, (64, ), (1, ))
        assert_size_stride(arg60_1, (256, 64), (64, 1))
        assert_size_stride(arg61_1, (256, ), (1, ))
        assert_size_stride(arg62_1, (64, 256), (256, 1))
        assert_size_stride(arg63_1, (64, ), (1, ))
        assert_size_stride(arg64_1, (64, 64), (64, 1))
        assert_size_stride(arg65_1, (64, ), (1, ))
        assert_size_stride(arg66_1, (128, 64), (64, 1))
        assert_size_stride(arg67_1, (128, ), (1, ))
        assert_size_stride(arg68_1, (64, 64), (64, 1))
        assert_size_stride(arg69_1, (64, ), (1, ))
        assert_size_stride(arg70_1, (32, 64), (64, 1))
        assert_size_stride(arg71_1, (32, ), (1, ))
        assert_size_stride(arg72_1, (64, 64), (64, 1))
        assert_size_stride(arg73_1, (64, ), (1, ))
        assert_size_stride(arg74_1, (32, 64), (64, 1))
        assert_size_stride(arg75_1, (32, ), (1, ))
        assert_size_stride(arg76_1, (1, 64), (64, 1))
        assert_size_stride(arg77_1, (1, ), (1, ))
        assert_size_stride(arg78_1, (32, 64), (64, 1))
        assert_size_stride(arg79_1, (32, ), (1, ))
        assert_size_stride(arg80_1, (32, 64), (64, 1))
        assert_size_stride(arg81_1, (32, ), (1, ))
        buf0 = empty_strided((1, 7424, 18), (133632, 18, 1), device='mps', dtype=torch.float32)
        mps_lib_0.generated_kernel(buf0, arg0_1, arg1_1, threads=[133632])
        del arg0_1
        del arg1_1
        buf1 = empty_strided((7424, 64), (64, 1), device='mps', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [node_feats, unsqueeze, gf, node_feats_1, input_1], Original ATen: [aten.slice, aten.unsqueeze, aten.expand, aten.cat, aten.view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf0, (7424, 18), (18, 1), 0), reinterpret_tensor(arg2_1, (18, 64), (1, 18), 0), out=buf1)
        del arg2_1
        del buf0
        buf2 = empty_strided((1, 7424, 64), (475136, 64, 1), device='mps', dtype=torch.float32)
        mps_lib_1.generated_kernel(buf2, buf1, arg3_1, threads=[475136])
        del arg3_1
        buf3 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.addmm(arg5_1, reinterpret_tensor(buf2, (7424, 64), (64, 1), 0), reinterpret_tensor(arg4_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf3)
        del arg4_1
        del arg5_1
        buf8 = empty_strided((7424, 128), (128, 1), device='mps', dtype=torch.float32)
        buf4 = reinterpret_tensor(buf8, (7424, 64), (128, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [input_3, x_flat, self_features], Original ATen: [aten.view, aten.squeeze, aten.t, aten.addmm]
        extern_kernels.addmm(arg11_1, buf3, reinterpret_tensor(arg10_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf4)
        del arg10_1
        del arg11_1
        buf6 = reinterpret_tensor(buf8, (7424, 64), (128, 1), 64)  # alias
        mps_lib_2.generated_kernel(buf6, threads=[475136])
        buf5 = reinterpret_tensor(buf2, (7424, 64), (64, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_3, x_flat, neighbor_features], Original ATen: [aten.view, aten.squeeze, aten.t, aten.addmm]
        extern_kernels.addmm(arg9_1, buf3, reinterpret_tensor(arg8_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf5)
        del arg8_1
        del arg9_1
        mps_lib_3.generated_kernel(buf6, arg6_1, buf5, arg7_1, threads=[3161664])
        del buf4
        del buf6
        buf9 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf8, reinterpret_tensor(arg12_1, (128, 64), (1, 128), 0), out=buf9)
        del arg12_1
        buf10 = empty_strided((7424, 64), (64, 1), device='mps', dtype=torch.float32)
        mps_lib_4.generated_kernel(buf10, buf9, arg13_1, threads=[475136])
        del arg13_1
        buf11 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.addmm, aten.gelu, aten.t]
        extern_kernels.mm(buf10, reinterpret_tensor(arg14_1, (64, 64), (1, 64), 0), out=buf11)
        del arg14_1
        buf12 = buf10; del buf10  # reuse
        mps_lib_5.generated_kernel(buf12, buf3, buf11, arg15_1, threads=[475136])
        del arg15_1
        buf17 = buf8; del buf8  # reuse
        buf13 = reinterpret_tensor(buf17, (7424, 64), (128, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [input_3, x_flat, input_6, out, h, x_flat_1, self_features_1], Original ATen: [aten.view, aten.squeeze, aten.addmm, aten.add, aten.unsqueeze, aten.t]
        extern_kernels.addmm(arg19_1, buf12, reinterpret_tensor(arg18_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf13)
        del arg18_1
        del arg19_1
        buf15 = reinterpret_tensor(buf17, (7424, 64), (128, 1), 64)  # alias
        mps_lib_6.generated_kernel(buf15, threads=[475136])
        buf14 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [input_3, x_flat, input_6, out, h, x_flat_1, neighbor_features_1], Original ATen: [aten.view, aten.squeeze, aten.addmm, aten.add, aten.unsqueeze, aten.t]
        extern_kernels.addmm(arg17_1, buf12, reinterpret_tensor(arg16_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf14)
        del arg16_1
        del arg17_1
        mps_lib_3.generated_kernel(buf15, arg6_1, buf14, arg7_1, threads=[3161664])
        del arg6_1
        del arg7_1
        del buf13
        del buf15
        buf18 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg20_1, (128, 64), (1, 128), 0), out=buf18)
        del arg20_1
        buf19 = buf11; del buf11  # reuse
        mps_lib_4.generated_kernel(buf19, buf18, arg21_1, threads=[475136])
        del arg21_1
        buf20 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.addmm, aten.gelu, aten.t]
        extern_kernels.mm(buf19, reinterpret_tensor(arg22_1, (64, 64), (1, 64), 0), out=buf20)
        del arg22_1
        buf24 = reinterpret_tensor(buf19, (1, 7424, 64), (475136, 64, 1), 0); del buf19  # reuse
        mps_lib_7.generated_kernel(buf24, buf12, buf20, arg23_1, arg24_1, arg25_1, threads=[7424, 64], group_size=[1, 64])
        del arg23_1
        del arg24_1
        del arg25_1
        buf25 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [input_3, x_flat, input_6, out, h, x_flat_1, input_9, out_1, h_1, h_2, h_3], Original ATen: [aten.view, aten.squeeze, aten.addmm, aten.add, aten.unsqueeze, aten.native_layer_norm, aten.t]
        extern_kernels.addmm(arg27_1, reinterpret_tensor(buf24, (7424, 64), (64, 1), 0), reinterpret_tensor(arg26_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf25)
        del arg26_1
        del arg27_1
        buf29 = buf24; del buf24  # reuse
        mps_lib_8.generated_kernel(buf29, buf25, arg32_1, arg33_1, threads=[7424, 64], group_size=[1, 64])
        del arg32_1
        del arg33_1
        buf30 = empty_strided((7424, 192), (192, 1), device='mps', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [h_3, layer_norm_1, x_blk, qkv_x], Original ATen: [aten.view, aten.native_layer_norm, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf29, (7424, 64), (64, 1), 0), reinterpret_tensor(arg34_1, (64, 192), (1, 64), 0), out=buf30)
        del arg34_1
        buf35 = reinterpret_tensor(buf29, (58, 8, 128, 8), (8192, 8, 64, 1), 0); del buf29  # reuse
        mps_lib_9.generated_kernel(buf35, buf30, arg35_1, threads=[475136])
        buf36 = reinterpret_tensor(buf12, (58, 8, 128, 8), (8192, 8, 64, 1), 0); del buf12  # reuse
        mps_lib_10.generated_kernel(buf36, buf30, arg35_1, threads=[475136])
        buf37 = empty_strided((58, 8, 128, 8), (8192, 8, 64, 1), device='mps', dtype=torch.float32)
        mps_lib_11.generated_kernel(buf37, buf30, arg35_1, threads=[475136])
        buf31 = empty_strided((1, 58, 128, 128, 4), (3801088, 65536, 512, 4, 1), device='mps', dtype=torch.float32)
        mps_lib_12.generated_kernel(buf31, arg31_1, threads=[3801088])
        del arg31_1
        mps_lib_13.generated_kernel(buf31, threads=[29696])
        buf33 = reinterpret_tensor(buf17, (1, 58, 128, 128), (950272, 16384, 128, 1), 0); del buf17  # reuse
        mps_lib_14.generated_kernel(buf33, buf31, threads=[950272])
        mps_lib_15.generated_kernel(buf33, threads=[7424])
        buf38 = empty_strided((58, 8, 128, 128), (131072, 16384, 128, 1), device='mps', dtype=torch.float32)
        mps_lib_16.generated_kernel(buf38, buf33, buf31, threads=[7602176])
        # Topologically Sorted Source Nodes: [qkv_x, getitem_10, q, reshape_4, transpose, q_f, getitem_11, k, reshape_5, transpose_1, k_f, getitem_12, v_spatial, reshape_6, transpose_2, v_f, logit_bias, x_soft], Original ATen: [aten.addmm, aten.view, aten.slice, aten.transpose, aten.clone, aten.select, aten.expand, aten._scaled_dot_product_attention_math_for_mps]
        buf39 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf35, buf36, buf37, buf38)
        del buf35
        del buf36
        del buf38
        buf40 = buf39[0]
        assert_size_stride(buf40, (58, 8, 128, 8), (8192, 1024, 8, 1), 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf40, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf39
        buf47 = reinterpret_tensor(buf37, (58, 8, 128, 1, 8, 1), (8192, 1024, 8, 8, 1, 1), 0); del buf37  # reuse
        mps_lib_17.generated_kernel(buf47, buf30, arg35_1, threads=[475136])
        del arg35_1
        del buf30
        buf42 = empty_strided((1, 58, 128, 128, 4), (3801088, 65536, 512, 4, 1), device='mps', dtype=torch.float32)
        mps_lib_18.generated_kernel(buf42, buf33, buf31, threads=[3801088])
        del buf31
        del buf33
        buf43 = empty_strided((950272, 16), (16, 1), device='mps', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf42, (950272, 4), (4, 1), 0), reinterpret_tensor(arg36_1, (4, 16), (1, 4), 0), out=buf43)
        del arg36_1
        del buf42
        buf44 = empty_strided((1, 58, 128, 128, 16), (15204352, 262144, 2048, 16, 1), device='mps', dtype=torch.float32)
        mps_lib_19.generated_kernel(buf44, buf43, arg37_1, threads=[15204352])
        del arg37_1
        del buf43
        buf45 = empty_strided((950272, 8), (8, 1), device='mps', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_10, input_11, input_12], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf44, (950272, 16), (16, 1), 0), reinterpret_tensor(arg38_1, (16, 8), (1, 16), 0), out=buf45)
        del arg38_1
        del buf44
        buf46 = empty_strided((58, 8, 128, 128, 1, 1), (131072, 16384, 128, 1, 1, 1), device='mps', dtype=torch.float32)
        mps_lib_20.generated_kernel(buf46, buf45, arg39_1, threads=[464, 16384])
        del arg39_1
        del buf45
        buf48 = empty_strided((464, 128, 8), (1024, 8, 1), device='mps', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [qkv_x, getitem_12, v_spatial, input_12, einsum_2], Original ATen: [aten.addmm, aten.view, aten.slice, aten.unsqueeze, aten.permute, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf46, (464, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf47, (464, 128, 8), (1024, 8, 1), 0), out=buf48)
        del buf46
        buf49 = reinterpret_tensor(buf47, (1, 58, 128, 8, 8), (475136, 8192, 64, 8, 1), 0); del buf47  # reuse
        mps_lib_21.generated_kernel(buf49, buf40, buf48, threads=[475136])
        buf50 = reinterpret_tensor(buf48, (7424, 64), (64, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [transpose_3, x_soft_1, einsum_2, x_mid, x_out, x_out_1], Original ATen: [aten.transpose, aten.view, aten.permute, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf49, (7424, 64), (64, 1), 0), reinterpret_tensor(arg40_1, (64, 64), (1, 64), 0), out=buf50)
        del arg40_1
        buf54 = reinterpret_tensor(buf49, (1, 7424, 64), (475136, 64, 1), 0); del buf49  # reuse
        mps_lib_7.generated_kernel(buf54, buf25, buf50, arg41_1, arg42_1, arg43_1, threads=[7424, 64], group_size=[1, 64])
        del arg42_1
        del arg43_1
        buf55 = empty_strided((7424, 256), (256, 1), device='mps', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [h_3, x_out_1, x_out_2, x_mid_d, z, input_13], Original ATen: [aten.view, aten.addmm, aten.add, aten.native_layer_norm, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf54, (7424, 64), (64, 1), 0), reinterpret_tensor(arg44_1, (64, 256), (1, 64), 0), out=buf55)
        del arg44_1
        buf56 = empty_strided((1, 7424, 256), (1900544, 256, 1), device='mps', dtype=torch.float32)
        mps_lib_22.generated_kernel(buf56, buf55, arg45_1, threads=[1900544])
        del arg45_1
        del buf55
        buf57 = reinterpret_tensor(buf54, (7424, 64), (64, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf56, (7424, 256), (256, 1), 0), reinterpret_tensor(arg46_1, (256, 64), (1, 256), 0), out=buf57)
        del arg46_1
        del buf56
        buf58 = reinterpret_tensor(buf40, (1, 7424, 64), (475136, 64, 1), 0); del buf40  # reuse
        mps_lib_23.generated_kernel(buf58, buf25, buf50, arg41_1, buf57, arg47_1, threads=[475136])
        del arg41_1
        del arg47_1
        del buf25
        del buf50
        buf59 = empty_strided((1, 64, 128, 64), (524288, 8192, 64, 1), device='mps', dtype=torch.float32)
        mps_lib_24.generated_kernel(buf59, buf58, threads=[524288])
        buf60 = empty_strided((234, 8192), (8192, 1), device='mps', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [row_p_1, h_k_2, h_k_3], Original ATen: [aten.unsqueeze, aten.view, aten.bmm, aten.constant_pad_nd, aten.permute]
        extern_kernels.mm(arg29_1, reinterpret_tensor(buf59, (64, 8192), (8192, 1), 0), out=buf60)
        del arg29_1
        buf61 = empty_strided((234, 8192), (8192, 1), device='mps', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [col_p_1], Original ATen: [aten.unsqueeze, aten.view, aten.bmm, aten.permute]
        extern_kernels.mm(arg30_1, reinterpret_tensor(buf59, (64, 8192), (8192, 1), 0), out=buf61)
        del arg30_1
        del buf59
        buf62 = empty_strided((234, 32, 64), (2048, 64, 1), device='mps', dtype=torch.float32)
        buf66 = empty_strided((234, 32, 64), (2048, 64, 1), device='mps', dtype=torch.float32)
        mps_lib_25.generated_kernel(buf62, buf66, buf60, buf61, arg48_1, arg49_1, threads=[7488, 64], group_size=[1, 64])
        del arg48_1
        del arg49_1
        buf67 = empty_strided((7488, 192), (192, 1), device='mps', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_3, x_blk_1, qkv_x_1], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf66, (7488, 64), (64, 1), 0), reinterpret_tensor(arg50_1, (64, 192), (1, 64), 0), out=buf67)
        del arg50_1
        buf73 = reinterpret_tensor(buf66, (234, 8, 32, 8), (2048, 8, 64, 1), 0); del buf66  # reuse
        mps_lib_9.generated_kernel(buf73, buf67, arg51_1, threads=[479232])
        buf74 = empty_strided((234, 8, 32, 8), (2048, 8, 64, 1), device='mps', dtype=torch.float32)
        mps_lib_10.generated_kernel(buf74, buf67, arg51_1, threads=[479232])
        buf75 = empty_strided((234, 8, 32, 8), (2048, 8, 64, 1), device='mps', dtype=torch.float32)
        mps_lib_11.generated_kernel(buf75, buf67, arg51_1, threads=[479232])
        buf69 = empty_strided((234, 32, 32, 4), (4096, 128, 4, 1), device='mps', dtype=torch.float32)
        mps_lib_26.generated_kernel(buf69, arg28_1, threads=[958464, 16], group_size=[1, 16])
        del arg28_1
        mps_lib_27.generated_kernel(buf69, threads=[29952])
        buf71 = empty_strided((234, 1, 32, 32), (1024, 1024, 32, 1), device='mps', dtype=torch.float32)
        mps_lib_14.generated_kernel(buf71, buf69, threads=[239616])
        mps_lib_28.generated_kernel(buf71, threads=[7488])
        buf76 = reinterpret_tensor(buf61, (234, 8, 32, 32), (8192, 1024, 32, 1), 0); del buf61  # reuse
        mps_lib_29.generated_kernel(buf76, buf71, buf69, threads=[1916928])
        # Topologically Sorted Source Nodes: [qkv_x_1, getitem_15, q_1, reshape_12, transpose_4, q_f_1, getitem_16, k_1, reshape_13, transpose_5, k_f_1, getitem_17, v_spatial_1, reshape_14, transpose_6, v_f_1, reshape, sub, unsqueeze_5, oe5, off_edge_feats, logit_bias_1, x_soft_2], Original ATen: [aten.addmm, aten.view, aten.slice, aten.transpose, aten.clone, aten.mean, aten.unsqueeze, aten.expand, aten.select, aten._scaled_dot_product_attention_math_for_mps]
        buf77 = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(buf73, buf74, buf75, buf76)
        del buf73
        buf78 = buf77[0]
        assert_size_stride(buf78, (234, 8, 32, 8), (2048, 256, 8, 1), 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        assert_alignment(buf78, 16, 'torch.ops.aten._scaled_dot_product_attention_math_for_mps.default')
        del buf77
        buf110 = empty_strided((7424, 1), (1, 1), device='mps', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [h_leaves_2, linear_29], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf58, (7424, 64), (64, 1), 0), reinterpret_tensor(arg76_1, (64, 1), (1, 64), 0), out=buf110)
        del arg76_1
        buf111 = empty_strided((1, 58, 128, 1), (7424, 128, 1, 1), device='mps', dtype=torch.float32)
        mps_lib_30.generated_kernel(buf111, buf110, arg77_1, threads=[7424])
        del arg77_1
        del buf110
        buf108 = empty_strided((7424, 32), (32, 1), device='mps', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [node_U], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg79_1, reinterpret_tensor(buf58, (7424, 64), (64, 1), 0), reinterpret_tensor(arg78_1, (64, 32), (1, 64), 0), alpha=1, beta=1, out=buf108)
        del arg78_1
        del arg79_1
        buf109 = empty_strided((7424, 32), (32, 1), device='mps', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [node_V], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg81_1, reinterpret_tensor(buf58, (7424, 64), (64, 1), 0), reinterpret_tensor(arg80_1, (64, 32), (1, 64), 0), alpha=1, beta=1, out=buf109)
        del arg80_1
        del arg81_1
        buf85 = reinterpret_tensor(buf75, (234, 8, 32, 1, 8, 1), (2048, 256, 8, 8, 1, 1), 0); del buf75  # reuse
        mps_lib_31.generated_kernel(buf85, buf67, arg51_1, threads=[479232])
        del arg51_1
        del buf67
        buf80 = empty_strided((234, 1, 32, 32, 4), (4096, 1, 128, 4, 1), device='mps', dtype=torch.float32)
        mps_lib_18.generated_kernel(buf80, buf71, buf69, threads=[958464])
        del buf69
        buf81 = empty_strided((239616, 16), (16, 1), device='mps', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [reshape, sub, unsqueeze_5, oe5, off_edge_feats, input_16], Original ATen: [aten.view, aten.mean, aten.unsqueeze, aten.expand, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf80, (239616, 4), (4, 1), 0), reinterpret_tensor(arg52_1, (4, 16), (1, 4), 0), out=buf81)
        del arg52_1
        del buf80
        buf82 = empty_strided((234, 1, 32, 32, 16), (16384, 1, 512, 16, 1), device='mps', dtype=torch.float32)
        mps_lib_19.generated_kernel(buf82, buf81, arg53_1, threads=[3833856])
        del arg53_1
        del buf81
        buf83 = reinterpret_tensor(buf76, (239616, 8), (8, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [input_16, input_17, input_18], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf82, (239616, 16), (16, 1), 0), reinterpret_tensor(arg54_1, (16, 8), (1, 16), 0), out=buf83)
        del arg54_1
        del buf82
        buf84 = reinterpret_tensor(buf60, (234, 8, 32, 32, 1, 1), (8192, 1024, 32, 1, 1, 1), 0); del buf60  # reuse
        mps_lib_32.generated_kernel(buf84, buf83, arg55_1, threads=[1872, 1024])
        del arg55_1
        buf86 = reinterpret_tensor(buf74, (1872, 32, 8), (256, 8, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [qkv_x_1, getitem_17, v_spatial_1, input_18, einsum_5], Original ATen: [aten.addmm, aten.view, aten.slice, aten.unsqueeze, aten.permute, aten.clone, aten._unsafe_view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf84, (1872, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf85, (1872, 32, 8), (256, 8, 1), 0), out=buf86)
        buf87 = reinterpret_tensor(buf85, (234, 1, 32, 8, 8), (2048, 1, 64, 8, 1), 0); del buf85  # reuse
        mps_lib_33.generated_kernel(buf87, buf78, buf86, threads=[479232])
        buf88 = reinterpret_tensor(buf86, (7488, 64), (64, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [transpose_7, x_soft_3, einsum_5, x_mid_1, x_out_3, x_out_4], Original ATen: [aten.transpose, aten.view, aten.permute, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf87, (7488, 64), (64, 1), 0), reinterpret_tensor(arg56_1, (64, 64), (1, 64), 0), out=buf88)
        del arg56_1
        buf96 = reinterpret_tensor(buf87, (234, 32, 64), (2048, 64, 1), 0); del buf87  # reuse
        mps_lib_7.generated_kernel(buf96, buf62, buf88, arg57_1, arg58_1, arg59_1, threads=[7488, 64], group_size=[1, 64])
        del arg58_1
        del arg59_1
        buf97 = reinterpret_tensor(buf84, (7488, 256), (256, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_out_4, x_out_5, x_mid_o, z_1, input_19], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf96, (7488, 64), (64, 1), 0), reinterpret_tensor(arg60_1, (64, 256), (1, 64), 0), out=buf97)
        del arg60_1
        buf98 = reinterpret_tensor(buf83, (234, 32, 256), (8192, 256, 1), 0); del buf83  # reuse
        mps_lib_22.generated_kernel(buf98, buf97, arg61_1, threads=[1916928])
        del arg61_1
        del buf97
        buf99 = reinterpret_tensor(buf96, (7488, 64), (64, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [input_19, input_20, input_21], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf98, (7488, 256), (256, 1), 0), reinterpret_tensor(arg62_1, (256, 64), (1, 256), 0), out=buf99)
        del arg62_1
        del buf98
        buf100 = reinterpret_tensor(buf78, (234, 32, 64), (2048, 64, 1), 0); del buf78  # reuse
        mps_lib_23.generated_kernel(buf100, buf62, buf88, arg57_1, buf99, arg63_1, threads=[479232])
        del arg57_1
        del arg63_1
        del buf62
        buf101 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [x_out_4, x_out_5, x_mid_o, input_21, off_stream, h_leaves_1, input_25], Original ATen: [aten.addmm, aten.view, aten.add, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf100, (7488, 64), (64, 1), 0), reinterpret_tensor(arg68_1, (64, 64), (1, 64), 0), out=buf101)
        del arg68_1
        buf102 = reinterpret_tensor(buf88, (234, 1, 32, 64), (2048, 1, 64, 1), 0); del buf88  # reuse
        mps_lib_34.generated_kernel(buf102, buf101, arg69_1, threads=[479232])
        del arg69_1
        del buf101
        buf103 = reinterpret_tensor(buf71, (7488, 32), (32, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [input_25, input_26, input_27], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.addmm(arg71_1, reinterpret_tensor(buf102, (7488, 64), (64, 1), 0), reinterpret_tensor(arg70_1, (64, 32), (1, 64), 0), alpha=1, beta=1, out=buf103)
        del arg70_1
        del arg71_1
        buf104 = reinterpret_tensor(buf102, (7488, 64), (64, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [x_out_4, x_out_5, x_mid_o, input_21, off_stream, h_leaves_1, input_28], Original ATen: [aten.addmm, aten.view, aten.add, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf100, (7488, 64), (64, 1), 0), reinterpret_tensor(arg72_1, (64, 64), (1, 64), 0), out=buf104)
        del arg72_1
        buf105 = reinterpret_tensor(buf100, (234, 1, 32, 64), (2048, 1, 64, 1), 0); del buf100  # reuse
        mps_lib_34.generated_kernel(buf105, buf104, arg73_1, threads=[479232])
        del arg73_1
        del buf104
        buf106 = empty_strided((7488, 32), (32, 1), device='mps', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_28, input_29, input_30], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.addmm(arg75_1, reinterpret_tensor(buf105, (7488, 64), (64, 1), 0), reinterpret_tensor(arg74_1, (64, 32), (1, 64), 0), alpha=1, beta=1, out=buf106)
        del arg74_1
        del arg75_1
        del buf105
        buf107 = empty_strided((234, 32, 32), (1024, 32, 1), device='mps', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_27, out_3, input_30, transpose_9], Original ATen: [aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf103, (234, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf106, (234, 32, 32), (1024, 1, 32), 0), out=buf107)
        del buf103
        del buf106
        buf92 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [h_leaves, input_22], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf58, (7424, 64), (64, 1), 0), reinterpret_tensor(arg64_1, (64, 64), (1, 64), 0), out=buf92)
        del arg64_1
        buf93 = reinterpret_tensor(buf58, (1, 58, 128, 64), (475136, 8192, 64, 1), 0); del buf58  # reuse
        mps_lib_34.generated_kernel(buf93, buf92, arg65_1, threads=[475136])
        del arg65_1
        del buf92
        buf94 = empty_strided((7424, 128), (128, 1), device='mps', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_22, input_23, input_24], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.addmm(arg67_1, reinterpret_tensor(buf93, (7424, 64), (64, 1), 0), reinterpret_tensor(arg66_1, (64, 128), (1, 64), 0), alpha=1, beta=1, out=buf94)
        del arg66_1
        del arg67_1
        del buf93
        buf95 = empty_strided((58, 128, 128), (16384, 128, 1), device='mps', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_24, out_2, transpose_8], Original ATen: [aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf94, (58, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf94, (58, 128, 128), (16384, 1, 128), 0), out=buf95)
        del buf94
        buf112 = empty_strided((1, 1672448), (1672448, 1), device='mps', dtype=torch.float32)
        mps_lib_35.generated_kernel(buf112, buf95, buf107, buf108, buf109, buf111, threads=[1672448])
        return (buf112, reinterpret_tensor(buf111, (1, 58, 128), (7424, 128, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 7424, 9), (67716, 9, 1), device='mps:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 12), (12, 1), device='mps:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, 18), (18, 1), device='mps:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg6_1 = rand_strided((2, 49401), (49401, 1), device='mps:0', dtype=torch.int64)
    arg7_1 = rand_strided((49401, ), (1, ), device='mps:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, 128), (128, 1), device='mps:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, 128), (128, 1), device='mps:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg27_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg28_1 = rand_strided((234, 128, 128, 4), (65536, 512, 4, 1), device='mps:0', dtype=torch.float32)
    arg29_1 = rand_strided((234, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg30_1 = rand_strided((234, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg31_1 = rand_strided((58, 128, 128, 4), (65536, 512, 4, 1), device='mps:0', dtype=torch.float32)
    arg32_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg33_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg34_1 = rand_strided((192, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg35_1 = rand_strided((192, ), (1, ), device='mps:0', dtype=torch.float32)
    arg36_1 = rand_strided((16, 4), (4, 1), device='mps:0', dtype=torch.float32)
    arg37_1 = rand_strided((16, ), (1, ), device='mps:0', dtype=torch.float32)
    arg38_1 = rand_strided((8, 16), (16, 1), device='mps:0', dtype=torch.float32)
    arg39_1 = rand_strided((8, ), (1, ), device='mps:0', dtype=torch.float32)
    arg40_1 = rand_strided((64, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg41_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, ), (1, ), device='mps:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, 256), (256, 1), device='mps:0', dtype=torch.float32)
    arg47_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg48_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg49_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg50_1 = rand_strided((192, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg51_1 = rand_strided((192, ), (1, ), device='mps:0', dtype=torch.float32)
    arg52_1 = rand_strided((16, 4), (4, 1), device='mps:0', dtype=torch.float32)
    arg53_1 = rand_strided((16, ), (1, ), device='mps:0', dtype=torch.float32)
    arg54_1 = rand_strided((8, 16), (16, 1), device='mps:0', dtype=torch.float32)
    arg55_1 = rand_strided((8, ), (1, ), device='mps:0', dtype=torch.float32)
    arg56_1 = rand_strided((64, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg57_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg58_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg59_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, ), (1, ), device='mps:0', dtype=torch.float32)
    arg62_1 = rand_strided((64, 256), (256, 1), device='mps:0', dtype=torch.float32)
    arg63_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg64_1 = rand_strided((64, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg65_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='mps:0', dtype=torch.float32)
    arg68_1 = rand_strided((64, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg69_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg70_1 = rand_strided((32, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg71_1 = rand_strided((32, ), (1, ), device='mps:0', dtype=torch.float32)
    arg72_1 = rand_strided((64, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg73_1 = rand_strided((64, ), (1, ), device='mps:0', dtype=torch.float32)
    arg74_1 = rand_strided((32, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg75_1 = rand_strided((32, ), (1, ), device='mps:0', dtype=torch.float32)
    arg76_1 = rand_strided((1, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg77_1 = rand_strided((1, ), (1, ), device='mps:0', dtype=torch.float32)
    arg78_1 = rand_strided((32, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg79_1 = rand_strided((32, ), (1, ), device='mps:0', dtype=torch.float32)
    arg80_1 = rand_strided((32, 64), (64, 1), device='mps:0', dtype=torch.float32)
    arg81_1 = rand_strided((32, ), (1, ), device='mps:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
