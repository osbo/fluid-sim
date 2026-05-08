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


cpp_fused_add_bitwise_and_le_0 = async_compile.cpp_pybinding(['const int64_t*', 'const int64_t*', 'const int64_t*', 'bool*'], '''
#include "/orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       const int64_t* in_ptr1,
                       const int64_t* in_ptr2,
                       bool* out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(992L); x0+=static_cast<int64_t>(16L))
        {
            auto tmp0 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            auto tmp1 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            auto tmp6 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            auto tmp2 = tmp0 + tmp1;
            auto tmp3 = static_cast<int64_t>(235);
            auto tmp4 = at::vec::VectorizedN<int64_t,2>(tmp3);
            auto tmp5 = at::vec::VecMask<int64_t,2>(tmp2 <= tmp4);
            auto tmp7 = tmp6 + tmp1;
            auto tmp8 = at::vec::VecMask<int64_t,2>(tmp7 <= tmp4);
            auto tmp9 = tmp5 & tmp8;
            tmp9.store(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
        }
        #pragma omp simd simdlen(8) 
        for(int64_t x0=static_cast<int64_t>(992L); x0<static_cast<int64_t>(996L); x0+=static_cast<int64_t>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
            auto tmp1 = in_ptr1[static_cast<int64_t>(x0)];
            auto tmp5 = in_ptr2[static_cast<int64_t>(x0)];
            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
            auto tmp3 = static_cast<int64_t>(235);
            auto tmp4 = tmp2 <= tmp3;
            auto tmp6 = decltype(tmp5)(tmp5 + tmp1);
            auto tmp7 = tmp6 <= tmp3;
            auto tmp8 = decltype(tmp4)(tmp4 & tmp7);
            out_ptr0[static_cast<int64_t>(x0)] = tmp8;
        }
    }
}
''')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/2q/c2qtemeubmocrnxm7ckznz5onk6o4mp4tu2udspm234c3kqi7beg.py
# Topologically Sorted Source Nodes: [off_diag_blocks_1], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   off_diag_blocks_1 => mul
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %view_6), kwargs = {})
triton_poi_fused_mul_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1019904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (262144 + x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/nh/cnhghlzrlokoaarnckn7o2pevshepdt6cvbjhtyohtqkfju3bpwk.py
# Topologically Sorted Source Nodes: [Wc], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   Wc => convert_element_type_2
# Graph fragment:
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%device_put_2, torch.float32), kwargs = {})
triton_poi_fused__to_copy_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 234060
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tl.store(in_out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/tq/ctqvhk4vdg4aoumoev265nttg7l346aftilwpqmk5f6xtme4g5nl.py
# Topologically Sorted Source Nodes: [getitem_4, index_add_, getitem_5, index_add__1], Original ATen: [aten.index, aten.index_add]
# Source node to ATen node mapping:
#   getitem_4 => index
#   getitem_5 => index_1
#   index_add_ => index_put
#   index_add__1 => index_put_1
# Graph fragment:
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_18, [None, %device_put_5]), kwargs = {})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%view_4, [None, %device_put_3], %index, True), kwargs = {})
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_19, [None, %device_put_5]), kwargs = {})
#   %index_put_1 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%view_21, [None, %device_put_4], %index_1, True), kwargs = {})
triton_poi_fused_index_index_add_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_index_add_3', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp1 = tl.full([XBLOCK], 235, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 235)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 235")
    tmp7 = tl.full([XBLOCK], 996, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert(((0 <= tmp10) & (tmp10 < 996)) | ~(xmask), "index out of bounds: 0 <= tmp10 < 996")
    tmp12 = tl.load(in_ptr2 + (x0 + (32*tmp10)), xmask)
    tmp14 = tmp13 + tmp1
    tmp15 = tmp13 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp13)
    tl.device_assert(((0 <= tmp16) & (tmp16 < 235)) | ~(xmask), "index out of bounds: 0 <= tmp16 < 235")
    tmp18 = tl.load(in_ptr4 + (x0 + (32*tmp10)), xmask)
    tl.atomic_add(out_ptr0 + (x0 + (32*tmp4)), tmp12, xmask, sem='relaxed')
    tl.atomic_add(out_ptr0 + (x0 + (32*tmp16)), tmp18, xmask, sem='relaxed')
''', device_str='cuda')


# kernel path: /orcd/home/002/osbo/ondemand/fluid-sim/Assets/Scripts/.torch_inductor_cache/zn/cznqjkvhs4qwd5ri4casj4v76edfymuhvdrfclzi7i34zdbc4v7q.py
# Topologically Sorted Source Nodes: [mul_2, y_1], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_2 => mul_2
#   y_1 => add_2
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_9, %unsqueeze), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_26, %mul_2), kwargs = {})
triton_poi_fused_add_mul_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (1282048 + x0), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask)
    tmp4 = tl.load(in_ptr3 + (x0), xmask)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1 = args
    args.clear()
    assert_size_stride(arg0_1, (7520, 1), (1, 1))
    assert_size_stride(arg1_1, (1, 1290240), (1290240, 1))
    assert_size_stride(arg2_1, (996, ), (1, ))
    assert_size_stride(arg3_1, (996, ), (1, ))
    assert_size_stride(arg4_1, (996, ), (1, ))
    assert_size_stride(arg5_1, (996, 256), (256, 1))
    assert_size_stride(arg6_1, (996, 256), (256, 1))
    assert_size_stride(arg7_1, (2562, ), (1, ))
    assert_size_stride(arg8_1, (2562, ), (1, ))
    assert_size_stride(arg9_1, (2562, ), (1, ))
    assert_size_stride(arg10_1, (1, 7520), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((235, 32, 1), (32, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_pool], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg1_1, (235, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(arg0_1, (235, 32, 1), (32, 1, 1), 0), out=buf0)
        buf1 = empty_strided_cuda((2562, ), (1, ), torch.int64)
        buf1.copy_(arg7_1)
        del arg7_1
    buf2 = empty_strided_cpu((996, ), (1, ), torch.bool)
    cpp_fused_add_bitwise_and_le_0(arg2_1, arg3_1, arg4_1, buf2)
    del arg2_1
    del arg3_1
    del arg4_1
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((996, ), (1, ), torch.bool)
        buf3.copy_(buf2)
        del buf2
        buf4 = empty_strided_cuda((1, 996, 32, 32), (1019904, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [off_diag_blocks_1], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_1.run(arg1_1, buf3, buf4, 1019904, grid=grid(1019904), stream=stream0)
        del buf3
        buf5 = empty_strided_cuda((996, 235), (235, 1), torch.float32)
        buf5.copy_(reinterpret_tensor(arg6_1, (996, 235), (256, 1), 0))
        del arg6_1
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [Wc], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_2.run(buf6, 234060, grid=grid(234060), stream=stream0)
        buf7 = empty_strided_cuda((1, 996, 32), (31872, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_col_strips], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (1, 996, 235), (0, 235, 1), 0), reinterpret_tensor(arg0_1, (1, 235, 32), (7520, 32, 1), 0), out=buf7)
        buf8 = empty_strided_cuda((996, 32, 1), (32, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_r], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf4, (996, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf7, (996, 32, 1), (32, 1, 1), 0), out=buf8)
        buf9 = empty_strided_cuda((2562, ), (1, ), torch.int64)
        buf9.copy_(arg9_1)
        del arg9_1
        buf11 = empty_strided_cuda((2562, ), (1, ), torch.int64)
        buf11.copy_(arg8_1)
        del arg8_1
        buf12 = buf6; del buf6  # reuse
        buf12.copy_(reinterpret_tensor(arg5_1, (996, 235), (256, 1), 0))
        del arg5_1
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [Wr], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_2.run(buf13, 234060, grid=grid(234060), stream=stream0)
        buf14 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [x_row_strips], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf13, (1, 996, 235), (0, 235, 1), 0), reinterpret_tensor(arg0_1, (1, 235, 32), (7520, 32, 1), 0), out=buf14)
        del buf13
        buf15 = empty_strided_cuda((996, 32, 1), (32, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_c], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf4, (996, 32, 32), (1024, 1, 32), 0), reinterpret_tensor(buf14, (996, 32, 1), (32, 1, 1), 0), out=buf15)
        del buf14
        del buf4
        # Topologically Sorted Source Nodes: [getitem_4, index_add_, getitem_5, index_add__1], Original ATen: [aten.index, aten.index_add]
        triton_poi_fused_index_index_add_3.run(buf1, buf9, buf8, buf11, buf15, buf0, 81984, grid=grid(81984), stream=stream0)
        del buf15
        del buf8
        buf17 = empty_strided_cuda((1, 7520, 1), (7520, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_2, y_1], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_4.run(buf0, arg1_1, arg10_1, arg0_1, buf17, 7520, grid=grid(7520), stream=stream0)
        del arg0_1
        del arg10_1
        del arg1_1
        del buf0
    return (reinterpret_tensor(buf17, (7520, 1), (1, 1), 0), buf1, buf11, buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((7520, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1290240), (1290240, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((996, ), (1, ), device='cpu', dtype=torch.int64)
    arg3_1 = rand_strided((996, ), (1, ), device='cpu', dtype=torch.int64)
    arg4_1 = rand_strided((996, ), (1, ), device='cpu', dtype=torch.int64)
    arg5_1 = rand_strided((996, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((996, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((2562, ), (1, ), device='cpu', dtype=torch.int64)
    arg8_1 = rand_strided((2562, ), (1, ), device='cpu', dtype=torch.int64)
    arg9_1 = rand_strided((2562, ), (1, ), device='cpu', dtype=torch.int64)
    arg10_1 = rand_strided((1, 7520), (8192, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
