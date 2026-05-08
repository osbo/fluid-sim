
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_56', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1785856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0 + (2048*x1)), None)
    tmp6 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (512 + x0 + (2048*x1)), None)
    tmp11 = tl.load(in_ptr3 + (512 + x0 + (2048*x1)), None)
    tmp15 = tl.load(in_ptr0 + (1024 + x0 + (2048*x1)), None)
    tmp17 = tl.load(in_ptr3 + (1024 + x0 + (2048*x1)), None)
    tmp21 = tl.load(in_ptr0 + (1536 + x0 + (2048*x1)), None)
    tmp23 = tl.load(in_ptr3 + (1536 + x0 + (2048*x1)), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp10 = tmp9 + tmp3
    tmp12 = tmp11 + tmp6
    tmp13 = tmp10 + tmp12
    tmp14 = tmp8 + tmp13
    tmp16 = tmp15 + tmp3
    tmp18 = tmp17 + tmp6
    tmp19 = tmp16 + tmp18
    tmp20 = tmp14 + tmp19
    tmp22 = tmp21 + tmp3
    tmp24 = tmp23 + tmp6
    tmp25 = tmp22 + tmp24
    tmp26 = tmp20 + tmp25
    tmp27 = 4.0
    tmp28 = tmp26 / tmp27
    tl.store(out_ptr0 + (x2), tmp28, None)
