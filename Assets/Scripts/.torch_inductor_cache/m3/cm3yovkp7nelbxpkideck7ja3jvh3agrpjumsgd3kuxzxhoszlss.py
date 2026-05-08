
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_35', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1785856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 512
    x3 = (xindex // 512)
    x2 = (xindex // 16384)
    x4 = xindex % 16384
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (2048*x3)), None)
    tmp3 = tl.load(in_ptr2 + (4*x3), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (4*x3), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (512 + x0 + (2048*x3)), None)
    tmp17 = tl.load(in_ptr1 + (512 + x0 + (2048*x3)), None)
    tmp19 = tl.load(in_ptr2 + (1 + (4*x3)), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (1 + (4*x3)), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (1024 + x0 + (2048*x3)), None)
    tmp30 = tl.load(in_ptr1 + (1024 + x0 + (2048*x3)), None)
    tmp32 = tl.load(in_ptr2 + (2 + (4*x3)), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr3 + (2 + (4*x3)), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (1536 + x0 + (2048*x3)), None)
    tmp43 = tl.load(in_ptr1 + (1536 + x0 + (2048*x3)), None)
    tmp45 = tl.load(in_ptr2 + (3 + (4*x3)), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr3 + (3 + (4*x3)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 512.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 / tmp6
    tmp23 = tmp22 + tmp8
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp20 * tmp24
    tmp26 = tmp25 * tmp12
    tmp27 = tmp26 + tmp14
    tmp28 = tmp15 + tmp27
    tmp31 = tmp29 + tmp30
    tmp33 = tmp31 - tmp32
    tmp35 = tmp34 / tmp6
    tmp36 = tmp35 + tmp8
    tmp37 = libdevice.rsqrt(tmp36)
    tmp38 = tmp33 * tmp37
    tmp39 = tmp38 * tmp12
    tmp40 = tmp39 + tmp14
    tmp41 = tmp28 + tmp40
    tmp44 = tmp42 + tmp43
    tmp46 = tmp44 - tmp45
    tmp48 = tmp47 / tmp6
    tmp49 = tmp48 + tmp8
    tmp50 = libdevice.rsqrt(tmp49)
    tmp51 = tmp46 * tmp50
    tmp52 = tmp51 * tmp12
    tmp53 = tmp52 + tmp14
    tmp54 = tmp41 + tmp53
    tmp55 = 4.0
    tmp56 = tmp54 / tmp55
    tl.store(out_ptr0 + (x4 + (16896*x2)), tmp56, None)
