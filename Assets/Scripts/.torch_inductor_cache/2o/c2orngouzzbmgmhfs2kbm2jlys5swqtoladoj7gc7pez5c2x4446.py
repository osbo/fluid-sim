
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_mean_24', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'B098E03CDA7B8ADC90DAFFDF24A2956451D1B13F297756A5DCC209498AA53705', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 16384)
    x0 = xindex % 512
    x1 = (xindex // 512) % 32
    x4 = (xindex // 512)
    x5 = xindex
    x3 = xindex % 16384
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (6 + x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr2 + (2*x4), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (2*x4), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (x2 + ((1 + (2*x1)) // 64)), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (6 + x2 + ((1 + (2*x1)) // 64)), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr2 + (1 + (2*x4)), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr3 + (1 + (2*x4)), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 4), "index out of bounds: 0 <= tmp4 < 4")
    tmp6 = tl.load(in_ptr1 + (x0 + (1024*x1) + (32768*tmp4)), None)
    tmp8 = tmp7 + tmp1
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 4), "index out of bounds: 0 <= tmp10 < 4")
    tmp12 = tl.load(in_ptr1 + (x0 + (1024*x1) + (32768*tmp10)), None)
    tmp13 = tmp6 + tmp12
    tmp15 = tmp13 - tmp14
    tmp17 = 512.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp15 * tmp21
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tmp28 = tmp27 + tmp1
    tmp29 = tmp27 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp27)
    tl.device_assert((0 <= tmp30) & (tmp30 < 4), "index out of bounds: 0 <= tmp30 < 4")
    tmp32 = tl.load(in_ptr1 + (512 + x0 + (1024*x1) + (32768*tmp30)), None)
    tmp34 = tmp33 + tmp1
    tmp35 = tmp33 < 0
    tmp36 = tl.where(tmp35, tmp34, tmp33)
    tl.device_assert((0 <= tmp36) & (tmp36 < 4), "index out of bounds: 0 <= tmp36 < 4")
    tmp38 = tl.load(in_ptr1 + (512 + x0 + (1024*x1) + (32768*tmp36)), None)
    tmp39 = tmp32 + tmp38
    tmp41 = tmp39 - tmp40
    tmp43 = tmp42 / tmp17
    tmp44 = tmp43 + tmp19
    tmp45 = libdevice.rsqrt(tmp44)
    tmp46 = tmp41 * tmp45
    tmp47 = tmp46 * tmp23
    tmp48 = tmp47 + tmp25
    tmp49 = tmp26 + tmp48
    tmp50 = 2.0
    tmp51 = tmp49 / tmp50
    tl.store(out_ptr0 + (x5), tmp51, None)
    tl.store(out_ptr1 + (x3 + (16896*x2)), tmp51, None)
