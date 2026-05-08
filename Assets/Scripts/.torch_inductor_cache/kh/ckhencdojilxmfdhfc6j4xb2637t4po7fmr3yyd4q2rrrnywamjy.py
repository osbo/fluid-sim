
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_53', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1785856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 512
    x3 = (xindex // 512)
    x2 = (xindex // 16384)
    x4 = xindex % 16384
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*x3)), None)
    tmp1 = tl.load(in_ptr1 + (4*x3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (4*x3), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (512 + x0 + (2048*x3)), None)
    tmp15 = tl.load(in_ptr1 + (1 + (4*x3)), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (1 + (4*x3)), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (1024 + x0 + (2048*x3)), None)
    tmp26 = tl.load(in_ptr1 + (2 + (4*x3)), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (2 + (4*x3)), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr0 + (1536 + x0 + (2048*x3)), None)
    tmp37 = tl.load(in_ptr1 + (3 + (4*x3)), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr2 + (3 + (4*x3)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp22 = tmp21 * tmp10
    tmp23 = tmp22 + tmp12
    tmp24 = tmp13 + tmp23
    tmp27 = tmp25 - tmp26
    tmp29 = tmp28 / tmp4
    tmp30 = tmp29 + tmp6
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp27 * tmp31
    tmp33 = tmp32 * tmp10
    tmp34 = tmp33 + tmp12
    tmp35 = tmp24 + tmp34
    tmp38 = tmp36 - tmp37
    tmp40 = tmp39 / tmp4
    tmp41 = tmp40 + tmp6
    tmp42 = libdevice.rsqrt(tmp41)
    tmp43 = tmp38 * tmp42
    tmp44 = tmp43 * tmp10
    tmp45 = tmp44 + tmp12
    tmp46 = tmp35 + tmp45
    tmp47 = 4.0
    tmp48 = tmp46 / tmp47
    tl.store(out_ptr0 + (x4 + (16896*x2)), tmp48, None)
