
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[65536, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_eq_masked_fill_mul_15', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 33024
    rnumel = 129
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 129
    x2 = (xindex // 1032)
    x4 = xindex
    _tmp27 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x5 = xindex % 1032
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp15 = tl.load(in_ptr1 + (r3 + (129*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr2 + (r3 + (129*x0) + (16672*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + (3 + (4*r3) + (516*x0) + (66592*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 128, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (r3 + (129*x0) + (16512*x2)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 129, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp6, tmp9, tmp10)
        tmp12 = tl.where(tmp4, tmp5, tmp11)
        tmp13 = 0.0
        tmp14 = tmp12 == tmp13
        tmp16 = 0.25
        tmp17 = tmp15 * tmp16
        tmp18 = tl.full([1, 1], 3, tl.int32)
        tmp19 = tmp18 == tmp18
        tmp22 = tl.where(tmp19, tmp20, tmp21)
        tmp23 = tmp17 + tmp22
        tmp24 = float("-inf")
        tmp25 = tl.where(tmp14, tmp24, tmp23)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = triton_helpers.maximum(_tmp27, tmp26)
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
    tmp27 = triton_helpers.max2(_tmp27, 1)[:, None]
    x6 = (xindex // 129)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp44 = tl.load(in_ptr1 + (r3 + (129*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp49 = tl.load(in_ptr2 + (r3 + (129*x0) + (16672*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp50 = tl.load(in_ptr3 + (3 + (4*r3) + (516*x0) + (66592*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp29 = x0
        tmp30 = tl.full([1, 1], 0, tl.int64)
        tmp31 = tmp29 >= tmp30
        tmp32 = tl.full([1, 1], 128, tl.int64)
        tmp33 = tmp29 < tmp32
        tmp34 = tl.load(in_ptr0 + (r3 + (129*x0) + (16512*x2)), rmask & tmp33 & xmask, eviction_policy='evict_last', other=0.0)
        tmp35 = tmp29 >= tmp32
        tmp36 = tl.full([1, 1], 129, tl.int64)
        tmp37 = tmp29 < tmp36
        tmp38 = 1.0
        tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
        tmp40 = tl.where(tmp35, tmp38, tmp39)
        tmp41 = tl.where(tmp33, tmp34, tmp40)
        tmp42 = 0.0
        tmp43 = tmp41 == tmp42
        tmp45 = 0.25
        tmp46 = tmp44 * tmp45
        tmp47 = tl.full([1, 1], 3, tl.int32)
        tmp48 = tmp47 == tmp47
        tmp51 = tl.where(tmp48, tmp49, tmp50)
        tmp52 = tmp46 + tmp51
        tmp53 = float("-inf")
        tmp54 = tl.where(tmp43, tmp53, tmp52)
        tmp55 = tmp54 - tmp27
        tmp56 = tl_math.exp(tmp55)
        tl.store(out_ptr1 + (r3 + (129*x0) + (16672*x6)), tmp56, rmask & xmask)
