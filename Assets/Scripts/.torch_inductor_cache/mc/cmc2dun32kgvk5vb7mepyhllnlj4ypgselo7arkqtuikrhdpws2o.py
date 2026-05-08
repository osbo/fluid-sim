
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_native_layer_norm_39', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 16, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3488
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 32)
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (r2 + (256*x3)), xmask, other=0.0)
    tmp7 = tl.load(in_ptr0 + (64 + r2 + (256*x3)), xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (64 + r2 + (256*x3)), xmask, other=0.0)
    tmp13 = tl.load(in_ptr0 + (128 + r2 + (256*x3)), xmask, other=0.0)
    tmp15 = tl.load(in_ptr2 + (128 + r2 + (256*x3)), xmask, other=0.0)
    tmp19 = tl.load(in_ptr0 + (192 + r2 + (256*x3)), xmask, other=0.0)
    tmp21 = tl.load(in_ptr2 + (192 + r2 + (256*x3)), xmask, other=0.0)
    tmp27 = tl.load(in_ptr3 + (r2 + (64*x3)), xmask, other=0.0)
    tmp28 = tl.load(in_ptr4 + (r2 + (64*x3)), xmask, other=0.0)
    tmp29 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr6 + (r2 + (64*x3)), xmask, other=0.0)
    tmp33 = tl.load(in_ptr7 + (r2), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr8 + (r2), None, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr9 + (r2), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 / tmp2
    tmp5 = tmp4 / tmp2
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7 / tmp2
    tmp10 = tmp9 / tmp2
    tmp11 = tmp8 + tmp10
    tmp12 = tmp6 + tmp11
    tmp14 = tmp13 / tmp2
    tmp16 = tmp15 / tmp2
    tmp17 = tmp14 + tmp16
    tmp18 = tmp12 + tmp17
    tmp20 = tmp19 / tmp2
    tmp22 = tmp21 / tmp2
    tmp23 = tmp20 + tmp22
    tmp24 = tmp18 + tmp23
    tmp25 = 4.0
    tmp26 = tmp24 / tmp25
    tmp30 = tmp28 + tmp29
    tmp31 = tmp27 + tmp30
    tmp34 = tmp32 + tmp33
    tmp35 = tmp31 + tmp34
    tmp36 = tmp26 + tmp35
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
    tmp39 = tl.where(xmask, tmp37, 0)
    tmp40 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
    tmp42 = tl.where(xmask, tmp40, 0)
    tmp43 = tl.sum(tmp42, 1)[:, None]
    tmp44 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tmp43 / tmp45
    tmp47 = tmp37 - tmp46
    tmp48 = tmp47 * tmp47
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK, RBLOCK])
    tmp51 = tl.where(xmask, tmp49, 0)
    tmp52 = tl.sum(tmp51, 1)[:, None]
    tmp53 = tmp36 - tmp46
    tmp54 = 64.0
    tmp55 = tmp52 / tmp54
    tmp56 = 1e-05
    tmp57 = tmp55 + tmp56
    tmp58 = libdevice.rsqrt(tmp57)
    tmp59 = tmp53 * tmp58
    tmp61 = tmp59 * tmp60
    tmp63 = tmp61 + tmp62
    tl.store(in_out_ptr0 + (r2 + (64*x3)), tmp36, xmask)
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp63, xmask)
