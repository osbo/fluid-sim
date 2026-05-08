
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_mean_native_layer_norm_sub_53', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 15, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 3488
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = xindex % 32
    x3 = (xindex // 32)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (r1 + (512*x0)), None)
    tmp6 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (r1 + (2048*x0)), None)
    tmp11 = tl.load(in_ptr5 + (r1 + (2048*x0)), None)
    tmp14 = tl.load(in_ptr4 + (512 + r1 + (2048*x0)), None)
    tmp15 = tl.load(in_ptr5 + (512 + r1 + (2048*x0)), None)
    tmp19 = tl.load(in_ptr4 + (1024 + r1 + (2048*x0)), None)
    tmp20 = tl.load(in_ptr5 + (1024 + r1 + (2048*x0)), None)
    tmp24 = tl.load(in_ptr4 + (1536 + r1 + (2048*x0)), None)
    tmp25 = tl.load(in_ptr5 + (1536 + r1 + (2048*x0)), None)
    tmp51 = tl.load(in_ptr6 + (r1), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr7 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tmp8 - tmp0
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp9
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16 + tmp9
    tmp18 = tmp13 + tmp17
    tmp21 = tmp19 + tmp20
    tmp22 = tmp21 + tmp9
    tmp23 = tmp18 + tmp22
    tmp26 = tmp24 + tmp25
    tmp27 = tmp26 + tmp9
    tmp28 = tmp23 + tmp27
    tmp29 = 4.0
    tmp30 = tmp28 / tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.broadcast_to(tmp31, [RBLOCK])
    tmp35 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp36 = tl.full([1], 512, tl.int32)
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 / tmp37
    tmp39 = tmp31 - tmp38
    tmp40 = tmp39 * tmp39
    tmp41 = tl.broadcast_to(tmp40, [RBLOCK])
    tmp43 = triton_helpers.promote_to_tensor(tl.sum(tmp41, 0))
    tmp44 = tmp30 - tmp38
    tmp45 = 512.0
    tmp46 = tmp43 / tmp45
    tmp47 = 1e-05
    tmp48 = tmp46 + tmp47
    tmp49 = libdevice.rsqrt(tmp48)
    tmp50 = tmp44 * tmp49
    tmp52 = tmp50 * tmp51
    tmp54 = tmp52 + tmp53
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp9, None)
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp30, None)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp54, None)
    tl.store(out_ptr4 + (r1 + (512*x2) + (16896*x3)), tmp54, None)
