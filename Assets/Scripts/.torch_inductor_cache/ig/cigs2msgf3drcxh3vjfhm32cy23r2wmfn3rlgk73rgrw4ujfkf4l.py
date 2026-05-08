
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_native_layer_norm_37', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 15, 'num_reduction': 4, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, xnumel, rnumel):
    xnumel = 7488
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), None)
    tmp3 = tl.load(in_ptr0 + (256 + r1 + (1024*x0)), None)
    tmp4 = tl.load(in_ptr1 + (256 + r1 + (1024*x0)), None)
    tmp7 = tl.load(in_ptr0 + (512 + r1 + (1024*x0)), None)
    tmp8 = tl.load(in_ptr1 + (512 + r1 + (1024*x0)), None)
    tmp11 = tl.load(in_ptr0 + (768 + r1 + (1024*x0)), None)
    tmp12 = tl.load(in_ptr1 + (768 + r1 + (1024*x0)), None)
    tmp17 = tl.load(in_ptr2 + (r1 + (256*x0)), None)
    tmp18 = tl.load(in_ptr3 + (r1 + (256*x0)), None)
    tmp19 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_out_ptr0 + (r1 + (256*x0)), None)
    tmp23 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr6 + (r1), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr7 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp20 = tmp18 + tmp19
    tmp21 = tmp17 + tmp20
    tmp24 = tmp22 + tmp23
    tmp25 = tmp21 + tmp24
    tmp26 = tmp16 + tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp32 = tl.full([1], 256, tl.int32)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 / tmp33
    tmp35 = tmp27 - tmp34
    tmp36 = tmp35 * tmp35
    tmp37 = tl.broadcast_to(tmp36, [RBLOCK])
    tmp39 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp40 = tmp26 - tmp34
    tmp41 = 256.0
    tmp42 = tmp39 / tmp41
    tmp43 = 1e-05
    tmp44 = tmp42 + tmp43
    tmp45 = libdevice.rsqrt(tmp44)
    tmp46 = tmp40 * tmp45
    tmp48 = tmp46 * tmp47
    tmp50 = tmp48 + tmp49
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp26, None)
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp50, None)
