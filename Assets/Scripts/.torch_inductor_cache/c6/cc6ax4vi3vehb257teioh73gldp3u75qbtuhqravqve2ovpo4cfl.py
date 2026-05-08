
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_layer_norm_25', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 11, 'num_reduction': 5, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6976
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
    x1 = (xindex // 64)
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (r2 + (128*x3)), xmask, other=0.0)
    tmp7 = tl.load(in_ptr0 + (64 + r2 + (128*x3)), xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (64 + r2 + (128*x3)), xmask, other=0.0)
    tmp38 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr5 + (x0 + (128*r2) + (8192*x1)), xmask, other=0.0)
    tmp44 = tl.load(in_ptr6 + (x0 + (128*r2) + (8192*x1)), xmask, other=0.0)
    tmp47 = tl.load(in_ptr5 + (64 + x0 + (128*r2) + (8192*x1)), xmask, other=0.0)
    tmp49 = tl.load(in_ptr6 + (64 + x0 + (128*r2) + (8192*x1)), xmask, other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 / tmp2
    tmp5 = tmp4 / tmp2
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7 / tmp2
    tmp10 = tmp9 / tmp2
    tmp11 = tmp8 + tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = 2.0
    tmp14 = tmp12 / tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp20 = tl.where(xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 / tmp23
    tmp25 = tmp15 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
    tmp29 = tl.where(xmask, tmp27, 0)
    tmp30 = tl.sum(tmp29, 1)[:, None]
    tmp31 = tmp14 - tmp24
    tmp32 = 64.0
    tmp33 = tmp30 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp43 = tmp42 / tmp2
    tmp45 = tmp44 / tmp2
    tmp46 = tmp43 + tmp45
    tmp48 = tmp47 / tmp2
    tmp50 = tmp49 / tmp2
    tmp51 = tmp48 + tmp50
    tmp52 = tmp46 + tmp51
    tmp53 = tmp52 / tmp13
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK, RBLOCK])
    tmp56 = tl.where(xmask, tmp54, 0)
    tmp57 = tl.sum(tmp56, 1)[:, None]
    tmp58 = tmp57 / tmp32
    tl.store(out_ptr0 + (r2 + (64*x3)), tmp14, xmask)
    tl.store(out_ptr3 + (r2 + (64*x3)), tmp41, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp58, xmask)
