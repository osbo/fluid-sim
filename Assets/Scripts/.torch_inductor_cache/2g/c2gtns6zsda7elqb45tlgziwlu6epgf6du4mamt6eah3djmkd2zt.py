
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[64, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_44', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 48
    xnumel = 1056
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x5 = xindex
    y4 = yindex
    x3 = (xindex // 33)
    x2 = xindex % 33
    y1 = (yindex // 8)
    y0 = yindex % 8
    tmp0 = tl.load(in_out_ptr0 + (x5 + (1056*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x3 + (32*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x3 + (32*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (y0 + (8*x5) + (8448*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tmp6 = x2
    tmp7 = tl.full([1, 1], 0, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tl.full([1, 1], 32, tl.int64)
    tmp10 = tmp6 < tmp9
    tmp11 = tl.load(in_ptr2 + ((32*x3) + (1024*y1) + x2), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp6 >= tmp9
    tmp13 = tl.full([1, 1], 33, tl.int64)
    tmp14 = tmp6 < tmp13
    tmp15 = tl.load(in_ptr3 + (x3 + (32*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.where(tmp10, tmp11, tmp15)
    tmp17 = 0.0
    tmp18 = tmp16 == tmp17
    tmp21 = tmp19 + tmp20
    tmp22 = tl.where(tmp18, tmp17, tmp21)
    tmp23 = tmp5 + tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5 + (1056*y4)), tmp23, xmask & ymask)
