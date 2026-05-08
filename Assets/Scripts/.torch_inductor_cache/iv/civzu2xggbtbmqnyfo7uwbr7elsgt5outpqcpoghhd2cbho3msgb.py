
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1024, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_masked_fill_37', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 872
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y1 = (yindex // 8)
    y3 = yindex
    y0 = yindex % 8
    tmp0 = tl.load(in_ptr0 + (x2 + (16384*y1)), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2 + (16384*y1)), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (3 + (4*x2) + (65536*y1)), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0 + (8*x2) + (131072*y1)), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1, 1], 3, tl.int32)
    tmp4 = tmp3 == tmp3
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = float("-inf")
    tmp9 = tl.where(tmp2, tmp8, tmp7)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.where(tmp2, tmp1, tmp12)
    tl.store(out_ptr0 + (x2 + (16384*y3)), tmp9, ymask)
    tl.store(out_ptr1 + (x2 + (16384*y3)), tmp13, ymask)
