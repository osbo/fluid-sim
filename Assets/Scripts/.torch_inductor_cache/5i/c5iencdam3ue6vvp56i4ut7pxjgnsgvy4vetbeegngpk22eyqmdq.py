
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mean_sub_55', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1785856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0 + (2048*x1)), None)
    tmp11 = tl.load(in_ptr5 + (x0 + (2048*x1)), None)
    tmp13 = tl.load(in_out_ptr1 + (x2), None)
    tmp16 = tl.load(in_ptr4 + (512 + x0 + (2048*x1)), None)
    tmp17 = tl.load(in_ptr5 + (512 + x0 + (2048*x1)), None)
    tmp22 = tl.load(in_ptr4 + (1024 + x0 + (2048*x1)), None)
    tmp23 = tl.load(in_ptr5 + (1024 + x0 + (2048*x1)), None)
    tmp28 = tl.load(in_ptr4 + (1536 + x0 + (2048*x1)), None)
    tmp29 = tl.load(in_ptr5 + (1536 + x0 + (2048*x1)), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tmp8 - tmp0
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14 + tmp9
    tmp18 = tmp16 + tmp17
    tmp19 = tmp18 + tmp13
    tmp20 = tmp19 + tmp9
    tmp21 = tmp15 + tmp20
    tmp24 = tmp22 + tmp23
    tmp25 = tmp24 + tmp13
    tmp26 = tmp25 + tmp9
    tmp27 = tmp21 + tmp26
    tmp30 = tmp28 + tmp29
    tmp31 = tmp30 + tmp13
    tmp32 = tmp31 + tmp9
    tmp33 = tmp27 + tmp32
    tmp34 = 4.0
    tmp35 = tmp33 / tmp34
    tl.store(in_out_ptr1 + (x2), tmp35, None)
