
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_triu_indices_20', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B098E03CDA7B8ADC90DAFFDF24A2956451D1B13F297756A5DCC209498AA53705', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 6, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp0.to(tl.float64)
    tmp6 = tl.full([1], 2.0, tl.float64)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full([1], 12.25, tl.float64)
    tmp9 = tmp8 - tmp7
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 3.5, tl.float64)
    tmp12 = tmp11 - tmp10
    tmp13 = libdevice.floor(tmp12)
    tmp14 = tmp13.to(tl.int64)
    tmp15 = tmp14 + tmp1
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp4, tmp15, tmp16)
    tmp18 = tmp0 >= tmp3
    tmp19 = tl.full([1], 12, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = (-6) + x0
    tmp22 = tmp21.to(tl.float64)
    tmp23 = tmp22 * tmp6
    tmp24 = tmp8 - tmp23
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp11 - tmp25
    tmp27 = libdevice.floor(tmp26)
    tmp28 = tl.full([1], 5.0, tl.float64)
    tmp29 = tmp28 - tmp27
    tmp30 = tmp29 * tmp27
    tmp31 = tl.full([1], 0.5, tl.float64)
    tmp32 = tmp30 * tmp31
    tmp33 = tmp22 - tmp32
    tmp34 = libdevice.floor(tmp33)
    tmp35 = tmp34.to(tl.int64)
    tmp36 = tl.full([1], 1, tl.int64)
    tmp37 = tmp35 + tmp36
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp18, tmp37, tmp38)
    tmp40 = tl.where(tmp4, tmp17, tmp39)
    tl.store(out_ptr0 + (x0), tmp40, xmask)
