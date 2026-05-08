
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.foreach(
    num_warps=8,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=142), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_for_fused_1', 'mutated_arg_names': [], 'backend_hash': 'EE1FF1A692BB318A52D1FC9AAB9D4731B4C6A59706D9DFCBED88CECB030EA339', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2):
    pid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    num_xblocks_0 = tl.cdiv(524288, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(524288, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(524288, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        xnumel = 524288
        rnumel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        x2 = xindex
        x0 = xindex % 128
        x1 = (xindex // 128)
        tmp0 = tl.load(in_ptr0 + (x2), None)
        tl.store(out_ptr0 + (x0 + (512*x1)), tmp0, None)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        xnumel = 524288
        rnumel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        x5 = xindex
        x3 = xindex % 128
        x4 = (xindex // 128)
        tmp1 = tl.load(in_ptr1 + (x5), None)
        tl.store(out_ptr1 + (x3 + (512*x4)), tmp1, None)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        xnumel = 524288
        rnumel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        x6 = xindex % 128
        x7 = (xindex // 128)
        tmp2 = tl.load(in_ptr2 + (x6), None, eviction_policy='evict_last')
        tl.store(out_ptr2 + (x6 + (512*x7)), tmp2, None)
    else:
        pass
