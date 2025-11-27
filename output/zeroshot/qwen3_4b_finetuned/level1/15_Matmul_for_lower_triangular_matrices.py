import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_tril_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096 * x1), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused_tril_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused_tril_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused_tril_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_5(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_6(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_7(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_9(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_10(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_11(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_12(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_13(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_14(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_15(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_16(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_17(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_18(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_19(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_20(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_21(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_22(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_23(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_24(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_25(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_26(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_27(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_28(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_29(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_30(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_31(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_32(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_33(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4096 * x1 + x0), tmp2 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_tril_34(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4096, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (