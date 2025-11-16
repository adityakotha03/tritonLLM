import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_clone_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + x2, tmp0, xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (512 + x2), tmp1, xmask)


@triton.jit
def triton_poi_fused_clone_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_8(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_9(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_11(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_12(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_13(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_14(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_15(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_16(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_17(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_18(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_19(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_20(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_21(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_22(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_23(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_24(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_25(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_26(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_27(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_28(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_29(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_30(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_31(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_32(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_33(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_34(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_35(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_36(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset