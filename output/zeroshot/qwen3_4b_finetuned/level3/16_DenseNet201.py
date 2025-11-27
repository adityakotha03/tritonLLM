import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_per_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 64
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64 * x1), xmask & ymask, eviction_policy
        ='evict_last')
    tl.store(out_ptr0 + (x1 + 1024 * y0), tmp0, xmask & ymask)


@triton.jit
def triton_per_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 64
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 64 * y3), xmask & ymask)
    tl.store(out_ptr0 + (y0 + 16 * x2 + 1024 * y1), tmp0, xmask & ymask)


@triton.jit
def triton_per_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 3 * x2 + 27 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 3 * x2 + 27 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 3
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 3 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 3 * x2 + 9 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 3
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 3 * y3), xmask, eviction_policy='evict_last'
        )
    tl.store(out_ptr0 + (y0 + 3 * x2 + 9 * y1), tmp0, xmask)


@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 3
    xnumel = 3
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 3 * x1), xmask & ymask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (x1 + 3 * y0), tmp0, xmask & ymask)


@triton.jit
def triton_per_fused_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_17(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_18(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_19(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_21(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_22(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_23(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_24(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_25(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_26(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_27(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_29(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_30(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_31(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 3 * x2 + 3072 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused_32(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = y