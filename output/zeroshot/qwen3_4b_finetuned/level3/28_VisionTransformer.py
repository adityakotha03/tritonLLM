import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 2
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 2
    y1 = yindex // 2
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y0 + 1024 * y1), xmask & ymask)
    tl.store(out_ptr0 + (y3 + 4096 * x2), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_clone_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 2
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 2
    y1 = yindex // 2
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y0 + 1024 * y1), xmask & ymask)
    tl.store(out_ptr0 + (y3 + 4096 * x2), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_clone_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_17(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_18(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_19(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_21(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_22(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_23(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_24(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_25(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_26(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_27(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_29(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_30(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y4 + 2048 * x2 + 4096 * y5), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_31(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y4 = yindex % 2
    y5 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask, eviction_policy=
