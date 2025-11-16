import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 4966112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_17(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_18(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_19(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_21(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_22(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_23(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_24(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_25(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_26(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_27(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_29(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_30(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x2 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_31(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr
    , XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
   