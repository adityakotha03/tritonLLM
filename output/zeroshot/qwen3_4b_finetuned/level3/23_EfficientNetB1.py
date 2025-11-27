import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_relu6_0(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_1(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_2(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_3(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_4(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_5(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_6(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_7(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_8(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_9(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_10(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_11(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_12(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_13(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_14(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_15(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_16(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_17(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_18(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_19(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_20(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_21(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_22(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_23(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_24(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_25(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 10.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + x3, tmp4, xmask)
    tl.store(out_ptr1 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_26(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1],