import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_per_fused__native_batch_norm_legit_0(in_out_ptr0, in_ptr0,
    in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 32
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64 * x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 64.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tl.store(in_out_ptr0 + x0, tmp24, xmask)
    tl.store(out_ptr1 + x0, tmp12, xmask)
    tl.store(out_ptr0 + x0, tmp18, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_1(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 221184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 576 % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x3, xmask)
    tmp4 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tl.store(in_out_ptr0 + x3, tmp2, xmask)
    tl.store(out_ptr0 + x3, tmp10, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_2(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 192.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp2 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp13 = tl.sum(tmp11, 0)[:, None]
    tl.store(in_out_ptr0 + x2, tmp6, xmask)
    tl.store(out_ptr0 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_3(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 192.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp2 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp13 = tl.sum(tmp11, 0)[:, None]
    tl.store(in_out_ptr0 + x2, tmp6, xmask)
    tl.store(out_ptr0 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_4(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 221184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 576 % 96
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x3, xmask)
    tmp4 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tl.store(in_out_ptr0 + x3, tmp2, xmask)
    tl.store(out_ptr0 + x3, tmp10, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_5(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 144.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp2 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp13 = tl.sum(tmp11, 0)[:, None]
    tl.store(in_out_ptr0 + x2, tmp6, xmask)
    tl.store(out_ptr0 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_6(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 144.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp2 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp13 = tl.sum(tmp11, 0)[:, None]
    tl.store(in_out_ptr0 + x2, tmp6, xmask)
    tl.store(out_ptr0 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_7(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 221184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 576 % 144
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x3, xmask)
    tmp4 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tl.store(in_out_ptr0 + x3, tmp2, xmask)
    tl.store(out_ptr0 + x3, tmp10, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_8(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 192.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp2 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp13 = tl.sum(tmp11, 0)[:, None]
    tl.store(in_out_ptr0 + x2, tmp6, xmask)
    tl.store(out_ptr0 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_9(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 192.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp2 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp13 = tl.sum(tmp11, 0)[:, None]
    tl.store(in_out_ptr0 + x2, tmp6, xmask)
    tl.store(out_ptr0 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_10(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 221184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 576 % 192
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x3, xmask)
    tmp4 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tl.store(in_out_ptr0 + x3, tmp2, xmask)
    tl.store(out_ptr0 + x3, tmp10, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_11(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 192.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp2 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp13 = tl.sum(tmp11, 0)[:, None]
    tl.store(in_out_ptr0 + x2, tmp6, xmask)
    tl.store(out_ptr0 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_12(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 192.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp2 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp13 = tl.sum(tmp11, 0)[:, None]
    tl.store(in_out_ptr0 + x2, tmp6, xmask)
    tl.store(out_ptr0 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_13(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 221184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 576 % 288
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x3, xmask)
    tmp4 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tl.store(in_out_ptr0 + x3, tmp2, xmask)
    tl.store(out_ptr0 + x3, tmp10, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_14(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 192.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp2 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp13 = tl.sum(tmp11, 0)[:, None]
    tl.store(in_out_ptr0 + x2, tmp6, xmask)
    tl.store(out_ptr0 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_15(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 192.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp2 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp13 = tl.sum(tmp11, 0)[:, None]
    tl.store(in_out_ptr0 + x2, tmp6, xmask)
    tl.store(out_ptr0 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_16(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 221184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 576 % 384
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x3, xmask)
    tmp4 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tl.store(in_out_ptr0 + x3, tmp2, xmask)
    tl.store(out_ptr0 + x3, tmp10, xmask)


@triton.jit
def triton_per_fused__native_batch_norm_legit_17(in_out_ptr0, in_ptr0,
    in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 1408
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 128 * x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 128.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tl.store(in_out_ptr0 + x0, tmp24, xmask)
    tl.store(out_ptr1 + x0, tmp12, xmask)
    tl.store(out_ptr0 + x0, tmp18, xmask)


@triton.jit
def triton_poi_fused__to_copy_18(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_19(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused_add_mean_20(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1408
    x1 = xindex // 1408
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1408 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.sum(tmp3, 0)[:, None]
    tmp6 = 2.0
    tmp7 = tmp5 / tmp6
    tl.store(out_ptr0 + x2, tmp7, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_threshold_backward_21(in_ptr0,
    in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tl.load(in_ptr2 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = 0.0
    tmp8 = tmp6 <= tmp7
    tl.store(out_ptr0 + x0, tmp6, xmask)
    tl.store(out_ptr1 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_22(in_out_ptr0, in_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 3 % 3
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15, primals_16, primals_17,
        primals_18, primals_19, primals_20, primals_21, primals_22,
        primals_23, primals_24, primals_25, primals_26, primals_27,
        primals_28, primals_29, primals_30, primals_31, primals_32,
        primals_33, primals_34, primals_35, primals_36, primals_37,
        primals_38, primals_39, primals_40, primals_41, primals_42,
        primals_43, primals_44, primals_45, primals_46, primals_47,
        primals_48, primals_49, primals_50, primals_51, primals_52,
        primals_53, primals_54, primals_55, primals_56, primals_57,
        primals_58, primals_59, primals_60, primals_61, primals_62,
        primals_63, primals_64, primals_65, primals_66, primals_67,
        primals_68, primals_69, primals_70, primals_71, primals_72,
        primals_73, primals_74, primals_75, primals_76, primals_77,
        primals_78, primals_79, primals_80, primals_81, prim