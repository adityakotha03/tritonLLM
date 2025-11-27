import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_0(in_ptr0,
    in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 36 // 3 % 32
    x0 = xindex % 36 % 3
    x4 = xindex // 36 // 3
    x5 = xindex // 36 % 3
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (32 + x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (64 + x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (96 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (x5 + 96 * x4), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr1 + (32 + x5 + 96 * x4), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr1 + (64 + x5 + 96 * x4), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr1 + (96 + x5 + 96 * x4), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr1 + (x5 + 96 * x4), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr1 + (32 + x5 + 96 * x4), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr1 + (64 + x5 + 96 * x4), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr1 + (96 + x5 + 96 * x4), xmask, eviction_policy=
        'evict_last')
    tmp3 = tmp2 + tmp3
    tmp5 = tmp3 + tmp5
    tmp7 = tmp5 + tmp7
    tmp8 = 4.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp10 + tmp2
    tmp11 = tmp11 + tmp3
    tmp13 = tmp11 + tmp5
    tmp15 = tmp13 + tmp7
    tmp16 = tmp15 / tmp8
    tmp17 = tmp10 - tmp16
    tmp18 = 3.0
    tmp19 = tmp17 / tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tmp21 + tmp2
    tmp22 = tmp22 + tmp3
    tmp24 = tmp22 + tmp5
    tmp26 = tmp24 + tmp7
    tmp27 = tmp26 / tmp8
    tmp28 = tmp21 - tmp27
    tmp29 = tmp28 / tmp18
    tmp30 = tmp29 * tmp29
    tmp31 = tmp20 + tmp30
    tmp32 = tmp10 * tmp10
    tmp33 = tmp22 * tmp22
    tmp34 = tmp32 + tmp33
    tmp35 = tmp5 * tmp5
    tmp36 = tmp34 + tmp35
    tmp37 = tmp7 * tmp7
    tmp38 = tmp36 + tmp37
    tmp39 = tmp38 / tmp8
    tmp40 = tmp31 - tmp39
    tmp41 = tmp40 / tmp18
    tmp42 = tmp41 * tmp41
    tmp43 = tmp31 + tmp42
    tmp44 = libdevice.sqrt(tmp43)
    tmp45 = 1e-05
    tmp46 = tmp44 + tmp45
    tmp47 = tmp19 / tmp46
    tmp48 = tmp1 - tmp16
    tmp49 = tmp48 / tmp18
    tmp50 = tmp49 * tmp49
    tmp51 = tmp50 * tmp43
    tmp52 = tmp47 - tmp51
    tmp53 = tmp0 - tmp9
    tmp54 = tmp53 / tmp18
    tmp55 = tmp54 * tmp44
    tmp56 = tmp52 + tmp55
    tmp57 = 0.0
    tmp58 = triton_helpers.maximum(tmp56, tmp57)
    tl.store(out_ptr0 + x3, tmp9, xmask)
    tl.store(out_ptr1 + x3, tmp58, xmask)
    tl.store(out_ptr2 + x3, tmp46, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_1(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 16 % 64
    x0 = xindex % 16
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 64 * x0 + 1024 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 3.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp12 = tmp10 - tmp11
    tmp13 = tmp12 * tmp8
    tmp14 = triton_helpers.maximum(tmp9, tmp13)
    tl.store(out_ptr0 + x3, tmp9, xmask)
    tl.store(out_ptr1 + x3, tmp14, xmask)
    tl.store(out_ptr2 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_2(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4 % 128
    x0 = xindex % 4
    x2 = xindex // 512
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 128 * x0 + 512 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 3.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp12 = tmp10 - tmp11
    tmp13 = tmp12 * tmp8
    tmp14 = triton_helpers.maximum(tmp9, tmp13)
    tl.store(out_ptr0 + x3, tmp9, xmask)
    tl.store(out_ptr1 + x3, tmp14, xmask)
    tl.store(out_ptr2 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_3(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 16 % 128
    x0 = xindex % 16
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 128 * x0 + 2048 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 3.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp12 = tmp10 - tmp11
    tmp13 = tmp12 * tmp8
    tmp14 = triton_helpers.maximum(tmp9, tmp13)
    tl.store(out_ptr0 + x3, tmp9, xmask)
    tl.store(out_ptr1 + x3, tmp14, xmask)
    tl.store(out_ptr2 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_4(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4 % 256
    x0 = xindex % 4
    x2 = xindex // 128
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 256 * x0 + 1024 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 3.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp12 = tmp10 - tmp11
    tmp13 = tmp12 * tmp8
    tmp14 = triton_helpers.maximum(tmp9, tmp13)
    tl.store(out_ptr0 + x3, tmp9, xmask)
    tl.store(out_ptr1 + x3, tmp14, xmask)
    tl.store(out_ptr2 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_5(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 16 % 256
    x0 = xindex % 16
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 256 * x0 + 4096 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 3.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp12 = tmp10 - tmp11
    tmp13 = tmp12 * tmp8
    tmp14 = triton_helpers.maximum(tmp9, tmp13)
    tl.store(out_ptr0 + x3, tmp9, xmask)
    tl.store(out_ptr1 + x3, tmp14, xmask)
    tl.store(out_ptr2 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_6(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4 % 512
    x0 = xindex % 4
    x2 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 512 * x0 + 2048 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 3.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp12 = tmp10 - tmp11
    tmp13 = tmp12 * tmp8
    tmp14 = triton_helpers.maximum(tmp9, tmp13)
    tl.store(out_ptr0 + x3, tmp9, xmask)
    tl.store(out_ptr1 + x3, tmp14, xmask)
    tl.store(out_ptr2 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_7(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4 % 512
    x0 = xindex % 4
    x2 = xindex // 8
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 512 * x0 + 2048 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 3.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp12 = tmp10 - tmp11
    tmp13 = tmp12 * tmp8
    tmp14 = triton_helpers.maximum(tmp9, tmp13)
    tl.store(out_ptr0 + x3, tmp9, xmask)
    tl.store(out_ptr1 + x3, tmp14, xmask)
    tl.store(out_ptr2 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_8(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4 % 512
    x0 = xindex % 4
    x2 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 512 * x0 + 2048 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 3.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp12 = tmp10 - tmp11
    tmp13 = tmp12 * tmp8
    tmp14 = triton_helpers.maximum(tmp9, tmp13)
    tl.store(out_ptr0 + x3, tmp9, xmask)
    tl.store(out_ptr1 + x3, tmp14, xmask)
    tl.store(out_ptr2 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_9(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4 % 512
    x0 = xindex % 4
    x2 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 512 * x0 + 2048 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 3.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp12 = tmp10 - tmp11
    tmp13 = tmp12 * tmp8
    tmp14 = triton_helpers.maximum(tmp9, tmp13)
    tl.store(out_ptr0 + x3, tmp9, xmask)
    tl.store(out_ptr1 + x3, tmp14, xmask)
    tl.store(out_ptr2 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_10(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4 % 1024
    x0 = xindex % 4
    x2 = xindex // 8
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 1024 * x0 + 4096 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 3.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp12 = tmp10 - tmp11
    tmp13 = tmp12 * tmp8
    tmp14 = triton_helpers.maximum(tmp9, tmp13)
    tl.store(out_ptr0 + x3, tmp9, xmask)
    tl.store(out_ptr1 + x3, tmp14, xmask)
    tl.store(out_ptr2 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_11(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4 % 1024
    x0 = xindex % 4
    x2 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 1024 * x0 + 4096 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 3.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp12 = tmp10 - tmp11
    tmp13 = tmp12 * tmp8
    tmp14 = triton_helpers.maximum(tmp9, tmp13)
    tl.store(out_ptr0 + x3, tmp9, xmask)
    tl.store(out_ptr1 + x3, tmp14, xmask)
    tl.store(out_ptr2 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_12(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 100
    xnumel = 100
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 25
    y3 = yindex // 25
    tmp0 = tl.load(in_ptr0 + (y0 + 100 * x1), xmask & ymask, eviction_policy
        ='evict_last')
    tmp1 = tl.load(in_ptr1 + y2, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x1 + 100 * y0), tmp2, xmask & ymask)


@triton.jit
def triton_poi_fused_avg_pool2d_13(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1000
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


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
        primals_78, primals_79, primals_80, primals_81, primals_82,
        primals_83, primals_84, primals_85, primals_86, primals_87,
        primals_88, primals_89, primals_90, primals_91, primals_92,
        primals_93, primals_94, primals_95, primals_96, primals_97,
        primals_98, primals_99, primals_100, primals_101, primals_102,
        primals_103, primals_104, primals_105, primals_106, primals_107,
        primals_108, primals_109, primals_110, primals_111, primals_112,
        primals_113, primals_114, primals_115, primals_116, primals_117,
        primals_118, primals_119, primals_120, primals_121, primals_122,
        primals_123, primals_124, primals_125, primals_126, primals_127,
        primals_128, primals_129, primals_130, primals_131, primals_132,
        primals_133, primals_134, primals_135, primals_136, primals_137,
        primals_138, primals_139, primals_140, primals_141, primals_142,
        primals_143, primals_144, primals_145, primals_146, primals_147,
        primals_148, primals_149, primals_150, primals_151, primals_152,
        primals_153, primals_154, primals_155, primals_156, primals_157,
        primals_158, primals_159, primals_160, primals_161, primals_162,
        primals_163, primals_164, primals_165, prim