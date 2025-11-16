import torch
from torch._inductor.select_algorithm import extern_kernels
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
def triton_poi_fused_convolution_relu6_0(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3840
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 32.0
    tmp6 = tmp4 > tmp5
    tl.store(in_out_ptr0 + x2, tmp4, xmask)
    tl.store(in_out_ptr0 + (3840 + x2), tmp6, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_1(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 320
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + 1280 + x2, xmask)
    tmp4 = tl.load(in_ptr1 + (320 + x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2560 + x2), xmask)
    tmp8 = tl.load(in_ptr1 + (640 + x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3840 + x2), xmask)
    tmp12 = tl.load(in_ptr1 + (960 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp5 = tmp3 - tmp4
    tmp6 = tmp2 * tmp5
    tmp9 = tmp7 - tmp8
    tmp10 = tmp6 + tmp9 * tmp9
    tmp13 = tmp11 - tmp12
    tmp14 = tmp10 + tmp13 * tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp17 = 0.0
    tmp18 = tmp16 - tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + x2, tmp18, xmask)
    tl.store(out_ptr1 + x2, tmp19, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)


@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 11468800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3840
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_3(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 16.0
    tmp6 = tmp4 > tmp5
    tl.store(in_out_ptr0 + x2, tmp4, xmask)
    tl.store(in_out_ptr0 + (1280 + x2), tmp6, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_4(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 256
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4096 + x2), xmask)
    tmp4 = tl.load(in_ptr1 + (320 + x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (8192 + x2), xmask)
    tmp8 = tl.load(in_ptr1 + (640 + x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (12288 + x2), xmask)
    tmp12 = tl.load(in_ptr1 + (960 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp5 = tmp3 - tmp4
    tmp6 = tmp2 * tmp5
    tmp9 = tmp7 - tmp8
    tmp10 = tmp6 + tmp9 * tmp9
    tmp13 = tmp11 - tmp12
    tmp14 = tmp10 + tmp13 * tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp17 = 0.0
    tmp18 = tmp16 - tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + x2, tmp18, xmask)
    tl.store(out_ptr1 + x2, tmp19, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)


@triton.jit
def triton_poi_fused_convolution_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 11648640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 2560
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_6(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 640
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 24.0
    tmp6 = tmp4 > tmp5
    tl.store(in_out_ptr0 + x2, tmp4, xmask)
    tl.store(in_out_ptr0 + (640 + x2), tmp6, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_7(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4096 + x2), xmask)
    tmp4 = tl.load(in_ptr1 + (1280 + x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (8192 + x2), xmask)
    tmp8 = tl.load(in_ptr1 + (2560 + x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (12288 + x2), xmask)
    tmp12 = tl.load(in_ptr1 + (3840 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp5 = tmp3 - tmp4
    tmp6 = tmp2 * tmp5
    tmp9 = tmp7 - tmp8
    tmp10 = tmp6 + tmp9 * tmp9
    tmp13 = tmp11 - tmp12
    tmp14 = tmp10 + tmp13 * tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp17 = 0.0
    tmp18 = tmp16 - tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + x2, tmp18, xmask)
    tl.store(out_ptr1 + x2, tmp19, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)


@triton.jit
def triton_poi_fused_convolution_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 11536384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_9(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 256
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 40.0
    tmp6 = tmp4 > tmp5
    tl.store(in_out_ptr0 + x2, tmp4, xmask)
    tl.store(in_out_ptr0 + (256 + x2), tmp6, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_10(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4096 + x2), xmask)
    tmp4 = tl.load(in_ptr1 + (1280 + x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (8192 + x2), xmask)
    tmp8 = tl.load(in_ptr1 + (2560 + x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (12288 + x2), xmask)
    tmp12 = tl.load(in_ptr1 + (3840 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp5 = tmp3 - tmp4
    tmp6 = tmp2 * tmp5
    tmp9 = tmp7 - tmp8
    tmp10 = tmp6 + tmp9 * tmp9
    tmp13 = tmp11 - tmp12
    tmp14 = tmp10 + tmp13 * tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp17 = 0.0
    tmp18 = tmp16 - tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + x2, tmp18, xmask)
    tl.store(out_ptr1 + x2, tmp19, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)


@triton.jit
def triton_poi_fused_convolution_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 11552256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_12(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 256
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 80.0
    tmp6 = tmp4 > tmp5
    tl.store(in_out_ptr0 + x2, tmp4, xmask)
    tl.store(in_out_ptr0 + (256 + x2), tmp6, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_13(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4096 + x2), xmask)
    tmp4 = tl.load(in_ptr1 + (1280 + x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (8192 + x2), xmask)
    tmp8 = tl.load(in_ptr1 + (2560 + x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (12288 + x2), xmask)
    tmp12 = tl.load(in_ptr1 + (3840 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp5 = tmp3 - tmp4
    tmp6 = tmp2 * tmp5
    tmp9 = tmp7 - tmp8
    tmp10 = tmp6 + tmp9 * tmp9
    tmp13 = tmp11 - tmp12
    tmp14 = tmp10 + tmp13 * tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp17 = 0.0
    tmp18 = tmp16 - tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + x2, tmp18, xmask)
    tl.store(out_ptr1 + x2, tmp19, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)


@triton.jit
def triton_poi_fused_convolution_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 11598528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_15(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 320
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 112.0
    tmp6 = tmp4 > tmp5
    tl.store(in_out_ptr0 + x2, tmp4, xmask)
    tl.store(in_out_ptr0 + (320 + x2), tmp6, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_16(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4096 + x2), xmask)
    tmp4 = tl.load(in_ptr1 + (1280 + x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (8192 + x2), xmask)
    tmp8 = tl.load(in_ptr1 + (2560 + x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (12288 + x2), xmask)
    tmp12 = tl.load(in_ptr1 + (3840 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp5 = tmp3 - tmp4
    tmp6 = tmp2 * tmp5
    tmp9 = tmp7 - tmp8
    tmp10 = tmp6 + tmp9 * tmp9
    tmp13 = tmp11 - tmp12
    tmp14 = tmp10 + tmp13 * tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp17 = 0.0
    tmp18 = tmp16 - tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + x2, tmp18, xmask)
    tl.store(out_ptr1 + x2, tmp19, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)


@triton.jit
def triton_poi_fused_convolution_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 11636576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_18(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 320
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 192.0
    tmp6 = tmp4 > tmp5
    tl.store(in_out_ptr0 + x2, tmp4, xmask)
    tl.store(in_out_ptr0 + (320 + x2), tmp6, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_19(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 320
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4096 + x2), xmask)
    tmp4 = tl.load(in_ptr1 + (320 + x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (8192 + x2), xmask)
    tmp8 = tl.load(in_ptr1 + (640 + x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (12288 + x2), xmask)
    tmp12 = tl.load(in_ptr1 + (960 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp5 = tmp3 - tmp4
    tmp6 = tmp2 * tmp5
    tmp9 = tmp7 - tmp8
    tmp10 = tmp6 + tmp9 * tmp9
    tmp13 = tmp11 - tmp12
    tmp14 = tmp10 + tmp13 * tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp17 = 0.0
    tmp18 = tmp16 - tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + x2, tmp18, xmask)
    tl.store(out_ptr1 + x2, tmp19, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)


@triton.jit
def triton_poi_fused_convolution_20(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 11677056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 320
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_21(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 320
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 320.0
    tmp6 = tmp4 > tmp5
    tl.store(in_out_ptr0 + x2, tmp4, xmask)
    tl.store(in_out_ptr0 + (320 + x2), tmp6, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_22(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 160
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (10240 + x2), xmask)
    tmp4 = tl.load(in_ptr1 + (1280 + x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (20480 + x2), xmask)
    tmp8 = tl.load(in_ptr1 + (2560 + x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (30720 + x2), xmask)
    tmp12 = tl.load(in_ptr1 + (3840 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp5 = tmp3 - tmp4
    tmp6 = tmp2 * tmp5
    tmp9 = tmp7 - tmp8
    tmp10 = tmp6 + tmp9 * tmp9
    tmp13 = tmp11 - tmp12
    tmp14 = tmp10 + tmp13 * tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp17 = 0.0
    tmp18 = tmp16 - tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + x2, tmp18, xmask)
    tl.store(out_ptr1 + x2, tmp19, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)


@triton.jit
def triton_poi_fused_convolution_23(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 11779200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 160
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_24(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 400
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 1280.0
    tmp6 = tmp4 > tmp5
    tl.store(in_out_ptr0 + x2, tmp4, xmask)
    tl.store(in_out_ptr0 + (400 + x2), tmp6, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15, primals_16, primals_17,
        primals_18, primals_19, primals_20, primals_21, primals_22,
        primals_23, primals_24, primals_25, primals_26, primals_27,
        primals_28) = args
    args.clear()
    assert_size_stride(primals_1, (3, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (32,), (1