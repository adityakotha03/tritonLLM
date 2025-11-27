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
def triton_poi_fused_clone_0(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 10
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (y0 + 2 * x2 + 2048 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 1024 * y3), tmp2, xmask & ymask)


@triton.jit
def triton_poi_fused_mul_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_2(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused_add_arange_clamp_mul_3(out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_4(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_5(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_6(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_7(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_8(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_9(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_10(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_11(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_12(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_13(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_14(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_15(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_16(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_17(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_18(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_19(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_20(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_21(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_22(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_23(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_24(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_25(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_26(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_27(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_28(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_29(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_30(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_31(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_32(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_33(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_34(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_35(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_36(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_37(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_38(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = tmp6.to(tl.int32)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_39(out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex