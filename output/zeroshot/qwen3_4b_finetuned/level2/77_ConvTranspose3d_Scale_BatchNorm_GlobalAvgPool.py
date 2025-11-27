import torch
from torch._inductor.select_algorithm import extern_kernels
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
def triton_poi_fused_mul_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_per_fused__native_batch_norm_legit_1(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64000 * x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 64000, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 64000.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tl.store(out_ptr2 + x0, tmp21, xmask)
    tl.store(out_ptr0 + x0, tmp10, xmask)
    tl.store(out_ptr1 + x0, tmp16, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_2(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tl.store(out_ptr0 + (64000 + x0), tmp7, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_3(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (128 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_4(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (256 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_5(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (384 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_6(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (512 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_7(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (640 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_8(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (768 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_9(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (896 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_10(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (1024 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_11(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (1152 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_12(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (1280 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_13(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (1408 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_14(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (1536 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_15(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (1664 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_16(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (1792 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_17(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (1920 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_18(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (2048 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_19(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (2176 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_20(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (2304 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_21(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (2432 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_22(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (2560 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_23(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (2688 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_24(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (2816 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_25(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (2944 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_26(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (3072 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_27(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (3200 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_28(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (3328 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_29(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp6 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (3456 + x0), tmp8, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_30(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x