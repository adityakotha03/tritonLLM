import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_0(in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 400 % 32
    x0 = xindex % 400
    x2 = xindex // 400
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tmp2 / tmp8
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 1.0
    tmp13 = tmp9 * tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tl.store(out_ptr0 + (x0 + 4096 * x2), tmp9, xmask)
    tl.store(out_ptr1 + (x0 + 4096 * x2), tmp15, xmask)
    tl.store(out_ptr2 + x1, tmp3, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_1(in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1600 % 24
    x0 = xindex % 1600
    x2 = xindex // 1600
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tmp2 / tmp8
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 1.0
    tmp13 = tmp9 * tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tl.store(out_ptr0 + (x0 + 16640 * x2), tmp9, xmask)
    tl.store(out_ptr1 + (x0 + 16640 * x2), tmp15, xmask)
    tl.store(out_ptr2 + x1, tmp3, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_2(in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1600 % 24
    x0 = xindex % 1600
    x2 = xindex // 1600
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tmp2 / tmp8
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 1.0
    tmp13 = tmp9 * tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tl.store(out_ptr0 + (x0 + 16640 * x2), tmp9, xmask)
    tl.store(out_ptr1 + (x0 + 16640 * x2), tmp15, xmask)
    tl.store(out_ptr2 + x1, tmp3, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_3(in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1600 % 40
    x0 = xindex % 1600
    x2 = xindex // 1600
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tmp2 / tmp8
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 1.0
    tmp13 = tmp9 * tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tl.store(out_ptr0 + (x0 + 16640 * x2), tmp9, xmask)
    tl.store(out_ptr1 + (x0 + 16640 * x2), tmp15, xmask)
    tl.store(out_ptr2 + x1, tmp3, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_4(in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1600 % 40
    x0 = xindex % 1600
    x2 = xindex // 1600
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tmp2 / tmp8
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 1.0
    tmp13 = tmp9 * tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tl.store(out_ptr0 + (x0 + 16640 * x2), tmp9, xmask)
    tl.store(out_ptr1 + (x0 + 16640 * x2), tmp15, xmask)
    tl.store(out_ptr2 + x1, tmp3, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_5(in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 7680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1600 % 80
    x0 = xindex % 1600
    x2 = xindex // 1600
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tmp2 / tmp8
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 1.0
    tmp13 = tmp9 * tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tl.store(out_ptr0 + (x0 + 16640 * x2), tmp9, xmask)
    tl.store(out_ptr1 + (x0 + 16640 * x2), tmp15, xmask)
    tl.store(out_ptr2 + x1, tmp3, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_6(in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 7680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1600 % 80
    x0 = xindex % 1600
    x2 = xindex // 1600
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tmp2 / tmp8
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 1.0
    tmp13 = tmp9 * tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tl.store(out_ptr0 + (x0 + 16640 * x2), tmp9, xmask)
    tl.store(out_ptr1 + (x0 + 16640 * x2), tmp15, xmask)
    tl.store(out_ptr2 + x1, tmp3, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_7(in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 11200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 19600 % 112
    x0 = xindex % 19600
    x2 = xindex // 19600
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tmp2 / tmp8
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 1.0
    tmp13 = tmp9 * tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tl.store(out_ptr0 + (x0 + 19600 * x2), tmp9, xmask)
    tl.store(out_ptr1 + (x0 + 19600 * x2), tmp15, xmask)
    tl.store(out_ptr2 + x1, tmp3, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_8(in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 11200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 19600 % 112
    x0 = xindex % 19600
    x2 = xindex // 19600
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tmp2 / tmp8
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 1.0
    tmp13 = tmp9 * tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tl.store(out_ptr0 + (x0 + 19600 * x2), tmp9, xmask)
    tl.store(out_ptr1 + (x0 + 19600 * x2), tmp15, xmask)
    tl.store(out_ptr2 + x1, tmp3, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_9(in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 19200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 19600 % 192
    x0 = xindex % 19600
    x2 = xindex // 19600
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tmp2 / tmp8
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 1.0
    tmp13 = tmp9 * tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tl.store(out_ptr0 + (x0 + 19600 * x2), tmp9, xmask)
    tl.store(out_ptr1 + (x0 + 19600 * x2), tmp15, xmask)
    tl.store(out_ptr2 + x1, tmp3, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_10(in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 19200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 19600 % 192
    x0 = xindex % 19600
    x2 = xindex // 19600
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tmp2 / tmp8
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 1.0
    tmp13 = tmp9 * tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tl.store(out_ptr0 + (x0 + 19600 * x2), tmp9, xmask)
    tl.store(out_ptr1 + (x0 + 19600 * x2), tmp15, xmask)
    tl.store(out_ptr2 + x1, tmp3, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_11(in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 19600 % 320
    x0 = xindex % 19600
    x2 = xindex // 19600
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tmp2 / tmp8
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 1.0
    tmp13 = tmp9 * tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tl.store(out_ptr0 + (x0 + 19600 * x2), tmp9, xmask)
    tl.store(out_ptr1 + (x0 + 19600 * x2), tmp15, xmask)
    tl.store(out_ptr2 + x1, tmp3, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_12(in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 19600 % 320
    x0 = xindex % 19600
    x2 = xindex // 19600
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tmp2 / tmp8
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 1.0
    tmp13 = tmp9 * tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tl.store(out_ptr0 + (x0 + 19600 * x2), tmp9, xmask)
    tl.store(out_ptr1 + (x0 + 19600 * x2), tmp15, xmask)
    tl.store(out_ptr2 + x1, tmp3, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_13(in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 400 % 1280
    x0 = xindex % 400
    x2 = xindex // 400
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tmp2 / tmp8
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 1.0
    tmp13 = tmp9 * tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tl.store(out_ptr0 + (x0 + 4096 * x2), tmp9, xmask)
    tl.store(out_ptr1 + (x0 + 4096 * x2), tmp15, xmask)
    tl.store(out_ptr2 + x1, tmp3, xmask)


@triton.jit
def triton_poi_fused__to_copy_14(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.0
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 + tmp2
    tmp6 = tmp5 * tmp2
    tmp7 = tmp6 + tmp2
    tmp8 = tmp7 * tmp2
    tmp9 = tmp8 + tmp2
    tmp10 = tmp9 * tmp2
    tmp11 = tmp10 + tmp2
    tmp12 = tmp11 * tmp2
    tmp13 = tmp12 + tmp2
    tmp14 = tmp13 * tmp2
    tmp15 = tmp14 + tmp2
    tmp16 = tmp15 * tmp2
    tmp17 = tmp16 + tmp2
    tmp18 = tmp17 * tmp2
    tmp19 = tmp18 + tmp2
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20 + tmp2
    tmp22 = tmp21 * tmp2
    tmp23 = tmp22 + tmp2
    tmp24 = tmp23 * tmp2
    tmp25 = tmp24 + tmp2
    tmp26 = tmp25 * tmp2
    tmp27 = tmp26 + tmp2
    tmp28 = tmp27 * tmp2
    tmp29 = tmp28 + tmp2
    tmp30 = tmp29 * tmp2
    tmp31 = tmp30 + tmp2
    tmp32 = tmp31 * tmp2
    tmp33 = tmp32 + tmp2
    tmp34 = tmp33 * tmp2
    tmp35 = tmp34 + tmp2
    tmp36 = tmp35 * tmp2
    tmp37 = tmp36 + tmp2
    tmp38 = tmp37 * tmp2
    tmp39 = tmp38 + tmp2
    tmp40 = tmp39 * tmp2
    tmp41 = tmp40 + tmp2
    tmp42 = tmp41 * tmp2
    tmp43 = tmp42 + tmp2
    tmp44 = tmp43 * tmp2
    tmp45 = tmp44 + tmp2
    tmp46 = tmp45 * tmp2
    tmp47 = tmp46 + tmp2
    tmp48 = tmp47 * tmp2
    tmp49 = tmp48 + tmp2
    tmp50 = tmp49 * tmp2
    tmp51 = tmp50 + tmp2
    tmp52 = tmp51 * tmp2
    tmp53 = tmp52 + tmp2
    tmp54 = tmp53 * tmp2
    tmp55 = tmp54 + tmp2
    tmp56 = tmp55 * tmp2
    tmp57 = tmp56 + tmp2
    tmp58 = tmp57 * tmp2
    tmp59 = tmp58 + tmp2
    tmp60 = tmp59 * tmp2
    tmp61 = tmp60 + tmp2
    tmp62 = tmp61 * tmp2
    tmp63 = tmp62 + tmp2
    tmp64 = tmp63 * tmp2
    tmp65 = tmp64 + tmp2
    tmp66 = tmp65 * tmp2
    tmp67 = tmp66 + tmp2
    tmp68 = tmp67 * tmp2
    tmp69 = tmp68 + tmp2
    tmp70 = tmp69 * tmp2
    tmp71 = tmp70 + tmp2
    tmp72 = tmp71 * tmp2
    tmp73 = tmp72 + tmp2
    tmp74 = tmp73 * tmp2
    tmp75 = tmp74 + tmp2
    tmp76 = tmp75 * tmp2
    tmp77 = tmp76 + tmp2
    tmp78 = tmp77 * tmp2
    tmp79 = tmp78 + tmp2
    tmp80 = tmp79 * tmp2
    tmp81 = tmp80 + tmp2
    tmp82 = tmp81 * tmp2
    tmp83 = tmp82 + tmp2
    tmp84 = tmp83 * tmp2
    tmp85 = tmp84 + tmp2
    tmp86 = tmp85 * tmp2
    tmp87 = tmp86 + tmp2
    tmp88 = tmp87 * tmp2
    tmp89 = tmp88 + tmp2
    tmp90 = tmp89 * tmp2
    tmp91 = tmp90 + tmp2
    tmp92 = tmp91 * tmp2
    tmp93 = tmp92 + tmp2
    tmp94 = tmp93 * tmp2
    tmp95 = tmp94 + tmp2
    tmp96 = tmp95 * tmp2
    tmp97 = tmp96 + tmp2
    tmp98 = tmp97 * tmp2
    tmp99 = tmp98 + tmp2
    tmp100 = tmp99 * tmp2
    tmp101 = tmp100 + tmp2
    tmp102 = tmp101 * tmp2
    tmp103 = tmp102 + tmp2
    tmp104 = tmp103 * tmp2
    tmp105 = tmp104 + tmp2
    tmp106 = tmp105 * tmp2
    tmp107 = tmp106 + tmp2
    tmp108 = tmp107 * tmp2
    tmp109 = tmp108 + tmp2
    tmp110 = tmp109 * tmp2
    tmp111 = tmp110 + tmp2
    tmp112 = tmp111 * tmp2
    tmp113 = tmp112 + tmp2
    tmp114 = tmp113 * tmp2
    tmp115 = tmp114 + tmp2
    tmp116 = tmp115 * tmp2
    tmp117 = tmp116 + tmp2
    tmp118 = tmp117 * tmp2
    tmp119 = tmp118 + tmp2
    tmp120 = tmp119 * tmp2
    tmp121 = tmp120 + tmp2
    tmp122 = tmp121 * tmp2
    tmp123 = tmp122 + tmp2
    tmp124 = tmp123 * tmp2
    tmp125 = tmp124 + tmp2
    tmp126 = tmp125 * tmp2
    tmp127 = tmp126 + tmp2
    tmp128 = tmp127 * tmp2
    tmp129 = tmp128 + tmp2
    tmp130 = tmp129 * tmp2
    tmp131 = tmp130 + tmp2
    tmp132 = tmp131 * tmp2
    tmp133 = tmp132 + tmp2
    tmp134 = tmp133 * tmp2
    tmp135 = tmp134 + tmp2
    tmp136 = tmp135 * tmp2
    tmp137 = tmp136 + tmp2
    tmp138 = tmp137 * tmp2
    tmp139 = tmp138 + tmp2
    tmp140 = tmp139 * tmp2
    tmp141 = tmp140 + tmp2
    tmp142 = tmp141 * tmp2
    tmp143 = tmp142 + tmp2
    tmp144 = tmp143 * tmp2
    tmp145 = tmp144 + tmp2
    tmp146 = tmp145 * tmp2
    tmp147 = tmp146 + tmp2
    tmp148 = tmp147 * tmp2
    tmp14