import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 1024
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = yindex // 1024
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 1024 * x2 + 131072 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 128 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (2048 + x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (3072 + x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp11 = tl.where(xmask, tmp9, float('-inf'))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp7 - tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = tmp14 == tmp12
    tmp16 = tmp15 & xmask
    tmp17 = tl.load(in_ptr0 + (1024 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp18 = tl.load(in_ptr0 + (2048 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp19 = tl.load(in_ptr0 + (3072 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp20 = tmp17 + tmp18
    tmp21 = tmp20 + tmp19
    tmp22 = tmp13 + tmp21
    tmp23 = tl.sigmoid(tmp22)
    tmp24 = tl.sigmoid(tmp22)
    tmp25 = tl.sigmoid(tmp22)
    tmp26 = tl.sigmoid(tmp22)
    tmp27 = tmp24 * tmp25
    tmp28 = tmp26 * tmp27
    tmp29 = tmp23 * tmp28
    tl.store(out_ptr0 + x2, tmp29, xmask)


@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (2048 + x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (3072 + x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp11 = tl.where(xmask, tmp9, float('-inf'))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp7 - tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = tmp14 == tmp12
    tmp16 = tmp15 & xmask
    tmp17 = tl.load(in_ptr0 + (1024 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp18 = tl.load(in_ptr0 + (2048 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp19 = tl.load(in_ptr0 + (3072 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp20 = tmp17 + tmp18
    tmp21 = tmp20 + tmp19
    tmp22 = tmp13 + tmp21
    tmp23 = tl.sigmoid(tmp22)
    tmp24 = tl.sigmoid(tmp22)
    tmp25 = tl.sigmoid(tmp22)
    tmp26 = tl.sigmoid(tmp22)
    tmp27 = tmp24 * tmp25
    tmp28 = tmp26 * tmp27
    tmp29 = tmp23 * tmp28
    tl.store(out_ptr0 + x2, tmp29, xmask)


@triton.jit
def triton_poi_fused__softmax_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (2048 + x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (3072 + x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp11 = tl.where(xmask, tmp9, float('-inf'))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp7 - tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = tmp14 == tmp12
    tmp16 = tmp15 & xmask
    tmp17 = tl.load(in_ptr0 + (1024 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp18 = tl.load(in_ptr0 + (2048 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp19 = tl.load(in_ptr0 + (3072 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp20 = tmp17 + tmp18
    tmp21 = tmp20 + tmp19
    tmp22 = tmp13 + tmp21
    tmp23 = tl.sigmoid(tmp22)
    tmp24 = tl.sigmoid(tmp22)
    tmp25 = tl.sigmoid(tmp22)
    tmp26 = tl.sigmoid(tmp22)
    tmp27 = tmp24 * tmp25
    tmp28 = tmp26 * tmp27
    tmp29 = tmp23 * tmp28
    tl.store(out_ptr0 + x2, tmp29, xmask)


@triton.jit
def triton_poi_fused_add_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused__softmax_5(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (2048 + x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (3072 + x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp11 = tl.where(xmask, tmp9, float('-inf'))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp7 - tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = tmp14 == tmp12
    tmp16 = tmp15 & xmask
    tmp17 = tl.load(in_ptr0 + (1024 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp18 = tl.load(in_ptr0 + (2048 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp19 = tl.load(in_ptr0 + (3072 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp20 = tmp17 + tmp18
    tmp21 = tmp20 + tmp19
    tmp22 = tmp13 + tmp21
    tmp23 = tl.sigmoid(tmp22)
    tmp24 = tl.sigmoid(tmp22)
    tmp25 = tl.sigmoid(tmp22)
    tmp26 = tl.sigmoid(tmp22)
    tmp27 = tmp24 * tmp25
    tmp28 = tmp26 * tmp27
    tmp29 = tmp23 * tmp28
    tl.store(out_ptr0 + x2, tmp29, xmask)


@triton.jit
def triton_poi_fused_add_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused__softmax_7(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (2048 + x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (3072 + x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp11 = tl.where(xmask, tmp9, float('-inf'))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp7 - tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = tmp14 == tmp12
    tmp16 = tmp15 & xmask
    tmp17 = tl.load(in_ptr0 + (1024 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp18 = tl.load(in_ptr0 + (2048 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp19 = tl.load(in_ptr0 + (3072 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp20 = tmp17 + tmp18
    tmp21 = tmp20 + tmp19
    tmp22 = tmp13 + tmp21
    tmp23 = tl.sigmoid(tmp22)
    tmp24 = tl.sigmoid(tmp22)
    tmp25 = tl.sigmoid(tmp22)
    tmp26 = tl.sigmoid(tmp22)
    tmp27 = tmp24 * tmp25
    tmp28 = tmp26 * tmp27
    tmp29 = tmp23 * tmp28
    tl.store(out_ptr0 + x2, tmp29, xmask)


@triton.jit
def triton_poi_fused_add_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused__softmax_9(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (2048 + x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (3072 + x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp11 = tl.where(xmask, tmp9, float('-inf'))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp7 - tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = tmp14 == tmp12
    tmp16 = tmp15 & xmask
    tmp17 = tl.load(in_ptr0 + (1024 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp18 = tl.load(in_ptr0 + (2048 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp19 = tl.load(in_ptr0 + (3072 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp20 = tmp17 + tmp18
    tmp21 = tmp20 + tmp19
    tmp22 = tmp13 + tmp21
    tmp23 = tl.sigmoid(tmp22)
    tmp24 = tl.sigmoid(tmp22)
    tmp25 = tl.sigmoid(tmp22)
    tmp26 = tl.sigmoid(tmp22)
    tmp27 = tmp24 * tmp25
    tmp28 = tmp26 * tmp27
    tmp29 = tmp23 * tmp28
    tl.store(out_ptr0 + x2, tmp29, xmask)


@triton.jit
def triton_poi_fused_add_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused__softmax_11(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (2048 + x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (3072 + x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp11 = tl.where(xmask, tmp9, float('-inf'))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp7 - tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = tmp14 == tmp12
    tmp16 = tmp15 & xmask
    tmp17 = tl.load(in_ptr0 + (1024 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp18 = tl.load(in_ptr0 + (2048 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp19 = tl.load(in_ptr0 + (3072 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp20 = tmp17 + tmp18
    tmp21 = tmp20 + tmp19
    tmp22 = tmp13 + tmp21
    tmp23 = tl.sigmoid(tmp22)
    tmp24 = tl.sigmoid(tmp22)
    tmp25 = tl.sigmoid(tmp22)
    tmp26 = tl.sigmoid(tmp22)
    tmp27 = tmp24 * tmp25
    tmp28 = tmp26 * tmp27
    tmp29 = tmp23 * tmp28
    tl.store(out_ptr0 + x2, tmp29, xmask)


@triton.jit
def triton_poi_fused_add_12(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused__softmax_13(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (2048 + x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (3072 + x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp11 = tl.where(xmask, tmp9, float('-inf'))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp7 - tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = tmp14 == tmp12
    tmp16 = tmp15 & xmask
    tmp17 = tl.load(in_ptr0 + (1024 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp18 = tl.load(in_ptr0 + (2048 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp19 = tl.load(in_ptr0 + (3072 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp20 = tmp17 + tmp18
    tmp21 = tmp20 + tmp19
    tmp22 = tmp13 + tmp21
    tmp23 = tl.sigmoid(tmp22)
    tmp24 = tl.sigmoid(tmp22)
    tmp25 = tl.sigmoid(tmp22)
    tmp26 = tl.sigmoid(tmp22)
    tmp27 = tmp24 * tmp25
    tmp28 = tmp26 * tmp27
    tmp29 = tmp23 * tmp28
    tl.store(out_ptr0 + x2, tmp29, xmask)


@triton.jit
def triton_poi_fused_add_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused__softmax_15(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (2048 + x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (3072 + x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp11 = tl.where(xmask, tmp9, float('-inf'))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp7 - tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = tmp14 == tmp12
    tmp16 = tmp15 & xmask
    tmp17 = tl.load(in_ptr0 + (1024 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp18 = tl.load(in_ptr0 + (2048 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp19 = tl.load(in_ptr0 + (3072 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp20 = tmp17 + tmp18
    tmp21 = tmp20 + tmp19
    tmp22 = tmp13 + tmp21
    tmp23 = tl.sigmoid(tmp22)
    tmp24 = tl.sigmoid(tmp22)
    tmp25 = tl.sigmoid(tmp22)
    tmp26 = tl.sigmoid(tmp22)
    tmp27 = tmp24 * tmp25
    tmp28 = tmp26 * tmp27
    tmp29 = tmp23 * tmp28
    tl.store(out_ptr0 + x2, tmp29, xmask)


@triton.jit
def triton_poi_fused_add_16(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused__softmax_17(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (2048 + x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (3072 + x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp11 = tl.where(xmask, tmp9, float('-inf'))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp7 - tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = tmp14 == tmp12
    tmp16 = tmp15 & xmask
    tmp17 = tl.load(in_ptr0 + (1024 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp18 = tl.load(in_ptr0 + (2048 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp19 = tl.load(in_ptr0 + (3072 + x2), tmp16, eviction_policy='evict_last'
        ).to(tl.float32)
    tmp20 = tmp17 + tmp18
    tmp21 = tmp20 + tmp19
    tmp22 = tmp13 + tmp21
    tmp23 = tl.sigmoid(tmp22)
    tmp24 = tl.sigmoid(tmp22)
    tmp25 = tl.sigmoid(tmp22)
    tmp26 = tl.sigmoid(tmp22)
    tmp27 = tmp24 * tmp25
    tmp28 = tmp26 * tmp27
    tmp29 = tmp23 * tmp28
    tl.store(out_ptr0 + x2, tmp29, xmask)


@triton.jit
def triton_poi_fused_add_18(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused__softmax_19(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (2048 +