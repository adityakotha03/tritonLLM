import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from itertools import repeat
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16 % 16
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_1(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp1 - tmp8
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 - tmp8
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = tmp5 - tmp8
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19 / tmp7
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = tl.sqrt(tmp22)
    tl.store(out_ptr0 + x0, tmp8, xmask)
    tl.store(out_ptr1 + x0, tmp23, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 12
    x0 = xindex % 12
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 / tmp3
    tmp5 = tmp4 * tmp5
    tmp6 = tmp5 + tmp7
    tl.store(out_ptr0 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused__softmax_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 12
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 12 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 12 * x1), xmask, eviction_policy='evict_last'
        )
    tmp4 = tl.load(in_ptr0 + (2 + 12 * x1), xmask, eviction_policy='evict_last'
        )
    tmp6 = tl.load(in_ptr0 + (3 + 12 * x1), xmask, eviction_policy='evict_last'
        )
    tmp8 = tl.load(in_ptr0 + (4 + 12 * x1), xmask, eviction_policy='evict_last'
        )
    tmp10 = tl.load(in_ptr0 + (5 + 12 * x1), xmask, eviction_policy='evict_last'
        )
    tmp12 = tl.load(in_ptr0 + (6 + 12 * x1), xmask, eviction_policy='evict_last'
        )
    tmp14 = tl.load(in_ptr0 + (7 + 12 * x1), xmask, eviction_policy='evict_last'
        )
    tmp16 = tl.load(in_ptr0 + (8 + 12 * x1), xmask, eviction_policy='evict_last'
        )
    tmp18 = tl.load(in_ptr0 + (9 + 12 * x1), xmask, eviction_policy='evict_last'
        )
    tmp20 = tl.load(in_ptr0 + (10 + 12 * x1), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr0 + (11 + 12 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp21 = tmp19 + tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tmp0 / tmp23
    tl.store(out_ptr0 + x2, tmp24, xmask)


@triton.jit
def triton_poi_fused__softmax_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 12
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 12 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 12 * x1), xmask, eviction_policy='evict_last'
        )
    tmp4 = tl.load(in_ptr0 + (2 + 12 * x1), xmask, eviction_policy='evict_last'
        )
    tmp6 = tl.load(in_ptr0 + (3 + 12 * x1), xmask, eviction_policy='evict_last'
        )
    tmp8 = tl.load(in_ptr0 + (4 + 12 * x1), xmask, eviction_policy='evict_last'
        )
    tmp10 = tl.load(in_ptr0 + (5 + 12 * x1), xmask, eviction_policy='evict_last'
        )
    tmp12 = tl.load(in_ptr0 + (6 + 12 * x1), xmask, eviction_policy='evict_last'
        )
    tmp14 = tl.load(in_ptr0 + (7 + 12 * x1), xmask, eviction_policy='evict_last'
        )
    tmp16 = tl.load(in_ptr0 + (8 + 12 * x1), xmask, eviction_policy='evict_last'
        )
    tmp18 = tl.load(in_ptr0 + (9 + 12 * x1), xmask, eviction_policy='evict_last'
        )
    tmp20 = tl.load(in_ptr0 + (10 + 12 * x1), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr0 + (11 + 12 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp21 = tmp19 + tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tmp0 / tmp23
    tl.store(out_ptr0 + x2, tmp24, xmask)


@triton.jit
def triton_poi_fused__softmax_5(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 12 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp2 = tmp0 + tmp1
    tmp3 = tl.load(in_ptr0 + (2 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp4 = tmp2 + tmp3
    tmp5 = tl.load(in_ptr0 + (3 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp6 = tmp4 + tmp5
    tmp7 = tl.load(in_ptr0 + (4 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp8 = tmp6 + tmp7
    tmp9 = tl.load(in_ptr0 + (5 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp10 = tmp8 + tmp9
    tmp11 = tl.load(in_ptr0 + (6 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp12 = tmp10 + tmp11
    tmp13 = tl.load(in_ptr0 + (7 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp14 = tmp12 + tmp13
    tmp15 = tl.load(in_ptr0 + (8 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp16 = tmp14 + tmp15
    tmp17 = tl.load(in_ptr0 + (9 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp18 = tmp16 + tmp17
    tmp19 = tl.load(in_ptr0 + (10 + 12 * x0), xmask, eviction_policy=
        'evict_last')
    tmp20 = tmp18 + tmp19
    tmp21 = tl.load(in_ptr0 + (11 + 12 * x0), xmask, eviction_policy=
        'evict_last')
    tmp22 = tmp20 + tmp21
    tmp23 = tmp0 / tmp22
    tmp24 = tl.sigmoid(tmp23)
    tl.store(out_ptr0 + x0, tmp24, xmask)


@triton.jit
def triton_poi_fused__softmax_6(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 12 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (3 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (4 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (5 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (6 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (7 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (8 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr0 + (9 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (10 + 12 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 12 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp23 = tmp0 / tmp22
    tmp24 = tl.sigmoid(tmp23)
    tl.store(out_ptr0 + x0, tmp24, xmask)


@triton.jit
def triton_poi_fused__softmax_7(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 12 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (3 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (4 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (5 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (6 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (7 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (8 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr0 + (9 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (10 + 12 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 12 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp23 = tmp0 / tmp22
    tmp24 = tl.sigmoid(tmp23)
    tl.store(out_ptr0 + x0, tmp24, xmask)


@triton.jit
def triton_poi_fused__softmax_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 12 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (3 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (4 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (5 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (6 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (7 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (8 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr0 + (9 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (10 + 12 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 12 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp23 = tmp0 / tmp22
    tmp24 = tl.sigmoid(tmp23)
    tl.store(out_ptr0 + x0, tmp24, xmask)


@triton.jit
def triton_poi_fused__softmax_9(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 12 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (3 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (4 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (5 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (6 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (7 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (8 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr0 + (9 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (10 + 12 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 12 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp23 = tmp0 / tmp22
    tmp24 = tl.sigmoid(tmp23)
    tl.store(out_ptr0 + x0, tmp24, xmask)


@triton.jit
def triton_poi_fused__softmax_10(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 12 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (3 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (4 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (5 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (6 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (7 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (8 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr0 + (9 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (10 + 12 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 12 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp23 = tmp0 / tmp22
    tmp24 = tl.sigmoid(tmp23)
    tl.store(out_ptr0 + x0, tmp24, xmask)


@triton.jit
def triton_poi_fused__softmax_11(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 12 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (3 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (4 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (5 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (6 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (7 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (8 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr0 + (9 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (10 + 12 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 12 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp23 = tmp0 / tmp22
    tmp24 = tl.sigmoid(tmp23)
    tl.store(out_ptr0 + x0, tmp24, xmask)


@triton.jit
def triton_poi_fused__softmax_12(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 12 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (3 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (4 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (5 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (6 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (7 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (8 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr0 + (9 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (10 + 12 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 12 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp23 = tmp0 / tmp22
    tmp24 = tl.sigmoid(tmp23)
    tl.store(out_ptr0 + x0, tmp24, xmask)


@triton.jit
def triton_poi_fused__softmax_13(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 12 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (3 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (4 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (5 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (6 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (7 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (8 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr0 + (9 + 12 * x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (10 + 12 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 12 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp1