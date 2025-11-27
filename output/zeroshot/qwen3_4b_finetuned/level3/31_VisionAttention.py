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
def triton_poi_fused_native_layer_norm_0(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp1 + tmp1
    tmp3 = tmp0 * tmp0
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp7 = tl.sum(tmp5, 0)[:, None]
    tmp8 = tl.full([XBLOCK], 128, tl.int32)
    tmp9 = tmp8 * tmp8
    tmp10 = tmp9 / tmp8
    tmp11 = tmp7 / tmp8
    tmp12 = tmp10 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tmp13 / tmp10
    tmp15 = tmp0 - tmp11
    tmp16 = tmp15 / tmp10
    tl.store(out_ptr0 + x0, tmp16, xmask)
    tl.store(out_ptr1 + x0, tmp14, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_11(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_12(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_13(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_14(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_15(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_17(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_18(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_19(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_20(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_21(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_22(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_23(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_24(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_25(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_26(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_27(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_28(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_29(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.0078125
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_30(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
