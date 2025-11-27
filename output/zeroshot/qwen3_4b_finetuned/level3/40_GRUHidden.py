import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_zeros_0(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused_zeros_1(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused_gru_cell_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tmp5 / tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp15
    tmp14 = libdevice.tanh(tmp13)
    tmp16 = tmp14 + tmp2
    tl.store(out_ptr0 + x2, tmp14, xmask)
    tl.store(out_ptr1 + x2, tmp16, xmask)
    tl.store(out_ptr2 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused_zeros_3(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused_gru_cell_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tmp5 / tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp15
    tmp14 = libdevice.tanh(tmp13)
    tmp16 = tmp14 + tmp2
    tl.store(out_ptr0 + x2, tmp14, xmask)
    tl.store(out_ptr1 + x2, tmp16, xmask)
    tl.store(out_ptr2 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused_gru_cell_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tmp5 / tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp15
    tmp14 = libdevice.tanh(tmp13)
    tmp16 = tmp14 + tmp2
    tl.store(out_ptr0 + x2, tmp14, xmask)
    tl.store(out_ptr1 + x2, tmp16, xmask)
    tl.store(out_ptr2 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused_gru_cell_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tmp5 / tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp15
    tmp14 = libdevice.tanh(tmp13)
    tmp16 = tmp14 + tmp2
    tl.store(out_ptr0 + x2, tmp14, xmask)
    tl.store(out_ptr1 + x2, tmp16, xmask)
    tl.store(out_ptr2 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused_gru_cell_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tmp5 / tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp15
    tmp14 = libdevice.tanh(tmp13)
    tmp16 = tmp14 + tmp2
    tl.store(out_ptr0 + x2, tmp14, xmask)
    tl.store(out_ptr1 + x2, tmp16, xmask)
    tl.store(out_ptr2 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused_gru_cell_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tmp5 / tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp15
    tmp14 = libdevice.tanh(tmp13)
    tmp16 = tmp14 + tmp2
    tl.store(out_ptr0 + x2, tmp14, xmask)
    tl.store(out_ptr1 + x2, tmp16, xmask)
    tl.store(out_ptr2 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused_gru_cell_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tmp5 / tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp15
    tmp14 = libdevice.tanh(tmp13)
    tmp16 = tmp14 + tmp2
    tl.store(out_ptr0 + x2, tmp14, xmask)
    tl.store(out_ptr1 + x2, tmp16, xmask)
    tl.store(out_ptr2 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused_gru_cell_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tmp5 / tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp15
    tmp14 = libdevice.tanh(tmp13)
    tmp16 = tmp14 + tmp2
    tl.store(out_ptr0 + x2, tmp14, xmask)
    tl.store(out_ptr1 + x2, tmp16, xmask)
    tl.store(out_ptr2 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused_gru_cell_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tmp5 / tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp15
    tmp14 = libdevice.tanh(tmp13)
    tmp16 = tmp14 + tmp2
    tl.store(out_ptr0 + x2, tmp14, xmask)
    tl.store(out_ptr1 + x2, tmp16, xmask)
    tl.store(out_ptr2 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused_gru_cell_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tmp5 / tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp15
    tmp14 = libdevice.tanh(tmp13)
    tmp16 = tmp14 + tmp2
    tl.store(out_ptr0 + x2, tmp14, xmask)
    tl.store(out_ptr1 + x2, tmp16, xmask)
    tl.store(out_ptr2 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused_gru_cell_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tmp5 / tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp15
    tmp14 = libdevice.tanh(tmp13)
    tmp16 = tmp14 + tmp2
    tl.store(out_ptr0 + x2, tmp14, xmask)
    tl.store(out_ptr1 + x2, tmp16, xmask)
    tl.store(out_ptr2 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused_gru_cell_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tmp5 / tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp15
    tmp14 = libdevice.tanh(tmp13)
    tmp16 = tmp14 + tmp2
    tl.store(out_ptr0 + x2, tmp14, xmask)
    tl.store(out_ptr1 + x2, tmp16, xmask)
    tl.store(out_ptr2 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused_gru_cell_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tmp5 / tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp15
    tmp14 = libdevice.tanh(tmp13)
    tmp16 = tmp14 + tmp2
    tl.store(out_ptr0 + x2, tmp14, xmask)
    tl.store(out_ptr1 + x2, tmp16, xmask)
    tl.store(out_ptr2 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused_gru_cell_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tmp5 / tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp15
    tmp14 = libdevice.tanh(tmp13)
    tmp16 = tmp14 + tmp2
    tl.store(out_ptr0 + x2, tmp14, xmask)
    tl.store(out_ptr1 + x2, tmp16, xmask)
    tl.store(out_ptr2 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused_gru_cell_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tmp5 / tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp15
    tmp14 = libdevice.tanh(tmp13)
    tmp16 = tmp14 + tmp2
    tl.store(out_ptr0 + x2, tmp14, xmask)
    tl.store(out_ptr1 + x2, tmp16, xmask)
    tl.store(out_ptr2 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused_gru_cell_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + x0