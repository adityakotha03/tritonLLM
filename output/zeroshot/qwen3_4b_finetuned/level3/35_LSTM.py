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
def triton_poi_fused__lstm_cell_forward_0(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = xindex // 6
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tmp3 * tmp3
    tmp11 = tmp8 - tmp9
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp15 = tmp10 + tmp14
    tmp16 = tmp5 + tmp15
    tl.store(out_ptr0 + x2, tmp7, xmask)
    tl.store(out_ptr1 + x2, tmp13, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)
    tl.store(out_ptr3 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__lstm_cell_forward_1(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = xindex // 6
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tmp3 * tmp3
    tmp11 = tmp8 - tmp9
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp15 = tmp10 + tmp14
    tmp16 = tmp5 + tmp15
    tl.store(out_ptr0 + x2, tmp7, xmask)
    tl.store(out_ptr1 + x2, tmp13, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)
    tl.store(out_ptr3 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__lstm_cell_forward_2(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = xindex // 6
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tmp3 * tmp3
    tmp11 = tmp8 - tmp9
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp15 = tmp10 + tmp14
    tmp16 = tmp5 + tmp15
    tl.store(out_ptr0 + x2, tmp7, xmask)
    tl.store(out_ptr1 + x2, tmp13, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)
    tl.store(out_ptr3 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__lstm_cell_forward_3(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = xindex // 6
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tmp3 * tmp3
    tmp11 = tmp8 - tmp9
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp15 = tmp10 + tmp14
    tmp16 = tmp5 + tmp15
    tl.store(out_ptr0 + x2, tmp7, xmask)
    tl.store(out_ptr1 + x2, tmp13, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)
    tl.store(out_ptr3 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__lstm_cell_forward_4(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = xindex // 6
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tmp3 * tmp3
    tmp11 = tmp8 - tmp9
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp15 = tmp10 + tmp14
    tmp16 = tmp5 + tmp15
    tl.store(out_ptr0 + x2, tmp7, xmask)
    tl.store(out_ptr1 + x2, tmp13, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)
    tl.store(out_ptr3 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__lstm_cell_forward_5(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = xindex // 6
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tmp3 * tmp3
    tmp11 = tmp8 - tmp9
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp15 = tmp10 + tmp14
    tmp16 = tmp5 + tmp15
    tl.store(out_ptr0 + x2, tmp7, xmask)
    tl.store(out_ptr1 + x2, tmp13, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)
    tl.store(out_ptr3 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__lstm_cell_forward_6(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = xindex // 6
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tmp3 * tmp3
    tmp11 = tmp8 - tmp9
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp15 = tmp10 + tmp14
    tmp16 = tmp5 + tmp15
    tl.store(out_ptr0 + x2, tmp7, xmask)
    tl.store(out_ptr1 + x2, tmp13, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)
    tl.store(out_ptr3 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__lstm_cell_forward_7(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = xindex // 6
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tmp3 * tmp3
    tmp11 = tmp8 - tmp9
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp15 = tmp10 + tmp14
    tmp16 = tmp5 + tmp15
    tl.store(out_ptr0 + x2, tmp7, xmask)
    tl.store(out_ptr1 + x2, tmp13, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)
    tl.store(out_ptr3 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__lstm_cell_forward_8(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = xindex // 6
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tmp3 * tmp3
    tmp11 = tmp8 - tmp9
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp15 = tmp10 + tmp14
    tmp16 = tmp5 + tmp15
    tl.store(out_ptr0 + x2, tmp7, xmask)
    tl.store(out_ptr1 + x2, tmp13, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)
    tl.store(out_ptr3 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__lstm_cell_forward_9(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = xindex // 6
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tmp3 * tmp3
    tmp11 = tmp8 - tmp9
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp15 = tmp10 + tmp14
    tmp16 = tmp5 + tmp15
    tl.store(out_ptr0 + x2, tmp7, xmask)
    tl.store(out_ptr1 + x2, tmp13, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)
    tl.store(out_ptr3 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__lstm_cell_forward_10(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = xindex // 6
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tmp3 * tmp3
    tmp11 = tmp8 - tmp9
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp15 = tmp10 + tmp14
    tmp16 = tmp5 + tmp15
    tl.store(out_ptr0 + x2, tmp7, xmask)
    tl.store(out_ptr1 + x2, tmp13, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)
    tl.store(out_ptr3 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__lstm_cell_forward_11(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = xindex // 6
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tmp3 * tmp3
    tmp11 = tmp8 - tmp9
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp15 = tmp10 + tmp14
    tmp16 = tmp5 + tmp15
    tl.store(out_ptr0 + x2, tmp7, xmask)
    tl.store(out_ptr1 + x2, tmp13, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)
    tl.store(out_ptr3 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__lstm_cell_forward_12(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = xindex // 6
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tmp3 * tmp3
    tmp11 = tmp8 - tmp9
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp15 = tmp10 + tmp14
    tmp16 = tmp5 + tmp15
    tl.store(out_ptr0 + x2, tmp7, xmask)
    tl.store(out_ptr1 + x2, tmp13, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)
    tl.store(out_ptr3 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__lstm_cell_forward_13(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = xindex // 6
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tmp3 * tmp3
    tmp11 = tmp8 - tmp9
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp15 = tmp10 + tmp14
    tmp16 = tmp5 + tmp15
    tl.store(out_ptr0 + x2, tmp7, xmask)
    tl.store(out_ptr1 + x2, tmp13, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)
    tl.store(out_ptr3 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__lstm_cell_forward_14(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = xindex // 6
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tmp3 * tmp3
    tmp11 = tmp8 - tmp9
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp15 = tmp10 + tmp14
    tmp16 = tmp5 + tmp15
    tl.store(out_ptr0 + x2, tmp7, xmask)
    tl.store(out_ptr1 + x2, tmp13, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)
    tl.store(out_ptr3 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__lstm_cell_forward_15(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = xindex // 6
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tmp3 * tmp3
    tmp11 = tmp8 - tmp9
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp15 = tmp10 + tmp14
    tmp16 = tmp5 + tmp15
    tl.store(out_ptr0 + x2, tmp7, xmask)
    tl.store(out_ptr1 + x2, tmp13, xmask)
    tl.store(out_ptr2 + x2, tmp16, xmask)
    tl.store(out_ptr3 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__lstm_cell_forward_16(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = xindex //