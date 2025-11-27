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
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 2056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1025
    x1 = xindex // 1025
    x2 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (1024 * x1 + x0), tmp4 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 1025, tl.int64)
    tmp9 = tl.load(in_ptr1 + (x0 + -1024 * tl.full([1], 1024, tl.int32) + 1024 *
        x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + x2, tmp10, xmask)


@triton.jit
def triton_poi_fused_add_1(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 2
    xnumel = 1025
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
    tmp0 = tl.load(in_ptr0 + (y0 + 2 * x2 + 2050 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 1025 * y3), tmp2, xmask & ymask)


@triton.jit
def triton_poi_fused__prelu_kernel_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = 1.0
    tmp7 = tmp3 * tmp6
    tmp8 = tl.where(tmp5, tmp3, tmp7)
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tl.store(out_ptr1 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused__prelu_kernel_4(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = 1.0
    tmp7 = tmp3 * tmp6
    tmp8 = tl.where(tmp5, tmp3, tmp7)
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tl.store(out_ptr1 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused__prelu_kernel_6(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = 1.0
    tmp7 = tmp3 * tmp6
    tmp8 = tl.where(tmp5, tmp3, tmp7)
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tl.store(out_ptr1 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused__prelu_kernel_8(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = 1.0
    tmp7 = tmp3 * tmp6
    tmp8 = tl.where(tmp5, tmp3, tmp7)
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tl.store(out_ptr1 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused__prelu_kernel_10(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = 1.0
    tmp7 = tmp3 * tmp6
    tmp8 = tl.where(tmp5, tmp3, tmp7)
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tl.store(out_ptr1 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused__prelu_kernel_12(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = 1.0
    tmp7 = tmp3 * tmp6
    tmp8 = tl.where(tmp5, tmp3, tmp7)
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tl.store(out_ptr1 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused__prelu_kernel_14(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = 1.0
    tmp7 = tmp3 * tmp6
    tmp8 = tl.where(tmp5, tmp3, tmp7)
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tl.store(out_ptr1 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused__prelu_kernel_16(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = 1.0
    tmp7 = tmp3 * tmp6
    tmp8 = tl.where(tmp5, tmp3, tmp7)
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tl.store(out_ptr1 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused__prelu_kernel_18(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = 1.0
    tmp7 = tmp3 * tmp6
    tmp8 = tl.where(tmp5, tmp3, tmp7)
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tl.store(out_ptr1 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused__prelu_kernel_20(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = 1.0
    tmp7 = tmp3 * tmp6
    tmp8 = tl.where(tmp5, tmp3, tmp7)
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tl.store(out_ptr1 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused__prelu_kernel_22(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = 1.0
    tmp7 = tmp3 * tmp6
    tmp8 = tl.where(tmp5, tmp3, tmp7)
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tl.store(out_ptr1 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused__prelu_kernel_24(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = 1.0
    tmp7 = tmp3 * tmp6
    tmp8 = tl.where(tmp5, tmp3, tmp7)
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tl.store(out_ptr1 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused__prelu_kernel_26(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = 1.0
    tmp7 = tmp3 * tmp6
    tmp8 = tl.where(tmp5, tmp3, tmp7)
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tl.store(out_ptr1 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused__prelu_kernel_28(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = 1.0
    tmp7 = tmp3 * tmp6
    tmp8 = tl.where(tmp5, tmp3, tmp7)
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tl.store(out_ptr1 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused__prelu_kernel_30(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * X