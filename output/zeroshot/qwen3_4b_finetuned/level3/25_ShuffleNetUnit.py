import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 1
    xnumel = 480
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 480 * y0), xmask & ymask, eviction_policy
        ='evict_last')
    tl.store(out_ptr0 + (x1 + 480 * y0), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_clone_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 1
    xnumel = 240
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 240 * y0), xmask & ymask, eviction_policy
        ='evict_last')
    tl.store(out_ptr0 + (x1 + 240 * y0), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_clone_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 480
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 480 * y0), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (x1 + 480 * y0), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 240
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 240 * y0), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (x1 + 240 * y0), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 1
    xnumel = 480
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 480 * y0), xmask & ymask, eviction_policy
        ='evict_last')
    tl.store(out_ptr0 + (x1 + 480 * y0), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_clone_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 480
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 480 * y0), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (x1 + 480 * y0), tmp0, xmask)


@triton.jit
def triton_poi_fused_relu_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 235200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_poi_fused_clone_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 480
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 480 * y0), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (x1 + 480 * y0), tmp0, xmask)


@triton.jit
def triton_poi_fused_clone_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 1
    xnumel = 240
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 240 * y0), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (x1 + 240 * y0), tmp0, xmask)


@triton.jit
def triton_poi_fused_add_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4,
    in_ptr5, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 235200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 480 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tl.load(in_ptr2 + x0, xmask)
    tmp5 = tl.load(in_ptr3 + x0, xmask)
    tmp8 = tl.load(in_ptr4 + x0, xmask)
    tmp10 = tl.load(in_ptr5 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl_math.relu(tmp6)
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 + tmp10
    tl.store(out_ptr0 + x0, tmp11, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15, primals_16, primals_17,
        primals_18, primals_19, primals_20, primals_21, primals_22,
        primals_23, primals_24, primals_25, primals_26) = args
    args.clear()
    assert_size_stride(primals_1, (240, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_2, (120,), (1,))
    assert_size_stride(primals_3, (10, 240, 224, 224), (125440, 512, 2, 1))
    assert_size_stride(primals_4, (240,), (1,))
    assert_size_stride(primals_5, (120, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_6, (120,), (1,))
    assert_size_stride(primals_7, (120, 120, 3, 3), (1080, 9, 3, 1))
    assert_size_stride(primals_8, (120,), (1,))
    assert_size_stride(primals_9, (120, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_10, (120,), (1,))
    assert_size_stride(primals_11, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_12, (480,), (1,))
    assert_size_stride(primals_13, (480,), (1,))
    assert_size_stride(primals_14, (10, 480, 224, 224), (107520, 224, 1, 1))
    assert_size_stride(primals_15, (10,), (1,))
    assert_size_stride(primals_16, (10, 1000), (1000, 1))
    assert_size_stride(primals_17, (10,), (1,))
    assert_size_stride(primals_18, (10, 1000), (1000, 1))
    assert_size_stride(primals_19, (10,), (1,))
    assert_size_stride(primals_20, (240, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_21, (480,), (1,))
    assert_size_stride(primals_22, (480,), (1,))
    assert_size_stride(primals_23, (480, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_24, (480,), (1,))
    assert_size_stride(primals_25, (480,), (1,))
    assert_size_stride(primals_26, (10, 480, 224, 224), (107520, 224, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf0, (10, 120, 224, 224), (592704, 480, 2, 1))
        buf1 = empty_strided_cuda((10, 120, 224, 224), (592704, 480, 2, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(1, 480)](buf0, buf1, 1, 480, XBLOCK=
            16, YBLOCK=1, num_warps=1, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((10, 120, 224, 224), (592704, 480, 2, 1),
            torch.float32)
        triton_poi_fused_clone_1[grid(1, 240)](primals_2, buf2, 1, 240,
            XBLOCK=16, YBLOCK=1, num_warps=1, num_stages=1)
        buf3 = extern_kernels.convolution(buf1, primals_5, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf3, (10, 120, 224, 224), (592704, 480, 2, 1))
        buf4 = empty_strided_cuda((10, 120, 224, 224), (592704, 480, 2, 1),
            torch.float32)
        triton_poi_fused_clone_2[grid(1, 480)](buf3, buf4, 1, 480, XBLOCK=
            16, YBLOCK=1, num_warps=1, num_stages=1)
        del buf3
        buf5 = empty_strided_cuda((10, 120, 224, 224), (592704, 480, 2, 1),
            torch.float32)
        triton_poi_fused_clone_3[grid(1, 240)](primals_6, buf5, 1, 240,
            XBLOCK=16, YBLOCK=1, num_warps=1, num_stages=1)
        buf6 = extern_kernels.convolution(buf4, primals_7, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf6, (10, 120, 224, 224), (592704, 480, 2, 1))
        buf7 = empty_strided_cuda((10, 120, 224, 224), (592704, 480, 2, 1),
            torch.float32)
        triton_poi_fused_clone_4[grid(1, 480)](buf6, buf7, 1, 480, XBLOCK=
            16, YBLOCK=1, num_warps=1, num_stages=1)
        del buf6
        buf8 = empty_strided_cuda((10, 120, 224, 224), (592704, 480, 2, 1),
            torch.float32)
        triton_poi_fused_clone_5[grid(1, 480)](primals_8, buf8, 1, 480,
            XBLOCK=16, YBLOCK=1, num_warps=1, num_stages=1)
        buf9 = empty_strided_cuda((10, 120, 224, 224), (592704, 480, 2, 1),
            torch.float32)
        extern_kernels.convolution(buf7, primals_9, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf9, (10, 120, 224, 224), (592704, 480, 2, 1))
        buf10 = empty_strided_cuda((10, 120, 224, 224), (592704, 480, 2, 1),
            torch.float32)
        triton_poi_fused_clone_7[grid(1, 480)](buf9, buf10, 1, 480, XBLOCK=
            16, YBLOCK=1, num_warps=1, num_stages=1)
        del buf9
        buf11 = empty_strided_cuda((10, 120, 224, 224), (592704, 480, 2, 1),
            torch.float32)
        triton_poi_fused_clone_8[grid(1, 240)](primals_10, buf11, 1, 240,
            XBLOCK=16, YBLOCK=1, num_warps=1, num_stages=1)
        buf12 = extern_kernels.convolution(buf10, primals_11, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf12, (10, 480, 224, 224), (1105920, 240, 2, 1))
        buf13 = empty_strided_cuda((10, 480, 224, 224), (1105920, 240, 2, 1
            ), torch.float32)
        triton_poi_fused_clone_0[grid(1, 480)](buf12, buf13, 1, 480, XBLOCK
            =16, YBLOCK=1, num_warps=1, num_stages=1)
        del buf12
        buf14 = extern_kernels.convolution(primals_14, primals_16, stride=(
            1, 1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=10, bias=None)
        assert_size_stride(buf14, (10, 1000, 224, 224), (4928000, 4928, 2, 1
            ))
        buf15 = empty_strided_cuda((10, 1000, 224, 224), (4928000, 4928, 2,
            1), torch.float32)
        triton_poi_fused_clone_0[grid(1, 480)](buf14, buf15, 1, 480, XBLOCK
            =16, YBLOCK=1, num_warps=1, num_stages=1)
        del buf14
        buf16 = extern_kernels.convolution(buf13, primals_19, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf16, (10, 480, 224, 224), (1105920, 240, 2, 1))
        buf17 = empty_strided_cuda((10, 480, 224, 224), (1105920, 240, 2, 1
            ), torch.float32)
        triton_poi_fused_clone_0[grid(1, 480)](buf16, buf17, 1, 480, XBLOCK
            =16, YBLOCK=1, num_warps=1, num_stages=1)
        del buf16
        buf18 = empty_strided_cuda((10, 480, 224, 224), (1105920, 240, 2, 1
            ), torch.float32)
        extern_kernels.convolution(buf15, primals_17, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=10, bias=None)
        assert_size_stride(buf18, (10, 480, 224, 224), (1105920, 240, 2, 1))
        buf19 = empty_strided_cuda((10, 480, 224, 224), (1105920, 240, 2, 1
            ), torch.float32)
        triton_poi_fused_clone_0[grid(1, 480)](buf18, buf19, 1, 480, XBLOCK
            =16, YBLOCK=1, num_warps=1, num_stages=1)
        del buf18
        buf20 = extern_kernels.convolution(buf19, primals_20, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf20, (10, 480, 224, 224), (1105920, 240, 2, 1))
        buf21 = empty_strided_cuda((10, 480, 224, 224), (1105920, 240, 2, 1
            ), torch.float32)
        triton_poi_fused_clone_0[grid(1, 480)](buf20, buf21, 1, 480, XBLOCK
            =16, YBLOCK=1, num_warps=1, num_stages=1)
        del buf20
        buf22 = extern_kernels.convolution(primals_13, primals_21, stride=(
            1, 1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=10, bias=None)
        assert_size_stride(buf22, (10, 1000, 224, 224), (4928000, 4928, 2, 1
            ))
        buf23 = empty_strided_cuda((10, 1000, 224, 224), (4928000, 4928, 2,
            1), torch.float32)
        triton_poi_fused_clone_0[grid(1, 480)](buf22, buf23, 1, 480, XBLOCK
            =16, YBLOCK=1, num_warps=1, num_stages=1)
        del buf22
        buf24 = extern_kernels.convolution(buf21, primals_23, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf24, (10, 480, 224, 224), (1105920, 240, 2, 1))
        buf25 = empty_strided_cuda((10, 480, 224, 224), (1105920, 240, 2, 1
            ), torch.float32)
        triton_poi_fused_clone_0[grid(1, 480)](buf24, buf25, 1, 480, XBLOCK
            =16, YBLOCK=1, num_warps=1, num_stages=1)
        del buf24
        buf26 = extern_kernels.convolution(buf25, primals_25, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf26, (10, 480, 224, 224), (1105920, 240, 2, 1))
        buf27 = reinterpret_tensor(buf26, (10, 480, 224, 224), (1105920, 240,
            2, 1), 0)
        del buf26
        triton_poi_fused_clone_0[grid(1, 480)](buf27, buf27, 1, 480, XBLOCK=
            16, YBLOCK=1, num_warps=1, num_stages=1)
        buf28 = extern_kernels.convolution(buf27, primals_18, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf28, (10, 480, 224, 224), (1105920, 240, 2, 1))
        buf29 = empty_strided_cuda((10, 480, 224, 224), (1105920, 240, 2, 1
            ), torch.float32)
        triton_poi_fused_clone_0[grid(1, 480)](buf28, buf29, 1, 480, XBLOCK
            =16, YBLOCK=1, num_warps=1, num_stages=1)
        del buf28
        buf30 = empty_strided_cuda((10, 480, 224, 224), (1105920, 240, 2, 1
            ), torch.float32)
        extern_kernels.convolution(buf23, primals_17, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=10, bias=None)
        assert_size_stride(buf30, (10, 480, 224, 224), (1105920, 240, 2, 1))
        buf31 = reinterpret_tensor(buf30, (10, 480, 224, 224), (1105920, 240,
            2, 1), 0)
        del buf30
        triton_poi_fused_clone_0[grid(1, 480)](buf31, buf31, 1, 480, XBLOCK
            =16, YBLOCK=1, num_warps=1, num_stages=1)
        buf32 = extern_kernels.convolution(buf31, primals_21, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf32, (10, 480, 224, 224), (1105920, 240, 2, 1))
        buf33 = empty_strided_cuda((10, 480, 224, 224), (1105920, 240, 2, 1
            ), torch.float32)
        triton_poi_fused_clone_0[grid(1, 480)](buf32, buf33, 1, 480, XBLOCK
            =16, YBLOCK=1, num_warps=1, num_stages=1)
        del buf32
        buf34 = extern_kernels.convolution(buf33, primals_24, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf34, (10, 480, 224, 224), (1105920, 240, 2, 1))
        buf35 = empty_strided_cuda((10, 480, 224, 224), (1105920, 240, 2, 1
            ), torch.float32)
        triton_poi_fused_clone_0[grid(1, 480)](buf34, buf35, 1, 480, XBLOCK
            =16, YBLOCK=1, num_warps=1, num_stages=1)
        del buf34
        buf36 = extern_kernels.convolution(buf35, primals_26, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf36, (10, 480, 224, 224), (1105920, 240, 2, 1))
        buf37 = empty_strided_cuda((10, 480, 224, 224), (1105920, 240, 2, 1
            ), torch.float32)
        triton_poi_fused_clone_0[grid(1, 480)](buf36, buf37, 1, 480, XBLOCK
            =16, YBLOCK=1, num_warps=1, num_stages=1)
        del buf36
        buf38 = empty_strided_cuda((10, 1000, 224, 224), (4928000, 4928, 2,
            1), torch.float32)
        triton_poi_fused_clone_0[grid(1, 480)](primals_26, buf38, 1, 480,
            XBLOCK=16, YBLOCK=1, num_warps=1, num_stages=1)
        buf39 = empty_strided_cuda((10, 1000, 224, 224), (4928000, 4928, 2,
            1), torch.float32)
        extern_kernels.convolution(buf37, primals_17, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf39, (10, 1000, 224, 224), (4928000, 4928,