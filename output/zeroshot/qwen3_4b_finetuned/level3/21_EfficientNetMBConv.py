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


@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 112
    xnumel = 112
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 7
    y1 = yindex // 7
    tmp0 = tl.load(in_ptr0 + (x2 + 112 * y3), xmask & ymask, eviction_policy
        ='evict_last')
    tl.store(out_ptr0 + (y0 + 7 * x2 + 784 * y1), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 112
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 7
    y1 = yindex // 7
    tmp0 = tl.load(in_ptr0 + (x2 + 192 * y3), xmask & ymask, eviction_policy
        ='evict_last')
    tl.store(out_ptr0 + (y0 + 7 * x2 + 1344 * y1), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 25
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = yindex // 112
    tmp0 = tl.load(in_ptr0 + (x2 + 25 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 112 * x2 + 2800 * y1), tmp0, xmask)


@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 25
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (x2 + 25 * y3), xmask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 192 * x2 + 4800 * y1), tmp0, xmask)


@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_relu_repeat_4(in_ptr0
    , in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel,
    rnumel, XBLOCK: tl.constexpr):
    xnumel = 25
    RBLOCK: tl.constexpr = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 192 * x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + r1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 25, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 25.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tl.store(out_ptr0 + x0, tmp12, xmask)
    tl.store(out_ptr1 + x0, tmp24, xmask)
    tl.store(out_ptr2 + (r1 + 192 * x0), tmp2, xmask)
    tl.store(out_ptr3 + x0, tmp18, xmask)
    tl.store(out_ptr4 + x0, tmp20, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_repeat_5(in_ptr0
    , in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 25
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + x0, xmask)
    tmp7 = tl.load(in_ptr3 + x0, xmask)
    tmp9 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 - tmp4
    tmp6 = 25.0
    tmp8 = tmp7 / tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tl.store(out_ptr0 + x0, tmp8, xmask)
    tl.store(out_ptr1 + x0, tmp12, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_repeat_6(in_ptr0
    , in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK
    : tl.constexpr):
    xnumel = 25
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2800 + 4 * x0), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + x0, xmask)
    tmp7 = tl.load(in_ptr3 + x0, xmask)
    tmp9 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 - tmp4
    tmp6 = 25.0
    tmp8 = tmp7 / tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tl.store(out_ptr0 + x0, tmp8, xmask)
    tl.store(out_ptr1 + x0, tmp12, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_relu_repeat_7(in_ptr0
    , in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK
    : tl.constexpr):
    xnumel = 25
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4800 + 4 * x0), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + x0, xmask)
    tmp7 = tl.load(in_ptr3 + x0, xmask)
    tmp9 = tl.load(in_ptr4 + x0, xmask)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 - tmp4
    tmp6 = 25.0
    tmp8 = tmp7 / tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tl.store(out_ptr0 + x0, tmp8, xmask)
    tl.store(out_ptr1 + x0, tmp12, xmask)


@triton.jit
def triton_poi_fused_add_8(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 28224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 112
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9) = args
    args.clear()
    assert_size_stride(primals_1, (112, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_2, (112, 112, 224, 224), (573440, 5312, 224, 
        1))
    assert_size_stride(primals_3, (112,), (1,))
    assert_size_stride(primals_4, (112,), (1,))
    assert_size_stride(primals_5, (112,), (1,))
    assert_size_stride(primals_6, (192, 112, 5, 5), (2800, 25, 5, 1))
    assert_size_stride(primals_7, (192,), (1,))
    assert_size_stride(primals_8, (192,), (1,))
    assert_size_stride(primals_9, (192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((112, 112, 1, 1), (112, 1, 1, 1), torch.
            float32)
        get_raw_stream(0)
        triton_poi_fused_0[grid(112, 112)](primals_1, buf0, 112, 112,
            XBLOCK=32, YBLOCK=32, num_warps=4, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((192, 112, 1, 1), (112, 1, 1, 1), torch.
            float32)
        triton_poi_fused_1[grid(112, 192)](primals_6, buf1, 112, 192,
            XBLOCK=32, YBLOCK=32, num_warps=4, num_stages=1)
        del primals_6
        buf2 = empty_strided_cuda((112, 192, 5, 5), (4800, 25, 5, 1), torch
            .float32)
        triton_poi_fused_2[grid(112, 25)](primals_2, buf2, 112, 25, XBLOCK=
            32, YBLOCK=32, num_warps=4, num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((192, 192, 5, 5), (4800, 25, 5, 1), torch
            .float32)
        triton_poi_fused_3[grid(192, 25)](primals_3, buf3, 192, 25, XBLOCK=
            32, YBLOCK=32, num_warps=4, num_stages=1)
        del primals_3
        buf4 = empty_strided_cuda((25, 112), (112, 1), torch.float32)
        buf5 = empty_strided_cuda((25, 112), (112, 1), torch.float32)
        buf6 = empty_strided_cuda((25, 112, 192), (2208, 192, 1), torch.float32
            )
        buf7 = empty_strided_cuda((25, 112), (112, 1), torch.float32)
        buf8 = empty_strided_cuda((25, 112), (112, 1), torch.float32)
        triton_per_fused__native_batch_norm_legit_convolution_relu_repeat_4[grid
            (25)](buf2, primals_4, buf4, buf5, buf6, buf7, buf8, 25, 192,
            XBLOCK=1, num_warps=2, num_stages=1)
        buf9 = empty_strided_cuda((25, 112), (112, 1), torch.float32)
        buf10 = empty_strided_cuda((25, 112), (112, 1), torch.float32)
        triton_poi_fused__native_batch_norm_legit_convolution_relu_repeat_5[grid
            (25)](buf6, primals_5, buf4, buf5, buf7, buf9, buf10, 25,
            XBLOCK=32, num_warps=1, num_stages=1)
        del buf5
        del primals_5
        buf11 = empty_strided_cuda((25, 192), (192, 1), torch.float32)
        buf12 = empty_strided_cuda((25, 192), (192, 1), torch.float32)
        triton_poi_fused__native_batch_norm_legit_convolution_relu_repeat_6[grid
            (25)](buf3, primals_7, buf8, buf9, buf10, buf11, buf12, 25,
            XBLOCK=32, num_warps=1, num_stages=1)
        del buf10
        del buf9
        del primals_7
        buf13 = empty_strided_cuda((25, 192), (192, 1), torch.float32)
        buf14 = empty_strided_cuda((25, 192), (192, 1), torch.float32)
        triton_poi_fused__native_batch_norm_legit_convolution_relu_repeat_7[grid
            (25)](buf1, primals_8, buf12, buf11, buf13, buf14, buf15, 25,
            XBLOCK=32, num_warps=1, num_stages=1)
        del buf11
        del primals_8
        buf15 = empty_strided_cuda((25, 192), (192, 1), torch.float32)
        del buf12
        buf16 = empty_strided_cuda((25, 192), (192, 1), torch.float32)
        triton_poi_fused_add_8[grid(28224)](buf15, primals_9, buf14, 28224,
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf14
        del primals_9
    return (buf15, buf4, buf6, buf8, buf13, buf16, primals_4, primals_7,
        primals_8)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        MBConv block implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size for the depthwise convolution.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the intermediate channels.
        """
        super(ModelNew, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, input_0):
        primals_1 = self.expand_conv.conv.weight
        primals_3 = self.expand_conv.bn.weight
        primals_4 = self.expand_conv.bn.bias
        primals_6 = self.depthwise_conv.conv.weight
        primals_5 = self.depthwise_conv.bn.weight
        primals_7 = self.depthwise_conv.bn.bias
        primals_2 = self.project_conv.conv.weight
        primals_8 = self.project_conv.bn.weight
        primals_9 = self.project_conv.bn.bias
        primals_10 = input_0
        output = call([primals_1, primals_10, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9])
        return output[0]
