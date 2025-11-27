import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 192
    xnumel = 112
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = yindex // 192
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 192 * x2 + 21564 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 112 * y3), tmp2, xmask & ymask)


@triton.jit
def triton_poi_fused_convolution_relu_1(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 192
    xnumel = 112
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = yindex // 192
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 192 * x2 + 21564 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(out_ptr0 + (x2 + 112 * y3), tmp6, xmask & ymask)


@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 192
    xnumel = 112
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = yindex // 192
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 192 * x2 + 21564 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 112 * y3), tmp2, xmask & ymask)


@triton.jit
def triton_poi_fused_convolution_relu_3(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 192
    xnumel = 112
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = yindex // 192
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 192 * x2 + 21564 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(out_ptr0 + (x2 + 112 * y3), tmp6, xmask & ymask)


@triton.jit
def triton_poi_fused_convolution_4(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 192
    xnumel = 112
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = yindex // 192
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 192 * x2 + 21564 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 112 * y3), tmp2, xmask & ymask)


@triton.jit
def triton_poi_fused_convolution_5(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_6(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused_convolution_7(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused_add_8(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 192
    x2 = xindex // 4032
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp3
    tmp4 = tmp0 + tmp2
    tl.store(in_out_ptr0 + x3, tmp4, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14) = args
    args.clear()
    assert_size_stride(primals_1, (192, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_2, (192,), (1,))
    assert_size_stride(primals_3, (112, 112, 5, 5), (2800, 25, 5, 1))
    assert_size_stride(primals_4, (112,), (1,))
    assert_size_stride(primals_5, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_6, (192,), (1,))
    assert_size_stride(primals_7, (112, 192, 5, 5), (4900, 25, 5, 1))
    assert_size_stride(primals_8, (112,), (1,))
    assert_size_stride(primals_9, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_10, (192,), (1,))
    assert_size_stride(primals_11, (10, 112, 224, 224), (250880, 2240, 10, 1
        ))
    assert_size_stride(primals_12, (192,), (1,))
    assert_size_stride(primals_13, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_14, (192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((10, 192, 1, 1), (192, 1, 192, 192), torch
            .float32)
        triton_poi_fused_convolution_0[grid] = lambda meta: (meta['YBLOCK'], 
            meta['XBLOCK'])
        triton_poi_fused_convolution_0[grid](primals_1, primals_3, buf0, 
            192, 112, XBLOCK=128, YBLOCK=16, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((10, 112, 1, 1), (112, 1, 1, 1), torch.float32
            )
        triton_poi_fused_convolution_0[grid] = lambda meta: (meta['YBLOCK'], 
            meta['XBLOCK'])
        triton_poi_fused_convolution_0[grid](primals_4, primals_2, buf1, 112,
            112, XBLOCK=128, YBLOCK=16, num_warps=4, num_stages=1)
        del primals_2
        del primals_4
        buf2 = empty_strided_cuda((10, 112, 1, 1), (112, 1, 1, 1), torch.bool)
        triton_poi_fused_convolution_relu_1[grid] = lambda meta: (meta['YBLOCK'
            ], meta['XBLOCK'])
        triton_poi_fused_convolution_relu_1[grid](buf0, primals_4, buf2, 192,
            112, XBLOCK=128, YBLOCK=16, num_warps=4, num_stages=1)
        buf3 = empty_strided_cuda((10, 192, 1, 1), (192, 1, 1, 1), torch.float32
            )
        triton_poi_fused_convolution_2[grid] = lambda meta: (meta['YBLOCK'], 
            meta['XBLOCK'])
        triton_poi_fused_convolution_2[grid](primals_5, primals_7, buf3, 192,
            112, XBLOCK=128, YBLOCK=16, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((10, 112, 1, 1), (112, 1, 1, 1), torch.bool)
        triton_poi_fused_convolution_relu_1[grid] = lambda meta: (meta['YBLOCK'
            ], meta['XBLOCK'])
        triton_poi_fused_convolution_relu_1[grid](buf3, primals_6, buf4, 192,
            112, XBLOCK=128, YBLOCK=16, num_warps=4, num_stages=1)
        del primals_6
        buf5 = empty_strided_cuda((10, 192, 1, 1), (192, 1, 1, 1), torch.float32
            )
        triton_poi_fused_convolution_4[grid] = lambda meta: (meta['YBLOCK'], 
            meta['XBLOCK'])
        triton_poi_fused_convolution_4[grid](primals_9, primals_13, buf5, 
            192, 112, XBLOCK=128, YBLOCK=16, num_warps=4, num_stages=1)
        buf6 = empty_strided_cuda((10, 112, 1, 1), (112, 1, 1, 1), torch.bool)
        triton_poi_fused_convolution_relu_3[grid] = lambda meta: (meta['YBLOCK'
            ], meta['XBLOCK'])
        triton_poi_fused_convolution_relu_3[grid](buf5, primals_10, buf6, 192,
            112, XBLOCK=128, YBLOCK=16, num_warps=4, num_stages=1)
        del primals_10
        buf7 = empty_strided_cuda((10, 192), (192, 1), torch.float32)
        triton_poi_fused_convolution_5[grid] = lambda meta: (meta['XBLOCK'],)
        triton_poi_fused_convolution_5[grid](primals_12, buf7, 192, XBLOCK=
            128, num_warps=4, num_stages=1)
        buf8 = empty_strided_cuda((10, 192), (192, 1), torch.float32)
        triton_poi_fused_convolution_6[grid] = lambda meta: (meta['XBLOCK'],)
        triton_poi_fused_convolution_6[grid](primals_14, buf8, 192, XBLOCK=
            128, num_warps=4, num_stages=1)
        buf9 = empty_strided_cuda((10, 192), (192, 1), torch.float32)
        triton_poi_fused_convolution_7[grid] = lambda meta: (meta['XBLOCK'],)
        triton_poi_fused_convolution_7[grid](primals_13, buf9, 192, XBLOCK=
            128, num_warps=4, num_stages=1)
        del primals_13
        buf10 = empty_strided_cuda((10, 112, 224, 224), (250880, 2240, 10, 1
            ), torch.float32)
        triton_poi_fused_add_8[grid] = lambda meta: (meta['XBLOCK'],)
        triton_poi_fused_add_8[grid](buf10, primals_11, buf7, 25088, XBLOCK=
            512, num_warps=4, num_stages=1)
        del primals_11
    return buf10, primals_1, primals_3, primals_5, primals_7, primals_8, primals_9, primals_12, primals_14, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9


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
        primals_1 = self.expand_conv[0].weight
        primals_2 = self.expand_conv[1].weight
        primals_3 = self.expand_conv[1].running_mean
        primals_4 = self.expand_conv[1].running_var
        primals_5 = self.depthwise_conv[0].weight
        primals_6 = self.depthwise_conv[1].weight
        primals_7 = self.depthwise_conv[1].running_mean
        primals_8 = self.depthwise_conv[1].running_var
        primals_9 = self.project_conv[0].weight
        primals_10 = self.project_conv[0].bias
        primals_12 = self.project_conv[1].weight
        primals_13 = self.project_conv[1].running_mean
        primals_14 = self.project_conv[1].running_var
        primals_11 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9,
            primals_10, primals_11, primals_12, primals_13, primals_14])
        return output[0]
