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
def triton_poi_fused_convolution_relu6_0(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 14922240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 14336 % 672
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = 0.5999999999999999
    tmp2 = tmp0 * tmp1
    tmp3 = tl.full([1], 1.0, tl.int32)
    tmp4 = tmp3 + tmp3
    tmp5 = tmp2 - tmp4
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = triton_helpers.minimum(tmp7, tmp3)
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 14922240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 14336 % 672
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_2(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1155840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 14336
    x1 = xindex // 14336
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = 0.5999999999999999
    tmp2 = tmp0 * tmp1
    tmp3 = tl.full([1], 1.0, tl.int32)
    tmp4 = tmp3 + tmp3
    tmp5 = tmp2 - tmp4
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = triton_helpers.minimum(tmp7, tmp3)
    tl.store(out_ptr0 + (x0 + 14336 * x1), tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1155840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 14336
    x1 = xindex // 14336
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_native_batch_norm_4(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 14336 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (14336 + 14336 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (28672 + 14336 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (43008 + 14336 * x0), xmask, eviction_policy=
        'evict_last')
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
    tmp23 = libdevice.rsqrt(tmp22)
    tl.store(out_ptr0 + x0, tmp8, xmask)
    tl.store(out_ptr1 + x0, tmp23, xmask)
    tl.store(out_ptr2 + x0, tmp22, xmask)


@triton.jit
def triton_poi_fused_native_batch_norm_backward_5(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 14336 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + 14336 * x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask)
    tmp6 = tl.load(in_ptr3 + x0, xmask)
    tmp9 = tl.load(in_ptr4 + x0, xmask)
    tmp15 = tl.load(in_ptr0 + (14336 + 14336 * x0), xmask,
        eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (14336 + 14336 * x0), xmask,
        eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (1 + x0), xmask)
    tmp22 = tl.load(in_ptr3 + (1 + x0), xmask)
    tmp25 = tl.load(in_ptr4 + (1 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4 * tmp6
    tmp7 = tmp5 * tmp9
    tmp8 = tmp7 + tmp1
    tmp10 = tmp8 * tmp6
    tmp11 = tmp10 + tmp1
    tmp12 = tmp8 - tmp11
    tmp13 = tl.load(in_ptr0 + (28672 + 14336 * x0), xmask,
        eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (28672 + 14336 * x0), xmask,
        eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (2 + x0), xmask)
    tmp20 = tl.load(in_ptr3 + (2 + x0), xmask)
    tmp23 = tl.load(in_ptr4 + (2 + x0), xmask)
    tmp18 = tmp13 + tmp14
    tmp21 = tmp18 * tmp17
    tmp24 = tmp21 * tmp20
    tmp26 = tmp24 * tmp23
    tmp27 = tmp26 + tmp1
    tmp28 = tmp27 * tmp20
    tmp29 = tmp28 + tmp1
    tmp30 = tmp27 - tmp29
    tmp31 = tl.load(in_ptr0 + (43008 + 14336 * x0), xmask,
        eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr1 + (43008 + 14336 * x0), xmask,
        eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr2 + (3 + x0), xmask)
    tmp38 = tl.load(in_ptr3 + (3 + x0), xmask)
    tmp41 = tl.load(in_ptr4 + (3 + x0), xmask)
    tmp33 = tmp31 + tmp32
    tmp34 = tmp33 * tmp35
    tmp36 = tmp34 * tmp38
    tmp37 = tmp36 * tmp41
    tmp39 = tmp37 + tmp1
    tmp40 = tmp39 * tmp38
    tmp42 = tmp40 + tmp1
    tmp43 = tmp39 - tmp42
    tmp44 = tmp12 + tmp30
    tmp45 = tmp44 + tmp43
    tmp46 = tmp8 - tmp45
    tl.store(out_ptr0 + x0, tmp46, xmask)


@triton.jit
def triton_poi_fused_add_native_batch_norm_6(in_out_ptr0, in_ptr0, in_ptr1,
    in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp0 + tmp3
    tmp6 = tmp5 + tmp4
    tl.store(in_out_ptr0 + x0, tmp6, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14) = args
    args.clear()
    assert_size_stride(primals_1, (112, 672, 1, 1), (672, 672, 672, 1))
    assert_size_stride(primals_2, (672,), (1,))
    assert_size_stride(primals_3, (672, 112, 5, 5), (2800, 2800, 560, 112))
    assert_size_stride(primals_4, (672,), (1,))
    assert_size_stride(primals_5, (672, 112, 5, 5), (2800, 2800, 560, 112))
    assert_size_stride(primals_6, (672,), (1,))
    assert_size_stride(primals_7, (192, 672, 1, 1), (672, 672, 672, 1))
    assert_size_stride(primals_8, (192,), (1,))
    assert_size_stride(primals_9, (192, 192, 1, 1), (192, 192, 192, 1))
    assert_size_stride(primals_10, (192,), (1,))
    assert_size_stride(primals_11, (192, 192, 1, 1), (192, 192, 192, 1))
    assert_size_stride(primals_12, (192,), (1,))
    assert_size_stride(primals_13, (10, 112, 224, 224), (250880, 2240, 10,
        1))
    assert_size_stride(primals_14, (192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((10, 672, 224, 224), (3145728, 4864, 14, 
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_relu6_0[grid(14922240)](primals_1,
            buf0, 14922240, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((10, 672, 224, 224), (3145728, 4864, 14, 
            1), torch.float32)
        extern_kernels.convolution(buf0, primals_3, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=672, bias=None)
        del buf0
        buf2 = reinterpret_tensor(buf1, (10, 672, 224, 224), (3145728, 4864,
            14, 1), 0)
        del buf1
        triton_poi_fused_convolution_1[grid(14922240)](buf2, primals_4, 
            14922240, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_4
        buf3 = empty_strided_cuda((10, 672, 224, 224), (3145728, 4864, 14, 
            1), torch.float32)
        extern_kernels.convolution(buf2, primals_5, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=672, bias=None)
        buf4 = empty_strided_cuda((672, 10, 14, 14), (1960, 196, 14, 1),
            torch.float32)
        buf5 = empty_strided_cuda((672, 10), (10, 1), torch.float32)
        buf6 = empty_strided_cuda((672,), (1,), torch.float32)
        triton_poi_fused_native_batch_norm_4[grid(576)](buf3, buf4, buf5,
            buf6, 576, XBLOCK=64, num_warps=1, num_stages=1)
        buf7 = reinterpret_tensor(buf3, (10, 672, 14, 14), (14336, 1, 1, 1), 
            0)
        del buf3
        triton_poi_fused_convolution_relu6_2[grid(1155840)](buf7, buf7, 
            1155840, XBLOCK=1024, num_warps=4, num_stages=1)
        buf8 = reinterpret_tensor(buf7, (10, 672, 14, 14), (14336, 1, 1, 1), 
            0)
        del buf7
        triton_poi_fused_convolution_3[grid(1155840)](buf8, primals_6, 
            1155840, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_6
        buf9 = reinterpret_tensor(buf8, (10, 672, 14, 14), (14336, 1, 1, 1), 
            0)
        del buf8
        triton_poi_fused_native_batch_norm_backward_5[grid(576)](buf9, buf4,
            buf5, buf6, primals_10, buf9, 576, XBLOCK=128, num_warps=4,
            num_stages=1)
        del buf4
        del buf5
        del buf6
        buf10 = reinterpret_tensor(buf9, (10, 672, 14, 14), (14336, 1, 1, 1), 
            0)
        del buf9
        triton_poi_fused_add_native_batch_norm_6[grid(576)](buf10, primals_7,
            primals_8, primals_9, primals_11, 576, XBLOCK=128, num_warps=4,
            num_stages=1)
        buf11 = empty_strided_cuda((10, 192, 14, 14), (3584, 18432, 10, 1),
            torch.float32)
        extern_kernels.convolution(buf10, primals_12, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        del primals_12
        buf12 = reinterpret_tensor(buf11, (10, 192, 14, 14), (3584, 1, 1, 1),
            0)
        del buf11
        triton_poi_fused_native_batch_norm_4[grid(576)](buf12, buf4, buf5,
            buf6, 576, XBLOCK=64, num_warps=1, num_stages=1)
        buf13 = reinterpret_tensor(buf12, (10, 192, 14, 14), (3584, 1, 1, 1),
            0)
        del buf12
        triton_poi_fused_convolution_relu6_2[grid(1155840)](buf13, buf13,
            1155840, XBLOCK=1024, num_warps=4, num_stages=1)
        buf14 = reinterpret_tensor(buf13, (10, 192, 14, 14), (3584, 1, 1, 1),
            0)
        del buf13
        triton_poi_fused_convolution_3[grid(1155840)](buf14, primals_14, 
            1155840, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_14
        buf15 = reinterpret_tensor(buf14, (10, 192, 14, 14), (3584, 1, 1, 1),
            0)
        del buf14
        triton_poi_fused_native_batch_norm_backward_5[grid(576)](buf15, buf4,
            buf5, buf6, primals_13, buf15, 576, XBLOCK=128, num_warps=4,
            num_stages=1)
        del buf4
        del buf5
        del buf6
        buf16 = empty_strided_cuda((10, 192, 14, 14), (3584, 18432, 10, 1),
            torch.float32)
        extern_kernels.convolution(buf15, primals_13, stride=(2, 2),
            padding=(2, 2), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        del primals_13
    return buf16, primals_2, primals_3, primals_5, primals_7, primals_8, (
        primals_9, primals_10), buf1, buf2, buf8, buf10, buf15, primals_11


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
        primals_3 = self.depthwise_conv[0].weight
        primals_4 = self.depthwise_conv[1].weight
        primals_5 = self.depthwise_conv[1].running_mean
        primals_6 = self.depthwise_conv[1].running_var
        primals_7 = self.project_conv[0].weight
        primals_8 = self.project_conv[1].weight
        primals_9 = self.project_conv[1].running_mean
        primals_10 = self.project_conv[1].running_var
        primals_11 = self.project_conv[1].num_batches_tracked
        primals_12 = self.project_conv[1].weight
        primals_13 = input_0
        primals_14 = self.project_conv[1].bias
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9,
            primals_10, primals_11, primals_12, primals_13, primals_14])
        return output[0]
