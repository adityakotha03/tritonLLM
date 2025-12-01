import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 131072 % 512
    x1 = xindex // 65536 % 512
    x0 = xindex % 131072
    y0 = xindex // 536870912
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 262144 * x1 + 131072 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 % 64 + 384 * y0 // 64), xmask,
        eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + y0, xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + y0, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (1 + y0 % 64 + 384 * y0 // 64), xmask,
        eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (1 + y0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr3 + (1 + y0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (2 + y0 % 64 + 384 * y0 // 64), xmask,
        eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (2 + y0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr3 + (2 + y0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr1 + (3 + y0 % 64 + 384 * y0 // 64), xmask,
        eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr2 + (3 + y0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr3 + (3 + y0), xmask, eviction_policy='evict_last')
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp5 = tmp4 + tmp10
    tmp6 = 2.0
    tmp7 = tmp1 * tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tmp8 + tmp5
    tmp11 = tmp9 + tmp12
    tmp12 = tmp1 * tmp2
    tmp15 = tmp14 * tmp2
    tmp16 = tmp13 * tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = tmp14 * tmp6
    tmp21 = tmp20 * tmp19
    tmp22 = tmp18 + tmp21
    tmp24 = tmp14 * tmp2
    tmp25 = tmp13 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp22 + tmp27
    tmp30 = tmp23 * tmp2
    tmp31 = tmp13 * tmp30
    tmp33 = tmp31 + tmp35
    tmp34 = tmp23 * tmp6
    tmp36 = tmp29 * tmp34
    tmp37 = tmp33 + tmp36
    tmp39 = tmp23 * tmp2
    tmp40 = tmp13 * tmp39
    tmp41 = tmp40 + tmp38
    tmp42 = tmp37 + tmp41
    tmp43 = tmp32 * tmp2
    tmp44 = tmp13 * tmp43
    tmp45 = tmp44 + tmp38
    tmp46 = tmp32 * tmp6
    tmp47 = tmp29 * tmp46
    tmp48 = tmp45 + tmp47
    tmp49 = tmp32 * tmp2
    tmp50 = tmp13 * tmp49
    tmp51 = tmp50 + tmp38
    tmp52 = tmp48 + tmp51
    tmp53 = tmp42 + tmp52
    tl.store(out_ptr0 + x4, tmp53, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_4, (128,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf0, (16, 64, 512, 512), (20971520, 327680, 64,
            1))
        buf1 = extern_kernels.convolution(primals_3, buf0, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (16, 128, 512, 512), (33554432, 262144, 512,
            1))
        buf2 = empty_strided_cuda((16, 128, 512, 512), (33554432, 262144,
            512, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(536870912)](primals_1,
            primals_2, primals_3, buf0, buf2, 536870912, XBLOCK=1024,
            num_warps=4, num_stages=1)
        del primals_1
        del primals_2
        del primals_3
        del buf0
    return buf2, primals_4


class ModelNew(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, input_0):
        primals_1 = self.depthwise.weight
        primals_2 = self.depthwise.bias
        primals_3 = self.pointwise.weight
        primals_4 = self.pointwise.bias
        output = call([primals_1, primals_2, primals_3, primals_4, input_0])
        return output[0]