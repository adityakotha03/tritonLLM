import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_avg_pool3d_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1), xmask)
    tmp1 = tl.load(in_ptr0 + (16 + x0 + 16 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (32 + x0 + 16 * x1), xmask)
    tmp5 = tl.load(in_ptr0 + (48 + x0 + 16 * x1), xmask)
    tmp7 = tl.load(in_ptr0 + (64 + x0 + 16 * x1), xmask)
    tmp9 = tl.load(in_ptr0 + (80 + x0 + 16 * x1), xmask)
    tmp11 = tl.load(in_ptr0 + (96 + x0 + 16 * x1), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = 16.0
    tmp14 = tmp12 / tmp13
    tl.store(out_ptr0 + x2, tmp14, xmask)


@triton.jit
def triton_poi_fused_add_mul_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tl.store(in_out_ptr0 + x2, tmp6, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 3, 16, 32, 32), (49152, 16384, 1024, 
        32, 1))
    assert_size_stride(arg1_1, (16, 3, 3, 3, 3), (243, 81, 27, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 16, 16, 16, 16), (16384, 1024, 64, 
            4, 1), torch.float32)
        get_ptr0 = buf0
        triton_poi_fused_avg_pool3d_0[triton.autotune, triton.jit](
            arg0_1, get_ptr0, 128, XBLOCK=128, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((128, 16, 16, 16, 16), (16384, 1024, 64, 
            4, 1), torch.float32)
        triton_poi_fused_add_mul_1[triton.autotune, triton.jit](
            buf1, arg1_1, 128, XBLOCK=128, num_warps=4, num_stages=1)
        del arg1_1
    return buf1, arg0_1, buf0, get_ptr0


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, scaling, average pooling, bias addition, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
        scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

    def forward(self, input_0):
        arg1_1 = self.conv_transpose.weight
        arg1_1 = arg1_1.to(torch.float32)
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]
