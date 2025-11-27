import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_leaky_relu_max_pool3d_mul_0(in_ptr0,
    in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1024 % 128
    x0 = xindex % 1024
    x2 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 0.2
    tmp5 = tmp2 > tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp2, tmp6)
    tmp8 = tmp7 * tmp3
    tmp9 = 0.2
    tmp10 = tmp8 > tmp9
    tmp11 = tl.where(tmp10, tmp8, tmp6)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp14 = tl.where(xmask, tmp12, float('-inf'))
    tmp15 = tl.load(out_ptr0 + (x3 + 16384), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(out_ptr0 + (16384 + x3), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(out_ptr0 + (32768 + x3), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(out_ptr1 + (x3 + 16384), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(out_ptr1 + (16384 + x3), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(out_ptr1 + (32768 + x3), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.where(xmask, tmp15, tmp14)
    tmp22 = tl.where(xmask, tmp16, tmp21)
    tmp23 = tl.where(xmask, tmp17, tmp22)
    tmp24 = tl.where(xmask, tmp18, tmp23)
    tmp25 = tl.where(xmask, tmp19, tmp24)
    tmp26 = tl.where(xmask, tmp20, tmp25)
    tl.store(out_ptr0 + x3, tmp26, xmask)
    tl.store(out_ptr1 + x3, tmp13, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (32, 16, 3, 3, 3), (432, 27, 9, 3, 1))
    assert_size_stride(primals_2, (16, 16, 16, 32, 32), (16384, 1024, 64, 4, 
        1))
    assert_size_stride(primals_3, (32,), (1,))
    assert_size_stride(primals_4, (32, 1, 1, 1), (1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 32, 17, 33, 33), (177856, 5468, 327, 
            10, 1), torch.float32)
        buf1 = empty_strided_cuda((16, 32, 17, 33, 33), (177856, 5468, 327,
            10, 1), torch.float32)
        get_ptr0 = buf0
        get_ptr1 = buf1
        triton_poi_fused_convolution_leaky_relu_max_pool3d_mul_0[ext_out_ptr0
            ](primals_2, primals_1, primals_3, get_ptr0, get_ptr1, 36864,
            XBLOCK=256, num_warps=4, num_stages=1)
        del primals_1
        del primals_3
    return get_ptr0, primals_2, primals_4, buf1


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies LeakyReLU, multiplies by a learnable parameter, 
    applies LeakyReLU again, and performs a max pooling operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
        output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_3 = self.conv_transpose.bias
        primals_4 = self.multiplier
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
