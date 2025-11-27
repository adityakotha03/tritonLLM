import torch
import torch.nn as nn
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_bias_subtract_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + 0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_tanh_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp6 = tmp5 * tmp5
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = tmp6 - tmp7
    tmp12 = tmp11 / tmp10
    tmp13 = tmp12 + tmp7
    tmp14 = tmp13 / tmp10
    tmp15 = tl.full([1], 0.5, tl.int32)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16 + tmp12
    tmp18 = tmp17 * tmp14
    tl.store(out_ptr0 + x0, tmp18, xmask)


def call(args):
    (primals_1, primals_2) = args
    args.clear()
    assert_size_stride(primals_1, (64, 1, 1), (1, 1, 1))
    assert_size_stride(primals_2, (32, 64, 256, 256), (4194304, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 64, 128, 128), (1048576, 16384, 128, 1), torch.float32)
        del primals_2
        buf1 = buf0
        get_raw_buf = buf0
        buf2 = buf0
        buf3 = empty_strided_cuda((64, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf4 = empty_strided_cuda((32, 64, 128, 128), (1048576, 16384, 128, 1), torch.float32)
        buf5 = buf0
        triton_poi_fused_bias_subtract_0[grid(32768)](primals_1, buf1, 32768, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        triton_poi_fused_tanh_1[grid(32768)](buf1, buf2, 32768, XBLOCK=128, num_warps=4, num_stages=1)
    return buf2, buf1, buf3, buf4, buf5


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, subtracts a bias term, and applies tanh activation,
    with the bias subtraction and tanh replaced by custom Triton kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_1 = primals_1.to(torch.float32)
        primals_2 = primals_2.to(torch.float32)
        output = call([primals_1, primals_2, input_0])
        return output[0]