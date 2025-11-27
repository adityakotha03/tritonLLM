import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_relu_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 106496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tl.store(out_ptr0 + x0, tmp3, xmask)
    tl.store(out_ptr0 + (106496 + x0), tmp5, xmask)


@triton.jit
def triton_poi_fused_add_leaky_relu_1(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 106496
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
    tmp6 = 0.01
    tmp7 = tmp3 * tmp6
    tmp8 = tl.where(tmp5, tmp3, tmp7)
    tl.store(out_ptr0 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_gelu_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 106496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 0.7071067811865476
    tmp7 = tmp3 * tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = 0.01
    tmp10 = tmp8 * tmp9
    tmp11 = 1.0
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 * tmp3
    tmp14 = tl.where(tmp5, tmp4, tmp13)
    tl.store(out_ptr0 + x0, tmp14, xmask)


@triton.jit
def triton_poi_fused_add_sigmoid_3(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 106496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tl.store(out_ptr0 + x0, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (32, 8, 3, 3, 3), (216, 27, 9, 3, 1))
    assert_size_stride(primals_2, (32,), (1,))
    assert_size_stride(primals_3, (64, 8, 32, 32, 32), (16384, 2048, 64, 2, 
        1))
    assert_size_stride(primals_4, (32,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 32, 30, 62, 62), (1191232, 37632, 393,
            6, 1), torch.float32)
        get_ptr0 = extern_kernels.convolution
        buf1 = empty_strided_cuda((64, 32, 30, 62, 62), (1191232, 37632, 393,
            6, 1), torch.float32)
        triton_poi_fused_add_relu_0[grid(106496)](primals_3, primals_1, buf0,
            106496, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        buf2 = get_ptr0(buf1, primals_3, buf0, [1191232, 2048, 393, 6, 1], 0)
        del primals_3
        buf3 = buf1
        del buf1
        triton_poi_fused_add_leaky_relu_1[grid(106496)](buf0, primals_2, buf2
            , 106496, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf4 = empty_strided_cuda((64, 32, 30, 62, 62), (1191232, 37632, 393,
            6, 1), torch.float32)
        triton_poi_fused_add_gelu_2[grid(106496)](buf2, primals_4, buf4, 
            106496, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_4
        buf5 = empty_strided_cuda((64, 32, 30, 62, 62), (1191232, 37632, 393,
            6, 1), torch.float32)
        triton_poi_fused_add_sigmoid_3[grid(106496)](buf4, primals_4, buf5,
            106496, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_4
    return buf5, primals_4, buf0, buf2, buf4, primals_4


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, and bias in sequence.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_4 = self.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
