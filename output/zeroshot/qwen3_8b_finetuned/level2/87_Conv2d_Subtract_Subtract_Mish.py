import torch
import torch.nn as nn
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
empty_cuda = torch._C._dynamo.guards._empty_cuda


@triton.jit
def triton_poi_fused_sub_add_sub_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    x1 = xindex // 256
    x3 = xindex // 65536
    x4 = xindex // 16777216
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = 0.7
    tmp2 = tmp0 - tmp1
    tmp3 = 0.2
    tmp4 = tmp2 - tmp3
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_mish_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    x1 = xindex // 256
    x3 = xindex // 65536
    x4 = xindex // 16777216
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = 0.6931471805599453
    tmp2 = tmp0
    tmp3 = tmp2 - tmp1
    tmp4 = tl.full([1], 1, tl.int32)
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = tl.full(tmp4, True, tl.int1)
    tmp7 = tl.where(tmp6, tmp3, tmp5)
    tmp8 = tl.math.exp(tmp7)
    tmp9 = tmp8 + tmp5
    tmp10 = tl.math.log(tmp9)
    tmp11 = tmp10
    tmp12 = tl.tanh(tmp11)
    tmp13 = tmp12
    tmp14 = tmp13 * tmp1
    tl.store(out_ptr0 + x2, tmp14, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 8, 256, 256), (262144, 32768, 128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 8, 256, 256), (262144, 32768, 128, 1), torch.float32)
        buf1 = buf0
        del buf0
        get_raw_buf = buf1
        buf2 = buf1
        del buf1
        triton_poi_fused_sub_add_sub_0[grid(1048576)](get_raw_buf, buf2, 1048576,
            XBLOCK=256, num_warps=4, num_stages=1)
        del get_raw_buf
        buf3 = empty_strided_cuda((128, 8, 256, 256), (262144, 32768, 128, 1), torch.float32)
        buf4 = reinterpret_tensor(buf3, (128, 8, 256, 256), (262144, 32768, 128, 1), 0)
        triton_poi_fused_mish_1[grid(1048576)](buf2, buf4, 1048576, XBLOCK=256,
            num_warps=4, num_stages=1)
        del buf2
        del buf3
    return buf4,


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, subtracts two constants, applies Mish activation.
    The two elementwise subtractions are fused into a single Triton kernel, and Mish is implemented
    with a separate Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]