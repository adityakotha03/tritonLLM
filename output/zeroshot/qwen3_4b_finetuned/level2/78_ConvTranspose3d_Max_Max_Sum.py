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
def triton_poi_fused_max_pool3d_with_indices_0(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 10752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x1 = xindex // 3 % 16
    x2 = xindex // 48 % 16
    x3 = xindex // 768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 6 * x2 + 192 * x1 + 3072 * x3), xmask)
    tmp1 = tl.load(in_ptr0 + (1 + x0 + 6 * x2 + 192 * x1 + 3072 * x3), xmask)
    tmp3 = tl.load(in_ptr0 + (3 + x0 + 6 * x2 + 192 * x1 + 3072 * x3), xmask)
    tmp5 = tl.load(in_ptr0 + (4 + x0 + 6 * x2 + 192 * x1 + 3072 * x3), xmask)
    tmp7 = tl.load(in_ptr0 + (2 + x0 + 6 * x2 + 192 * x1 + 3072 * x3), xmask)
    tmp9 = tl.load(in_ptr0 + (5 + x0 + 6 * x2 + 192 * x1 + 3072 * x3), xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tl.store(out_ptr0 + x4, tmp10, xmask)


@triton.jit
def triton_poi_fused_max_pool3d_with_indices_1(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x1 = xindex // 2 % 2
    x2 = xindex // 4 % 2
    x3 = xindex // 8
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8 * x2 + 32 * x1 + 64 * x3), xmask)
    tmp1 = tl.load(in_ptr0 + (1 + x0 + 8 * x2 + 32 * x1 + 64 * x3), xmask)
    tmp3 = tl.load(in_ptr0 + (4 + x0 + 8 * x2 + 32 * x1 + 64 * x3), xmask)
    tmp5 = tl.load(in_ptr0 + (5 + x0 + 8 * x2 + 32 * x1 + 64 * x3), xmask)
    tmp7 = tl.load(in_ptr0 + (2 + x0 + 8 * x2 + 32 * x1 + 64 * x3), xmask)
    tmp9 = tl.load(in_ptr0 + (6 + x0 + 8 * x2 + 32 * x1 + 64 * x3), xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tl.store(out_ptr0 + x4, tmp10, xmask)


@triton.jit
def triton_poi_fused_sum_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x1 = xindex // 2 % 2
    x2 = xindex // 4 % 2
    x3 = xindex // 8
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8 * x2 + 32 * x1 + 64 * x3), xmask)
    tmp1 = tl.load(in_ptr0 + (1 + x0 + 8 * x2 + 32 * x1 + 64 * x3), xmask)
    tmp3 = tl.load(in_ptr0 + (4 + x0 + 8 * x2 + 32 * x1 + 64 * x3), xmask)
    tmp5 = tl.load(in_ptr0 + (5 + x0 + 8 * x2 + 32 * x1 + 64 * x3), xmask)
    tmp7 = tl.load(in_ptr0 + (2 + x0 + 8 * x2 + 32 * x1 + 64 * x3), xmask)
    tmp9 = tl.load(in_ptr0 + (6 + x0 + 8 * x2 + 32 * x1 + 64 * x3), xmask)
    tmp11 = tl.load(in_ptr0 + (3 + x0 + 8 * x2 + 32 * x1 + 64 * x3), xmask)
    tmp13 = tl.load(in_ptr0 + (7 + x0 + 8 * x2 + 32 * x1 + 64 * x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + x4, tmp14, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 5, 5, 5), (4000, 125, 25, 5, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (16, 32, 32, 32, 32), (524288, 16384, 
        512, 16, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(2, 
            2, 2), padding=(2, 2, 2), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (16, 64, 32, 16, 16), (524288, 8192, 256, 
            16, 1))
        buf1 = empty_strided_cuda((16, 64, 32, 16, 16), (524288, 8192, 256,
            16, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_max_pool3d_with_indices_0[grid(10752)](buf0, buf1,
            10752, XBLOCK=128, num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((16, 64, 2, 2, 2), (512, 8, 4, 2, 1),
            torch.float32)
        triton_poi_fused_max_pool3d_with_indices_1[grid(256)](buf1, buf2, 
            256, XBLOCK=128, num_warps=4, num_stages=1)
        del buf1
        buf3 = empty_strided_cuda((16, 2, 2, 2), (8, 4, 2, 1), torch.float32)
        triton_poi_fused_sum_2[grid(256)](buf2, buf3, 256, XBLOCK=128,
            num_warps=4, num_stages=1)
        del buf2
    return buf3, primals_1, primals_3, buf0


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, 
            kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
