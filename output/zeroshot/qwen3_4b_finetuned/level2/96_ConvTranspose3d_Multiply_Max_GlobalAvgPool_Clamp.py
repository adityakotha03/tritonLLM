import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_clamp_max_0(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tl.store(in_out_ptr0 + x2, tmp0, xmask)
    tl.store(in_out_ptr0 + (x0 + 1024), tmp1, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8) = args
    args.clear()
    assert_size_stride(primals_1, (128, 3, 16, 32, 32), (49152, 16384, 3276,
        1024, 1))
    assert_size_stride(primals_2, (16, 3, 3, 3, 3), (27, 9, 3, 1, 1))
    assert_size_stride(primals_3, (16,), (1,))
    assert_size_stride(primals_4, (2,), (1,))
    assert_size_stride(primals_5, (2, 2, 2), (4, 2, 1))
    assert_size_stride(primals_6, (1,), (1,))
    assert_size_stride(primals_7, (2,), (1,))
    assert_size_stride(primals_8, (1, 1, 1), (1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(2, 
            2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 16, 21, 34, 34), (1166256, 729, 3496,
            102, 1))
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((128, 16, 21, 34, 34), (1166256, 729, 3496,
            102, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clamp_max_0[grid(1024)](buf1, 1024, XBLOCK=128,
            num_warps=4, num_stages=1)
        del buf1
        buf3 = empty_strided_cuda((128, 16, 2, 2, 2), (102, 6, 3, 1, 1), torch
            .float32)
        extern_kernels.max_pool3d(reinterpret_tensor(buf2, (128, 16, 21, 34,
            34), (1166256, 729, 3496, 102, 1), 0), reinterpret_tensor(
            primals_5, (2, 2, 2), (4, 2, 1), 0), stride=(2, 2, 2), padding=(
            0, 0, 0), dilation=(1, 1, 1), ceilling_mode=True, kernel_size=(
            2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), stride_out=(2, 2,
            2), output_padding=(0, 0, 0), ceil_mode=True, div_ceiling=True)
        del buf2
        del primals_5
        buf4 = extern_kernels.avg_pool3d(reinterpret_tensor(buf3, (128, 16, 2,
            2, 2), (102, 6, 3, 1, 1), 0), reinterpret_tensor(primals_8, (1, 1,
            1), (1, 1, 1), 0), stride=(1, 1, 1), padding=(0, 0, 0), dilation=
            (1, 1, 1), output_padding=(0, 0, 0), ceil_mode=True, kernel_size=(
            1, 1, 1))
        assert_size_stride(buf4, (128, 16, 1, 1, 1), (16, 1, 1, 1, 1))
        buf5 = reinterpret_tensor(buf4, (128, 16), (16, 1), 0)
        del buf4
        buf6 = buf5
        del buf5
        triton_poi_fused_clamp_min_0 = triton_poi_fused_clamp_max_0
        triton_poi_fused_clamp_min_0[grid(128)](buf6, 128, XBLOCK=128,
            num_warps=4, num_stages=1)
        del buf6
    return reinterpret_tensor(buf3, (128, 16, 1, 1, 1), (16, 1, 1, 1, 1), 0
        ), primals_1, primals_2, primals_3, primals_4, primals_6, primals_8


class ModelNew(nn.Module):
    """
    Model that performs a transposed 3D convolution, multiplies by a scalar, applies max pooling, 
    global average pooling, and clamps the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0
        self.clamp_max = 1

    def forward(self, input_0):
        primals_2 = self.conv_transpose.weight
        primals_3 = self.conv_transpose.bias
        primals_5 = self.maxpool.kernel_size
        primals_6 = self.global_avg_pool.output_size
        primals_8 = self.global_avg_pool.output_padding
        primals_4 = self.conv_transpose.stride
        primals_1 = input_0
        primals_7 = self.conv_transpose.dilation
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8])
        return output[0]
