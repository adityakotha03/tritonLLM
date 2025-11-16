import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_tanh_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 48
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = triton_helpers.tanh(tmp2)
    tmp5 = tmp3 * tmp4
    tl.store(in_out_ptr0 + x2, tmp5, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7) = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (128, 3, 16, 64, 64), (393216, 131072, 
        2464, 384, 6))
    assert_size_stride(primals_3, (16,), (1,))
    assert_size_stride(primals_4, (16,), (1,))
    assert_size_stride(primals_5, (16,), (1,))
    assert_size_stride(primals_6, (16,), (1,))
    assert_size_stride(primals_7, (16,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 
            1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 16, 14, 62, 62), (764416, 48360, 3460,
            56, 1))
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((128, 16, 14, 62, 62), (764416, 48360, 
            3460, 56, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_tanh_0[grid(36864)](buf1, primals_3, 36864, XBLOCK=
            256, num_warps=4, num_stages=1)
        del primals_3
    return buf2, primals_1, primals_2, primals_4, primals_5, primals_6, primals_7


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, scales the output, applies tanh, multiplies by a scaling factor, and applies sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_3 = self.conv.bias
        primals_4 = self.bias
        primals_5 = self.scaling_factor
        primals_6 = torch.ops.tanh
        primals_7 = torch.ops._softmax
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5,
            primals_6, primals_7])
        return output[0]
