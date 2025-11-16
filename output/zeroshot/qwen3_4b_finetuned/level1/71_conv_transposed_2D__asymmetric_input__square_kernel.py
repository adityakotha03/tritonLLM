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
def triton_poi_fused__unsafe_index_addmm_convolution_0(in_out_ptr0, in_ptr0,
    in_ptr1, in_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 122880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 10240 % 32
    x0 = xindex % 10240
    x2 = xindex // 10240
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (1024 + x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x2, xmask, eviction_policy='evict_last')
    tmp4 = tmp1 + tmp3
    tmp5 = tmp2 + tmp4
    tmp6 = tmp0 + tmp5
    tl.store(in_out_ptr0 + x3, tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (32,), (1,))
    assert_size_stride(primals_3, (8, 32, 512, 1024), (16777216, 524288,
        1024, 1))
    assert_size_stride(primals_4, (32, 32, 3, 3), (27, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(reinterpret_tensor(primals_3, (8,
            32, 512, 1024), (16777216, 524288, 1024, 1), 0), reinterpret_tensor(
            primals_1, (32, 3, 3, 3), (27, 9, 3, 1), 0), stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=True,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 32, 513, 1025), (16777216, 524288, 1025,
            1))
        buf1 = empty_strided_cuda((8, 32, 513, 1025), (16777216, 524288, 1025,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__unsafe_index_addmm_convolution_0[grid(122880)](buf1,
            primals_2, buf0, primals_4, 122880, XBLOCK=256, num_warps=4,
            num_stages=1)
        del primals_2
        del primals_4
    return reinterpret_tensor(buf1, (8, 32, 513, 1025), (16777216, 524288,
        1025, 1), 0), primals_1, primals_3, buf0


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        
    def forward(self, input_0):
        primals_1 = self.conv_transpose2d.weight
        primals_2 = self.conv_transpose2d.bias
        primals_4 = self.conv_transpose2d.weight_1
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
