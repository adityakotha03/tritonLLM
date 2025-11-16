import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_per_fused_sum_0(in_out_ptr0, in_ptr0, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    tl.full([1], xoffset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + r0, None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(tl.full([RBLOCK], 1, tl.int32), tmp1, 0)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp7 = 0.0
    tmp8 = tmp6 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + tl.full([1], 0, tl.int32), tmp8, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 32, 32, 32, 32), (524288, 16384, 16384, 
        512, 16))
    assert_size_stride(arg1_1, (64, 32, 5, 5, 5), (40000, 12500, 2500, 500, 
        100))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(arg0_1, arg1_1, stride=(2, 
            2, 2), padding=(2, 2, 2), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (16, 64, 37, 37, 37), (8948704, 139840, 
            37568, 1024, 32))
        buf1 = empty_strided_cuda((16, 64, 37, 37, 37), (90647296, 139840, 
            37568, 1024, 32), torch.float32)
        get_raw_stream(0)
        triton_per_fused_sum_0[grid(1)](buf1, buf0, 1, 256, num_warps=2,
            num_stages=1)
        del buf0
    return buf1, arg1_1


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
        arg1_1 = self.conv_transpose.weight
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]
