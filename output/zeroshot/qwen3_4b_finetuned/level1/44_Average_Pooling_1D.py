import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_per_fused_avg_pool1d_with_indices_0(in_ptr0, out_ptr0, out_ptr1,
    xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    tl.full([1], xoffset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + r0, None)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.sum(tmp3, 0)[:, None]
    tl.store(out_ptr0 + r0, tmp5, None)
    tl.store(out_ptr1 + r0, tmp2, None)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (64, 128, 65536), (8388608, 65536, 1))
    assert_size_stride(primals_2, (8,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 128, 65536), (8388608, 65536, 1),
            torch.float32)
        buf1 = empty_strided_cuda((64, 128, 65532), (8388608, 65536, 1),
            torch.float32)
        buf2 = empty_strided_cuda((64, 128, 65532), (8388608, 65536, 1),
            torch.float32)
        get_raw_stream(0)
        triton_per_fused_avg_pool1d_with_indices_0[grid(256)](primals_1,
            buf1, buf2, 256, 256, num_warps=2, num_stages=1)
    return buf2, primals_1, buf1, primals_2


class ModelNew(nn.Module):
    """
    Simple model that performs 1D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride,
            padding=padding)

    def forward(self, input_0):
        primals_2 = self.avg_pool.weight
        primals_1 = input_0
        output = call([primals_1, primals_2])
        return output[0]
