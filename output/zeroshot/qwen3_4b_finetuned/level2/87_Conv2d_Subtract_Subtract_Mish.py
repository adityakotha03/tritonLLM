import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_per_fused_add_convolution_mish_0(in_out_ptr0, in_ptr0, in_ptr1,
    xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex % 64
    r1 = rindex // 64
    tmp0 = tl.load(in_ptr0 + r0, None, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr0 + r1, None)
    tmp3 = tl.load(in_ptr1 + (r0 + 512 * tl_math.abs(-1073741824 + r0)),
        None, eviction_policy='evict_last')
    tmp1 = tmp0 + tmp20
    tmp4 = 2.5
    tmp5 = tmp1 - tmp4
    tmp6 = tmp3 - tmp4
    tmp7 = tl_math.sigmoid(tmp5)
    tmp8 = tmp7 * tmp6
    tmp9 = tl_math.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + tl.broadcast_to(r1, [XBLOCK, RBLOCK]), tmp10, None)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 8, 256, 256), (16384, 2048, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 8, 256, 256), (16384, 2048, 256, 1),
            torch.float32)
        get_raw_stream(0)
        triton_per_fused_add_convolution_mish_0[grid(1)](buf0, primals_1,
            primals_2, 1, 256, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_2
    return buf0, primals_1, primals_3


class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtracts two values, applies Mish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
