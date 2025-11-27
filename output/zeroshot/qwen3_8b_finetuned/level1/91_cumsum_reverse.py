import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def cumsum_flip_kernel(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = x2 - 1
    x3 = xindex
    x1 = x3
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp2 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(out_ptr0 + x1, xmask)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp0 + tmp4
    tl.store(out_ptr0 + x1, tmp5, xmask)


def triton_cumsum_flip(in_out_ptr0, in_ptr0, primals_3, primals_4):
    arg0_1, arg1_1 = primals_3
    args.clear()
    assert_size_stride(arg0_1, (32768,), (1,))
    assert_size_stride(arg1_1, (32768,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((), (), torch.int64)
        buf1 = empty_strided_cuda((), (), torch.int64)
        buf2 = empty_strided_cuda((), (), torch.int64)
        buf3 = empty_strided_cuda((), (), torch.int64)
        buf4 = empty_strided_cuda((), (), torch.int64)
        del primals_3
        buf5 = buf0
        del buf0
        buf6 = buf1
        del buf1
        buf7 = buf2
        del buf2
        buf8 = buf3
        del buf3
        buf9 = buf4
        del buf4
        get_raw_stream(0)
        cumsum_flip_kernel[grid(32768)](arg1_1, buf5, 32768, 32768, XBLOCK=128)
        del arg1_1
        del buf5
        del buf6
        del buf7
        del buf8
        del buf9
    return reinterpret_tensor(buf0, (1, 32768), (32768, 1), 0), reinterpret_tensor(
        buf1, (1, 32768), (32768, 1), 0
    ), reinterpret_tensor(buf2, (1, 32768), (32768, 1), 0), reinterpret_tensor(
        buf3, (1, 32768), (32768, 1), 0
    ), reinterpret_tensor(buf4, (1, 32768), (32768, 1), 0
    ), reinterpret_tensor(buf6, (1, 32768), (32768, 1), 0), reinterpret_tensor(
        buf7, (1, 32768), (32768, 1), 0
    ), reinterpret_tensor(buf8, (1, 32768), (32768, 1), 0), reinterpret_tensor(
        buf9, (1, 32768), (32768, 1), 0
    )


class ModelNew(nn.Module):
    """
    A model that performs a reverse cumulative sum operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, input_0):
        primals_3 = (input_0, input_0)
        primals_4 = primals_3
        output = triton_cumsum_flip(primals_3, primals_4)
        return output[0]