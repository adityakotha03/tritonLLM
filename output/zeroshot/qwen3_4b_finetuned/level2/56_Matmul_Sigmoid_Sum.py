import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_per_fused_sigmoid_sum_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 128
    rnumel = 32768
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 32768 * x0), rmask & xmask, other=0.0)
    tmp1 = libdevice.sigmoid(tmp0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp2, 1))
    tl.store(out_ptr0 + x0, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (32768, 32768), (32768, 1))
    assert_size_stride(primals_2, (32768,), (1,))
    assert_size_stride(primals_3, (128, 32768), (32768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768, 32768), (32768, 1), torch.float32)
        extern_kernels.addmm(primals_2, reinterpret_tensor(primals_3, (32768,
            32768), (1, 32768), 0), reinterpret_tensor(primals_1, (32768,
            32768), (1, 32768), 0), alpha=1, beta=1, out=buf0)
        del primals_1
        del primals_2
        buf1 = empty_strided_cuda((128, 1), (1, 128), torch.float32)
        get_raw_stream(0)
        triton_per_fused_sigmoid_sum_0[grid(128)](buf0, buf1, 128, 32768,
            XBLOCK=1, num_warps=2, num_stages=1)
        del buf0
    return buf1, reinterpret_tensor(primals_3, (32768, 32768), (1, 32768), 0
        ), primals_2, buf1


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies sigmoid, and sums the result.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, input_0):
        primals_1 = self.linear.weight
        primals_2 = self.linear.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
