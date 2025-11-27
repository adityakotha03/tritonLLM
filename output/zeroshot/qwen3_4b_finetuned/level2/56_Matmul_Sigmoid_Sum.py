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
def triton_poi_fused_sum_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 32768 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tl.load(in_ptr0 + (1 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp1 + tmp3
    tmp5 = tl.load(in_ptr0 + (2 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp4 + tmp6
    tl.store(out_ptr0 + x0, tmp7, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (32768, 32768), (32768, 1))
    assert_size_stride(primals_2, (32768,), (1,))
    assert_size_stride(primals_3, (128, 32768), (32768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        extern_kernels.addmm(primals_2, primals_3, reinterpret_tensor(
            primals_1, (32768, 32768), (1, 32768), 0), alpha=1, beta=1,
            out=buf0)
        del primals_2
        buf1 = empty_strided_cuda((128, 1), (1, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_sum_0[grid(128)](buf0, buf1, 128, XBLOCK=128,
            num_warps=4, num_stages=1)
        del buf0
    return buf1, primals_1, primals_3


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
