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
def triton_poi_fused__logsumexp_clamp_mul_neg_0(in_out_ptr0, in_ptr0,
    in_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp6 = tl.sum(tmp4, 0)[:, None]
    tmp7 = tl.full([1], 0, tl.int64)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp9 = -10.0
    tmp10 = libdevice.log1p(tl_math.exp(tmp8 - tmp9))
    tmp11 = tmp0 * tmp3
    tmp12 = tmp11 * tmp10
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x0, tmp12, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (1024, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192, 8192), (8192, 1))
    assert_size_stride(primals_3, (8192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.mm(primals_1, reinterpret_tensor(primals_2, (8192, 
            8192), (1, 8192), 0), out=buf0)
        del primals_2
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused__logsumexp_clamp_mul_neg_0[grid(1024)](buf1,
            primals_3, primals_1, 1024, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
    return buf1, primals_3


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, scales the result, adds a residual connection, clamps the output,
    applies LogSumExp, and finally applies the Mish activation function.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, input_0):
        primals_2 = self.matmul.weight
        primals_1 = self.matmul.bias
        primals_3 = self.scale_factor
        output = call([primals_1, primals_2, primals_3])
        return output[0]
