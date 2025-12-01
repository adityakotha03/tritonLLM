import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
tl_math = torch._C._dynamo.guards._import_tls_lib
triton_helpers math = triton_helpers
triton_poi_fused_div_gelu_0[grid(8388608)] = triton_per_fused_div_gelu_0


@triton.jit
def triton_per_fused_div_gelu_0(in_out_ptr0, in_ptr1, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 / tmp2
    tmp4 = 0.5
    tmp5 = tmp3 * tmp4
    tmp6 = 1.4142135623730951
    tmp7 = tmp3 / tmp6
    tmp8 = libdevice.erf(tmp7)
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = tmp5 * tmp10
    tl.store(in_out_ptr0 + x0, tmp11, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8388608, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.mm(primals_3, primals_1, out=buf0)
        del primals_1
        del primals_3
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_per_fused_div_gelu_0[grid(8388608)](buf1, primals_2, 8388608,
            XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
    return buf1, buf1


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, divides by a scalar, and applies GELU activation.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, input_0):
        primals_1 = self.linear.weight
        primals_2 = self.linear.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]