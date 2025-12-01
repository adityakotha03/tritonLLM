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
def triton_poi_fused_add_min_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = triton_helpers.minimum(tmp0, tmp2)
    tmp4 = tmp3 - tmp2
    tl.store(out_ptr0 + x0, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 16384), (16384, 1))
    assert_size_stride(primals_2, (16384, 16384), (16384, 1))
    assert_size_stride(primals_3, (1,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        extern_kernels.mm(primals_1, reinterpret_tensor(primals_2, (16384, 
            16384), (1, 16384), 0), out=buf0)
        del primals_2
        buf1 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_min_0[grid(2097152)](buf0, primals_3, buf1, 
            2097152, XBLOCK=1024, num_warps=4, num_stages=1)
        del buf0
        del primals_3
    return buf1, primals_1


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies minimum, and subtracts a constant.
    """
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))

    def forward(self, input_0):
        primals_2 = self.linear.weight
        primals_3 = self.linear.bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]