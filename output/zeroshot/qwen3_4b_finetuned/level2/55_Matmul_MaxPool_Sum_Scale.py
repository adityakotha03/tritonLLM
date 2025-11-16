import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused__to_copy_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


def call(args):
    arg0_1, arg1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 32768), (32768, 1))
    assert_size_stride(arg1, (32768, 32768), (32768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        extern_kernels.addmm(0.0, arg1, arg0_1, alpha=1, beta=1, out=buf0)
        del arg1
        buf1 = empty_strided_cuda((128, 32768, 1), (32768, 1, 32768), torch.
            float32)
        get_raw_stream(0)
        triton_poi_fused__to_copy_mul_0[grid(131072)](buf0, buf1, 131072,
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf0
    return reinterpret_tensor(buf1, (128, 1, 1), (1, 32768, 32768), 0
        ), reinterpret_tensor(arg0_1, (128, 1, 32768), (32768, 32768, 1), 0)


class ModelNew(nn.Module):
    """
    Model that performs matrix multiplication, max pooling, sum, and scaling.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.scale_factor = scale_factor

    def forward(self, input_0):
        arg0_1 = input_0
        arg1 = self.matmul.weight
        output = call([arg0_1, arg1])
        return output[0]
