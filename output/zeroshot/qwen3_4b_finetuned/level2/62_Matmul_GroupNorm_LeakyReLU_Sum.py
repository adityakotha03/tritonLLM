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


@triton.jit
def triton_poi_fused_leaky_relu_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8192
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.01
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 - tmp8
    tl.store(in_out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr0 + x2, tmp9, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.mm(primals_3, primals_1)
        assert_size_stride(buf0, (1024, 8192), (8192, 1))
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_leaky_relu_0[grid(8388608)](buf1, primals_2, buf2,
            8388608, XBLOCK=2048, num_warps=10, num_stages=1)
        del primals_2
    return buf1, primals_1, primals_3, buf2


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, group normalization, leaky ReLU activation, and element-wise sum.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5,
        negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size,
            eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, input_0):
        primals_1 = self.fc.weight
        primals_2 = self.fc.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
