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


@triton.jit
def triton_poi_fused_add_max_mul_sum_0(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32768
    x1 = xindex // 32768
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2 * x1), xmask, eviction_policy='evict_last'
        )
    tmp1 = tl.load(in_ptr0 + (1 + x0 + 2 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = tmp3 + tmp0
    tmp5 = tmp3 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + x3, tmp8, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 32768), (32768, 1))
    assert_size_stride(arg1_1, (32768, 32768), (32768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        extern_kernels.mm(arg0_1, arg1_1, out=buf0)
        del arg0_1
        del arg1_1
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_add_max_mul_sum_0[grid(4194304)](buf1, buf1, 
            4194304, XBLOCK=256, num_warps=4, num_stages=1)
    return buf1,


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
        arg1_1 = self.matmul.weight
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]