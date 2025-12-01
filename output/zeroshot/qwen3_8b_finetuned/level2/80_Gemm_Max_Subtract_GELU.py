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
def triton_poi_fused__to_copy_add_div_max_mean_mul_sub_0(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1024 + x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2048 + x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3072 + x0), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = tl.load(in_ptr0 + (4096 + x0), xmask, eviction_policy='evict_last')
    tmp8 = tmp6 / 4.0
    tmp9 = tmp7 - tmp8
    tmp10 = tmp0 - tmp9
    tl.store(out_ptr0 + x0, tmp10, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.mm(arg0_1, arg0_1, out=empty_strided_cuda((1024,
            8192), (8192, 1), torch.float32))
        del arg0_1
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__to_copy_add_div_max_mean_mul_sub_0[grid(8388608)](
            buf0, buf1, 8388608, XBLOCK=256, num_warps=4, num_stages=1)
        del buf0
    return buf1,


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, followed by a max operation, subtraction, and GELU activation.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, input_0):
        arg0_1 = self.gemm.weight
        arg1_1 = input_0
        self.gemm.weight = None
        buf0 = extern_kernels.mm(arg1_1, arg0_1, out=empty_strided_cuda((1024,
            8192), (8192, 1), torch.float32))
        del arg0_1
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__to_copy_add_div_max_mean_mul_sub_0[grid(8388608)](
            buf0, buf1, 8388608, XBLOCK=256, num_warps=4, num_stages=1)
        del buf0
        return buf1, input_0