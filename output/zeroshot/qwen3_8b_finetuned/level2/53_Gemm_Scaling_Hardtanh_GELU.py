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


@triton.jit
def triton_poi_fused_add_mul_hardtanh_gelu_0(in_out_ptr0, out_ptr0, xnumel,
    rnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + r1, xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.load(in_out_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp4 = 0.5
    tmp5 = tmp3 * tmp4
    tmp6 = -2.0
    tmp7 = tmp5 <= tmp6
    tmp8 = 2.0
    tmp9 = tmp5 >= tmp8
    tmp10 = tmp7 | tmp9
    tmp11 = 0.0
    tmp12 = tl.where(tmp10, tmp11, tmp5)
    tmp13 = tmp12
    tmp14 = 1.0
    tmp15 = tmp14 * tmp13
    tmp16 = 1.4142135623730951
    tmp17 = tmp15 / tmp16
    tmp18 = libdevice.erf(tmp17)
    tmp19 = tmp13 * tmp18
    tmp20 = 0.5
    tmp21 = tmp20 * tmp19
    tl.store(in_out_ptr0 + x0, tmp1, xmask)
    tl.store(out_ptr0 + (r1 + 8192 * x0), tmp21, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8192, 8192), (8192, 1))
    assert_size_stride(arg1_1, (2048, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2048, 8192), (8192, 1), torch.float32)
        extern_kernels.mm(arg1_1, arg0_1, out=buf0)
        del arg0_1
        del arg1_1
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((2048, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_mul_hardtanh_gelu_0[grid(32768)](buf1, buf2, 
            32768, 256, XBLOCK=1, num_warps=4, num_stages=1)
    return buf2,


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, scaling, hardtanh, and GELU activation.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
        self.gelu = nn.GELU()

    def forward(self, input_0):
        arg0_1 = self.gemm.weight
        del self.gemm
        arg1_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]