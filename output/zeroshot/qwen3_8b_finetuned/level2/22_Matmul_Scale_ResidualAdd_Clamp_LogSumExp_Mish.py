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
def triton_poi_fused_add_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask)
    tmp2 = 2.0
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3 + tmp1
    tl.store(out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_poi_fused_add_clamp_log_mish_mul_1(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 8192 * x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x2, xmask)
    tmp4 = tl.load(in_ptr1 + 8192 * x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp5 = tmp3 + tmp4
    tmp6 = triton_helpers.maximum(tmp5, -10.0)
    tmp7 = triton_helpers.minimum(tmp6, 10.0)
    tmp8 = 4.0
    tmp9 = tmp7 / tmp8
    tmp10 = libdevice.tanh(tmp9)
    tmp11 = tmp5 * tmp10
    tmp12 = tmp10 + 1.0
    tmp13 = tmp11 * tmp12
    tmp14 = tmp2 + tmp13
    tmp15 = tmp14 / tmp8
    tmp16 = libdevice.tanh(tmp15)
    tmp17 = tmp14 * tmp16
    tmp18 = tmp16 + 1.0
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr0 + x2, tmp19, xmask)
    tl.store(out_ptr1 + x2, tmp15, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.mm(arg0_1, arg0_1, out=buf0)
        del arg0_1
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_mul_0[grid(8388608)](buf0, buf1, 8388608,
            XBLOCK=512, num_warps=8, num_stages=1)
        buf2 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        buf3 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_add_clamp_log_mish_mul_1[grid(8388608)](buf0, buf1,
            buf2, buf3, 8388608, XBLOCK=512, num_warps=8, num_stages=1)
    return buf2, buf3, buf0, buf1


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, scales the result, adds a residual connection, clamps the output,
    applies LogSumExp, and finally applies the Mish activation function.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min,
        clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]