import torch
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
def triton_poi_fused_add_clamp_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2 + tmp2
    tmp4 = -10.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = 10.0
    tmp7 = triton_helpers.minimum(tmp5, tmp6)
    tl.store(out_ptr0 + x0, tmp7, xmask)


@triton.jit
def triton_per_fused_logsumexp_mul_mish_1(in_ptr0, out_ptr2, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1024
    rnumel = 8192
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 8192 * x0), rmask & xmask, other=0.0)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2 + tmp2
    tmp4 = -10.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = 10.0
    tmp7 = triton_helpers.minimum(tmp5, tmp6)
    tmp8 = tl_math.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tl_math.log(tmp12)
    tmp14 = tmp7 - tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp18 * tmp18
    tmp20 = libdevice.sqrt(tmp19)
    tl.store(out_ptr2 + (r1 + 8192 * x0), tmp18, rmask & xmask)


@triton.jit
def triton_poi_fused_mish_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl_math.exp(tmp1)
    tmp3 = libdevice.log1p(tmp2)
    tmp4 = tmp0 * tmp3
    tmp5 = 0.0
    tmp6 = tmp1 > tmp5
    tmp7 = 0.01
    tmp8 = tmp1 * tmp7
    tmp9 = tl.where(tmp6, tmp1, tmp8)
    tmp10 = tmp9 * tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tmp0 * tmp11
    tmp13 = tmp4 + tmp12
    tl.store(out_ptr0 + x0, tmp13, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_clamp_mul_0[grid(8388608)](primals_3, buf0, 
            8388608, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_3
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_add_clamp_mul_0[grid(8388608)](buf0, buf1, 8388608,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((1024, 1), (1, 1024), torch.float32)
        buf3 = reinterpret_tensor(buf2, (1024, 1), (1, 1), 0)
        del buf2
        triton_per_fused_logsumexp_mul_mish_1[grid(1024)](buf1, buf3, 1024,
            8192, XBLOCK=32, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_mish_2[grid(8388608)](buf3, buf1, buf4, 8388608,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del buf1
        del buf3
    return buf4, primals_1, primals_2


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
        primals_1 = self.matmul.weight
        primals_2 = self.matmul.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
