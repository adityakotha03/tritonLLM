import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_mul_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x3, tmp4, xmask)


@triton.jit
def triton_poi_fused_clamp_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x5, xmask)
    tmp1 = tl.load(in_ptr0 + x4, xmask)
    tmp2 = tmp0 <= tmp1
    tmp3 = -10.0
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = tl.full([1], -10.0, tl.int32)
    tmp7 = tl.where(tmp5, tmp0, tmp6)
    tmp8 = 10.0
    tmp9 = tmp7 <= tmp8
    tmp10 = tmp7 >= tmp3
    tmp11 = tmp9 & tmp10
    tmp12 = tl.where(tmp11, tmp7, tmp8)
    tl.store(out_ptr0 + x5, tmp12, xmask)


@triton.jit
def triton_poi_fused_logsumexp_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x6 = xindex
    x2 = xindex
    x1 = xindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x6, xmask)
    tmp1 = tl.load(in_ptr0 + x2, xmask)
    tmp2 = tl.load(in_ptr0 + x1, xmask)
    tmp3 = tl.load(in_ptr0 + x0, xmask)
    tmp4 = tmp1 + tmp2
    tmp5 = tmp4 + tmp3
    tmp6 = tmp0 + tmp5
    tmp7 = tl_math.exp(tmp6)
    tl.store(in_ptr0 + x6, tmp7, xmask)
    tmp8 = tl.load(in_ptr0 + x2, xmask)
    tmp9 = tl.load(in_ptr0 + x1, xmask)
    tmp10 = tl.load(in_ptr0 + x0, xmask)
    tmp11 = tmp8 + tmp9
    tmp12 = tmp11 + tmp10
    tmp13 = tmp0 + tmp12
    tmp14 = tl_math.exp(tmp13)
    tl.store(in_ptr0 + x2, tmp14, xmask)
    tmp15 = tl.load(in_ptr0 + x1, xmask)
    tmp16 = tl.load(in_ptr0 + x0, xmask)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp0 + tmp17
    tmp19 = tl_math.exp(tmp18)
    tl.store(in_ptr0 + x1, tmp19, xmask)
    tmp20 = tl.load(in_ptr0 + x0, xmask)
    tmp21 = tmp0 + tmp20
    tmp22 = tl_math.exp(tmp21)
    tl.store(in_ptr0 + x0, tmp22, xmask)
    tmp23 = tl.load(in_ptr0 + x6, xmask)
    tmp24 = tl.load(in_ptr0 + x2, xmask)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.load(in_ptr0 + x1, xmask)
    tmp27 = tmp25 + tmp26
    tmp28 = tl.load(in_ptr0 + x0, xmask)
    tmp29 = tmp27 + tmp28
    tmp30 = tl_math.log(tmp29)
    tl.store(out_ptr0 + x6, tmp30, xmask)


@triton.jit
def triton_poi_fused_mul_mish_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x7 = xindex
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + x7, xmask)
    tmp1 = tl.load(in_ptr0 + x6, xmask)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tl_math.exp(tmp3)
    tmp5 = tmp4 + 1.0
    tmp6 = tl_math.log(tmp5)
    tmp7 = tmp1 * tmp6
    tmp8 = tmp7 * tmp1
    tmp9 = 0.0
    tmp10 = tmp7 <= tmp9
    tmp11 = tmp7 >= tmp9
    tmp12 = tmp10 & tmp11
    tmp13 = tl.where(tmp12, tmp7, tmp9)
    tmp14 = tl.load(in_ptr1 + x6, xmask)
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + x7, tmp15, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (1024, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(1024)](primals_1, buf0, 1024, XBLOCK=128,
            num_warps=4, num_stages=1)
        del primals_1
        buf1 = buf0
        del buf0
        buf2 = buf1
        del buf1
        buf3 = buf2
        del buf2
        buf4 = buf3
        del buf3
        triton_poi_fused_add_mul_1[grid(1024)](buf4, primals_2, 1024,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf5 = buf4
        del buf4
        buf6 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_clamp_2[grid(1024)](buf5, buf6, 1024, XBLOCK=128,
            num_warps=4, num_stages=1)
        del buf5
        buf7 = buf6
        del buf6
        buf8 = empty_strided_cuda((1024, 1), (1, 1024), torch.float32)
        triton_poi_fused_logsumexp_3[grid(1024)](buf7, buf8, 1024, XBLOCK=256,
            num_warps=4, num_stages=1)
        del buf7
        buf9 = buf8
        del buf8
        buf10 = buf9
        del buf9
        triton_poi_fused_mul_mish_4[grid(8192)](buf10, buf6, buf10, 8192,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf6
    return buf10, primals_2


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, scales the result, adds a residual connection, clamps the output,
    applies LogSumExp, and finally applies the Mish activation function.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min,
        clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)

    def forward(self, input_0):
        primals_1 = self.matmul.weight
        primals_2 = self.matmul.bias
        primals_3 = self.scale_factor
        primals_4 = self.clamp_min
        primals_5 = self.clamp_max
        output = call([input_0, primals_1, primals_2])
        return output[0]