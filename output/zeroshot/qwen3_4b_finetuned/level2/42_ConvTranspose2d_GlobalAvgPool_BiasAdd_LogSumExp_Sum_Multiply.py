import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 1297952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 262144 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_per_fused_add_mean_1(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0,
    xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = xindex % 64
    x3 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (r1 + 1024 * x0), xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + x2, xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, float('-inf'))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tmp16 = 1024.0
    tmp17 = tmp15 / tmp16
    tmp18 = tmp17 + tmp24
    tmp19 = 0.0
    tmp20 = tmp18 + tmp19
    tmp21 = 0.015625
    tmp22 = tmp20 * tmp21
    tmp23 = tmp22 * tmp21
    tmp25 = tmp23 + tmp19
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0 + 128 * x3), tmp25, xmask)
    tl.store(out_ptr0 + (r1 + 1024 * x0), tmp25, xmask)


@triton.jit
def triton_poi_fused_add_mul_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 10.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (16, 64, 512, 512), (16777216, 262144, 
        512, 1))
    assert_size_stride(primals_4, (128, 1, 1), (1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_3, primals_1, [1,
            1], 1, 1, 0, 1024, 1)
        buf1 = buf0
        del buf0
        buf2 = buf1
        del buf1
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(1297952)](buf2, primals_2, 
            1297952, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((128, 1, 1), (1, 128, 128), torch.float32)
        buf4 = buf3
        del buf3
        buf5 = empty_strided_cuda((128, 1024, 1), (1024, 1, 1024), torch.
            float32)
        triton_per_fused_add_mean_1[grid(128)](buf4, buf2, primals_4, buf5,
            128, 1024, XBLOCK=1, num_warps=2, num_stages=1)
        del buf2
        del primals_4
        buf6 = empty_strided_cuda((128, 1, 1), (1, 1, 1), torch.float32)
        triton_poi_fused_add_mul_2[grid(128)](buf4, buf6, 128, XBLOCK=128,
            num_warps=4, num_stages=1)
        del buf4
    return buf6, primals_1, primals_3, buf5


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, global average pooling, adds a bias, applies log-sum-exp, sum, and multiplication.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, input_0):
        primals_4 = self.bias
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
