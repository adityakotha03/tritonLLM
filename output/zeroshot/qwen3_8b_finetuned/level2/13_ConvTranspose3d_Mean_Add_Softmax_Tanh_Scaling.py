import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x1, x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1, x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_mean_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x1, x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, 16])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_sub_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x1, x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1, x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_softmax_3(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    x1 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x1, x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, 16])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = tl.where(xmask, tmp4, 0)
    tmp6 = tmp0 - tmp5
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, 16])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.where(xmask, tmp11, 0)
    tmp13 = tmp7 / tmp12
    tl.store(in_out_ptr0 + x3, tmp13, xmask)


@triton.jit
def triton_poi_fused_tanh_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x1, x0), xmask, eviction_policy='evict_last')
    tmp1 = tl_math.tanh(tmp0)
    tl.store(out_ptr0 + x2, tmp1, xmask)


@triton.jit
def triton_poi_fused_mul_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x1, x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1, x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (16, 16, 32, 128, 128), (262144, 16384, 512, 64, 1))
    assert_size_stride(primals_2, (1, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(primals_3, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 64, 1, 128, 128), (1048576, 16384, 256, 2, 1), torch.float32)
        buf1 = empty_strided_cuda((16, 64, 1, 128, 128), (1048576, 16384, 256, 2, 1), torch.float32)
        buf2 = empty_strided_cuda((16, 64, 1, 128, 128), (1048576, 16384, 256, 2, 1), torch.float32)
        buf3 = empty_strided_cuda((16, 64, 1, 128, 128), (1048576, 16384, 256, 2, 1), torch.float32)
        buf4 = empty_strided_cuda((16, 64, 1, 128, 128), (1048576, 16384, 256, 2, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(16384)](primals_1, primals_2, buf0, 16384, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        triton_poi_fused_mean_1[grid(16384)](buf0, buf1, 16384, XBLOCK=128, num_warps=4, num_stages=1)
        triton_poi_fused_sub_2[grid(16384)](buf1, primals_1, buf2, 16384, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        triton_poi_fused_softmax_3[grid(16384)](buf2, 16384, XBLOCK=128, num_warps=4, num_stages=1)
        triton_poi_fused_tanh_4[grid(16384)](buf3, 16384, XBLOCK=128, num_warps=4, num_stages=1)
        triton_poi_fused_mul_5[grid(16384)](buf4, primals_3, buf3, 16384, XBLOCK=128, num_warps=4, num_stages=1)
    return buf4, primals_3,


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.bias
        primals_3 = self.scaling_factor
        primals_4 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]