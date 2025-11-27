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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 1536000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 160000 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_max_pool3d_with_indices_1(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 384000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 64
    x2 = xindex // 8192 % 16
    x3 = xindex // 131072
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 512 * x2 + 8192 * x1 + 524288 * x3),
        xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 512 * x2 + 8192 * x1 + 524288 *
        x3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (256 + 2 * x0 + 512 * x2 + 8192 * x1 + 524288 *
        x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (257 + 2 * x0 + 512 * x2 + 8192 * x1 + 524288 *
        x3), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + x4, tmp6, xmask)
    tl.store(out_ptr1 + x4, tmp16, xmask)


@triton.jit
def triton_per_fused_logsumexp_relu_2(in_ptr0, out_ptr0, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    xnumel = 4
    rnumel = 384000
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 384000 * x0), rmask & xmask,
        eviction_policy='evict_first', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float('-inf'))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl_math.log(tmp10)
    tmp12 = tmp4 - tmp11
    tmp13 = tl.full([1, 1], 0, tl.int32)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tl.store(out_ptr0 + x0, tmp14, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (4, 32, 32, 128, 128), (1638400, 51200, 
        1600, 128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_3, primals_1, [1,
            1, 1], [1, 1, 1], bias=None)
        buf1 = buf0
        del buf0
        buf2 = reinterpret_tensor(buf1, (4, 64, 32, 128, 128), (3276800, 
            51200, 1600, 128, 1), 0)
        del buf1
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(1536000)](buf2, primals_2, 
            1536000, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((4, 64, 32, 128, 128), (3276800, 51200, 
            1600, 128, 1), torch.float32)
        buf4 = empty_strided_cuda((4, 64, 32, 128, 128), (3276800, 51200, 
            1600, 128, 1), torch.int8)
        triton_poi_fused_max_pool3d_with_indices_1[grid(384000)](buf2, buf3,
            buf4, 384000, XBLOCK=512, num_warps=8, num_stages=1)
        del buf2
        buf5 = empty_strided_cuda((4, 1, 32, 128, 128), (524288, 524288, 
            16384, 128, 1), torch.float32)
        triton_per_fused_logsumexp_relu_2[grid(4)](buf3, buf5, 4, 384000,
            XBLOCK=1, num_warps=2, num_stages=1)
    return buf5, primals_1, primals_3, buf3, buf4


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, max pooling, log sum exp, and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
