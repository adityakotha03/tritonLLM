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
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 8192 % 16
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 8192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 8192 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (8193 + 8192 * x1), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (16384 + 8192 * x1), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (24576 + 8192 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + x2, tmp9, xmask)


@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 8192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 8192 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (8193 + 8192 * x1), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (16384 + 8192 * x1), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (24576 + 8192 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_max_pool3d_with_indices_3(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32 * x1 + 512 * x2), xmask)
    tmp1 = tl.load(in_ptr0 + (16 + x0 + 32 * x1 + 512 * x2), xmask)
    tmp3 = tl.load(in_ptr0 + (8192 + x0 + 32 * x1 + 512 * x2), xmask)
    tmp5 = tl.load(in_ptr0 + (8193 + x0 + 32 * x1 + 512 * x2), xmask)
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
    tl.store(out_ptr0 + x3, tmp6, xmask)
    tl.store(out_ptr1 + x3, tmp16, xmask)


@triton.jit
def triton_poi_fused_max_pool3d_with_indices_4(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x1 = xindex // 8 % 8
    x2 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr0 + (8 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp3 = tl.load(in_ptr0 + (128 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp5 = tl.load(in_ptr0 + (136 + x0 + 16 * x1 + 256 * x2), xmask)
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
    tl.store(out_ptr0 + x3, tmp6, xmask)
    tl.store(out_ptr1 + x3, tmp16, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (16,), (1,))
    assert_size_stride(primals_3, (128, 3, 16, 32, 32), (49152, 16384, 1024,
        32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 16, 14, 30, 30), (1007040, 62940, 4496,
            149, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(524288)](buf1, primals_2, 524288,
            XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((128, 16, 14, 30, 30), (1007040, 62940, 
            4496, 149, 1), torch.float32)
        triton_poi_fused__softmax_1[grid(524288)](buf1, buf2, 524288, XBLOCK
            =512, num_warps=8, num_stages=1)
        buf3 = buf1
        del buf1
        triton_poi_fused__softmax_2[grid(524288)](buf2, buf3, 524288, XBLOCK
            =512, num_warps=8, num_stages=1)
        del buf2
        buf4 = empty_strided_cuda((128, 16, 14, 30, 30), (1007040, 62940, 
            4496, 149, 1), torch.float32)
        buf5 = empty_strided_cuda((128, 16, 14, 30, 30), (1007040, 62940, 
            4496, 149, 1), torch.int8)
        triton_poi_fused_max_pool3d_with_indices_3[grid(131072)](buf3, buf4,
            buf5, 131072, XBLOCK=512, num_warps=8, num_stages=1)
        buf6 = empty_strided_cuda((128, 16, 8, 15, 15), (192000, 12000, 
            1500, 100, 1), torch.float32)
        buf7 = empty_strided_cuda((128, 16, 8, 15, 15), (192000, 12000, 
            1500, 100, 1), torch.int8)
        triton_poi_fused_max_pool3d_with_indices_4[grid(65536)](buf4, buf6,
            buf7, 65536, XBLOCK=256, num_warps=4, num_stages=1)
        del buf4
    return (reinterpret_tensor(buf6, (128, 16, 8, 15, 15), (192000, 12000, 
        1500, 100, 1), 0), primals_1, primals_3, buf3, buf5, buf7)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Softmax, and performs two max pooling operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool1 = nn.MaxPool3d(pool_kernel_size)
        self.pool2 = nn.MaxPool3d(pool_kernel_size)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
