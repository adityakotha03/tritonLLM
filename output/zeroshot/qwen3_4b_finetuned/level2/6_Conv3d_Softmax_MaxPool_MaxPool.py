import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, in_ptr1, out_ptr0, ynumel,
    xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 16848
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (y3 + 16848 * x2 + 512 * y0), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 9 * y3), tmp2, xmask & ymask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 36992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 4 * x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4 * x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 4 * x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 4 * x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1
    tmp4 = tmp2 < tmp3
    tmp5 = tl.where(tmp4, tmp3, tmp2)
    tmp7 = tmp3 > tmp6
    tmp8 = tl.where(tmp4, tmp6, tmp5)
    tmp10 = tl.full([1], 3, tl.int64)
    tmp11 = tmp9 > tmp10
    tmp12 = tl.where(tmp7, tmp10, tmp8)
    tmp13 = tl.where(tmp11, tmp9, tmp12)
    tmp14 = tl.where(tmp11, tmp9, tmp12)
    tmp15 = tl.where(tmp7, tmp6, tmp13)
    tmp16 = tl.where(tmp4, tmp15, tmp5)
    tmp17 = tmp16.to(tl.int32)
    tl.store(out_ptr0 + x2, tmp16, xmask)
    tl.store(out_ptr1 + x2, tmp17, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_2(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 36992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 4 * x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4 * x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 4 * x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 4 * x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1
    tmp4 = tmp2 < tmp3
    tmp5 = tl.where(tmp4, tmp3, tmp2)
    tmp7 = tmp3 > tmp6
    tmp8 = tl.where(tmp4, tmp6, tmp5)
    tmp10 = tl.full([1], 3, tl.int64)
    tmp11 = tmp9 > tmp10
    tmp12 = tl.where(tmp7, tmp10, tmp8)
    tmp13 = tl.where(tmp11, tmp9, tmp12)
    tmp14 = tl.where(tmp11, tmp9, tmp12)
    tmp15 = tl.where(tmp7, tmp6, tmp13)
    tmp16 = tl.where(tmp4, tmp15, tmp5)
    tmp17 = tmp16.to(tl.int32)
    tl.store(out_ptr0 + x2, tmp16, xmask)
    tl.store(out_ptr1 + x2, tmp17, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5) = args
    args.clear()
    assert_size_stride(primals_1, (3, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (3,), (1,))
    assert_size_stride(primals_3, (128, 3, 16, 32, 32), (15744, 512, 32, 1, 
        1))
    assert_size_stride(primals_4, (16,), (1,))
    assert_size_stride(primals_5, (16,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 16, 15, 30, 30), (86400, 5400, 360,
            12, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(16848, 9)](primals_3, primals_1,
            buf0, 16848, 9, XBLOCK=64, YBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((128, 16, 15, 30, 30), (86400, 5400, 360,
            12, 1), torch.float32)
        extern_kernels.softmax(buf0, 1, out=buf1)
        del buf0
        buf2 = empty_strided_cuda((128, 16, 7, 15, 15), (28800, 1800, 256,
            16, 1), torch.int32)
        buf3 = empty_strided_cuda((128, 16, 7, 15, 15), (28800, 1800, 256,
            16, 1), torch.int64)
        triton_poi_fused_max_pool2d_with_indices_1[grid(36992)](buf1, buf2,
            buf3, 36992, XBLOCK=256, num_warps=4, num_stages=1)
        del buf1
        buf4 = empty_strided_cuda((128, 16, 3, 7, 7), (16800, 1050, 360, 56,
            1), torch.int32)
        buf5 = empty_strided_cuda((128, 16, 3, 7, 7), (16800, 1050, 360, 56,
            1), torch.int64)
        triton_poi_fused_max_pool2d_with_indices_2[grid(36992)](buf2, buf4,
            buf5, 36992, XBLOCK=256, num_warps=4, num_stages=1)
        del buf2
    return reinterpret_tensor(buf3, (128, 16, 7, 15, 15), (28800, 1800, 256,
        16, 1), 0), primals_2, primals_4, primals_5, buf4, buf5, primals_3


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
        primals_4 = self.pool1.padding
        primals_5 = self.pool2.padding
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5])
        return output[0]
