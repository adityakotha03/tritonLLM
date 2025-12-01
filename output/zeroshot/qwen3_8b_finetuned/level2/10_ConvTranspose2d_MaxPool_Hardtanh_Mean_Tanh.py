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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_hardtanh_max_pool2d_with_indices_2(in_ptr0, out_ptr0,
    out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 128
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (2 * x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 2 * x0 + 256 * x1), xmask, eviction_policy
        ='evict_last')
    tmp5 = tl.load(in_ptr0 + (128 + 2 * x0 + 256 * x1), xmask,
        eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (129 + 2 * x0 + 256 * x1), xmask,
        eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp3)
    tmp4 = triton_helpers.maximum(tmp2, tmp5)
    tmp6 = triton_helpers.maximum(tmp4, tmp7)
    tmp8 = triton_helpers.maximum(tmp0, tmp6)
    tmp9 = -1.0
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp11 = 1.0
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tmp13 = tmp8 > tmp6
    tmp14 = tl.full([1], 0, tl.int64)
    tmp15 = tl.full([1], 1, tl.int64)
    tmp16 = tmp13 == tmp14
    tmp17 = tmp13 == tmp15
    tmp18 = tmp16 | tmp17
    tmp19 = tl.full([1], 2, tl.int64)
    tmp20 = tmp13 == tmp19
    tmp21 = tmp18 | tmp20
    tmp22 = tl.full([1], 3, tl.int64)
    tmp23 = tmp13 == tmp22
    tmp24 = tmp21 | tmp23
    tl.store(out_ptr0 + x3, tmp12, xmask)
    tl.store(out_ptr1 + x3, tmp13, xmask)
    tl.store(out_ptr2 + x3, tmp24, xmask)


@triton.jit
def triton_poi_fused_mean_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    x1 = xindex // 64
    tmp0 = tl.load(in_ptr0 + 4 * x2, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4 * x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 4 * x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 4 * x2), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 + tmp3
    tmp2 = tmp1 + tmp6
    tmp4 = tmp3 + tmp6
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 + tmp9
    tmp8 = tmp5 + tmp7
    tmp10 = 4.0
    tmp11 = tmp8 / tmp10
    tl.store(out_ptr0 + (x0 + 64 * x1), tmp11, xmask)


@triton.jit
def triton_poi_fused_tanh_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.tanh(tmp0)
    tl.store(out_ptr0 + x0, tmp1, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 256, 256), (4194304, 65536, 256,
        1))
    assert_size_stride(primals_2, (64, 64, 3, 3), (576, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=True,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 256, 256), (4194304, 65536, 256, 
            1))
        buf1 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 8192,
            1), torch.float16)
        buf2 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 8192,
            1), torch.float16)
        buf3 = empty_strided_cuda((128, 64, 1, 1), (8192, 128, 128, 1),
            torch.float16)
        buf4 = empty_strided_cuda((128, 64, 1, 1), (8192, 128, 128, 1),
            torch.float16)
        get_raw_stream(0)
        triton_poi_fused_hardtanh_max_pool2d_with_indices_2[grid(1048576)](buf0,
            buf1, buf2, buf3, 1048576, XBLOCK=128, num_warps=4, num_stages=1)
        buf5 = empty_strided_cuda((128, 64, 1, 1), (8192, 128, 128, 1),
            torch.float16)
        triton_poi_fused_mean_3[grid(8192)](buf2, buf5, 8192, XBLOCK=256,
            num_warps=4, num_stages=1)
        buf6 = empty_strided_cuda((128, 64, 1, 1), (8192, 128, 128, 1),
            torch.float16)
        triton_poi_fused_tanh_4[grid(8192)](buf5, buf6, 8192, XBLOCK=256,
            num_warps=4, num_stages=1)
        del buf5
    return buf6, primals_1, primals_2, buf0, buf1, buf2, buf3


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, followed by max pooling, hardtanh activation, mean operation, and tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, input_0):
        primals_2 = self.conv_transpose.weight
        primals_1 = input_0
        output = call([primals_1, primals_2])
        return output[0]