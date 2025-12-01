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
def triton_poi_fused__log_softmax_2(in_ptr0, out_ptr0, xnumel, ynumel,
    XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
    xnumel = 1024
    ynumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + 32768 * y0 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, YBLOCK])
    tmp3 = tl.where(xmask & ymask, tmp1, float('-inf'))
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 1)[:, None])
    tmp5 = tmp0 - tmp4
    tmp6 = tl.full([1, 1], 0, tl.int32)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tl.store(out_ptr0 + (x2 + 32768 * y3), tmp7, xmask & ymask)


@triton.jit
def triton_poi_fused__log_softmax_2_1(in_ptr0, out_ptr0, xnumel, ynumel,
    XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
    xnumel = 1024
    ynumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + 32768 * y0 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, YBLOCK])
    tmp3 = tl.where(xmask & ymask, tmp1, float('-inf'))
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 1)[:, None])
    tmp5 = tmp0 - tmp4
    tmp6 = tl.full([1, 1], 0, tl.int32)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tl.store(out_ptr0 + (x2 + 32768 * y3), tmp7, xmask & ymask)


@triton.jit
def triton_poi_fused__log_softmax_2_2(in_ptr0, out_ptr0, xnumel, ynumel,
    XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
    xnumel = 1024
    ynumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + 32768 * y0 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, YBLOCK])
    tmp3 = tl.where(xmask & ymask, tmp1, float('-inf'))
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 1)[:, None])
    tmp5 = tmp0 - tmp4
    tmp6 = tl.full([1, 1], 0, tl.int32)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tl.store(out_ptr0 + (x2 + 32768 * y3), tmp7, xmask & ymask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 32, 3, 32, 128, 128), (32768, 1024, 32, 2,
        1, 1))
    assert_size_stride(arg1_1, (64, 32, 3, 3), (288, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(arg0_1, arg1_1, stride=(1, 1, 1),
            padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 32, 128, 128), (32768, 512, 16, 4, 1))
        buf1 = empty_strided_cuda((64, 16, 64, 64), (65536, 4096, 64, 1),
            torch.float32)
        buf2 = buf1
        del buf1
        extern_kernels.max_pool3d_with_indices(buf0, (2, 2, 2), (2, 2, 2),
            (0, 0, 0), (1, 1, 1), (0, 0, 0), buf2, None)
        del buf0
        buf3 = empty_strided_cuda((4, 64, 16, 64, 64), (65536, 1024, 64, 1, 
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__log_softmax_2[grid(1024, 64)](buf2, buf3, 1024, 
            64, XBLOCK=128, YBLOCK=64, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((4, 64, 16, 64, 64), (65536, 1024, 64, 1, 
            1), torch.float32)
        triton_poi_fused__log_softmax_2_1[grid(1024, 64)](buf2, buf4, 1024,
            64, XBLOCK=128, YBLOCK=64, num_warps=4, num_stages=1)
        buf5 = empty_strided_cuda((4, 64, 16, 64, 64), (65536, 1024, 64, 1, 
            1), torch.float32)
        triton_poi_fused__log_softmax_2_2[grid(1024, 64)](buf2, buf5, 1024,
            64, XBLOCK=128, YBLOCK=64, num_warps=4, num_stages=1)
        del buf2
    return buf5, arg0_1, arg1_1


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, max pooling, log sum exp, and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, input_0):
        arg1_1 = self.conv.weight
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]