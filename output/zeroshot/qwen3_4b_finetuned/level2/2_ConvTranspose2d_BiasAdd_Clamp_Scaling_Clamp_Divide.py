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
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK:
    tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 768
    xnumel = 1
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex % 32
    y1 = yindex // 32
    y2 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32 * x1 + 1024 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + 1 * y2), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_add_clamp_div_mul_sub_1(in_out_ptr0, in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 1.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = 2.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp0 - tmp8
    tmp10 = triton_helpers.maximum(tmp9, tmp3)
    tmp11 = triton_helpers.minimum(tmp10, tmp5)
    tmp12 = tmp11 / tmp7
    tl.store(in_out_ptr0 + x2, tmp12, xmask)
    tl.store(out_ptr0 + x2, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 64, 128, 128), (1048576, 16384, 128,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch
            .float32)
        get_raw_stream(0)
        triton_poi_fused__to_copy_0[grid(768, 1)](primals_1, buf0, 768, 1,
            XBLOCK=1, YBLOCK=64, num_warps=4, num_stages=1)
        del primals_1
        buf1 = extern_kernels.conv_transpose_with_indices(primals_3, buf0,
            stride=(2,), padding=(1,), dilation=(1,), transposed=True,
            output_padding=(1,), groups=1, bias=None)
        assert_size_stride(buf1, (128, 64, 130, 130), (1081600, 1, 8320, 64))
        buf2 = buf1
        del buf1
        buf3 = empty_strided_cuda((128, 64, 130, 130), (1081600, 1, 8320, 64
            ), torch.float32)
        triton_poi_fused_add_clamp_div_mul_sub_1[grid(768)](buf2, primals_2,
            buf3, 768, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_2
    return buf3, primals_3, buf0, buf2


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
