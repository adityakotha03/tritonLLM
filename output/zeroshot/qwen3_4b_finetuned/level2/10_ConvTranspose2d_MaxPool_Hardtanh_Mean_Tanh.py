import torch
from torch._inductor.select_algorithm import extern_kernels
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
def triton_poi_fused_hardtanh_max_pool2d_with_indices_mean_0(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 256 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = libdevice.tanh(tmp4)
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = triton_helpers.minimum(tmp7, tmp3)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x3, tmp8, xmask)
    tl.store(out_ptr0 + x3, tmp9, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_3, (128, 64, 256, 256), (4096, 64, 16, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 64, 258, 258), (415296, 64, 16, 1),
            torch.float32)
        extern_kernels.convolution(reinterpret_tensor(primals_3, (128, 64,
            256, 256), (4096, 1, 16, 64), 0), primals_1, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=True,
            output_padding=(0, 0), groups=1, bias=None, stride_cuda=(1, 1),
            padding_cuda=(1, 1), stride_strided=(1, 1), padding_strided=(1, 1
            ))
        del primals_1
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 81,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_hardtanh_max_pool2d_with_indices_mean_0[grid(256)](
            buf1, primals_2, buf2, 256, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
    return reinterpret_tensor(buf2, (128, 64, 1, 1), (64, 1, 64, 64), 0
        ), reinterpret_tensor(primals_3, (128, 64, 256, 256), (4096, 1, 16, 64
        ), 0), buf1


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, followed by max pooling, hardtanh activation, mean operation, and tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size,
            stride=maxpool_stride)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.weight
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
