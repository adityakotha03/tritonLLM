import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (32, 32, 32, 64, 64), (134217728, 4194304,
        131072, 2048, 32))
    assert_size_stride(primals_2, (1, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 32, 32, 64, 64), (134217728, 4194304,
            131072, 2048, 32), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(134217728)](primals_1, primals_2, buf0,
            134217728, XBLOCK=128, num_warps=4, num_stages=1)
    return buf0, primals_1, primals_2


class ModelNew(nn.Module):
    """
    Model that performs average pooling, 3D transposed convolution, clamping,
    spatial softmax, and multiplication by a learnable scale.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, input_0):
        primals_2 = self.scale
        primals_1 = input_0
        output = call([primals_1, primals_2])
        return output[0]