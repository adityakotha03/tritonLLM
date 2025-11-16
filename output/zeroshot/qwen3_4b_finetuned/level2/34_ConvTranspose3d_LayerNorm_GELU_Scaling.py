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
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 256 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_gelu_layer_norm_1(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + 0)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp1 = tmp0 + tmp4
    tmp2 = 1e-05
    tmp5 = tmp1 * tmp1
    tmp6 = tl_math.exp(-tmp5)
    tmp7 = -tmp6
    tmp8 = 0.0
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = 0.0
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + x2, tmp13, xmask)
    tl.store(out_ptr1 + x2, tmp1, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 4, 4, 4), (2048, 64, 64, 16, 4))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (32, 32, 16, 32, 32), (524288, 16384, 32768,
        1024, 32))
    assert_size_stride(primals_4, (64,), (1,))
    assert_size_stride(primals_5, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 64, 17, 33, 33), (3660864, 57601, 
            1024, 32, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(16384)](buf0, primals_1,
            16384, XBLOCK=1024, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((32, 64, 17, 33, 33), (3660864, 57601, 
            1024, 32, 1), torch.float32)
        buf2 = empty_strided_cuda((64, 1), (1,), torch.float32)
        triton_poi_fused_add_gelu_layer_norm_1[grid(16384)](buf1, primals_2,
            buf2, buf0, 16384, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        del buf2
        del primals_2
    return buf1, primals_1, primals_3, primals_4, primals_5


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, layer normalization, GELU activation, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
        bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_4 = self.layer_norm.weight
        primals_5 = self.layer_norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
