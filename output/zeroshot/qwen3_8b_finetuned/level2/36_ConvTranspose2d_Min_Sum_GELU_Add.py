import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_gelu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex // 131072
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 131072), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x0 + 262144), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0 + 393216), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp4 = 1.0
    tmp6 = 1.4142135623731027
    tmp7 = tmp6 * tmp2
    tmp8 = tmp0 * tmp7
    tmp9 = tmp0 * tmp4
    tmp10 = tmp1 - tmp8
    tmp11 = tmp10 * tmp9
    tmp12 = tmp1 * tmp9
    tmp13 = tmp3 * tmp9
    tmp14 = tmp5 * tmp9
    tmp15 = tmp11 + tmp12
    tmp16 = tmp13 + tmp15
    tmp17 = tmp16 + tmp14
    tmp18 = tmp17 * tmp1
    tmp19 = 0.5
    tmp20 = tmp19 + tmp18
    tmp21 = tmp12 * tmp14
    tmp22 = tmp14 * tmp13
    tmp23 = tmp21 + tmp22
    tmp24 = tmp20 * tmp23
    tl.store(out_ptr0 + x3, tmp24, xmask)


@triton.jit
def triton_poi_fused_add_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex // 131072
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x3, xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


def call(args):
    (primals_1, primals_2) = args
    args.clear()
    assert_size_stride(primals_1, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_2, (1, 1, 128, 128), (16384, 128, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 1, 1, 128), (131072, 131072, 131072, 1), torch.float32)
        buf1 = empty_strided_cuda((16, 1, 1, 128), (131072, 131072, 131072, 1), torch.float32)
        buf2 = empty_strided_cuda((16, 1, 1, 128), (131072, 131072, 131072, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_gelu_0[grid(2048)](primals_2, buf0, 2048, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        triton_poi_fused_add_1[grid(2048)](buf0, primals_1, buf1, 2048, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        del buf0
        buf3 = buf1
        del buf1
    return reinterpret_tensor(buf3, (16, 1, 1, 128), (131072, 131072, 131072, 1), 0)


class ModelNew(nn.Module):
    """
    A model that performs a convolution transpose, minimum operation, sum operation, GELU activation and addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, input_0):
        primals_1 = self.bias
        primals_2 = self.conv_transpose.weight
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output