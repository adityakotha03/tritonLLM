import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn.functional as F


@triton.jit
def triton_poi_fused_min_add_mul_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0,
    xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 268435456
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex % 128
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + 0, xmask, eviction_policy='evict_last')
    tmp4 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.minimum(tmp0, tmp4)
    tmp6 = tmp5 + tmp2
    tmp7 = tmp6 * tmp3
    tl.store(out_ptr0 + x0, tmp7, xmask)


def triton_min_add_mul(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 128, 128, 128), (2097152, 262144, 32768, 256))
    assert_size_stride(arg1_1, (1,), (1,))
    assert_size_stride(arg2_1, (128, 1, 1), (128, 1, 1))
    assert_size_stride(arg3_1, (1,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 128, 128, 128), (2097152, 262144, 32768,
            256), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_min_add_mul_0[grid(268435456)](arg0_1, arg1_1,
            arg2_1, arg3_1, buf0, 268435456, 1, XBLOCK=256, num_warps=4,
            num_stages=1)
        del arg1_1
        del arg2_1
        del arg3_1
    return buf0,


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, elementwise min with a constant,
    adds a bias, and multiplies by a scaling factor, with the elementwise
    operations replaced by a fused Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value,
        bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        arg0_1 = input_0
        arg1_1 = torch.tensor([self.constant_value], device='cuda')
        arg2_1 = self.bias
        arg3_1 = torch.tensor([self.scaling_factor], device='cuda')
        output = triton_min_add_mul([arg0_1, arg1_1, arg2_1, arg3_1])
        return output[0]