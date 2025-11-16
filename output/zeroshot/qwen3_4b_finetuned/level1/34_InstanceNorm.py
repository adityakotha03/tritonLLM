import torch
from torch._inductor.select_algorithm import extern_kernels
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
def triton_poi_fused_div_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 120688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl_math.div(tmp0, 64.0)
    tl.store(out_ptr0 + x0, tmp1, xmask)


@triton.jit
def triton_poi_fused_add_mean_sqrt_sub_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 120688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32768
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp2
    tmp6 = tmp3 + tmp4
    tmp7 = tmp6 / 64.0
    tmp8 = triton_helpers.maximum(tmp7, 1e-05)
    tmp9 = tl_math.sqrt(tmp8)
    tmp10 = tmp5 / tmp9
    tmp11 = tmp2 / tmp9
    tl.store(out_ptr0 + x2, tmp10, xmask)
    tl.store(out_ptr0 + (128 * x2 + 128), tmp11, xmask)


@triton.jit
def triton_poi_fused_add_div_mean_sqrt_sub_2(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 120688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32768
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 / tmp5
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + x2, tmp7, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (112, 64, 512, 512), (16777216, 262144, 512, 
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 112, 512, 512), (65536, 512, 1, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_div_0[grid(120688)](arg0_1, buf0, 120688, XBLOCK=
            128, num_warps=4, num_stages=1)
        del arg0_1
        buf1 = empty_strided_cuda((112, 64, 512, 512), (32768, 512, 1, 1),
            torch.float32)
        triton_poi_fused_add_mean_sqrt_sub_1[grid(120688)](buf0, buf0, buf0,
            buf0, buf1, 120688, XBLOCK=512, num_warps=8, num_stages=1)
        buf2 = empty_strided_cuda((112, 64, 512, 512), (32768, 512, 1, 1),
            torch.float32)
        triton_poi_fused_add_div_mean_sqrt_sub_2[grid(120688)](buf0, buf1,
            buf1, buf1, buf2, 120688, XBLOCK=512, num_warps=8, num_stages=1)
        del buf1
        buf3 = extern_kernels.addmm(0.0, buf0, buf0, stride=(65536, 512), 
            alpha=1, beta=1)
        assert_size_stride(buf3, (112, 64, 512, 512), (32768, 512, 1, 1))
    return buf2, buf0


class ModelNew(nn.Module):
    """
    Simple model that performs Instance Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.inorm = nn.InstanceNorm2d(num_features=num_features)

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
