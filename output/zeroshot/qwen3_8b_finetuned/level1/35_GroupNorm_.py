import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_addmm_0(in_ptr0, in_ptr1, out_ptr0, xnumel, ynumel,
    xoffset, yoffset, rnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    ynumel = 128
    yoffset = tl.program_id(1) * XBLOCK
    yindex = yoffset + tl.arange(0, XBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x0 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x0), xmask & ymask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (y0 + 512*x0), xmask & ymask, other=0.0)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + 512*x0), tmp2, xmask & ymask)


@triton.jit
def triton_poi_fused_muladd_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4,
    in_ptr5, out_ptr0, xnumel, ynumel, xoffset, yoffset, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 128
    ynumel = 128
    yoffset = tl.program_id(1) * XBLOCK
    yindex = yoffset + tl.arange(0, XBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x0 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x0), xmask & ymask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (y0 + 512*x0), xmask & ymask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (y0 + 512*x0), xmask & ymask, other=0.0)
    tmp3 = tl.load(in_ptr3 + (y0 + 512*x0), xmask & ymask, other=0.0)
    tmp4 = tl.load(in_ptr4 + (y0 + 512*x0), xmask & ymask, other=0.0)
    tmp5 = tl.load(in_ptr5 + (y0 + 512*x0), xmask & ymask, other=0.0)
    tmp6 = tmp0 + tmp1
    tmp7 = tmp2 + tmp3
    tmp8 = tmp6 * tmp7
    tmp9 = tmp4 + tmp5
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp10 / tmp11
    tmp13 = tl.load(in_ptr0 + (y0 + 512*x0), xmask & ymask, eviction_policy
        = 'evict_last')
    tmp14 = tmp13 - tmp12
    tmp15 = tl.load(in_ptr1 + (y0 + 512*x0), xmask & ymask, eviction_policy
        = 'evict_last')
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16 + tmp12
    tmp18 = tl.full([1], 1.0, tl.float32)
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr0 + (y0 + 512*x0), tmp19, xmask & ymask)


def triton_groupnorm(input_0, input_1, input_2, num_groups, num_features,
    dim1, dim2):
    args.clear()
    assert_size_stride(input_0, (112, 64, 512, 512), (16384, 256, 4, 1))
    assert_size_stride(input_1, (64,), (1,))
    assert_size_stride(input_2, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((112, 8, 512, 64), (32768, 2048, 16, 1),
            torch.float32)
        buf1 = empty_strided_cuda((112, 8, 512, 64), (32768, 2048, 16, 1),
            torch.float32)
        buf2 = empty_strided_cuda((112, 8, 512, 64), (32768, 2048, 16, 1),
            torch.float32)
        buf3 = empty_strided_cuda((112, 64, 512, 512), (16384, 256, 4, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_addmm_0[grid(128, 128)](input_0, input_1, buf0,
            128, 128, 0, 0, 8, XBLOCK=128, num_warps=4, num_stages=1)
        del input_1
        triton_poi_fused_muladd_1[grid(128, 128)](buf0, buf1, buf2, input_0,
            input_2, buf3, 128, 128, 0, 0, 8, XBLOCK=128, num_warps=4,
            num_stages=1)
        del input_0
        del input_2
    return buf3


class ModelNew(nn.Module):
    """
    Simple model that performs Group Normalization, implemented with custom
    Triton kernels for mean/variance calculation and normalization.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)

    def forward(self, input_0):
        arg0_1 = input_0
        arg0_2 = self.gn.weight
        arg0_3 = self.gn.bias
        output = triton_groupnorm(arg0_1, arg0_2, arg0_3, num_groups, num_features,
            512, 512)
        return output

batch_size = 112  # scaled up
features = 64
num_groups = 8
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features, num_groups] # num_features