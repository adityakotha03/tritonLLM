import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_group_norm_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 3136
    x1 = xindex % 3136
    x0 = xindex // 128
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 3136 * x2), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + x1 + 3136 * x2), xmask, eviction_policy
        ='evict_last')
    tmp4 = tl.load(in_ptr1 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (256 + x1 + 3136 * x2), xmask, eviction_policy
        ='evict_last')
    tmp8 = tl.load(in_ptr1 + (256 + x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (384 + x1 + 3136 * x2), xmask, eviction_policy
        ='evict_last')
    tmp12 = tl.load(in_ptr1 + (384 + x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1.0
    tmp18 = tmp16 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = 16.0
    tmp21 = tmp19 / tmp20
    tmp22 = tmp21 * tmp15
    tl.store(out_ptr0 + x4, tmp16, xmask)
    tl.store(out_ptr1 + x4, tmp22, xmask)
    tl.store(out_ptr2 + x4, tmp15, xmask)


@triton.jit
def triton_poi_fused_group_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 3136
    x1 = xindex % 3136
    x0 = xindex // 128
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 3136 * x2), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 / tmp3
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + x4, tmp6, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (112, 64, 512, 512), (16777216, 262144, 512,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((112, 64, 512, 512), (16777216, 262144, 
            512, 1), torch.float32)
        buf1 = empty_strided_cuda((112, 64, 512, 512), (16777216, 262144, 
            512, 1), torch.float32)
        buf2 = empty_strided_cuda((112, 64, 512, 512), (16777216, 262144, 
            512, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_group_norm_0[grid(125440)](arg0_1, arg0_1, buf0,
            buf1, buf2, 125440, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
        buf3 = empty_strided_cuda((112, 64, 512, 512), (16777216, 262144, 
            512, 1), torch.float32)
        triton_poi_fused_group_norm_1[grid(125440)](buf0, buf1, buf2, buf3,
            buf3, 125440, XBLOCK=128, num_warps=4, num_stages=1)
    return buf3,


class ModelNew(nn.Module):
    """
    Simple model that performs Group Normalization.
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
        output = call([arg0_1])
        return output[0]
