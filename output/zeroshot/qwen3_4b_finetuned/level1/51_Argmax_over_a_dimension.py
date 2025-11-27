import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_argmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 51200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4095
    x1 = xindex // 4095
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096 * x1), xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x0 + 4096 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (2048 + x0 + 4096 * x1), xmask)
    tmp5 = tl.load(in_ptr0 + (3072 + x0 + 4096 * x1), xmask)
    tmp7 = tl.load(in_ptr0 + (x0 + 4096 * x1), xmask, eviction_policy=
        'evict_last')
    tmp8 = tl.load(in_ptr0 + (1024 + x0 + 4096 * x1), xmask, eviction_policy
        ='evict_last')
    tmp10 = tl.load(in_ptr0 + (2048 + x0 + 4096 * x1), xmask,
        eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (3072 + x0 + 4096 * x1), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 > tmp1
    tmp4 = tmp0 > tmp3
    tmp6 = tmp0 > tmp5
    tmp9 = tmp7 > tmp8
    tmp11 = tmp7 > tmp10
    tmp13 = tmp7 > tmp12
    tmp14 = tmp9 > tmp11
    tmp15 = tmp9 > tmp13
    tmp16 = tmp11 > tmp13
    tmp17 = tmp2 | tmp4
    tmp18 = tmp17 | tmp6
    tmp19 = tmp18 | tmp14
    tmp20 = tmp19 | tmp15
    tmp21 = tmp20 | tmp16
    tl.store(out_ptr0 + x2, tmp21, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 4096, 4095), (16796160, 4095, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 4095), (4095, 1), torch.bool)
        get_raw_stream(0)
        triton_poi_fused_argmax_0[grid(51200)](arg0_1, buf0, 51200, XBLOCK=
            128, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs Argmax over a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
