import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_abs_mean_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 65535
    x1 = xindex // 65535
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 65535 * x1), xmask)
    tmp1 = tl.load(in_ptr0 + (65535 + x0 + 65535 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (131070 + x0 + 65535 * x1), xmask)
    tmp5 = tl.load(in_ptr0 + (196605 + x0 + 65535 * x1), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + x2, tmp12, xmask)


@triton.jit
def triton_poi_fused_div_mean_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 65535
    x1 = xindex // 65535
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (65535 + x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (1 + x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (131070 + x0), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr1 + (2 + x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (196605 + x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr1 + (3 + x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + x2, xmask)
    tmp11 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp12 = tmp11 / tmp1
    tmp13 = tmp0 / tmp12
    tmp14 = tmp2 / tmp3
    tmp15 = tmp13 + tmp14
    tmp16 = tmp5 / tmp6
    tmp17 = tmp15 + tmp16
    tmp18 = tmp8 / tmp9
    tmp19 = tmp17 + tmp18
    tmp20 = tmp10 / tmp19
    tl.store(out_ptr0 + x2, tmp20, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (32768, 65535), (65535, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768, 1), (1, 65535), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_abs_mean_0[grid(65536)](arg0_1, buf0, 65536,
            XBLOCK=128, num_warps=8, num_stages=1)
        del arg0_1
        buf1 = empty_strided_cuda((32768, 65535), (65535, 1), torch.float32)
        triton_poi_fused_div_mean_1[grid(2097152)](buf0, buf0, buf1, 2097152,
            XBLOCK=1024, num_warps=4, num_stages=1)
    return buf1,


class ModelNew(nn.Module):
    """
    Simple model that performs L1 normalization.
    """
    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
