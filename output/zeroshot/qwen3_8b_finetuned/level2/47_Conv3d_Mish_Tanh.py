import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_mish_tanh_0(in_out_ptr0, in_ptr0, out_ptr0,
    xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 117869600
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp9 = 1.0
    tmp10 = tmp2 + tmp9
    tmp11 = tmp4 + tmp9
    tmp12 = tmp6 + tmp9
    tmp13 = tmp8 + tmp9
    tmp14 = tl.math.exp(tmp10)
    tmp15 = tl.math.log(tmp11)
    tmp16 = tl.math.tanh(tmp14)
    tmp17 = tmp16 + tmp15
    tmp18 = tmp12 * tmp17
    tmp19 = tl.math.tanh(tmp18)
    tmp20 = tmp13 * tmp19
    tl.store(out_ptr0 + x0, tmp20, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 32, 32, 64, 64), (131072, 4096, 128, 2, 1))
    assert_size_stride(arg1_1, (64, 3, 3, 3, 32), (1536, 512, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 64, 30, 62, 62), (115296, 1860, 62, 1, 1),
            torch.float32)
        buf1 = buf0
        del buf0
        buf2 = buf1
        del buf1
        buf3 = buf2
        del buf2
        get_raw_stream(0)
        triton_poi_fused_convolution_mish_tanh_0[grid(117869600)](buf3,
            arg0_1, arg1_1, 117869600, 1, XBLOCK=128, num_warps=4,
            num_stages=1)
        del arg0_1
        del arg1_1
    return buf3, buf2, buf3


class ModelNew(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        del kwargs
        assert_size_stride = torch._C._dynamo.guards.assert_size_stride
        self._criterion = None
        self.conv = nn.Conv3d(32, 64, (3, 3, 3), stride=(1, 1, 1), padding=(0,
            0, 0), bias=False, _inplace=False)

    def forward(self, input_0):
        arg0_1 = input_0
        arg1_1 = self.conv.weight
        output = call([arg0_1, arg1_1])
        return output[0]