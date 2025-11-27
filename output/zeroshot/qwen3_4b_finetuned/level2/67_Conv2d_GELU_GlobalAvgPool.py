import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 4096 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_gelu_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + x0, tmp8, xmask)


@triton.jit
def triton_per_fused_mean_2(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 64
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 16 * x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 16.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x0, tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 8, 256, 256), (524288, 65536, 256, 
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_3, primals_1, [1,
            1], 1, 1, 1, 1, 0)
        buf1 = buf0
        del buf0
        buf2 = buf1
        del buf1
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(16777216)](buf2, primals_2, 
            16777216, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((128, 64, 256, 256), (4194304, 65536, 
            256, 1), torch.float32)
        triton_poi_fused_gelu_1[grid(16777216)](buf2, buf3, 16777216,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf2
        buf4 = empty_strided_cuda((128, 64), (64, 1), torch.float32)
        buf5 = buf4
        del buf4
        triton_per_fused_mean_2[grid(64)](buf5, buf3, 64, 16, XBLOCK=1,
            num_warps=2, num_stages=1)
        del buf3
    return buf5, primals_1, primals_3


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies GELU, and then performs global average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
