import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_hardswish_tanh_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 114688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp4 = tl.load(in_ptr0 + (5120 + x0), xmask)
    tmp5 = tl.load(in_ptr1 + (5120 + x0), xmask)
    tmp8 = tl.load(in_ptr0 + (10240 + x0), xmask)
    tmp9 = tl.load(in_ptr1 + (10240 + x0), xmask)
    tmp2 = tl_math.tanh(tmp1)
    tmp3 = tmp0 + tmp2
    tmp6 = tl_math.tanh(tmp5)
    tmp7 = tmp4 + tmp6
    tmp10 = tl_math.tanh(tmp9)
    tmp11 = tmp8 + tmp10
    tmp12 = tl.where(xmask, tmp3, tmp7)
    tmp13 = tl.where(xmask, tmp3, tmp11)
    tmp14 = 0.0
    tmp15 = tl.full([1], 1, tl.int64)
    tmp16 = triton_helpers.promote_to_tensor(tl.broadcast_to(tmp15, [1]))
    tmp17 = libdevice.log1p(tmp14)
    tmp18 = tmp16 * tmp16
    tmp19 = tmp14 + tmp18
    tmp20 = libdevice.log1p(tmp19)
    tmp21 = tmp17 + tmp20
    tmp22 = tl.where(xmask, tmp13, tmp14)
    tmp23 = tl.where(xmask, tmp12, tmp14)
    tmp24 = tmp23 + tmp22
    tmp25 = libdevice.log1p(tmp24)
    tmp26 = tmp21 + tmp25
    tmp27 = tmp16 * tmp26
    tl.store(out_ptr0 + x0, tmp27, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (16, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_2, (16, 8), (8, 1))
    assert_size_stride(primals_3, (128, 8, 128, 128), (131072, 16384, 128,
        1))
    assert_size_stride(primals_4, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
        1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 126, 126), (10077696, 157921, 
        126, 1))
        buf1 = empty_strided_cuda((128, 64, 126, 126), (10077696, 157921, 
            126, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_hardswish_tanh_0[grid(114688)](buf0, primals_2,
            buf1, 114688, XBLOCK=512, num_warps=8, num_stages=1)
    return buf1, primals_1, primals_2, primals_3, primals_4, buf0


class ModelNew(nn.Module):
    """
    Model that performs a convolution, applies Group Normalization, Tanh, HardSwish, 
    Residual Addition, and LogSumExp.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = input_0
        primals_4 = self.group_norm.weight
        primals_5 = self.group_norm.bias
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
