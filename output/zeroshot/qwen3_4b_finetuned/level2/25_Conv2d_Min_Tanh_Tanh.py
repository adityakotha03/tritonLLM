import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16384 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_minimum_tanh_tanh_1(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = -1.0
    tmp2 = tmp0 < tmp1
    tmp3 = tl.where(tmp2, tmp0, tmp1)
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = 1.0
    tmp7 = tmp3 * tmp6
    tmp8 = tmp7 - tmp4
    tmp9 = 1.0000000000000002
    tmp10 = tmp8 * tmp9
    tmp11 = 0.0
    tmp12 = tmp10 > tmp11
    tmp13 = 0.7615941559557649
    tmp14 = tmp10 * tmp13
    tmp15 = tl.where(tmp12, tmp14, tmp11)
    tmp16 = tmp15 > tmp4
    tmp17 = tmp15 * tmp9
    tmp18 = tmp17 - tmp4
    tmp19 = tmp18 * tmp13
    tmp20 = tl.where(tmp16, tmp19, tmp11)
    tl.store(out_ptr0 + x0, tmp20, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 16, 256, 256), (1048576, 65536, 256,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_3, primals_1, [1,
            1, 1], 1, 0, 0, 1, 1)
        buf1 = buf0
        del buf0
        buf2 = buf1
        del buf1
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(131072)](buf2, primals_2, 
            131072, XBLOCK=512, num_warps=4, num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((128, 64, 256, 256), (4194304, 65536, 256,
            1), torch.float32)
        triton_poi_fused_minimum_tanh_tanh_1[grid(131072)](buf2, buf3, 
            131072, XBLOCK=512, num_warps=4, num_stages=1)
        del buf2
    return buf3, primals_1, primals_3


class ModelNew(nn.Module):
    """
    Model that performs a convolution, applies minimum operation, Tanh, and another Tanh.
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
