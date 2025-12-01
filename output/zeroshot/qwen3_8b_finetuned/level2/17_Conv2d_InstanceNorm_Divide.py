import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_div_0(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 2496384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex // 15876 % 128
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tmp0 + tmp1
    tmp6 = tmp5 - tmp2
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp10 = tl.sum(tmp9, 0)[:, None]
    tmp11 = tl.full([XBLOCK, 1], 15876, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 15875.0
    tmp14 = tmp12 / tmp13
    tmp15 = tmp10 * tmp14
    tmp16 = tmp6 * tmp6
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp19 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp20 = tl.sum(tmp19, 0)[:, None]
    tmp21 = tmp20 * tmp14
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp6 * tmp24
    tmp26 = tmp25 * tmp3
    tmp27 = tmp26 + tmp4
    tl.store(in_out_ptr0 + x0, tmp27, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (128, 128, 126, 126), (2032128, 15876, 
        126, 1))
    assert_size_stride(primals_4, (128,), (1,))
    assert_size_stride(primals_5, (128,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.convolution(primals_3, primals_1,
            stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 128, 126, 126), (2032128, 15876, 126,
            1))
        buf1 = buf0
        del buf0
        buf6 = empty_strided_cuda((128, 128, 126, 126), (2032128, 1, 15876,
            126), torch.float32)
        buf7 = empty_strided_cuda((128, 128, 126, 126), (2032128, 1, 15876,
            126), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_convolution_div_0[grid(2496384)](
            buf1, primals_2, primals_2, primals_4, primals_5, 2496384,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        del primals_4
        del primals_5
    return buf1, primals_1, primals_3, buf6, buf7


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies Instance Normalization, and divides by a constant.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_4 = self.instance_norm.weight
        primals_5 = self.instance_norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]