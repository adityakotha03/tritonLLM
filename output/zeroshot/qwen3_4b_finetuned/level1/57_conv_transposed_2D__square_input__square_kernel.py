import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_per_fused__unsafe_index_convolution_0(in_out_ptr0, in_ptr0,
    out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 12
    rnumel = 384
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 128 * x0), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (128 + r1 + 128 * x0), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr0 + (256 + r1 + 128 * x0), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr0 + (384 + r1 + 128 * x0), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr0 + (r1 + 128 * x0), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr0 + (128 + r1 + 128 * x0), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr0 + (256 + r1 + 128 * x0), rmask & xmask, other=0.0)
    tmp22 = tl.load(in_ptr0 + (384 + r1 + 128 * x0), rmask & xmask, other=0.0)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp5 = tmp6 + tmp4
    tmp7 = tmp17 + tmp16
    tmp8 = tmp7 + tmp1
    tmp9 = tmp19 + tmp8
    tmp10 = tmp22 + tmp9
    tmp11 = tmp5 + tmp10
    tmp12 = tmp1 + tmp7
    tmp13 = tmp12 + tmp5
    tmp14 = tmp22 + tmp12
    tmp15 = tmp14 + tmp10
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp16
    tmp21 = tmp20 + tmp19
    tmp23 = tmp21 + tmp17
    tmp24 = tmp23 + tmp22
    tmp25 = tmp24 + tmp2
    tmp26 = tmp25 + tmp15
    tmp27 = tmp26 + tmp4
    tmp28 = tmp27 + tmp13
    tmp29 = tmp28 + tmp26
    tmp30 = tmp29 + tmp11
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.sum(tmp31, 1)[:, None]
    tmp34 = 3.125 * tmp33
    tmp35 = tmp30 - tmp34
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x0, tmp34, xmask)
    tl.store(out_ptr0 + (r1 + 384 * x0), tmp35, rmask & xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (8, 64, 1024, 1024), (67108864, 1024, 1024, 
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = buf1 = empty_strided_cuda((8, 64, 1025, 1025), (67108864, 
            1024, 1025, 1), torch.float32)
        buf2 = buf0
        del buf0
        get_raw_stream(0)
        triton_per_fused__unsafe_index_convolution_0[grid(12)](buf2,
            primals_1, buf1, 12, 384, XBLOCK=1, num_warps=2, num_stages=1)
        del primals_1
        del primals_2
    return buf1, primals_3, buf2


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, input_0):
        primals_1 = self.conv_transpose2d.weight
        primals_2 = self.conv_transpose2d.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
