import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_convolution_sigmoid_0(in_out_ptr0, in_ptr0,
    in_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 65536 % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tmp0 + tmp1
    tmp4 = tmp3 * tmp2
    tmp5 = tl.sigmoid(tmp4)
    tl.store(in_out_ptr0 + x3, tmp5, xmask)


@triton.jit
def triton_poi_fused_group_norm_1(in_ptr0, out_ptr0, out_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x2 = xindex // 32
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x2), xmask)
    tmp1 = tl.load(in_ptr0 + (32 + x0 + 128 * x2), xmask)
    tmp3 = tl.load(in_ptr0 + (64 + x0 + 128 * x2), xmask)
    tmp5 = tl.load(in_ptr0 + (96 + x0 + 128 * x2), xmask)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp2 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp4 - tmp8
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp6 - tmp13
    tmp15 = tmp14 / tmp7
    tmp16 = 1e-05
    tmp17 = tmp13 + tmp16
    tmp18 = 1.0
    tmp19 = tmp18 / tmp17
    tmp20 = tmp15 * tmp19
    tmp21 = tmp1 * tmp20
    tmp22 = tmp3 * tmp20
    tmp23 = tmp22 + tmp21
    tmp24 = tmp5 * tmp20
    tmp25 = tmp23 + tmp24
    tl.store(out_ptr0 + (x0 + 128 * x1), tmp8, xmask)
    tl.store(out_ptr1 + (x0 + 128 * x1), tmp20, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6 = args
    args.clear()
    assert_size_stride(primals_1, (32, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_2, (32,), (1,))
    assert_size_stride(primals_3, (128, 8, 256, 256), (524288, 65536, 256, 1
        ))
    assert_size_stride(primals_4, (32,), (1,))
    assert_size_stride(primals_5, (32,), (1,))
    assert_size_stride(primals_6, (32,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_3, primals_1,
            stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 32, 254, 254), (2097152, 65536, 256, 1
            ))
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((32, 1, 1), (1, 1, 1), torch.float32)
        buf3 = empty_strided_cuda((32, 1, 1), (1, 1, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_convolution_sigmoid_0[grid(1048576)](buf1,
            primals_2, primals_4, 1048576, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        del primals_4
        buf4 = buf1
        del buf1
        buf5 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 1, 1), torch.float32
            )
        buf6 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 1, 1), torch.float32
            )
        triton_poi_fused_group_norm_1[grid(4096)](primals_5, buf4, buf5,
            4096, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_5
    return buf6, primals_1, primals_3, primals_6, buf2, buf3, buf4, buf5


class ModelNew(nn.Module):
    """
    Model that performs a convolution, adds a bias term, scales, applies sigmoid, and performs group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_4 = self.scale
        primals_5 = self.group_norm.weight
        primals_6 = self.group_norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5,
            primals_6])
        return output[0]
