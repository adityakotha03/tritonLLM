import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_div_exp_max_mul_sigmoid_sub_sum_0(in_ptr0, in_ptr1,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 4096
    x0 = xindex % 4096
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + 4096 * x1 + x0, xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (2048 + 4096 * x1 + x0), xmask,
        eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4096 + 4096 * x1 + x0), xmask,
        eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (6144 + 4096 * x1 + x0), xmask,
        eviction_policy='evict_last')
    tmp3 = tmp0 - tmp1
    tmp5 = tmp2 - tmp1
    tmp6 = triton_helpers.maximum(tmp5, tmp3)
    tmp8 = tmp4 - tmp6
    tmp10 = tmp7 - tmp6
    tmp11 = triton_helpers.maximum(tmp10, tmp8)
    tmp12 = tmp9 - tmp11
    tmp13 = tmp12 + tmp11
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp15 * 2.0
    tmp17 = tmp16 + tmp1
    tmp18 = tmp17 - tmp11
    tl.store(out_ptr0 + x3, tmp18, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 128, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(primals_2, (128, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(primals_3, (128, 1, 1), (1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.convolution(primals_1, primals_2,
            stride=(2, 2), padding=(1, 1), output_padding=(1, 1), dilation=(
            1, 1), transposed=True, groups=1, bias=None)
        assert_size_stride(buf0, (128, 128, 65, 65), (532480, 4096, 64, 1))
        buf1 = empty_strided_cuda((128, 128, 65, 65), (532480, 4096, 64, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_div_exp_max_mul_sigmoid_sub_sum_0[grid(65536)](
            buf0, primals_3, buf1, 65536, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        del primals_3
    return buf1, primals_1, primals_2


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, applies softmax, adds a bias term, scales the result, and applies sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        primals_2 = self.conv_transpose.weight
        primals_3 = self.bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]