import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_group_norm_hardswish_hardtanh_0(in_out_ptr0,
    in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 55296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x1 = xindex // 4096 % 16
    x2 = xindex // 65536
    x3 = xindex % 4096
    x5 = xindex // 16
    tmp0 = tl.load(in_out_ptr0 + x4, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x2, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + x5, xmask, eviction_policy='evict_last')
    tmp3 = tmp0 + tmp1
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 + tmp2
    tmp8 = 4.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = 1.0
    tmp14 = tmp13 / tmp12
    tmp15 = tmp5 * tmp14
    tmp16 = 3.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp7 - tmp10
    tmp19 = tmp18 * tmp14
    tmp20 = tmp17 + tmp19
    tmp21 = 0.0
    tmp22 = triton_helpers.maximum(tmp20, tmp21)
    tmp23 = 6.0
    tmp24 = triton_helpers.minimum(tmp22, tmp23)
    tmp25 = tmp24 - tmp21
    tmp26 = 0.5
    tmp27 = tmp25 * tmp26
    tmp28 = tmp27 + tmp21
    tl.store(in_out_ptr0 + x4, tmp28, xmask)
    tl.store(in_out_ptr1 + x4, tmp15, xmask)
    tl.store(out_ptr0 + x4, tmp24, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (4, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (16,), (1,))
    assert_size_stride(primals_3, (128, 3, 16, 32, 32), (16384, 5462, 1024,
        32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(2, 
            2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=4, bias=None)
        assert_size_stride(buf0, (128, 16, 16, 34, 34), (112640, 7040, 4480,
            136, 4))
        buf1 = buf0
        del buf0
        buf2 = extern_kernels.convolution(primals_3, primals_1, stride=(2, 
            2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=4, bias=None)
        del primals_1
        buf3 = empty_strided_cuda((16, 16), (16, 1), torch.float32)
        extern_kernels.mm(buf2, primals_2, out=buf3)
        buf4 = empty_strided_cuda((16, 16, 16, 16, 32), (40960, 2048, 128, 8,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_group_norm_hardswish_hardtanh_0[grid(55296
            )](buf1, buf2, buf3, primals_2, buf4, 55296, XBLOCK=256,
            num_warps=4, num_stages=1)
        del buf2
        del primals_2
    return buf4, primals_3, buf1, buf3


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies Swish activation, 
    group normalization, and then HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)

    def forward(self, input_0):
        primals_2 = self.conv_transpose.weight
        primals_1 = self.conv_transpose.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
