import torch
from torch._inductor.select_algorithm import extern_kernels
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
def triton_poi_fused_add_clamp_div_hardtanh_mul_log_sigmoid_logsumexp_0(
    in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (16 + x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (32 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (48 + x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (64 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (80 + x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (96 + x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (112 + x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr1 + 0)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tmp0 + tmp17
    tmp19 = tmp1 + tmp18
    tmp20 = libdevice.log(tmp19)
    tmp21 = tmp18 - tmp20
    tmp22 = tmp2 + tmp21
    tmp23 = 3.0
    tmp24 = tmp22 + tmp23
    tmp25 = tl.sigmoid(tmp24)
    tmp26 = tmp21 * tmp25
    tmp27 = 6.0
    tmp28 = tmp26 / tmp27
    tmp31 = tmp28 - tmp30
    tmp32 = -1.0
    tmp33 = triton_helpers.minimum(tmp31, tmp32)
    tmp34 = 1.0
    tmp35 = triton_helpers.maximum(tmp33, tmp34)
    tl.store(in_out_ptr0 + x2, tmp35, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 16, 16, 32, 32), (16384, 1024, 64, 2,
        1))
    assert_size_stride(primals_2, (16, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_3, (1, 1, 1, 1), (1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(2, 
            2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 16, 16, 32, 32), (16384, 1024, 64, 2,
            1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_add_clamp_div_hardtanh_mul_log_sigmoid_logsumexp_0[grid
            (1048576)](buf1, primals_1, primals_3, 1048576, XBLOCK=256,
            num_warps=4, num_stages=1)
        del primals_1
        del primals_3
    return buf1, primals_2


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, LogSumExp, HardSwish, subtraction, clamp operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1)) 

    def forward(self, input_0):
        primals_2 = self.conv_transpose.weight
        primals_3 = self.bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]