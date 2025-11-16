import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 256 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 256 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (257 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (514 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp1 - tmp8
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 - tmp8
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = tmp5 - tmp8
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19 / tmp7
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tl.store(out_ptr0 + x0, tmp8, xmask)
    tl.store(out_ptr1 + x0, tmp23, xmask)
    tl.store(out_ptr2 + x0, tmp22, xmask)


@triton.jit
def triton_poi_fused_convolution_mean_mul_1(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 64
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.sum(tmp3, 0)[:, None]
    tmp6 = 4.0
    tmp7 = tmp5 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x2, tmp7, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_2, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_3, (16, 64, 128, 128), (1048576, 16384, 128,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(2, 
            2), padding=(1, 1), dilation=(1, 1), transposed=True,
            output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf0, (16, 128, 131, 131), (2131360, 16384, 163,
            1))
        buf1 = empty_strided_cuda((16, 128, 131, 131), (2131360, 16384, 163,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_0[grid(512)](buf0, buf1,
            buf2, buf1, 512, XBLOCK=256, num_warps=4, num_stages=1)
        del buf2
        buf3 = extern_kernels.convolution(buf1, primals_2, stride=(2, 2),
            padding=(1, 1), dilation=(1, 1), transposed=True,
            output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf3, (16, 128, 133, 133), (2763856, 16384, 133,
            1))
        buf4 = empty_strided_cuda((16, 128), (128, 1), torch.float32)
        triton_poi_fused_convolution_mean_mul_1[grid(128)](buf4, primals_3,
            128, XBLOCK=128, num_warps=4, num_stages=1)
        buf5 = buf3
        del buf3
        triton_poi_fused_convolution_mean_mul_1[grid(128)](buf5, primals_3,
            128, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_3
    return buf5, primals_1, primals_2, buf0, buf1, buf4


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, multiplies by a scalar, applies global average pooling, 
    another global average pooling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, 
            kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.weight
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
