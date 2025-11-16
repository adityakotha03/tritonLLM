import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_mul_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 87479680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 2359296 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_div_mean_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1126400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 10240 % 10240
    x0 = xindex % 10240
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (x1 + 102400), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (x0 + 102400), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (10240 + x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (20480 + x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (30720 + x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 + tmp9
    tmp8 = tmp7 + tmp13
    tmp10 = 1.0
    tmp11 = tmp8 * tmp10
    tmp12 = 10240.0
    tmp14 = tmp11 / tmp12
    tmp15 = tmp8 / tmp12
    tl.store(out_ptr0 + x2, tmp15, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 5, 5, 5), (8000, 128, 256, 51,
        1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (16, 64, 16, 32, 32), (32768, 512, 16, 4,
        1))
    assert_size_stride(primals_4, (128,), (1,))
    assert_size_stride(primals_5, (128,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1, 1), padding=(2, 2, 2), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (16, 128, 17, 35, 35), (852800, 64, 21200,
            616, 1))
        buf1 = empty_strided_cuda((128,), (1,), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(87479680)](buf0, primals_2, 87479680,
            XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((16, 128, 17, 35, 35), (852800, 64, 21200,
            616, 1), torch.float32)
        extern_kernels.batch_norm(buf0, primals_4, buf1, primals_5, 1.0, 
            0.0, True, 0, 0)
        del primals_5
        buf3 = empty_strided_cuda((16, 128, 1, 1, 1), (128, 1, 128, 128, 
            128), torch.float32)
        triton_poi_fused_add_div_mean_1[grid(1126400)](buf2, buf3, 1126400,
            XBLOCK=256, num_warps=4, num_stages=1)
    return buf3, primals_1, primals_3, reinterpret_tensor(buf2, (16, 128, 1,
        1, 1), (128, 1, 128, 128, 1), 0), buf1


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, scales the output, applies batch normalization, 
    and then performs global average pooling. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_4 = self.batch_norm.weight
        primals_5 = self.batch_norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
