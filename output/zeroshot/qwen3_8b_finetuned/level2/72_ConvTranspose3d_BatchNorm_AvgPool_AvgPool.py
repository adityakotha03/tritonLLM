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
def triton_poi_fused_convolution_native_batch_norm_0(in_ptr0, in_ptr1,
    in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = xindex // 128 % 16
    tmp0 = tl.load(in_ptr0 + x0, None)
    tmp1 = tl.load(in_ptr1 + x1, None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp10 = tl.sum(tmp9, 0)[:, None]
    tmp11 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp13 = tl.sum(tmp11, 0)[:, None]
    tmp14 = 128.0
    tmp15 = tmp13 / tmp14
    tmp16 = tmp10 * tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp21 = tl.sum(tmp19, 0)[:, None]
    tmp22 = tmp21 / tmp14
    tmp23 = 1.0
    tmp24 = tmp22 * tmp23
    tmp25 = tmp6 - tmp24
    tl.store(out_ptr0 + x0, tmp25, None)
    tl.store(out_ptr1 + x1, tmp16, None)
    tl.store(out_ptr2 + x1, tmp22, None)


@triton.jit
def triton_poi_fused_avg_pool2d_1(in_ptr0, out_ptr0, out_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = xindex // 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1), None)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128 * x1), None)
    tmp3 = tl.load(in_ptr0 + (x0 + 192 * x1), None)
    tmp5 = tl.load(in_ptr0 + (64 + x0 + 192 * x1), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = 1.0
    tmp10 = tmp8 - tmp9
    tmp11 = 0.0
    tmp12 = tmp10 <= tmp11
    tmp13 = tl.full([1], 0, tl.int32)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = triton_helpers.maximum(tmp12, tmp13)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tl.store(out_ptr0 + x2, tmp17, None)
    tl.store(out_ptr1 + x2, tmp8, None)


@triton.jit
def triton_poi_fused_avg_pool2d_2(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = xindex // 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 128 * x1), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.load(in_ptr0 + (64 + x0 + 128 * x1), None)
    tmp4 = tl.load(in_ptr1 + (64 + x0 + 128 * x1), None)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.load(in_ptr0 + (x0 + 192 * x1), None)
    tmp8 = tl.load(in_ptr1 + (x0 + 192 * x1), None)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr0 + (64 + x0 + 192 * x1), None)
    tmp11 = tl.load(in_ptr1 + (64 + x0 + 192 * x1), None)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = tmp6 + tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tl.store(out_ptr0 + x2, tmp16, None)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (64, 3, 32, 32, 32), (98304, 32768, 1024,
        32, 1))
    assert_size_stride(primals_3, (16,), (1,))
    assert_size_stride(primals_4, (16,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(2, 
            2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (64, 16, 64, 64, 64), (4194304, 262144, 
            4096, 64, 1))
        buf1 = empty_strided_cuda((64, 16, 64, 64, 64), (4194304, 262144, 
            4096, 64, 1), torch.float32)
        buf2 = empty_strided_cuda((16, 64, 64, 64), (262144, 4096, 64, 1),
            torch.float32)
        buf3 = empty_strided_cuda((16, 64, 64, 64), (262144, 4096, 64, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_native_batch_norm_0[grid(4194304)](buf0,
            primals_3, primals_4, buf2, buf1, buf2, buf3, 4194304, XBLOCK=
            128, num_warps=4, num_stages=1)
        del buf2
        del buf3
        del primals_4
        buf4 = empty_strided_cuda((64, 16, 32, 32, 32), (524288, 32768, 1024,
            32, 1), torch.float32)
        buf5 = empty_strided_cuda((64, 16, 32, 32, 32), (524288, 32768, 1024,
            32, 1), torch.float32)
        triton_poi_fused_avg_pool2d_1[grid(524288)](buf1, buf4, buf5, 524288,
            XBLOCK=128, num_warps=4, num_stages=1)
        buf6 = empty_strided_cuda((64, 16, 16, 16, 16), (4096, 256, 16, 1, 
            1), torch.float32)
        triton_poi_fused_avg_pool2d_2[grid(4096)](buf4, buf5, buf6, 4096,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf4
        del buf5
    return buf6, primals_1, primals_2, buf0, buf1, buf6


class ModelNew(nn.Module):
    """
    A model that performs a 3D transposed convolution, followed by batch normalization, 
    two average pooling layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_3 = self.batch_norm.weight
        primals_4 = self.batch_norm.bias
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]