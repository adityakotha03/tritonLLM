import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_hardtanh_hardtanh_backward_0(in_ptr0,
    in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 12969 % 128
    x0 = xindex % 12969
    x4 = xindex // 12969
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 20.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = 0.5
    tmp8 = tmp6 - tmp7
    tmp9 = 0.0
    tmp10 = tmp8 > tmp9
    tmp11 = 1.0
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12 * tmp11
    tmp14 = tl.where(tmp10, tmp13, tmp9)
    tmp15 = tmp14 - tmp7
    tmp16 = tmp15 * tmp14
    tl.store(out_ptr0 + (x0 + 13024 * x4), tmp16, xmask)
    tl.store(out_ptr1 + x3, tmp10, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_mish_1(in_ptr0, in_ptr1,
    out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 393152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = xindex // 64 % 63
    x2 = xindex // 4032
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 256 * x1 + 13024 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 20.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp6 - tmp3
    tmp8 = tmp7 > tmp3
    tmp9 = 1.0
    tmp10 = tmp7 * tmp9
    tmp11 = tmp10 * tmp10
    tmp12 = tmp11 * tmp10
    tmp13 = tl.where(tmp8, tmp12, tmp3)
    tmp14 = tmp13 * tmp6
    tl.store(out_ptr0 + x3, tmp13, xmask)
    tl.store(out_ptr1 + x3, tmp14, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (128, 64, 128, 128), (1048576, 16384, 128,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 128, 127, 127), (2097152, 16384, 127,
            1))
        buf1 = empty_strided_cuda((128, 128, 127, 127), (2097152, 16384, 
            127, 1), torch.float32)
        buf2 = empty_strided_cuda((128, 128, 127, 127), (2097152, 16384, 
            127, 1), torch.bool)
        get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardtanh_backward_0[grid(786432)
            ](buf0, primals_2, buf1, buf2, 786432, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del buf0
        del primals_2
        buf3 = empty_strided_cuda((128, 128, 63, 63), (50176, 393152, 63, 1
            ), torch.float32)
        buf4 = empty_strided_cuda((128, 128, 63, 63), (50176, 393152, 63, 1
            ), torch.float32)
        triton_poi_fused_max_pool2d_with_indices_mish_1[grid(393152)](buf1,
            primals_3, buf3, buf4, 393152, XBLOCK=512, num_warps=8, num_stages=1
            )
        del buf1
        del primals_3
    return buf4, primals_1, buf2, reinterpret_tensor(buf3, (128, 128, 63, 
        63), (50176, 393152, 63, 1), 0)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtracts a value, applies HardSwish, MaxPool, and Mish activation functions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
