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
def triton_poi_fused_add_hardtanh_hardswish_mish_max_pool2d_with_indices_0(
    in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 262144096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 - tmp2
    tmp4 = 6.0
    tmp5 = tmp3 / tmp4
    tmp6 = -3.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 3.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = tmp9 + tmp6
    tmp11 = tmp9 * tmp10
    tmp12 = 0.5
    tmp13 = tmp3 * tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tmp14 * tmp12
    tmp16 = tmp13 + tmp15
    tmp17 = tmp16 * tmp12
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp20 = tl.where(xmask, tmp18, 0)
    tl.store(out_ptr0 + x0, tmp17, xmask)
    tl.store(out_ptr1 + x0, tmp20, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (128, 64, 128, 128), (1048576, 16384, 128,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 128, 128, 128), (2097152, 16384, 128,
            1))
        buf1 = empty_strided_cuda((128, 128, 128, 128), (2097152, 1, 8192,
            64), torch.float32)
        buf3 = empty_strided_cuda((128, 128, 128, 128), (2097152, 1, 8192, 
            64), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_hardtanh_hardswish_mish_max_pool2d_with_indices_0[
            grid(262144096)](buf0, primals_2, buf1, buf3, 262144096,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        del primals_2
    return buf3, primals_1, primals_3, buf1


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