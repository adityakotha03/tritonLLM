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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_per_fused_clamp_min_min_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 132832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = triton_helpers.minimum(tmp0, tmp4)
    tmp6 = tmp5
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp9 = triton_helpers.minimum(tmp8, tmp2)
    tl.store(in_out_ptr0 + x0, tmp9, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (128, 3, 16, 64, 64), (196608, 65536, 4096,
        64, 1))
    assert_size_stride(primals_2, (16, 3, 3, 3, 3), (243, 81, 27, 9, 1))
    assert_size_stride(primals_3, (16,), (1,))
    assert_size_stride(primals_4, (8,), (1,))
    assert_size_stride(primals_5, (8,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 
            1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 16, 16, 64, 64), (2097152, 131072, 
            8192, 128, 2))
        buf1 = empty_strided_cuda((128, 16, 16, 64, 64), (2097152, 131072, 
            8192, 128, 2), torch.float32)
        get_raw_stream(0)
        triton_per_fused_clamp_min_min_0[grid(132832)](buf1, primals_3, 
            132832, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_3
        buf2 = empty_strided_cuda((128, 16, 16, 64, 64), (2097152, 131072, 
            8192, 128, 2), torch.float32)
        extern_kernels.convolution(buf0, primals_4, stride=(1, 1, 1),
            padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=8, bias=None)
        buf3 = empty_strided_cuda((128, 16, 16, 64, 64), (2097152, 131072, 
            8192, 128, 2), torch.float32)
        extern_kernels.convolution(buf2, primals_5, stride=(1, 1, 1),
            padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=8, bias=None)
        buf4 = reinterpret_tensor(buf2, (128, 16, 16, 64, 64), (2097152, 
            131072, 8192, 128, 2), 0)
        del buf2
        triton_per_fused_clamp_min_min_0[grid(132832)](buf4, primals_5, 
            132832, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_5
    return buf4, primals_1, primals_2, primals_4, buf0, buf1, buf3


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, minimum, clamp, and dropout.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value,
        max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_0):
        primals_2 = self.conv.weight
        primals_3 = self.conv.bias
        primals_4 = self.norm.weight
        primals_5 = self.norm.bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
