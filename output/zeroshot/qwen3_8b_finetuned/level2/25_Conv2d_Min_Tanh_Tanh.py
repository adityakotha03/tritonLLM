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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_min_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 4096
    x2 = xindex // 1048576
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 65536 * x2), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (256 * x0 + 1048576 * x2), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (512 * x0 + 1048576 * x2), xmask,
        eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (1024 * x0 + 1048576 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = triton_helpers.minimum(tmp1, tmp3)
    tmp4 = triton_helpers.minimum(tmp2, tmp5)
    tmp6 = triton_helpers.maximum(tmp1, tmp3)
    tmp8 = triton_helpers.maximum(tmp6, tmp5)
    tmp9 = tmp4 + tmp8
    tmp10 = 2.0
    tmp11 = tmp9 / tmp10
    tmp12 = 0.125
    tmp13 = tmp11 * tmp12
    tmp14 = tmp0 - tmp13
    tl.store(out_ptr0 + x3, tmp14, xmask)


@triton.jit
def triton_poi_fused_add_tanh_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 4096
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = libdevice.tanh(tmp2)
    tmp4 = libdevice.tanh(tmp3)
    tl.store(out_ptr0 + x3, tmp4, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_2, (128, 16, 256, 256), (1048576, 65536, 256,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 256, 256), (4194304, 65536, 256,
            1))
        buf1 = empty_strided_cuda((128, 64, 256, 256), (4194304, 65536, 256,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_min_0[grid(536870912)](buf0, buf1, 536870912,
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((128, 64, 256, 256), (4194304, 65536, 256,
            1), torch.float32)
        triton_poi_fused_add_tanh_1[grid(536870912)](buf1, primals_1, buf2,
            536870912, XBLOCK=1024, num_warps=4, num_stages=1)
        del buf1
        del primals_1
    return buf2, reinterpret_tensor(primals_2, (64, 16, 3, 3), (144, 9, 3, 1),
        0)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = input_0
        output = call([primals_1, primals_2])
        return output[0]