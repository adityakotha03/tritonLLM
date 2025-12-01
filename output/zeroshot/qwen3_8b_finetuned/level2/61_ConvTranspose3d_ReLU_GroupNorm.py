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
def triton_poi_fused_convolution_relu_0(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_group_norm_1(in_out_ptr0, in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 128 % 34
    x0 = xindex % 128
    x2 = xindex // 4624
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + x2, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp4 = tmp0 + tmp1
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp6 - tmp2
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.sum(tmp9, 0)[:, None]
    tmp12 = 255.0
    tmp13 = tmp11 / tmp12
    tmp14 = tmp7 - tmp13
    tmp15 = 1e-05
    tmp16 = tmp14 * tmp14
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp19 = tl.sum(tmp17, 0)[:, None]
    tmp20 = tmp19 / tmp12
    tmp21 = tmp13 + tmp15
    tmp22 = tmp21 + tmp20
    tmp23 = 0.5
    tmp24 = tmp22 * tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp14 * tmp25
    tmp27 = tmp26 * tmp3
    tl.store(in_out_ptr0 + x3, tmp27, xmask)
    tl.store(out_ptr0 + (x1 + 34 * x0), tmp25, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (16, 128, 34, 34, 34), (5331936, 4096, 156,
        4.617914583333333, 1.3743483333333333))
    assert_size_stride(primals_2, (128, 64, 3, 3, 3), (1728, 27, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 
            1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (16, 128, 34, 34, 34), (5331936, 4096, 156,
            4.617914583333333, 1.3743483333333333))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_relu_0[grid(6291456)](buf1, primals_2, 
            6291456, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((16, 128, 34, 34, 34), (5331936, 1, 156,
            4.617914583333333, 1.3743483333333333), torch.float32)
        buf3 = empty_strided_cuda((16, 128, 34, 34, 34), (5331936, 1, 156,
            4.617914583333333, 1.3743483333333333), torch.float32)
        triton_poi_fused_convolution_group_norm_1[grid(6291456)](buf1,
            primals_2, buf2, buf3, buf3, 6291456, XBLOCK=256, num_warps=4,
            num_stages=1)
        del buf2
        del primals_2
    return reinterpret_tensor(buf1, (16, 128, 34, 34, 34), (5331936, 1, 156,
        4.617914583333333, 1.3743483333333333), 0), reinterpret_tensor(buf3,
        (16, 128, 34, 34, 34), (5331936, 1, 156, 4.617914583333333,
        1.3743483333333333), 0)


class ModelNew(nn.Module):
    """
    Model that performs a transposed 3D convolution, applies ReLU, and then applies group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

    def forward(self, input_0):
        primals_2 = self.conv_transpose.weight
        primals_1 = input_0
        output = call([primals_1, primals_2])
        return output[0]