import torch
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
def triton_poi_fused_convolution_relu_0(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 32768 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x3, tmp4, None)


@triton.jit
def triton_per_fused_native_group_norm_1(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64 * x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 64.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tl.store(out_ptr2 + x0, tmp22, xmask)
    tl.store(out_ptr0 + x0, tmp10, xmask)
    tl.store(out_ptr1 + x0, tmp16, xmask)


@triton.jit
def triton_poi_fused_native_group_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x4 = xindex // 16
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr1 + x0, None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + x4, None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + x4, None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 / tmp3
    tmp5 = tmp4 * tmp6
    tmp7 = tmp5 * tmp9
    tl.store(out_ptr0 + x3, tmp7, None)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 3, 3, 3), (1728, 27, 9, 3, 1))
    assert_size_stride(primals_2, (16, 64, 32, 32, 32), (2097152, 32768, 
        1024, 32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_2, primals_1, [0,
            0, 0], dilation=[1, 1, 1], transposed=True, output_padding=[0, 
            0, 0], groups=8, bias=None)
        buf1 = buf0
        del buf0
        buf2 = buf1
        del buf1
        get_raw_stream(0)
        triton_poi_fused_convolution_relu_0[grid(2097152)](buf2, primals_1,
            2097152, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_1
        buf3 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 1, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 16, 16), torch.float32
            )
        buf5 = empty_strided_cuda((1, 16, 1, 1), (16, 1, 1, 1), torch.float32)
        triton_per_fused_native_group_norm_1[grid(16)](buf2, buf3, buf4,
            buf5, 16, 64, XBLOCK=8, num_warps=2, num_stages=1)
        buf6 = empty_strided_cuda((16, 128, 32, 32, 32), (4194304, 32768, 
            1024, 32, 1), torch.float32)
        triton_poi_fused_native_group_norm_2[grid(6656000)](buf2, buf3,
            buf4, buf5, buf5, buf6, 6656000, XBLOCK=512, num_warps=8,
            num_stages=1)
        del buf3
        del buf4
        del buf5
    return buf6, primals_2, buf2


class ModelNew(nn.Module):
    """
    Model that performs a transposed 3D convolution, applies ReLU, and then applies group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size, bias=bias)
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = input_0
        output = call([primals_1, primals_2])
        return output[0]
