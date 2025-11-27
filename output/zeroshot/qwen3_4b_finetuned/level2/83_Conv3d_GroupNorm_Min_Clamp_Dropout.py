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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 1536000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 153600 % 16
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_per_fused_native_group_norm_1(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 128
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
def triton_poi_fused_clamp_min_max_minimum_2(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1536000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = 1.0
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tl.store(out_ptr0 + x0, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (16,), (1,))
    assert_size_stride(primals_3, (128, 3, 16, 64, 64), (196608, 65536, 4096,
        64, 1))
    assert_size_stride(primals_4, (16,), (1,))
    assert_size_stride(primals_5, (16,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 16, 16, 64, 64), (1536000, 95616,
            5976, 96, 1), torch.float32)
        buf1 = reinterpret_tensor(buf0, (128, 16, 16, 64, 64), (1536000, 
            95616, 5976, 96, 1), 0)
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(1536000)](buf1, primals_1, 
            1536000, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_1
        buf2 = empty_strided_cuda((1, 128, 1, 1, 1), (128, 1, 1, 1, 1),
            torch.float32)
        buf3 = empty_strided_cuda((1, 128, 1, 1, 1), (128, 1, 128, 128, 
            128), torch.float32)
        buf4 = empty_strided_cuda((1, 128, 1, 1, 1), (128, 1, 1, 1, 1),
            torch.float32)
        triton_per_fused_native_group_norm_1[grid(128)](buf1, buf2, buf3,
            buf4, 128, 64, XBLOCK=32, num_warps=4, num_stages=1)
        del buf1
        buf5 = empty_strided_cuda((128, 16, 16, 64, 64), (1536000, 95616,
            5976, 96, 1), torch.float32)
        triton_poi_fused_clamp_min_max_minimum_2[grid(1536000)](buf3, buf5,
            1536000, XBLOCK=512, num_warps=8, num_stages=1)
        del buf3
    return buf5, primals_2, primals_3, primals_4, primals_5, buf2, buf4


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, minimum, clamp, and dropout.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_4 = self.norm.weight
        primals_5 = self.norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
