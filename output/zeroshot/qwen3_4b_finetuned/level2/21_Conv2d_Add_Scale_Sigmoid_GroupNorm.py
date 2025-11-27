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
def triton_poi_fused_add_mul_sigmoid_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 65536 % 32
    x0 = xindex % 65536
    x4 = xindex // 65536
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 65536 * x4), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.sigmoid(tmp4)
    tl.store(out_ptr0 + x3, tmp5, xmask)


@triton.jit
def triton_per_fused_native_group_norm_1(in_ptr0, out_ptr0, out_ptr1,
    xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 32
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 128 * x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 128.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tl.store(out_ptr0 + x0, tmp10, xmask)
    tl.store(out_ptr1 + x0, tmp21, xmask)


@triton.jit
def triton_poi_fused_native_group_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex // 4096
    x4 = xindex % 4096
    x5 = xindex
    x6 = xindex // 128
    x7 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x4 + 131072 * x3), xmask)
    tmp1 = tl.load(in_ptr1 + x6, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x6, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + x7, xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + x7, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 128.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + x5, tmp13, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7) = args
    args.clear()
    assert_size_stride(primals_1, (32, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_2, (32,), (1,))
    assert_size_stride(primals_3, (128, 8, 256, 256), (524288, 65536, 256, 1
        ))
    assert_size_stride(primals_4, (32, 1, 1), (1, 1, 1))
    assert_size_stride(primals_5, (32, 1, 1), (1, 1, 1))
    assert_size_stride(primals_6, (32,), (1,))
    assert_size_stride(primals_7, (32,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_3, primals_1, [1,
            1], dilation=[1, 1], transposed=False, output_padding=[0, 0],
            groups=1, bias=None)
        assert_size_stride(buf0, (128, 32, 254, 254), (2097152, 65536, 256, 1
            ))
        buf1 = empty_strided_cuda((128, 32, 254, 254), (2097152, 65536, 256,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_0[grid(131072)](buf0, primals_2,
            primals_4, buf1, 131072, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
        del primals_4
        buf2 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 128, 128), torch.
            float32)
        buf3 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 128, 128), torch.
            float32)
        triton_per_fused_native_group_norm_1[grid(32)](buf1, buf2, buf3, 32,
            128, XBLOCK=32, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((128, 32, 1, 1), (32, 1, 1, 1), torch.float32
            )
        triton_poi_fused_native_group_norm_2[grid(4096)](buf1, buf2, buf3,
            primals_5, primals_6, buf4, 4096, XBLOCK=256, num_warps=4,
            num_stages=1)
        del buf2
        del buf3
        del primals_6
        buf5 = reinterpret_tensor(buf0, (128, 32, 254, 254), (2097152, 65536,
            256, 1), 0)
        del buf0
        triton_poi_fused_add_mul_sigmoid_0[grid(131072)](buf4, primals_7,
            primals_5, buf5, 131072, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_7
        del primals_5
    return buf5, primals_1, primals_3, buf1, buf4


class ModelNew(nn.Module):
    """
    Model that performs a convolution, adds a bias term, scales, applies sigmoid, and performs group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups,
        bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_4 = self.bias
        primals_5 = self.scale
        primals_6 = self.group_norm.weight
        primals_7 = self.group_norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7])
        return output[0]
