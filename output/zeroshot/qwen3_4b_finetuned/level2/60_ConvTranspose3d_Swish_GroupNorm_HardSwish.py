import torch
from torch._inductor.select_algorithm import extern_kernels
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
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 5308416 % 16
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_per_fused_native_group_norm_sigmoid_sigmoid_mul_1(in_ptr0,
    out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = xindex % 8
    x3 = xindex // 8
    tmp0 = tl.load(in_ptr0 + (r1 + 16 * x0), xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp1 * tmp0
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 16, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 16.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tl.store(out_ptr0 + x0, tmp12, xmask)
    tl.store(out_ptr1 + x0, tmp23, xmask)


@triton.jit
def triton_poi_fused_hardtanh_hardtanh_backward_2(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = 3.0
    tmp7 = tmp5 * tmp6
    tmp8 = 0.0
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = triton_helpers.minimum(tmp9, tmp6)
    tmp11 = tmp2 - tmp5
    tmp12 = tmp11 * tmp6
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = 0.7071067811865476
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16 * tmp13
    tmp18 = tmp10 * tmp13
    tmp19 = tmp17 - tmp18
    tl.store(out_ptr0 + x0, tmp10, xmask)
    tl.store(out_ptr1 + x0, tmp19, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (16,), (1,))
    assert_size_stride(primals_3, (128, 3, 16, 32, 32), (49152, 16384, 512,
        16, 1))
    assert_size_stride(primals_4, (16,), (1,))
    assert_size_stride(primals_5, (16,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(reinterpret_tensor(primals_3, (1,
            3, 5308416), (16384064, 5308416, 1), 0), primals_1, stride=(2, 
            2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=4, bias=None)
        assert_size_stride(buf0, (128, 16, 5308416), (85734656, 5308416, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(134217728)](buf1, primals_2, 
            134217728, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((128, 16), (16, 1), torch.float32)
        buf3 = empty_strided_cuda((128, 16), (16, 1), torch.float32)
        triton_per_fused_native_group_norm_sigmoid_sigmoid_mul_1[grid(128)](buf1
            , buf2, buf3, 128, 16, XBLOCK=1, num_warps=2, num_stages=1)
        buf4 = empty_strided_cuda((128, 16, 16, 16, 16), (4194304, 262144,
            16384, 1024, 64), torch.float32)
        buf5 = empty_strided_cuda((128, 16, 16, 16, 16), (4194304, 262144,
            16384, 1024, 64), torch.float32)
        triton_poi_fused_hardtanh_hardtanh_backward_2[grid(134217728)](buf2,
            buf3, buf1, primals_5, buf4, buf5, 134217728, XBLOCK=512,
            num_warps=8, num_stages=1)
        del buf2
        del buf3
        del primals_5
    return buf5, primals_1, reinterpret_tensor(primals_3, (1, 3, 5308416),
        (16384064, 5308416, 1), 0), buf1, buf4


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies Swish activation, 
    group normalization, and then HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_4 = self.group_norm.weight
        primals_5 = self.group_norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
