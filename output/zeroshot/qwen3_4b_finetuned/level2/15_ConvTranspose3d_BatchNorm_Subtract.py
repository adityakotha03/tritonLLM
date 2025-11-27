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
    xnumel = 338560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 21952 % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_per_fused__native_batch_norm_legit_1(in_ptr0, out_ptr0, out_ptr1,
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
    tmp0 = tl.load(in_ptr0 + (r1 + 64000 * x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 64000, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 64000.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tl.store(out_ptr2 + x0, tmp21, xmask)
    tl.store(out_ptr0 + x0, tmp10, xmask)
    tl.store(out_ptr1 + x0, tmp16, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_sub_2(in_ptr0, in_ptr1,
    in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 512000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex // 64000
    x4 = xindex % 64000
    x0 = xindex % 64000
    x2 = xindex // 1024000
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x4 + 64000 * x3), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + x2, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x2, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 64000.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 - tmp10
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr0 + x5, tmp13, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (32, 16, 3, 3, 3), (432, 27, 9, 3, 1))
    assert_size_stride(primals_2, (32,), (1,))
    assert_size_stride(primals_3, (16, 16, 16, 32, 32), (131072, 8192, 512,
        32, 1))
    assert_size_stride(primals_4, (32,), (1,))
    assert_size_stride(primals_5, (32,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(reinterpret_tensor(
            primals_3, (1, 16, 16, 32, 32), (131072, 8192, 512, 32, 1), 0),
            primals_1, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1,
            1), transposed=True, output_padding=(0, 0, 0), groups=1,
            bias=None)
        buf1 = buf0
        del buf0
        buf2 = reinterpret_tensor(buf1, (16, 32, 1, 1, 1), (0, 1, 1024000, 
            1024000, 1024000), 0)
        del buf1
        buf3 = empty_strided_cuda((16, 32, 1, 1, 1), (32, 1, 1, 1, 1), torch
            .float32)
        buf4 = empty_strided_cuda((16, 32, 1, 1, 1), (32, 1, 1, 1, 1), torch
            .float32)
        buf5 = empty_strided_cuda((16, 32, 1, 1, 1), (32, 1, 1, 1, 1), torch
            .float32)
        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_1[grid(16)](buf2, buf3,
            buf4, buf5, 16, 64000, XBLOCK=1, num_warps=2, num_stages=1)
        buf6 = empty_strided_cuda((16, 32, 16, 32, 32), (1024000, 32000, 
            2000, 64, 1), torch.float32)
        triton_poi_fused__native_batch_norm_legit_sub_2[grid(512000)](buf2,
            buf3, buf5, primals_4, primals_5, buf6, 512000, XBLOCK=512,
            num_warps=4, num_stages=1)
        del buf2
        del buf3
        del buf5
        del primals_5
    return buf6, primals_1, primals_2, primals_4, reinterpret_tensor(
        primals_3, (1, 16, 16, 32, 32), (131072, 8192, 512, 32, 1), 0
        ), buf4


class ModelNew(nn.Module):
    """
    A 3D convolutional transpose layer followed by Batch Normalization and subtraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_4 = self.batch_norm.weight
        primals_5 = self.batch_norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
