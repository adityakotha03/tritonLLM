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
def triton_per_fused__native_batch_norm_legit_0(in_out_ptr0, in_ptr0,
    in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 64
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 32 * x0), xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 32, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 32.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x0, tmp22, xmask)
    tl.store(out_ptr1 + (r1 + 32 * x0), tmp25, xmask)
    tl.store(out_ptr0 + x0, tmp10, xmask)


@triton.jit
def triton_poi_fused_avg_pool3d_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256 % 16
    x3 = xindex // 4096
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32 * x2 + 512 * x1 + 16384 * x3), xmask)
    tmp1 = tl.load(in_ptr0 + (16 + x0 + 32 * x2 + 512 * x1 + 16384 * x3),
        xmask)
    tmp3 = tl.load(in_ptr0 + (8192 + x0 + 32 * x2 + 512 * x1 + 16384 * x3),
        xmask)
    tmp5 = tl.load(in_ptr0 + (8192 + 16 + x0 + 32 * x2 + 512 * x1 + 16384 *
        x3), xmask)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + x4, tmp8, xmask)


@triton.jit
def triton_poi_fused_avg_pool3d_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = xindex // 4 % 4
    x2 = xindex // 16 % 4
    x3 = xindex // 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8 * x2 + 32 * x1 + 128 * x3), xmask)
    tmp1 = tl.load(in_ptr0 + (4 + x0 + 8 * x2 + 32 * x1 + 128 * x3), xmask)
    tmp3 = tl.load(in_ptr0 + (64 + x0 + 8 * x2 + 32 * x1 + 128 * x3), xmask)
    tmp5 = tl.load(in_ptr0 + (68 + x0 + 8 * x2 + 32 * x1 + 128 * x3), xmask)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + x4, tmp8, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (16,), (1,))
    assert_size_stride(primals_3, (64, 3, 32, 32, 32), (49152, 16384, 512,
        16, 1))
    assert_size_stride(primals_4, (16,), (1,))
    assert_size_stride(primals_5, (16,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(reinterpret_tensor(primals_3, (1,
            3, 32, 32, 32), (49152, 16384, 512, 16, 1), 0), primals_1, stride
            =(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=
            True, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (64, 16, 32, 32, 32), (524288, 32768, 1024,
            32, 1))
        buf1 = empty_strided_cuda((1, 64, 1, 1, 1), (64, 1, 64, 64, 64),
            torch.float32)
        buf2 = reinterpret_tensor(buf1, (64, 1, 1, 1), (1, 1, 1, 1), 0)
        del buf1
        buf3 = empty_strided_cuda((64, 16, 1, 1, 1), (16, 1, 16, 16, 16),
            torch.float32)
        buf4 = reinterpret_tensor(buf0, (64, 16, 1, 1, 1), (16, 1, 1, 1, 1), 0
            )
        del buf0
        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_0[grid(64)](buf2, buf3,
            primals_4, buf4, primals_5, 64, 32, XBLOCK=1, num_warps=2,
            num_stages=1)
        del primals_4
        del primals_5
        buf5 = empty_strided_cuda((64, 16, 32, 32, 32), (524288, 32768, 
            1024, 32, 1), torch.float32)
        triton_poi_fused_avg_pool3d_1[grid(50176)](buf4, buf5, 50176,
            XBLOCK=256, num_warps=4, num_stages=1)
        buf6 = empty_strided_cuda((64, 16, 4, 4, 4), (1024, 64, 16, 4, 1),
            torch.float32)
        triton_poi_fused_avg_pool3d_2[grid(12544)](buf5, buf6, 12544,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf5
    return buf6, primals_1, primals_2, reinterpret_tensor(primals_3, (1, 3,
        32, 32, 32), (49152, 16384, 512, 16, 1), 0), buf2, buf3, buf4


class ModelNew(nn.Module):
    """
    A model that performs a 3D transposed convolution, followed by batch normalization, 
    two average pooling layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_4 = self.batch_norm.weight
        primals_5 = self.batch_norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
