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
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 16384
    x2 = xindex // 65536
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 65536 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (16384 + x0 + 65536 * x2), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (32768 + x0 + 65536 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp0 - tmp5
    tmp7 = tl_math.exp(tmp6)
    tl.store(out_ptr0 + x3, tmp7, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 16384
    x2 = xindex // 65536
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 65536 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (16384 + x0 + 65536 * x2), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (32768 + x0 + 65536 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 / tmp5
    tl.store(out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_add_mul_tanh_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 16384
    x2 = xindex // 65536
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 16384 * x2), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr1 + (8192 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (16384 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp1 + tmp3
    tmp4 = tmp2 + tmp6
    tmp5 = 20.0
    tmp7 = tmp4 > tmp5
    tmp8 = libdevice.log1p(tmp4)
    tmp9 = -tmp8
    tmp10 = tl.where(tmp7, tmp4, tmp9)
    tmp11 = tl_math.exp(tmp10)
    tmp12 = libdevice.log1p(tmp11)
    tmp13 = -tmp12
    tmp14 = tl.where(tmp7, tmp4, tmp13)
    tmp15 = tmp0 - tmp14
    tmp16 = 1e-30
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = tmp17 * tmp17
    tmp19 = tmp14 * tmp17
    tmp20 = tl.where(tmp7, tmp15, tmp19)
    tmp21 = tl.load(in_ptr2 + x3, xmask)
    tmp22 = tl.load(in_ptr3 + x3, xmask)
    tmp23 = tl.load(in_ptr4 + x3, xmask)
    tmp24 = tmp21 + tmp22
    tmp25 = tmp20 * tmp24
    tmp26 = tmp23 + tmp25
    tl.store(out_ptr0 + x3, tmp26, xmask)


@triton.jit
def triton_per_fused_native_batch_norm_3(in_ptr0, out_ptr0, out_ptr1,
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
    tmp11 = tl.load(in_ptr0 + (r1 + 64 * x0), xmask, other=0.0)
    tmp0 = x0
    tl.store(out_ptr2 + (r1 + 64 * x0), tmp0, xmask)
    tmp3 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 / tmp8
    tmp10 = tmp3 - tmp9
    tmp12 = tmp10 * tmp10
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 64.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tl.store(out_ptr0 + x0, tmp9, xmask)
    tl.store(out_ptr1 + x0, tmp21, xmask)
    tl.store(out_ptr2 + (r1 + 64 * x0), tmp0, xmask)


@triton.jit
def triton_poi_fused_native_batch_norm_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = xindex % 128
    x0 = xindex % 64
    x1 = xindex // 64 % 2
    x2 = xindex // 128
    x5 = xindex % 512
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x4, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + x3, tmp8, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (64, 128, 128, 128), (2097152, 16384, 128,
        1))
    assert_size_stride(primals_4, (128,), (1,))
    assert_size_stride(primals_5, (128,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (64, 128, 126, 126), (2097152, 16384, 126, 
            1))
        buf1 = empty_strided_cuda((64, 128, 126, 126), (2097152, 16384, 
            126, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__softmax_0[grid(524288)](buf0, buf1, 524288,
            XBLOCK=512, num_warps=8, num_stages=1)
        buf2 = buf0
        del buf0
        triton_poi_fused__softmax_1[grid(524288)](buf1, buf2, 524288,
            XBLOCK=512, num_warps=8, num_stages=1)
        buf3 = empty_strided_cuda((64, 128, 126, 126), (2097152, 16384, 126,
            1), torch.float32)
        triton_poi_fused_add_mul_tanh_2[grid(524288)](buf2, buf1, primals_3,
            primals_1, primals_2, buf3, 524288, XBLOCK=512, num_warps=8,
            num_stages=1)
        del buf1
        del buf2
        del primals_2
        del primals_1
        del primals_3
        buf4 = empty_strided_cuda((128, 1), (1, 1), torch.float32)
        buf5 = empty_strided_cuda((128, 1), (1, 1), torch.float32)
        buf8 = empty_strided_cuda((128, 1), (1, 1), torch.float32)
        buf7 = reinterpret_tensor(buf8, (128, 1, 1), (1, 1, 1), 0)
        del buf8
        triton_per_fused_native_batch_norm_3[grid(128)](buf3, buf4, buf5,
            buf7, 128, 64, XBLOCK=1, num_warps=2, num_stages=1)
        buf9 = empty_strided_cuda((64, 128, 126, 126), (2097152, 16384, 126,
            1), torch.float32)
        triton_poi_fused_native_batch_norm_4[grid(32768)](buf3, buf4, buf5,
            buf7, primals_5, buf9, 32768, XBLOCK=256, num_warps=4,
            num_stages=1)
        del buf3
        del buf4
        del buf5
        del buf7
        del primals_5
    return buf9, primals_4, reinterpret_tensor(buf6, (128, 128, 126, 126),
        (2097152, 16384, 126, 1), 0)


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies activation, and then applies Batch Normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_4 = self.bn.weight
        primals_5 = self.bn.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
