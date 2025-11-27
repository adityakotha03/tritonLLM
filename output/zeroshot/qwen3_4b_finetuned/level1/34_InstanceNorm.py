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
def triton_per_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 112
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = xindex % 14
    x3 = xindex // 14
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
    tl.store(out_ptr2 + (r1 + 64 * x2 + 4096 * x3), tmp17, xmask)
    tl.store(out_ptr3 + (r1 + 64 * x2 + 4096 * x3), tmp22, xmask)
    tl.store(out_ptr0 + x0, tmp10, xmask)
    tl.store(out_ptr1 + x0, tmp16, xmask)


@triton.jit
def triton_per_fused__native_batch_norm_legit_1(in_out_ptr0, in_ptr0,
    in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 7168
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex % 14
    x2 = xindex // 14 % 512
    x3 = xindex // 7168
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64 * x0), None, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (r1 + 64 * x0), None, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr2 + (r1 + 64 * x0), None, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_out_ptr0 + (r1 + 64 * x2 + 32768 * x3), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (64 * x0 + 896 * x2), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr1 + (64 * x0 + 896 * x2), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr2 + (64 * x0 + 896 * x2), xmask, eviction_policy=
        'evict_last')
    tmp7 = tmp4 + tmp5
    tmp8 = tmp7 + tmp6
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tmp1 + tmp2
    tmp17 = tmp16 + tmp3
    tmp18 = tmp17 - tmp15
    tmp19 = 64.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp0 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = 64.0
    tmp28 = tmp26 / tmp27
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tl.store(in_out_ptr0 + (r1 + 64 * x2 + 32768 * x3), tmp17, xmask)
    tl.store(out_ptr0 + x4, tmp31, xmask)


@triton.jit
def triton_per_fused__native_batch_norm_legit_2(in_out_ptr0, in_ptr0,
    in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 7168
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex % 14
    x2 = xindex // 14 % 512
    x3 = xindex // 7168
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64 * x0), None, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (r1 + 64 * x0), None, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr2 + (r1 + 64 * x0), None, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_out_ptr0 + (r1 + 64 * x2 + 32768 * x3), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (64 * x0 + 896 * x2), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr1 + (64 * x0 + 896 * x2), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr2 + (64 * x0 + 896 * x2), xmask, eviction_policy=
        'evict_last')
    tmp7 = tmp4 + tmp5
    tmp8 = tmp7 + tmp6
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tmp1 + tmp2
    tmp17 = tmp16 + tmp3
    tmp18 = tmp17 - tmp15
    tmp19 = 64.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp0 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = 64.0
    tmp28 = tmp26 / tmp27
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tl.store(in_out_ptr0 + (r1 + 64 * x2 + 32768 * x3), tmp17, xmask)
    tl.store(out_ptr0 + x4, tmp31, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (112, 64, 512, 512), (16777216, 262144, 512,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((112,), (1,), torch.float32)
        buf1 = empty_strided_cuda((112,), (1,), torch.float32)
        buf3 = empty_strided_cuda((112, 64, 512, 512), (2097152, 32768, 
            512, 1), torch.float32)
        buf4 = empty_strided_cuda((112, 64, 512, 512), (2097152, 32768, 
            512, 1), torch.float32)
        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_0[grid(112)](arg0_1, buf0,
            buf1, buf3, buf4, 112, 64, XBLOCK=1, num_warps=2, num_stages=1)
        del arg0_1
        buf5 = buf4
        del buf4
        buf6 = empty_strided_cuda((112, 64, 512, 512), (2097152, 32768, 
            512, 1), torch.float32)
        triton_per_fused__native_batch_norm_legit_1[grid(7168)](buf5, buf0,
            buf1, buf3, buf6, 7168, 64, XBLOCK=64, num_warps=4, num_stages=1)
        del buf0
        del buf1
        del buf3
        buf7 = buf5
        del buf5
        buf8 = empty_strided_cuda((112, 64, 512, 512), (2097152, 32768, 
            512, 1), torch.float32)
        triton_per_fused__native_batch_norm_legit_2[grid(7168)](buf7, buf0,
            buf1, buf6, buf8, 7168, 64, XBLOCK=64, num_warps=4, num_stages=1)
        del buf0
        del buf1
        del buf6
    return buf8,


class ModelNew(nn.Module):
    """
    Simple model that performs Instance Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.inorm = nn.InstanceNorm2d(num_features=num_features)

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
