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
    xnumel = 64
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 1024 * x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 1024, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 1024.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tl.store(out_ptr2 + x0, tmp10, xmask)
    tl.store(out_ptr3 + x0, tmp22, xmask)
    tl.store(out_ptr0 + x0, tmp7, xmask)
    tl.store(out_ptr1 + x0, tmp16, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_1(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x2 = xindex // 32768
    x4 = xindex % 32768
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x4, xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + x3, tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 64, 512, 512), (16777216, 262144, 
        512, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        buf1 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        buf3 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_0[grid(64)](primals_1,
            buf0, buf1, buf3, primals_2, 64, 1024, XBLOCK=1, num_warps=2,
            num_stages=1)
        buf2 = empty_strided_cuda((64, 64, 512, 512), (16777216, 262144, 
            512, 1), torch.float32)
        triton_poi_fused__native_batch_norm_legit_1[grid(32768)](primals_1,
            buf0, primals_3, buf1, primals_2, buf2, 32768, XBLOCK=256,
            num_warps=4, num_stages=1)
        del buf0
        del buf1
        del primals_2
        del primals_3
    return buf2, primals_1


class ModelNew(nn.Module):
    """
    Simple model that performs Batch Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, input_0):
        primals_2 = self.bn.weight
        primals_3 = self.bn.bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
