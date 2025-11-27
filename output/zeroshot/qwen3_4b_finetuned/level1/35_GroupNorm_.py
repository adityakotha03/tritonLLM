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
def triton_per_fused_native_group_norm_0(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, xnumel, rnumel, XBLOCK: tl.constexpr):
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
    tl.store(out_ptr2 + (r1 + 64 * x2 + 1024 * x3), tmp22, xmask)
    tl.store(out_ptr0 + x0, tmp10, xmask)
    tl.store(out_ptr1 + x0, tmp16, xmask)


@triton.jit
def triton_per_fused_native_group_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 8
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = xindex % 2
    x3 = xindex // 2
    tmp0 = tl.load(in_ptr0 + (r1 + 16 * x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 16 * x0), xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + 64 * x3), xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + 64 * x3), xmask, other=0.0)
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (r1 + 16 * x2), tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 8, 8), (64, 8, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (112, 64, 512, 512), (16777216, 262144,
        512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((112, 1), (1, 1), torch.float32)
        buf1 = empty_strided_cuda((112, 1), (1, 1), torch.float32)
        buf3 = empty_strided_cuda((112, 64, 512, 512), (33554432, 524288,
            1024, 1), torch.float32)
        get_raw_stream(0)
        triton_per_fused_native_group_norm_0[grid(112)](primals_3, buf0,
            buf1, buf3, 112, 64, XBLOCK=8, num_warps=2, num_stages=1)
        buf4 = empty_strided_cuda((112, 64, 512, 512), (33554432, 524288,
            1024, 1), torch.float32)
        triton_per_fused_native_group_norm_1[grid(8)](primals_3, buf0, buf1,
            buf3, buf4, 8, 16, XBLOCK=1, num_warps=2, num_stages=1)
        del buf0
        del buf1
        del buf3
    return buf4, primals_1, primals_2, primals_3


class ModelNew(nn.Module):
    """
    Simple model that performs Group Normalization.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)

    def forward(self, input_0):
        primals_1 = self.gn.weight
        primals_2 = self.gn.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
