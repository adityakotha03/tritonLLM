import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_div_mean_mul_rsub_sqrt_sub_0(in_ptr0, in_ptr1,
    in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel,
    rnumel, XBLOCK: tl.constexpr):
    xnumel = 64
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    x0 = xindex
    tmp3 = tl.load(in_ptr0 + (r0 + 262144 * x0), xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp0 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp2 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.where(xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = tl.full([XBLOCK, 1], 262144, tl.int32)
    tmp8 = tmp6.to(tl.float32)
    tmp9 = tmp5 / tmp8
    tmp11 = tmp0 - tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 262143.0
    tmp18 = tmp16 / tmp17
    tmp19 = tmp18 + 1e-05
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp11 * tmp20
    tmp22 = tmp21 * tmp7
    tmp23 = tmp22 + tmp10
    tmp24 = tmp3 - tmp9
    tmp25 = tmp24 * tmp20
    tmp26 = tmp25 * tmp7
    tmp27 = tmp26 + tmp10
    tmp28 = tmp27 - tmp14
    tmp29 = tmp28 * tmp20
    tmp30 = tmp29 * tmp7
    tmp31 = tmp30 + tmp14
    tl.store(out_ptr0 + (x0 + 64 * r0), tmp9, xmask)
    tl.store(out_ptr1 + (x0 + 64 * r0), tmp19, xmask)
    tl.store(out_ptr2 + (x0 + 64 * r0), tmp20, xmask)
    tl.store(out_ptr3 + (x0 + 64 * r0), tmp31, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (64,), (1,))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (64, 64, 512, 512), (16777216, 262144, 
        512, 1))
    assert_size_stride(primals_4, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        buf1 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        buf2 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        buf3 = empty_strided_cuda((64, 64, 512, 512), (16777216, 262144, 
            512, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_div_mean_mul_rsub_sqrt_sub_0[grid(64)](primals_3,
            primals_1, primals_2, primals_4, buf0, buf1, buf2, buf3, 64, 
            256, XBLOCK=1, num_warps=2, num_stages=1)
    return buf3, primals_1, primals_2, primals_4, buf0, buf1, buf2


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
        primals_1 = self.bn.weight
        primals_2 = self.bn.bias
        primals_4 = self.bn.running_mean
        primals_3 = self.bn.running_var
        primals_1 = self.bn.weight
        primals_2 = self.bn.bias
        primals_4 = self.bn.running_mean
        primals_3 = self.bn.running_var
        output = call([primals_1, primals_2, input_0, primals_4])
        return output[0]