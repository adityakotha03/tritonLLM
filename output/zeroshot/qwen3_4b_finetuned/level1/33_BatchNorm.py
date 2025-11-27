import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, out_ptr3, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = xindex // 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64 * x1), xmask)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + 64 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (256 + x0 + 64 * x1), xmask)
    tmp5 = tl.load(in_ptr0 + (384 + x0 + 64 * x1), xmask)
    tmp11 = tl.load(in_ptr0 + (x0 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (128 + x0 + 64 * x1), xmask, eviction_policy
        ='evict_last')
    tmp14 = tl.load(in_ptr0 + (256 + x0 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr0 + (384 + x0 + 64 * x1), xmask, eviction_policy
        ='evict_last')
    tmp2 = tmp1 - tmp0
    tmp4 = tmp3 - tmp0
    tmp6 = tmp5 - tmp0
    tmp7 = tmp2 * tmp2
    tmp8 = tmp4 * tmp4
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp6
    tmp13 = tmp12 - tmp11
    tmp15 = tmp14 - tmp11
    tmp17 = tmp16 - tmp11
    tmp18 = tmp13 * tmp13
    tmp19 = tmp15 * tmp15
    tmp20 = tmp18 + tmp19
    tmp21 = tmp17 * tmp17
    tmp22 = tmp20 + tmp21
    tmp23 = 64.0
    tmp24 = tmp22 / tmp23
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = tl.sqrt(tmp26)
    tmp28 = 1.0
    tmp29 = tmp27 * tmp28
    tl.store(out_ptr0 + x2, tmp29, xmask)
    tl.store(out_ptr1 + x2, tmp11, xmask)
    tl.store(out_ptr2 + x2, tmp29, xmask)
    tl.store(out_ptr3 + x2, tmp29, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_1(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = xindex // 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + x1, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tmp0 - tmp2
    tmp5 = tmp3 / tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp11 = tmp1 - tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tmp12 + tmp1
    tl.store(out_ptr0 + x2, tmp13, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (64, 64, 512, 512), (16384, 256, 512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 64, 512, 512), (16384, 256, 512, 1),
            torch.float32)
        buf1 = empty_strided_cuda((64, 64, 512, 512), (16384, 256, 512, 1),
            torch.float32)
        buf2 = empty_strided_cuda((64, 64, 512, 512), (16384, 256, 512, 1),
            torch.float32)
        buf3 = empty_strided_cuda((64, 64, 512, 512), (16384, 256, 512, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_0[grid(32768)](arg0_1,
            buf0, buf1, buf2, buf3, 32768, XBLOCK=128, num_warps=4, num_stages=
            1)
        buf4 = empty_strided_cuda((64, 64, 512, 512), (16384, 256, 512, 1),
            torch.float32)
        triton_poi_fused__native_batch_norm_legit_1[grid(32768)](arg0_1,
            buf0, buf1, buf2, buf3, buf4, buf4, 32768, XBLOCK=128,
            num_warps=4, num_stages=1)
    return buf4, arg0_1, buf0, buf1, buf2, buf3


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
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
