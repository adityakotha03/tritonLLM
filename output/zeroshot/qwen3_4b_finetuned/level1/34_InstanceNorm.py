import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 32768
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = yindex // 128
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128 * x2 + 16384 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 128 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_add_mean_sub_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.sum(tmp3, 0)[:, None]
    tmp6 = 128.0
    tmp7 = tmp5 / tmp6
    tmp8 = tmp2 - tmp7
    tl.store(out_ptr0 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_div_mul_pow_rsub_sqrt_2(in_out_ptr0, in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp3 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tmp0 + tmp3
    tmp2 = 1.0
    tmp4 = tmp1 * tmp2
    tmp6 = tmp4 - tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 * tmp15
    tmp17 = tmp6 * tmp2
    tmp18 = tmp17 * tmp17
    tmp19 = 127.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = tl.sqrt(tmp22)
    tmp24 = tmp16 / tmp23
    tl.store(in_out_ptr0 + x2, tmp24, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (112, 64, 512, 512), (16384, 256, 512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((112, 128, 512, 512), (16384, 128, 256, 1
            ), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(32768, 128)](arg0_1, buf0, 32768, 128,
            XBLOCK=64, YBLOCK=128, num_warps=8, num_stages=1)
        buf1 = empty_strided_cuda((112, 128, 512, 512), (16384, 128, 256, 1
            ), torch.float32)
        triton_poi_fused_add_mean_sub_1[grid(32768)](buf0, buf1, 32768,
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf0
        del buf1
        buf2 = empty_strided_cuda((112, 64, 512, 512), (16384, 256, 1, 1),
            torch.float32)
        triton_poi_fused_add_div_mul_pow_rsub_sqrt_2[grid(32768)](buf2,
            arg0_1, buf1, buf1, buf1, buf1, buf1, buf1, buf1, 32768, XBLOCK=
            256, num_warps=4, num_stages=1)
        del arg0_1
        del buf1
    return buf2,


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
