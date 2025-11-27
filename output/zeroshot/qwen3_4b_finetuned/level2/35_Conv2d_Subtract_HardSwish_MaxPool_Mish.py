import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    constexpr):
    xnumel = 1966080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16384 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_hardswish_max_pool2d_with_indices_mish_1(in_ptr0,
    out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1966080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16384 % 128
    x0 = xindex % 16384
    x2 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x1 + 128 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + x1 + 128 * x0), xmask, eviction_policy
        ='evict_last')
    tmp6 = tl.load(in_ptr0 + (256 + x1 + 128 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (x1 + 128 * (x0 // 4 + 4 * x2)), xmask,
        eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (128 + x1 + 128 * (x0 // 4 + 4 * x2)), xmask,
        eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (256 + x1 + 128 * (x0 // 4 + 4 * x2)), xmask,
        eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (x1 + 128 * (x0 % 4 + 4 * (x0 // 4 + 4 * x2))),
        xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (128 + x1 + 128 * (x0 % 4 + 4 * (x0 // 4 + 4 *
        x2))), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (256 + x1 + 128 * (x0 % 4 + 4 * (x0 // 4 + 4 *
        x2))), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (x1 + 128 * (x0 % 4 + 4 * (x0 // 4 + 4 * x2))),
        xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (128 + x1 + 128 * (x0 % 4 + 4 * (x0 // 4 + 4 *
        x2))), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (256 + x1 + 128 * (x0 % 4 + 4 * (x0 // 4 + 4 *
        x2))), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (x1 + 128 * (x0 // 4 + 4 * (x0 // 4 + 4 * x2))),
        xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (128 + x1 + 128 * (x0 // 4 + 4 * (x0 // 4 + 4 *
        x2))), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (256 + x1 + 128 * (x0 // 4 + 4 * (x0 // 4 + 4 *
        x2))), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (x1 + 128 * (x0 % 4 + 4 * (x0 // 4 + 4 * x2))),
        xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr0 + (128 + x1 + 128 * (x0 % 4 + 4 * (x0 // 4 + 4 *
        x2))), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (256 + x1 + 128 * (x0 % 4 + 4 * (x0 // 4 + 4 *
        x2))), xmask, eviction_policy='evict_last')
    tmp4 = tmp0 - 0.5
    tmp5 = 3.0
    tmp7 = tmp5 * tmp4
    tmp8 = tl.where(tmp1 <= tmp7, tmp1, tmp7)
    tmp10 = tmp5 * tmp3
    tmp12 = tmp5 * tmp6
    tmp13 = tl.where(tmp8 <= tmp10, tmp8, tmp10)
    tmp15 = tl.where(tmp13 <= tmp12, tmp13, tmp12)
    tmp16 = tl.where(tmp15 <= tmp9, tmp15, tmp9)
    tmp18 = tl.where(tmp16 <= tmp11, tmp16, tmp11)
    tmp19 = tl.where(tmp18 <= tmp14, tmp18, tmp14)
    tmp21 = tl.where(tmp19 <= tmp17, tmp19, tmp17)
    tmp22 = tl.where(tmp21 <= tmp20, tmp21, tmp20)
    tmp24 = tl.where(tmp22 <= tmp23, tmp22, tmp23)
    tmp25 = tl.where(tmp24 <= tmp26, tmp24, tmp26)
    tmp26 = tl.where(tmp25 <= tmp29, tmp25, tmp29)
    tmp27 = tl.where(tmp26 <= tmp32, tmp26, tmp32)
    tmp35 = tmp0 - 0.5
    tmp36 = 3.0
    tmp37 = tmp35 * tmp36
    tmp38 = tl.where(tmp27 <= tmp37, tmp27, tmp37)
    tmp39 = 0.0
    tmp40 = tl.where(tmp38 <= tmp39, tmp38, tmp39)
    tmp41 = 1.0
    tmp42 = tl.where(tmp40 <= tmp41, tmp40, tmp41)
    tl.store(out_ptr0 + x3, tmp38, xmask)
    tl.store(out_ptr1 + x3, tmp42, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 64, 128, 128), (1048576, 16384, 128, 
        1))
    assert_size_stride(arg1_1, (128, 64, 3, 3), (576, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 128, 128, 128), (2097152, 16384, 
            128, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(1966080)](buf0, arg1_1, 1966080,
            XBLOCK=512, num_warps=8, num_stages=1)
        del arg1_1
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((128, 128, 128, 128), (2097152, 16384, 
            128, 1), torch.float32)
        triton_poi_fused_hardswish_max_pool2d_with_indices_mish_1[grid(1966080)](
            buf1, buf2, buf1, 1966080, XBLOCK=512, num_warps=8, num_stages=1)
    return buf2, arg0_1, buf1


class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtracts a value, applies HardSwish, MaxPool, and Mish activation functions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value,
        pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, input_0):
        arg1_1 = self.conv.weight
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]
