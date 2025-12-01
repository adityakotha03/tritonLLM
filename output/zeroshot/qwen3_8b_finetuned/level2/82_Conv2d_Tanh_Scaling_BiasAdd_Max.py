import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_mul_tanh_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = libdevice.tanh(tmp3)
    tmp5 = 2.0
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 268736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 63
    x1 = xindex // 63 % 64
    x2 = xindex // 4032
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4 * x0 + 16128 * x1 + 63 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4 * x0 + 16128 * x1 + 63 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4 * x0 + 16128 * x1 + 63 * x2), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4 * x0 + 16128 * x1 + 63 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 <= tmp0
    tmp8 = tl.full([1], True, tl.int1)
    tmp9 = tmp7 == tmp8
    tmp10 = tmp9 == tmp8
    tmp11 = tmp1 <= tmp2
    tmp12 = tmp11 == tmp8
    tmp13 = tmp12 == tmp8
    tmp14 = tmp1 <= tmp4
    tmp15 = tmp14 == tmp8
    tmp16 = tmp15 == tmp8
    tmp17 = tmp1 <= tmp6
    tmp18 = tmp17 == tmp8
    tmp19 = tmp18 == tmp8
    tmp20 = tmp0 <= tmp1
    tmp21 = tmp20 == tmp8
    tmp22 = tmp21 == tmp8
    tmp23 = tmp0 <= tmp2
    tmp24 = tmp23 == tmp8
    tmp25 = tmp24 == tmp8
    tmp26 = tmp0 <= tmp4
    tmp27 = tmp26 == tmp8
    tmp28 = tmp27 == tmp8
    tmp29 = tmp0 <= tmp6
    tmp30 = tmp29 == tmp8
    tmp31 = tmp30 == tmp8
    tmp32 = tmp2 <= tmp1
    tmp33 = tmp32 == tmp8
    tmp34 = tmp33 == tmp8
    tmp35 = tmp2 <= tmp0
    tmp36 = tmp35 == tmp8
    tmp37 = tmp36 == tmp8
    tmp38 = tmp2 <= tmp4
    tmp39 = tmp38 == tmp8
    tmp40 = tmp39 == tmp8
    tmp41 = tmp2 <= tmp6
    tmp42 = tmp41 == tmp8
    tmp43 = tmp42 == tmp8
    tmp44 = tmp4 <= tmp1
    tmp45 = tmp44 == tmp8
    tmp46 = tmp45 == tmp8
    tmp47 = tmp4 <= tmp0
    tmp48 = tmp47 == tmp8
    tmp49 = tmp48 == tmp8
    tmp50 = tmp4 <= tmp2
    tmp51 = tmp50 == tmp8
    tmp52 = tmp51 == tmp8
    tmp53 = tmp4 <= tmp6
    tmp54 = tmp53 == tmp8
    tmp55 = tmp54 == tmp8
    tmp56 = tmp6 <= tmp1
    tmp57 = tmp56 == tmp8
    tmp58 = tmp57 == tmp8
    tmp59 = tmp6 <= tmp0
    tmp60 = tmp59 == tmp8
    tmp61 = tmp60 == tmp8
    tmp62 = tmp6 <= tmp2
    tmp63 = tmp62 == tmp8
    tmp64 = tmp63 == tmp8
    tmp65 = tmp6 <= tmp4
    tmp66 = tmp65 == tmp8
    tmp67 = tmp66 == tmp8
    tmp68 = tl.where(tmp67, tl.full([1], 2, tl.int64), tl.where(tmp64,
        tl.full([1], 3, tl.int64), tl.where(tmp61, tl.full([1], 4, tl.int64),
        tl.where(tmp58, tl.full([1], 0, tl.int64), tl.where(tmp55, tl.full([
        1], 1, tl.int64), tl.where(tmp52, tl.full([1], 2, tl.int64), tl.where
        (tmp49, tl.full([1], 3, tl.int64), tl.where(tmp46, tl.full([1], 4,
        tl.int64), tl.where(tmp43, tl.full([1], 0, tl.int64), tl.where(
        tmp39, tl.full([1], 1, tl.int64), tl.where(tmp36, tl.full([1], 2,
        tl.int64), tl.where(tmp33, tl.full([1], 3, tl.int64), tl.where(
        tmp30, tl.full([1], 4, tl.int64), tl.where(tmp28, tl.full([1], 0,
        tl.int64), tl.where(tmp25, tl.full([1], 1, tl.int64), tl.where(
        tmp22, tl.full([1], 2, tl.int64), tl.where(tmp19, tl.full([1], 3,
        tl.int64), tl.where(tmp16, tl.full([1], 4, tl.int64), tl.where(
        tmp13, tl.full([1], 0, tl.int64), tl.where(tmp10, tl.full([1], 1,
        tl.int64), tl.where(tmp8, tl.full([1], 2, tl.int64), tl.where(tmp5,
        tl.full([1], 3, tl.int64), tl.full([1], 4, tl.int64))))))))))))))))))
    tl.store(out_ptr0 + x3, tmp6, xmask)
    tl.store(out_ptr1 + x3, tmp68, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg1_1, (128, 8, 256, 256), (523776, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(arg1_1, arg0_1, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 256, 256), (4194304, 65536, 256, 
            1))
        buf2 = empty_strided_cuda((128, 64, 256, 256), (4194304, 65536, 256,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_mul_tanh_0[grid(1048576)](buf0, arg1_1, buf2,
            1048576, XBLOCK=128, num_warps=4, num_stages=1)
        del arg1_1
        buf3 = extern_kernels.convolution(arg2_1, arg0_1, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (128, 64, 256, 256), (4194304, 65536, 256, 
            1))
        buf4 = empty_strided_cuda((128, 64, 63, 63), (268736, 4032, 63, 1),
            torch.float32)
        buf5 = empty_strided_cuda((128, 64, 63, 63), (268736, 4032, 63, 1),
            torch.bool)
        triton_poi_fused_max_pool2d_with_indices_1[grid(268736)](buf2, buf4,
            buf5, 268736, XBLOCK=128, num_warps=4, num_stages=1)
        del buf2
    return buf4, arg0_1, buf0, buf3, buf5


class ModelNew(nn.Module):
    """
    A model that performs a convolution, applies tanh, scaling, adds a bias term, and then max-pools.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, input_0):
        arg0_1 = self.conv.weight
        arg1_1 = input_0
        arg2_1 = self.bias
        output = call([arg0_1, arg1_1, arg2_1])
        return output[0]