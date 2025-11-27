import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_tanh_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp1 + tmp0
    tmp3 = tl.full([1], 1, tl.int32)
    tmp4 = tmp3 + tmp2
    tmp5 = tmp2 + tmp3
    tmp6 = tl.full([1], 2, tl.int32)
    tmp7 = tmp6 + tmp2
    tmp8 = tmp2 + tmp6
    tmp9 = tmp4 < tmp6
    tmp10 = tmp8 < tmp3
    tmp11 = tmp9 & tmp10
    tmp12 = tmp8 >= tmp6
    tmp13 = tmp4 >= tmp3
    tmp14 = tmp12 & tmp13
    tmp15 = tmp11 | tmp14
    tmp16 = tmp2 < tmp3
    tmp17 = tmp16 & tmp15
    tmp18 = tl.load(in_ptr0 + (-1 + x0), tmp17 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp19 = tmp2 < tmp6
    tmp20 = tmp19 & tmp15
    tmp21 = tl.load(in_ptr0 + (-2 + x0), tmp20 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp22 = tmp2 >= tmp6
    tmp23 = tmp22 & tmp15
    tmp24 = tl.load(in_ptr0 + (1 + x0), tmp23 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp25 = tmp2 >= tmp3
    tmp26 = tmp25 & tmp15
    tmp27 = tl.load(in_ptr0 + (2 + x0), tmp26 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp28 = tmp17 & tmp22
    tmp29 = tmp18 + tmp28
    tmp30 = tmp21 + tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp24 + tmp28
    tmp33 = tmp27 + tmp28
    tmp34 = tmp31 + tmp32
    tmp35 = tmp34 + tmp33
    tmp36 = tmp35 / tmp3
    tmp37 = tmp2 + tmp36
    tmp38 = tmp2 - tmp36
    tmp39 = tl.sigmoid(tmp38)
    tmp40 = tmp39 * tmp39
    tmp41 = tmp39 - tmp40
    tmp42 = tmp36 * tmp41
    tl.store(out_ptr0 + x0, tmp42, xmask)


@triton.jit
def triton_poi_fused_avg_pool2d_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + x0 + 129 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (129 + x0 + 129 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (258 + x0 + 129 * x1), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (387 + x0 + 129 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (516 + x0 + 129 * x1), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (645 + x0 + 129 * x1), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (774 + x0 + 129 * x1), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (903 + x0 + 129 * x1), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (1032 + x0 + 129 * x1), xmask, eviction_policy
        ='evict_last')
    tmp17 = tl.load(in_ptr0 + (1161 + x0 + 129 * x1), xmask, eviction_policy
        ='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = 16.0
    tmp20 = tmp18 / tmp19
    tl.store(out_ptr0 + x2, tmp20, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 128, 128), (1048576, 16384, 128,
        1))
    assert_size_stride(primals_2, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_3, (128,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_1, primals_2, [0],
            [0], 1, 1, padding=(1, 1), stride=(1, 1), dilation=(1, 1),
            transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 128, 128, 128), (2097152, 16384, 128,
            1))
        buf1 = empty_strided_cuda((128, 128, 128, 128), (2097152, 16384, 128,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_tanh_0[grid(2048)](buf0, buf1, 2048, XBLOCK=128,
            num_warps=4, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((128, 128, 64, 64), (524288, 4096, 64, 1),
            torch.float32)
        triton_poi_fused_avg_pool2d_1[grid(16384)](buf1, buf2, 16384,
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf1
    return buf2, primals_1, primals_2, primals_3


class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtraction, tanh activation, subtraction and average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value,
        subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)

    def forward(self, input_0):
        primals_2 = self.conv.weight
        primals_3 = self.conv.bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
