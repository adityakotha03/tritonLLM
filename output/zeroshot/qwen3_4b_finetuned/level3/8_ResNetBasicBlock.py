import torch
from torch._inductor.select_algorithm import extern_kernels
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
def triton_per_fused__native_batch_norm_legit_relu_0(in_out_ptr0, in_ptr0,
    in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = xindex % 16
    x3 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (r1 + 64 * x0), xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + x2, xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + x2, xmask, eviction_policy='evict_last')
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
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tl.full([1, 1], 0, tl.int32)
    tmp29 = triton_helpers.maximum(tmp28, tmp27)
    tl.store(in_out_ptr0 + (r1 + 64 * x0), tmp23, xmask)
    tl.store(out_ptr1 + (r1 + 64 * x0), tmp29, xmask)
    tl.store(out_ptr0 + x0, tmp10, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_1(in_ptr0, in_ptr1, in_ptr2,
    out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + 0)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp18 = tl.load(in_ptr0 + (1 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr1 + 1)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp22 = tl.load(in_ptr2 + 1)
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp35 = tl.load(in_ptr0 + (2 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp36 = tl.load(in_ptr1 + 2)
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp39 = tl.load(in_ptr2 + 2)
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK])
    tmp52 = tl.load(in_ptr0 + (3 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp53 = tl.load(in_ptr1 + 3)
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK])
    tmp56 = tl.load(in_ptr2 + 3)
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK])
    tmp3 = tmp0 - tmp2
    tmp5 = tmp3 * tmp5
    tmp6 = tmp4 - tmp5
    tmp7 = tmp4 * tmp4
    tmp8 = tmp5 * tmp5
    tmp9 = tmp6 - tmp7
    tmp10 = tmp8 - tmp9
    tmp11 = 64.0
    tmp12 = tmp10 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp3 * tmp15
    tmp17 = tmp16 * tmp2
    tmp21 = tmp18 - tmp20
    tmp24 = tmp21 * tmp23
    tmp25 = tmp22 - tmp24
    tmp26 = tmp22 * tmp22
    tmp27 = tmp24 * tmp24
    tmp28 = tmp25 - tmp26
    tmp29 = tmp27 - tmp28
    tmp30 = tmp29 / tmp11
    tmp31 = tmp30 + tmp13
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tmp21 * tmp32
    tmp34 = tmp33 * tmp20
    tmp38 = tmp35 - tmp37
    tmp41 = tmp38 * tmp40
    tmp42 = tmp39 - tmp41
    tmp43 = tmp39 * tmp39
    tmp44 = tmp41 * tmp41
    tmp45 = tmp42 - tmp43
    tmp46 = tmp44 - tmp45
    tmp47 = tmp46 / tmp11
    tmp48 = tmp47 + tmp13
    tmp49 = libdevice.rsqrt(tmp48)
    tmp50 = tmp38 * tmp49
    tmp51 = tmp50 * tmp37
    tmp55 = tmp52 - tmp54
    tmp58 = tmp55 * tmp57
    tmp59 = tmp56 - tmp58
    tmp60 = tmp56 * tmp56
    tmp61 = tmp58 * tmp58
    tmp62 = tmp59 - tmp60
    tmp63 = tmp61 - tmp62
    tmp64 = tmp63 / tmp11
    tmp65 = tmp64 + tmp13
    tmp66 = libdevice.rsqrt(tmp65)
    tmp67 = tmp55 * tmp66
    tmp68 = tmp67 * tmp54
    tmp69 = tmp17 + tmp34
    tmp70 = tmp69 + tmp51
    tmp71 = tmp70 + tmp68
    tl.store(out_ptr0 + x0, tmp15, xmask)
    tl.store(out_ptr1 + x0, tmp66, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_add_relu_2(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + x3, xmask)
    tmp12 = tl.load(in_ptr3 + x3, xmask)
    tmp14 = tl.load(in_ptr4 + x3, xmask)
    tmp16 = tl.load(in_ptr5 + x3, xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = 64.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp11 = tmp10 * tmp9
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + x3, tmp19, xmask)
    tl.store(out_ptr0 + x3, tmp9, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9) = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (10, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(primals_4, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_5, (64,), (1,))
    assert_size_stride(primals_6, (64,), (1,))
    assert_size_stride(primals_7, (64,), (1,))
    assert_size_stride(primals_8, (64,), (1,))
    assert_size_stride(primals_9, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (10, 64, 224, 224), (3276800, 50176, 224, 1))
        buf1 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf2 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf3 = reinterpret_tensor(buf0, (1, 64, 224, 224), (3276800, 50176,
            224, 1), 0)
        del buf0
        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_relu_0[grid(64)](buf3,
            primals_3, primals_2, primals_4, buf1, buf2, 64, 64, XBLOCK=1,
            num_warps=2, num_stages=1)
        del primals_2
        buf4 = extern_kernels.convolution(buf3, primals_4, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (10, 64, 224, 224), (3276800, 50176, 224, 1))
        buf5 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        buf6 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        triton_poi_fused__native_batch_norm_legit_1[grid(64)](primals_5,
            primals_6, primals_7, buf5, buf6, 64, XBLOCK=64, num_warps=1,
            num_stages=1)
        del primals_5
        buf7 = buf3
        del buf3
        buf8 = empty_strided_cuda((10, 64, 224, 224), (3276800, 50176, 224,
            1), torch.float32)
        triton_poi_fused__native_batch_norm_legit_add_relu_2[grid(4096)](buf7,
            primals_5, primals_6, primals_7, buf4, primals_8, primals_9,
            buf8, 4096, XBLOCK=256, num_warps=4, num_stages=1)
        del buf4
        del primals_9
    return buf8, primals_1, primals_3, primals_4, primals_6, primals_7, buf1, buf2, buf5, buf6, buf7


class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        :param downsample: Downsample layer for the shortcut connection
        """
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride

    def forward(self, input_0):
        primals_1 = self.conv1.weight
        primals_2 = self.bn1.weight
        primals_6 = self.bn1.bias
        primals_4 = self.conv2.weight
        primals_5 = self.bn2.weight
        primals_7 = self.bn2.bias
        primals_8 = self.downsample.conv2d.weight
        primals_9 = self.downsample.bn2d.weight
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9])
        return output[0]
