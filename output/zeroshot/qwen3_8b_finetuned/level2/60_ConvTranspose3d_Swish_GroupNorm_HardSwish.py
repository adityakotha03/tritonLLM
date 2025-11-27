import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn.functional as F


@triton.jit
def triton_poi_fused_add_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0 * tmp0
    tmp2 = 1.0
    tmp3 = tmp1 - tmp2
    tmp4 = 0.0001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full([1], 16, tl.int64)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full([1], 0.0, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp0 * tmp9
    tl.store(out_ptr0 + x0, tmp10, xmask)


@triton.jit
def triton_poi_fused_hardswish_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 6, tl.int64)
    tmp2 = tmp0 <= tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.full([1], 0.0, tl.float32)
    tmp7 = tmp0 <= tmp6
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tmp0 >= tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tl.full([1], 0.0, tl.float32)
    tmp12 = tmp0 <= tmp11
    tmp13 = tl.full([1], 0, tl.int64)
    tmp14 = tmp0 >= tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp5 | tmp15
    tmp17 = tl.full([1], 6, tl.int64)
    tmp18 = tmp0 <= tmp17
    tmp19 = tl.full([1], 0, tl.int64)
    tmp20 = tmp0 >= tmp19
    tmp21 = tmp18 & tmp20
    tmp22 = tmp16 | tmp21
    tmp23 = tl.where(tmp22, tmp0, tmp6)
    tmp24 = tl.where(tmp22, tmp0, tmp11)
    tmp25 = tmp23 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = 0.0001
    tmp28 = tmp26 + tmp27
    tmp29 = tl.sqrt(tmp28)
    tmp30 = tmp0 - tmp23
    tmp31 = tmp30 * tmp29
    tmp32 = 6.0
    tmp33 = tmp31 / tmp32
    tl.store(out_ptr0 + x0, tmp33, xmask)


@triton.jit
def triton_poi_fused_sigmoid_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = tl.full([1], 0.0, tl.float32)
    tmp4 = tmp0 <= tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = tl.full([1], 0.0, tl.float32)
    tmp7 = tmp0 <= tmp6
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tmp0 >= tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp5 | tmp10
    tmp12 = tl.full([1], 6.0, tl.float32)
    tmp13 = tmp0 <= tmp12
    tmp14 = tl.full([1], 0, tl.int64)
    tmp15 = tmp0 >= tmp14
    tmp16 = tmp13 & tmp15
    tmp17 = tmp11 | tmp16
    tmp18 = tl.full([1], 0.0, tl.float32)
    tmp19 = tmp0 <= tmp18
    tmp20 = tl.full([1], 0, tl.int64)
    tmp21 = tmp0 >= tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tmp17 | tmp22
    tmp24 = tl.where(tmp23, tmp0, tmp6)
    tmp25 = tmp0 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = 0.0001
    tmp28 = tmp26 + tmp27
    tmp29 = tl.sqrt(tmp28)
    tmp30 = tmp0 - tmp24
    tmp31 = tmp30 * tmp29
    tmp32 = 6.0
    tmp33 = tmp31 / tmp32
    tmp34 = tl.where(tmp23, tmp33, tmp6)
    tmp35 = tmp24 + tmp34
    tl.store(out_ptr0 + x0, tmp35, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (128, 3, 16, 32, 32), (262144, 8192, 512, 16, 1))
    assert_size_stride(primals_2, (16,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 16, 8, 16, 16), (2048, 128, 256, 16, 1), torch.float32)
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(262144)](primals_1, buf1, 262144, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_1
        buf2 = buf1
        del buf1
        buf3 = buf2
        del buf2
        triton_poi_fused_sigmoid_2[grid(262144)](buf3, buf0, 262144, XBLOCK=256, num_warps=4, num_stages=1)
        del buf3
        buf4 = buf0
        del buf0
        triton_poi_fused_hardswish_1[grid(262144)](buf4, buf3, 262144, XBLOCK=256, num_warps=4, num_stages=1)
        del buf4
    return buf3, primals_2,


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        output = call([primals_1, primals_2, input_0])
        return output[0]