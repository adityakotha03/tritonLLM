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
def triton_per_fused__native_batch_norm_legit_relu_0(in_ptr0, out_ptr0,
    out_ptr2, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 128
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
    tmp17 = 1024.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp0 - tmp10
    tmp23 = tmp22 * tmp21
    tl.store(out_ptr2 + (r1 + 1024 * x0), tmp23, xmask)
    tl.store(out_ptr0 + x0, tmp21, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_1(in_ptr0, in_ptr1, in_ptr2,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (1 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr1 + 1)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp14 = tl.load(in_ptr2 + (1 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (2 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp20 = tl.load(in_ptr1 + 2)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp23 = tl.load(in_ptr2 + (2 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp28 = tl.load(in_ptr0 + (3 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp29 = tl.load(in_ptr1 + 3)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp32 = tl.load(in_ptr2 + (3 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp1 * tmp1
    tmp7 = tmp6 * tmp1
    tmp8 = tmp5 - tmp7
    tmp9 = tmp8 * tmp8
    tmp13 = tmp10 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp12 * tmp12
    tmp17 = tmp16 * tmp12
    tmp18 = tmp15 - tmp17
    tmp22 = tmp19 + tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = tmp21 * tmp21
    tmp26 = tmp25 * tmp21
    tmp27 = tmp24 - tmp26
    tmp29 = tmp28 + tmp30
    tmp31 = tmp29 + tmp32
    tmp33 = tmp30 * tmp30
    tmp34 = tmp33 * tmp30
    tmp35 = tmp31 - tmp34
    tmp36 = tmp9 + tmp18
    tmp37 = tmp36 + tmp27
    tmp38 = tmp37 + tmp35
    tmp39 = 4.0
    tmp40 = tmp38 / tmp39
    tl.store(out_ptr0 + x0, tmp40, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_2(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 2
    x0 = xindex % 2
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp5 = 1e-05
    tmp6 = tmp3 + tmp5
    tmp7 = libdevice.rsqrt(tmp6)
    tmp8 = tmp4 * tmp7
    tmp9 = tmp8 * tmp10
    tmp11 = tmp9 + tmp12
    tl.store(out_ptr0 + x2, tmp11, xmask)


@triton.jit
def triton_poi_fused_avg_pool2d_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x1 = xindex // 2 % 2
    x2 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 8 * x1 + 32 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 8 * x1 + 32 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4 + 2 * x0 + 8 * x1 + 32 * x2), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (5 + 2 * x0 + 8 * x1 + 32 * x2), xmask,
        eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 * x0 + 8 * x1 + 32 * x2), xmask,
        eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + 2 * x0 + 8 * x1 + 32 * x2), xmask,
        eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (4 + 2 * x0 + 8 * x1 + 32 * x2), xmask,
        eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (5 + 2 * x0 + 8 * x1 + 32 * x2), xmask,
        eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (2 * x0 + 8 * x1 + 32 * x2), xmask,
        eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (1 + 2 * x0 + 8 * x1 + 32 * x2), xmask,
        eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (4 + 2 * x0 + 8 * x1 + 32 * x2), xmask,
        eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (5 + 2 * x0 + 8 * x1 + 32 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp1 + tmp1
    tmp4 = tmp3 + tmp3
    tmp6 = tmp5 + tmp5
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp21 = 4.0
    tmp22 = tmp20 / tmp21
    tl.store(out_ptr0 + x3, tmp22, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (128, 32, 256, 256), (2097152, 65536, 256,
        1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_4, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 1), (1, 1), torch.float32)
        buf2 = empty_strided_cuda((128, 1), (1, 128), torch.float32)
        buf4 = empty_strided_cuda((128, 32, 256, 256), (2097152, 65536, 256,
            1), torch.float32)
        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_relu_0[grid(128)](primals_1,
            buf0, buf2, 128, 1024, XBLOCK=1, num_warps=2, num_stages=1)
        buf3 = empty_strided_cuda((128, 1), (1, 1), torch.float32)
        triton_poi_fused__native_batch_norm_legit_1[grid(128)](primals_1,
            primals_2, buf2, buf3, 128, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf5 = empty_strided_cuda((128, 64, 2, 2), (256, 4, 2, 1), torch.
            float32)
        triton_poi_fused__native_batch_norm_legit_2[grid(512)](buf4, buf3,
            buf2, primals_3, primals_4, buf5, 512, XBLOCK=256, num_warps=4,
            num_stages=1)
        del buf2
        del primals_4
        buf6 = empty_strided_cuda((128, 64, 256, 256), (4194304, 65536, 256,
            1), torch.float32)
        triton_poi_fused_avg_pool2d_3[grid(8192)](buf5, buf6, 8192, XBLOCK=
            128, num_warps=4, num_stages=1)
        del buf5
    return buf6, primals_1, primals_3, buf0, buf3, buf4


class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(ModelNew, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, input_0):
        primals_3 = self.transition.conv2d.weight
        primals_2 = self.transition.batch_norm.weight
        primals_4 = self.transition.batch_norm.bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
