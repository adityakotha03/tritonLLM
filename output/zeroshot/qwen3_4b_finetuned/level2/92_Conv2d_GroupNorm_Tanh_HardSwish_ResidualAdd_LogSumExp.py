import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16384 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_native_group_norm_1(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 16 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (3 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (4 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (5 + 16 * x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (6 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (8 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (9 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (10 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (12 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (13 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (14 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (15 + 16 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp24 = tmp23 + tmp22
    tmp26 = tmp25 + tmp24
    tmp28 = tmp27 + tmp26
    tmp30 = tmp29 + tmp28
    tmp31 = 16.0
    tmp32 = tmp30 / tmp31
    tmp33 = tmp1 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tmp0 - tmp32
    tmp36 = tmp35 * tmp35
    tmp37 = tmp34 + tmp36
    tmp38 = tmp3 - tmp32
    tmp39 = tmp38 * tmp38
    tmp40 = tmp37 + tmp39
    tmp41 = tmp5 - tmp32
    tmp42 = tmp41 * tmp41
    tmp43 = tmp40 + tmp42
    tmp44 = tmp7 - tmp32
    tmp45 = tmp44 * tmp44
    tmp46 = tmp43 + tmp45
    tmp47 = tmp9 - tmp32
    tmp48 = tmp47 * tmp47
    tmp49 = tmp46 + tmp48
    tmp50 = tmp11 - tmp32
    tmp51 = tmp50 * tmp50
    tmp52 = tmp49 + tmp51
    tmp53 = tmp13 - tmp32
    tmp54 = tmp53 * tmp53
    tmp55 = tmp52 + tmp54
    tmp56 = tmp15 - tmp32
    tmp57 = tmp56 * tmp56
    tmp58 = tmp55 + tmp57
    tmp59 = tmp17 - tmp32
    tmp60 = tmp59 * tmp59
    tmp61 = tmp58 + tmp60
    tmp62 = tmp19 - tmp32
    tmp63 = tmp62 * tmp62
    tmp64 = tmp61 + tmp63
    tmp65 = tmp21 - tmp32
    tmp66 = tmp65 * tmp65
    tmp67 = tmp64 + tmp66
    tmp68 = tmp23 - tmp32
    tmp69 = tmp68 * tmp68
    tmp70 = tmp67 + tmp69
    tmp71 = tmp25 - tmp32
    tmp72 = tmp71 * tmp71
    tmp73 = tmp70 + tmp72
    tmp74 = tmp27 - tmp32
    tmp75 = tmp74 * tmp74
    tmp76 = tmp73 + tmp75
    tmp77 = tmp29 - tmp32
    tmp78 = tmp77 * tmp77
    tmp79 = tmp76 + tmp78
    tmp80 = 15.0
    tmp81 = tmp79 / tmp80
    tl.store(out_ptr0 + x0, tmp32, xmask)
    tl.store(out_ptr1 + x0, tmp81, xmask)


@triton.jit
def triton_poi_fused_native_group_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 16
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_hardtanh_hardswish_3(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = 1.0
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = 7.0
    tmp6 = tmp4 * tmp5
    tmp7 = 0.0
    tmp8 = tmp6 > tmp7
    tmp9 = tl_math.abs(tmp4)
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl_math.log(tmp11)
    tmp13 = 0.0
    tmp14 = tl.where(tmp8, tmp6, tmp13)
    tmp15 = tmp14 - tmp12
    tmp16 = tmp15 * tmp15
    tmp17 = tmp16 * tmp15
    tmp18 = tl.where(tmp8, tmp14, tmp13)
    tmp19 = triton_helpers.maximum(tmp18, tmp13)
    tl.store(out_ptr0 + x0, tmp19, xmask)


@triton.jit
def triton_per_fused_add_logsumexp_4(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    rnumel, XBLOCK: tl.constexpr):
    xnumel = 64
    rnumel = 16384
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 16384 * x0), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 16384 * x0), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, float('-inf'))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tl_math.log(tmp12)
    tmp14 = tmp6 - tmp13
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x0, tmp14, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 8, 128, 128), (131072, 16384, 128,
        1))
    assert_size_stride(primals_4, (64,), (1,))
    assert_size_stride(primals_5, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128,
            1), torch.float32)
        buf1 = reinterpret_tensor(buf0, (128, 64, 128, 128), (1048576, 16384,
            128, 1), 0)
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(1048576)](buf1, primals_1, 
            1048576, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_1
        buf2 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        buf3 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32
            )
        triton_poi_fused_native_group_norm_1[grid(32)](buf1, buf2, buf3, 32,
            XBLOCK=32, num_warps=1, num_stages=1)
        buf4 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128,
            1), torch.float32)
        triton_poi_fused_native_group_norm_2[grid(4096)](buf1, buf2, buf3,
            primals_2, primals_4, buf4, 4096, XBLOCK=256, num_warps=4,
            num_stages=1)
        del buf2
        del buf3
        del primals_4
        buf5 = reinterpret_tensor(buf1, (128, 64, 128, 128), (1048576, 16384,
            128, 1), 0)
        del buf1
        triton_poi_fused_hardtanh_hardswish_3[grid(1048576)](buf4, buf5, 
            1048576, XBLOCK=1024, num_warps=4, num_stages=1)
        buf6 = reinterpret_tensor(buf4, (128, 64, 1, 1), (64, 1, 64, 64), 0)
        del buf4
        triton_per_fused_add_logsumexp_4[grid(64)](buf6, buf5, primals_3, 
            64, 16384, XBLOCK=8, num_warps=2, num_stages=1)
        del primals_3
    return buf6, primals_2, buf5, primals_5


class ModelNew(nn.Module):
    """
    Model that performs a convolution, applies Group Normalization, Tanh, HardSwish, 
    Residual Addition, and LogSumExp.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_4 = self.group_norm.weight
        primals_5 = self.group_norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
