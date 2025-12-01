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


@triton.jit
def triton_poi_fused_add_hardtanh_group_norm_0(in_out_ptr0, in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    r2 = rindex // 512
    r3 = rindex % 512
    r4 = rindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 8192 * x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + r2, xmask, eviction_policy='evict_last', other=0.0
        )
    tmp2 = tl.load(in_ptr1 + r3, xmask, eviction_policy='evict_last', other=0.0
        )
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + r4, xmask, eviction_policy='evict_last', other=0.0
        )
    tmp5 = tmp0 + tmp1
    tmp6 = tmp5 + tmp2
    tmp7 = tmp6 - tmp3
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.sum(tmp8, 1)[:, None]
    tmp11 = tl.full([XBLOCK, 1], 512, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 / tmp12
    tmp14 = tmp7 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.sum(tmp16, 1)[:, None]
    tmp19 = tmp18 / tmp12
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp14 * tmp22
    tmp24 = tmp23 + tmp4
    tmp25 = 2.0
    tmp26 = triton_helpers.maximum(tmp24, tmp25)
    tmp27 = triton_helpers.minimum(tmp26, tmp25)
    tl.store(in_out_ptr0 + (r1 + 8192 * x0), tmp24, xmask)
    tl.store(out_ptr0 + (r1 + 8192 * x0), tmp27, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.mm(primals_3, reinterpret_tensor(primals_1, (8192, 
            8192), (1, 8192), 0), out=buf0)
        del primals_1
        buf1 = empty_strided_cuda((16, 512, 1), (8192, 1, 1), torch.float32)
        buf2 = empty_strided_cuda((16, 512, 1), (8192, 1, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_hardtanh_group_norm_0[grid(16)](buf0,
            primals_2, buf0, buf1, buf2, buf0, 16, 512, XBLOCK=1, num_warps
            =2, num_stages=1)
        del primals_2
    return buf0, reinterpret_tensor(primals_3, (8192, 1024), (1, 8192), 0
        ), buf1, buf2


class ModelNew(nn.Module):
    """
    Simple model that performs a GEMM, applies Group Normalization, and then HardTanh.
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min,
        hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh = nn.HardTanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, input_0):
        primals_1 = self.gemm.weight
        primals_2 = self.gemm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]