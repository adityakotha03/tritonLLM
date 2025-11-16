import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_group_norm_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2,
    out_ptr3, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp5 = tl.load(in_ptr0 + (512 + x0), xmask)
    tmp11 = tl.load(in_ptr0 + (1024 + x0), xmask)
    tmp17 = tl.load(in_ptr0 + (1536 + x0), xmask)
    tmp2 = 1e-05
    tmp3 = tmp0 * tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 * tmp5
    tmp7 = tmp6 + tmp2
    tmp8 = tmp4 + tmp7
    tmp9 = tmp0 - 0.0
    tmp10 = tmp9 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp12 + tmp2
    tmp14 = tmp10 + tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tmp10 / tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp18 + tmp2
    tmp20 = tmp16 + tmp19
    tmp21 = tmp14 / tmp20
    tmp22 = tmp16 - tmp21
    tl.store(out_ptr0 + x0, tmp21, xmask)
    tl.store(out_ptr1 + x0, tmp22, xmask)
    tl.store(out_ptr2 + x0, tmp15, xmask)
    tl.store(out_ptr3 + x0, tmp20, xmask)


@triton.jit
def triton_poi_fused_add_min_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (2048 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (3072 + x0), xmask)
    tmp7 = tl.load(in_ptr0 + (4096 + x0), xmask)
    tmp9 = tl.load(in_ptr0 + (5120 + x0), xmask)
    tmp11 = tl.load(in_ptr0 + (6208 + x0), xmask)
    tmp13 = tl.load(in_ptr0 + (7232 + x0), xmask)
    tmp15 = tl.load(in_ptr0 + (8256 + x0), xmask)
    tmp17 = tl.load(in_ptr0 + (9280 + x0), xmask)
    tmp19 = tl.load(in_ptr0 + (10240 + x0), xmask)
    tmp2 = tl.where(xmask, tmp1, tmp0)
    tmp4 = tl.where(xmask, tmp3, tmp2)
    tmp6 = tl.where(xmask, tmp5, tmp4)
    tmp8 = tl.where(xmask, tmp7, tmp6)
    tmp10 = tl.where(xmask, tmp9, tmp8)
    tmp12 = tl.where(xmask, tmp11, tmp10)
    tmp14 = tl.where(xmask, tmp13, tmp12)
    tmp16 = tl.where(xmask, tmp15, tmp14)
    tmp18 = tl.where(xmask, tmp17, tmp16)
    tmp20 = tl.where(xmask, tmp19, tmp18)
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tl.store(out_ptr0 + x0, tmp22, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1, 8192, 1, 1), (8192, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.addmm(primals_2, reinterpret_tensor(primals_1, (1024,
            8192), (8192, 1), 0), reinterpret_tensor(primals_3, (8192, 1),
            (1, 8192), 0), alpha=1, beta=1, out=buf0)
        del primals_1
        del primals_2
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        buf2 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        buf3 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_group_norm_0[grid(1024)](buf0, buf1, buf2, buf3, 
            1024, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        buf4 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_add_min_1[grid(1024)](buf3, buf4, 1024, XBLOCK=128,
            num_warps=4, num_stages=1)
        del buf3
    return buf4, primals_3, buf1, buf2, reinterpret_tensor(buf4, (1024, 8192),
        (8192, 1), 0)


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, Group Normalization, Minimum operation, and Bias addition.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, input_0):
        primals_1 = self.gemm.weight
        primals_2 = self.gemm.bias
        primals_3 = self.bias
        output = call([primals_1, primals_2, primals_3])
        return output[0]
