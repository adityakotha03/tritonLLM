import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_mean_rsub_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp1 - tmp8
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 - tmp8
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = tmp5 - tmp8
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19 / tmp7
    tl.store(out_ptr0 + x0, tmp20, xmask)


@triton.jit
def triton_poi_fused_div_mul_sub_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 8192
    x0 = xindex % 8192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = 1e-05
    tmp9 = tmp6 + tmp8
    tmp10 = tl_math.rsqrt(tmp9)
    tmp11 = tmp7 * tmp10
    tl.store(out_ptr0 + x2, tmp11, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8) = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    assert_size_stride(primals_4, (8192,), (1,))
    assert_size_stride(primals_5, (8192,), (1,))
    assert_size_stride(primals_6, (8192,), (1,))
    assert_size_stride(primals_7, (8192,), (1,))
    assert_size_stride(primals_8, (8192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_3, (1024, 8192), (1, 
            8192), 0), reinterpret_tensor(primals_1, (8192, 8192), (1, 8192
            ), 0), out=buf0)
        del primals_1
        del primals_3
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(8388608)](buf0, primals_2, buf1, 
            8388608, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((1, 8192), (8192, 1), torch.float32)
        triton_poi_fused_mean_rsub_1[grid(8192)](buf1, buf2, 8192, XBLOCK=
            128, num_warps=4, num_stages=1)
        buf3 = empty_strided_cuda((1, 8192), (8192, 1), torch.float32)
        triton_poi_fused_mean_rsub_1[grid(8192)](buf1, buf3, 8192, XBLOCK=
            128, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_div_mul_sub_2[grid(8388608)](buf1, primals_4,
            buf2, primals_5, primals_6, buf4, 8388608, XBLOCK=1024,
            num_warps=4, num_stages=1)
        del buf2
        del primals_5
        del primals_6
        buf5 = empty_strided_cuda((1, 8192), (8192, 1), torch.float32)
        triton_poi_fused_mean_rsub_1[grid(8192)](buf4, buf5, 8192, XBLOCK=
            128, num_warps=4, num_stages=1)
        buf6 = empty_strided_cuda((1, 8192), (8192, 1), torch.float32)
        triton_poi_fused_mean_rsub_1[grid(8192)](buf4, buf6, 8192, XBLOCK=
            128, num_warps=4, num_stages=1)
        buf7 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_div_mul_sub_2[grid(8388608)](buf4, primals_4,
            buf5, primals_7, primals_8, buf7, 8388608, XBLOCK=1024,
            num_warps=4, num_stages=1)
        del buf5
        del primals_7
        del primals_8
    return buf7, reinterpret_tensor(primals_4, (1, 8192), (1, 8192), 0
        ), reinterpret_tensor(buf1, (1, 8192), (1, 8192), 0
        ), reinterpret_tensor(buf4, (1, 8192), (1, 8192), 0
        ), reinterpret_tensor(primals_4, (8192,), (1, 8192), 0
        ), reinterpret_tensor(buf6, (1, 8192), (1, 8192), 0
        ), buf0, primals_4, primals_5, primals_6, primals_8


class ModelNew(nn.Module):
    """
    Simple model that performs a GEMM (general matrix multiplication), applies scaling, 
    and then batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, input_0):
        primals_1 = self.gemm.weight
        primals_2 = self.gemm.bias
        primals_4 = self.scale
        primals_5 = self.bn.running_mean
        primals_6 = self.bn.running_var
        primals_7 = self.bn.weight
        primals_8 = self.bn.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8])
        return output[0]