import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn.functional as F
import triton.language as tl
import triton
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_leaky_relu_log_sum_exp_0(in_out_ptr0, in_ptr0,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0, None))
    tmp7 = tmp3 - tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0, None))
    tmp12 = libdevice.log(tmp11)
    tmp13 = tmp7 - tmp12
    tmp14 = 0.01
    tmp15 = tmp13 * tmp14
    tmp16 = tmp13 > 0.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp13 <= 0.0
    tmp19 = tmp13 * tmp18
    tmp20 = tmp17 + tmp19
    tl.store(in_out_ptr0 + x0, tmp20, xmask)
    tl.store(out_ptr0 + x0, tmp12, xmask)


@triton.jit
def triton_poi_fused_leaky_relu_1(in_out_ptr0, in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.01
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 > tmp5
    tmp7 = tmp2 * tmp6
    tmp8 = tmp4 <= tmp5
    tmp9 = tmp4 * tmp8
    tmp10 = tmp7 + tmp9
    tl.store(in_out_ptr0 + x2, tmp10, xmask)
    tl.store(out_ptr0 + x2, tmp10, xmask)


@triton.jit
def triton_poi_fused_gelu_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + x0, tmp10, xmask)


@triton.jit
def triton_poi_fused_gelu_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + x2, tmp10, xmask)


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
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((1024, 1), (1, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_leaky_relu_log_sum_exp_0[grid(1024)](buf1,
            primals_2, buf2, 1024, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_2
        buf3 = buf1
        del buf1
        buf4 = empty_strided_cuda((1024, 1), (1, 1), torch.float32)
        triton_poi_fused_leaky_relu_1[grid(1024)](buf3, buf2, buf4, 1024,
            XBLOCK=128, num_warps=1, num_stages=1)
        buf5 = buf2
        del buf2
        triton_poi_fused_gelu_2[grid(1024)](buf3, buf4, 1024, XBLOCK=128,
            num_warps=1, num_stages=1)
        buf6 = buf4
        del buf4
        triton_poi_fused_gelu_3[grid(1024)](buf3, buf6, 1024, XBLOCK=128,
            num_warps=1, num_stages=1)
    return buf3, buf3, primals_3, buf6, buf5, buf3


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), followed by LogSumExp, LeakyReLU, 
    LeakyReLU, GELU, and GELU activations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input_0):
        primals_1 = self.linear.weight
        primals_2 = self.linear.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]