import torch
from torch._inductor.select_algorithm import extern_kernels
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
def triton_poi_fused_gelu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8398080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8398080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 8192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 8192 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (8193 + 8192 * x1), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (16385 + 8192 * x1), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (24577 + 8192 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + x2, tmp9, xmask)


@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8398080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 8192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 8192 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (8193 + 8192 * x1), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (16385 + 8192 * x1), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (24577 + 8192 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.addmm(primals_2, primals_3, reinterpret_tensor(
            primals_1, (8192, 8192), (1, 8192), 0), alpha=1, beta=1, out=buf0)
        del primals_1
        del primals_2
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_gelu_0[grid(8398080)](buf0, buf1, 8398080, XBLOCK=
            1024, num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused__softmax_1[grid(8398080)](buf1, buf2, 8398080,
            XBLOCK=512, num_warps=8, num_stages=1)
        buf3 = reinterpret_tensor(buf1, (1024, 8192), (8192, 1), 0)
        del buf1
        triton_poi_fused__softmax_2[grid(8398080)](buf2, buf3, 8398080,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del buf2
    return buf3, primals_3, buf0


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input_0):
        primals_1 = self.linear.weight
        primals_2 = self.linear.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
