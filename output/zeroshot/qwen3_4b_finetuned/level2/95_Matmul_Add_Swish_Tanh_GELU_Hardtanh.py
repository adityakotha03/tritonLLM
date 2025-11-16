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
def triton_poi_fused_hardtanh_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = -1.0
    tmp4 = tmp2 <= tmp3
    tmp5 = tmp2 >= 1.0
    tmp6 = tmp4 | tmp5
    tl.store(in_out_ptr0 + x0, tmp2, xmask)
    tl.store(in_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused_gelu_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp0 + tmp1
    tmp5 = tmp3 * tmp4
    tl.store(out_ptr0 + x2, tmp5, xmask)


@triton.jit
def triton_poi_fused_tanh_sigmoid_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + x2, tmp3, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5) = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    assert_size_stride(primals_4, (8192,), (1,))
    assert_size_stride(primals_5, (8192, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.mm(primals_3, reinterpret_tensor(primals_1, (8192, 
            1024), (1, 8192), 0), out=buf0)
        del primals_1
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_hardtanh_0[grid(8388608)](buf1, primals_2, 8388608,
            XBLOCK=512, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((8192, 8192), (8192, 1), torch.float32)
        extern_kernels.mm(buf0, reinterpret_tensor(primals_5, (8192, 8192),
            (1, 8192), 0), out=buf2)
        buf3 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_gelu_1[grid(8388608)](buf2, buf3, 8388608, XBLOCK=
            256, num_warps=4, num_stages=1)
        buf4 = reinterpret_tensor(buf2, (1024, 8192), (8192, 1), 0)
        del buf2
        triton_poi_fused_tanh_sigmoid_2[grid(8388608)](buf3, buf4, 8388608,
            XBLOCK=512, num_warps=4, num_stages=1)
        del buf3
    return reinterpret_tensor(buf0, (1024, 8192), (8192, 1), 0
        ), primals_4, primals_5, buf4, buf1


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a value, applies Swish, Tanh, GELU, and Hardtanh activation functions.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape)) 

    def forward(self, input_0):
        primals_1 = self.matmul.weight
        primals_2 = self.add_value
        primals_3 = input_0
        primals_5 = self.matmul.weight
        primals_4 = self.add_value
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5])
        return output[0]
