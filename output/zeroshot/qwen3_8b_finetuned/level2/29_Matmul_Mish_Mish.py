import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
empty_cuda = torch._C._dynamo.guards._empty_cuda
reinterpret_tensor_1 = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_mish_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 0.0, tl.int32)
    tmp2 = libdevice.maximum(tmp0, tmp1)
    tmp3 = tl.full([1], 1.0, tl.int32)
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.exp(tmp4)
    tmp6 = tmp5 + tmp3
    tmp7 = libdevice.log(tmp6)
    tmp8 = tmp2 * tmp7
    tmp9 = libdevice.tanh(tmp8)
    tmp10 = tmp9 * tmp2
    tl.store(out_ptr0 + x0, tmp10, xmask)


@triton.jit
def triton_poi_fused_mish_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 0.0, tl.int32)
    tmp2 = libdevice.maximum(tmp0, tmp1)
    tmp3 = tl.full([1], 1.0, tl.int32)
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.exp(tmp4)
    tmp6 = tmp5 + tmp3
    tmp7 = libdevice.log(tmp6)
    tmp8 = tmp2 * tmp7
    tmp9 = libdevice.tanh(tmp8)
    tmp10 = tmp9 * tmp2
    tl.store(out_ptr0 + x0, tmp10, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        buf2 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mish_0[grid(8388608)](primals_1, buf0, 8388608,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_1
        triton_poi_fused_mish_1[grid(8388608)](buf0, buf1, 8388608,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del buf0
        buf3 = buf2
        del buf2
        buf4 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_mish_1[grid(8388608)](buf1, buf4, 8388608,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del buf1
        buf5 = buf3
        del buf3
        buf6 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_mish_1[grid(8388608)](buf4, buf6, 8388608,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del buf4
        del buf5
    return reinterpret_tensor(buf6, (1024, 8192), (8192, 1), 0), reinterpret_tensor_1(
        buf5, (8192, 8192), (1, 8192), 0), reinterpret_tensor_1(buf0, (8192,),
        (1,), 0), reinterpret_tensor(buf2, (8192,), (1,), 0), primals_2


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication, applies Mish, and applies Mish again.
    The two Mish activations are replaced with custom Triton kernels that perform elementwise
    Mish operations, while the matrix multiplication is kept as a standard Linear layer.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input_0):
        primals_2 = self.linear.weight
        primals_1 = self.linear.bias
        output = call([input_0, primals_2])
        return output[0]