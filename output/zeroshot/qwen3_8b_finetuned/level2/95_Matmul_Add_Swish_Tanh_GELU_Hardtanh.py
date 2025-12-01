import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_hardtanh_sigmoid_tanh_gelu_swish_0(
    in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = -tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tl_math.sigmoid(tmp7)
    tmp9 = tmp3 * tmp8
    tmp10 = 0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11 * tmp9
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14 * tmp10
    tmp16 = tl_math.tanh(tmp15)
    tmp17 = tmp16 * tmp15
    tmp18 = 1.4142135623731027
    tmp19 = tmp17 / tmp18
    tmp20 = libdevice.erf(tmp19)
    tmp21 = tmp20 + tmp13
    tmp22 = tmp17 * tmp21
    tmp23 = -1.0
    tmp24 = triton_helpers.minimum(tmp22, tmp16)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(out_ptr0 + x0, tmp25, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_hardtanh_sigmoid_tanh_gelu_swish_0[grid(8388608)](
            primals_3, primals_2, buf0, 8388608, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
    return buf0, primals_1, primals_3


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
        output = call([primals_1, primals_2, primals_3])
        return output[0]