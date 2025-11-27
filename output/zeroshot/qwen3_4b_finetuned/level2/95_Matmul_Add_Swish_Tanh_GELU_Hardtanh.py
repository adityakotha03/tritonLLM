import torch
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
def triton_poi_fused_add_hardtanh_mul_sigmoid_tanh_0(in_ptr0, in_ptr1,
    out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp3 * tmp2
    tmp5 = libdevice.tanh(tmp4)
    tmp6 = 0.5
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tmp7 <= tmp8
    tmp10 = 1.0
    tmp11 = tmp7 >= tmp10
    tmp12 = tmp9 | tmp11
    tmp13 = -1.0
    tmp14 = tmp7 >= tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp5 * tmp5
    tmp17 = 0.7071067811865476
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18 * tmp5
    tmp20 = tmp19 + tmp6
    tmp21 = tl.where(tmp15, tmp20, tmp5)
    tmp22 = tl.where(tmp15, tmp10, tmp13)
    tmp23 = tl.where(tmp15, tmp21, tmp22)
    tl.store(out_ptr0 + x0, tmp4, xmask)
    tl.store(out_ptr1 + x0, tmp23, xmask)


@triton.jit
def triton_poi_fused_add_hardtanh_mul_sigmoid_tanh_1(in_ptr0, in_ptr1,
    out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp3 * tmp2
    tmp5 = libdevice.tanh(tmp4)
    tmp6 = 0.5
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tmp7 <= tmp8
    tmp10 = 1.0
    tmp11 = tmp7 >= tmp10
    tmp12 = tmp9 | tmp11
    tmp13 = -1.0
    tmp14 = tmp7 >= tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp5 * tmp5
    tmp17 = 0.7071067811865476
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18 * tmp5
    tmp20 = tmp19 + tmp6
    tmp21 = tl.where(tmp15, tmp20, tmp5)
    tmp22 = tl.where(tmp15, tmp10, tmp13)
    tmp23 = tl.where(tmp15, tmp21, tmp22)
    tl.store(out_ptr0 + x0, tmp4, xmask)
    tl.store(out_ptr1 + x0, tmp23, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    assert_size_stride(primals_4, (8192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_hardtanh_mul_sigmoid_tanh_0[grid(8388608)](
            primals_3, primals_4, buf0, primals_1, 8388608, XBLOCK=1024,
            num_warps=4, num_stages=1)
        del primals_4
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_add_hardtanh_mul_sigmoid_tanh_1[grid(8388608)](
            primals_3, primals_1, buf1, buf0, 8388608, XBLOCK=1024,
            num_warps=4, num_stages=1)
        del primals_1
        del primals_3
    return reinterpret_tensor(buf1, (1024, 8192), (8192, 1), 0
        ), reinterpret_tensor(buf0, (8192, 1024), (1, 8192), 0)


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
        primals_2 = self.matmul.bias
        primals_4 = self.add_value
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
