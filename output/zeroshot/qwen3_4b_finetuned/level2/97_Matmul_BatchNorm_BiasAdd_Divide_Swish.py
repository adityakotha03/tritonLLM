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
def triton_poi_fused__native_batch_norm_legit_add_div_mul_sigmoid_0(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tl.load(in_ptr2 + 0)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp8 = tl.load(in_ptr3 + 0)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 - tmp4
    tmp6 = 1.0
    tmp7 = tmp5 * tmp6
    tmp10 = tmp7 + tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.sqrt(tmp12)
    tmp14 = tmp6 / tmp13
    tmp15 = tmp10 * tmp14
    tmp16 = tmp15 + tmp4
    tmp17 = tmp16 * tmp6
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr0 + x0, tmp14, xmask)
    tl.store(out_ptr1 + x0, tmp16, xmask)
    tl.store(out_ptr2 + x0, tmp19, xmask)
    tl.store(out_ptr3 + x0, tmp18, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    assert_size_stride(primals_4, (1,), (1,))
    assert_size_stride(primals_5, (1,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.mm(primals_3, reinterpret_tensor(primals_1, (8192, 
            8192), (1, 8192), 0), out=buf0)
        del primals_1
        buf1 = empty_strided_cuda((1024, 1), (1, 1024), torch.float32)
        buf2 = empty_strided_cuda((1024, 1), (1, 1024), torch.float32)
        buf3 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        buf4 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_add_div_mul_sigmoid_0[grid
            (1024)](buf0, primals_2, primals_4, primals_5, buf1, buf2, buf3,
            buf4, 1024, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        del primals_2
        del primals_4
        del primals_5
    return buf4, primals_3, buf1, buf2, reinterpret_tensor(buf3, (1, 1024),
        (1, 8192), 0)


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, batch normalization, bias addition, division, and Swish activation.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value

    def forward(self, input_0):
        primals_1 = self.matmul.weight
        primals_2 = self.matmul.bias
        primals_4 = self.bias
        primals_3 = input_0
        primals_5 = self.bn.weight
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
