import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused__unsafe_index_add_mul_0(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + 0)
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + 0)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp6 = tl.load(in_ptr3 + 0)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp8 = tl.load(in_ptr0 + x0, xmask)
    tmp9 = tl.load(in_ptr1 + x0, xmask)
    tmp10 = tmp7 + tmp8
    tmp11 = tmp1 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp5
    tl.store(out_ptr0 + x0, tmp13, xmask)
    tmp14 = tl.load(in_ptr0 + x0, xmask)
    tmp15 = tmp3 + tmp14
    tmp16 = tmp15 * tmp9
    tl.store(out_ptr0 + (8192 + x0), tmp16, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    assert_size_stride(primals_4, (8192,), (1,))
    assert_size_stride(primals_5, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_3, (1024, 8192), (8192,
            1), 0), reinterpret_tensor(primals_1, (8192, 8192), (1, 8192), 
            0), out=buf0)
        del primals_1
        buf1 = reinterpret_tensor(buf0, (1024, 1, 1, 8192), (8192, 8192, 1, 
            1), 0)
        del buf0
        get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_0[grid(8192)](primals_4,
            primals_5, buf1, primals_2, buf1, 8192, XBLOCK=256, num_warps=4,
            num_stages=1)
        del primals_2
        del primals_4
    return buf1, reinterpret_tensor(primals_3, (1024, 8192), (8192, 1), 0
        ), primals_5, buf1


class ModelNew(nn.Module):
    """
    Model that performs a batch matrix multiplication, instance normalization, summation, residual addition, and multiplication.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)

    def forward(self, input_0, input_1):
        primals_1 = self.bmm.weight
        primals_2 = self.bmm.bias
        primals_3 = input_0
        primals_4 = self.instance_norm.weight
        primals_5 = self.instance_norm.bias
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
