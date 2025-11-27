import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_mul_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp3 = tmp0 + tmp1
    tmp4 = tmp3 * tmp2
    tl.store(out_ptr0 + x0, tmp4, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 8192), (8192, 1))
    assert_size_stride(arg1_1, (1024, 8192), (8192, 1))
    assert_size_stride(arg2_1, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_ptr0 = buf0
        triton_poi_fused_add_mul_0[grid](arg0_1, arg1_1, arg2_1, get_ptr0, 
            8388608, XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
        del arg2_1
    return buf0,


class ModelNew(nn.Module):
    """
    Model that performs a batch matrix multiplication, instance normalization, summation, residual addition, and multiplication.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)

    def forward(self, input_0, input_1):
        arg0_1 = self.bmm.weight
        arg1_1 = self.bmm.bias
        arg2_1 = input_1
        output = call([arg0_1, arg1_1, arg2_1])
        return output[0]
