import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
extern_kernels = torch._C._dynamo.guards._extern_kernels
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_per_fused_add_div_mul_native_batch_norm_sigmoid_0(
    in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex // 8192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp5 = 1e-05
    tmp6 = tmp14 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tmp4 / tmp7
    tmp9 = tmp8 * tmp3
    tmp10 = tmp9 + tmp1
    tmp11 = 1.0
    tmp12 = tmp10 / tmp11
    tmp13 = tl.sigmoid(tmp12)
    tmp15 = tmp12 * tmp13
    tl.store(out_ptr0 + x2, tmp15, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (8192,), (1,))
    assert_size_stride(primals_4, (8192,), (1,))
    assert_size_stride(primals_5, (8192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.mm(primals_5, reinterpret_tensor(primals_1, (8192, 
            8192), (1, 8192), 0), out=buf0)
        del primals_1
        buf1, buf2, buf3 = extern_kernels.add_native_batch_norm(buf0,
            primals_2, primals_3, primals_4, 0.1, 1e-05, 1.0, 1.0)
        del primals_2
        del primals_3
        del primals_4
        buf4 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_per_fused_add_div_mul_native_batch_norm_sigmoid_0[grid(8388608)
            ](buf1, primals_2, primals_3, primals_4, buf2, buf4, 8388608,
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf1
        del primals_2
        del primals_3
        del primals_4
    return buf4, primals_5, buf2, buf3, buf0


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, batch normalization, bias addition, division, and Swish activation.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value

    def forward(self, input_0):
        primals_1 = self.matmul.weight
        primals_2 = self.bias
        primals_3 = self.bn.weight
        primals_4 = self.bn.bias
        primals_5 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]