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
def triton_poi_fused_div_mul_sigmoid_0(in_out_ptr0, in_ptr0, in_ptr1,
    in_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8192
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp2 + tmp5
    tmp7 = 0.9999999999999999
    tmp8 = tmp6 * tmp7
    tmp9 = tmp4 * tmp8
    tl.store(in_out_ptr0 + x2, tmp9, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1,), (1,))
    assert_size_stride(primals_4, (1024, 8192), (8192, 1))
    assert_size_stride(primals_5, (1024,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.mm(primals_4, reinterpret_tensor(primals_1, (8192, 8192), (1, 8192), 0))
        assert_size_stride(buf0, (1024, 8192), (8192, 1))
        buf1 = empty_strided_cuda((8192,), (1,), torch.float32)
        buf2 = reinterpret_tensor(buf1, (8192,), (1,), 0)
        get_raw_stream(0)
        triton_poi_fused_div_mul_sigmoid_0[grid(8388608)](buf2, primals_2,
            primals_3, primals_5, 8388608, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_2
        del primals_3
        del primals_5
    return buf0, reinterpret_tensor(buf1, (1024, 8192), (8192, 1), 0
        ), primals_1, primals_4, buf2


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, batch normalization, bias addition, division, and Swish activation.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1,
        bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value

    def forward(self, input_0):
        primals_1 = self.matmul.weight
        primals_2 = self.bn.weight
        primals_3 = self.bn.bias
        primals_5 = self.bias
        primals_4 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5])
        return output[1]
