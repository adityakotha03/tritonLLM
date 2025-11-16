import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_clamp_div_mul_sigmoid_threshold_backward_tanh_0(in_out_ptr0,
    in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8192
    x1 = xindex // 8192
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 1.0
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = tmp5 > tmp6
    tmp8 = 1.0
    tmp9 = tmp5 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tl.where(tmp10, tmp5, tmp2)
    tmp12 = 2.0
    tmp13 = tmp11 / tmp12
    tmp14 = -1.0
    tmp15 = tmp13 < tmp14
    tmp16 = 1.0
    tmp17 = tmp13 > tmp16
    tmp18 = tmp15 | tmp17
    tmp19 = tl.where(tmp18, tmp13, tmp11)
    tmp20 = tmp2 > tmp4
    tmp21 = tl.where(tmp20, tmp2, tmp1)
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = tmp20 & tmp22
    tmp24 = tl.where(tmp23, tmp21, tmp19)
    tmp25 = tl.tanh(tmp24)
    tmp26 = tmp25 < tmp14
    tmp27 = tmp26 | tmp18
    tmp28 = tl.where(tmp27, tmp25, tmp24)
    tl.store(in_out_ptr0 + x2, tmp19, xmask)
    tl.store(out_ptr0 + x2, tmp28, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8192, 8192), (8192, 1))
    assert_size_stride(arg1_1, (8192,), (1,))
    assert_size_stride(arg2_1, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.mm(arg2_1, arg0_1)
        assert_size_stride(buf0, (1024, 8192), (8192, 1))
        buf1 = buf0
        del buf0
        buf4 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clamp_div_mul_sigmoid_threshold_backward_tanh_0[grid(8
            * 1024)](buf1, arg1_1, arg2_1, buf4, 8388608, XBLOCK=1024,
            num_warps=4, num_stages=1)
        del arg1_1
        del arg2_1
    return buf4, buf1


class ModelNew(nn.Module):
    """
    Simple model that performs a gemm, swish, divide, clamp, tanh, and clamp operations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input_0):
        arg0_1 = self.gemm.weight
        arg1_1 = self.gemm.bias
        arg2_1 = input_0
        output = call([arg0_1, arg1_1, arg2_1])
        return output[0]
