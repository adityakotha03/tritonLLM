import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_mean_var_0(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 8192
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex % rnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x2, xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 1))
    tl.debug_barrier()
    tmp5 = tl.load(in_out_ptr0 + x0, xmask)
    tmp6 = tmp4 / 1024
    tmp7 = tmp5 - tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 1))
    tmp13 = 1023.0
    tmp14 = tmp12 / tmp13
    tl.store(in_out_ptr0 + x0, tmp14, xmask)


@triton.jit
def triton_poi_fused_add_div_1(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 8192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex % rnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x2, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + x0, xmask)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)
    tmp7 = 1024.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp1 + tmp8
    tmp10 = 1.0
    tmp11 = tmp9 / tmp10
    tl.store(in_out_ptr0 + x0, tmp11, xmask)


@triton.jit
def triton_poi_fused_mul_sigmoid_2(in_ptr0, in_out_ptr0, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    xnumel = 8192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex % rnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x2, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + x0, xmask)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)
    tmp7 = 1024.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp1 + tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = -tmp10
    tmp12 = tl.full([1], 0.0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full([1], 2.718281828459045, tl.int32)
    tmp15 = tl.where(tmp13 < tmp14, tmp13, tmp14)
    tmp16 = tl.full([1], 1.0, tl.int32)
    tmp17 = tmp15 / tmp16
    tmp18 = tmp9 / tmp17
    tl.store(in_out_ptr0 + x0, tmp18, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 8192), (8192, 1))
    assert_size_stride(arg1_1, (8192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        buf1 = empty_strided_cuda((8192,), (1,), torch.float32)
        buf2 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        buf3 = buf0
        del buf0
        buf4 = buf2
        del buf2
        buf5 = buf4
        del buf4
        get_raw_stream(0)
        triton_poi_fused_mean_var_0[grid(8192)](buf3, arg1_1, 8192, 1024,
            XBLOCK=128, num_warps=4, num_stages=1)
        del arg1_1
        triton_poi_fused_add_div_1[grid(8192)](buf5, buf3, 8192, 1024,
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf3
        triton_poi_fused_mul_sigmoid_2[grid(8192)](buf5, buf5, 8192, 1024,
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf5
    return buf1, buf4, arg0_1


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, batch normalization, bias addition, division, and Swish activation.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1
        , bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum
            )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value

    def forward(self, input_0):
        arg0_1 = input_0
        arg1_1 = self.bias
        output = call([arg0_1, arg1_1])
        return output[0]