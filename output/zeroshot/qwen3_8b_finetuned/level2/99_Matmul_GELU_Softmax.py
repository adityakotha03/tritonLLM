import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8192
    x1 = xindex // 8192
    tmp0 = tl.load(in_ptr0 + (x0 + 8192 * x2), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_gelu_softmax_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8192
    x1 = xindex // 8192
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp12 = tl.where(xmask, tmp11, 0)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.max2(tmp16, 1, 0))
    tmp18 = tmp10 - tmp17
    tmp19 = tl_math.exp(tmp18)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp21 = tl.where(xmask, tmp20, 0)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp23 = tl.where(xmask, tmp22, 0)
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp19 / tmp25
    tl.store(in_out_ptr0 + x2, tmp26, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192, 8192), (687070464, 8192, 1),
            torch.float32)
        buf1 = reinterpret_tensor(buf0, (1024, 8192), (8388608, 1), 0)
        del buf0
        get_raw_stream(0)
        triton_poi_fused_0[grid(134217728)](primals_3, primals_1, buf1, 
            134217728, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_1
        buf2 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        buf3 = buf2
        del buf2
        triton_poi_fused_gelu_softmax_1[grid(8388608)](buf3, primals_2, 
            8388608, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
    return buf3, primals_3


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input_0):
        primals_1 = self.linear.weight
        primals_2 = self.linear.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]