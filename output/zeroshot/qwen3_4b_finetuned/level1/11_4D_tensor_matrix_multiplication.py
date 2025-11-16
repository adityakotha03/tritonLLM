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
def triton_poi_fused_mul_sum_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16384 % 256
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl_math.mul(tmp0, tmp1)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 1))
    tl.store(out_ptr0 + x3, tmp6, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 256, 512, 256), (65536, 256, 512, 1))
    assert_size_stride(arg1_1, (256, 768), (768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 256, 512, 768), (100352, 4, 2, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_sum_0[grid(49152)](arg0_1, arg1_1, buf0, 49152,
            XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
    return reinterpret_tensor(buf0, (8, 256, 512, 768), (100352, 4, 2, 1), 0),
        reinterpret_tensor(buf0, (8, 512, 256, 768), (100352, 2, 1, 0), 1)


class ModelNew(nn.Module):
    """
    Performs 4D tensor-matrix multiplication: 
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)

    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]
