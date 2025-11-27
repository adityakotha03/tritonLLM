import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 13762880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256 % 512
    x2 = xindex // 131072
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256 * x1 + 65536 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.load(in_ptr0 + (256 + x0 + 256 * x1 + 65536 * x2), xmask)
    tmp5 = tl.load(in_ptr1 + (256 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tmp3 + tmp7
    tmp9 = tl.load(in_ptr0 + (512 + x0 + 256 * x1 + 65536 * x2), xmask)
    tmp10 = tl.load(in_ptr1 + (512 + x0), xmask, eviction_policy='evict_last')
    tmp11 = tmp9 + tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tmp8 + tmp12
    tmp14 = tl.load(in_ptr0 + (768 + x0 + 256 * x1 + 65536 * x2), xmask)
    tmp15 = tl.load(in_ptr1 + (768 + x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tmp13 + tmp17
    tl.store(out_ptr0 + x3, tmp18, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 256, 512, 256), (32768, 128, 256, 1))
    assert_size_stride(arg1_1, (256, 768), (768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 256, 512, 256), (32768, 128, 256, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_mul_0[grid(13762880)](arg0_1, arg1_1, buf0, 
            13762880, XBLOCK=512, num_warps=8, num_stages=1)
        del arg0_1
        del arg1_1
    return buf0,


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
