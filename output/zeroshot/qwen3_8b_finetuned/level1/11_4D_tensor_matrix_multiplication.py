import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 819200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 768
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8, 256, 512, 256), (32768000, 128, 256, 1
        ))
    assert_size_stride(primals_2, (256, 768), (768, 1))
    assert_size_stride(primals_3, (768,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1048576, 256), (256, 1), torch.float32)
        extern_kernels.reshape_strided_cuda(primals_1, (1048576, 256), buf0)
        del primals_1
        buf1 = empty_strided_cuda((8, 256, 512, 768), (98304, 393216, 768, 1
            ), torch.float32)
        extern_kernels.bmm(buf0, primals_2, out=buf1)
        del buf0
        del primals_2
        buf2 = buf1
        del buf1
        get_raw_stream(0)
        triton_poi_fused_add_relu_0[grid(819200)](buf2, primals_3, 819200,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_3
    return reinterpret_tensor(buf2, (8, 256, 512, 768), (98304, 393216, 768, 
        1), 0)


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
        primals_3 = input_1
        primals_1 = input_0
        primals_2 = primals_3
        del primals_3
        output = call([primals_1, primals_2])
        return output