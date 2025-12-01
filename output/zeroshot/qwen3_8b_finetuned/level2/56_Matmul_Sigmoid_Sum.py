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
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32768
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_sigmoid_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 32768
    tmp0 = tl.load(in_ptr0 + x4, xmask)
    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = tmp0 * tmp2
    tl.store(out_ptr0 + x4, tmp4, xmask)


@triton.jit
def triton_poi_fused_sum_2(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 32768
    tmp0 = tl.load(in_ptr0 + (r2 + 32768 * x3), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + x3, tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 32768), (32768, 1))
    assert_size_stride(primals_2, (32768, 32768), (32768, 1))
    assert_size_stride(primals_3, (32768,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        extern_kernels.mm(primals_1, reinterpret_tensor(primals_2, (32768, 
            32768), (1, 32768), 0), out=buf0)
        del primals_2
        buf1 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(4194304)](buf0, primals_3, buf1, 
            4194304, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_3
        buf2 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_sigmoid_1[grid(4194304)](buf1, primals_3, buf2, 
            4194304, XBLOCK=128, num_warps=4, num_stages=1)
        buf3 = empty_strided_cuda((128, 1), (1, 1), torch.float32)
        triton_poi_fused_sum_2[grid(128)](buf2, primals_3, buf3, 128, 64,
            XBLOCK=16, num_warps=4, num_stages=1)
        del buf2
    return buf3, primals_1, buf0, buf1, buf3


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies sigmoid, and sums the result.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, input_0):
        primals_2 = self.linear.weight
        primals_3 = self.linear.bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]