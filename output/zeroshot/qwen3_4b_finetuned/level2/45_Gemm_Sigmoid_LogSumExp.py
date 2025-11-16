import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_exp_log_sigmoid_0(in_out_ptr0, in_ptr0, in_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tmp0 + tmp1
    tmp4 = libdevice.exp(tmp3)
    tmp5 = tmp4 + tmp2
    tmp6 = tl.sigmoid(tmp3)
    tmp7 = tmp0 - tmp6
    tmp8 = tmp5 + tmp7
    tmp9 = libdevice.log(tmp8)
    tmp10 = tmp0 - tmp9
    tmp11 = tmp0 - tmp10
    tl.store(in_out_ptr0 + x2, tmp11, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (4096, 2048), (2048, 1))
    assert_size_stride(primals_2, (4096,), (1,))
    assert_size_stride(primals_3, (16384, 2048), (2048, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        extern_kernels.mm(primals_3, reinterpret_tensor(primals_1, (2048, 
            4096), (1, 2048), 0), out=buf0)
        del primals_1
        buf1 = empty_strided_cuda((16384, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(buf0, reinterpret_tensor(primals_2, (4096, 1024),
            (1, 4096), 0), out=buf1)
        del primals_2
        buf2 = reinterpret_tensor(buf0, (16384, 4096), (4096, 1), 0)
        del buf0
        get_raw_stream(0)
        triton_poi_fused_add_exp_log_sigmoid_0[grid(8192)](buf2, primals_3,
            primals_3, 8192, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_3
    return buf1, reinterpret_tensor(buf0, (16384, 4096), (4096, 1), 0), buf2


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), applies Sigmoid,
    another Gemm, and computes LogSumExp over features.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, input_0):
        primals_1 = self.linear1.weight
        primals_2 = self.linear2.weight
        primals_3 = self.linear1.bias
        primals_4 = self.linear2.bias
        primals_10 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
