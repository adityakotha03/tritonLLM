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
def triton_poi_fused_add_group_norm_leaky_relu_0(in_ptr0, in_ptr1, in_ptr2,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp6 = tl.where(xmask, tmp3, 0)
    tmp7 = tl.sum(tmp6, 0)[:, None]
    tmp8 = tmp4 - tmp7
    tmp9 = tmp8 * tmp5
    tmp10 = 0.01
    tmp11 = tmp9 * tmp10
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp9 * tmp13
    tmp15 = tmp14 + tmp14
    tl.store(out_ptr0 + x2, tmp15, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (8192,), (1,))
    assert_size_stride(primals_4, (1024, 8192), (8388608, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.mm(primals_4, reinterpret_tensor(primals_1, (8192, 
            8192), (1, 8192), 0), out=buf0)
        del primals_1
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_group_norm_leaky_relu_0[grid(8388608)](buf0,
            primals_2, primals_3, buf1, 8388608, XBLOCK=1024, num_warps=8,
            num_stages=1)
        del buf0
        del primals_2
        del primals_3
    return buf1, primals_4, buf1, primals_1


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, group normalization, leaky ReLU activation, and element-wise sum.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, input_0):
        primals_1 = self.fc.weight
        primals_2 = self.fc.bias
        primals_3 = self.gn.weight
        primals_4 = self.gn.bias
        primals_4 = self.gn.bias
        primals_1 = self.fc.weight
        primals_2 = self.fc.bias
        primals_3 = self.gn.weight
        primals_4 = self.gn.bias
        output = call([primals_1, primals_2, primals_3, primals_4, input_0])
        return output[0]