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
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16384
    x1 = xindex // 16384
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16384 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16384
    x2 = xindex // 16384
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16384 * x2), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = tl.where(xmask, tmp1, float('-inf'))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tmp9 / tmp13
    tl.store(out_ptr0 + x4, tmp14, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (16384, 16384), (16384, 1))
    assert_size_stride(primals_2, (16384,), (1,))
    assert_size_stride(primals_3, (128, 16384), (16384, 1))
    assert_size_stride(primals_4, (16384,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16384, 16384), (16384, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_3, (16384, 128), (1, 
            16384), 0), reinterpret_tensor(primals_1, (128, 16384), (1, 
            128), 0), out=buf0)
        del primals_1
        del primals_3
        buf1 = empty_strided_cuda((128, 16384), (16384, 1), torch.bool)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(2097152)](buf0, buf1, 2097152,
            XBLOCK=512, num_warps=4, num_stages=1)
        buf2 = reinterpret_tensor(buf0, (128, 16384), (16384, 1), 0)
        del buf0
        triton_poi_fused__softmax_1[grid(2097152)](primals_4, buf2, 2097152,
            XBLOCK=256, num_warps=4, num_stages=1)
        del primals_4
    return reinterpret_tensor(buf2, (128, 16384), (16384, 1), 0
        ), reinterpret_tensor(primals_2, (16384,), (1,), 0), reinterpret_tensor(
        buf1, (16384, 128), (1, 16384), 0), buf2


class ModelNew(nn.Module):
    """
    A model that performs matrix multiplication, applies dropout, and then applies softmax.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_0):
        primals_1 = self.matmul.weight
        primals_2 = self.matmul.bias
        primals_3 = input_0
        primals_4 = self.dropout.p
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
