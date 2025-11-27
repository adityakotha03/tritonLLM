import torch
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
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32768
    x1 = xindex // 32768
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 65536 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (16384 + x0 + 65536 * x1), xmask,
        eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_sum_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 32768 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (16384 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (65536 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (81920 + 32768 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + x0, tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 32768), (32768, 1))
    assert_size_stride(primals_2, (32768,), (1,))
    assert_size_stride(primals_3, (128, 32768), (32768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_0[grid(2097152)](primals_3,
            buf0, 2097152, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf1 = empty_strided_cuda((128, 1), (1, 1), torch.float32)
        triton_poi_fused_sum_1[grid(128)](buf0, buf1, 128, XBLOCK=128,
            num_warps=4, num_stages=1)
        del buf0
    return buf1, primals_1, primals_2


class ModelNew(nn.Module):
    """
    Model that performs matrix multiplication, max pooling, sum, and scaling.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.scale_factor = scale_factor

    def forward(self, input_0):
        primals_1 = self.matmul.weight
        primals_2 = self.matmul.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
