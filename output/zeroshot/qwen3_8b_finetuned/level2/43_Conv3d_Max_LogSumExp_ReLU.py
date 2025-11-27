import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_logsumexp_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x2 = xindex // 4096
    x1 = xindex // 64
    x3 = xindex // 16
    x4 = xindex // 1
    tmp0 = tl.load(in_ptr0 + (x2 + 64 * x0 + 4096 * x1 + 16384 * x3 + 
        16384 * x4), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, 64])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.max(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl.full([XBLOCK, 64], 1e-05, tl.int32)
    tmp7 = tl.full([XBLOCK, 64], 1e-05, tl.float32)
    tmp8 = tl.where(tmp5 < tmp6, tmp7, tmp5)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tmp9 + tmp4
    tmp11 = tl.log(tmp10)
    tl.store(out_ptr0 + x0, tmp11, xmask)


@triton.jit
def triton_poi_fused_relu_1(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1
    x1 = xindex // 1
    tmp0 = tl.load(in_out_ptr0 + x1, xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = tl.where(tmp2, tmp0, tmp1)
    tl.store(in_out_ptr0 + x1, tmp3, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (4, 32, 32, 128, 128), (32768, 1024, 32, 1, 
        1))
    assert_size_stride(primals_2, (64, 32, 3, 3, 3), (27648, 432, 144, 36, 9))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 64, 32, 128, 128), (262144, 4096, 128, 
            4, 1), torch.float32)
        get_raw_stream(0)
        buf1 = buf0
        del buf0
        buf2 = buf1
        del buf1
        buf3 = buf2
        del buf2
        buf4 = buf3
        del buf3
        buf5 = buf4
        del buf4
        buf6 = buf5
        del buf5
        del input_0
        triton_poi_fused_logsumexp_0[grid(262144)](buf6, buf4, 262144, 
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf6
        buf7 = buf4
        del buf4
        triton_poi_fused_relu_1[grid(262144)](buf7, 262144, XBLOCK=256, 
            num_warps=4, num_stages=1)
        del buf7
        del primals_1
        del primals_2
    return buf5, reinterpret_tensor(buf0, (4, 64, 32, 128, 128), (262144, 
        4096, 128, 4, 1), 0), reinterpret_tensor(buf5, (4, 64, 32, 128, 
        128), (262144, 4096, 128, 4, 1), 0)


class ModelNew(nn.Module):
    """
    Optimized version of the original 3D convolution + max pooling + logsumexp + ReLU
    model, where the logsumexp and ReLU are implemented as Triton kernels.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=
            stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = self.max_pool.weight
        primals_4 = self.max_pool.bias
        output = call([input_0, primals_1, primals_2])
        return output[0]