import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_mean_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 264196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex % 514
    x3 = xindex // 514
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = tl.sum(tmp1, 0)[:, None]
    tmp4 = 128.0
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr0 + x0, tmp5, xmask)


@triton.jit
def triton_poi_fused_add_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex % 128
    x1 = xindex // 128
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x1, x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_log_sum_exp_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, 1])
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK, 1, 1])
    tmp4 = tl.where(xmask, tmp3, 0)
    tmp5 = tl.max_root(tmp4, 0)[:, None, None]
    tmp6 = tmp0 - tmp5
    tmp7 = tl.full([XBLOCK, 1, 1], 1.0)
    tmp8 = tl.where(xmask, tmp6, tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, 1])
    tmp11 = tl.sum(tmp9, 0)[:, None]
    tmp12 = tl.where(xmask, tmp11, 0)
    tmp13 = tl.log(tmp12)
    tmp14 = 10.0
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + x0, tmp15, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (16, 64, 512, 512), (32768, 512, 1, 1))
    assert_size_stride(primals_2, (128, 1, 1), (1, 1, 128))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mean_0[grid(264196)](primals_1, buf0, 264196,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((16, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        triton_poi_fused_add_1[grid(2048)](buf0, primals_2, buf1, 2048,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        del primals_2
        buf2 = empty_strided_cuda((16, 1, 1, 1), (1, 1, 1, 16), torch.float32)
        triton_poi_fused_log_sum_exp_2[grid(16)](buf1, buf2, 16, XBLOCK=256,
            num_warps=1, num_stages=1)
        del buf1
    return buf2,


class ModelNew(nn.Module):
    """
    Optimized model that replaces the global mean, bias addition, log-sum-exp, and sum operations
    with Triton kernels for improved performance on NVIDIA A100-80GB GPU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)

    def forward(self, input_0):
        primals_1 = input_0
        primals_2 = self.bias
        output = call([primals_1, primals_2])
        return output[0]