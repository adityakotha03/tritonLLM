import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_mean_pow_rsqrt_sub_add_sqrt_0(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1155136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 16448 % 64
    x2 = xindex // 1024
    x3 = xindex // 16448
    x4 = xindex % 16448
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (64 * x2 + x1 + 1048576 * x3), xmask,
        eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.where(xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = 64.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tmp0 * tmp0
    tmp11 = 1.0
    tmp12 = tmp11 / tmp10
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp16 * tmp16
    tmp18 = 1048576.0
    tmp19 = tmp17 / tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = tl.sqrt(tmp22)
    tl.store(out_ptr0 + x5, tmp23, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (112, 64, 512, 512), (16777216, 262144, 
        512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((112, 64, 1, 1), (64, 1, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mean_pow_rsqrt_sub_add_sqrt_0[grid(1155136)](arg0_1,
            buf0, 1155136, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs RMS Normalization.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
