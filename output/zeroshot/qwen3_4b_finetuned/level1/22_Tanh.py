import torch
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
def triton_poi_fused_tanh_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1580132160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 393216
    x1 = xindex // 393216
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 393216 * x1), xmask)
    tmp1 = libdevice.tanh(tmp0)
    tl.store(out_ptr0 + x2, tmp1, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 393216), (393216, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_tanh_0[grid(1580132160)](arg0_1, buf0, 1580132160,
            XBLOCK=512, num_warps=16, num_stages=1)
        del arg0_1
    return reinterpret_tensor(buf0, (4096, 393216), (393216, 1), 0),
    reinterpret_tensor(buf0, (393216, 4096), (1, 393216), 0)


class ModelNew(nn.Module):
    """
    Simple model that performs a Tanh activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
