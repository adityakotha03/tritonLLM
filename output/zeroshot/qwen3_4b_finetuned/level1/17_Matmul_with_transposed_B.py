import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_clone_0(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 8192
    xnumel = 8192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 8192 * x1), xmask & ymask, eviction_policy
        ='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x1 + 8192 * y0), tmp2, xmask & ymask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8192, 8192), (8192, 1))
    assert_size_stride(arg1_1, (8192, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8192, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(8192, 8192)](arg1_1, arg0_1, buf0, 
            8192, 8192, XBLOCK=32, YBLOCK=32, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
    return reinterpret_tensor(buf0, (8192, 8192), (8192, 1), 0),


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]
