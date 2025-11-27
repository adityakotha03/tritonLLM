import torch
import torch.nn as nn
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_triu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex // 4096
    x2 = xindex % 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = x1 >= x2
    tmp2 = tl.full([1], True, tl.int1)
    tmp3 = tmp1 & tmp2
    tmp4 = tl.where(tmp3, tmp0, 0.0)
    tl.store(out_ptr0 + x3, tmp4, xmask)


def triton_triu(input_0):
    arg0_1, arg1_1 = input_0, empty_strided_cuda((4096, 4096), (4096, 1), torch.float32)
    args.clear()
    assert_size_stride(arg0_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg1_1, (4096, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float32)
        get_rawbuf = buf0
        triton_poi_fused_triu_0[grid(16777216)](arg0_1, get_rawbuf, 16777216,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
    return buf0


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        arg0_2 = arg0_1
        arg1_2 = arg1_1
        buf0 = torch.mm(arg0_2, arg1_2)
        del arg0_2
        del arg1_2
        buf1 = triton_triu(buf0)
        return buf1