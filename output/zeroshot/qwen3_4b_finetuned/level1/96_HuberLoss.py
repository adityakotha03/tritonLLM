import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_abs_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 - tmp1
    tmp3 = tl.abs(tmp2)
    tmp4 = 0.5
    tmp5 = tmp3 * tmp4
    tmp6 = 1.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tl.where(tmp7, tmp8, tmp5)
    tl.store(out_ptr0 + x0, tmp9, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32768,), (1,))
    assert_size_stride(arg1_1, (32768,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768,), (1,), torch.float32)
        get_input = buf0
        triton_poi_fused_abs_add_0[triton.ops._triton_helpers.grid(32768)](arg0_1,
            arg1_1, get_input, 32768, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
    return get_input,


class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]
