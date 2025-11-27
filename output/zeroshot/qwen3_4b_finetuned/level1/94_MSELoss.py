import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__unsafe_index_add_mean_mul_pow_sub_0(in_ptr0, in_ptr1,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32768
    x1 = xindex // 32768
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32768 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 32768 * x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp6 = tl.sum(tmp4, 0)[:, None]
    tmp7 = 32768.0
    tmp8 = tmp6 / tmp7
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr0 + x2, tmp10, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32768,), (1,))
    assert_size_stride(arg1_1, (32768,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((), (), torch.float32)
        buf1 = buf0
        del buf0
        triton_poi_fused__unsafe_index_add_mean_mul_pow_sub_0[ext_fn](
            arg0_1, arg1_1, buf1, 1048576, XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
    return buf1,


class ModelNew(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks.

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
