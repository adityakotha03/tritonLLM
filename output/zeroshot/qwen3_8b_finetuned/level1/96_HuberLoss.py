import torch
import torch.nn as nn
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__log_softmax_backward_data_0(in_ptr0, in_ptr1, out_ptr0,
    xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK, 1], xoffset, tl.int32)
    tl.full([XBLOCK, 1], XBLOCK, tl.int32)
    x = xindex
    tmp0 = tl.load(in_ptr0 + x, None)
    tmp1 = tl.load(in_ptr1 + x, None)
    tmp2 = tl.full([1], 1, tl.int32)
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, 1])
    tmp5 = tl.where(tmp3 < tmp2, tmp3, tmp3)
    tmp6 = tmp1 - tmp5
    tmp7 = 1.0
    tmp8 = tmp6 <= tmp7
    tmp9 = tl.full([1], 0.5, tl.float32)
    tmp10 = 1.0
    tmp11 = tmp6 - tmp10
    tmp12 = tl.where(tmp8, tmp9, tmp11)
    tmp13 = 2.0
    tmp14 = tmp12 * tmp13
    tmp15 = tl.full([1], 0.5, tl.float32)
    tmp16 = tmp12 - tmp15
    tmp17 = tl.where(tmp8, tmp14, tmp16)
    tmp18 = tmp0 * tmp17
    tl.store(out_ptr0 + x, tmp18, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32768,), (1,))
    assert_size_stride(arg1_1, (32768,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 32768), (32768, 1), torch.float32)
        buf1 = buf0
        del buf0
        get_raw_buf = buf1
        buf2 = buf1
        triton_poi_fused__log_softmax_backward_data_0[grid(32768)](arg0_1,
            arg1_1, buf2, 32768, 1, XBLOCK=128)
        del arg0_1
        del arg1_1
    return buf2, get_raw_buf,


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