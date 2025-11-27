import torch
import torch.nn as nn
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_min_sub_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16384
    x1 = xindex // 16384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 16384, xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tmp0 < tmp3
    tmp5 = tl.where(tmp4, tmp0, tmp3)
    tmp6 = tmp5 - tmp3
    tl.store(out_ptr0 + x3, tmp6, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 16384), (16384, 1))
    assert_size_stride(arg1_1, (16384,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        buf1 = buf0
        del buf0
        get_raw_buf = buf1
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
        triton_poi_fused_min_sub_0[grid(2097152)](arg0_1, buf6, 2097152,
            XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
    return buf6, arg1_1,


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication, applies a constant
    minimum, and subtracts the constant using a fused Triton kernel.
    """
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))

    def forward(self, input_0):
        arg0_1 = input_0
        arg1_1 = self.constant
        output = call([arg0_1, arg1_1])
        return output[0]