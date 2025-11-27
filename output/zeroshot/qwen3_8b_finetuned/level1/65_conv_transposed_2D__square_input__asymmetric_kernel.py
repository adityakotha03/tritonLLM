import torch
import torch.nn as nn
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, 
    xoffset, xindex, xmask):
    x0 = xindex
    x2 = xindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x3, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(arg1_1, (64, 64, 3, 7), (1344, 21, 7, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 64, 512, 512), (16777216, 262144, 512, 1),
            torch.float32)
        buf1 = buf0
        del buf0
        get_raw_buf = torch._C._dynamo.guards._get_raw_buf
        buf2 = get_raw_buf(buf1, 0, 8, 64, 512, 512, 262144, 512, 1, 16777216,
            16777216, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        buf3 = buf1
        del buf1
        buf4 = buf2
        del buf2
        triton_poi_fused_convolution_add_0[grid(16777216)](buf4, buf3, buf0,
            16777216, 0, xnumel=16777216, xoffset=0, xmask=tl.full([1], True,
            tl.int1))
        del buf4
        del buf3
        del buf0
        buf5 = buf3
        del buf3
        del buf1
        del buf0
        return buf5, arg1_1, arg0_1,


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]