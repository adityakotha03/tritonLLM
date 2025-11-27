import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_relu_0(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex % 256
    x1 = xindex // 256
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_poi_fused_sigmoid_1(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl
    .constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = -tmp0
    tmp2 = tl.full([1], 0.0, tl.float32)
    tmp3 = triton_helpers.maximum(tmp2, tmp1)
    tmp4 = tl.sigmoid(tmp3)
    tl.store(out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_poi_fused_add_relu_sigmoid_2(in_out_ptr0, in_ptr0, in_ptr1,
    in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex % 256
    x1 = xindex // 256
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tl.load(in_ptr2 + (x1), xmask)
    tmp6 = tmp4 + tmp5
    tmp7 = tl.load(in_ptr3 + (x1), xmask)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + x0, tmp10, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (512, 10, 128), (1280, 128, 1))
    assert_size_stride(arg1_1, (6, 10, 256), (2560, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((512, 10, 256), (2560, 256, 1), torch.float32)
        buf1 = empty_strided_cuda((6, 10, 256), (2560, 256, 1), torch.float32)
        buf2 = empty_strided_cuda((512, 10, 256), (2560, 256, 1), torch.float32)
        buf3 = empty_strided_cuda((6, 10, 256), (2560, 256, 1), torch.float32)
        del arg1_1
        get_rawbuf = buf0
        triton_poi_fused_add_relu_0[grid(49152, 256)](arg0_1, buf1, buf2,
            49152, 256, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
        buf4 = buf2
        triton_poi_fused_sigmoid_1[grid(2048, 1)](buf4, buf3, 2048, 1,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf4
        buf5 = buf3
        triton_poi_fused_add_relu_sigmoid_2[grid(65536, 256)](buf5, buf0,
            buf1, buf2, arg1_1, 65536, 256, XBLOCK=256, num_warps=4,
            num_stages=1)
        del arg1_1
    return buf5, buf0, buf1, buf2, buf3


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True,
        batch_first=False):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias,
            batch_first, dropout=0, bidirectional=False)
    
    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]