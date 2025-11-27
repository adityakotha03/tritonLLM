import torch
import torch.nn as nn
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
reinterpret_tensor_1 = torch._C._dynamo.guards._reinterpret_tensor
empty_cuda = torch._C._dynamo.guards._empty_cuda
to_tensor = torch._C._dynamo.guards.to_tensor
reinterpret_tensor_2 = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK: tl
    .constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr1 + x2, None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, None)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 512, 128), (65536, 128, 1))
    assert_size_stride(arg1_1, (6, 1, 256), (256, 1, 1))
    assert_size_stride(arg2_1, (6, 1, 256), (256, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 512, 256), (131072, 256, 1), torch.float32
            )
        buf1 = empty_strided_cuda((1, 512, 256), (131072, 256, 1), torch.float32
            )
        buf2 = empty_strided_cuda((6, 1, 256), (256, 1, 1), torch.float32)
        buf3 = empty_strided_cuda((1, 512, 256), (131072, 256, 1), torch.float32
            )
        buf4 = empty_strided_cuda((1, 512, 256), (131072, 256, 1), torch.float32
            )
        buf5 = empty_strided_cuda((6, 1, 256), (256, 1, 1), torch.float32)
        del arg1_1
        del arg2_1
        buf6 = buf0
        del buf0
        buf7 = buf1
        del buf1
        triton_poi_fused_add_0[grid(131072)](buf6, buf7, buf4, 131072, 1,
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf6
        del buf7
        buf8 = buf2
        del buf2
        buf9 = buf3
        del buf3
        triton_poi_fused_add_0[grid(256)](buf8, buf9, buf5, 256, 1, XBLOCK=256,
            num_warps=4, num_stages=1)
        del buf8
        del buf9
        buf10 = buf4
        del buf4
        buf11 = buf5
        del buf5
        buf12 = buf10
        del buf10
        del buf11
        buf13 = buf12
        del buf12
        buf14 = buf13
        del buf13
        del arg0_1
        get_raw_buf = buf14
        del buf14
    return buf10, buf11, buf12, buf13, buf14, get_raw_buf


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout
        =0.0):
        super(ModelNew, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True
            , dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_0, input_1, input_2):
        arg0_1 = input_0
        arg1_1 = input_1
        arg2_1 = input_2
        output = call([arg0_1, arg1_1, arg2_1])
        return output[0]