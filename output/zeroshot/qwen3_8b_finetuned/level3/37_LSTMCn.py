import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + (x0 % 256), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (6, 10, 256), (2560, 256, 1))
    assert_size_stride(primals_2, (256,), (1,))
    assert_size_stride(primals_3, (10, 256), (256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((6, 10, 256), (2560, 256, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(32768)](primals_1, primals_2, buf0, 
            32768, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((6, 10, 256), (2560, 256, 1), torch.float32)
        triton_poi_fused_add_0[grid(32768)](buf1, primals_1, buf2, 32768,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        buf3 = empty_strided_cuda((6, 10, 256), (2560, 256, 1), torch.float32)
        triton_poi_fused_add_0[grid(32768)](buf2, primals_1, buf3, 32768,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        buf4 = empty_strided_cuda((6, 10, 256), (2560, 256, 1), torch.float32)
        triton_poi_fused_add_0[grid(32768)](buf3, primals_1, buf4, 32768,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        buf5 = empty_strided_cuda((6, 10, 256), (2560, 256, 1), torch.float32)
        triton_poi_fused_add_0[grid(32768)](buf4, primals_1, buf5, 32768,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        buf6 = empty_strided_cuda((6, 10, 256), (2560, 256, 1), torch.float32)
        triton_poi_fused_add_0[grid(32768)](buf5, primals_1, buf6, 32768,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        buf7 = empty_strided_cuda((6, 10, 256), (2560, 256, 1), torch.float32)
        triton_poi_fused_add_0[grid(32768)](buf6, primals_1, buf7, 32768,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        buf8 = torch.ops.aten.mm.default(buf7, primals_3, out=None)
        del buf7
    return buf8, primals_3, buf1, buf2, buf3, buf4, buf5, buf6, buf8


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model.

        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param num_layers: Number of recurrent layers
        :param output_size: The number of output features
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to `dropout`
        """
        super(ModelNew, self).__init__()
        # Initialize hidden state with random values
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_0, input_1, input_2):
        primals_1 = self.lstm.weight_hh_l0
        primals_2 = self.lstm.weight_ih_l0
        primals_3 = self.lstm.bias_hh_l0
        primals_4 = self.lstm.bias_ih_l0
        primals_5 = self.fc.weight
        primals_6 = self.fc.bias
        primals_1 = self.lstm.weight_hh_l0
        primals_2 = self.lstm.weight_ih_l0
        primals_3 = self.lstm.bias_hh_l0
        primals_4 = self.lstm.bias_ih_l0
        primals_5 = self.fc.weight
        primals_6 = self.fc.bias
        output = call([input_0, input_1, input_2, primals_1, primals_2,
            primals_3, primals_4, primals_5, primals_6])
        return output[0]