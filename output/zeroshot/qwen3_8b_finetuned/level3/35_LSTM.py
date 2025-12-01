import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 10
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (10, 256), (256, 1))
    assert_size_stride(primals_2, (10,), (1,))
    assert_size_stride(primals_3, (10, 128, 512, 256), (16384, 128, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(2560)](primals_3, primals_2, buf0, 2560,
            XBLOCK=256, num_warps=4, num_stages=1)
        del primals_2
    return buf0, primals_1, primals_3


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model.

        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param num_layers: Number of recurrent layers
        :param output_size: The number of output features
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
        """
        super(ModelNew, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_0):
        primals_1 = self.fc.weight
        primals_2 = self.fc.bias
        primals_3 = self.lstm.weight_hh_l0
        primals_4 = self.lstm.weight_ih_l0
        primals_5 = self.lstm.bias_hh_l0
        primals_6 = self.lstm.bias_ih_l0
        primals_7 = self.lstm.weight_hh_l1
        primals_8 = self.lstm.weight_ih_l1
        primals_9 = self.lstm.bias_hh_l1
        primals_10 = self.lstm.bias_ih_l1
        primals_11 = self.lstm.weight_hh_l2
        primals_12 = self.lstm.weight_ih_l2
        primals_13 = self.lstm.bias_hh_l2
        primals_14 = self.lstm.bias_ih_l2
        primals_15 = self.lstm.weight_hh_l3
        primals_16 = self.lstm.weight_ih_l3
        primals_17 = self.lstm.bias_hh_l3
        primals_18 = self.lstm.bias_ih_l3
        primals_19 = self.lstm.weight_hh_l4
        primals_20 = self.lstm.weight_ih_l4
        primals_21 = self.lstm.bias_hh_l4
        primals_22 = self.lstm.bias_ih_l4
        primals_23 = self.lstm.weight_hh_l5
        primals_24 = self.lstm.weight_ih_l5
        primals_25 = self.lstm.bias_hh_l5
        primals_26 = self.lstm.bias_ih_l5
        primals_27 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9,
            primals_10, primals_11, primals_12, primals_13, primals_14,
            primals_15, primals_16, primals_17, primals_18, primals_19,
            primals_20, primals_21, primals_22, primals_23, primals_24,
            primals_25])
        return output[0]