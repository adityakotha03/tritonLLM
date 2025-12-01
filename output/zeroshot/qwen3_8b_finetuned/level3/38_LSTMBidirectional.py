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
    xnumel = 30720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (12, 10, 256), (2560, 256, 1))
    assert_size_stride(primals_2, (256,), (1,))
    assert_size_stride(primals_3, (12, 10, 256), (2560, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((12, 10, 256), (2560, 256, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(30720)](primals_1, primals_2, buf0, 
            30720, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf1 = empty_strided_cuda((12, 10, 256), (2560, 256, 1), torch.float32)
        triton_poi_fused_add_0[grid(30720)](primals_3, primals_2, buf1, 
            30720, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
    return buf0, buf1, primals_1, primals_3


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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, input_0, input_1, input_2):
        primals_2 = self.fc.weight
        primals_3 = self.fc.bias
        primals_1 = input_1
        primals_4 = input_2
        primals_5 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]