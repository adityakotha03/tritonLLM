import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_linalg_vector_matrix_mul_0(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 128 % 256
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + 128 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + x3, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6 = args
    args.clear()
    assert_size_stride(primals_1, (10, 512, 128), (65536, 128, 1))
    assert_size_stride(primals_2, (256, 128), (128, 1))
    assert_size_stride(primals_3, (6, 10, 256), (2560, 256, 1))
    assert_size_stride(primals_4, (6, 10, 256), (2560, 256, 1))
    assert_size_stride(primals_5, (10, 256), (256, 1))
    assert_size_stride(primals_6, (6, 10, 256), (2560, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.bmm(primals_1, primals_2)
        assert_size_stride(buf0, (10, 512, 256), (131072, 256, 1))
        buf1 = empty_strided_cuda((10, 256, 6), (1536, 6, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_linalg_vector_matrix_mul_0[grid(32768)](buf0,
            primals_3, buf1, 32768, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_3
        buf2 = extern_kernels.bmm(buf0, primals_4)
        assert_size_stride(buf2, (10, 512, 256), (131072, 256, 1))
        del primals_4
        triton_poi_fused_linalg_vector_matrix_mul_0[grid(32768)](buf2,
            primals_6, buf1, 32768, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_6
        buf3 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        extern_kernels.mm(buf1, primals_5, out=buf3)
        del buf1
    return buf3, primals_1, primals_2, primals_5, buf0, buf2


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
        primals_2 = self.lstm.weight_ih
        primals_4 = self.lstm.weight_hh
        primals_3 = self.lstm.bias_ih
        primals_6 = self.lstm.bias_hh
        primals_5 = self.fc.weight
        primals_1 = input_0
        primals_0 = input_1
        primals_10 = input_2
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6])
        return output[0]
