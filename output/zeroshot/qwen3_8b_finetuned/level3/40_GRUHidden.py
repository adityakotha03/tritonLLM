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
def triton_poi_fused_relu_threshold_backward_0(in_out_ptr0, in_ptr0,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x2 = xindex
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x2, tmp4, xmask)
    tl.store(out_ptr0 + x2, tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 256), (256, 1))
    assert_size_stride(primals_2, (512, 10, 128), (1280, 128, 1))
    assert_size_stride(primals_3, (256,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((61440, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_2, (61440, 128), (128,
            1), 0), reinterpret_tensor(primals_1, (128, 256), (1, 128), 0),
            out=buf0)
        del primals_1
        del primals_2
        buf1 = buf0
        del buf0
        buf4 = empty_strided_cuda((512, 10, 256), (25600, 2560, 1), torch.
            bool)
        get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_0[grid(15728640)](buf1,
            primals_3, buf4, 15728640, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
    return buf1, buf4


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True,
        batch_first=False):
        """
        :param input_size: The number of expected features in the input x
        :param hidden_size: The number of features in the hidden state h
        :param num_layers: Number of recurrent layers (default: 1)
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh (default: True)
        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) (default: False)
        """
        super(ModelNew, self).__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=False)
    
    def forward(self, input_0, input_1):
        primals_1 = self.gru.weight_ih
        primals_3 = self.gru.bias_ih
        primals_2 = self.gru.weight_hh
        primals_4 = self.gru.bias_hh
        primals_5 = input_0
        primals_6 = input_1
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6])
        return output[0]