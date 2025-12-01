import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_relu_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 256 % 256
    x0 = xindex % 256
    x2 = xindex // 65536
    tmp0 = tl.load(in_ptr0 + (256 * x1 + 65536 * x2 + x0), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (256 * x1 + 65536 * x2 + x0), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + x3, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (256, 256), (256, 1))
    assert_size_stride(primals_2, (256, 256), (256, 1))
    assert_size_stride(primals_3, (256, 256), (256, 1))
    assert_size_stride(primals_4, (512, 10, 256), (25600, 2560, 1))
    assert_size_stride(primals_5, (6, 10, 256), (2560, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((512, 10, 256), (25600, 2560, 1), torch
            .float32)
        get_raw_stream(0)
        triton_poi_fused_add_relu_0[grid(1310720)](primals_1, primals_2,
            buf0, 1310720, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_1
        del primals_2
        buf1 = empty_strided_cuda((512, 10, 256), (25600, 2560, 1), torch
            .float32)
        triton_poi_fused_add_relu_0[grid(1310720)](primals_3, primals_5,
            buf1, 1310720, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_5
    return buf0, buf1, primals_3, primals_4


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
        primals_1 = self.gru.weight_ih_l0
        primals_2 = self.gru.weight_hh_l0
        primals_3 = self.gru.bias_ih_l0
        primals_4 = self.gru.bias_hh_l0
        primals_5 = input_1
        primals_6 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_6])
        return output[0], output[1]