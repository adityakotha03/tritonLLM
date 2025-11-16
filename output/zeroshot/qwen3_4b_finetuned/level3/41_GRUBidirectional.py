import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_gru_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (512 * x1 + x0), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (256 * x1 + x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + x2, xmask)
    tmp4 = tmp0 + tmp1
    tmp5 = 1.0
    tmp6 = tmp2 * tmp5
    tmp7 = tmp4 + tmp6
    tmp8 = tmp7 + tmp3
    tl.store(out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_gru_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1024000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 256 % 512
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (512, 10, 128), (1280, 128, 1))
    assert_size_stride(primals_2, (128, 256), (256, 1))
    assert_size_stride(primals_3, (6, 10, 256), (2560, 256, 1))
    assert_size_stride(primals_4, (12, 256), (256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((512, 10, 256), (2560, 256, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_gru_0[grid(524288)](primals_1, primals_2,
            primals_3, primals_4, buf0, 524288, XBLOCK=1024, num_warps=4,
            num_stages=1)
        buf1 = empty_strided_cuda((6, 10, 256), (2560, 256, 1), torch.float32)
        triton_poi_fused_add_gru_1[grid(1024000)](primals_1, primals_3, buf1,
            1024000, XBLOCK=2048, num_warps=8, num_stages=1)
        del primals_1
        del primals_3
    return buf1, primals_2, primals_4, buf0


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
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first,
            dropout=0, bidirectional=True)
        self.h0 = torch.randn((num_layers * 2, batch_size, hidden_size))
    
    def forward(self, input_0, input_1):
        primals_2 = self.gru.weight_ih_l0
        primals_4 = self.gru.weight_hh_l0
        primals_3 = self.gru.weight_ih_l1
        primals_1 = self.gru.weight_hh_l1
        primals_0 = self.gru.bias_ih_l0
        primals_5 = self.gru.bias_hh_l0
        primals_6 = self.gru.bias_ih_l1
        primals_7 = self.gru.bias_hh_l1
        primals_8 = self.gru.bias_ih_l2
        primals_9 = self.gru.bias_hh_l2
        primals_10 = self.gru.bias_ih_l3
        primals_11 = self.gru.bias_hh_l3
        primals_12 = self.gru.bias_ih_l4
        primals_13 = self.gru.bias_hh_l4
        primals_14 = self.gru.bias_ih_l5
        primals_15 = self.gru.bias_hh_l5
        primals_16 = self.h0
        output = call([primals_1, primals_2, primals_16, primals_15])
        return output[0]
