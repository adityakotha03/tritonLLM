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
    xnumel = 199680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1920
    x1 = xindex // 1920
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (3072 * x0 + 3072 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1 + 6048 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 + tmp0
    tmp6 = tmp5 + tmp4
    tl.store(out_ptr0 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_add_gru_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4,
    in_ptr5, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 256
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr3 + (x2 + 6048 * x1 + 12096 * x0), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr5 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp7 = tmp3 + tmp4
    tmp8 = tmp2 + tmp7
    tmp9 = tmp5 + tmp6
    tmp10 = tmp8 + tmp9
    tl.store(out_ptr0 + x3, tmp10, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6 = args
    args.clear()
    assert_size_stride(primals_1, (6, 256, 256), (65536, 256, 1))
    assert_size_stride(primals_2, (6, 256), (256, 1))
    assert_size_stride(primals_3, (512, 10, 128), (1280, 128, 1))
    assert_size_stride(primals_4, (6, 10, 256), (2560, 256, 1))
    assert_size_stride(primals_5, (6, 256), (256, 1))
    assert_size_stride(primals_6, (6, 10, 256), (2560, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((10, 6, 256), (1536, 256, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_gru_0[grid(199680)](primals_1, primals_2,
            primals_4, primals_5, buf0, 199680, XBLOCK=256, num_warps=4,
            num_stages=1)
        del primals_1
        del primals_2
        del primals_4
        del primals_5
        buf1 = empty_strided_cuda((6048, 10, 256), (2560, 256, 1), torch.
            float32)
        triton_poi_fused_add_gru_1[grid(20480)](primals_3, primals_6, buf0,
            primals_3, buf1, primals_3, buf1, 20480, XBLOCK=256, num_warps
            =4, num_stages=1)
        del primals_6
        del primals_3
        buf2 = empty_strided_cuda((6, 256, 10), (2560, 10, 256), torch.float32)
        triton_poi_fused_add_gru_0[grid(199680)](primals_1, primals_2,
            primals_4, primals_5, buf2, 199680, XBLOCK=256, num_warps=4,
            num_stages=1)
        del primals_1
        del primals_2
        del primals_4
        del primals_5
    return buf2, primals_3, primals_6, buf0, buf1


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
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
        primals_2 = self.gru.bias_ih_l0
        primals_4 = self.gru.weight_hh_l0
        primals_5 = self.gru.bias_hh_l0
        primals_3 = input_0
        primals_6 = input_1
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6])
        return output[0]
