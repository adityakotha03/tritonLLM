import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 13107200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tmp0 + tmp1
    tmp4 = tmp3 + tmp2
    tl.store(out_ptr0 + x0, tmp4, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15) = args
    args.clear()
    assert_size_stride(primals_1, (512, 10, 128), (12800, 128, 1))
    assert_size_stride(primals_2, (256, 128), (128, 1))
    assert_size_stride(primals_3, (256,), (1,))
    assert_size_stride(primals_4, (256, 256), (256, 1))
    assert_size_stride(primals_5, (256,), (1,))
    assert_size_stride(primals_6, (256, 128), (128, 1))
    assert_size_stride(primals_7, (256,), (1,))
    assert_size_stride(primals_8, (256, 256), (256, 1))
    assert_size_stride(primals_9, (256,), (1,))
    assert_size_stride(primals_10, (256, 128), (128, 1))
    assert_size_stride(primals_11, (256,), (1,))
    assert_size_stride(primals_12, (256, 256), (256, 1))
    assert_size_stride(primals_13, (256,), (1,))
    assert_size_stride(primals_14, (256, 128), (128, 1))
    assert_size_stride(primals_15, (256,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((256, 10, 512), (51200, 512, 1), torch.
            float32)
        extern_kernels.addmm(primals_3, reinterpret_tensor(primals_1, (512,
            128), (128, 1), 0), reinterpret_tensor(primals_2, (128, 256),
            (1, 128), 0), alpha=1, beta=1, out=buf0)
        del primals_2
        del primals_3
        buf1 = empty_strided_cuda((256, 10, 512), (51200, 512, 1), torch.
            float32)
        extern_kernels.addmm(primals_5, reinterpret_tensor(primals_1, (512,
            128), (128, 1), 0), reinterpret_tensor(primals_6, (128, 256),
            (1, 128), 0), alpha=1, beta=1, out=buf1)
        del primals_5
        buf2 = empty_strided_cuda((256, 10, 512), (51200, 512, 1), torch.
            float32)
        extern_kernels.addmm(primals_7, reinterpret_tensor(primals_1, (512,
            128), (128, 1), 0), reinterpret_tensor(primals_8, (128, 256),
            (1, 128), 0), alpha=1, beta=1, out=buf2)
        del primals_7
        buf3 = empty_strided_cuda((256, 10, 512), (51200, 512, 1), torch.
            float32)
        extern_kernels.addmm(primals_9, reinterpret_tensor(primals_1, (512,
            128), (128, 1), 0), reinterpret_tensor(primals_10, (128, 256),
            (1, 128), 0), alpha=1, beta=1, out=buf3)
        del primals_9
        buf4 = empty_strided_cuda((256, 10, 512), (51200, 512, 1), torch.
            float32)
        extern_kernels.addmm(primals_11, reinterpret_tensor(primals_1, (512,
            128), (128, 1), 0), reinterpret_tensor(primals_12, (128, 256),
            (1, 128), 0), alpha=1, beta=1, out=buf4)
        del primals_11
        buf5 = empty_strided_cuda((256, 10, 512), (51200, 512, 1), torch.
            float32)
        extern_kernels.addmm(primals_13, reinterpret_tensor(primals_1, (512,
            128), (128, 1), 0), reinterpret_tensor(primals_14, (128, 256),
            (1, 128), 0), alpha=1, beta=1, out=buf5)
        del primals_13
        buf6 = empty_strided_cuda((12, 10, 256), (2560, 256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_15, (12, 256), (256, 
            1), 0), reinterpret_tensor(buf0, (256, 256), (1, 256), 0),
            out=buf6)
        del primals_15
        buf7 = empty_strided_cuda((12, 10, 256), (2560, 256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_15, (12, 256), (256, 
            1), 0), reinterpret_tensor(buf1, (256, 256), (1, 256), 0),
            out=buf7)
        buf8 = empty_strided_cuda((12, 10, 256), (2560, 256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_15, (12, 256), (256, 
            1), 0), reinterpret_tensor(buf2, (256, 256), (1, 256), 0),
            out=buf8)
        buf9 = empty_strided_cuda((12, 10, 256), (2560, 256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_15, (12, 256), (256, 
            1), 0), reinterpret_tensor(buf3, (256, 256), (1, 256), 0),
            out=buf9)
        buf10 = empty_strided_cuda((12, 10, 256), (2560, 256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_15, (12, 256), (256, 
            1), 0), reinterpret_tensor(buf4, (256, 256), (1, 256), 0),
            out=buf10)
        buf11 = empty_strided_cuda((12, 10, 256), (2560, 256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_15, (12, 256), (256, 
            1), 0), reinterpret_tensor(buf5, (256, 256), (1, 256), 0),
            out=buf11)
        buf12 = empty_strided_cuda((512, 10, 256), (25600, 256, 1), torch.
            float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(13107200)](buf6, buf7, primals_14,
            buf12, 13107200, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_14
        buf13 = reinterpret_tensor(buf6, (512, 10, 256), (25600, 256, 1), 0)
        del buf6
        triton_poi_fused_add_0[grid(13107200)](buf8, buf9, primals_10,
            buf13, 13107200, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_10
        buf14 = reinterpret_tensor(buf8, (512, 10, 256), (25600, 256, 1), 0)
        del buf8
        triton_poi_fused_add_0[grid(13107200)](buf10, buf11, primals_6,
            buf14, 13107200, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_6
        buf15 = reinterpret_tensor(buf10, (512, 10, 256), (25600, 256, 1), 0)
        del buf10
        triton_poi_fused_add_0[grid(13107200)](buf11, buf12, primals_2,
            buf15, 13107200, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        del buf12
    return (buf15, reinterpret_tensor(primals_15, (12, 256), (256, 1), 0),
        primals_1, buf0, buf1, buf2, buf3, buf4, buf5, buf7, buf9, buf11,
        buf13, buf14)


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
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)
        self.h0 = torch.randn((num_layers*2, batch_size, hidden_size))
    
    def forward(self, input_0, input_1):
        primals_2 = self.gru.weight_ih_l0
        primals_3 = self.gru.bias_ih_l0
        primals_4 = self.gru.weight_hh_l0
        primals_5 = self.gru.bias_hh_l0
        primals_6 = self.gru.weight_ih_l1
        primals_7 = self.gru.bias_ih_l1
        primals_8 = self.gru.weight_hh_l1
        primals_9 = self.gru.bias_hh_l1
        primals_10 = self.gru.weight_ih_l2
        primals_11 = self.gru.bias_ih_l2
        primals_12 = self.gru.weight_hh_l2
        primals_13 = self.gru.bias_hh_l2
        primals_14 = self.gru.weight_ih_l3
        primals_15 = self.gru.bias_ih_l3
        primals_16 = self.gru.weight_hh_l3
        primals_17 = self.gru.bias_hh_l3
        primals_18 = self.gru.weight_ih_l4
        primals_19 = self.gru.bias_ih_l4
        primals_20 = self.gru.weight_hh_l4
        primals_21 = self.gru.bias_hh_l4
        primals_22 = self.gru.weight_ih_l5
        primals_23 = self.gru.bias_ih_l5
        primals_24 = self.gru.weight_hh_l5
        primals_25 = self.gru.bias_hh_l5
        primals_1 = input_0
        primals_26 = input_1
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9,
            primals_10, primals_11, primals_12, primals_13, primals_14,
            primals_15, primals_16, primals_17, primals_18, primals_19,
            primals_20, primals_21, primals_22, primals_23, primals_24,
            primals_25])
        return output[0]