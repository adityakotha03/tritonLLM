import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_max_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    x3 = xindex // 10240
    tmp0 = tl.load(in_ptr0 + (x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (128 + x0 + 256 * x1), xmask,
        eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + 128 + x0, xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + 128 + x0, xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (256 + x0 + 256 * x1), xmask,
        eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + 256 + x0, xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr2 + 256 + x0, xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (384 + x0 + 256 * x1), xmask,
        eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr1 + 384 + x0, xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + 384 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tmp0 + tmp1
    tmp4 = tmp3 + tmp2
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = 1.0
    tmp9 = tmp8 - tmp7
    tmp13 = tmp10 + tmp11
    tmp14 = tmp13 + tmp12
    tmp15 = triton_helpers.maximum(tmp5, tmp14)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp9 * tmp16
    tmp18 = tmp7 * tmp17
    tmp22 = tmp19 + tmp20
    tmp23 = tmp22 + tmp21
    tmp27 = tmp24 + tmp25
    tmp28 = tmp27 + tmp26
    tmp29 = triton_helpers.maximum(tmp5, tmp28)
    tmp30 = tl.sigmoid(tmp29)
    tmp31 = tmp9 * tmp30
    tmp32 = tmp18 + tmp31
    tl.store(out_ptr0 + x2, tmp32, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (128, 256), (256, 1))
    assert_size_stride(primals_2, (256, 256), (256, 1))
    assert_size_stride(primals_3, (256,), (1,))
    assert_size_stride(primals_4, (256,), (1,))
    assert_size_stride(primals_5, (12, 10, 256), (2560, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((512, 10, 256), (25600, 2560, 1), torch.
            float32)
        extern_kernels.mm(reinterpret_tensor(primals_5, (10, 128), (128, 1),
            0), reinterpret_tensor(primals_1, (128, 256), (1, 128), 0), out
            =buf0)
        del primals_1
        buf1 = empty_strided_cuda((10240, 256), (256, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(1310720)](buf0, primals_3, buf1, 1310720,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf2 = empty_strided_cuda((10240, 256), (256, 1), torch.float32)
        triton_poi_fused_add_0[grid(1310720)](buf1, primals_4, buf2, 1310720,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_4
        buf3 = empty_strided_cuda((10240, 256), (256, 1), torch.float32)
        triton_poi_fused_add_max_relu_1[grid(1310720)](buf2, primals_2,
            buf0, primals_2, buf3, 1310720, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del buf2
        del primals_2
    return reinterpret_tensor(buf3, (512, 10, 256), (25600, 2560, 1), 0
        ), primals_5, reinterpret_tensor(buf0, (10, 128, 256), (32768, 256,
        1), 0)


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
    
    def forward(self, input_0, input_1):
        primals_1 = self.gru.weight_ih_l0
        primals_2 = self.gru.weight_hh_l0
        primals_3 = self.gru.bias_ih_l0
        primals_4 = self.gru.bias_hh_l0
        primals_5 = input_1
        primals_1 = self.gru.weight_ih_l0
        primals_2 = self.gru.weight_hh_l0
        primals_3 = self.gru.bias_ih_l0
        primals_4 = self.gru.bias_hh_l0
        primals_5 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]