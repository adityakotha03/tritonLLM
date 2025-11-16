import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = xindex // 512 % 256
    x2 = xindex // 131072
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 262144 * x2), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x1 + 256 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.load(in_ptr2 + (x0 + 262144 * x2), xmask, eviction_policy=
        'evict_last')
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + x3, tmp4, xmask)


@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x0, tmp1, xmask)


@triton.jit
def triton_poi_fused_add_mul_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 10
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + 0)
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tl.load(in_ptr0 + 1)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + 1)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp10 = tl.load(in_ptr0 + 2)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp13 = tl.load(in_ptr1 + 2)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp16 = tl.load(in_ptr0 + 3)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp19 = tl.load(in_ptr1 + 3)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp22 = tl.load(in_ptr0 + 4)
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp25 = tl.load(in_ptr1 + 4)
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp27 = tmp0 + tmp2
    tmp28 = tmp4 + tmp8
    tmp29 = tmp10 + tmp14
    tmp30 = tmp16 + tmp20
    tmp31 = tmp22 + tmp26
    tmp32 = tl_math.sigmoid(tmp27)
    tmp33 = tl_math.sigmoid(tmp28)
    tmp34 = tl_math.sigmoid(tmp29)
    tmp35 = tl_math.sigmoid(tmp30)
    tmp36 = tl_math.sigmoid(tmp31)
    tmp37 = tmp27 * tmp32
    tmp38 = tmp28 * tmp33
    tmp39 = tmp37 + tmp38
    tmp40 = tmp29 * tmp34
    tmp41 = tmp39 + tmp40
    tmp42 = tmp30 * tmp35
    tmp43 = tmp41 + tmp42
    tmp44 = tmp31 * tmp36
    tmp45 = tmp43 + tmp44
    tl.store(out_ptr0 + x0, tmp45, xmask)


@triton.jit
def triton_poi_fused_mul_sub_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 256 % 256
    x0 = xindex % 256
    x2 = xindex // 65536
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + x3, tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7
    = args
    args.clear()
    assert_size_stride(primals_1, (6, 256, 256), (65536, 256, 1))
    assert_size_stride(primals_2, (6, 256), (256, 1))
    assert_size_stride(primals_3, (256, 256), (256, 1))
    assert_size_stride(primals_4, (6, 10, 256), (2560, 256, 1))
    assert_size_stride(primals_5, (10,), (1,))
    assert_size_stride(primals_6, (10,), (1,))
    assert_size_stride(primals_7, (6, 10), (10, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((10, 256, 512), (131072, 512, 1), torch.
            float32)
        buf3 = empty_strided_cuda((10, 256, 512), (131072, 512, 1), torch.
            float32)
        buf1 = empty_strided_cuda((6, 10, 256), (2560, 256, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(786432)](primals_1, primals_2,
            primals_3, buf0, 786432, XBLOCK=512, num_warps=8, num_stages=1)
        buf2 = empty_strided_cuda((6, 256, 256), (65536, 256, 1), torch.float32)
        triton_poi_fused__to_copy_1[grid(131072)](primals_1, buf2, 131072,
            XBLOCK=512, num_warps=8, num_stages=1)
        buf4 = empty_strided_cuda((6, 10, 256), (2560, 256, 1), torch.float32)
        triton_poi_fused__to_copy_1[grid(131072)](primals_4, buf4, 131072,
            XBLOCK=512, num_warps=8, num_stages=1)
        del primals_4
        buf5 = empty_strided_cuda((6, 10), (10, 1), torch.float32)
        triton_poi_fused_add_mul_2[grid(10)](primals_5, primals_6, buf5, 10,
            XBLOCK=10, num_warps=2, num_stages=1)
        del primals_6
        buf6 = empty_strided_cuda((6, 10, 256), (2560, 256, 1), torch.float32)
        triton_poi_fused_mul_sub_3[grid(65536)](buf1, primals_2, buf5,
            buf4, 65536, XBLOCK=512, num_warps=8, num_stages=1)
        del buf5
        buf7 = empty_strided_cuda((6, 256), (256, 1), torch.float32)
        triton_poi_fused_mul_sub_3[grid(65536)](buf2, primals_2, primals_5,
            primals_7, 65536, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_5
    return buf6, primals_1, primals_2, primals_3, buf0, buf1, primals_7, buf2, buf3, buf4, primals_2, primals_3


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
        primals_1 = self.lstm.weight_ih_l0
        primals_2 = self.lstm.weight_hh_l0
        primals_3 = self.lstm.bias_ih_l0
        primals_4 = self.lstm.weight_hh_l0
        primals_5 = self.lstm.bias_hh_l0
        primals_6 = self.fc.weight
        primals_7 = self.fc.bias
        primals_12 = input_0
        primals_13 = input_1
        primals_14 = input_2
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5,
            primals_6, primals_7, primals_12, primals_13, primals_14])
        return output[0]
