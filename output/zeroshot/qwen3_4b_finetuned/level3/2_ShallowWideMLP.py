import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_relu_threshold_backward_0(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tmp6 = 0.0
    tmp7 = tmp5 <= tmp6
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tl.store(out_ptr0 + x0 + 524288, tmp7, xmask)


@triton.jit
def triton_poi_fused_add_relu_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32768
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_add_relu_threshold_backward_2(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tmp6 = 0.0
    tmp7 = tmp5 <= tmp6
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tl.store(out_ptr1 + x0, tmp7, xmask)


@triton.jit
def triton_poi_fused_add_relu_threshold_backward_3(in_ptr0, in_ptr1,
    out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tmp6 = 0.0
    tmp7 = tmp5 <= tmp6
    tl.store(out_ptr0 + x0, tmp5, xmask)
    tl.store(out_ptr1 + x0, tmp7, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15) = args
    args.clear()
    assert_size_stride(primals_1, (32768, 16384), (16384, 1))
    assert_size_stride(primals_2, (32768,), (1,))
    assert_size_stride(primals_3, (16384, 32768), (32768, 1))
    assert_size_stride(primals_4, (16384,), (1,))
    assert_size_stride(primals_5, (32768, 16384), (16384, 1))
    assert_size_stride(primals_6, (32768,), (1,))
    assert_size_stride(primals_7, (16384, 32768), (32768, 1))
    assert_size_stride(primals_8, (16384,), (1,))
    assert_size_stride(primals_9, (16384, 16384), (16384, 1))
    assert_size_stride(primals_10, (16384,), (1,))
    assert_size_stride(primals_11, (16384, 16384), (16384, 1))
    assert_size_stride(primals_12, (16384,), (1,))
    assert_size_stride(primals_13, (16384, 16384), (16384, 1))
    assert_size_stride(primals_14, (16384,), (1,))
    assert_size_stride(primals_15, (16384, 16384), (16384, 1))
    assert_size_stride(primals_16, (16384,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_0[grid(524288)](primals_1,
            primals_2, buf0, 524288, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf1 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_add_relu_1[grid(262144)](buf1, primals_3, 262144,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf2 = buf0
        del buf0
        triton_poi_fused_add_relu_1[grid(262144)](buf2, primals_5, 262144,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf3 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_add_relu_1[grid(262144)](buf3, primals_7, 262144,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_7
        buf4 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_add_relu_threshold_backward_2[grid(524288)](primals_9,
            primals_10, buf4, buf5, 524288, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_10
        buf6 = buf4
        del buf4
        triton_poi_fused_add_relu_threshold_backward_3[grid(524288)](primals_11,
            primals_12, buf6, buf7, 524288, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_12
        buf7 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_add_relu_threshold_backward_3[grid(524288)](primals_13,
            primals_14, buf7, buf8, 524288, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_14
        buf8 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_add_relu_threshold_backward_3[grid(524288)](primals_15,
            primals_16, buf8, buf9, 524288, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_16
    return (buf8, primals_1, primals_4, primals_6, primals_8, primals_9,
        primals_11, primals_13, primals_15, buf1, buf2, buf3, buf5, buf6,
        buf7, buf9)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        
        layers = []
        current_input_size = input_size
        
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_size = hidden_size
        
        layers.append(nn.Linear(current_input_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, input_0):
        primals_1 = self.network[0].weight
        primals_2 = self.network[0].bias
        primals_3 = self.network[2].weight
        primals_4 = self.network[2].bias
        primals_5 = self.network[4].weight
        primals_6 = self.network[4].bias
        primals_7 = self.network[6].weight
        primals_8 = self.network[6].bias
        primals_9 = self.network[8].weight
        primals_10 = self.network[8].bias
        primals_11 = self.network[10].weight
        primals_12 = self.network[10].bias
        primals_13 = self.network[12].weight
        primals_14 = self.network[12].bias
        primals_15 = self.network[14].weight
        primals_16 = self.network[14].bias
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9,
            primals_10, primals_11, primals_12, primals_13, primals_14,
            primals_15, primals_16], )
        return output[0]
