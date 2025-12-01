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
def triton_poi_fused_add_mul_reciprocal_rsub_0(in_ptr0, in_ptr1, in_ptr2,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16384
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x2, xmask)
    tmp6 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tmp2 + 1.0
    tmp5 = tmp3 / tmp4
    tmp7 = tmp6 * tmp5
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp8 - tmp9
    tl.store(out_ptr0 + x2, tmp10, xmask)


@triton.jit
def triton_poi_fused_add_relu_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16384
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8) = args
    args.clear()
    assert_size_stride(primals_1, (16384, 16384), (16384, 1))
    assert_size_stride(primals_2, (16384,), (1,))
    assert_size_stride(primals_3, (128, 16384), (16384, 1))
    assert_size_stride(primals_4, (16384, 16384), (16384, 1))
    assert_size_stride(primals_5, (16384,), (1,))
    assert_size_stride(primals_6, (16384, 16384), (16384, 1))
    assert_size_stride(primals_7, (16384,), (1,))
    assert_size_stride(primals_8, (8192, 16384), (16384, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_mul_reciprocal_rsub_0[grid(2097152)](primals_3,
            primals_1, primals_2, buf0, 2097152, XBLOCK=128, num_warps=4,
            num_stages=1)
        del primals_2
        buf1 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_add_mul_reciprocal_rsub_0[grid(2097152)](buf0,
            primals_4, primals_5, buf1, 2097152, XBLOCK=128, num_warps=4,
            num_stages=1)
        del primals_5
        buf2 = empty_strided_cuda((128, 8192), (8192, 1), torch.float32)
        triton_poi_fused_add_mul_reciprocal_rsub_0[grid(1048576)](buf1,
            primals_6, primals_7, buf2, 1048576, XBLOCK=128, num_warps=4,
            num_stages=1)
        del primals_7
        buf3 = buf1
        del buf1
        triton_poi_fused_add_relu_1[grid(2097152)](buf3, primals_8, 2097152,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_8
    return buf3, primals_1, primals_3, primals_4, primals_6, buf0, buf2


class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        
        layers = []
        current_input_size = input_size
        
        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_input_size, layer_size))
            layers.append(nn.ReLU())
            current_input_size = layer_size
        
        layers.append(nn.Linear(current_input_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, input_0):
        primals_1 = self.network[0].weight
        primals_2 = self.network[0].bias
        primals_4 = self.network[2].weight
        primals_5 = self.network[2].bias
        primals_6 = self.network[4].weight
        primals_7 = self.network[4].bias
        primals_8 = self.network[6].weight
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8])
        return output[0]