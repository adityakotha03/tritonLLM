import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32768
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (32768, 16384), (16384, 1))
    assert_size_stride(primals_2, (32768,), (1,))
    assert_size_stride(primals_3, (16384, 32768), (32768, 1))
    assert_size_stride(primals_4, (16384,), (1,))
    assert_size_stride(primals_5, (16384, 16384), (16384, 1))
    assert_size_stride(primals_5, (16384,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((131072, 16384), (16384, 1), torch.float32)
        extern_kernels.mm(primals_1, reinterpret_tensor(primals_2, (16384,
            1), (1, 16384), 0), out=buf0)
        del primals_1
        del primals_2
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_relu_0[grid(131072)](buf1, primals_3, 131072,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
        buf2 = empty_strided_cuda((131072, 16384), (16384, 1), torch.float32)
        extern_kernels.addmm(primals_4, buf1, reinterpret_tensor(primals_5,
            (16384, 1), (1, 16384), 0), alpha=1, beta=1, out=buf2)
        del primals_4
    return reinterpret_tensor(buf2, (128, 16384), (16384, 1), 0
        ), reinterpret_tensor(primals_5, (16384, 1), (1, 16384), 0
        ), buf1, primals_5


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
        primals_7 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5])
        return output[0]
