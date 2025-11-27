import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_relu_threshold_backward_0(in_out_ptr0, in_ptr0,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4194304
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
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x2, tmp4, xmask)
    tl.store(out_ptr0 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_relu_threshold_backward_1(in_out_ptr0, in_ptr0,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4194304
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
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x2, tmp4, xmask)
    tl.store(out_ptr0 + x2, tmp6, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7) = args
    args.clear()
    assert_size_stride(primals_1, (32768, 16384), (16384, 1))
    assert_size_stride(primals_2, (32768,), (1,))
    assert_size_stride(primals_3, (128, 16384), (16384, 1))
    assert_size_stride(primals_4, (32768,), (1,))
    assert_size_stride(primals_5, (32768, 32768), (32768, 1))
    assert_size_stride(primals_6, (32768,), (1,))
    assert_size_stride(primals_7, (16384, 32768), (32768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_0[grid(4194304)](buf0,
            primals_1, primals_2, 4194304, XBLOCK=512, num_warps=4, num_stages=1
            )
        del primals_1
        del primals_2
        buf1 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_0[grid(4194304)](buf1,
            primals_3, primals_4, 4194304, XBLOCK=512, num_warps=4, num_stages=1
            )
        del primals_3
        del primals_4
        buf2 = empty_strided_cuda((128, 32768), (32768, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_1[grid(4194304)](buf2,
            primals_5, primals_6, 4194304, XBLOCK=512, num_warps=4, num_stages=1
            )
        del primals_5
        del primals_6
        buf3 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused_relu_threshold_backward_1[grid(2097152)](buf3,
            primals_7, primals_7, 2097152, XBLOCK=512, num_warps=4, num_stages=1
            )
        del primals_7
    return reinterpret_tensor(buf3, (128, 16384), (16384, 1), 0
        ), buf0, buf1, buf2, primals_6, reinterpret_tensor(primals_3, (16384,
        128), (1, 16384), 0), reinterpret_tensor(primals_1, (32768, 32768),
        (1, 32768), 0)


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
        primals_9 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7])
        return output[0]
