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
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32768
    x1 = xindex // 32768
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 16384, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16384 * x1), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 32768, tl.int64)
    tmp9 = tl.load(in_ptr1 + (x0 - 16384 + 16384 * x1), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + x3, xmask, eviction_policy='evict_last')
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp6, tmp11, tmp5)
    tl.store(out_ptr0 + x3, tmp12, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (16384, 16384), (16384, 1))
    assert_size_stride(primals_2, (16384,), (1,))
    assert_size_stride(primals_3, (256, 16384), (16384, 1))
    assert_size_stride(primals_4, (256, 16384), (16384, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((256, 32768), (32768, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_cat_0[grid(8388608)](primals_3, primals_4,
            primals_2, buf0, 8388608, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_2
        del primals_3
        del primals_4
    return buf0, primals_1


class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Vanilla RNN model.
        
        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = torch.randn((batch_size, hidden_size))
        
        # Define the RNN cell components (input to hidden, hidden to hidden, and hidden to output)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # Input to hidden
        self.h2o = nn.Linear(hidden_size, output_size)  # Hidden to output
        self.tanh = nn.Tanh()  # Activation function for hidden state
    
    def forward(self, input_0):
        primals_1 = self.i2h.weight
        primals_2 = self.i2h.bias
        primals_4 = self.h2o.weight
        primals_3 = self.h2o.bias
        primals_1 = self.i2h.weight
        primals_2 = self.i2h.bias
        primals_4 = self.h2o.weight
        primals_3 = self.h2o.bias
        primals_1 = self.i2h.weight
        primals_2 = self.i2h.bias
        primals_4 = self.h2o.weight
        primals_3 = self.h2o.bias
        buf0, primals_1 = call([primals_1, primals_2, input_0, primals_4])
        del primals_1
        del input_0
        del primals_2
        del primals_4
        return buf0