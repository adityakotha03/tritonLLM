import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_tanh_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask)
    tmp3 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = 0.5
    tmp5 = tmp2 * tmp4
    tmp6 = tl.math.tanh(tmp5)
    tl.store(in_out_ptr0 + x0, tmp6, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7) = args
    args.clear()
    assert_size_stride(primals_1, (256, 16384), (16384, 1))
    assert_size_stride(primals_2, (256, 16384), (16384, 1))
    assert_size_stride(primals_3, (16384, 16384), (16384, 1))
    assert_size_stride(primals_4, (16384,), (1,))
    assert_size_stride(primals_5, (8192, 16384), (16384, 1))
    assert_size_stride(primals_6, (8192,), (1,))
    assert_size_stride(primals_7, (8192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((256, 16384), (16384, 1), torch.float32)
        extern_kernels.mm(primals_2, reinterpret_tensor(primals_3, (16384,
            16384), (1, 16384), 0), out=buf0)
        del primals_3
        buf1 = empty_strided_cuda((256, 16384), (16384, 1), torch.float32)
        extern_kernels.mm(primals_1, reinterpret_tensor(primals_5, (16384,
            8192), (1, 16384), 0), out=buf1)
        buf2 = empty_strided_cuda((256, 16384), (16384, 1), torch.float32)
        buf4 = reinterpret_tensor(buf2, (256, 16384), (16384, 1), 0)
        del buf2
        get_raw_stream(0)
        triton_poi_fused_tanh_0[grid(4194304)](buf4, primals_1, buf1, 4194304
            , XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_1
        del buf1
    return buf4, primals_2, buf0, primals_5, primals_6, primals_7


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
        primals_3 = self.i2h.weight
        primals_4 = self.i2h.bias
        primals_5 = self.h2o.weight
        primals_6 = self.h2o.bias
        primals_1 = input_0
        primals_2 = self.hidden
        primals_7 = self.tanh.weight
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7])
        return output[0]
