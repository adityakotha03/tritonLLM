import torch
from torch._inductor.select_algorithm import extern_kernels
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
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = xindex // 1024
    x2 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tmp0 + 1024 * x1), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 1024, tl.int64)
    tmp9 = tl.load(in_ptr1 + (x0 % 256 + 256 * x1), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + x2, tmp10, xmask)


@triton.jit
def triton_poi_fused_tanh_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = libdevice.tanh(tmp2)
    tl.store(in_out_ptr0 + x2, tmp3, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5) = args
    args.clear()
    assert_size_stride(primals_1, (1, 128), (128, 1))
    assert_size_stride(primals_2, (1, 128), (128, 1))
    assert_size_stride(primals_3, (1, 1), (1, 1))
    assert_size_stride(primals_4, (256, 129), (129, 1))
    assert_size_stride(primals_5, (128, 256), (256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((256, 8, 1025), (8192, 1024, 1), torch.
            float32)
        get_raw_stream(0)
        triton_poi_fused_cat_0[grid(262144)](primals_1, primals_2, buf0, 
            262144, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_1
        del primals_2
        buf1 = empty_strided_cuda((256, 8, 256), (2048, 256, 1), torch.float32)
        extern_kernels.addmm(primals_3, buf0, reinterpret_tensor(primals_4,
            (1025, 256), (1, 1025), 0), alpha=1, beta=1, out=buf1)
        del primals_3
        buf2 = reinterpret_tensor(buf1, (256, 8, 256), (2048, 256, 1), 0)
        del buf1
        triton_poi_fused_tanh_2[grid(262144)](buf2, primals_4, 262144,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_4
        buf3 = empty_strided_cuda((256, 8, 128), (1024, 128, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf2, (262144, 256), (256, 1),
            0), reinterpret_tensor(primals_5, (256, 128), (1, 256), 0),
            out=buf3)
        buf4 = reinterpret_tensor(buf3, (256, 8, 128), (1024, 128, 1), 0)
        del buf3
        buf5 = reinterpret_tensor(buf0, (256, 8, 1025), (8192, 1024, 1), 0)
        del buf0
        extern_kernels.addmm(primals_3, buf4, reinterpret_tensor(primals_5,
            (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf5)
        del primals_3
    return buf5, reinterpret_tensor(buf2, (262144, 256), (256, 1), 0
        ), reinterpret_tensor(primals_5, (256, 128), (1, 256), 0
        ), reinterpret_tensor(buf4, (256, 128), (128, 1), 0)


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

        # Define the RNN cell components (input to hidden, hidden to hidden, and hidden to output)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # Input to hidden
        self.h2o = nn.Linear(hidden_size, output_size)  # Hidden to output
        self.tanh = nn.Tanh()  # Activation function for hidden state

    def forward(self, input_0, input_1):
        primals_4 = self.i2h.weight
        primals_5 = self.h2o.weight
        primals_3 = self.h2o.bias
        primals_1 = input_1
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]