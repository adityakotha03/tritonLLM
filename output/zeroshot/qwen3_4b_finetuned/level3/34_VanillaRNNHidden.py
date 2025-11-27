import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + 1024 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = libdevice.tanh(tmp2)
    tmp5 = tmp4 + tmp3
    tl.store(out_ptr0 + x2, tmp5, xmask)


@triton.jit
def triton_poi_fused_add_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8) = args
    args.clear()
    assert_size_stride(primals_1, (256, 1024 + 256), (1024 + 256, 1))
    assert_size_stride(primals_2, (256,), (1,))
    assert_size_stride(primals_3, (256, 128), (128, 1))
    assert_size_stride(primals_4, (256, 1024 + 256), (1024 + 256, 1))
    assert_size_stride(primals_5, (256,), (1,))
    assert_size_stride(primals_6, (256, 256), (256, 1))
    assert_size_stride(primals_7, (256,), (1,))
    assert_size_stride(primals_8, (128,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_cat_0[grid(2097152)](primals_4, primals_2,
            primals_1, buf0, 2097152, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
        del primals_1
        buf1 = empty_strided_cuda((256, 128), (128, 1), torch.float32)
        triton_poi_fused_add_1[grid(2097152)](buf1, primals_3, 2097152,
            XBLOCK=512, num_warps=8, num_stages=1)
        del primals_3
        buf2 = empty_strided_cuda((256, 128), (128, 1), torch.float32)
        triton_poi_fused_add_1[grid(2097152)](buf2, primals_5, 2097152,
            XBLOCK=512, num_warps=8, num_stages=1)
        del primals_5
        buf3 = empty_strided_cuda((256, 128), (128, 1), torch.float32)
        triton_poi_fused_add_1[grid(2097152)](buf3, primals_6, 2097152,
            XBLOCK=512, num_warps=8, num_stages=1)
        del primals_6
        buf4 = empty_strided_cuda((256, 128), (128, 1), torch.float32)
        triton_poi_fused_add_1[grid(2097152)](buf4, primals_8, 2097152,
            XBLOCK=512, num_warps=8, num_stages=1)
        del primals_8
    return buf4, primals_7, buf0, buf1, buf2, buf3, primals_4, reinterpret_tensor(
        buf0, (1, 256, 256), (256, 1, 1), 0)


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
        primals_1 = self.i2h.weight
        primals_2 = self.i2h.bias
        primals_3 = self.h2o.weight
        primals_5 = self.h2o.bias
        primals_4 = input_0
        primals_6 = input_1
        primals_7 = primals_2
        primals_8 = primals_5
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8])
        return output[0]
