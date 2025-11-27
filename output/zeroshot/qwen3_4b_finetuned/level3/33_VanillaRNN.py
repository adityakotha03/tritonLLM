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
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 4096000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16384
    x1 = xindex // 16384
    x2 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 16384, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (16384 * x1 + x0), tmp4 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 32768, tl.int64)
    tmp9 = tl.load(in_ptr1 + (16384 * x1 + (-16384 + x0)), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + x2, tmp10, xmask)


@triton.jit
def triton_poi_fused_tanh_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 4096000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16384
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp7 = -1.0
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6 = args
    args.clear()
    assert_size_stride(primals_1, (256, 16384), (16384, 1))
    assert_size_stride(primals_2, (256, 16384), (16384, 1))
    assert_size_stride(primals_3, (16384, 32768), (32768, 1))
    assert_size_stride(primals_4, (16384,), (1,))
    assert_size_stride(primals_5, (8192, 16384), (16384, 1))
    assert_size_stride(primals_6, (8192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((256, 32768), (32768, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_cat_0[grid(4096000)](primals_1, primals_2, buf0,
            4096000, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_1
        del primals_2
        buf1 = empty_strided_cuda((256, 16384), (16384, 1), torch.float32)
        extern_kernels.mm(buf0, reinterpret_tensor(primals_3, (32768, 16384
            ), (1, 32768), 0), out=buf1)
        del primals_3
        buf2 = buf1
        del buf1
        triton_poi_fused_tanh_1[grid(4096000)](buf2, primals_4, 4096000,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_4
        buf3 = empty_strided_cuda((256, 8192), (8192, 1), torch.float32)
        extern_kernels.mm(buf2, reinterpret_tensor(primals_5, (16384, 8192),
            (1, 16384), 0), out=buf3)
        buf4 = empty_strided_cuda((256, 16384), (16384, 1), torch.float32)
        triton_poi_fused_cat_2[grid(16384)](primals_6, buf3, buf4, 16384,
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf3
        del primals_6
    return buf4, buf0, buf2, primals_5


class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Vanilla RNN model.
        
        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = torch.randn((batch_size, hidden_size))
        
        # Define the RNN cell components (input to hidden, hidden to hidden, and hidden to output)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # Input to hidden
        self.h2o = nn.Linear(hidden_size, output_size)  # Hidden to output
        self.tanh = nn.Tanh()  # Activation function for hidden state
    
    def forward(self, input_0, input_1):
        primals_3 = self.i2h.weight
        primals_4 = self.i2h.bias
        primals_5 = self.h2o.weight
        primals_6 = self.h2o.bias
        primals_1 = input_0
        primals_2 = input_1
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6])
        return output[0]
