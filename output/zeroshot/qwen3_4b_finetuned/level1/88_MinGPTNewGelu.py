import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_per_fused_tanh_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 8192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = 1.0
    tmp2 = 0.5
    tmp3 = tmp0 * tmp1
    tmp4 = 0.044715
    tmp5 = tmp0 * tmp0
    tmp6 = tmp5 * tmp0
    tmp7 = tmp4 * tmp6
    tmp8 = tmp3 + tmp7
    tmp9 = 0.7978845608028654
    tmp10 = tmp8 * tmp9
    tmp11 = 2.0
    tmp12 = 1.0 / tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = triton_helpers.tanh(tmp13)
    tmp15 = tmp14 + tmp1
    tmp16 = tmp3 * tmp15
    tl.store(out_ptr0 + x2, tmp16, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (8192, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8192, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_per_fused_tanh_0[grid(67108864)](arg0_1, buf0, 67108864,
            XBLOCK=512, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


def get_inputs():
    return [torch.rand(8192, 8192, dtype=torch.float32)]


def get_init_inputs():
    return []


def run(*args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.cuda._DeviceGuard(0)
        del buf0
    return call([get_inputs()[0]])


class ModelNew(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
