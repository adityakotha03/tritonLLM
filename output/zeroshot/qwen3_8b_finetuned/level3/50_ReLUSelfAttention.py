import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import math
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 3
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 4096 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_mul_rsub_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 3
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 4096 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = 0.5
    tmp3 = tmp0 * tmp2
    tmp4 = 0.044715
    tmp5 = tmp1 * tmp4
    tmp6 = tmp1 * tmp1
    tmp7 = tmp6 * tmp1
    tmp8 = tmp5 + tmp7
    tmp9 = 1.4142135623730951
    tmp10 = tmp8 * tmp9
    tmp11 = 2.5066282746310002
    tmp12 = tmp10 / tmp11
    tmp13 = tl_math.tanh(tmp12)
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = tmp3 * tmp15
    tl.store(out_ptr0 + x2, tmp16, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (3, 768, 768), (589824, 768, 1))
    assert_size_stride(primals_2, (3, 768), (768, 1))
    assert_size_stride(primals_3, (3, 768), (768, 1))
    assert_size_stride(primals_4, (768, 768), (768, 1))
    assert_size_stride(primals_5, (768,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 1024, 3), (3072, 3, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(589824)](primals_1, primals_2, buf0, 
            589824, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf1 = empty_strided_cuda((16, 1024, 3), (3072, 3, 1), torch.float32)
        triton_poi_fused_add_0[grid(589824)](primals_1, primals_3, buf1, 
            589824, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_3
        buf2 = empty_strided_cuda((16, 1024, 3), (3072, 3, 1), torch.float32)
        triton_poi_fused_add_0[grid(589824)](primals_1, primals_3, buf2, 
            589824, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_3
        buf3 = empty_strided_cuda((16, 1024, 3), (3072, 3, 1), torch.float32)
        triton_poi_fused_add_0[grid(589824)](primals_1, primals_2, buf3, 
            589824, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf4 = empty_strided_cuda((16, 1024, 3), (3072, 3, 1), torch.float32)
        triton_poi_fused_add_0[grid(589824)](primals_1, primals_3, buf4, 
            589824, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        buf5 = empty_strided_cuda((16, 1024, 768), (786432, 768, 1),
            torch.float32)
        triton_poi_fused_add_mul_rsub_1[grid(589824)](buf0, buf5, 589824,
            XBLOCK=128, num_warps=4, num_stages=1)
        buf6 = reinterpret_tensor(buf0, (16, 1024, 768), (786432, 768, 1), 0)
        del buf0
        triton_poi_fused_add_mul_rsub_1[grid(589824)](buf1, buf6, 589824,
            XBLOCK=128, num_warps=4, num_stages=1)
        buf7 = reinterpret_tensor(buf1, (16, 1024, 768), (786432, 768, 1), 0)
        del buf1
        triton_poi_fused_add_mul_rsub_1[grid(589824)](buf2, buf7, 589824,
            XBLOCK=128, num_warps=4, num_stages=1)
        buf8 = reinterpret_tensor(buf2, (16, 1024, 768), (786432, 768, 1), 0)
        del buf2
        triton_poi_fused_add_mul_rsub_1[grid(589824)](buf3, buf8, 589824,
            XBLOCK=128, num_warps=4, num_stages=1)
        buf9 = empty_strided_cuda((16, 1024, 768), (786432, 768, 1),
            torch.float32)
        buf10 = reinterpret_tensor(buf9, (16, 768, 1024), (786432, 1024, 1),
            0)
        del buf9
        extern_kernels.addmm(primals_5, buf5, reinterpret_tensor(primals_4,
            (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf10)
        del primals_5
    return reinterpret_tensor(buf10, (16, 1024, 768), (786432, 1, 1024), 0
        ), primals_4, buf4, buf6, buf7, buf8, reinterpret_tensor(primals_4,
        (768, 768), (1, 768), 0)


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class ModelNew(nn.Module):
    """
    A multi-head masked self-attention layer with a projection at the end that uses ReLU instead of Softmax.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, input_0):
        primals_1 = self.c_attn.weight
        primals_2 = self.c_attn.bias
        primals_3 = self.c_proj.weight
        primals_4 = self.c_proj.bias
        primals_5 = self.bias
        output = call([input_0, primals_1, primals_2, primals_3, primals_4,
            primals_5])
        return output[0]