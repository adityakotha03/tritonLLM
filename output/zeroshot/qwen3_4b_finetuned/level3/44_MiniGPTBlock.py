import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
import torch.nn.functional as F
import math
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_tanh_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.28854335818583508
    tmp2 = tmp0 * tmp1
    tmp3 = 0.4471523819932116
    tmp4 = tmp0 * tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tmp4 * tmp5
    tmp7 = tmp2 + tmp6
    tmp8 = tmp7 * tmp4
    tmp9 = tmp8 + tmp2
    tmp10 = libdevice.tanh(tmp9)
    tmp11 = 1.0
    tmp12 = tmp10 + tmp11
    tl.store(out_ptr0 + x0, tmp12, xmask)


@triton.jit
def triton_poi_fused_tanh_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0 * 0.28854335818583508
    tmp2 = tmp0 * 0.4471523819932116
    tmp3 = tmp2 * tmp2
    tmp4 = tmp2 * tmp3
    tmp5 = tmp1 + tmp4
    tmp6 = tmp5 * tmp2
    tmp7 = tmp6 + tmp1
    tmp8 = libdevice.tanh(tmp7)
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tl.store(out_ptr0 + x0, tmp10, xmask)


@triton.jit
def triton_poi_fused_mul_2(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 409600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 2560 % 2048
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_out_ptr0 + (x1, x3 % 2048 + 2048 * (x3 // 2048), 2560),
        xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_mul_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 409600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 2560 % 2048
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x3, tmp4, xmask)


@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 409600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 2560 % 2048
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x1, x3 % 2048 + 2048 * (x3 // 2048), 2560),
        xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + x3, tmp4, xmask)


@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 409600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 2560 % 2048
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x1, x3 % 2048 + 2048 * (x3 // 2048), 2560),
        xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + x3, tmp4, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15, primals_16) = args
    args.clear()
    assert_size_stride(primals_1, (768, 768), (768, 1))
    assert_size_stride(primals_2, (768,), (1,))
    assert_size_stride(primals_3, (128, 512, 768), (393216, 768, 1))
    assert_size_stride(primals_4, (768, 768), (768, 1))
    assert_size_stride(primals_5, (768,), (1,))
    assert_size_stride(primals_6, (768, 768), (768, 1))
    assert_size_stride(primals_7, (768,), (1,))
    assert_size_stride(primals_8, (8, 256), (256, 1))
    assert_size_stride(primals_9, (8, 256), (256, 1))
    assert_size_stride(primals_10, (8, 256), (256, 1))
    assert_size_stride(primals_11, (8, 256), (256, 1))
    assert_size_stride(primals_12, (256, 768), (768, 1))
    assert_size_stride(primals_13, (256,), (1,))
    assert_size_stride(primals_14, (768, 256), (256, 1))
    assert_size_stride(primals_15, (768,), (1,))
    assert_size_stride(primals_16, (768, 256), (256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 512, 768), (393216, 768, 1), torch.
            float32)
        extern_kernels.addmm(primals_2, primals_3, reinterpret_tensor(
            primals_1, (768, 768), (1, 768), 0), alpha=1, beta=1,
            out=buf0)
        del primals_1
        del primals_2
        buf1 = empty_strided_cuda((128, 512, 768), (393216, 768, 1), torch.
            float32)
        extern_kernels.addmm(primals_5, buf0, reinterpret_tensor(primals_4,
            (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf1)
        del primals_5
        del primals_4
        buf2 = empty_strided_cuda((128, 8, 512, 256), (1048576, 131072, 256,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_tanh_0[grid(65536)](buf1, buf2, 65536, XBLOCK=512,
            num_warps=8, num_stages=1)
        buf3 = empty_strided_cuda((128, 8, 512, 256), (1048576, 131072, 256,
            1), torch.float32)
        triton_poi_fused_tanh_0[grid(65536)](buf1, buf3, 65536, XBLOCK=512,
            num_warps=8, num_stages=1)
        buf4 = empty_strided_cuda((128, 8, 2048, 256), (4194304, 524288, 256,
            1), torch.float32)
        triton_poi_fused_0[grid(409600)](buf2, buf4, 409600, XBLOCK=1024,
            num_warps=16, num_stages=1)
        buf5 = empty_strided_cuda((128, 8, 512, 256), (1048576, 131072, 256,
            1), torch.float32)
        triton_poi_fused_0[grid(409600)](buf3, buf5, 409600, XBLOCK=1024,
            num_warps=16, num_stages=1)
        buf6 = empty_strided_cuda((128, 8, 512, 256), (1048576, 131072, 256,
            1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf4, (128, 8, 2048), (1048576
            , 131072), 0), reinterpret_tensor(buf5, (128, 8, 2048), (1048576,
            131072), 0), out=buf6)
        buf7 = reinterpret_tensor(buf6, (128, 8, 512, 256), (1048576, 131072,
            256, 1), 0)
        del buf6
        triton_poi_fused_mul_2[grid(409600)](buf7, buf7, 409600, XBLOCK=256,
            num_warps=4, num_stages=1)
        buf8 = reinterpret_tensor(buf7, (128, 512, 256), (32768, 256, 1), 0)
        del buf7
        triton_poi_fused_mul_3[grid(409600)](buf8, primals_7, 409600,
            XBLOCK=256, num_warps=4, num_stages=1)
        buf9 = reinterpret_tensor(buf4, (128, 8, 2048), (1048576, 131072, 1),
            0)
        del buf4
        extern_kernels.bmm(buf5, buf8, out=buf9)
        buf10 = reinterpret_tensor(buf8, (128, 512, 256), (32768, 256, 1), 0)
        del buf8
        triton_poi_fused_mul_3[grid(409600)](buf10, primals_7, 409600,
            XBLOCK=256, num_warps=4, num_stages=1)
        buf11 = reinterpret_tensor(buf5, (128, 8, 512), (524288, 65536, 1), 0)
        del buf5
        extern_kernels.bmm(buf9, primals_12, out=buf11)
        buf12 = empty_strided_cuda((128, 512, 768), (393216, 768, 1), torch.
            float32)
        extern_kernels.addmm(primals_13, buf11, reinterpret_tensor(
            primals_14, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf12)
        buf13 = empty_strided_cuda((128, 512, 768), (393216, 768, 1), torch.
            float32)
        extern_kernels.addmm(primals_15, primals_16, reinterpret_tensor(
            primals_13, (256, 768), (768, 1), 0), alpha=1, beta=1, out=buf13)
    return (buf13, primals_3, primals_7, primals_12, primals_13, primals_15,
        primals_16, buf0, buf1, reinterpret_tensor(buf2, (128, 8, 512, 256),
        (1048576, 131072, 256, 1), 0), reinterpret_tensor(buf3, (128, 8, 
        512, 256), (1048576, 131072, 256, 1), 0), buf9, reinterpret_tensor(
        buf11, (128, 512, 256), (32768, 256, 1), 0))


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class ModelNew(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, input_0):
        primals_1 = self.ln_1.weight
        primals_2 = self.ln_1.bias
        primals_4 = self.attn.c_attn.weight
        primals_5 = self.attn.c_attn.bias
        primals_7 = self.attn.c_proj.weight
        primals_8 = self.attn.c_proj.bias
        primals_12 = self.ln_2.weight
        primals_13 = self.ln_2.bias
        primals_14 = self.mlp.c_fc.weight
        primals_15 = self.mlp.c_fc.bias
        primals_16 = self.mlp.c_proj.weight
        primals_17 = self.mlp.c_proj.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_7, primals_8, primals_12, primals_13,
            primals_14, primals_15, primals_16, primals_17])
        return output[0]
