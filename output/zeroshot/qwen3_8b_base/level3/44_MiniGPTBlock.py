import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

class NewGELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

@triton.jit
def gelu_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute GELU approximation
    x = x * (1.0 + tl.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
    tl.store(out_ptr + offsets, x, mask=mask)

def triton_gelu(x: torch.Tensor):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

@triton.jit
def matmul_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    n_head, head_size, 
    BLOCK_SIZE: tl.constexpr,
):
    # Get the current program ID
    pid = tl.program_id(0)
    # Compute the block index
    block_idx = pid // (n_head * head_size)
    head_idx = (pid // head_size) % n_head
    # Compute the offset for the current block
    block_start = block_idx * BLOCK_SIZE
    # Compute the offset for the current head
    head_offset = head_idx * head_size
    # Compute the offset for the current block in the head
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    # Compute the mask
    mask = offset < (n_head * head_size)
    # Load q, k, v
    q = tl.load(q_ptr + head_offset + offset, mask=mask, other=0.0)
    k = tl.load(k_ptr + head_offset + offset, mask=mask, other=0.0)
    v = tl.load(v_ptr + head_offset + offset, mask=mask, other=0.0)
    # Compute qk
    qk = tl.dot(q, k)
    # Scale qk
    qk = qk * (1.0 / math.sqrt(head_size))
    # Apply softmax
    softmax = tl.softmax(qk, axis=-1)
    # Apply dropout
    # (Assume dropout is applied in the main model)
    # Compute attention
    out = tl.dot(softmax, v)
    # Store the result
    tl.store(out_ptr + head_offset + offset, out, mask=mask)

def triton_attention(q, k, v, n_head, head_size):
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Tensors must be on CUDA."
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)
    n_elements = q.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_head * head_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    matmul_kernel[grid](q, k, v, out, n_head, head_size, BLOCK_SIZE=BLOCK_SIZE)
    return out

class CausalSelfAttentionNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_size)
        k = k.view(B, T, self.n_head, self.head_size)
        v = v.view(B, T, self.n_head, self.head_size)
        # Apply attention
        out = triton_attention(q, k, v, self.n_head, self.head_size)
        out = out.view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttentionNew(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        self.mlpf = lambda x: self.mlp.dropout(self.mlp.c_proj(self.mlp.act(self.mlp.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x