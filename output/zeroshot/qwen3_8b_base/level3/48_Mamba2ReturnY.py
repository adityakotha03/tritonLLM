import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange

@triton.jit
def einsum_kernel(
    A_ptr, B_ptr, C_ptr, 
    out_ptr, 
    n_heads, n_heads_b, n_heads_c, 
    d_head, d_state, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    mask = tl.arange(0, BLOCK_SIZE) < n_heads * d_state * d_head

    # Load A and B
    A = tl.load(A_ptr + offset + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    B = tl.load(B_ptr + offset + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    C = tl.load(C_ptr + offset + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

    # Compute einsum
    out = tl.dot(A, B, C)

    # Store result
    tl.store(out_ptr + offset + tl.arange(0, BLOCK_SIZE), out, mask=mask)

def triton_einsum(A, B, C, n_heads, d_head, d_state):
    assert A.is_cuda and B.is_cuda and C.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    
    out = torch.empty_like(A)
    n_elements = A.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    einsum_kernel[grid](A, B, C, out, n_heads, n_heads, n_heads, d_head, d_state, BLOCK_SIZE=BLOCK_SIZE)
    return out

@triton.jit
def segsum_kernel(
    A_ptr, 
    out_ptr, 
    n_heads, n_heads_b, n_heads_c, 
    d_head, d_state, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    mask = tl.arange(0, BLOCK_SIZE) < n_heads * d_state * d_head

    A = tl.load(A_ptr + offset + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    cumsum = tl.cumsum(A, axis=0)
    segsum = cumsum[:, :, None] - cumsum[:, None, :]
    mask = tl.tril(tl.ones((d_state, d_state), dtype=tl.bool), diagonal=0)
    segsum = segsum.masked_fill(~mask, -tl.inf)

    tl.store(out_ptr + offset + tl.arange(0, BLOCK_SIZE), segsum, mask=mask)

def triton_segsum(A, n_heads, d_state):
    assert A.is_cuda, "Tensor must be on CUDA."
    A = A.contiguous()
    
    out = torch.empty_like(A)
    n_elements = A.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    segsum_kernel[grid](A, out, n_heads, n_heads, n_heads, d_head, d_state, BLOCK_SIZE=BLOCK_SIZE)
    return out

@triton.jit
def exp_kernel(
    A_ptr, 
    out_ptr, 
    n_heads, n_heads_b, n_heads_c, 
    d_head, d_state, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    mask = tl.arange(0, BLOCK_SIZE) < n_heads * d_state * d_head

    A = tl.load(A_ptr + offset + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    out = tl.exp(A)

    tl.store(out_ptr + offset + tl.arange(0, BLOCK_SIZE), out, mask=mask)

def triton_exp(A, n_heads, d_state):
    assert A.is_cuda, "Tensor must be on CUDA."
    A = A.contiguous()
    
    out = torch.empty_like(A)
    n_elements = A.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    exp_kernel[grid](A, out, n_heads, n_heads, n_heads, d_head, d_state, BLOCK_SIZE=BLOCK_SIZE)
    return out

@triton.jit
def einsum2_kernel(
    A_ptr, B_ptr, C_ptr, 
    out_ptr, 
    n_heads, n_heads_b, n_heads_c, 
    d_head, d_state, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    mask = tl.arange(0, BLOCK_SIZE) < n_heads * d_state * d_head

    # Load A and B
    A = tl.load(A_ptr + offset + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    B = tl.load(B_ptr + offset + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    C = tl.load(C_ptr + offset + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

    # Compute einsum
    out = tl.dot(A, B, C)

    # Store result
    tl.store(out_ptr + offset + tl.arange(0, BLOCK_SIZE), out, mask=mask)

def triton_einsum2(A, B, C, n_heads, d_head, d_state):
    assert A.is_cuda and B.is_cuda and C.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    
    out = torch.empty_like(A)
    n_elements = A.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    einsum2_kernel[grid](A, B, C, out, n_heads, n_heads, n_heads, d_head, d_state, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super(ModelNew, self).__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        
    def segsum(self, x):
        return triton_segsum(x, self.n_heads, self.d_state)
    
    def forward(self, X, initial_states=None):
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = triton_exp(triton_segsum(A_blocks, self.n_heads, self.d_state), self.n_heads, self.d_state)
        
        L = triton_exp(self.segsum(A_blocks))
        Y_diag = triton_einsum(C_blocks, B_blocks, L, self.n_heads, self.d_head, self.d_state)
        
        decay_states = triton_exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = triton_einsum2(B_blocks, decay_states, X_blocks, self.n_heads, self.d_head, self.d_state)
        
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        
        decay_chunk = triton_exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = triton_einsum2(decay_chunk, states, torch.ones_like(states), self.n_heads, self.d_head, self.d_state)
        states = new_states[:, :-1]
        
        state_decay_out = triton_exp(A_cumsum)
        Y_off = triton_einsum(C_blocks, states, state_decay_out, self.n_heads, self.d_head, self.d_state)
        
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        
        return Y