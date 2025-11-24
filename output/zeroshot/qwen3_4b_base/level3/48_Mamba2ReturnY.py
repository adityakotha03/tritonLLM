import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < m

    # Load A: (m, k) -> (BLOCK_SIZE, k)
    a = tl.load(a_ptr + offsets[:, None] * k + tl.arange(0, k)[None, :], mask=mask, other=0.0)
    
    # Load B: (k, n) -> (k, BLOCK_SIZE)
    b = tl.load(b_ptr + tl.arange(0, k)[None, :] * n + offsets[:, None], mask=mask, other=0.0)
    
    # Compute C: (m, n)
    c = tl.dot(a, b)
    
    # Store result
    tl.store(c_ptr + offsets[:, None] * n + tl.arange(0, n)[None, :], c, mask=mask)


@triton.jit
def exp_cumsum_kernel(
    x_ptr, y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input x (cumsum of A_blocks)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute exponential of cumulative sum
    # We compute exp(x) element-wise
    exp_x = tl.exp(x)
    
    # Store result
    tl.store(y_ptr + offsets, exp_x, mask=mask)


@triton.jit
def einsum_diag_kernel(
    c_ptr, b_ptr, l_ptr, x_ptr,
    m, n, k,
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel computes: C @ B @ L @ X
    # Shape: (m, n) = (m, k) @ (k, n) -> (m, n)
    # We assume input tensors are rearranged to block form (b, h, c, l)
    
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < m

    # Load C: (m, k)
    c = tl.load(c_ptr + offsets[:, None] * k + tl.arange(0, k)[None, :], mask=mask, other=0.0)
    
    # Load B: (k, n)
    b = tl.load(b_ptr + tl.arange(0, k)[None, :] * n + offsets[:, None], mask=mask, other=0.0)
    
    # Load L: (m, n)
    l = tl.load(l_ptr + offsets[:, None] * n + tl.arange(0, n)[None, :], mask=mask, other=0.0)
    
    # Load X: (m, n)
    x = tl.load(x_ptr + offsets[:, None] * n + tl.arange(0, n)[None, :], mask=mask, other=0.0)
    
    # Compute: C @ B @ L @ X
    # Step 1: C @ B -> (m, n)
    c_b = tl.dot(c, b)
    
    # Step 2: (C @ B) @ L -> (m, n)
    c_b_l = tl.dot(c_b, l)
    
    # Step 3: (C @ B @ L) @ X -> (m, n)
    c_b_l_x = tl.dot(c_b_l, x)
    
    # Store result
    tl.store(c_b_l_x + offsets[:, None] * n + tl.arange(0, n)[None, :], c_b_l_x, mask=mask)


@triton.jit
def einsum_state_kernel(
    b_ptr, decay_ptr, x_ptr,
    m, n, k,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < m

    # Load B: (m, k)
    b = tl.load(b_ptr + offsets[:, None] * k + tl.arange(0, k)[None, :], mask=mask, other=0.0)
    
    # Load decay: (m, k)
    decay = tl.load(decay_ptr + offsets[:, None] * k + tl.arange(0, k)[None, :], mask=mask, other=0.0)
    
    # Load X: (m, k)
    x = tl.load(x_ptr + offsets[:, None] * k + tl.arange(0, k)[None, :], mask=mask, other=0.0)
    
    # Compute: B @ decay @ X
    b_decay = tl.dot(b, decay)
    result = tl.dot(b_decay, x)
    
    tl.store(result + offsets[:, None] * k + tl.arange(0, k)[None, :], result, mask=mask)


@triton.jit
def einsum_inter_chunk_kernel(
    decay_ptr, states_ptr,
    m, n, k,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < m

    # Load decay: (m, k)
    decay = tl.load(decay_ptr + offsets[:, None] * k + tl.arange(0, k)[None, :], mask=mask, other=0.0)
    
    # Load states: (m, k)
    states = tl.load(states_ptr + offsets[:, None] * k + tl.arange(0, k)[None, :], mask=mask, other=0.0)
    
    # Compute: decay @ states
    result = tl.dot(decay, states)
    
    tl.store(result + offsets[:, None] * k + tl.arange(0, k)[None, :], result, mask=mask)


@triton.jit
def einsum_output_kernel(
    c_ptr, states_ptr, decay_ptr,
    m, n, k,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < m

    # Load C: (m, k)
    c = tl.load(c_ptr + offsets[:, None] * k + tl.arange(0, k)[None, :], mask=mask, other=0.0)
    
    # Load states: (m, k)
    states = tl.load(states_ptr + offsets[:, None] * k + tl.arange(0, k)[None, :], mask=mask, other=0.0)
    
    # Load decay: (m, k)
    decay = tl.load(decay_ptr + offsets[:, None] * k + tl.arange(0, k)[None, :], mask=mask, other=0.0)
    
    # Compute: C @ states @ decay
    c_states = tl.dot(c, states)
    result = tl.dot(c_states, decay)
    
    tl.store(result + offsets[:, None] * k + tl.arange(0, k)[None, :], result, mask=mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()
    
    m, k = a.shape
    k, n = b.shape
    
    BLOCK_SIZE = 128
    grid = lambda meta: ((m + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    out = torch.empty(m, n, device=a.device, dtype=a.dtype)
    matmul_kernel[grid](a.data_ptr(), b.data_ptr(), out.data_ptr(), m, n, k, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_exp_cumsum(x: torch.Tensor):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    y = torch.empty_like(x)
    exp_cumsum_kernel[grid](x.data_ptr(), y.data_ptr(), n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y


def triton_einsum_diag(c: torch.Tensor, b: torch.Tensor, l: torch.Tensor, x: torch.Tensor):
    m, k = c.shape
    k, n = b.shape
    BLOCK_SIZE = 128
    grid = lambda meta: ((m + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    out = torch.empty(m, n, device=c.device, dtype=c.dtype)
    einsum_diag_kernel[grid](c.data_ptr(), b.data_ptr(), l.data_ptr(), x.data_ptr(), m, n, k, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_einsum_state(b: torch.Tensor, decay: torch.Tensor, x: torch.Tensor):
    m, k = b.shape
    k, n = x.shape
    BLOCK_SIZE = 128
    grid = lambda meta: ((m + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    out = torch.empty(m, n, device=b.device, dtype=b.dtype)
    einsum_state_kernel[grid](b.data_ptr(), decay.data_ptr(), x.data_ptr(), m, n, k, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_einsum_inter_chunk(decay: torch.Tensor, states: torch.Tensor):
    m, k = decay.shape
    k, n = states.shape
    BLOCK_SIZE = 128
    grid = lambda meta: ((m + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    out = torch.empty(m, n, device=decay.device, dtype=decay.dtype)
    einsum_inter_chunk_kernel[grid](decay.data_ptr(), states.data_ptr(), m, n, k, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_einsum_output(c: torch.Tensor, states: torch.Tensor, decay: torch.Tensor):
    m, k = c.shape
    k, n = states.shape
    BLOCK_SIZE = 128
    grid = lambda meta: ((m + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    out = torch.empty(m, n, device=c.device, dtype=c.dtype)
    einsum_output_kernel[grid](c.data_ptr(), states.data_ptr(), decay.data_ptr(), m, n, k, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super(ModelNew, self).__init__()
        
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        # Initialize parameters
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, device='cuda', dtype=torch.float32))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state, device='cuda', dtype=torch.float32))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state, device='cuda', dtype=torch.float32))
        
    def segsum(self, x):
        """Naive segment sum calculation using Triton kernel for performance."""
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum
    
    def forward(self, X, initial_states=None):
        """
        Forward pass implementing the SSD operation with custom Triton kernels.
        
        :param X: Input tensor of shape (batch, length, n_heads, d_head)
        :param initial_states: Optional initial states
        :return: Output tensor Y and final state
        """
        # Rearrange into blocks/chunks
        X_blocks = rearrange(X, "b (c l) h d -> b c l h d", l=self.block_len)
        A_blocks = rearrange(self.A, "b (c l) h -> b h c l", l=self.block_len)
        B_blocks = rearrange(self.B, "b (c l) h d -> b h c l d", l=self.block_len)
        C_blocks = rearrange(self.C, "b (c l) h d -> b h c l d", l=self.block_len)
        
        # Compute A_cumsum
        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        
        # 1. Compute diagonal block outputs
        L = triton_exp_cumsum(self.segsum(A_blocks))
        Y_diag = triton_einsum_diag(C_blocks, B_blocks, L, X_blocks)
        
        # 2. Compute intra-chunk states
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = triton_einsum_state(B_blocks, decay_states, X_blocks)
        
        # 3. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        
        decay_chunk = triton_exp_cumsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))
        new_states = triton_einsum_inter_chunk(decay_chunk, states)
        states = new_states[:, :-1]
        
        # 4. Compute state-to-output conversion
        state_decay_out = triton_exp_cumsum(A_cumsum)
        Y_off = triton_einsum_output(C_blocks, states, state_decay_out)
        
        # Combine diagonal and off-diagonal terms
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        
        return Y