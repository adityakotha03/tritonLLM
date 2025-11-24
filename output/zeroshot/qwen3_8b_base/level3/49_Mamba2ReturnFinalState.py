import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange

@triton.jit
def segsum_kernel(
    A_ptr,  # Pointer to A
    out_ptr,  # Pointer to output
    n_blocks,  # Number of blocks
    n_heads,  # Number of heads
    d_state,  # Dimension of state
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_blocks

    # Load A
    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    # Compute cumulative sum
    cumsum = tl.cumsum(A, axis=0)
    # Compute segment sum
    cumsum_upper = tl.expand_dims(cumsum, axis=1)
    cumsum_lower = tl.expand_dims(cumsum, axis=2)
    segsum = cumsum_upper - cumsum_lower
    # Apply mask
    segsum = segsum * tl.where(tl.tril(tl.ones((n_blocks, n_blocks), dtype=tl.int32), diagonal=0), 1.0, 0.0)
    # Store result
    tl.store(out_ptr + offsets, segsum, mask=mask)

def triton_segsum(A):
    """
    Custom Triton kernel for segment sum operation.
    """
    assert A.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    n_blocks = A.size(0)
    n_heads = A.size(1)
    d_state = A.size(2)

    out = torch.empty((n_blocks, n_heads, n_blocks, d_state), dtype=A.dtype, device=A.device)

    # Choose block size
    BLOCK_SIZE = 128

    # Determine grid size
    grid = lambda meta: ((n_blocks + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the kernel
    segsum_kernel[grid](A, out, n_blocks, n_heads, d_state, BLOCK_SIZE=BLOCK_SIZE)
    return out

@triton.jit
def einsum_kernel(
    C_ptr,  # Pointer to C
    B_ptr,  # Pointer to B
    L_ptr,  # Pointer to L
    X_ptr,  # Pointer to X
    out_ptr,  # Pointer to output
    n_blocks,  # Number of blocks
    n_heads,  # Number of heads
    d_state,  # Dimension of state
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_blocks

    # Load C, B, L, X
    C = tl.load(C_ptr + offsets, mask=mask, other=0.0)
    B = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    L = tl.load(L_ptr + offsets, mask=mask, other=0.0)
    X = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # Compute einsum
    out = tl.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_einsum(C, B, L, X):
    """
    Custom Triton kernel for einsum operation.
    """
    assert C.is_cuda and B.is_cuda and L.is_cuda and X.is_cuda, "Tensors must be on CUDA."
    C = C.contiguous()
    B = B.contiguous()
    L = L.contiguous()
    X = X.contiguous()
    n_blocks = C.size(0)
    n_heads = C.size(1)
    d_state = C.size(2)

    out = torch.empty((n_blocks, n_heads, n_blocks, d_state), dtype=C.dtype, device=C.device)

    # Choose block size
    BLOCK_SIZE = 128

    # Determine grid size
    grid = lambda meta: ((n_blocks + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the kernel
    einsum_kernel[grid](C, B, L, X, out, n_blocks, n_heads, d_state, BLOCK_SIZE=BLOCK_SIZE)
    return out

@triton.jit
def einsum2_kernel(
    B_ptr,  # Pointer to B
    decay_states_ptr,  # Pointer to decay_states
    X_ptr,  # Pointer to X
    out_ptr,  # Pointer to output
    n_blocks,  # Number of blocks
    n_heads,  # Number of heads
    d_state,  # Dimension of state
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_blocks

    # Load B, decay_states, X
    B = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    decay_states = tl.load(decay_states_ptr + offsets, mask=mask, other=0.0)
    X = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # Compute einsum
    out = tl.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_einsum2(B, decay_states, X):
    """
    Custom Triton kernel for einsum operation.
    """
    assert B.is_cuda and decay_states.is_cuda and X.is_cuda, "Tensors must be on CUDA."
    B = B.contiguous()
    decay_states = decay_states.contiguous()
    X = X.contiguous()
    n_blocks = B.size(0)
    n_heads = B.size(1)
    d_state = B.size(2)

    out = torch.empty((n_blocks, n_heads, n_blocks, d_state), dtype=B.dtype, device=B.device)

    # Choose block size
    BLOCK_SIZE = 128

    # Determine grid size
    grid = lambda meta: ((n_blocks + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the kernel
    einsum2_kernel[grid](B, decay_states, X, out, n_blocks, n_heads, d_state, BLOCK_SIZE=BLOCK_SIZE)
    return out

@triton.jit
def einsum3_kernel(
    decay_chunk_ptr,  # Pointer to decay_chunk
    states_ptr,  # Pointer to states
    out_ptr,  # Pointer to output
    n_blocks,  # Number of blocks
    n_heads,  # Number of heads
    d_state,  # Dimension of state
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_blocks

    # Load decay_chunk, states
    decay_chunk = tl.load(decay_chunk_ptr + offsets, mask=mask, other=0.0)
    states = tl.load(states_ptr + offsets, mask=mask, other=0.0)

    # Compute einsum
    out = tl.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_einsum3(decay_chunk, states):
    """
    Custom Triton kernel for einsum operation.
    """
    assert decay_chunk.is_cuda and states.is_cuda, "Tensors must be on CUDA."
    decay_chunk = decay_chunk.contiguous()
    states = states.contiguous()
    n_blocks = decay_chunk.size(0)
    n_heads = decay_chunk.size(1)
    d_state = decay_chunk.size(2)

    out = torch.empty((n_blocks, n_heads, n_blocks, d_state), dtype=decay_chunk.dtype, device=decay_chunk.device)

    # Choose block size
    BLOCK_SIZE = 128

    # Determine grid size
    grid = lambda meta: ((n_blocks + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the kernel
    einsum3_kernel[grid](decay_chunk, states, out, n_blocks, n_heads, d_state, BLOCK_SIZE=BLOCK_SIZE)
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
        """Custom segment sum calculation using Triton kernel."""
        return triton_segsum(x)
    
    def forward(self, X, initial_states=None):
        # Rearrange into blocks/chunks
        X_blocks = rearrange(X, "b (c l) ... -> b c l ...", l=self.block_len)
        A_blocks = rearrange(self.A, "b (c l) h -> b h c l", l=self.block_len)
        B_blocks = rearrange(self.B, "b (c l) h s -> b h c l s", l=self.block_len)
        C_blocks = rearrange(self.C, "b (c l) h s -> b h c l s", l=self.block_len)
        
        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        
        # 1. Compute diagonal block outputs
        L = torch.exp(self.segsum(A_blocks))
        Y_diag = triton_einsum(C_blocks, B_blocks, L, X_blocks)
        
        # 2. Compute intra-chunk states
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = triton_einsum2(B_blocks, decay_states, X_blocks)
        
        # 3. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        
        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = triton_einsum3(decay_chunk, states)
        return new_states[:, -1]