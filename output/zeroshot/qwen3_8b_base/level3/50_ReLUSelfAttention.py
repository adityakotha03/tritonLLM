import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def gelu_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute GELU
    x_cubed = x * x * x
    x_pow3 = x_cubed * 0.044715
    x_term = x + x_pow3
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    tanh_term = tl.tanh(sqrt_2_over_pi * x_term)
    out = 0.5 * x * (1.0 + tanh_term)
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(x: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def matmul_relu_kernel(
    q_ptr,  # Pointer to query
    k_ptr,  # Pointer to key
    v_ptr,  # Pointer to value
    out_ptr,  # Pointer to output
    B,  # Batch size
    T,  # Sequence length
    H,  # Number of heads
    HS,  # Head size
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Compute the block index in the sequence dimension
    seq_idx = pid
    # Compute the block index in the head dimension
    head_idx = tl.program_id(1)
    # Compute the block index in the batch dimension
    batch_idx = tl.program_id(2)

    # Compute the offset for the current block
    offset = batch_idx * B * T * H * HS + head_idx * T * HS + seq_idx * HS
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = offset + tl.arange(0, BLOCK_SIZE)

    # Load q, k, v
    q = tl.load(q_ptr + offsets, mask=offsets < B * T * H * HS, other=0.0)
    k = tl.load(k_ptr + offsets, mask=offsets < B * T * H * HS, other=0.0)
    v = tl.load(v_ptr + offsets, mask=offsets < B * T * H * HS, other=0.0)

    # Compute attention scores
    qk = q @ k
    qk = qk * (1.0 / math.sqrt(HS))
    # Apply mask
    mask = (seq_idx < T) & (seq_idx >= 0)
    qk = tl.where(mask, qk, float('-inf'))
    # Apply ReLU
    qk = tl.maximum(qk, 0.0)

    # Compute attention weights
    attn_weights = tl.softmax(qk, axis=-1)

    # Compute attention output
    out = attn_weights @ v

    # Store the result
    tl.store(out_ptr + offsets, out, mask=offsets < B * T * H * HS)


def triton_matmul_relu(q, k, v, B, T, H, HS):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Tensors must be on CUDA."
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Prepare output tensor
    out = torch.empty_like(q)

    # Compute the grid
    num_blocks_seq = (T + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks_head = H
    num_blocks_batch = B
    grid = (num_blocks_seq, num_blocks_head, num_blocks_batch)

    # Launch the Triton kernel
    matmul_relu_kernel[grid](q, k, v, out, B, T, H, HS, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    A multi-head masked self-attention layer with a projection at the end that uses ReLU instead of Softmax.
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
        self.HS = n_embd // n_head
        self.BLOCK_SIZE = 128  # Tunable parameter for block size

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.HS).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.HS).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.HS).transpose(1, 2) # (B, nh, T, hs)

        # Compute attention using Triton kernel
        y = triton_matmul_relu(q, k, v, B, T, self.n_head, self.HS)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        return self.c_proj(y)