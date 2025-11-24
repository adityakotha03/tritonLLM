import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to first input matrix (B, H, S)
    b_ptr,  # Pointer to second input matrix (S, H)
    out_ptr,  # Pointer to output matrix (B, H)
    B, H, S,  # Dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Compute the block offset
    block_idx = pid * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_idx + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < B * H
    # Load input values
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    # Perform the elementwise addition
    out = a + b
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    # Prepare output tensor
    B, H, S = a.shape
    out = torch.empty((B, H), device=a.device, dtype=a.dtype)

    # Number of elements in the tensor
    n_elements = B * H
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    matmul_kernel[grid](a, b, out, B, H, S, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def softmax_kernel(
    x_ptr,  # Pointer to input matrix (B, H)
    out_ptr,  # Pointer to output matrix (B, H)
    B, H,  # Dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Compute the block offset
    block_idx = pid * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_idx + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < B * H
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute softmax
    max_val = tl.max(x, axis=0)
    x -= max_val
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    out = exp_x / sum_exp
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_softmax(x: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    B, H = x.shape
    out = torch.empty((B, H), device=x.device, dtype=x.dtype)

    # Number of elements in the tensor
    n_elements = B * H
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    softmax_kernel[grid](x, out, B, H, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Attention Block using Multihead Self-Attention.
        :param embed_dim: Embedding dimension (the number of channels)
        :param num_heads: Number of attention heads
        """
        super(ModelNew, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def forward(self, x):
        """
        Forward pass of the AttentionBlock.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of the same shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(2, 0, 1)  # (seq_len, batch_size, embed_dim)
        # Custom matmul and softmax
        seq_len, batch_size, embed_dim = x.shape
        # Reshape to (seq_len, batch_size, num_heads, head_dim)
        x = x.reshape(seq_len, batch_size, self.num_heads, self.head_dim)
        # Transpose to (num_heads, seq_len, batch_size, head_dim)
        x = x.transpose(0, 1)
        # Compute attention weights
        q = x
        k = x
        v = x
        # Custom matmul for QK^T
        attn_weights = triton_matmul(q, k)
        # Apply softmax
        attn_weights = triton_softmax(attn_weights)
        # Compute attention output
        attn_output = triton_matmul(attn_weights, v)
        # Transpose back to (seq_len, batch_size, embed_dim)
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim)
        # Add and norm
        x = self.norm(attn_output + x)  # (seq_len, batch_size, embed_dim)
        x = x.permute(1, 2, 0).view(B, C, H, W)
        return x