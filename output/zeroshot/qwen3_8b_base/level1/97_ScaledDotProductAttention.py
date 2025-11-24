import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scaled_dot_product_attention_kernel(
    Q_ptr, K_ptr, V_ptr, out_ptr,
    batch_size, num_heads, sequence_length, embedding_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance handles a block of data
    pid = tl.program_id(0)
    # Each block handles a specific head and position
    head_idx = pid % num_heads
    pos_idx = pid // num_heads

    # Compute the offset for the current head and position
    offset = head_idx * sequence_length * embedding_dim + pos_idx * embedding_dim
    # Compute the block start and end indices
    block_start = tl.arange(0, BLOCK_SIZE)
    block_offsets = block_start + offset
    # Mask to ensure we don't go out of bounds
    mask = block_offsets < (batch_size * num_heads * sequence_length * embedding_dim)

    # Load Q, K, V
    Q = tl.load(Q_ptr + block_offsets, mask=mask, other=0.0)
    K = tl.load(K_ptr + block_offsets, mask=mask, other=0.0)
    V = tl.load(V_ptr + block_offsets, mask=mask, other=0.0)

    # Compute attention weights
    attn_weights = tl.dot(Q, K)
    # Apply softmax
    attn_weights = tl.softmax(attn_weights, axis=-1)
    # Compute output
    out = tl.dot(attn_weights, V)

    # Store the result
    tl.store(out_ptr + block_offsets, out, mask=mask)


def triton_scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    """
    This function wraps the Triton kernel call for scaled dot product attention.
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Tensors must be on CUDA."
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    # Prepare output tensor
    out = torch.empty_like(Q)

    # Parameters
    batch_size = Q.size(0)
    num_heads = Q.size(1)
    sequence_length = Q.size(2)
    embedding_dim = Q.size(3)

    # Choose block size (power of 2)
    BLOCK_SIZE = 128

    # Determine the number of blocks needed
    num_blocks = (batch_size * num_heads * sequence_length * embedding_dim + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    grid = (num_blocks,)
    scaled_dot_product_attention_kernel[grid](
        Q, K, V, out,
        batch_size, num_heads, sequence_length, embedding_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return triton_scaled_dot_product_attention(Q, K, V)