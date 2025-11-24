import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def sdpa_kernel(
    Q_ptr, K_ptr, V_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    embed_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes one block of sequence length
    bid = tl.program_id(0)  # batch index
    head_id = tl.program_id(1)  # head index

    # Compute the global indices
    q_offset = bid * seq_len + tl.arange(0, seq_len)
    k_offset = bid * seq_len + tl.arange(0, seq_len)
    v_offset = bid * seq_len + tl.arange(0, seq_len)

    # Load Q, K, V for the current batch and head
    q = tl.load(Q_ptr + bid * num_heads * seq_len * embed_dim + head_id * seq_len * embed_dim + q_offset * embed_dim, mask=q_offset < seq_len, other=0.0)
    k = tl.load(K_ptr + bid * num_heads * seq_len * embed_dim + head_id * seq_len * embed_dim + k_offset * embed_dim, mask=k_offset < seq_len, other=0.0)
    v = tl.load(V_ptr + bid * num_heads * seq_len * embed_dim + head_id * seq_len * embed_dim + v_offset * embed_dim, mask=v_offset < seq_len, other=0.0)

    # Compute attention scores (Q @ K^T)
    # We use a block-wise computation to avoid full matrix multiplication
    # Each block of size BLOCK_SIZE computes dot products over a segment of the sequence
    q = q.reshape(seq_len, embed_dim)
    k = k.reshape(seq_len, embed_dim)

    # Compute attention scores (Q @ K^T) using a block-wise dot product
    # We compute scores in a tiled fashion
    scores = tl.zeros((seq_len, seq_len), dtype=tl.float16)
    for i in range(0, seq_len, BLOCK_SIZE):
        i_start = i
        i_end = min(i + BLOCK_SIZE, seq_len)
        # Load a block of Q
        q_block = q[i_start:i_end]
        # Compute dot product with K
        k_t = k.transpose(0, 1)
        scores_i = tl.dot(q_block, k_t)
        scores_i = scores_i.to(tl.float16)
        scores[i_start:i_end] = scores_i

    # Softmax over the sequence dimension
    # We compute the softmax in a block-wise fashion
    # First, compute the max for numerical stability
    max_scores = tl.max(scores, axis=1, keepdim=True)
    exp_scores = tl.exp(scores - max_scores)
    sum_scores = tl.sum(exp_scores, axis=1, keepdim=True)
    softmax_scores = exp_scores / sum_scores

    # Compute output: softmax @ V
    v = v.reshape(seq_len, embed_dim)
    out = tl.zeros((seq_len, embed_dim), dtype=tl.float16)
    for i in range(0, seq_len, BLOCK_SIZE):
        i_start = i
        i_end = min(i + BLOCK_SIZE, seq_len)
        # Load a block of V
        v_block = v[i_start:i_end]
        # Compute dot product with softmax scores
        out_block = tl.dot(softmax_scores[i_start:i_end], v_block)
        out[i_start:i_end] = out_block

    # Store output
    out_ptr_base = bid * num_heads * seq_len * embed_dim + head_id * seq_len * embed_dim
    out_offset = tl.arange(0, seq_len)
    tl.store(out_ptr + out_ptr_base + out_offset * embed_dim, out, mask=out_offset < seq_len)


def triton_sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    """
    Custom Triton kernel for scaled dot-product attention.
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "All tensors must be on CUDA."
    assert Q.dtype == torch.float16 and K.dtype == torch.float16 and V.dtype == torch.float16, "Input tensors must be float16."

    batch_size = Q.shape[0]
    num_heads = Q.shape[1]
    seq_len = Q.shape[2]
    embed_dim = Q.shape[3]

    # Ensure tensors are contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    # Output tensor
    out = torch.empty_like(Q)

    # Define block size (power of 2, optimized for Ampere)
    BLOCK_SIZE = 128

    # Grid definition: (batch_size, num_heads)
    grid = lambda meta: (
        (batch_size * num_heads + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    # Launch kernel
    sdpa_kernel[grid](Q, K, V, out, batch_size, num_heads, seq_len, embed_dim, BLOCK_SIZE=BLOCK_SIZE)

    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return triton_sdpa(Q, K, V)