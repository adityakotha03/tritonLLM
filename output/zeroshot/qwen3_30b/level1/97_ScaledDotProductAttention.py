import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scaled_dot_product_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    Q_row_stride, K_row_stride, V_row_stride, Out_row_stride,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
    BLOCK_SIZE_HEAD: tl.constexpr,
    BLOCK_SIZE_BLOCK: tl.constexpr,
):
    # Define thread block indices
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_block = tl.program_id(2)

    # Calculate offsets for Q, K, V, and output
    # Each block processes a portion of the sequence length (query and key)
    q_block_offset = pid_block * BLOCK_SIZE_BLOCK
    k_block_offset = q_block_offset
    v_block_offset = q_block_offset

    # Ensure we don't go out of bounds
    q_mask = q_block_offset + tl.arange(0, BLOCK_SIZE_BLOCK) < seq_len
    k_mask = k_block_offset + tl.arange(0, BLOCK_SIZE_BLOCK) < seq_len
    v_mask = v_block_offset + tl.arange(0, BLOCK_SIZE_BLOCK) < seq_len

    # Load Q, K, V tiles with masking
    q_ptrs = Q_ptr + (
        pid_batch * Q_row_stride +
        pid_head * (seq_len * head_dim) +
        (q_block_offset + tl.arange(0, BLOCK_SIZE_BLOCK)) * head_dim
    )
    k_ptrs = K_ptr + (
        pid_batch * K_row_stride +
        pid_head * (seq_len * head_dim) +
        (k_block_offset + tl.arange(0, BLOCK_SIZE_BLOCK)) * head_dim
    )
    v_ptrs = V_ptr + (
        pid_batch * V_row_stride +
        pid_head * (seq_len * head_dim) +
        (v_block_offset + tl.arange(0, BLOCK_SIZE_BLOCK)) * head_dim
    )

    # Load Q, K, V tiles
    q = tl.load(q_ptrs + tl.arange(0, head_dim)[:, None], mask=q_mask[None, :], other=0.0)
    k = tl.load(k_ptrs + tl.arange(0, head_dim)[None, :], mask=k_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + tl.arange(0, head_dim)[None, :], mask=v_mask[None, :], other=0.0)

    # Perform matmul Q @ K.T
    # Use block-wise accumulation to reduce register pressure
    qk = tl.dot(q, k.T)  # [BLOCK_SIZE_BLOCK, BLOCK_SIZE_BLOCK]

    # Scale by sqrt(head_dim)
    qk_scale = qk * (1.0 / tl.sqrt(tl.float32(head_dim)))

    # Apply causal mask if needed (for autoregressive generation)
    # Here we assume no causal masking; can be added if needed
    # For now, just apply softmax over keys

    # Apply softmax (online: use max and exp to avoid overflow)
    qk_max = tl.max(qk, axis=1, keep_dims=True)
    qk_exp = tl.exp(qk - qk_max)
    qk_sum = tl.sum(qk_exp, axis=1, keep_dims=True)
    qk_norm = qk_exp / qk_sum

    # Multiply by V: [BLOCK_SIZE_BLOCK, BLOCK_SIZE_BLOCK] @ [BLOCK_SIZE_BLOCK, head_dim]
    out = tl.dot(qk_norm, v)  # [BLOCK_SIZE_BLOCK, head_dim]

    # Store output
    out_ptrs = Out_ptr + (
        pid_batch * Out_row_stride +
        pid_head * (seq_len * head_dim) +
        (q_block_offset + tl.arange(0, BLOCK_SIZE_BLOCK)) * head_dim
    )
    tl.store(out_ptrs + tl.arange(0, head_dim)[None, :], out, mask=q_mask[None, :])


# Define custom attention implementation
def triton_scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Tensors must be on CUDA"
    assert Q.dtype == torch.float16, "Only FP16 supported for this kernel"
    assert Q.shape == K.shape == V.shape, "Q, K, V must have same shape"

    batch_size, num_heads, seq_len, head_dim = Q.shape

    # Ensure contiguous tensors
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    # Output tensor
    Out = torch.empty_like(Q)

    # Calculate strides
    Q_row_stride = Q.stride(0) * Q.shape[0] + Q.stride(1) * Q.shape[1] + Q.stride(2) * Q.shape[2]
    K_row_stride = K.stride(0) * K.shape[0] + K.stride(1) * K.shape[1] + K.stride(2) * K.shape[2]
    V_row_stride = V.stride(0) * V.shape[0] + V.stride(1) * V.shape[1] + V.stride(2) * V.shape[2]
    Out_row_stride = Out.stride(0) * Out.shape[0] + Out.stride(1) * Out.shape[1] + Out.stride(2) * Out.shape[2]

    # Configure block sizes
    BLOCK_SIZE_BLOCK = 128  # tile size for sequence dimension
    BLOCK_SIZE_HEAD = 128  # tile size for head dimension
    BLOCK_SIZE_Q = 128
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_V = 128

    # Grid configuration: (batch, head, block)
    grid = lambda meta: (
        batch_size,
        num_heads,
        (seq_len + meta["BLOCK_SIZE_BLOCK"] - 1) // meta["BLOCK_SIZE_BLOCK"]
    )

    # Launch kernel
    scaled_dot_product_attention_kernel[grid](
        Q, K, V, Out,
        Q_row_stride, K_row_stride, V_row_stride, Out_row_stride,
        batch_size, num_heads, seq_len, head_dim,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        BLOCK_SIZE_HEAD=BLOCK_SIZE_HEAD,
        BLOCK_SIZE_BLOCK=BLOCK_SIZE_BLOCK
    )
    return Out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return triton_scaled_dot_product_attention(Q, K, V)