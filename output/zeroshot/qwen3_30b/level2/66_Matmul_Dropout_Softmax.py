import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_softmax_dropout_kernel(
    x_ptr,  # Input matrix (batch, in_features)
    w_ptr,  # Weight matrix (in_features, out_features)
    out_ptr,  # Output matrix (batch, out_features)
    dropout_mask_ptr,  # Dropout mask (batch, out_features)
    batch_size: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    dropout_p: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION_SCALE: tl.constexpr,
):
    # Shared memory for the matmul accumulation (tiling)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(batch_size, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(out_features, BLOCK_SIZE_N)

    # Calculate the block indices
    block_m = pid // num_pid_n
    block_n = pid % num_pid_n

    # Calculate offsets for this block
    offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize the accumulator for the matmul
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension (in_features) in tiles
    for k in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        # Load input: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        x_ptrs = x_ptr + (offs_m[:, None] * in_features + offs_k[None, :])
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < batch_size) & (offs_k[None, :] < in_features), other=0.0)

        # Load weights: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        w_ptrs = w_ptr + (offs_k[:, None] * out_features + offs_n[None, :])
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < in_features) & (offs_n[None, :] < out_features), other=0.0)

        # Perform matmul: (BLOCK_SIZE_M, BLOCK_SIZE_K) @ (BLOCK_SIZE_K, BLOCK_SIZE_N)
        accumulator += tl.dot(x, w)

    # Apply activation scaling if needed (optional)
    accumulator = accumulator * ACTIVATION_SCALE

    # Perform online softmax over dim=1 (features) for each batch
    # Use the maximum for numerical stability
    row_max = tl.max(accumulator, axis=1)
    exp_x = tl.exp(accumulator - row_max[:, None])

    # Normalize by sum
    row_sum = tl.sum(exp_x, axis=1)
    softmax = exp_x / row_sum[:, None]

    # Apply dropout
    # Create mask from dropout_ptr (assumed to be a boolean tensor)
    dropout_mask = tl.load(dropout_mask_ptr + offs_m[:, None] * out_features + offs_n[None, :],
                           mask=(offs_m[:, None] < batch_size) & (offs_n[None, :] < out_features),
                           other=1.0)  # 1.0 means keep, 0.0 means drop
    dropout_mask = (dropout_mask > 0.5).to(tl.float32)  # Convert to float32

    # Apply dropout: multiply by mask (and scale by 1/(1-dropout_p) if desired)
    dropout_scale = 1.0 / (1.0 - dropout_p)
    output = softmax * dropout_mask * dropout_scale

    # Store result
    out_ptrs = out_ptr + (offs_m[:, None] * out_features + offs_n[None, :])
    tl.store(out_ptrs, output, mask=(offs_m[:, None] < batch_size) & (offs_n[None, :] < out_features))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
    ],
    key=['batch_size', 'in_features', 'out_features'],
)
def triton_matmul_softmax_dropout(x, w, dropout_mask, dropout_p, ACTIVATION_SCALE=1.0):
    # Ensure inputs are contiguous and on GPU
    assert x.is_cuda and w.is_cuda and dropout_mask.is_cuda, "All inputs must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    dropout_mask = dropout_mask.contiguous()

    # Get dimensions
    batch_size, in_features = x.shape
    out_features = w.shape[1]

    # Define block sizes (chosen to fit A100 tensor cores and shared memory)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64

    # Grid setup: one block per tile
    grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_SIZE_M']) * triton.cdiv(out_features, meta['BLOCK_SIZE_N']),)

    # Launch kernel
    matmul_softmax_dropout_kernel[grid](
        x, w, x, dropout_mask, batch_size, in_features, out_features, dropout_p,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ACTIVATION_SCALE
    )

    return x


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.float16).cuda())
        self.dropout_p = dropout_p
        self.register_buffer('dropout_mask', None)

    def forward(self, x):
        # Create or re-use dropout mask for the forward pass
        if self.dropout_mask is None or self.dropout_mask.shape != x.shape:
            # Create random dropout mask
            self.dropout_mask = torch.bernoulli(torch.full(x.shape, 1.0 - self.dropout_p, device=x.device)).to(torch.float32)

        # Use Triton kernel: fused matmul + softmax + dropout
        return triton_matmul_softmax_dropout(x, self.weight, self.dropout_mask, self.dropout_p, ACTIVATION_SCALE=1.0)