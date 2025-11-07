import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_sum_max_mean_logsumexp_kernel(
    x_ptr,  # Input (batch_size, in_features)
    w_ptr,  # Weight (in_features, out_features)
    out_ptr,  # Output (batch_size, 1)
    batch_size,  # Number of batches
    in_features,  # Input feature dimension
    out_features,  # Output feature dimension
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for output feature dimension
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input feature dimension
):
    # Block index for batch dimension
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for the current block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Load weights (in_features, out_features) into shared memory
    w_ptrs = w_ptr + (offs_k[:, None] * out_features + offs_n[None, :])
    w = tl.load(w_ptrs, mask=(offs_k[:, None] < in_features) & (offs_n[None, :] < out_features), other=0.0)

    # Accumulator for dot product
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over chunks of input features
    for k in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        # Compute offsets for input data
        k_offset = k * BLOCK_SIZE_K
        offs_k = k_offset + tl.arange(0, BLOCK_SIZE_K)
        x_ptrs = x_ptr + (offs_m[:, None] * in_features + offs_k[None, :])
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < batch_size) & (offs_k[None, :] < in_features), other=0.0)

        # Perform matrix multiplication: acc += x @ w
        acc += tl.dot(x, w, allow_tf32=True)

    # Reduce sum over output features (dim=1) -> (batch_size, 1)
    # Use block-level reduction with shared memory for sum
    sum_out = tl.sum(acc, axis=1)  # (BLOCK_SIZE_M,)
    sum_out = tl.broadcast_to(sum_out[:, None], (BLOCK_SIZE_M, 1))

    # Max reduction across output features (dim=1) -> (batch_size, 1)
    # This requires reducing along n-axis, but we only have 1 output feature here
    # Since out_features is not reduced in kernel, we can just pass sum_out as max
    max_out = sum_out

    # Mean over output features (dim=1) -> (batch_size, 1)
    # Since we are reducing across a single dimension and have already reduced to 1, mean is same as max
    mean_out = max_out

    # LogSumExp: log(sum(exp(x))) over output features (dim=1)
    # We are collapsing along output features, so we have already reduced to one value per batch
    # So logsumexp of a single number is log(exp(x)) = x
    logsumexp1 = mean_out

    # LogSumExp again (same as above)
    logsumexp2 = logsumexp1

    # Store result
    out_ptrs = out_ptr + (offs_m[:, None] * 1 + tl.arange(0, 1))
    mask = offs_m[:, None] < batch_size
    tl.store(out_ptrs, logsumexp2, mask=mask)


def triton_linear_sum_max_mean_logsumexp(x, w):
    # Ensure inputs are contiguous and on GPU
    x = x.contiguous()
    w = w.contiguous()

    # Output shape: (batch_size, 1)
    out = torch.empty(x.shape[0], 1, dtype=x.dtype, device=x.device)

    # Determine block sizes for optimal performance on A100
    # Use 128x128 blocks for better occupancy and Tensor Core usage
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128

    # Number of blocks
    grid_m = triton.cdiv(x.shape[0], BLOCK_SIZE_M)
    grid_n = triton.cdiv(w.shape[1], BLOCK_SIZE_N)

    # Launch kernel
    linear_sum_max_mean_logsumexp_kernel[
        (grid_m, grid_n)
    ](
        x, w, out,
        x.shape[0], x.shape[1], w.shape[1],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        # Apply fused Triton kernel: linear + sum + max + mean + logsumexp + logsumexp
        return triton_linear_sum_max_mean_logsumexp(x, self.linear.weight)