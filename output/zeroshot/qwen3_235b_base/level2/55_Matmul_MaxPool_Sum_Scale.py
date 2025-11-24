import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_maxpool_sum_scale_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_features, out_features, kernel_size,
    scale_factor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 2D block ID
    pid_batch = tl.program_id(0)
    pid_out = tl.program_id(1)

    # Compute pointers for this block
    offs_m = pid_batch + tl.arange(0, BLOCK_M)
    offs_n = pid_out * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Matmul: x[batch, :in_features] @ weight[:in_features, :out_features]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, in_features, BLOCK_K):
        x_ptrs = x_ptr + offs_m[:, None] * in_features + (offs_k + k)[None, :]
        weight_ptrs = weight_ptr + (offs_k + k)[:, None] * out_features + offs_n[None, :]
        
        x_mask = (offs_m < batch_size)[:, None] & ((offs_k + k) < in_features)[None, :]
        w_mask = ((offs_k + k) < in_features)[:, None] & (offs_n < out_features)[None, :]

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(weight_ptrs, mask=w_mask, other=0.0)

        accumulator = tl.dot(x, w, acc=accumulator)

    # Add bias
    if bias_ptr is not None:
        bias_ptrs = bias_ptr + offs_n
        bias_mask = offs_n < out_features
        bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
        accumulator = accumulator + bias[None, :]

    # Apply ReLU-like non-linearity (not needed here, but placeholder if desired)
    accumulator = accumulator

    # Max Pooling: 1D max pooling with kernel_size along feature dim
    # We do it per row: reduce each row of `accumulator` with max-pool of size `kernel_size`
    pooled = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, BLOCK_N, kernel_size):
        start_idx = i
        end_idx = tl.minimum(i + kernel_size, BLOCK_N)
        current_block = accumulator[:, start_idx:end_idx]
        pooled_row = tl.max(current_block, axis=1)
        # Only write if within valid output bounds
        if start_idx < BLOCK_N:
            pooled[:, start_idx // kernel_size] = pooled_row

    # Now sum over the pooled features (dim=1), scale and write to output
    sum_val = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for i in range(0, (out_features + kernel_size - 1) // kernel_size):
        sum_val += pooled[:, i]

    # Scale by scale_factor
    sum_val *= scale_factor

    # Write output
    output_ptrs = output_ptr + offs_m
    output_mask = offs_m < batch_size
    tl.store(output_ptrs, sum_val, mask=output_mask)


class ModelNew(nn.Module):
    """
    Optimized model using fused Triton kernel for matmul + maxpool1d + sum + scale.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        # Linear layer parameters (we'll access weight and bias directly)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Fused forward pass: matmul -> max_pool1d -> sum(dim=1) -> scale
        """
        assert x.is_cuda, "Input must be on GPU"

        batch_size = x.size(0)
        # Output is a scalar per batch element
        output = torch.empty((batch_size,), dtype=x.dtype, device=x.device)

        # Launch kernel
        # Heuristic block sizes
        BLOCK_M = 1
        BLOCK_N = 1024
        BLOCK_K = 64

        # Number of output blocks for out_features
        grid_m = batch_size
        grid_n = triton.cdiv(self.out_features, BLOCK_N)

        grid = (grid_m, grid_n)

        matmul_maxpool_sum_scale_kernel[grid](
            x, self.weight, self.bias, output,
            batch_size, self.in_features, self.out_features, self.kernel_size,
            self.scale_factor,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        return output