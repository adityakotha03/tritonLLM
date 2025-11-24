import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_fused_gelu_avgpool_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr, pool_out_ptr,
    batch_stride, in_features, out_features, pool_kernel_size,
    scale_factor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < batch_stride
    mask_n = n_offsets < out_features

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, in_features, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < in_features
        x = tl.load(
            x_ptr + m_offsets[:, None] * in_features + k_offsets[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )
        w = tl.load(
            weight_ptr + n_offsets[:, None] + k_offsets[None, :] * out_features,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0
        )
        acc += tl.dot(x, w, out_dtype=tl.float32)

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + n_offsets, mask=mask_n, other=0.0)
        acc = acc + bias[None, :]

    out = acc

    # Apply GELU using the approximate formula: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    x3 = out * out * out
    inner = sqrt_2_over_pi * (out + 0.044715 * x3)
    tanh_inner = tl.tanh(inner)
    gelu_out = out * 0.5 * (1.0 + tanh_inner)

    # Scale
    gelu_out = gelu_out * scale_factor

    # Store the full GELU+scale output (we'll do max reduction later)
    tl.store(
        out_ptr + m_offsets[:, None] * out_features + n_offsets[None, :],
        gelu_out,
        mask=mask_m[:, None] & mask_n[None, :]
    )

    # Perform avg pool: reduce over groups of `pool_kernel_size` in the output dim
    # We assume out_features % pool_kernel_size == 0
    pooled_size = out_features // pool_kernel_size
    pool_n_offsets = n_offsets // pool_kernel_size
    within_pool = n_offsets % pool_kernel_size

    valid_pool = pool_n_offsets < pooled_size
    pool_mask = mask_n & valid_pool

    # Sum within each pool group
    pool_idx = pool_n_offsets
    pool_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    pool_count = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # We accumulate only if within valid pool group
    pool_acc = tl.where(pool_mask[None, :], gelu_out, 0.0)
    pool_count = tl.where(pool_mask[None, :], 1.0, 0.0)

    # Reduce within block over the N dimension grouped by pool_idx
    # We do elementwise sum per group
    for p in range(pooled_size):
        mask_p = (pool_idx == p) & pool_mask
        sum_p = tl.sum(tl.where(mask_p[None, :], pool_acc, 0.0), axis=1)
        cnt_p = tl.sum(tl.where(mask_p[None, :], pool_count, 0.0), axis=1)
        avg_p = tl.where(cnt_p > 0, sum_p / cnt_p, 0.0)
        tl.store(pool_out_ptr + m_offsets * pooled_size + p, avg_p, mask=mask_m)


@triton.jit
def max_reduction_kernel(
    x_ptr, output_ptr, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    row_start = pid_b * BLOCK_SIZE
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    max_val = tl.max(row, axis=0)
    first_thread = tl.program_id(0) == 0
    if first_thread:
        tl.store(output_ptr, max_val)


class ModelNew(nn.Module):
    """
    Optimized version of Model using fused Triton kernels for matmul+gelu+avgpool and max reduction.
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        self.pooled_size = out_features // pool_kernel_size

        assert out_features % pool_kernel_size == 0, "out_features must be divisible by pool_kernel_size"

        # Initialize Linear layer parameters manually for kernel access
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Fused computation: Matmul -> GELU -> Scale -> AvgPool1d -> Max Reduction
        """
        batch_size, in_features = x.shape
        out_features = self.out_features
        pooled_size = self.pooled_size

        # Allocate output for full GELU+scale (batch_size, out_features)
        fused_output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)

        # Allocate output for pooled result (batch_size, pooled_size)
        pooled_output = torch.empty((batch_size, pooled_size), device=x.device, dtype=x.dtype)

        # Launch fused matmul + gelu + scale + avg_pool kernel
        def grid_matmul(M, N):
            return (
                triton.cdiv(batch_size, M),
                triton.cdiv(out_features, N)
            )

        # Use autotuning heuristics
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        matmul_fused_gelu_avgpool_kernel[grid_matmul(BLOCK_M, BLOCK_N)](
            x_ptr=x,
            weight_ptr=self.weight,
            bias_ptr=self.bias,
            out_ptr=fused_output,
            pool_out_ptr=pooled_output,
            batch_stride=batch_size,
            in_features=in_features,
            out_features=out_features,
            pool_kernel_size=self.pool_kernel_size,
            scale_factor=self.scale_factor,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K
        )

        # Now do max reduction over the pooled output: (batch_size, pooled_size) -> (batch_size,)
        max_output = torch.empty((batch_size,), device=x.device, dtype=x.dtype)

        def grid_max(BLOCK_SIZE):
            return (batch_size,)

        max_reduction_kernel[grid_max(1024)](
            x_ptr=pooled_output,
            output_ptr=max_output,
            n_cols=pooled_size,
            BLOCK_SIZE=1024
        )

        return max_output