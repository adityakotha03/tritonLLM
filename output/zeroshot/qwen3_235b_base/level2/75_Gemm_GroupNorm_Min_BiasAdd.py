import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_groupnorm_min_bias_kernel(
    x_ptr, weight_ptr, bias_ptr, gamma_ptr, beta_ptr,
    out_ptr, bias_out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    stride_bias_out_n,
    num_groups,
    eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # Re-order program ID for better L2 performance
    width = GROUP_SIZE_M
    group_id = pid // width
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(grid_m - first_pid_m, width)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = pid // grid_m

    # Offsets for matrix multiplication
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = weight_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    # Block-local storage to accumulate output
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load tiles of A and B
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        # Perform GEMM
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Convert to proper dtype
    c = accumulator.to(tl.float32)

    # Store GEMM output to shared buffer (used for GroupNorm)
    offs_out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + (offs_out_m[:, None] * stride_om + offs_out_n[None, :] * stride_on)
    out_mask = (offs_out_m[:, None] < M) & (offs_out_n[None, :] < N)
    tl.store(out_ptrs, c, mask=out_mask)

    # Synchronize all threads in the block
    tl.debug_barrier()

    # Now apply GroupNorm, Min reduction, and Bias in a fused fashion
    if pid_n == 0:  # Only one block per column (N-dim) does reduction and norm
        for start_m in range(0, M, BLOCK_M):
            offs_m_curr = start_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m_curr < M
            row_mask = mask_m[:, None] & (offs_n[None, :] < N)

            # Load GEMM output
            out_row_ptrs = out_ptr + (offs_m_curr[:, None] * stride_om + offs_n[None, :] * stride_on)
            out_row = tl.load(out_row_ptrs, mask=row_mask, other=0.0)

            # GroupNorm: reshape to (M, num_groups, N // num_groups)
            group_size_n = N // num_groups
            for g in range(num_groups):
                offs_g = g * group_size_n + tl.arange(0, group_size_n)
                mask_g = offs_g < N
                x_g = tl.load(out_ptr + (offs_m_curr[:, None] * stride_om + offs_g[None, :] * stride_on),
                              mask=mask_m[:, None] & mask_g[None, :], other=0.0)
                mean = tl.sum(x_g, axis=1) / group_size_n
                diff = x_g - mean[:, None]
                var = tl.sum(diff * diff, axis=1) / group_size_n
                inv_std = 1.0 / tl.sqrt(var + eps)

                # Normalize and apply affine
                gamma = tl.load(gamma_ptr + offs_g, mask=mask_g, other=1.0)
                beta = tl.load(beta_ptr + offs_g, mask=mask_g, other=0.0)
                x_hat = (x_g - mean[:, None]) * inv_std[:, None]
                y_g = x_hat * gamma[None, :] + beta[None, :]

                # Store normalized result back
                tl.store(out_row_ptrs + offs_g[None, :], y_g, mask=mask_m[:, None] & mask_g[None, :])

            # After GroupNorm, compute min over dim=1 (N-dim) -> (M, 1)
            # We do this per row block
            min_val = tl.full([BLOCK_M], value=tl.inf32, dtype=tl.float32)
            for n_idx in range(N):
                col_val = tl.load(out_ptr + (offs_m_curr * stride_om + n_idx * stride_on), mask=mask_m, other=tl.inf32)
                min_val = tl.minimum(min_val, col_val)

            # Add bias (broadcasted from (1, N, 1, 1) -> we treat bias as per feature)
            bias_val = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            min_val = min_val[:, None] + bias_val[None, :]

            # Store final result to bias_out_ptr: shape (M, N)
            out_bias_ptrs = bias_out_ptr + (offs_m_curr[:, None] * N + offs_n[None, :])
            tl.store(out_bias_ptrs, min_val, mask=mask_m[:, None] & (offs_n[None, :] < N))


def triton_matmul_groupnorm_min_bias(x, weight, bias, gamma, beta, bias_add):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert gamma.is_cuda and beta.is_cuda and bias_add.is_cuda
    x = x.contiguous()
    weight = weight.t().contiguous()  # Transpose for better memory layout
    M, K = x.shape
    K, N = weight.shape

    # Allocate output buffers
    out_gemm = torch.empty((M, N), device=x.device, dtype=torch.float32)
    out_final = torch.empty((M, N), device=x.device, dtype=torch.float32)

    # Launch kernel
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    # Heuristics for block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_SIZE_M = 8
    eps = 1e-5
    num_groups = gamma.shape[0] * (gamma.shape[1] // 128)  # assuming gamma is (num_groups, N // num_groups)

    matmul_groupnorm_min_bias_kernel[grid](
        x, weight, bias, gamma, beta, out_gemm, out_final,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out_gemm.stride(0), out_gemm.stride(1),
        out_final.stride(1),
        num_groups, eps,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return out_final


class ModelNew(nn.Module):
    """
    Optimized model with fused Triton kernel for GEMM + GroupNorm + Min(dim=1) + Bias.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_gemm = nn.Parameter(torch.empty(out_features))
        self.group_norm_weight = nn.Parameter(torch.ones(num_groups, out_features // num_groups))  # gamma
        self.group_norm_bias = nn.Parameter(torch.zeros(num_groups, out_features // num_groups))  # beta
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias_gemm, -bound, bound)
        nn.init.constant_(self.group_norm_bias, 0.)
        nn.init.constant_(self.group_norm_weight, 1.)

    def forward(self, x):
        # Fused kernel call
        x = triton_matmul_groupnorm_min_bias(
            x, self.weight, self.bias_gemm,
            self.group_norm_weight, self.group_norm_bias, self.bias.squeeze()
        )
        return x