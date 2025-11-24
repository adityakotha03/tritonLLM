import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_groupnorm_hardtanh_kernel(
    x_ptr, weight_ptr, bias_ptr, gamma_ptr, beta_ptr,
    out_ptr,
    batch_size, in_features, out_features, num_groups,
    B: tl.constexpr, I: tl.constexpr, O: tl.constexpr,
    G: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Pointers into input and weight matrices
    x_ptrs = x_ptr + offs_m[:, None] * I + offs_k[None, :]
    w_ptrs = weight_ptr + offs_k[:, None] * O + offs_n[None, :]

    # Accumulate matrix multiplication result
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Matrix multiplication loop with tiling
    for k in range(0, tl.cdiv(I, BLOCK_SIZE_K)):
        # Load tiles with masks to avoid out-of-bounds
        x_mask = (offs_m < B)[:, None] & (offs_k < I)[None, :]
        w_mask = (offs_k < I)[:, None] & (offs_n < O)[None, :]
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_SIZE_K
        w_ptrs += BLOCK_SIZE_K * O

    # Add bias
    if pid_n == 0:
        bias_ptrs = bias_ptr + offs_m * O + offs_n
        bias_mask = (offs_m < B) & (offs_n < O)
        bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
        accumulator += bias[None, :]

    # Reshape accumulator for group norm: treat each group as a channel
    group_size = O // G
    offs_o = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    group_idx = offs_o // group_size
    within_group_idx = offs_o % group_size

    # Compute mean per group
    mean = tl.zeros((BLOCK_SIZE_M, G), dtype=tl.float32)
    count = 0
    for g in range(G):
        mask = (offs_m < B)[:, None] & (group_idx[None, :] == g)
        group_vals = tl.where(mask, tl.trans(accumulator), 0.0)
        mean += tl.sum(group_vals, axis=1)[:, None]
        count += tl.sum(mask, axis=1)[:, None]
    mean /= count

    # Compute variance per group
    var = tl.zeros((BLOCK_SIZE_M, G), dtype=tl.float32)
    for g in range(G):
        mask = (offs_m < B)[:, None] & (group_idx[None, :] == g)
        centered = tl.where(mask, tl.trans(accumulator) - mean[:, g][:, None], 0.0)
        var += tl.sum(centered * centered, axis=1)[:, None]
    var /= count
    inv_std = 1.0 / tl.sqrt(var + 1e-5)

    # Normalize and apply affine transform (gamma, beta)
    gamma_ptrs = gamma_ptr + group_idx
    beta_ptrs = beta_ptr + group_idx
    gamma = tl.load(gamma_ptrs, mask=offs_o < O, other=1.0)
    beta = tl.load(beta_ptrs, mask=offs_o < O, other=0.0)
    g_idx = group_idx
    normalized = (tl.trans(accumulator) - mean[:, g_idx]) * inv_std[:, g_idx]
    fused_out = gamma * normalized + beta

    # HardTanh: clamp between hardtanh_min and hardtanh_max
    hardtanh_min = -2.0
    hardtanh_max = 2.0
    fused_out = tl.maximum(fused_out, hardtanh_min)
    fused_out = tl.minimum(fused_out, hardtanh_max)

    # Store output
    out_ptrs = out_ptr + offs_m[:, None] * O + offs_n[None, :]
    out_mask = (offs_m < B)[:, None] & (offs_n < O)[None, :]
    tl.store(out_ptrs, fused_out, mask=out_mask)


class ModelNew(nn.Module):
    """
    Optimized version of Model using fused Triton kernel for GEMM + GroupNorm + HardTanh.
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

        # Linear layer parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # GroupNorm parameters
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Ensure input is contiguous
        x = x.contiguous()

        # Output tensor
        out = torch.empty(x.shape[0], self.out_features, device=x.device, dtype=x.dtype)

        # Launch Triton kernel
        def grid(meta):
            return (
                triton.cdiv(x.shape[0], meta['BLOCK_SIZE_M']),
                triton.cdiv(self.out_features, meta['BLOCK_SIZE_N']),
            )

        # Autotune kernel for best performance
        matmul_groupnorm_hardtanh_kernel_opt = triton.autotune(
            configs=[
                triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
                triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
            ],
            key=['B', 'I', 'O'],
        )(matmul_groupnorm_hardtanh_kernel)

        matmul_groupnorm_hardtanh_kernel_opt[
            grid
        ](
            x, self.weight, self.bias, self.gamma, self.beta,
            out,
            x.shape[0], self.in_features, self.out_features, self.num_groups,
            B=x.shape[0], I=self.in_features, O=self.out_features, G=self.num_groups
        )
        return out