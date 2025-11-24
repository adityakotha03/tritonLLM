import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_gn_leaky_relu_kernel(
    x_ptr, weight_ptr, bias_ptr,
    gamma_ptr, beta_ptr,
    out_ptr,
    batch_size, hidden_size, input_size,
    num_groups,
    eps,
    negative_slope,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Matrix multiplication tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * input_size + offs_k[None, :]
    w_ptrs = weight_ptr + offs_k[:, None] * hidden_size + offs_n[None, :]
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, input_size, BLOCK_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < batch_size) & (offs_k[None, :] < input_size), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < input_size) & (offs_n[None, :] < hidden_size), other=0.0)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * hidden_size

    c = accumulator.to(tl.float32)
    if pid_n == 0:
        c += tl.load(bias_ptr + offs_n, mask=offs_n < hidden_size, other=0.0)[None, :]

    # Reshape for group norm: (batch_size, num_groups, group_size)
    group_size = hidden_size // num_groups
    off_gn = (offs_m[:, None] // batch_size) * num_groups * group_size + \
             (offs_m[:, None] % batch_size) * group_size + \
             (offs_n[None, :] // group_size) * group_size + \
             (offs_n[None, :] % group_size)
    c = tl.load(c + off_gn, mask=(offs_m[:, None] < batch_size) & (offs_n[None, :] < hidden_size), other=0.0)

    # GroupNorm: compute mean and variance
    mean = tl.zeros((BLOCK_M,), dtype=tl.float32)
    var = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for g in range(num_groups):
        mask = (offs_m < batch_size) & (offs_n == g * group_size)
        group_vals = tl.load(c + offs_m[:, None] * hidden_size + g * group_size + (offs_n % group_size)[None, :], 
                             mask=(offs_m[:, None] < batch_size) & (offs_n[None, :] < group_size), other=0.0)
        mean += tl.sum(tl.sum(group_vals, axis=1), axis=0) / (group_size * num_groups)
        var += tl.sum(tl.sum(group_vals * group_vals, axis=1), axis=0) / (group_size * num_groups)
    mean = mean[:, None]
    var = var[:, None]
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine
    normalized = (c - mean) * inv_std
    gamma = tl.load(gamma_ptr + offs_n, mask=offs_n < hidden_size, other=1.0)
    beta = tl.load(beta_ptr + offs_n, mask=offs_n < hidden_size, other=0.0)
    out = normalized * gamma + beta

    # Leaky ReLU
    out = tl.where(out >= 0, out, out * negative_slope)

    # Element-wise sum: out = out + out
    out *= 2.0

    # Store result
    tl.store(out_ptr + offs_m[:, None] * hidden_size + offs_n[None, :], out, 
             mask=(offs_m[:, None] < batch_size) & (offs_n[None, :] < hidden_size))


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope

    def forward(self, x):
        batch_size, input_size = x.shape
        hidden_size = self.hidden_size
        num_groups = self.num_groups

        # Launch Triton kernel
        out = torch.empty((batch_size, hidden_size), device=x.device, dtype=x.dtype)

        # Tile sizes
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(batch_size, BLOCK_M), triton.cdiv(hidden_size, BLOCK_N))

        matmul_gn_leaky_relu_kernel[grid](
            x, self.fc.weight, self.fc.bias,
            self.gn.weight, self.gn.bias,
            out,
            batch_size, hidden_size, input_size,
            num_groups,
            self.eps,
            self.negative_slope,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return out