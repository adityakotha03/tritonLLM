import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate the starting indices for this block
    block_start_m = pid * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Create offsets for this block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create masks for bounds checking
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Calculate the current K offset
        k_offset = k * BLOCK_SIZE_K
        offs_k_current = offs_k + k_offset

        # Load A and B
        a = tl.load(
            a_ptr + (offs_m[:, None] * stride_am + offs_k_current[None, :] * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k_current[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + (offs_k_current[:, None] * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k_current[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        # Update accumulator
        accumulator += tl.dot(a, b)

    # Store result
    c = accumulator.to(tl.float32)
    tl.store(
        c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn),
        c,
        mask=mask,
    )


@triton.jit
def relu_kernel(
    x_ptr, y_ptr, n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv2d(x, weight, bias):
    # Input: (B, C_in, H, W)
    # Weight: (C_out, C_in, K, K)
    # Bias: (C_out, 1, 1)
    B, C_in, H, W = x.shape
    C_out, _, K, _ = weight.shape

    # Output dimensions
    H_out = H - K + 1
    W_out = W - K + 1

    # Reshape input for convolution: (B, C_in, H, W) -> (B, C_in, H_out, W_out, K, K)
    x = x.unfold(2, K, 1).unfold(3, K, 1)  # Shape: (B, C_in, H_out, W_out, K, K)
    x = x.permute(0, 1, 4, 5, 2, 3)  # Shape: (B, C_in, K, K, H_out, W_out)
    x = x.reshape(B, C_in * K * K, H_out * W_out)  # Shape: (B, C_in * K * K, H_out * W_out)

    # Weight: (C_out, C_in, K, K) -> (C_out, C_in * K * K)
    weight = weight.reshape(C_out, C_in * K * K)

    # Perform batched matrix multiplication: (B, C_out, H_out*W_out)
    out = torch.empty(B, C_out, H_out * W_out, dtype=x.dtype, device=x.device)

    # Get strides
    stride_xm, stride_xk, stride_xh = x.stride(0), x.stride(1), x.stride(2)
    stride_wk, stride_wn = weight.stride(0), weight.stride(1)
    stride_om, stride_on = out.stride(0), out.stride(1)

    # Launch kernel
    grid = lambda meta: (B, meta['C_out'], meta['H_out'] * meta['W_out'])
    matmul_kernel[grid](
        x, weight, out,
        B, C_out, C_in * K * K,
        stride_xm, stride_xk, stride_wk, stride_wn,
        stride_om, stride_on,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
    )

    # Reshape output back: (B, C_out, H_out * W_out) -> (B, C_out, H_out, W_out)
    out = out.view(B, C_out, H_out, W_out)

    # Apply bias: (B, C_out, H_out, W_out) + (C_out, 1, 1)
    out = out + bias.unsqueeze(0)

    # Apply ReLU
    out = out.clamp(min=0.0)

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return triton_conv2d(x, self.conv.weight, self.bias)