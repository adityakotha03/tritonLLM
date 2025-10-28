import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# ---- Fused Add + LayerNorm Kernel and Wrapper ----

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def add_layernorm_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,
    sum_weight, eps,
    M, N,
    stride_x_m, stride_x_n,
    stride_y_m, stride_y_n,
    BLOCK_SIZE_N: tl.constexpr
):
    """
    Triton kernel for fused element-wise addition and LayerNorm.
    Normalization is performed over the last dimension (N).
    Each program instance processes one row of the input tensor.
    """
    row_idx = tl.program_id(0)
    
    X = X_ptr + row_idx * stride_x_m
    Y = Y_ptr + row_idx * stride_y_m
    
    # --- Pass 1: Compute mean ---
    mean = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        x = tl.load(X + cols * stride_x_n, mask=mask, other=0.0).to(tl.float32)
        x_plus_weight = x + sum_weight
        mean += x_plus_weight
    row_mean = tl.sum(mean, axis=0) / N
    
    # --- Pass 2: Compute variance ---
    var = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        x = tl.load(X + cols * stride_x_n, mask=mask, other=0.0).to(tl.float32)
        x_plus_weight = x + sum_weight
        x_centered = x_plus_weight - row_mean
        var += x_centered * x_centered
    row_var = tl.sum(var, axis=0) / N
    
    rstd = 1.0 / tl.sqrt(row_var + eps)
    
    # --- Pass 3: Normalize, scale, and shift ---
    for off in range(0, N, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        
        x = tl.load(X + cols * stride_x_n, mask=mask, other=0.0)
        x_plus_weight = x + sum_weight
        
        w = tl.load(W_ptr + cols, mask=mask)
        b = tl.load(B_ptr + cols, mask=mask)
        
        x_normalized = (x_plus_weight - row_mean) * rstd
        y = x_normalized * w + b
        
        tl.store(Y + cols * stride_y_n, y, mask=mask)


def triton_add_layernorm(x: torch.Tensor, sum_weight: torch.Tensor, norm_weight: torch.Tensor, norm_bias: torch.Tensor, eps: float):
    assert x.is_cuda and norm_weight.is_cuda and norm_bias.is_cuda, "Tensors must be on CUDA"
    
    x_2d = x.view(-1, x.shape[-1])
    M, N = x_2d.shape
    
    x_2d = x_2d.contiguous()
    
    y = torch.empty_like(x)
    y_2d = y.view(M, N)
    
    norm_weight = norm_weight.contiguous()
    norm_bias = norm_bias.contiguous()

    grid = lambda meta: (M, )
    
    add_layernorm_kernel[grid](
        x_2d, y_2d, norm_weight, norm_bias,
        sum_weight.item(), eps,
        M, N,
        x_2d.stride(0), x_2d.stride(1),
        y_2d.stride(0), y_2d.stride(1),
    )
    
    return y


# ---- GELU Kernel and Wrapper ----

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    inner = 0.7978845608 * (x + 0.044715 * x * x * x)
    cdf = 0.5 * (1.0 + tl.tanh(inner))
    out = x * cdf
    
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    gelu_kernel[grid](x, out, n_elements)
    
    return out


# ---- Optimized Model with Triton Kernels ----

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Add+LayerNorm fusion and GELU.
    The ConvTranspose3d and AvgPool3d operations are kept from PyTorch
    as their cuDNN-backed implementations are highly optimized. The main
    performance gain comes from fusing the memory-bound Add and LayerNorm
    operations, which saves a full memory read/write cycle of the intermediate tensor.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight, dtype=torch.float32))
        
        self.norm_weight = nn.Parameter(torch.ones(norm_shape))
        self.norm_bias = nn.Parameter(torch.zeros(norm_shape))
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_add_layernorm(x, self.sum_weight, self.norm_weight, self.norm_bias, self.eps)
        x = self.avg_pool(x)
        x = triton_gelu(x)
        return x