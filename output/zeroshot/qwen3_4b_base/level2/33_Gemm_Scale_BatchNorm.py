import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    x_ptr,  # pointer to input tensor (batch, in_features)
    w_ptr,  # pointer to weight matrix (out_features, in_features)
    out_ptr,  # pointer to output tensor (batch, out_features)
    batch_size: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID for block
    pid = tl.program_id(0)
    # Compute which row of output this block is responsible for
    row = pid // (batch_size * out_features // BLOCK_SIZE_M)
    # Compute which column of output this block is responsible for
    col = pid % (out_features // BLOCK_SIZE_N)

    # Compute the offset in the output tensor
    offset_row = row * BLOCK_SIZE_M
    offset_col = col * BLOCK_SIZE_N

    # Load weights in a tiled fashion
    # We'll tile the weight matrix and compute the dot product
    # Each thread computes one element of the output
    # We use a loop over the inner dimension to compute the dot product
    # Using a block-level loop to avoid out-of-bounds
    # We assume that the input is batched and we process one batch at a time
    # This kernel assumes batch_size is known at compile time

    # Compute the output row and column
    row_start = offset_row
    row_end = row_start + BLOCK_SIZE_M
    col_start = offset_col
    col_end = col_start + BLOCK_SIZE_N

    # Compute the output element
    out = 0.0
    for k in range(in_features):
        # Load input x[i, k]
        x_val = tl.load(x_ptr + (row_start + tl.arange(0, BLOCK_SIZE_M)) * in_features + k, mask=(row_start + tl.arange(0, BLOCK_SIZE_M)) < batch_size * in_features, other=0.0)
        # Load weight w[k, j]
        w_val = tl.load(w_ptr + k * out_features + (col_start + tl.arange(0, BLOCK_SIZE_N)), mask=(k < in_features) & (col_start + tl.arange(0, BLOCK_SIZE_N)) < out_features, other=0.0)
        out += x_val * w_val
    # Store the result
    tl.store(out_ptr + (row_start + tl.arange(0, BLOCK_SIZE_M)) * out_features + col_start, out, mask=(row_start + tl.arange(0, BLOCK_SIZE_M)) < batch_size * out_features)


@triton.jit
def scale_kernel(
    x_ptr,  # pointer to input tensor (batch, out_features)
    scale_ptr,  # pointer to scale parameter (out_features,)
    out_ptr,  # pointer to output tensor (batch, out_features)
    batch_size: tl.constexpr,
    out_features: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * out_features
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + offsets % out_features, mask=offsets % out_features < out_features, other=1.0)
    out = x * scale
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def batch_norm_kernel(
    x_ptr,  # pointer to input (batch, out_features)
    mean_ptr,  # pointer to mean (out_features,)
    var_ptr,  # pointer to variance (out_features,)
    gamma_ptr,  # pointer to gamma (out_features,)
    beta_ptr,  # pointer to beta (out_features,)
    out_ptr,  # pointer to output (batch, out_features)
    batch_size: tl.constexpr,
    out_features: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * out_features
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Load parameters
    gamma = tl.load(gamma_ptr + offsets % out_features, mask=offsets % out_features < out_features, other=1.0)
    beta = tl.load(beta_ptr + offsets % out_features, mask=offsets % out_features < out_features, other=0.0)
    mean = tl.load(mean_ptr + offsets % out_features, mask=offsets % out_features < out_features, other=0.0)
    var = tl.load(var_ptr + offsets % out_features, mask=offsets % out_features < out_features, other=1.0)
    # Compute batch norm
    std = tl.sqrt(var + 1e-5)
    out = (x - mean) / std * gamma + beta
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gemm(x: torch.Tensor, w: torch.Tensor):
    """
    Custom GEMM kernel using Triton.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    batch_size, in_features = x.shape
    out_features = w.shape[0]

    # Use FP16 for better performance on A100 Tensor Cores
    x = x.half()
    w = w.half()

    # Grid size
    grid = lambda meta: ((batch_size * out_features + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],)
    # Use BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128

    # Launch GEMM kernel
    gemm_kernel[grid](
        x_ptr=x.data_ptr(),
        w_ptr=w.data_ptr(),
        out_ptr=torch.empty(batch_size, out_features, dtype=torch.float16, device=x.device).data_ptr(),
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return torch.empty(batch_size, out_features, dtype=torch.float16, device=x.device)


def triton_scale(x: torch.Tensor, scale: torch.Tensor):
    """
    Custom scaling kernel using Triton.
    """
    assert x.is_cuda and scale.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    scale = scale.contiguous()

    batch_size, out_features = x.shape
    scale = scale.half()

    # Grid size
    grid = lambda meta: ((batch_size * out_features + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    BLOCK_SIZE = 128

    # Launch scaling kernel
    scale_kernel[grid](
        x_ptr=x.data_ptr(),
        scale_ptr=scale.data_ptr(),
        out_ptr=torch.empty(batch_size, out_features, dtype=torch.float16, device=x.device).data_ptr(),
        batch_size=batch_size,
        out_features=out_features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return torch.empty(batch_size, out_features, dtype=torch.float16, device=x.device)


def triton_batch_norm(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
    """
    Custom batch norm kernel using Triton.
    """
    assert x.is_cuda and mean.is_cuda and var.is_cuda and gamma.is_cuda and beta.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    batch_size, out_features = x.shape
    # Use FP16 for performance
    x = x.half()
    mean = mean.half()
    var = var.half()
    gamma = gamma.half()
    beta = beta.half()

    grid = lambda meta: ((batch_size * out_features + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    BLOCK_SIZE = 128

    out = torch.empty(batch_size, out_features, dtype=torch.float16, device=x.device)
    batch_norm_kernel[grid](
        x_ptr=x.data_ptr(),
        mean_ptr=mean.data_ptr(),
        var_ptr=var.data_ptr(),
        gamma_ptr=gamma.data_ptr(),
        beta_ptr=beta.data_ptr(),
        out_ptr=out.data_ptr(),
        batch_size=batch_size,
        out_features=out_features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # We use a learned weight matrix instead of nn.Linear
        self.weight = torch.randn(out_features, in_features, dtype=torch.float16).cuda()
        self.scale = nn.Parameter(torch.randn(scale_shape, dtype=torch.float16))
        # Precompute mean and variance for batch norm (can be updated during training)
        self.register_buffer("mean", torch.zeros(out_features, dtype=torch.float16))
        self.register_buffer("var", torch.ones(out_features, dtype=torch.float16))
        self.gamma = torch.randn(out_features, dtype=torch.float16)
        self.beta = torch.zeros(out_features, dtype=torch.float16)

    def forward(self, x):
        # Convert input to FP16
        x = x.half()

        # GEMM using custom Triton kernel
        x = triton_gemm(x, self.weight)

        # Apply scaling
        x = triton_scale(x, self.scale)

        # Apply batch normalization
        x = triton_batch_norm(x, self.mean, self.var, self.gamma, self.beta)

        return x