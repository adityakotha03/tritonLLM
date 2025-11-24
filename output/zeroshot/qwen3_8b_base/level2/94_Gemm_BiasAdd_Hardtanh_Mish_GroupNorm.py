import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_bias_add_kernel(
    a_ptr, b_ptr, bias_ptr, out_ptr,
    n_rows, n_cols, k, 
    stride_a, stride_b, stride_bias, stride_out,
    BLOCK_SIZE: tl.constexpr
):
    # Each thread block processes a block of rows
    row_idx = tl.program_id(0)
    # Each thread processes a column
    col_idx = tl.program_id(1)
    # Compute the offset for the current row and column
    row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    # Compute the matrix multiplication
    for i in range(k):
        a = tl.load(a_ptr + row_offsets[:, None] * stride_a + i * stride_a, mask=(row_offsets < n_rows)[:, None], other=0.0)
        b = tl.load(b_ptr + col_offsets[None, :] * stride_b + i * stride_b, mask=(col_offsets < n_cols)[None, :], other=0.0)
        acc += tl.dot(a, b)
    # Add bias
    bias = tl.load(bias_ptr + row_offsets, mask=(row_offsets < n_rows), other=0.0)
    acc += bias[None, :]
    # Store the result
    tl.store(out_ptr + row_offsets[:, None] * stride_out + col_offsets[None, :], acc, mask=(row_offsets < n_rows)[:, None] & (col_offsets < n_cols)[None, :])


def triton_gemm_bias_add(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, out: torch.Tensor):
    assert a.is_cuda and b.is_cuda and bias.is_cuda and out.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()
    bias = bias.contiguous()
    out = out.contiguous()
    
    n_rows = a.shape[0]
    n_cols = b.shape[1]
    k = a.shape[1]
    
    # Determine block size
    BLOCK_SIZE = 128
    
    # Compute grid size
    grid = (tl.cdiv(n_rows, BLOCK_SIZE), tl.cdiv(n_cols, BLOCK_SIZE))
    
    # Launch the kernel
    gemm_bias_add_kernel[grid](a, b, bias, out, n_rows, n_cols, k, 
                               a.stride(0), b.stride(0), bias.stride(0), out.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def hardtanh_mish_kernel(
    x_ptr, out_ptr,
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Hardtanh: max(0, min(1, x))
    x = tl.where(x > 1, 1, x)
    x = tl.where(x < 0, 0, x)
    # Mish: x * tanh(softplus(x))
    x = x * tl.nn.functional.softplus(x)
    x = tl.tanh(x)
    # Store the result
    tl.store(out_ptr + offsets, x, mask=mask)


def triton_hardtanh_mish(x: torch.Tensor, out: torch.Tensor):
    assert x.is_cuda and out.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    out = out.contiguous()
    
    n_elements = x.numel()
    BLOCK_SIZE = 128
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    hardtanh_mish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def groupnorm_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    n_groups, n_channels, n_elements_per_group, 
    group_size, 
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements_per_group
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute mean and variance
    mean = tl.sum(x) / BLOCK_SIZE
    var = tl.sum((x - mean) * (x - mean)) / BLOCK_SIZE
    # Normalize
    x = (x - mean) / tl.sqrt(var + eps)
    # Scale and shift
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
    x = x * gamma + beta
    # Store the result
    tl.store(out_ptr + offsets, x, mask=mask)


def triton_groupnorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, out: torch.Tensor):
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda and out.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    out = out.contiguous()
    
    n_groups = x.shape[1] // 256
    n_channels = x.shape[1]
    n_elements_per_group = x.shape[0] * 256
    group_size = x.shape[0]
    
    BLOCK_SIZE = 128
    
    grid = lambda meta: ((n_elements_per_group + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    groupnorm_kernel[grid](x, gamma, beta, out, n_groups, n_channels, n_elements_per_group, group_size, 1e-5, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_shape = bias_shape
        self.num_groups = num_groups
        
        # Custom Triton kernel for GEMM + BiasAdd
        self.gemm_bias_add = None
        # Custom Triton kernel for Hardtanh + Mish
        self.hardtanh_mish = None
        # Custom Triton kernel for GroupNorm
        self.groupnorm = None

    def forward(self, x):
        # Custom GEMM + BiasAdd
        x = triton_gemm_bias_add(x, torch.randn(self.out_features, self.in_features).cuda(), torch.randn(self.out_features).cuda(), x)
        # Custom Hardtanh + Mish
        x = triton_hardtanh_mish(x, x)
        # Custom GroupNorm
        x = triton_groupnorm(x, torch.randn(self.out_features).cuda(), torch.randn(self.out_features).cuda(), x)
        return x


batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)
num_groups = 256

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, bias_shape, num_groups]