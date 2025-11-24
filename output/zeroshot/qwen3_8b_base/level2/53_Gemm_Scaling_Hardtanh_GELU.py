import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    a_stride, b_stride, c_stride,
    m, n, k,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block dimensions
    block_m = m // (tl.num_programs(0) * BLOCK_SIZE)
    block_n = n // (tl.num_programs(0) * BLOCK_SIZE)
    # Compute the block start
    block_start_m = pid // block_n * BLOCK_SIZE
    block_start_n = pid % block_n * BLOCK_SIZE
    # Compute the block offsets
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE)
    # Load A and B
    a = tl.load(a_ptr + offsets_m[:, None] * a_stride + tl.arange(0, k)[None, :], mask=offsets_m < m, other=0.0)
    b = tl.load(b_ptr + tl.arange(0, k)[None, :] * b_stride + offsets_n[None, :], mask=offsets_n < n, other=0.0)
    # Compute C = A @ B
    c = tl.dot(a, b)
    # Store C
    tl.store(c_ptr + offsets_m[:, None] * c_stride + offsets_n[None, :], c, mask=offsets_m < m & offsets_n < n)

def triton_gemm(a: torch.Tensor, b: torch.Tensor, m, n, k):
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty((m, n), dtype=a.dtype, device=a.device)
    # Determine the block size
    BLOCK_SIZE = 128
    # Determine the number of blocks
    num_blocks = (m + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the kernel
    gemm_kernel[ (num_blocks, ) ](a, b, c, a.stride(0), b.stride(0), c.stride(0), m, n, k, BLOCK_SIZE=BLOCK_SIZE)
    return c

@triton.jit
def scaling_kernel(
    x_ptr, y_ptr,
    n_elements,
    scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Perform scaling
    y = x * scaling_factor
    # Store the result
    tl.store(y_ptr + offsets, y, mask=mask)

def triton_scaling(x: torch.Tensor, scaling_factor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    y = torch.empty_like(x)
    # Number of elements in the tensor
    n_elements = x.numel()
    # Determine the number of blocks
    BLOCK_SIZE = 128
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the kernel
    scaling_kernel[ (num_blocks, ) ](x, y, n_elements, scaling_factor, BLOCK_SIZE=BLOCK_SIZE)
    return y

@triton.jit
def hardtanh_kernel(
    x_ptr, y_ptr,
    n_elements,
    min_val, max_val,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply Hardtanh
    y = tl.where(x < min_val, min_val, tl.where(x > max_val, max_val, x))
    # Store the result
    tl.store(y_ptr + offsets, y, mask=mask)

def triton_hardtanh(x: torch.Tensor, min_val, max_val):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    y = torch.empty_like(x)
    # Number of elements in the tensor
    n_elements = x.numel()
    # Determine the number of blocks
    BLOCK_SIZE = 128
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the kernel
    hardtanh_kernel[ (num_blocks, ) ](x, y, n_elements, min_val, max_val, BLOCK_SIZE=BLOCK_SIZE)
    return y

@triton.jit
def gelu_kernel(
    x_ptr, y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply GELU
    y = 0.5 * x * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    # Store the result
    tl.store(y_ptr + offsets, y, mask=mask)

def triton_gelu(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    y = torch.empty_like(x)
    # Number of elements in the tensor
    n_elements = x.numel()
    # Determine the number of blocks
    BLOCK_SIZE = 128
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the kernel
    gelu_kernel[ (num_blocks, ) ](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        # Perform GEMM
        x = triton_gemm(x, torch.randn(self.in_features, self.out_features, device=x.device), x.size(0), self.out_features, self.in_features)
        # Apply scaling
        x = triton_scaling(x, self.scaling_factor)
        # Apply Hardtanh
        x = triton_hardtanh(x, self.hardtanh_min, self.hardtanh_max)
        # Apply GELU
        x = triton_gelu(x)
        return x

batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]