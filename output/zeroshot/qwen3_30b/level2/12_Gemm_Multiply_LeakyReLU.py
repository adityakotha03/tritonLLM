import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gemm_mul_relu_kernel(
    x_ptr,       # Input pointer
    w_ptr,       # Weight pointer
    b_ptr,       # Bias pointer (optional)
    out_ptr,     # Output pointer
    n_elements,  # Total number of elements in input
    n_rows,      # Number of rows in the output (batch_size * out_features)
    n_cols,      # Number of columns in the output (in_features)
    multiplier,  # Scaling factor
    negative_slope,  # LeakyReLU negative slope
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Define program ID and offsets
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate block offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Create offsets for the current block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create masks to handle boundaries
    mask_m = offs_m < n_rows
    mask_n = offs_n < n_cols
    mask = mask_m[:, None] & mask_n[None, :]

    # Allocate shared memory for tiles
    shared_A = tl.load(
        tl.make_block_ptr(x_ptr, (n_rows, n_cols), (n_cols, 1), (block_start_m, 0), (BLOCK_SIZE_M, BLOCK_SIZE_K), (0, 0))
    )
    shared_B = tl.load(
        tl.make_block_ptr(w_ptr, (n_cols, n_rows), (1, n_cols), (0, block_start_n), (BLOCK_SIZE_K, BLOCK_SIZE_N), (0, 0))
    )

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Perform matrix multiplication with tiling
    for k in range(0, n_cols, BLOCK_SIZE_K):
        # Load data
        a = tl.load(
            tl.make_block_ptr(x_ptr, (n_rows, n_cols), (n_cols, 1), (block_start_m, k), (BLOCK_SIZE_M, BLOCK_SIZE_K), (0, 0)),
            mask=(offs_m[:, None] < n_rows) & (offs_k[None, :] < n_cols),
            other=0.0
        )
        b = tl.load(
            tl.make_block_ptr(w_ptr, (n_cols, n_rows), (1, n_cols), (k, block_start_n), (BLOCK_SIZE_K, BLOCK_SIZE_N), (0, 0)),
            mask=(offs_k[:, None] < n_cols) & (offs_n[None, :] < n_rows),
            other=0.0
        )

        # Perform matmul: accumulator += a @ b
        accumulator += tl.dot(a, b)

    # Apply bias if present
    if b_ptr is not None:
        bias = tl.load(
            tl.make_block_ptr(b_ptr, (n_cols,), (1,), (block_start_n,), (BLOCK_SIZE_N,), (0,))
        )
        accumulator += bias[None, :]

    # Scale by multiplier
    accumulator *= multiplier

    # Apply LeakyReLU
    out = tl.where(accumulator > 0, accumulator, negative_slope * accumulator)

    # Store output
    tl.store(
        tl.make_block_ptr(out_ptr, (n_rows, n_cols), (n_cols, 1), (block_start_m, block_start_n), (BLOCK_SIZE_M, BLOCK_SIZE_N), (0, 0)),
        out,
        mask=mask
    )


def triton_gemm_mul_relu(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, multiplier: float, negative_slope: float):
    """
    Custom Triton kernel wrapper for fused Gemm, multiply, and LeakyReLU.
    """
    assert x.is_cuda and w.is_cuda and (b is None or b.is_cuda), "All tensors must be on CUDA."

    # Ensure contiguity
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous() if b is not None else None

    # Output shape
    batch_size, in_features = x.shape
    out_features = w.shape[0]

    # Compute output shape
    n_rows = batch_size * out_features
    n_cols = in_features
    n_elements = n_rows * n_cols

    # Set block sizes for optimal performance on A100
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64

    # Grid setup
    grid = lambda meta: (
        triton.cdiv(batch_size, meta["BLOCK_SIZE_M"]),
        triton.cdiv(out_features, meta["BLOCK_SIZE_N"]),
    )

    # Launch kernel
    gemm_mul_relu_kernel[grid](
        x, w, b, x, n_elements, n_rows, n_cols, multiplier, negative_slope,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    # Reshape output to original shape
    return x.view(batch_size, out_features)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        # Fuse Gemm, Multiply, and LeakyReLU into a single Triton kernel
        return triton_gemm_mul_relu(
            x, 
            self.gemm.weight, 
            self.gemm.bias, 
            self.multiplier, 
            self.leaky_relu.negative_slope
        )