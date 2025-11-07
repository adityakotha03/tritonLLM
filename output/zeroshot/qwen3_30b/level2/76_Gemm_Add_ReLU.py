import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_add_relu_kernel(
    x_ptr,  # Input matrix pointer
    w_ptr,  # Weight matrix pointer
    bias_ptr,  # Bias pointer
    out_ptr,  # Output pointer
    batch_size,  # Batch size
    in_features,  # Input feature size
    out_features,  # Output feature size
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    TILE_SIZE_M: tl.constexpr,
    TILE_SIZE_N: tl.constexpr,
    TILE_SIZE_K: tl.constexpr,
):
    # Shared memory for tiling: used to store tiles of x and w
    # Note: Triton implicitly handles shared memory via tiling and caching
    # We'll use the block-level caching for matrix multiplication

    # Define program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute block offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_k = pid_k * BLOCK_SIZE_K

    # Create indices for M and N dimensions
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)

    # Create masks for boundary conditions
    mask_m = offs_m < batch_size
    mask_n = offs_n < out_features
    mask_k = offs_k < in_features

    # Load input and weights with masking
    x = tl.load(
        x_ptr + (offs_m[:, None] * in_features + offs_k[None, :]),
        mask=(mask_m[:, None] & mask_k[None, :]),
        other=0.0
    )
    w = tl.load(
        w_ptr + (offs_k[:, None] * out_features + offs_n[None, :]),
        mask=(mask_k[:, None] & mask_n[None, :]),
        other=0.0
    )

    # Perform matrix multiplication with fused reduction
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over tiles of K dimension
    for _ in range(0, (in_features + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K):
        # Compute dot product: x_k * w_k
        dot = tl.dot(x, w, allow_tf32=True)
        accumulator += dot

        # Advance to next tile of K
        block_start_k += BLOCK_SIZE_K
        offs_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < in_features

        # Load next tile of x and w
        x = tl.load(
            x_ptr + (offs_m[:, None] * in_features + offs_k[None, :]),
            mask=(mask_m[:, None] & mask_k[None, :]),
            other=0.0
        )
        w = tl.load(
            w_ptr + (offs_k[:, None] * out_features + offs_n[None, :]),
            mask=(mask_k[:, None] & mask_n[None, :]),
            other=0.0
        )

    # Add bias
    bias = tl.load(
        bias_ptr + offs_n,
        mask=mask_n,
        other=0.0
    )
    accumulator += bias[None, :]

    # Apply ReLU
    accumulator = tl.where(accumulator > 0, accumulator, 0.0)

    # Store output
    out_ptrs = out_ptr + (offs_m[:, None] * out_features + offs_n[None, :])
    tl.store(out_ptrs, accumulator, mask=(mask_m[:, None] & mask_n[None, :]))


def triton_gemm_add_relu(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor):
    # Ensure inputs are contiguous and on GPU
    assert x.is_cuda and w.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    bias = bias.contiguous()

    # Define shape parameters
    batch_size, in_features = x.shape
    out_features = bias.shape[0]

    # Define block sizes (tuned for A100)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    TILE_SIZE_M = 128
    TILE_SIZE_N = 256
    TILE_SIZE_K = 64

    # Compute number of blocks
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (in_features + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    # Define grid (3D grid for 3D tiling)
    grid = (grid_m, grid_n, grid_k)

    # Launch kernel
    matmul_add_relu_kernel[grid](
        x,
        w,
        bias,
        torch.empty_like(x),
        batch_size,
        in_features,
        out_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        TILE_SIZE_M=TILE_SIZE_M,
        TILE_SIZE_N=TILE_SIZE_N,
        TILE_SIZE_K=TILE_SIZE_K,
    )


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float32, device='cuda'))
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32, device='cuda'))

    def forward(self, x):
        # Use Triton kernel for fused gemm + bias + relu
        return triton_gemm_add_relu(x, self.weight, self.bias)