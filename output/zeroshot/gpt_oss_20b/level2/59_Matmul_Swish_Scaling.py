import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel: fused matmul + swish + scaling
@triton.jit
def matmul_swish_kernel(
    a_ptr,        # (M, K)
    b_ptr,        # (N, K) – weight transposed
    out_ptr,      # (M, N)
    M, N, K,      # matrix dimensions
    scaling_factor: tl.constexpr,          # scalar to multiply
    BLOCK_SIZE: tl.constexpr,              # tile size
):
    # Program indices
    block_m = tl.program_id(0)
    block_n = tl.program_id(1)

    # Row and column indices for this tile
    row = block_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = block_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask for boundaries
    mask_row = row < M
    mask_col = col < N

    # Accumulator
    acc = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)

    # Iterate over K in tiles
    for k in range(0, K, BLOCK_SIZE):
        k_off = k + tl.arange(0, BLOCK_SIZE)

        # Load tiles from A (M x K) and B (N x K) – BF16 for speed
        a_tile = tl.load(
            a_ptr + row[:, None] * K + k_off[None, :],
            mask=mask_row[:, None] & (k_off[None, :] < K),
            other=0.0,
            device=self.device,
            dtype=tl.bfloat16,
        )
        b_tile = tl.load(
            b_ptr + k_off[:, None] * N + col[None, :],
            mask=(k_off[:, None] < K) & mask_col[None, :],
            other=0.0,
            device=self.device,
            dtype=tl.bfloat16,
        )

        # Convert to FP32 for accumulation
        a_tile = a_tile.to(tl.float32)
        b_tile = b_tile.to(tl.float32)

        # Matrix multiply for the tile
        acc += tl.dot(a_tile, b_tile)

    # Apply Swish (x * sigmoid(x)) and scaling
    acc = acc * tl.sigmoid(acc) * scaling_factor

    # Store the result
    tl.store(
        out_ptr + row[:, None] * N + col[None, :],
        acc,
        mask=mask_row[:, None] & mask_col[None, :],
    )


# Helper to launch the kernel
def triton_linear_swish(x: torch.Tensor, weight: torch.Tensor, scaling_factor: float):
    assert x.is_cuda and weight.is_cuda
    M, K = x.shape
    N, _ = weight.shape
    # Transpose weight to (N, K) for the kernel
    weight_t = weight.t()

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 128  # can be autotuned
    grid = lambda meta: (
        (M + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    matmul_swish_kernel[grid](
        x,
        weight_t,
        out,
        M,
        N,
        K,
        scaling_factor=scaling_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# New model definition
class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int, scaling_factor: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        # Use fp32 weight but store as BF16 for kernel
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor):
        # Ensure input is float32
        x = x.to(torch.float32)
        return triton_linear_swish(x, self.weight, self.scaling_factor)