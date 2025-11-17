# Best Kernel for 90_cumprod
# Generated: 20251116_172729
# Speedup: 0.16x
# Runtime: 55.3000 ms
# Round: 0
# Idea: Given the PyTorch code, replace the operation with a custom Triton kernel

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumprod_pass1_kernel(
    x_ptr,          # *const T, input [M, N]
    y_ptr,          # *T, partial output [M, N]
    tile_prod_ptr,  # *T, tile products [M, T]
    M: tl.constexpr,
    N: tl.constexpr,
    T: tl.constexpr,             # number of tiles per row
    BLOCK_N: tl.constexpr,       # tile size along N
):
    pid_m = tl.program_id(0)  # row id [0..M)
    pid_t = tl.program_id(1)  # tile id [0..T)

    # Base linear index of the row
    row_base = pid_m * N
    tile_start = pid_t * BLOCK_N

    # Initialize running product for this tile
    # Use 1 as multiplicative identity; dtype will be inferred by ops
    carry = 1.0

    # Process the tile sequentially within the program
    # Note: we rely on mask to avoid OOB accesses
    i = 0
    while i < BLOCK_N:
        col = tile_start + i
        mask = (pid_m < M) & (col < N)
        # Load current value; if masked-out, load identity (1.0)
        val = tl.load(x_ptr + row_base + col, mask=mask, other=1.0)
        carry = carry * val
        # Store partial cumulative product for this position
        tl.store(y_ptr + row_base + col, carry, mask=mask)
        i += 1

    # Store the product of the entire tile (including masked positions which multiply by 1)
    # This is used to compute inter-tile prefix multipliers in pass2
    # tile_prod layout is [M, T]
    if pid_m < M:
        tl.store(tile_prod_ptr + pid_m * T + pid_t, carry)


@triton.jit
def cumprod_pass2_kernel(
    y_ptr,          # *T, partial output [M, N] from pass1
    tile_prod_ptr,  # *T, tile products [M, T]
    out_ptr,        # *T, final output [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    T: tl.constexpr,             # number of tiles per row
    BLOCK_N: tl.constexpr,       # tile size along N
):
    pid_m = tl.program_id(0)  # row id [0..M)

    row_base = pid_m * N
    # Running multiplier from previous tiles (exclusive product of prior tiles)
    carry = 1.0

    t = 0
    while t < T:
        tile_start = t * BLOCK_N
        # Vectorized load/store over the BLOCK_N elements of this tile
        offs = tile_start + tl.arange(0, BLOCK_N)
        mask = (pid_m < M) & (offs < N)
        y = tl.load(y_ptr + row_base + offs, mask=mask, other=1.0)
        out = y * carry
        tl.store(out_ptr + row_base + offs, out, mask=mask)

        # Update carry with the product of the current tile
        tile_prod = tl.load(tile_prod_ptr + pid_m * T + t, mask=(pid_m < M))
        carry = carry * tile_prod
        t += 1


def triton_cumprod_last_dim(x2d: torch.Tensor) -> torch.Tensor:
    """
    Compute cumprod along the last dimension of a 2D, row-major contiguous tensor using Triton.
    x2d: [M, N], contiguous, CUDA
    """
    assert x2d.is_cuda and x2d.is_contiguous(), "Input must be CUDA and contiguous."
    assert x2d.dim() == 2, "Input must be 2D."

    M, N = x2d.shape
    # Tunable tile size; fixed here to keep T consistent across passes.
    BLOCK_N = 1024
    T = (N + BLOCK_N - 1) // BLOCK_N

    # Allocate output and temporary buffers
    out = torch.empty_like(x2d)
    # We will write partial results directly into 'out' in pass1, and then apply multipliers in-place
    tile_prod = torch.empty((M, T), dtype=x2d.dtype, device=x2d.device)

    # Launch pass1: per-tile sequential scan + tile products
    grid1 = (M, T)
    cumprod_pass1_kernel[grid1](
        x2d, out, tile_prod,
        M, N, T,
        BLOCK_N=BLOCK_N,
        num_warps=2,
        num_stages=1,
    )

    # Launch pass2: per-row tile multipliers application (in-place ok)
    grid2 = (M,)
    cumprod_pass2_kernel[grid2](
        out, tile_prod, out,
        M, N, T,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=1,
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized cumulative product along a specified dimension using Triton kernels.
    If the dimension is not the last, the tensor is temporarily permuted to make the
    target dimension last, processed, and then permuted back.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim
        ndim = x.ndim
        if dim < 0:
            dim += ndim
        assert 0 <= dim < ndim, "Invalid dim"

        # If already last dimension and contiguous, we can directly view as 2D
        if dim != ndim - 1:
            # Move the target dimension to the last for contiguous scanning
            x_perm = x.movedim(dim, -1)
        else:
            x_perm = x

        # Ensure CUDA and contiguous layout
        assert x_perm.is_cuda, "Input must be on CUDA device."
        x_perm = x_perm.contiguous()

        # Collapse all leading dimensions into one batch dimension: [S, N]
        N = x_perm.shape[-1]
        S = x_perm.numel() // N
        x2d = x_perm.view(S, N)

        # Triton cumprod along last dim
        out2d = triton_cumprod_last_dim(x2d)

        # Reshape back to original permuted shape
        out_perm = out2d.view_as(x_perm)

        # Move the last dimension back to original position if needed
        if dim != ndim - 1:
            out = out_perm.movedim(-1, dim)
        else:
            out = out_perm
        return out


# Keep the same helpers to align with the original interface
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape, device='cuda')]

def get_init_inputs():
    return [dim]