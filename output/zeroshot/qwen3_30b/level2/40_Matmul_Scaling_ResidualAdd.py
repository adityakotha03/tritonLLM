import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_add_kernel(
    x_ptr,  # Pointer to input
    w_ptr,  # Pointer to weights
    out_ptr,  # Pointer to output
    batch_size,  # Number of batches
    in_features,  # Input feature dimension
    out_features,  # Output feature dimension
    scaling_factor,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Define program ID along the batch dimension
    pid = tl.program_id(0)
    # Determine which batch we are processing
    batch_offset = pid * batch_size
    # Create thread offsets for M and N dimensions
    offsets_m = tl.arange(0, BLOCK_SIZE_M)
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    # Calculate row and column indices for the current block
    row_idx = offsets_m[:, None] + batch_offset
    col_idx = offsets_n[None, :]
    # Load input data (x) and weights (w)
    x_ptrs = x_ptr + row_idx * in_features + col_idx
    w_ptrs = w_ptr + col_idx * in_features + offsets_m
    # Create mask to avoid out-of-bounds memory access
    mask_x = (row_idx < batch_size)[:, None] & (col_idx < in_features)
    mask_w = (col_idx < out_features)[:, None] & (offsets_m < in_features)
    # Load data with masking
    x = tl.load(x_ptrs, mask=mask_x, other=0.0)
    w = tl.load(w_ptrs, mask=mask_w, other=0.0)
    # Perform matrix multiplication with reduction over K
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, in_features, BLOCK_SIZE_K):
        x = tl.load(x_ptrs + k, mask=(mask_x & (offsets_m[:, None] < in_features - k)), other=0.0)
        w = tl.load(w_ptrs + k * out_features, mask=(mask_w & (offsets_m < in_features - k)), other=0.0)
        acc += tl.dot(x, w)
    # Apply scaling and residual addition
    acc = acc * scaling_factor + acc
    # Store the result
    out_ptrs = out_ptr + row_idx * out_features + col_idx
    mask_out = (row_idx < batch_size)[:, None] & (col_idx < out_features)
    tl.store(out_ptrs, acc, mask=mask_out)


def triton_matmul_add(x: torch.Tensor, w: torch.Tensor, scaling_factor: float):
    """
    Custom Triton kernel wrapper for matmul + scaling + residual add.
    Combines multiple operations into one kernel to reduce memory traffic.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)
    # Get tensor dimensions
    batch_size = x.size(0)
    in_features = x.size(1)
    out_features = w.size(0)
    # Choose block sizes based on A100 capabilities
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    # Grid size: one block per batch
    grid = lambda meta: (batch_size,)

    # Launch kernel
    matmul_add_kernel[grid](
        x, w, out, batch_size, in_features, out_features, scaling_factor,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features).cuda(), requires_grad=False)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Use Triton kernel for matmul + scaling + residual add
        return triton_matmul_add(x, self.weight, self.scaling_factor)