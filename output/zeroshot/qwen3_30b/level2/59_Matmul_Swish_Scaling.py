import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_swish_kernel(
    x_ptr,  # Input matrix pointer
    w_ptr,  # Weight matrix pointer
    out_ptr,  # Output matrix pointer
    n_elements,  # Total number of elements
    n_cols_x,  # Number of columns in input x
    n_rows_w,  # Number of rows in weights w
    n_cols_w,  # Number of columns in weights w
    scaling_factor,  # Scaling factor for output
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Calculate program ID for this block
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute block offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Create offsets within block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create mask to avoid out-of-bounds access
    mask_m = offs_m < n_rows_w
    mask_n = offs_n < n_cols_w
    mask = mask_m[:, None] & mask_n[None, :]

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension in blocks
    for k in range(0, n_cols_x, BLOCK_SIZE_K):
        # Load input X
        offs_k_local = k + offs_k
        x = tl.load(
            x_ptr + (offs_m[:, None] * n_cols_x + offs_k_local[None, :]),
            mask=(offs_m[:, None] < n_rows_w) & (offs_k_local[None, :] < n_cols_x),
            other=0.0
        )

        # Load weights W
        w = tl.load(
            w_ptr + (offs_k_local[:, None] * n_cols_w + offs_n[None, :]),
            mask=(offs_k_local[:, None] < n_cols_x) & (offs_n[None, :] < n_cols_w),
            other=0.0
        )

        # Perform matrix multiplication (FMA)
        acc += tl.dot(x, w)

    # Apply Swish activation: x * sigmoid(x)
    # Online computation of sigmoid for numerical stability
    acc_f32 = acc.to(tl.float32)
    x = acc_f32
    # Use the logit-based sigmoid approximation
    # sigmoid(x) = 1 / (1 + exp(-x)) -> numerically stable via softplus
    exp_neg_x = tl.exp(-tl.abs(x))
    sigmoid_x = tl.where(x >= 0, 1 / (1 + exp_neg_x), exp_neg_x / (1 + exp_neg_x))
    # Swish: x * sigmoid(x)
    swish = x * sigmoid_x

    # Scale output
    scaled = swish * scaling_factor

    # Store result with mask
    out_ptrs = out_ptr + (offs_m[:, None] * n_cols_w + offs_n[None, :])
    tl.store(out_ptrs, scaled, mask=mask)


def triton_matmul_swish(x: torch.Tensor, w: torch.Tensor, scaling_factor: float):
    """
    Custom Triton kernel for matmul + swish + scaling.
    Performs fused operation: output = (x @ w) * sigmoid(x @ w) * scaling_factor
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA"
    assert x.dtype == torch.float32 and w.dtype == torch.float32, "Only FP32 supported for now"
    assert x.shape[1] == w.shape[0], "Incompatible dimensions"

    # Ensure contiguous tensors
    x = x.contiguous()
    w = w.contiguous()

    # Extract dimensions
    batch_size, in_features = x.shape
    out_features = w.shape[1]

    # Prepare output
    out = torch.empty(batch_size, out_features, dtype=torch.float32, device=x.device)

    # Define block sizes - tuned for A100
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64

    # Grid configuration: 2D grid of thread blocks
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)

    # Launch kernel
    matmul_swish_kernel[grid](
        x, w, out,
        batch_size * out_features,
        in_features,  # n_cols_x
        batch_size,   # n_rows_w (not used, but for clarity)
        out_features, # n_cols_w
        scaling_factor,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        # Initialize weights using standard linear layer
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float32))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Fuse matmul, swish, and scaling into a single Triton kernel
        return triton_matmul_swish(x, self.weight, self.scaling_factor)