import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_relu_kernel(
    x_ptr,  # Input pointer
    w_ptr,  # Weight pointer
    b_ptr,  # Bias pointer (optional, set to None if not used)
    out_ptr,  # Output pointer
    n_elements,  # Total number of elements
    batch_size: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    subtract_value: tl.constexpr,
    multiply_value: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    # Program ID for the current block
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Block offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_k = pid_k * BLOCK_SIZE_K

    # Offsets for the current block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)

    # Load input batch (batch_size * in_features) - only load required portion
    # Assume input is [batch_size, in_features], process one row per thread block
    # Use row-major layout: [batch_size, in_features]
    # Only process one row at a time? No, process entire matrix multiplication efficiently

    # We need to restructure: instead of per-row, we tile matrix multiply across M, N, K
    # We will handle the batch loop implicitly via grid

    # Actually, better: use batched matmul via block-wise tiling
    # Each program_id handles a block of the output matrix (BLOCK_SIZE_M x BLOCK_SIZE_N)

    # But since batch is outermost, we can loop over batch and use block tiling for matmul

    # For simplicity, we assume batch_size is processed in a single kernel launch
    # So we treat the entire matrix as one big matrix of (batch_size x in_features) @ (in_features x out_features)

    # Total number of output elements: batch_size * out_features
    # Each thread block computes BLOCK_SIZE_M x BLOCK_SIZE_N output elements

    # Create row and col indices
    row = tl.load(x_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] * in_features + 
                  (tl.arange(0, BLOCK_SIZE_K)[None, :] * out_features))
    col = tl.load(w_ptr + (tl.arange(0, BLOCK_SIZE_K)[:, None] * out_features + 
                           (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :]))

    # Use shared memory to cache tile of X and W
    # Use shared memory for X tile: (BLOCK_SIZE_M x BLOCK_SIZE_K)
    x_shared = tl.load(x_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] * in_features + 
                       (tl.arange(0, BLOCK_SIZE_K)[None, :]), 
                       mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < batch_size) & 
                             (tl.arange(0, BLOCK_SIZE_K)[None, :] < in_features), 
                       other=0.0)
    w_shared = tl.load(w_ptr + (tl.arange(0, BLOCK_SIZE_K)[:, None] * out_features + 
                                (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :]), 
                       mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < in_features) & 
                             (tl.arange(0, BLOCK_SIZE_N)[None, :] < out_features), 
                       other=0.0)

    # Perform matmul: (BLOCK_SIZE_M x BLOCK_SIZE_K) @ (BLOCK_SIZE_K x BLOCK_SIZE_N)
    # Use tensor cores via matmul with fp16/bf16
    # But input is fp32, so we can still use tensor cores with casting

    # Use tl.dot to perform matrix multiply with tensor core support
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, in_features, BLOCK_SIZE_K):
        # Load X tile (BLOCK_SIZE_M x BLOCK_SIZE_K)
        x_tile = tl.load(
            x_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] * in_features + 
            (tl.arange(0, BLOCK_SIZE_K)[None, :] + k),
            mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < batch_size) & 
                  (tl.arange(0, BLOCK_SIZE_K)[None, :] + k < in_features),
            other=0.0
        )
        # Load W tile (BLOCK_SIZE_K x BLOCK_SIZE_N)
        w_tile = tl.load(
            w_ptr + (tl.arange(0, BLOCK_SIZE_K)[:, None] + k) * out_features + 
            (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :],
            mask=(tl.arange(0, BLOCK_SIZE_K)[None, :] + k < in_features) & 
                  (tl.arange(0, BLOCK_SIZE_N)[None, :] < out_features),
            other=0.0
        )
        # Compute partial matmul
        accumulator += tl.dot(x_tile, w_tile, allow_tf32=True)

    # Add bias if needed
    if USE_BIAS:
        bias = tl.load(b_ptr + tl.arange(0, out_features))
        accumulator += bias[None, :]

    # Apply linear transformation: accumulator now is (BLOCK_SIZE_M x BLOCK_SIZE_N)
    # Now apply: subtract_value, multiply_value, ReLU
    accumulator = accumulator - subtract_value
    accumulator = accumulator * multiply_value
    accumulator = tl.where(accumulator > 0, accumulator, 0.0)

    # Write output
    output_ptr = out_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] * out_features + \
                 (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :]
    tl.store(
        output_ptr,
        accumulator,
        mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < batch_size) & 
              (tl.arange(0, BLOCK_SIZE_N)[None, :] < out_features)
    )


def triton_fused_linear_relu(x, w, b, subtract_value, multiply_value):
    """
    Fused linear + subtract + multiply + ReLU using Triton.
    Assumes input x shape: (batch_size, in_features)
    Weights w shape: (in_features, out_features)
    Bias b shape: (out_features,)
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA"
    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()

    batch_size, in_features = x.shape
    out_features = w.shape[1]

    # Define block sizes for optimal performance on A100
    # Use powers of 2, multiple of 16 for tensor core efficiency
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32  # Multiple of 16, good for tensor cores

    # Compute number of blocks
    num_blocks_m = triton.cdiv(batch_size, BLOCK_SIZE_M)
    num_blocks_n = triton.cdiv(out_features, BLOCK_SIZE_N)
    num_blocks_k = triton.cdiv(in_features, BLOCK_SIZE_K)

    # Grid: (num_blocks_m, num_blocks_n, num_blocks_k)
    grid = (num_blocks_m, num_blocks_n, num_blocks_k)

    # Launch kernel
    fused_linear_relu_kernel[grid](
        x, w, b, x,  # Output overwrites input? No, we need a new output
        batch_size * in_features * out_features,
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        subtract_value=subtract_value,
        multiply_value=multiply_value,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        USE_BIAS=(b is not None),
    )

    # Create output tensor and return
    out = torch.empty_like(x)
    # But we need to recompute the kernel with correct output pointer
    # Fix: the kernel above writes to x — that's wrong. Let's restructure.

    # Actually, let's rewrite with correct output
    # Let me fix the kernel: change out_ptr to a new output tensor


# Let me rework the kernel with a correct output pointer
@triton.jit
def fused_linear_relu_kernel(
    x_ptr,        # Input: (batch_size, in_features)
    w_ptr,        # Weight: (in_features, out_features)
    b_ptr,        # Bias: (out_features,)
    out_ptr,      # Output: (batch_size, out_features)
    batch_size: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    subtract_value: tl.constexpr,
    multiply_value: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)  # M block (batch size)
    pid_n = tl.program_id(1)  # N block (out features)
    pid_k = tl.program_id(2)  # K block (in features tiling)

    # Offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_k = pid_k * BLOCK_SIZE_K

    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)

    # Accumulator for matmul
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K blocks
    for k in range(0, in_features, BLOCK_SIZE_K):
        # Load X tile: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        x_tile = tl.load(
            x_ptr + (offs_m[:, None] * in_features + (offs_k[None, :] + k)),
            mask=(offs_m[:, None] < batch_size) & (offs_k[None, :] + k < in_features),
            other=0.0
        )
        # Load W tile: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        w_tile = tl.load(
            w_ptr + ((offs_k[:, None] + k) * out_features + offs_n[None, :]),
            mask=(offs_k[:, None] + k < in_features) & (offs_n[None, :] < out_features),
            other=0.0
        )
        # Perform matmul: (BLOCK_SIZE_M x BLOCK_SIZE_K) @ (BLOCK_SIZE_K x BLOCK_SIZE_N)
        accumulator += tl.dot(x_tile, w_tile, allow_tf32=True)

    # Add bias if present
    if USE_BIAS:
        bias = tl.load(b_ptr + offs_n)
        accumulator += bias[None, :]

    # Apply non-linearity: subtract, multiply, ReLU
    accumulator = accumulator - subtract_value
    accumulator = accumulator * multiply_value
    accumulator = tl.where(accumulator > 0, accumulator, 0.0)

    # Store result
    output_ptr = out_ptr + (offs_m[:, None] * out_features + offs_n[None, :])
    tl.store(
        output_ptr,
        accumulator,
        mask=(offs_m[:, None] < batch_size) & (offs_n[None, :] < out_features)
    )


def triton_fused_linear_relu(x, w, b, subtract_value, multiply_value):
    """
    Fused linear + subtract + multiply + ReLU using Triton.
    x: (batch_size, in_features)
    w: (in_features, out_features)
    b: (out_features,) or None
    returns: (batch_size, out_features)
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA"
    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()

    batch_size, in_features = x.shape
    out_features = w.shape[1]

    # Choose optimal block sizes for A100
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    # Compute grid dimensions
    num_blocks_m = triton.cdiv(batch_size, BLOCK_SIZE_M)
    num_blocks_n = triton.cdiv(out_features, BLOCK_SIZE_N)
    num_blocks_k = triton.cdiv(in_features, BLOCK_SIZE_K)

    grid = (num_blocks_m, num_blocks_n, num_blocks_k)

    # Create output tensor
    out = torch.empty_like(x)

    # Launch kernel
    fused_linear_relu_kernel[grid](
        x, w, b, out,
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        subtract_value=subtract_value,
        multiply_value=multiply_value,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        USE_BIAS=(b is not None),
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        # Replace all operations with a single fused Triton kernel
        return triton_fused_linear_relu(
            x, self.linear.weight, self.linear.bias, self.subtract_value, self.multiply_value
        )