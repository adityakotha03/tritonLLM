import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_sum_scale_kernel(
    x_ptr,       # Pointer to input x
    w_ptr,       # Pointer to weight matrix (hidden_size, input_size)
    out_ptr,     # Pointer to output
    batch_size,  # Number of batches
    input_size,  # Input dimension
    hidden_size, # Output dimension
    scaling_factor,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Map the program to a block of data
    pid_m = tl.program_id(0)  # Block ID along M dimension
    pid_n = tl.program_id(1)  # Block ID along N dimension
    pid_k = tl.program_id(2)  # Block ID along K dimension

    # Calculate offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_k = pid_k * BLOCK_SIZE_K

    # Create indices for current block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K blocks
    for k in range(0, tl.cdiv(input_size, BLOCK_SIZE_K)):
        # Calculate actual offsets
        k_offset = k * BLOCK_SIZE_K
        k_mask = k_offset + offs_k < input_size

        # Load x and w slices
        x = tl.load(
            x_ptr + (offs_m[:, None] * input_size + offs_k[None, :]),
            mask=tl.expand_dims(k_mask, 0),
            other=0.0,
            cache_read=True
        )
        w = tl.load(
            w_ptr + (offs_k[:, None] * hidden_size + offs_n[None, :]),
            mask=tl.expand_dims(k_mask, 1),
            other=0.0,
            cache_read=True
        )

        # Perform matrix multiplication in fp16 (bfloat16 for better precision)
        accumulator += tl.dot(x, w, out_dtype=tl.float32)

    # Accumulate sum along N (i.e., input_size dimension) to reduce to per-row sum
    # But we only compute one output per batch, so we use reduction over N
    # We sum across the N dimension (input_size) → we want sum of each row
    # Note: Since we're doing matmul (B x K) @ (K x H), result is (B x H)
    # We then sum across H dimension (but in this case, we only want sum per batch, so we sum across H)

    # Compute sum over the hidden_size dimension
    # But we need to sum over the hidden_size dimension after matmul
    # So we sum across the H dimension → we do a reduction over N
    sum_output = tl.sum(accumulator, axis=1)  # (BLOCK_SIZE_M,)

    # Scale by the scaling factor
    sum_output = sum_output * scaling_factor

    # Write output
    output_ptr = out_ptr + offs_m
    tl.store(output_ptr, sum_output, mask=offs_m < batch_size)


def triton_matmul_sum_scale(x: torch.Tensor, w: torch.Tensor, scaling_factor: float):
    """
    Wrapper function for the Triton kernel.
    Combines matmul, sum over hidden_size, and scaling in a single fused kernel.
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA"
    x = x.contiguous()
    w = w.contiguous()

    batch_size, input_size = x.shape
    hidden_size = w.shape[0]

    # Output tensor
    out = torch.empty((batch_size, 1), dtype=x.dtype, device=x.device)

    # Define block sizes for optimal performance on A100
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64

    # Grid: (M_blocks, N_blocks, K_blocks)
    grid_m = (triton.cdiv(batch_size, BLOCK_SIZE_M),)
    grid_n = (triton.cdiv(hidden_size, BLOCK_SIZE_N),)
    grid_k = (triton.cdiv(input_size, BLOCK_SIZE_K),)

    # Launch kernel
    matmul_sum_scale_kernel[
        (grid_m[0], grid_n[0], grid_k[0]),
    ](
        x,
        w,
        out,
        batch_size,
        input_size,
        hidden_size,
        scaling_factor,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size).to(torch.bfloat16))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Cast input to bfloat16 for optimal Tensor Core performance
        x = x.to(torch.bfloat16)
        return triton_matmul_sum_scale(x, self.weight, self.scaling_factor)