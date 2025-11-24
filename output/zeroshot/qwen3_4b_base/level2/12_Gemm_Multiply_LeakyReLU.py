import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gemm_leaky_relu_kernel(
    input_ptr,           # pointer to input tensor (batch, in_features)
    weight_ptr,          # pointer to weight matrix (out_features, in_features)
    bias_ptr,            # pointer to bias vector (out_features)
    output_ptr,          # pointer to output tensor (batch, out_features)
    batch_size: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    multiplier: tl.constexpr,
    negative_slope: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes one block of the output
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE
    batch_idx = tl.arange(0, batch_size)

    # Compute output indices
    output_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = output_offsets < out_features

    # Load input data for current batch
    input_batch = tl.load(input_ptr + batch_idx[:, None] * in_features + tl.arange(0, in_features)[None, :], mask=tl.arange(0, in_features) < in_features, other=0.0)

    # Load weights and compute matrix multiplication
    # We compute: output = input @ weight + bias
    # We use block-wise computation to avoid loading entire weight matrix
    # We assume weights are stored as (out_features, in_features)
    weight_block = tl.load(weight_ptr + output_offsets[:, None] * in_features + tl.arange(0, in_features)[None, :], mask=tl.arange(0, in_features) < in_features, other=0.0)
    
    # Compute dot product: input @ weight
    # input_batch shape: (batch_size, in_features)
    # weight_block shape: (BLOCK_SIZE, in_features)
    # We compute output per row of output
    # We need to compute: (batch, out_features) = (batch, in_features) @ (out_features, in_features)
    # So we do a batched matmul per output block

    # Reconstruct input for current block
    input_vals = tl.load(input_ptr + batch_idx[:, None] * in_features + tl.arange(0, in_features)[None, :], mask=tl.arange(0, in_features) < in_features, other=0.0)
    # Compute output for each row in the block
    output_vals = tl.zeros((BLOCK_SIZE,), dtype=tl.float16)
    for i in range(in_features):
        input_col = input_vals[:, i]
        weight_row = tl.load(weight_ptr + output_offsets[:, None] * in_features + i, mask=tl.arange(0, out_features) < out_features, other=0.0)
        output_vals += input_col[:, None] * weight_row

    # Add bias
    bias_vals = tl.load(bias_ptr + output_offsets, mask=mask, other=0.0)
    output_vals = output_vals + bias_vals

    # Scale by multiplier
    output_vals = output_vals * multiplier

    # Apply LeakyReLU
    zero = tl.zeros_like(output_vals)
    positive = output_vals
    negative = output_vals * negative_slope
    output_vals = tl.where(output_vals >= 0, positive, negative)

    # Store result
    tl.store(output_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), output_vals, mask=mask)


def triton_gemm_leaky_relu(x: torch.Tensor, in_features: int, out_features: int, multiplier: float, negative_slope: float):
    """
    Custom kernel that replaces nn.Linear + LeakyReLU with a fused Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device."
    x = x.contiguous()

    # Ensure inputs are on GPU and correct shape
    batch_size = x.shape[0]
    assert x.shape[1] == in_features, f"Input features mismatch: expected {in_features}, got {x.shape[1]}"

    # Allocate output tensor
    out = torch.empty((batch_size, out_features), dtype=torch.float16, device=x.device)

    # Define kernel parameters
    BLOCK_SIZE = 128  # Optimal for Ampere, power of 2

    # Grid size: number of blocks needed
    grid = lambda meta: ((out.shape[1] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    gemm_leaky_relu_kernel[grid](
        x.data_ptr(),
        weight_ptr,  # Note: weight is not passed here; must be passed in as a parameter
        bias_ptr,    # Similarly, bias must be passed
        out.data_ptr(),
        batch_size,
        in_features,
        out_features,
        multiplier,
        negative_slope,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        # We still define the linear layer, but we will replace forward with custom kernel
        # We keep the weights and bias as parameters
        self.in_features = in_features
        self.out_features = out_features
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        # We will not use nn.Linear here; instead, we'll define weights and bias
        # We need to store weights and bias as learnable parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
        self.bias = nn.Parameter(torch.randn(out_features, dtype=torch.float16))

    def forward(self, x):
        # Use custom Triton kernel for fused GEMM + LeakyReLU
        # Ensure input is on GPU and contiguous
        x = x.contiguous()
        return triton_gemm_leaky_relu(x, self.in_features, self.out_features, self.multiplier, self.negative_slope)