import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_sigmoid_scaling_residual_add_kernel(
    x_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    out_ptr,  # Pointer to output tensor
    batch_size,  # Number of batches
    input_size,  # Input size
    hidden_size,  # Hidden size
    scaling_factor,  # Scaling factor
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0)
    block_idx = pid % num_blocks
    block_start = block_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Compute the row index in the matrix
    row_idx = tl.arange(0, BLOCK_SIZE)
    col_idx = tl.arange(0, hidden_size)

    # Load weights and biases
    weight = tl.load(weight_ptr + col_idx, mask=col_idx < hidden_size, other=0.0)
    bias = tl.load(bias_ptr + col_idx, mask=col_idx < hidden_size, other=0.0)

    # Compute the matrix multiplication for this block
    x = tl.load(x_ptr + row_idx * input_size + col_idx, mask=col_idx < input_size, other=0.0)
    matmul = tl.dot(x, weight)
    matmul = matmul + bias

    # Apply sigmoid
    matmul = tl.sigmoid(matmul)

    # Apply scaling
    matmul = matmul * scaling_factor

    # Add residual
    residual = tl.load(x_ptr + row_idx * input_size + col_idx, mask=col_idx < input_size, other=0.0)
    out = matmul + residual

    # Store the result
    tl.store(out_ptr + row_idx * hidden_size + col_idx, out, mask=col_idx < hidden_size)


def triton_gemm_sigmoid_scaling_residual_add(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scaling_factor: float,
):
    """
    Triton kernel for Gemm + Sigmoid + Scaling + Residual Add
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Output tensor shape
    batch_size = x.size(0)
    input_size = x.size(1)
    hidden_size = weight.size(0)

    # Output tensor
    out = torch.empty(batch_size, hidden_size, device=x.device, dtype=x.dtype)

    # Determine block size
    BLOCK_SIZE = 64  # Tunable parameter for block size

    # Number of blocks needed
    num_blocks = (hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Grid size
    grid = (num_blocks,)

    # Launch kernel
    gemm_sigmoid_scaling_residual_add_kernel[grid](
        x,
        weight,
        bias,
        out,
        batch_size,
        input_size,
        hidden_size,
        scaling_factor,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Perform Gemm + Sigmoid + Scaling + Residual Add using Triton kernel
        return triton_gemm_sigmoid_scaling_residual_add(
            x,
            self.weight,
            self.bias,
            self.scaling_factor
        )