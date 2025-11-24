import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    x_ptr,           # Pointer to input tensor (batch_size, input_size)
    weight_ptr,      # Pointer to weight matrix (hidden_size, input_size)
    output_ptr,      # Pointer to output tensor (batch_size, hidden_size)
    batch_size: tl.constexpr,
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes one block of rows (batch elements)
    batch_idx = tl.program_id(0)
    # If we exceed the batch size, skip
    if batch_idx >= batch_size:
        return

    # Compute the current batch index
    batch_offset = batch_idx

    # Load the entire input row for this batch
    x = tl.load(x_ptr + batch_offset * input_size, mask=None, other=0.0)
    # Reshape x to (input_size,) for processing
    x = x.reshape(input_size)

    # Compute the output row (hidden_size)
    # We will compute each output element via dot product with weight.T
    # weight.T has shape (input_size, hidden_size)
    # So we can compute: output[i] = sum_j x[j] * weight.T[j][i]

    # Create a range of indices for the input dimension
    input_offsets = tl.arange(0, input_size)
    # Load weight.T for the current batch's output row
    # weight_ptr points to (hidden_size, input_size) -> we need to load by column
    # So for each output element i, we load weight.T[j][i] = weight[i][j]
    # We can do this by iterating over output indices
    output_offsets = tl.arange(0, hidden_size)

    # For each output element
    for i in output_offsets:
        # Compute dot product: sum_j x[j] * weight[i][j]
        # Load weight[i] as a row
        weight_row = tl.load(weight_ptr + i * input_size + input_offsets, mask=input_offsets < input_size, other=0.0)
        # Compute dot product
        dot_product = tl.dot(x, weight_row)
        # Store result
        tl.store(output_ptr + batch_offset * hidden_size + i, dot_product)

    # The kernel computes the full row for the batch element


@triton.jit
def divide_and_sum_kernel(
    x_ptr,            # Pointer to input (batch_size, hidden_size)
    output_ptr,       # Pointer to output (batch_size, 1)
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return

    # Load the entire row of x for this batch
    x = tl.load(x_ptr + batch_idx * hidden_size, mask=None, other=0.0)
    # Divide by 2
    x_div_2 = x / 2.0
    # Sum over hidden_size dimension
    sum_val = tl.sum(x_div_2)
    # Store result in output (batch_idx, 0)
    tl.store(output_ptr + batch_idx, sum_val)


def triton_gemm(x: torch.Tensor, weight: torch.Tensor):
    """
    Custom Triton kernel for matrix multiplication: x @ weight.T
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()

    batch_size, input_size = x.shape
    hidden_size = weight.shape[0]

    # Output tensor: (batch_size, hidden_size)
    output = torch.empty((batch_size, hidden_size), device=x.device)

    # Determine block size (power of 2)
    BLOCK_SIZE = 128

    # Grid: number of blocks = batch_size
    grid = lambda meta: (batch_size,)

    # Launch the kernel
    gemm_kernel[grid](
        x_ptr=x.data_ptr(),
        weight_ptr=weight.data_ptr(),
        output_ptr=output.data_ptr(),
        batch_size=batch_size,
        input_size=input_size,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def triton_divide_and_sum(x: torch.Tensor, scaling_factor: float):
    """
    Custom Triton kernel for dividing by 2 and summing along dim=1, then scaling.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    batch_size, hidden_size = x.shape

    # Output tensor: (batch_size, 1)
    output = torch.empty((batch_size, 1), device=x.device)

    # Determine block size
    BLOCK_SIZE = 128

    # Grid: number of blocks = batch_size
    grid = lambda meta: (batch_size,)

    # Launch kernel
    divide_and_sum_kernel[grid](
        x_ptr=x.data_ptr(),
        output_ptr=output.data_ptr(),
        batch_size=batch_size,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Scale by scaling_factor
    output = output * scaling_factor

    return output


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Step 1: Matrix multiplication x @ weight.T
        x = triton_gemm(x, self.weight)
        # Step 2: Divide by 2
        x = x / 2.0
        # Step 3: Sum over dim=1 (hidden_size), keepdim=True
        x = torch.sum(x, dim=1, keepdim=True)
        # Step 4: Scale by scaling_factor
        x = x * self.scaling_factor
        return x