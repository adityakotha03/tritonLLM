import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(
    input_ptr,  # Pointer to input tensor
    gamma_ptr,  # Pointer to gamma (scale) tensor
    beta_ptr,   # Pointer to beta (shift) tensor
    output_ptr, # Pointer to output tensor
    n_batch,    # Number of batches
    n_channels, # Number of channels
    n_elements_per_channel, # Number of elements per channel
    eps,        # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    # Calculate the offset for the current program
    offset = pid * BLOCK_SIZE
    # Create a range of offsets within the block
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements_per_channel

    # Load input values
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Compute mean and variance for the current channel
    mean = tl.math.avg(input, mask=mask)
    var = tl.math.var(input, mean, mask=mask, eps=eps)
    # Normalize the input
    inv_std = tl.math.rsqrt(var)
    normalized = (input - mean) * inv_std
    # Load gamma and beta
    gamma = tl.load(gamma_ptr + pid, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + pid, mask=mask, other=0.0)
    # Scale and shift
    output = normalized * gamma + beta
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def leaky_relu_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr, # Pointer to output tensor
    n_elements, # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    # Calculate the offset for the current program
    offset = pid * BLOCK_SIZE
    # Create a range of offsets within the block
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements

    # Load input values
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Apply leaky ReLU
    output = tl.where(input >= 0, input, input * 0.01)
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def matmul_add_kernel(
    a_ptr,  # Pointer to first input tensor
    b_ptr,  # Pointer to second input tensor
    out_ptr,  # Pointer to output tensor
    n_batch,  # Number of batches
    n_channels,  # Number of channels
    n_elements_per_channel,  # Number of elements per channel
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    # Calculate the offset for the current program
    offset = pid * BLOCK_SIZE
    # Create a range of offsets within the block
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements_per_channel

    # Load input values
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    # Perform the elementwise addition
    out = a + b
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_group_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
    """
    This function wraps the Triton kernel call for group normalization.
    """
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    # Prepare output tensor
    output = torch.empty_like(x)

    # Parameters
    n_batch = x.size(0)
    n_channels = x.size(1)
    n_elements_per_channel = x.size(2) * x.size(3)

    # Determine the number of blocks needed
    grid = lambda meta: ((n_channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    group_norm_kernel[grid](x, gamma, beta, output, n_batch, n_channels, n_elements_per_channel, 1e-5, BLOCK_SIZE=128)
    return output


def triton_leaky_relu(x: torch.Tensor):
    """
    This function wraps the Triton kernel call for leaky ReLU.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    output = torch.empty_like(x)

    # Determine the number of blocks needed
    grid = lambda meta: ((x.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    leaky_relu_kernel[grid](x, output, x.numel(), BLOCK_SIZE=128)
    return output


def triton_matmul_add(a: torch.Tensor, b: torch.Tensor):
    """
    This function wraps the Triton kernel call for matrix multiplication and element-wise addition.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    # Prepare output tensor
    output = torch.empty_like(a)

    # Parameters
    n_batch = a.size(0)
    n_channels = a.size(1)
    n_elements_per_channel = a.size(2) * a.size(3)

    # Determine the number of blocks needed
    grid = lambda meta: ((n_channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    matmul_add_kernel[grid](a, b, output, n_batch, n_channels, n_elements_per_channel, BLOCK_SIZE=128)
    return output


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        x = self.fc(x)
        x = triton_group_norm(x, self.gamma, self.beta)
        x = triton_leaky_relu(x)
        x = triton_matmul_add(x, x)
        return x