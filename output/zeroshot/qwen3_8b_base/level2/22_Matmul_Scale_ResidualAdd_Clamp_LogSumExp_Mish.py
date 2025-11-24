import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to first input (batch, input_size)
    b_ptr,  # Pointer to second input (input_size, hidden_size)
    out_ptr,  # Pointer to output (batch, hidden_size)
    batch_size,  # Number of batches
    input_size,  # Size of input dimension
    hidden_size,  # Size of hidden dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Compute the row index in the output
    row_idx = pid * BLOCK_SIZE
    # Create a range of offsets for the row
    offsets = tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < hidden_size
    # Load the row from a
    a = tl.load(a_ptr + row_idx + offsets, mask=mask, other=0.0)
    # Load the column from b
    col_idx = tl.arange(0, BLOCK_SIZE)
    mask = col_idx < input_size
    b = tl.load(b_ptr + col_idx, mask=mask, other=0.0)
    # Compute the dot product
    out = tl.dot(a, b)
    # Store the result
    tl.store(out_ptr + row_idx + offsets, out, mask=mask)


@triton.jit
def scale_add_kernel(
    x_ptr,  # Pointer to input (batch, hidden_size)
    out_ptr,  # Pointer to output (batch, hidden_size)
    scale_factor,  # Scale factor
    batch_size,  # Number of batches
    hidden_size,  # Size of hidden dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Compute the row index in the output
    row_idx = pid * BLOCK_SIZE
    # Create a range of offsets for the row
    offsets = tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < hidden_size
    # Load the row from x
    x = tl.load(x_ptr + row_idx + offsets, mask=mask, other=0.0)
    # Scale and add
    out = x * scale_factor
    # Store the result
    tl.store(out_ptr + row_idx + offsets, out, mask=mask)


@triton.jit
def clamp_kernel(
    x_ptr,  # Pointer to input (batch, hidden_size)
    out_ptr,  # Pointer to output (batch, hidden_size)
    clamp_min,  # Minimum value
    clamp_max,  # Maximum value
    batch_size,  # Number of batches
    hidden_size,  # Size of hidden dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Compute the row index in the output
    row_idx = pid * BLOCK_SIZE
    # Create a range of offsets for the row
    offsets = tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < hidden_size
    # Load the row from x
    x = tl.load(x_ptr + row_idx + offsets, mask=mask, other=0.0)
    # Clamp the values
    out = tl.where(x < clamp_min, clamp_min, x)
    out = tl.where(out > clamp_max, clamp_max, out)
    # Store the result
    tl.store(out_ptr + row_idx + offsets, out, mask=mask)


@triton.jit
def logsumexp_kernel(
    x_ptr,  # Pointer to input (batch, hidden_size)
    out_ptr,  # Pointer to output (batch, hidden_size)
    batch_size,  # Number of batches
    hidden_size,  # Size of hidden dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Compute the row index in the output
    row_idx = pid * BLOCK_SIZE
    # Create a range of offsets for the row
    offsets = tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < hidden_size
    # Load the row from x
    x = tl.load(x_ptr + row_idx + offsets, mask=mask, other=0.0)
    # Compute logsumexp
    max_val = tl.max(x, mask=mask)
    out = x - max_val
    out = tl.exp(out)
    out = tl.sum(out, mask=mask)
    out = tl.log(out)
    out = out + max_val
    # Store the result
    tl.store(out_ptr + row_idx + offsets, out, mask=mask)


@triton.jit
def mish_kernel(
    x_ptr,  # Pointer to input (batch, hidden_size)
    out_ptr,  # Pointer to output (batch, hidden_size)
    batch_size,  # Number of batches
    hidden_size,  # Size of hidden dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Compute the row index in the output
    row_idx = pid * BLOCK_SIZE
    # Create a range of offsets for the row
    offsets = tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < hidden_size
    # Load the row from x
    x = tl.load(x_ptr + row_idx + offsets, mask=mask, other=0.0)
    # Compute Mish activation
    out = x * tl.tanh(tl.nn.functional.softplus(x))
    # Store the result
    tl.store(out_ptr + row_idx + offsets, out, mask=mask)


def matmul(x: torch.Tensor, weight: torch.Tensor):
    """
    Custom Triton kernel for matrix multiplication.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty((x.size(0), weight.size(1)), device=x.device, dtype=x.dtype)
    n_blocks = (x.size(1) + 128 - 1) // 128
    matmul_kernel[triton.make_kernel(n_blocks)](x, weight, out, x.size(0), x.size(1), weight.size(1), BLOCK_SIZE=128)
    return out


def scale_add(x: torch.Tensor, scale_factor: float):
    """
    Custom Triton kernel for scaling and adding.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_blocks = (x.size(1) + 128 - 1) // 128
    scale_add_kernel[triton.make_kernel(n_blocks)](x, out, scale_factor, x.size(0), x.size(1), BLOCK_SIZE=128)
    return out


def clamp(x: torch.Tensor, clamp_min: float, clamp_max: float):
    """
    Custom Triton kernel for clamping.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_blocks = (x.size(1) + 128 - 1) // 128
    clamp_kernel[triton.make_kernel(n_blocks)](x, out, clamp_min, clamp_max, x.size(0), x.size(1), BLOCK_SIZE=128)
    return out


def logsumexp(x: torch.Tensor):
    """
    Custom Triton kernel for logsumexp.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_blocks = (x.size(1) + 128 - 1) // 128
    logsumexp_kernel[triton.make_kernel(n_blocks)](x, out, x.size(0), x.size(1), BLOCK_SIZE=128)
    return out


def mish(x: torch.Tensor):
    """
    Custom Triton kernel for Mish activation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_blocks = (x.size(1) + 128 - 1) // 128
    mish_kernel[triton.make_kernel(n_blocks)](x, out, x.size(0), x.size(1), BLOCK_SIZE=128)
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        # Custom matrix multiplication
        weight = torch.randn(self.input_size, self.hidden_size, device=x.device, dtype=x.dtype)
        x = matmul(x, weight)
        # Scale and add
        x = scale_add(x, self.scale_factor)
        # Clamp
        x = clamp(x, self.clamp_min, self.clamp_max)
        # LogSumExp
        x = logsumexp(x)
        # Mish activation
        x = mish(x)
        return x