import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_add_kernel(
    a_ptr,  # Pointer to first input (batch, in_features)
    b_ptr,  # Pointer to second input (in_features, out_features)
    out_ptr,  # Pointer to output (batch, out_features)
    batch_size,  # Number of samples in the batch
    in_features,  # Number of input features
    out_features,  # Number of output features
    scaling_factor,  # Scaling factor
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < out_features

    # Compute the row index in the batch
    row_idx = pid
    # Compute the offset in the output tensor
    out_offset = row_idx * out_features + offsets

    # Load the row from a (batch, in_features)
    a_row = tl.load(a_ptr + row_idx * in_features + offsets, mask=mask, other=0.0)

    # Load the column from b (in_features, out_features)
    b_col = tl.load(b_ptr + offsets * in_features, mask=mask, other=0.0)

    # Compute the matrix multiplication
    matmul = tl.dot(a_row, b_col)

    # Apply scaling and add original x
    scaled = matmul * scaling_factor
    original = tl.load(out_ptr + row_idx * out_features + offsets, mask=mask, other=0.0)
    out = scaled + original

    # Store the result
    tl.store(out_ptr + out_offset, out, mask=mask)


def triton_matmul_add(a: torch.Tensor, b: torch.Tensor, scaling_factor: float, out: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert a.is_cuda and b.is_cuda and out.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()
    out = out.contiguous()

    # Number of elements in the tensor
    batch_size = a.size(0)
    in_features = a.size(1)
    out_features = b.size(1)

    # Determine the number of blocks needed
    grid = lambda meta: (batch_size,)

    # Launch the Triton kernel
    matmul_add_kernel[grid](a, b, out, batch_size, in_features, out_features, scaling_factor, BLOCK_SIZE=1024)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Initialize output tensor
        out = torch.empty_like(x)
        # Perform matrix multiplication, scaling, and residual addition using Triton kernel
        triton_matmul_add(x, torch.randn(self.in_features, self.out_features).cuda(), self.scaling_factor, out)
        return out