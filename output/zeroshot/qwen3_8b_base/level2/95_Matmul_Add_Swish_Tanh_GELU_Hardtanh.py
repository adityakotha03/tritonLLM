import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_activation_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Swish: x * sigmoid(x)
    sig_x = tl.sigmoid(x)
    swish_x = x * sig_x
    # Tanh
    tanh_swish_x = tl.tanh(swish_x)
    # GELU
    gelu_tanh_x = 0.5 * tanh_swish_x * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    # Hardtanh
    hardtanh_gelu_x = tl.where(gelu_tanh_x > 1.0, 1.0, tl.where(gelu_tanh_x < -1.0, -1.0, gelu_tanh_x))
    # Store the result
    tl.store(out_ptr + offsets, hardtanh_gelu_x, mask=mask)


def triton_fused_activation(x: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the input is contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    fused_activation_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication, adds a value, and applies a fused activation function using a custom Triton kernel.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape)) 

    def forward(self, x):
        x = self.matmul(x)
        x = x + self.add_value
        x = triton_fused_activation(x)
        return x