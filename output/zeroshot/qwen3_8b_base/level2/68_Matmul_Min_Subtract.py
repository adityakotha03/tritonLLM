import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_min_sub_kernel(
    x_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    constant_ptr,  # Pointer to constant
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in output
    in_features,  # Number of input features
    out_features,  # Number of output features
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load weights and bias
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)

    # Compute linear transformation
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = tl.dot(x, weight) + bias

    # Load constant
    constant = tl.load(constant_ptr, mask=mask, other=0.0)

    # Compute min and subtract
    x = tl.minimum(x, constant)
    x = x - constant

    # Store result
    tl.store(out_ptr + offsets, x, mask=mask)


def triton_fused_linear_min_sub(x: torch.Tensor, in_features, out_features, constant):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty((x.size(0), out_features), device=x.device, dtype=x.dtype)

    # Number of elements in the tensor
    n_elements = out.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    fused_linear_min_sub_kernel[grid](
        x,  # Input tensor
        torch.nn.Linear(in_features, out_features).weight,  # Weight tensor
        torch.nn.Linear(in_features, out_features).bias,  # Bias tensor
        torch.tensor(constant, device=x.device),  # Constant
        out,  # Output tensor
        n_elements,
        in_features,
        out_features,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.constant = constant

    def forward(self, x):
        # Use the optimized Triton kernel instead of the original operations
        return triton_fused_linear_min_sub(x, self.in_features, self.out_features, self.constant)