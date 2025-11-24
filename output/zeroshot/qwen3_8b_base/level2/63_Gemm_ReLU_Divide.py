import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_matmul_relu_div_kernel(
    x_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    out_ptr,  # Pointer to output tensor
    bias_ptr,  # Pointer to bias tensor (optional)
    divisor,  # Divisor value
    n_elements,  # Total number of elements in output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to avoid out-of-bounds
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Load weight and bias
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    else:
        bias = 0.0

    # Compute matmul + bias
    matmul = tl.dot(x, weight)
    matmul = matmul + bias

    # Apply ReLU
    matmul = tl.maximum(matmul, 0.0)

    # Apply division
    matmul = matmul / divisor

    # Store output
    tl.store(out_ptr + offsets, matmul, mask=mask)


def triton_fused_matmul_relu_div(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, divisor: float):
    """
    Triton kernel for fused matmul + ReLU + division.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Output tensor
    out = torch.empty_like(x)

    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the kernel
    fused_matmul_relu_div_kernel[grid](x, weight, out, bias, divisor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using a custom Triton kernel for fused matmul, ReLU, and division.
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.divisor = divisor

    def forward(self, x):
        # Use the Triton kernel for matmul + ReLU + division
        x = triton_fused_matmul_relu_div(x, self.weight, self.bias, self.divisor)
        return x