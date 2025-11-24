import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gelu_div_matmul_kernel(
    x_ptr,  # Pointer to input tensor
    w_ptr,  # Pointer to weight tensor
    out_ptr,  # Pointer to output tensor
    batch_size,
    input_size,
    output_size,
    divisor,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    # Compute the block start index
    block_start = pid * BLOCK_SIZE
    # Compute the offset for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to prevent out-of-bounds access
    mask = offsets < (batch_size * input_size)

    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Load weights
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)

    # Compute matrix multiplication
    acc = tl.dot(x, w)
    # Apply division
    acc = acc / divisor
    # Apply GELU approximation: 0.5 * x * (1 + erf(x / sqrt(2)))
    # Using Taylor approximation for GELU: 0.5 * x * (1 + tanh(sqrt(2 / pi) * x * (1 - 0.044715 * x * x)))
    x = acc
    x = x * (1.0 + tl.tanh(0.7978845608 * x * (1.0 - 0.044715 * x * x))) * 0.5
    # Store result
    tl.store(out_ptr + offsets, x, mask=mask)


def triton_gelu_div_matmul(x: torch.Tensor, weight: torch.Tensor, divisor: float):
    """
    Triton kernel for matrix multiplication, division, and GELU activation.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    # Output tensor with same shape as x
    out = torch.empty_like(x)

    # Compute grid size
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Define grid
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the kernel
    gelu_div_matmul_kernel[grid](x, weight, out, x.size(0), x.size(1), weight.size(0), divisor, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.divisor = divisor

    def forward(self, x):
        # Use Triton kernel for matrix multiplication, division, and GELU
        x = triton_gelu_div_matmul(x, self.weight, self.divisor)
        return x

    def _init_weights(self):
        # Initialize weights using Kaiming normal initialization
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')