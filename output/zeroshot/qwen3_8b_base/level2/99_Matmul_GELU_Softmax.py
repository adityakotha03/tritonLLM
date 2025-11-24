import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gelu_softmax_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Apply GELU approximation: x * (1 + tanh(sqrt(2 / pi) * (x + 0.044713 * x**3)))
    x_squared = x * x
    x_cubed = x * x_squared
    temp = x + 0.044713 * x_cubed
    temp = tl.sqrt(tl.constant(2.0 / tl.constant(3.141592653589793))) * temp
    temp = tl.tanh(temp)
    gelu = x * (1.0 + temp)
    
    # Apply Softmax (approximate with log-sum-exp)
    max_val = tl.max(gelu, axis=0)
    exp_gelu = tl.exp(gelu - max_val)
    sum_exp = tl.sum(exp_gelu, axis=0)
    softmax = exp_gelu / sum_exp

    # Store output
    tl.store(out_ptr + offsets, softmax, mask=mask)


def triton_gelu_softmax(x: torch.Tensor):
    """
    Custom Triton kernel for GELU + Softmax.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Output tensor
    out = torch.empty_like(x)

    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter

    # Determine grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    gelu_softmax_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = triton_gelu_softmax(x)
        return x