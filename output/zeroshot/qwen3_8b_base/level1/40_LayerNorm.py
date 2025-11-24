import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    input_ptr,  # Pointer to input tensor
    mean_ptr,   # Pointer to mean tensor
    var_ptr,    # Pointer to variance tensor
    gamma_ptr,  # Pointer to gamma (scale) tensor
    beta_ptr,   # Pointer to beta (shift) tensor
    output_ptr, # Pointer to output tensor
    n_elements, # Total number of elements in the input
    normalized_shape: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input data
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Compute mean and variance
    mean = tl.sum(input, axis=0) / n_elements
    var = tl.sum(tl.square(input - mean), axis=0) / n_elements

    # Store mean and variance
    tl.store(mean_ptr + offsets, mean, mask=mask)
    tl.store(var_ptr + offsets, var, mask=mask)

    # Compute normalized input
    inv_std = tl.rsqrt(var + 1e-5)
    normalized = (input - mean) * inv_std

    # Apply gamma and beta
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
    output = normalized * gamma + beta

    # Store output
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_layer_norm(x: torch.Tensor, normalized_shape, eps=1e-5):
    """
    Applies Layer Normalization using a custom Triton kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()

    # Compute mean and variance
    mean = torch.zeros(1, device=x.device)
    var = torch.zeros(1, device=x.device)

    # Compute gamma and beta (initialized to 1 and 0)
    gamma = torch.ones(normalized_shape, device=x.device)
    beta = torch.zeros(normalized_shape, device=x.device)

    # Output tensor
    output = torch.empty_like(x)

    # Number of elements in the input
    n_elements = x.numel()
    normalized_shape = normalized_shape[0]  # Assume normalized_shape is a tuple with one element

    # Determine the number of blocks needed
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    layer_norm_kernel[ num_blocks ](
        x,
        mean,
        var,
        gamma,
        beta,
        output,
        n_elements,
        normalized_shape,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_layer_norm(x, self.normalized_shape)