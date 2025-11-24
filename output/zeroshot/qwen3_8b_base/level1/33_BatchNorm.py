import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batch_norm_kernel(
    input_ptr,  # Pointer to input tensor
    gamma_ptr,  # Pointer to gamma (scale) parameter
    beta_ptr,   # Pointer to beta (shift) parameter
    mean_ptr,   # Pointer to running mean
    var_ptr,    # Pointer to running variance
    output_ptr, # Pointer to output tensor
    n_channels: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)

    # Compute mean and variance (for simplicity, using only the current block)
    mean = tl.sum(input, axis=0) / tl.sum(tl.ones_like(input), axis=0)
    var = tl.sum((input - mean) * (input - mean), axis=0) / tl.sum(tl.ones_like(input), axis=0)

    # Save running mean and variance (simplified for this example)
    tl.store(mean_ptr, mean)
    tl.store(var_ptr, var)

    # Normalize
    inv_std = tl.rsqrt(var + eps)
    normalized = (input - mean) * inv_std

    # Scale and shift
    output = normalized * gamma + beta

    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_batch_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor):
    """
    Applies Batch Normalization using a custom Triton kernel.
    """
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda and running_mean.is_cuda and running_var.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    running_mean = running_mean.contiguous()
    running_var = running_var.contiguous()

    # Output tensor
    output = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    n_channels = x.size(1)
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    batch_norm_kernel[grid](x, gamma, beta, running_mean, running_var, output, n_channels, n_elements, BLOCK_SIZE=BLOCK_SIZE, eps=1e-5)
    return output


class ModelNew(nn.Module):
    """
    Optimized model that performs Batch Normalization using a custom Triton kernel.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer with custom Triton kernel.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.register_buffer('gamma', torch.ones(num_features))
        self.register_buffer('beta', torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization using the custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        return triton_batch_norm(x, self.gamma, self.beta, self.running_mean, self.running_var)