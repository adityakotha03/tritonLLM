import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    input_ptr,  # Pointer to input tensor
    gamma_ptr,  # Pointer to gamma (scale)
    beta_ptr,   # Pointer to beta (shift)
    output_ptr, # Pointer to output tensor
    batch_size: tl.constexpr,
    num_features: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID (block ID)
    pid = tl.program_id(0)
    # Compute the block's starting index
    block_start = pid * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < (height * width)

    # Compute the 2D index for the current block
    # For each offset, compute (i, j) = (offset // width, offset % width)
    i = (offsets // width)
    j = (offsets % width)

    # Load input values
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Load gamma and beta (broadcasted across the feature dimension)
    gamma = tl.load(gamma_ptr + tl.arange(0, num_features), mask=tl.arange(0, num_features) < num_features, other=1.0)
    beta = tl.load(beta_ptr + tl.arange(0, num_features), mask=tl.arange(0, num_features) < num_features, other=0.0)

    # Compute mean and variance for the current feature map
    # For each feature map (batch, feature, i, j)
    # We need to compute mean over (i, j) for each feature
    # We'll use the same block size for all features
    # So for each feature, we process the same block of (i, j)

    # Compute mean across (i, j)
    mean = tl.sum(input_val, axis=0) / tl.numel(input_val)

    # Compute variance across (i, j)
    var = tl.sum((input_val - mean) * (input_val - mean), axis=0) / tl.numel(input_val)

    # Normalize
    normalized = (input_val - mean) * tl.rsqrt(var + 1e-5)

    # Apply gamma and beta
    output = normalized * gamma + beta

    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_instance_norm(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and gamma.is_cuda and beta.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    # Prepare output tensor
    output = torch.empty_like(input)

    # Get dimensions
    batch_size = input.size(0)
    num_features = input.size(1)
    height = input.size(2)
    width = input.size(3)

    # Choose block size (power of 2)
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    num_blocks = (height * width + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    grid = lambda meta: (num_blocks,)
    instance_norm_kernel[grid](input, gamma, beta, output, batch_size, num_features, height, width, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        # Initialize gamma and beta for instance normalization
        self.register_buffer('gamma', torch.ones(num_features))
        self.register_buffer('beta', torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply instance normalization using the custom Triton kernel
        return triton_instance_norm(x, self.gamma, self.beta)