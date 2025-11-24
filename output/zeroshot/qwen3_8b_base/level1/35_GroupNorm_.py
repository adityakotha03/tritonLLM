import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(
    x_ptr,  # Pointer to input tensor
    gamma_ptr,  # Pointer to gamma (scale)
    beta_ptr,  # Pointer to beta (shift)
    out_ptr,  # Pointer to output tensor
    n_groups,  # Number of groups
    group_size,  # Number of features per group
    n_channels,  # Total number of channels
    n_elements,  # Total number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the group index and the position within the group
    pid = tl.program_id(0)
    group_id = pid // n_channels
    channel_id = pid % n_channels

    # Compute the offset within the group
    offset = tl.arange(0, BLOCK_SIZE)
    offset = offset + group_id * group_size * n_elements + channel_id * n_elements

    # Load input values
    x = tl.load(x_ptr + offset, mask=offset < n_elements, other=0.0)

    # Compute mean and variance
    mean = tl.sum(x) / BLOCK_SIZE
    var = tl.sum(tl.square(x - mean)) / BLOCK_SIZE

    # Normalize
    x_norm = (x - mean) / tl.sqrt(var + 1e-5)

    # Apply gamma and beta
    gamma = tl.load(gamma_ptr + channel_id, mask=channel_id < n_channels, other=1.0)
    beta = tl.load(beta_ptr + channel_id, mask=channel_id < n_channels, other=0.0)
    out = gamma * x_norm + beta

    # Store the result
    tl.store(out_ptr + offset, out, mask=offset < n_elements)


def triton_group_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
    """
    Applies Group Normalization using a custom Triton kernel.
    """
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Calculate parameters
    n_groups = x.size(1)
    group_size = x.size(1) // n_groups
    n_channels = x.size(1)
    n_elements = x.size(2) * x.size(3)

    # Determine the block size
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    num_blocks = (n_channels + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    grid = lambda meta: (num_blocks,)
    group_norm_kernel[grid](x, gamma, beta, out, n_groups, group_size, n_channels, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model with custom Triton kernel for Group Normalization.
    """
    def __init__(self, num_features: int, num_groups: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.group_size = num_features // num_groups

        # Initialize gamma and beta
        self.gamma = torch.nn.Parameter(torch.ones(num_features))
        self.beta = torch.nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply Group Normalization using custom Triton kernel
        return triton_group_norm(x, self.gamma, self.beta)