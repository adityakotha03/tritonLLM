import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_swish_add_groupnorm_kernel(
    x_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    output_ptr,  # Pointer to output tensor
    group_norm_output_ptr,  # Pointer to group norm output tensor
    batch_size,  # Batch size
    in_features,  # Input features
    out_features,  # Output features
    num_groups,  # Number of groups for group norm
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block offset
    block_offset = pid * BLOCK_SIZE
    # Create a range of offsets for the current block
    offsets = block_offset + tl.arange(0, BLOCK_SIZE)
    # Mask to avoid out-of-bounds access
    mask = offsets < (batch_size * in_features)

    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute matmul (x @ weight)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    x = x * weight
    # Sum over features
    x = tl.sum(x, axis=1)
    # Apply Swish activation: x * sigmoid(x)
    x = x * tl.sigmoid(x)
    # Add bias
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    x = x + bias

    # Compute group norm
    # Reshape x to (batch_size, out_features)
    x = x.reshape((batch_size, out_features))
    # Compute mean and variance per group
    group_size = out_features // num_groups
    for g in range(num_groups):
        group_start = g * group_size
        group_end = (g + 1) * group_size
        group_mask = (offsets >= group_start) & (offsets < group_end)
        group_x = x[group_mask]
        mean = tl.mean(group_x)
        var = tl.var(group_x)
        # Normalize
        x[group_mask] = (group_x - mean) * tl.rsqrt(var + 1e-5)
    # Reshape back
    x = x.reshape((batch_size * in_features,))

    # Store the result
    tl.store(group_norm_output_ptr + offsets, x, mask=mask)


def triton_matmul_swish_add_groupnorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, out_features: int, num_groups: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Prepare output tensor
    output = torch.empty_like(x)
    group_norm_output = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    matmul_swish_add_groupnorm_kernel[grid](x, weight, bias, output, group_norm_output, x.shape[0], x.shape[1], out_features, num_groups, BLOCK_SIZE=BLOCK_SIZE)
    return group_norm_output


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(*bias_shape))
        self.out_features = out_features
        self.num_groups = num_groups

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Call the Triton kernel
        return triton_matmul_swish_add_groupnorm(x, self.weight, self.bias, self.out_features, self.num_groups)