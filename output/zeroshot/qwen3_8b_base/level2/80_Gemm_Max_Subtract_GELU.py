import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_max_sub_mean_gelu_kernel(
    x_ptr, y_ptr, out_ptr,
    batch_size, in_features, out_features,
    max_dim, BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    # Compute the block start and end indices
    block_start = pid * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE
    # Compute the offset for the current block
    offset = tl.arange(0, BLOCK_SIZE)
    # Compute the indices for the current block
    indices = offset + block_start
    # Compute the batch index
    batch_idx = indices // (in_features)
    # Compute the feature index
    feature_idx = indices % (in_features)
    # Load the values from x
    x = tl.load(x_ptr + indices, mask=indices < batch_size * in_features, other=0.0)
    # Compute the max along the max_dim dimension
    max_val = tl.max(x, axis=max_dim)
    # Compute the mean along the dim=1 dimension
    mean_val = tl.mean(x, axis=1)
    # Subtract the mean from the max
    x = max_val - mean_val
    # Apply GELU activation
    x = x * tl.where(x > 0, 1.0, 2.0 / (tl.math.sqrt(2 * tl.math.pi) + 1e-5) * (x + 0.010752 * x**3))
    # Store the result
    tl.store(out_ptr + indices, x, mask=indices < batch_size * in_features)


def triton_gemm_max_sub_mean_gelu(x: torch.Tensor, in_features, out_features, max_dim):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
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
    gemm_max_sub_mean_gelu_kernel[grid](x, x, out, x.size(0), in_features, out_features, max_dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_dim = max_dim

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        x = triton_gemm_max_sub_mean_gelu(x, self.in_features, self.out_features, self.max_dim)
        return x