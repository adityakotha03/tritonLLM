import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_swish_clamp_tanh_kernel(
    x_ptr,  # Pointer to input tensor
    w_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    out_ptr,  # Pointer to output tensor
    batch_size,  # Number of batches
    in_features,  # Input features per sample
    out_features,  # Output features per sample
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    # Get the block index within the batch
    block_idx = tl.program_id(1)
    # Compute the offset for the current block
    offset = batch_idx * in_features + block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offset < batch_idx * in_features + in_features

    # Load weights and bias
    weights = tl.load(w_ptr + block_idx * in_features * out_features + tl.arange(0, in_features)[:, None] * out_features + tl.arange(0, out_features), mask=mask, other=0.0)
    bias = tl.load(bias_ptr + block_idx * out_features + tl.arange(0, out_features), mask=mask, other=0.0)

    # Compute matrix multiplication
    x = tl.load(x_ptr + batch_idx * in_features + offset, mask=mask, other=0.0)
    matmul = tl.dot(x, weights)
    matmul = matmul + bias

    # Apply Swish activation
    swish = matmul * tl.sigmoid(matmul)

    # Apply divide by 2.0
    swish = swish / 2.0

    # First clamp
    clamped = tl.where(swish < -1.0, -1.0, tl.where(swish > 1.0, 1.0, swish))

    # Apply tanh
    tanh = tl.tanh(clamped)

    # Second clamp (redundant, but included as per original model)
    final = tl.where(tanh < -1.0, -1.0, tl.where(tanh > 1.0, 1.0, tanh))

    # Store the result
    tl.store(out_ptr + batch_idx * out_features + block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), final, mask=mask)


def triton_gemm_swish_clamp_tanh(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
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
    out = torch.empty((x.size(0), weight.size(0)), dtype=x.dtype, device=x.device)

    # Determine the number of blocks needed
    BLOCK_SIZE = 128
    num_blocks = (x.size(1) + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_batches = x.size(0)

    # Define the grid
    grid = (num_batches, num_blocks)

    # Launch the Triton kernel
    gemm_swish_clamp_tanh_kernel[grid](x, weight, bias, out, x.size(0), x.size(1), weight.size(0), BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return triton_gemm_swish_clamp_tanh(x, self.weight, self.bias)