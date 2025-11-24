import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_gemm_relu_kernel(
    x_ptr,  # pointer to input matrix (batch_size, in_features)
    w_ptr,  # pointer to weight matrix (in_features, out_features)
    bias_ptr,  # pointer to bias (out_features)
    out_ptr,  # pointer to output matrix (batch_size, out_features)
    batch_size,  # number of samples
    in_features,  # number of input features
    out_features,  # number of output features
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    # Compute the batch index
    batch_idx = pid
    # Compute the offset for the current batch
    batch_offset = batch_idx * in_features
    # Compute the offset for the current block within the batch
    block_start = tl.program_id(1) * BLOCK_SIZE
    # Create a range of offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < in_features

    # Load weights
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    # Load input
    x = tl.load(x_ptr + batch_offset + offsets, mask=mask, other=0.0)
    # Compute matrix multiplication
    acc = tl.dot(x, w)
    # Load bias
    bias = tl.load(bias_ptr, mask=mask, other=0.0)
    # Add bias
    acc += bias
    # Apply ReLU
    acc = tl.maximum(acc, 0.0)
    # Store output
    tl.store(out_ptr + batch_offset + offsets, acc, mask=mask)


def triton_fused_gemm_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
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
    out = torch.empty((x.size(0), weight.size(1)), dtype=x.dtype, device=x.device)

    # Define block size (power of 2 for optimal performance)
    BLOCK_SIZE = 128

    # Calculate number of blocks needed for in_features
    num_blocks = (in_features + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    grid = (x.size(0), num_blocks)
    fused_gemm_relu_kernel[grid](
        x,
        weight,
        bias,
        out,
        x.size(0),
        in_features,
        weight.size(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        return triton_fused_gemm_relu(x, self.weight, self.bias)