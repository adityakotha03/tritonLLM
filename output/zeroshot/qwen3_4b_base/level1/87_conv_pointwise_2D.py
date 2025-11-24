import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    input_ptr,  # Pointer to input tensor (batch, in_channels, H, W)
    weight_ptr,  # Pointer to weight tensor (out_channels, in_channels, 1, 1)
    bias_ptr,    # Pointer to bias tensor (out_channels) - optional
    output_ptr,  # Pointer to output tensor (batch, out_channels, H, W)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes a block of output elements
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)

    # Load the output channel weight and bias (if bias exists)
    w = tl.load(weight_ptr + (out_channel_idx * in_channels * 1 * 1), mask=tl.arange(0, in_channels) < in_channels, other=0.0)
    if bias_ptr is not None:
        b = tl.load(bias_ptr + out_channel_idx, mask=tl.arange(0, 1) < 1, other=0.0)
    else:
        b = 0.0

    # Compute the output for each position in the spatial dimensions
    # We process each output position (h, w) in a block
    h_start = tl.program_id(2) * BLOCK_SIZE
    w_start = tl.program_id(3) * BLOCK_SIZE

    # Create offsets for spatial dimensions
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE)
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE)

    # Ensure we stay within bounds
    h_mask = h_offsets < height
    w_mask = w_offsets < width

    # Broadcast batch and channel indices
    batch_idx = tl.full((BLOCK_SIZE, BLOCK_SIZE), batch_idx, dtype=tl.int32)
    in_channel_idx = tl.arange(0, in_channels)

    # Load input data for current batch and spatial positions
    # input: (batch, in_channels, H, W)
    # We load input in a tiled fashion across in_channels and spatial dims
    # For each in_channel, we load the entire input slice
    input_offsets = (batch_idx, in_channel_idx, h_offsets, w_offsets)
    input_mask = (h_offsets < height) & (w_offsets < width) & (in_channel_idx < in_channels)

    # Load input values
    input_vals = tl.load(
        input_ptr + (batch_idx * in_channels * height * width) +
        (in_channel_idx * height * width) +
        (h_offsets * width) +
        w_offsets,
        mask=input_mask,
        other=0.0
    )

    # Perform element-wise multiplication with weights and sum over in_channels
    # Weight is (out_channels, in_channels, 1, 1), so we use it directly
    # We compute: output[h, w] = sum(in_channels) input[h, w] * weight[in_channel]
    # Since kernel_size=1, no spatial convolution, just pointwise
    # We compute sum over in_channels
    out_val = tl.sum(input_vals * w, axis=1)  # sum over in_channels
    out_val = out_val + b

    # Store result at output position
    output_offsets = (batch_idx, out_channel_idx, h_offsets, w_offsets)
    output_mask = (h_offsets < height) & (w_offsets < width)
    tl.store(
        output_ptr + (batch_idx * out_channels * height * width) +
        (out_channel_idx * height * width) +
        (h_offsets * width) +
        w_offsets,
        out_val,
        mask=output_mask
    )


def triton_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None
):
    """
    Custom Triton kernel for pointwise 2D convolution (kernel_size=1).
    Replaces nn.Conv2d with a fused, optimized kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels = weight.shape[0]

    # Define kernel parameters
    BLOCK_SIZE = 128  # Optimal for Ampere, power of 2

    # Grid dimensions: (batch, out_channel, h_block, w_block)
    grid = lambda meta: (
        (batch_size, out_channels, (height + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], (width + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"])
    )

    # Launch kernel
    conv1d_kernel[
        grid
    ](
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr() if bias is not None else None,
        torch.empty_like(x).data_ptr(),
        batch_size, in_channels, out_channels, height, width, BLOCK_SIZE=BLOCK_SIZE
    )
    return x.new_empty(batch_size, out_channels, height, width)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights and bias
        self.weight = torch.randn(out_channels, in_channels, 1, 1, device='cuda', dtype=torch.float16)
        self.bias = torch.randn(out_channels, device='cuda', dtype=torch.float16) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs pointwise 2D convolution using custom Triton kernel.
        """
        return triton_conv1d(x, self.weight, self.bias)