import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    input_ptr,      # pointer to input tensor (batch, in_channels, H, W)
    weight_ptr,     # pointer to weight tensor (in_channels, out_channels, kernel_size, kernel_size)
    output_ptr,     # pointer to output tensor (batch, out_channels, H_out, W_out)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute block and thread indices
    pid = tl.program_id(0)
    block_id = pid // (H_out * W_out)
    h_idx = (pid % (H_out * W_out)) // W_out
    w_idx = pid % W_out

    # Compute the output position
    h_out = h_idx
    w_out = w_idx

    # Compute the input positions (with padding and stride)
    h_start = h_out * stride
    w_start = w_out * stride
    h_end = h_start + kernel_size
    w_end = w_start + kernel_size

    # Adjust for padding (we use zero padding)
    h_start = max(0, h_start - padding)
    w_start = max(0, w_start - padding)
    h_end = min(H_in, h_end + padding)
    w_end = min(W_in, w_end + padding)

    # Compute the range of input indices that contribute to this output
    h_range = tl.arange(0, kernel_size)
    w_range = tl.arange(0, kernel_size)

    # Compute input indices using offset
    h_input = h_start + h_range
    w_input = w_start + w_range

    # Clip input indices to valid range
    h_input = tl.clip(h_input, 0, H_in - 1)
    w_input = tl.clip(w_input, 0, W_in - 1)

    # Create mask for valid input positions
    h_mask = (h_input >= 0) & (h_input < H_in)
    w_mask = (w_input >= 0) & (w_input < W_in)
    valid_mask = h_mask & w_mask

    # Load input and weights
    # Input: (batch, in_channels, H_in, W_in)
    # We loop over in_channels and compute per channel
    input_offsets = (block_id * in_channels + tl.arange(0, in_channels)) * (H_in * W_in)
    input_idx = input_offsets + (h_input * W_in + w_input)

    # Load input values
    input_vals = tl.load(input_ptr + input_idx, mask=valid_mask, other=0.0)

    # Load weights: (in_channels, out_channels, k, k)
    # For each output channel, we load weights per input channel
    weight_offsets = (tl.arange(0, in_channels) * (out_channels * kernel_size * kernel_size) +
                      tl.arange(0, out_channels) * (kernel_size * kernel_size) +
                      h_range * kernel_size + w_range)
    weight_vals = tl.load(weight_ptr + weight_offsets, mask=valid_mask, other=0.0)

    # Compute output per channel
    output_vals = tl.zeros((out_channels,), dtype=tl.float32)
    for i in range(in_channels):
        # For each input channel, compute contribution to each output channel
        channel_weight = weight_vals[i]
        channel_input = input_vals[i]
        output_vals += channel_input * channel_weight

    # Store output
    output_offset = block_id * out_channels + tl.arange(0, out_channels)
    output_idx = output_offset + h_out * W_out + w_out
    tl.store(output_ptr + output_idx, output_vals, mask=tl.arange(0, out_channels) < out_channels)


def triton_depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    kernel_size: int = 3,
    out_channels: int = 128,
    in_channels: int = 128,
    batch_size: int = 64,
    height_in: int = 256,
    width_in: int = 512,
) -> torch.Tensor:
    """
    Custom Triton kernel for depthwise 2D convolution.
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA device."
    x = x.contiguous()
    weight = weight.contiguous()

    # Compute output dimensions
    H_out = (height_in + 2 * padding - kernel_size) // stride + 1
    W_out = (width_in + 2 * padding - kernel_size) // stride + 1

    # Ensure dimensions match
    assert x.shape[1] == in_channels, f"Input channels {x.shape[1]} must match in_channels {in_channels}"
    assert weight.shape[0] == in_channels, f"Weight input channels {weight.shape[0]} must match in_channels {in_channels}"
    assert weight.shape[1] == out_channels, f"Weight output channels {weight.shape[1]} must match out_channels {out_channels}"
    assert weight.shape[2] == kernel_size, f"Weight kernel size {weight.shape[2]} must match kernel_size {kernel_size}"
    assert weight.shape[3] == kernel_size, f"Weight kernel size {weight.shape[3]} must match kernel_size {kernel_size}"

    # Prepare output tensor
    out = torch.empty(
        (batch_size, out_channels, H_out, W_out),
        dtype=x.dtype,
        device=x.device
    )

    # Define block size (power of 2 for optimal performance)
    BLOCK_SIZE = 128

    # Grid: number of blocks needed
    grid = lambda meta: (
        (batch_size * out_channels * H_out * W_out + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )

    # Launch kernel
    depthwise_conv2d_kernel[
        grid
    ](
        x_ptr=x.data_ptr(),
        weight_ptr=weight.data_ptr(),
        output_ptr=out.data_ptr(),
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        H_in=height_in,
        W_in=width_in,
        H_out=H_out,
        W_out=W_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # We create a weight tensor manually (in_channels, out_channels, k, k)
        self.weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, dtype=torch.float16, device='cuda')
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs depthwise 2D convolution using custom Triton kernel.
        """
        return triton_depthwise_conv2d(
            x=x,
            weight=self.weight,
            stride=self.stride,
            padding=self.padding,
            kernel_size=self.kernel_size,
            out_channels=self.weight.shape[1],
            in_channels=self.weight.shape[0],
            batch_size=x.shape[0],
            height_in=x.shape[2],
            width_in=x.shape[3],
        )