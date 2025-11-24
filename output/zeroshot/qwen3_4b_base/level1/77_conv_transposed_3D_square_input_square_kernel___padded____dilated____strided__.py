import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    input_ptr,       # pointer to input (batch, in_channels, d, h, w)
    weight_ptr,      # pointer to weight (out_channels, in_channels, k, k, k)
    bias_ptr,        # pointer to bias (out_channels) - optional
    output_ptr,      # pointer to output (batch, out_channels, d_out, h_out, w_out)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output dimensions
    d_out = (input_ptr.shape[2] - 1) * stride + padding * 2 + (kernel_size - 1) * dilation
    h_out = (input_ptr.shape[3] - 1) * stride + padding * 2 + (kernel_size - 1) * dilation
    w_out = (input_ptr.shape[4] - 1) * stride + padding * 2 + (kernel_size - 1) * dilation

    # Define block and thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    d_idx = tl.program_id(2)
    h_idx = tl.program_id(3)
    w_idx = tl.program_id(4)

    # Compute the output position
    d_out_idx = d_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    h_out_idx = h_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    w_out_idx = w_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Compute input indices using reverse convolution mapping
    # For transposed convolution: output (d, h, w) -> input (d_in, h_in, w_in)
    d_in = (d_out_idx - padding) // stride
    h_in = (h_out_idx - padding) // stride
    w_in = (w_out_idx - padding) // stride

    # Apply dilation to kernel indices
    d_kernel = tl.arange(0, kernel_size)
    h_kernel = tl.arange(0, kernel_size)
    w_kernel = tl.arange(0, kernel_size)

    # Expand to full 3D kernel indices
    d_kernel = d_kernel + tl.arange(0, kernel_size) * dilation
    h_kernel = h_kernel + tl.arange(0, kernel_size) * dilation
    w_kernel = w_kernel + tl.arange(0, kernel_size) * dilation

    # Create 3D kernel index offsets
    d_kernel = d_kernel[:, None, None, None]
    h_kernel = h_kernel[None, :, None, None]
    w_kernel = w_kernel[None, None, :, None]

    # Compute input indices with dilation
    d_in_idx = d_in + d_kernel
    h_in_idx = h_in + h_kernel
    w_in_idx = w_in + w_kernel

    # Create mask to avoid out-of-bounds access
    d_mask = (d_in_idx >= 0) & (d_in_idx < input_ptr.shape[2])
    h_mask = (h_in_idx >= 0) & (h_in_idx < input_ptr.shape[3])
    w_mask = (w_in_idx >= 0) & (w_in_idx < input_ptr.shape[4])

    # Combine masks
    valid_mask = d_mask & h_mask & w_mask

    # Load input features
    input_features = tl.load(input_ptr + batch_idx * in_channels * input_ptr.shape[2] * input_ptr.shape[3] * input_ptr.shape[4] +
                             channel_idx * input_ptr.shape[2] * input_ptr.shape[3] * input_ptr.shape[4] +
                             d_in_idx * input_ptr.shape[3] * input_ptr.shape[4] +
                             h_in_idx * input_ptr.shape[4] +
                             w_in_idx,
                             mask=valid_mask, other=0.0)

    # Load weights
    weight = tl.load(weight_ptr + channel_idx * out_channels * kernel_size * kernel_size * kernel_size +
                     batch_idx * out_channels * kernel_size * kernel_size * kernel_size +
                     d_kernel * kernel_size * kernel_size +
                     h_kernel * kernel_size +
                     w_kernel,
                     mask=valid_mask, other=0.0)

    # Compute output for each valid position
    output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        d_in_val = d_in_idx[i]
        h_in_val = h_in_idx[i]
        w_in_val = w_in_idx[i]
        d_k = d_kernel[i]
        h_k = h_kernel[i]
        w_k = w_kernel[i]
        # Only compute if valid
        if d_mask[i] and h_mask[i] and w_mask[i]:
            # Compute the weighted sum
            output[i] = tl.sum(input_features[i] * weight[i], axis=0)

    # Store output
    output_ptr_offset = batch_idx * out_channels * d_out * h_out * w_out + \
                        channel_idx * d_out * h_out * w_out + \
                        d_out_idx * h_out * w_out + \
                        h_out_idx * w_out + \
                        w_out_idx
    tl.store(output_ptr + output_ptr_offset, output, mask=valid_mask)


@triton.jit
def conv_transpose3d_kernel_fused(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output dimensions
    d_out = (input_ptr.shape[2] - 1) * stride + padding * 2 + (kernel_size - 1) * dilation
    h_out = (input_ptr.shape[3] - 1) * stride + padding * 2 + (kernel_size - 1) * dilation
    w_out = (input_ptr.shape[4] - 1) * stride + padding * 2 + (kernel_size - 1) * dilation

    # Compute output position
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    d_out_idx = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    h_out_idx = tl.program_id(3) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    w_out_idx = tl.program_id(4) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Compute input indices
    d_in = (d_out_idx - padding) // stride
    h_in = (h_out_idx - padding) // stride
    w_in = (w_out_idx - padding) // stride

    # Apply dilation to kernel indices
    d_kernel = tl.arange(0, kernel_size)
    h_kernel = tl.arange(0, kernel_size)
    w_kernel = tl.arange(0, kernel_size)

    # Dilation offsets
    d_kernel = d_kernel + d_kernel * dilation
    h_kernel = h_kernel + h_kernel * dilation
    w_kernel = w_kernel + w_kernel * dilation

    # Compute valid input indices
    d_in_idx = d_in + d_kernel
    h_in_idx = h_in + h_kernel
    w_in_idx = w_in + w_kernel

    # Create masks
    d_mask = (d_in_idx >= 0) & (d_in_idx < input_ptr.shape[2])
    h_mask = (h_in_idx >= 0) & (h_in_idx < input_ptr.shape[3])
    w_mask = (w_in_idx >= 0) & (w_in_idx < input_ptr.shape[4])
    valid_mask = d_mask & h_mask & w_mask

    # Load input and weights
    input_val = tl.load(input_ptr + batch_idx * in_channels * input_ptr.shape[2] * input_ptr.shape[3] * input_ptr.shape[4] +
                        channel_idx * input_ptr.shape[2] * input_ptr.shape[3] * input_ptr.shape[4] +
                        d_in_idx * input_ptr.shape[3] * input_ptr.shape[4] +
                        h_in_idx * input_ptr.shape[4] +
                        w_in_idx,
                        mask=valid_mask, other=0.0)

    weight_val = tl.load(weight_ptr + channel_idx * out_channels * kernel_size * kernel_size * kernel_size +
                         d_kernel * kernel_size * kernel_size +
                         h_kernel * kernel_size +
                         w_kernel,
                         mask=valid_mask, other=0.0)

    # Compute output
    output = tl.sum(input_val * weight_val, axis=0)

    # Store output
    output_offset = batch_idx * out_channels * d_out * h_out * w_out + \
                    channel_idx * d_out * h_out * w_out + \
                    d_out_idx * h_out * w_out + \
                    h_out_idx * w_out + \
                    w_out_idx
    tl.store(output_ptr + output_offset, output, mask=valid_mask)


def triton_conv_transpose3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    kernel_size: int = 3,
) -> torch.Tensor:
    """
    Custom Triton kernel for 3D transposed convolution.
    """
    assert input.is_cuda and weight.is_cuda, "Inputs must be on CUDA device."
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, in_channels, d, h, w = input.shape
    out_channels, _, k, k, k = weight.shape

    # Compute output dimensions
    d_out = (d - 1) * stride + padding * 2 + (k - 1) * dilation
    h_out = (h - 1) * stride + padding * 2 + (k - 1) * dilation
    w_out = (w - 1) * stride + padding * 2 + (k - 1) * dilation

    # Output tensor
    output = torch.empty((batch_size, out_channels, d_out, h_out, w_out), dtype=input.dtype, device=input.device)

    # Define block size
    BLOCK_SIZE = 16  # Power of 2, optimized for small kernel and memory access

    # Grid dimensions
    grid = lambda meta: (
        (batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (out_channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (d_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (h_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (w_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    # Launch kernel
    conv_transpose3d_kernel_fused[
        grid
    ](
        input_ptr=input.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias.data_ptr() if bias is not None else None,
        output_ptr=output.data_ptr(),
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Define weight tensor (out_channels, in_channels, k, k, k)
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = torch.nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            kernel_size=self.kernel_size
        )