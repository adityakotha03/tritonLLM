import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_transpose_kernel(
    x_ptr,  # Input pointer (batch, in_channels, length)
    w_ptr,  # Weight pointer (out_channels, in_channels, kernel_size)
    out_ptr,  # Output pointer (batch, out_channels, length_out)
    batch_size,  # Number of batches
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    length,  # Input sequence length
    length_out,  # Output sequence length
    kernel_size,  # Kernel size
    stride,  # Stride
    padding,  # Padding
    dilation,  # Dilation
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
):
    # Each program processes one block of output elements in a single channel
    pid_batch = tl.program_id(0)
    pid_out_channel = tl.program_id(1)
    pid_block = tl.program_id(2)

    # Compute output position: (batch, out_channel, output_idx)
    batch_offset = pid_batch * BLOCK_SIZE_OUT * length_out
    out_channel_offset = pid_out_channel * BLOCK_SIZE_OUT
    block_offset = pid_block * BLOCK_SIZE

    # Output indices in the sequence
    out_indices = block_offset + tl.arange(0, BLOCK_SIZE)
    out_mask = out_indices < length_out

    # Input indices: compute corresponding input indices from the output index
    # Formula: input_idx = out_idx * stride - padding + (kernel_size - 1) * dilation
    input_indices = out_indices * stride - padding
    # Adjust input indices to include dilation
    input_indices = input_indices[:, None] + tl.arange(0, kernel_size) * dilation  # Shape: (BLOCK_SIZE, kernel_size)
    input_indices = input_indices + tl.arange(0, in_channels)[:, None, None] * length * out_channels  # Channel offset
    input_indices = input_indices + (pid_batch * in_channels + pid_out_channel) * length  # Batch offset
    input_indices = input_indices + (tl.arange(0, length)[:, None, None] * in_channels * out_channels * batch_size)  # Full offset

    # Mask for valid input indices
    valid_input_mask = (input_indices >= 0) & (input_indices < in_channels * out_channels * batch_size * length)
    valid_input_mask = valid_input_mask & (input_indices < (batch_size * in_channels * length))

    # Load weights: (out_channels, in_channels, kernel_size) -> (in_channels, kernel_size)
    w_offset = pid_out_channel * in_channels * kernel_size
    w_ptrs = w_ptr + w_offset + tl.arange(0, in_channels)[:, None] * kernel_size + tl.arange(0, kernel_size)
    w_vals = tl.load(w_ptrs, mask=tl.arange(0, in_channels)[:, None] < in_channels, other=0.0)
    w_vals = tl.reshape(w_vals, (in_channels * kernel_size,))

    # Output accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over input channels and kernel size
    for k in range(kernel_size):
        for c in range(in_channels):
            # Compute input index
            idx = out_indices * stride - padding + k * dilation
            idx_mask = (idx >= 0) & (idx < length)

            # Load input data
            input_ptr = x_ptr + pid_batch * in_channels * length + c * length + idx
            x_vals = tl.load(input_ptr, mask=idx_mask, other=0.0)

            # Weight value
            w_val = w_vals[c * kernel_size + k]

            # Accumulate
            acc += x_vals * w_val

    # Store output
    out_ptrs = out_ptr + batch_offset + out_channel_offset + block_offset
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_conv1d_transpose(x: torch.Tensor, w: torch.Tensor, stride: int, padding: int, dilation: int, length_out: int) -> torch.Tensor:
    # Ensure input and weights are on GPU and contiguous
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    batch_size, in_channels, length = x.shape
    out_channels, _, kernel_size = w.shape

    # Create output tensor
    out = torch.empty(batch_size, out_channels, length_out, dtype=x.dtype, device=x.device)

    # Set up kernel launch parameters
    BLOCK_SIZE = 128  # Tuneable, power of 2
    BLOCK_SIZE_OUT = 32  # Number of output elements per block

    # Grid: (batch_size, out_channels, ceil(length_out / BLOCK_SIZE))
    grid = lambda meta: (
        batch_size,
        out_channels,
        (length_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    )

    # Launch Triton kernel
    conv1d_transpose_kernel[grid](
        x,
        w,
        out,
        batch_size,
        in_channels,
        out_channels,
        length,
        length_out,
        kernel_size,
        stride,
        padding,
        dilation,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_SIZE_OUT=BLOCK_SIZE_OUT
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        # Use learnable weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # Store hyperparameters
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output length
        length_out = (x.shape[2] - 1) * self.stride - 2 * self.padding + self.kernel_size * self.dilation
        length_out = (length_out + self.dilation - 1) // self.dilation

        # Run custom Triton kernel
        out = triton_conv1d_transpose(
            x, self.weight, self.stride, self.padding, self.dilation, length_out
        )

        # Add bias if needed
        if self.bias is not None:
            out = out + self.bias[None, :, None]

        return out