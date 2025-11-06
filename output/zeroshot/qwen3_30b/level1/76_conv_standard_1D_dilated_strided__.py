import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weight tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    length,  # Input length
    kernel_size,  # Kernel size
    stride,  # Stride
    dilation,  # Dilation
    output_length,  # Output length
    BLOCK_SIZE: tl.constexpr,
    TILE_OUT_CH: tl.constexpr,
    TILE_IN_CH: tl.constexpr,
    TILE_LENGTH: tl.constexpr,
):
    # Grid: one block per output channel group (TILE_OUT_CH), one thread per output element (BLOCK_SIZE)
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_x = tl.program_id(2)

    # Thread indices within block
    pid = pid_batch * (out_channels // TILE_OUT_CH) * (output_length // BLOCK_SIZE) + pid_out_ch * (output_length // BLOCK_SIZE) + pid_x

    # Offset for output tensor
    out_ch_start = pid_out_ch * TILE_OUT_CH
    out_ch_end = min(out_ch_start + TILE_OUT_CH, out_channels)
    x_start = pid_x * BLOCK_SIZE
    x_end = min(x_start + BLOCK_SIZE, output_length)

    # Compute the corresponding input range for this output position
    # For each output position, gather input features from a region of length kernel_size
    input_start = x_start * stride - (kernel_size - 1) * dilation
    input_end = input_start + kernel_size * dilation

    # Load weights: [out_channels, in_channels, kernel_size] -> [TILE_OUT_CH, TILE_IN_CH, kernel_size]
    w_ptrs = w_ptr + (out_ch_start * in_channels * kernel_size) + (tl.arange(0, TILE_IN_CH)[:, None] * kernel_size + tl.arange(0, kernel_size)[None, :])
    w = tl.load(w_ptrs, mask=(tl.arange(0, TILE_IN_CH)[:, None] < in_channels) & (tl.arange(0, kernel_size)[None, :] < kernel_size), other=0.0)

    # Output accumulator
    acc = tl.zeros((TILE_OUT_CH, BLOCK_SIZE), dtype=tl.float32)

    # Iterate over input channels and kernel positions
    for ic in range(0, in_channels, TILE_IN_CH):
        # Compute input channel slice
        ic_start = ic
        ic_end = min(ic + TILE_IN_CH, in_channels)

        # Load input data: [batch_size, in_channels, length] -> [batch_size, TILE_IN_CH, kernel_size]
        x_ptrs = x_ptr + (pid_batch * in_channels * length + ic_start * length + (input_start + tl.arange(0, kernel_size) * dilation)[None, :])
        x = tl.load(x_ptrs, mask=(tl.arange(0, TILE_IN_CH)[:, None] < in_channels - ic_start) & (tl.arange(0, kernel_size)[None, :] < kernel_size) & (input_start + tl.arange(0, kernel_size) * dilation + tl.arange(0, TILE_IN_CH)[:, None] < length), other=0.0)

        # Perform convolution: [TILE_OUT_CH, TILE_IN_CH] x [TILE_IN_CH, kernel_size] -> [TILE_OUT_CH, kernel_size]
        # Then convolve over kernel_size
        for k in range(kernel_size):
            x_k = x[:, k]
            w_k = w[:, k]
            acc += tl.dot(w_k, x_k, allow_tf32=True)

    # Store output: [TILE_OUT_CH, BLOCK_SIZE] -> [batch_size, out_channels, length]
    out_ptrs = out_ptr + (pid_batch * out_channels * output_length + out_ch_start * output_length + tl.arange(0, BLOCK_SIZE))
    out = acc[:, :x_end - x_start]
    tl.store(out_ptrs, out, mask=(tl.arange(0, TILE_OUT_CH)[:, None] < out_ch_end - out_ch_start) & (tl.arange(0, BLOCK_SIZE)[None, :] < x_end - x_start))


def triton_conv1d(x: torch.Tensor, w: torch.Tensor, stride: int, dilation: int, padding: int, out_channels: int, kernel_size: int, output_length: int) -> torch.Tensor:
    # Ensure inputs are on GPU and contiguous
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA"
    x = x.contiguous()
    w = w.contiguous()

    # Output tensor
    out = torch.empty(x.size(0), out_channels, output_length, dtype=x.dtype, device=x.device)

    # Parameters
    batch_size, in_channels, length = x.shape
    BLOCK_SIZE = 128
    TILE_OUT_CH = 32
    TILE_IN_CH = 32
    TILE_LENGTH = 128

    # Grid dimensions
    grid = lambda meta: (
        batch_size,
        (out_channels + meta["TILE_OUT_CH"] - 1) // meta["TILE_OUT_CH"],
        (output_length + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    # Launch kernel
    conv1d_kernel[grid](
        x, w, out,
        batch_size, in_channels, out_channels, length, kernel_size,
        stride, dilation, output_length,
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_OUT_CH=TILE_OUT_CH,
        TILE_IN_CH=TILE_IN_CH,
        TILE_LENGTH=TILE_LENGTH,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias

        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, device='cuda'))

        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, device='cuda'))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate output length
        length = x.size(2)
        output_length = (length + 2 * 0 - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        # Perform convolution using Triton kernel
        out = triton_conv1d(x, self.weight, self.stride, self.dilation, 0, self.out_channels, self.kernel_size, output_length)

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)

        return out