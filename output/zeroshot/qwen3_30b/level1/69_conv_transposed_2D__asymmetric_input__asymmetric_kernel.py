import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_2d_kernel(
    x_ptr,      # Pointer to input tensor (batch, in_channels, h_in, w_in)
    w_ptr,      # Pointer to weight tensor (out_channels, in_channels, kh, kw)
    out_ptr,    # Pointer to output tensor (batch, out_channels, h_out, w_out)
    batch_size, # Number of batches
    in_channels, # Number of input channels
    out_channels, # Number of output channels
    h_in,       # Input height
    w_in,       # Input width
    h_out,      # Output height
    w_out,      # Output width
    kh,         # Kernel height
    kw,         # Kernel width
    stride_h,   # Stride height
    stride_w,   # Stride width
    pad_h,      # Padding height
    pad_w,      # Padding width
    dilation_h, # Dilation height
    dilation_w, # Dilation width
    output_padding_h, # Output padding height
    output_padding_w, # Output padding width
    BLOCK_SIZE: tl.constexpr,
):
    # Block index and thread indices
    pid_batch = tl.program_id(0)
    pid_outch = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Thread indices
    pid = pid_batch * (out_channels * h_out * w_out) + pid_outch * (h_out * w_out) + pid_h * w_out + pid_w

    # Shared memory for storing tiles of input and weights
    # We'll use shared memory for input tiles and weights
    x_shared = tl.load(tl.make_block_ptr(x_ptr, (batch_size, in_channels, h_in, w_in), (in_channels * h_in * w_in, h_in * w_in, w_in, 1), (pid_batch, 0, 0, 0), (1, in_channels, BLOCK_SIZE, BLOCK_SIZE), (0, 0, 0, 0), 'x'))
    w_shared = tl.load(tl.make_block_ptr(w_ptr, (out_channels, in_channels, kh, kw), (in_channels * kh * kw, kh * kw, kw, 1), (pid_outch, 0, 0, 0), (1, in_channels, BLOCK_SIZE, BLOCK_SIZE), (0, 0, 0, 0), 'x'))

    # Output buffer
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Loop over input channels and kernel size
    for c in range(0, in_channels, BLOCK_SIZE):
        for kh_idx in range(kh):
            for kw_idx in range(kw):
                # Compute output position
                h_out_idx = pid_h * stride_h - pad_h + kh_idx * dilation_h
                w_out_idx = pid_w * stride_w - pad_w + kw_idx * dilation_w

                # Check bounds for output indices
                if h_out_idx < 0 or h_out_idx >= h_out or w_out_idx < 0 or w_out_idx >= w_out:
                    continue

                # Load input value
                h_in_idx = h_out_idx // stride_h + (kh_idx * dilation_h - pad_h) // stride_h
                w_in_idx = w_out_idx // stride_w + (kw_idx * dilation_w - pad_w) // stride_w

                # Check input bounds
                if h_in_idx < 0 or h_in_idx >= h_in or w_in_idx < 0 or w_in_idx >= w_in:
                    continue

                # Load input and weight
                x_val = tl.load(x_ptr + (pid_batch * in_channels + c) * h_in * w_in + h_in_idx * w_in + w_in_idx)
                w_val = tl.load(w_ptr + (pid_outch * in_channels + c) * kh * kw + kh_idx * kw + kw_idx)

                # Accumulate
                acc += x_val * w_val

    # Store output
    out_offset = (pid_batch * out_channels + pid_outch) * h_out * w_out + pid_h * w_out + pid_w
    tl.store(out_ptr + out_offset, acc)


def triton_conv_transpose_2d(x: torch.Tensor, w: torch.Tensor, stride: tuple, padding: tuple, dilation: tuple, output_padding: tuple, groups: int = 1, bias: bool = False):
    """
    Performs transposed 2D convolution using Triton kernel.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    # Shape and parameters
    batch_size, in_channels, h_in, w_in = x.shape
    out_channels, _, kh, kw = w.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    out_pad_h, out_pad_w = output_padding

    # Compute output shape
    h_out = (h_in - 1) * stride_h - 2 * pad_h + kh + out_pad_h
    w_out = (w_in - 1) * stride_w - 2 * pad_w + kw + out_pad_w

    # Create output tensor
    out = torch.empty(batch_size, out_channels, h_out, w_out, device=x.device, dtype=x.dtype)

    # Grid setup
    grid = lambda meta: (batch_size, out_channels, (h_out + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], (w_out + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'])

    # Launch kernel
    conv_transpose_2d_kernel[grid](x, w, out, batch_size, in_channels, out_channels, h_in, w_in, h_out, w_out, kh, kw, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, out_pad_h, out_pad_w, BLOCK_SIZE=64)

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))

        # Optionally add bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton kernel for transposed convolution
        out = triton_conv_transpose_2d(
            x, self.weight, 
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            output_padding=self.output_padding,
            groups=self.groups,
            bias=self.bias is not None
        )

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.view(1, self.out_channels, 1, 1)

        return out