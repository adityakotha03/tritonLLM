import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv_kernel(
    X_ptr,          # input tensor pointer (float32)
    W_ptr,          # weight tensor pointer (float32)
    B_ptr,          # bias tensor pointer (float32) or None
    out_ptr,        # output tensor pointer (float32)
    stride: tl.constexpr,
    padding: tl.constexpr,
    ks: tl.constexpr,              # kernel size
    n_channels: tl.constexpr,
    H_in: tl.constexpr,            # input height
    W_in: tl.constexpr,            # input width
    H_out: tl.constexpr,           # output height
    W_out: tl.constexpr,           # output width
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one output element
    program_id = tl.program_id(0)
    total_out = n_channels * H_out * W_out
    if program_id >= total_out:
        return

    # Decode channel and spatial index
    out_idx = program_id
    channel = out_idx // (H_out * W_out)
    idx = out_idx % (H_out * W_out)
    h_out = idx // W_out
    w_out = idx % W_out

    # Compute input top-left corner
    h_in = h_out * stride - padding
    w_in = w_out * stride - padding

    # Stride offsets
    stride_in = H_in * W_in
    stride_ch = stride_in * n_channels

    # Load bias if present
    bias_val = tl.zeros([1], dtype=tl.float32)
    if B_ptr is not None:
        bias_val = tl.load(B_ptr + channel)

    # Accumulate convolution
    acc = bias_val
    for kh in range(ks):
        for kw in range(ks):
            # Compute global input coordinates
            h = h_in + kh
            w = w_in + kw

            # Check bounds
            if h < 0 or h >= H_in or w < 0 or w >= W_in:
                continue

            # Load input
            in_idx = channel * stride_ch + h * W_in + w
            inp = tl.load(X_ptr + in_idx)

            # Load weight
            w_idx = channel * ks * ks + kh * ks + kw
            wgt = tl.load(W_ptr + w_idx)

            acc += inp * wgt

    # Store result
    out_idx_global = channel * H_out * W_out + idx
    tl.store(out_ptr + out_idx_global, acc)


def triton_depthwise_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
):
    """
    Depthwise convolution implemented with Triton.
    Assumes groups == in_channels (depthwise).
    """
    assert x.is_cuda and weight.is_cuda, "Input and weight must be on CUDA."
    batch, in_channels, H_in, W_in = x.shape
    ks = weight.shape[2]  # kernel size (assumed square)
    # Compute output dimensions
    H_out = (H_in + 2 * padding - ks) // stride + 1
    W_out = (W_in + 2 * padding - ks) // stride + 1

    # Allocate output
    out = torch.empty((batch, in_channels, H_out, W_out), device=x.device, dtype=x.dtype)

    # Flatten batch for simplicity
    x_flat = x.reshape(-1, in_channels, H_in, W_in).contiguous()
    out_flat = out.reshape(-1, in_channels, H_out, W_out).contiguous()

    BLOCK_SIZE = 256
    num_programs = in_channels * H_out * W_out

    grid = lambda meta: (num_programs,)

    # Launch kernel for each batch element
    for b in range(batch):
        depthwise_conv_kernel[grid](
            x_flat[b].ptr(),
            weight.ptr(),
            bias.ptr() if bias is not None else None,
            out_flat[b].ptr(),
            stride=stride,
            padding=padding,
            ks=ks,
            n_channels=in_channels,
            H_in=H_in,
            W_in=W_in,
            H_out=H_out,
            W_out=W_out,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out


class ModelNew(nn.Module):
    """
    Depthwise 2D convolution replaced by a custom Triton kernel.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Keep weight and bias as parameters
        self.weight = nn.Parameter(
            torch.randn(in_channels, 1, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )