import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
#  Triton kernel for a 2‑D transposed convolution (NCHW) with square kernel.
#  The implementation uses a direct “im2col + matmul” approach.  For every
#  output element a kernel thread block loads the required input patch
#  and accumulates the weighted sum.  Tensor‑core friendly precision
#  (fp16/bf16) is used when available.
# --------------------------------------------------------------------------- #

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    ],
    key=["N", "C_in", "H_out", "W_out", "K"],
)
@triton.jit
def conv_transpose2d_kernel(
    input_ptr,      # (B, C_in, H_in, W_in)
    weight_ptr,     # (C_in, C_out, K, K)
    bias_ptr,       # (C_out,) or None
    output_ptr,     # (B, C_out, H_out, W_out)
    B, C_in, C_out, H_in, W_in, H_out, W_out, K,
    stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr,
    dtype_in: tl.constexpr,
    dtype_out: tl.constexpr,
):
    """
    Kernel layout:
        - Each program handles BLOCK_SIZE output positions.
        - The program iterates over all output positions, loading the
          corresponding input patch (im2col) and accumulating over
          the kernel dimensions and input channels.
    """
    # ---------------------------------------------------------- #
    #  Compute a linear index for the output positions handled
    #  by this program.  We flatten the batch, channel, height,
    #  and width dimensions in a single dimension.
    # ---------------------------------------------------------- #
    total_out = B * C_out * H_out * W_out
    out_start = tl.program_id(0) * BLOCK_SIZE
    offsets = out_start + tl.arange(0, BLOCK_SIZE)

    # Mask to ignore out‑of‑bounds programs
    mask = offsets < total_out
    if not tl.any(mask):
        return

    # ---------------------------------------------------------- #
    #  Decode the linear index into batch, output channel, y, x.
    # ---------------------------------------------------------- #
    b = offsets // (C_out * H_out * W_out)
    rem = offsets % (C_out * H_out * W_out)
    co = rem // (H_out * W_out)
    rem2 = rem % (H_out * W_out)
    cy = rem2 // W_out
    cx = rem2 % W_out

    # ---------------------------------------------------------- #
    #  Load the weight matrix into registers for fast access.
    # ---------------------------------------------------------- #
    weight = tl.load(weight_ptr, mask=tl.arange(0, K * K) < K * K)
    # weight shape: (C_in, C_out, K, K) => we will index as [ci, co, ky, kx]

    # ---------------------------------------------------------- #
    #  Accumulate convolution over kernel, input channels.
    # ---------------------------------------------------------- #
    acc = tl.zeros([1], dtype=dtype_out)

    for ci in range(C_in):
        for ky in range(K):
            for kx in range(K):
                # Compute corresponding input coordinates
                hy = cy * stride - padding + ky * dilation
                wx = cx * stride - padding + kx * dilation

                # Boundary check
                if hy < 0 or hy >= H_in or wx < 0 or wx >= W_in:
                    continue

                # Load input value
                inp_idx = (
                    b * (C_in * H_in * W_in) +
                    ci * (H_in * W_in) +
                    hy * W_in + wx
                )
                inp_val = tl.load(input_ptr + inp_idx, mask=True)

                # Load weight value
                w_idx = (
                    ci * (C_out * K * K) +
                    co * (K * K) +
                    ky * K + kx
                )
                w_val = tl.load(weight_ptr + w_idx, mask=True)

                acc += inp_val * w_val

    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + co, mask=True)
        acc += bias_val

    # Store the result
    out_idx = (
        b * (C_out * H_out * W_out) +
        co * (H_out * W_out) +
        cy * W_out + cx
    )
    tl.store(output_ptr + out_idx, acc, mask=mask)


# --------------------------------------------------------------------------- #
#  Wrapper to launch the kernel.  Handles dtype conversion and tensor shapes.
# --------------------------------------------------------------------------- #

def triton_conv_transpose2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
):
    """
    Parameters
    ----------
    input : torch.Tensor
        Input of shape (B, C_in, H_in, W_in).
    weight : torch.Tensor
        Weight of shape (C_in, C_out, K, K).
    bias : torch.Tensor, optional
        Bias of shape (C_out,).
    stride, padding, dilation : int
        Convolution parameters.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on GPU."
    B, C_in, H_in, W_in = input.shape
    _, C_out, K, _ = weight.shape

    # Compute output spatial dimensions
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1

    # Allocate output tensor
    output = torch.empty((B, C_out, H_out, W_out), dtype=input.dtype, device=input.device)

    # Ensure tensors are contiguous
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Launch kernel
    BLOCK_SIZE = 256
    grid = lambda meta: ((B * C_out * H_out * W_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    conv_transpose2d_kernel[grid](
        input,
        weight,
        bias,
        output,
        B, C_in, C_out, H_in, W_in, H_out, W_out, K,
        stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE,
        dtype_in=input.dtype,
        dtype_out=output.dtype,
    )
    return output


# --------------------------------------------------------------------------- #
#  New model definition using the Triton transposed convolution.
# --------------------------------------------------------------------------- #

class ModelNew(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input
    and square kernel, supporting dilation, padding, and stride.
    The implementation uses a custom Triton kernel for better performance.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )