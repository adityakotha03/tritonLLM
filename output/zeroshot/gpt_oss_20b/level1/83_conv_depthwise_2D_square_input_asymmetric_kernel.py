import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------- Triton kernel ----------
@triton.jit
def depthwise_conv_kernel(
    input_ptr,          # (batch, channel, height, width)
    kernel_ptr,         # (channel, kernel_h, 1)
    bias_ptr,           # (channel,) or None
    output_ptr,         # (batch, channel, height_out, width_out)
    batch: tl.constexpr,
    channel: tl.constexpr,
    height_in: tl.constexpr,
    width_in: tl.constexpr,
    height_out: tl.constexpr,
    width_out: tl.constexpr,
    kernel_h: tl.constexpr,
    stride_h: tl.constexpr,
    padding_h: tl.constexpr,
    dilation_h: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,   # must be 1 for a single output pixel per program
):
    # Program indices
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Stride of the output in the flattened tensor
    out_stride_b = channel * height_out * width_out
    out_stride_c = height_out * width_out
    out_stride_h = width_out
    # Index of the current output pixel
    out_idx = b * out_stride_b + c * out_stride_c + oh * out_stride_h + ow

    # Accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Compute each output pixel
    for kh in range(kernel_h):
        # Compute input row coordinate
        ih = oh * stride_h - padding_h + kh * dilation_h
        # Mask for valid input rows
        mask = (ih >= 0) & (ih < height_in)

        # Input stride (batch, channel, height, width)
        in_stride_b = channel * height_in * width_in
        in_stride_c = height_in * width_in
        in_stride_h = width_in

        # Index of the first element of the kernel row
        kernel_row_idx = c * kernel_h + kh

        # Load input element
        in_idx = b * in_stride_b + c * in_stride_c + ih * in_stride_h + ow
        inp = tl.load(input_ptr + in_idx, mask=mask, other=0.0)

        # Load kernel weight
        k_val = tl.load(kernel_ptr + kernel_row_idx, mask=mask, other=0.0)

        acc += inp * k_val

    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + c)
        acc += bias_val

    # Store result
    tl.store(output_ptr + out_idx, acc, mask=mask)


# ---------- Wrapper ----------
def triton_depthwise_conv(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    """
    Depthwise 2D convolution (kernel size (k,1)) implemented with Triton.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA"
    batch, channel, h_in, w_in = input.shape
    k_h = weight.shape[1]

    # Output size
    h_out = (h_in + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    w_out = w_in  # kernel width is 1, so no change

    output = torch.empty((batch, channel, h_out, w_out), device=input.device, dtype=input.dtype)

    BLOCK_SIZE = 1  # one output pixel per program

    grid = lambda meta: (
        batch,                # dim 0: batch
        channel,              # dim 1: channel
        h_out,                # dim 2: output height
        w_out,                # dim 3: output width
    )

    depthwise_conv_kernel[grid](
        input,
        weight,
        bias,
        output,
        batch,
        channel,
        h_in,
        w_in,
        h_out,
        w_out,
        k_h,
        stride,
        padding,
        dilation,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


# ---------- Optimised model ----------
class ModelNew(nn.Module):
    """
    Depthwise 2D convolution implemented with a custom Triton kernel.
    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.randn(in_channels, kernel_size, 1, device="cuda", dtype=torch.float32)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(in_channels, device="cuda", dtype=torch.float32)
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )