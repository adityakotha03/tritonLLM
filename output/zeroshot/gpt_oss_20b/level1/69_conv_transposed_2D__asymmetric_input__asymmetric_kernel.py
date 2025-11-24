import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    ],
    key=["N", "C"],
)
@triton.jit
def add_bias_kernel(
    out_ptr,      # pointer to output tensor
    bias_ptr,     # pointer to bias tensor
    n_elements,   # total elements in the output tensor
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # load output
    out = tl.load(out_ptr + offsets, mask=mask, other=0.0)

    # compute channel index for each element
    # output shape: (B, C, H, W)
    B, C, H, W = tl.shape(out_ptr)
    # stride of each dimension
    stride_b = C * H * W
    stride_c = H * W
    stride_h = W
    stride_w = 1

    # linear offset to 3D indices
    b_idx = (offsets // stride_b) % B
    c_idx = (offsets // stride_c) % C
    h_idx = (offsets // stride_h) % H
    w_idx = offsets % W

    # load bias for this channel
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    # add bias
    out = out + bias

    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution operation with asymmetric input and kernel size.
    The bias addition is fused into a custom Triton kernel for higher performance.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        output_padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if bias:
            # register bias as a buffer so it can be accessed by Triton
            self.register_buffer("bias_buffer", self.conv_transpose2d.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution with fused bias addition.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        out = self.conv_transpose2d(x)

        if self.conv_transpose2d.bias is not None:
            # prepare grid for bias addition
            n_elements = out.numel()
            BLOCK_SIZE = 256  # this will be autotuned by Triton

            grid = lambda meta: (
                (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
            )

            add_bias_kernel[grid](
                out,
                self.bias_buffer,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        return out