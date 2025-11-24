import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fuse_min_bias_scale(
    conv_ptr,
    bias_ptr,
    out_ptr,
    constant_value,
    scaling_factor,
    n_elements,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fuse the operations:
        y = torch.min(conv, constant_value) + bias + scaling_factor
    where `bias` has shape (C, 1, 1) and is broadcast over the spatial dimensions.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load conv value
    conv_val = tl.load(conv_ptr + offsets, mask=mask, other=0.0)

    # Compute channel index for each element
    # idx = n * C * H * W + c * H * W + h * W + w
    # channel = (idx // (H*W)) % C
    idx = offsets
    h_w = H * W
    channel = (idx // h_w) % C

    # Load bias for the channel
    bias_val = tl.load(bias_ptr + channel, mask=mask, other=0.0)

    # Apply min with constant
    min_val = tl.min(conv_val, constant_value)

    # Add bias and scale
    out_val = (min_val + bias_val) * scaling_factor

    tl.store(out_ptr + offsets, out_val, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model using a Triton kernel for the min+bias+scale fusion.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        constant_value,
        bias_shape,
        scaling_factor,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Conv2d
        conv_out = F.conv2d(x, self.conv.weight, bias=None, stride=self.conv.stride,
                           padding=self.conv.padding, dilation=self.conv.dilation,
                           groups=self.conv.groups)

        # Flatten for Triton kernel
        out = torch.empty_like(conv_out)
        n_elements = conv_out.numel()
        H, W = conv_out.shape[2], conv_out.shape[3]
        C = conv_out.shape[1]

        BLOCK_SIZE = 128

        grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

        fuse_min_bias_scale[grid](
            conv_out,
            self.bias,
            out,
            tl.constexpr(self.constant_value),
            tl.constexpr(self.scaling_factor),
            tl.constexpr(n_elements),
            tl.constexpr(H),
            tl.constexpr(W),
            tl.constexpr(C),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return out