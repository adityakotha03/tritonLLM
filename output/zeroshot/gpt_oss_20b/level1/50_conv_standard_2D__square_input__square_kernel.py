import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton kernel for a generic 2‑D convolution with padding handled by a pre‑pad step.
@triton.jit
def conv2d_kernel(
    X_ptr,          # input tensor (after padding), flattened
    W_ptr,          # weight tensor, flattened
    Y_ptr,          # output tensor, flattened
    B: tl.constexpr,        # batch size
    C_in: tl.constexpr,     # number of input channels
    H_in_pad: tl.constexpr, # padded input height
    W_in_pad: tl.constexpr, # padded input width
    C_out: tl.constexpr,    # number of output channels
    KH: tl.constexpr,       # kernel height
    KW: tl.constexpr,       # kernel width
    stride: tl.constexpr,   # stride
    H_out: tl.constexpr,    # output height
    W_out: tl.constexpr,    # output width
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of output elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    n_out = B * C_out * H_out * W_out
    mask = offsets < n_out
    if tl.any(mask):
        # decode linear offset into 4‑D indices
        b = offsets // (C_out * H_out * W_out)
        rem = offsets % (C_out * H_out * W_out)
        c_out = rem // (H_out * W_out)
        rem2 = rem % (H_out * W_out)
        h_out = rem2 // W_out
        w_out = rem2 % W_out

        h_in_start = h_out * stride
        w_in_start = w_out * stride

        acc = tl.zeros([], dtype=tl.float32)

        for ci in range(C_in):
            for kh in range(KH):
                h_in = h_in_start + kh
                for kw in range(KW):
                    w_in = w_in_start + kw
                    # compute flattened indices
                    x_idx = ((b * C_in + ci) * H_in_pad + h_in) * W_in_pad + w_in
                    w_idx = ((c_out * C_in + ci) * KH + kh) * KW + kw
                    x_val = tl.load(X_ptr + x_idx)
                    w_val = tl.load(W_ptr + w_idx)
                    acc += x_val * w_val

        y_idx = ((b * C_out + c_out) * H_out + h_out) * W_out + w_out
        tl.store(Y_ptr + y_idx, acc, mask=True)


def conv2d_triton(x: torch.Tensor,
                 weight: torch.Tensor,
                 stride: int = 1,
                 pad: int = 0,
                 kernel_size: int = 3) -> torch.Tensor:
    """
    Perform 2‑D convolution using the custom Triton kernel.
    x:      Input tensor of shape (B, C_in, H_in, W_in)
    weight: Weight tensor of shape (C_out, C_in, KH, KW)
    """
    B, C_in, H_in, W_in = x.shape
    C_out, _, KH, KW = weight.shape
    assert KH == KW == kernel_size, "Kernel size mismatch"

    # Output spatial dimensions
    H_out = (H_in + 2 * pad - KH) // stride + 1
    W_out = (W_in + 2 * pad - KW) // stride + 1

    # Pad input on GPU (no additional kernel load)
    padded = F.pad(x, (pad, pad, pad, pad))
    H_in_pad = H_in + 2 * pad
    W_in_pad = W_in + 2 * pad

    out = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 256
    grid = lambda meta: ((B * C_out * H_out * W_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    conv2d_kernel[grid](
        padded, weight, out,
        B, C_in, H_in_pad, W_in_pad,
        C_out, KH, KW,
        stride, H_out, W_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class Conv2dTriton(nn.Module):
    """
    Convolution layer that uses the custom Triton kernel.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size,
                        device="cuda")
        )
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv2d_triton(
            x,
            self.weight,
            stride=self.stride,
            pad=self.padding,
            kernel_size=self.kernel_size,
        )


class ModelNew(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(ModelNew, self).__init__()
        self.conv1 = Conv2dTriton(
            in_channels=3,
            out_channels=96,
            kernel_size=11,
            stride=4,
            padding=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return x