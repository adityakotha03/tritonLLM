import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    inp_ptr,
    weight_ptr,
    out_ptr,
    batch,
    channels,
    h_in,
    w_in,
    h_out,
    w_out,
    stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one output element (b, c, h, w)
    idx = tl.program_id(0)

    # Decode indices
    b = idx // (channels * h_out * w_out)
    rem = idx % (channels * h_out * w_out)
    c = rem // (h_out * w_out)
    rem2 = rem % (h_out * w_out)
    h = rem2 // w_out
    w = rem2 % w_out

    # Compute input base offset (top-left corner of kernel)
    base_in = (
        ((b * channels + c) * h_in + h * stride) * w_in
        + w * stride
    )

    # Load 3x3 patch and accumulate
    acc = tl.zeros([1], dtype=tl.float32)
    for i in range(3):
        for j in range(3):
            inp_offset = base_in + i * w_in + j
            wgt_offset = (c * 9) + i * 3 + j
            inp_val = tl.load(inp_ptr + inp_offset)
            wgt_val = tl.load(weight_ptr + wgt_offset)
            acc += inp_val * wgt_val

    # Write output
    out_offset = ((b * channels + c) * h_out + h) * w_out + w
    tl.store(out_ptr + out_offset, acc)


def triton_depthwise_conv2d(
    inp: torch.Tensor,
    weight: torch.Tensor,
    stride: int,
    padding: int,
):
    """
    inp:   (batch, in_channels, h_in, w_in)
    weight: (out_channels, 1, kH, kW)  -> depthwise conv
    """
    assert inp.is_cuda and weight.is_cuda
    batch, channels, h_in, w_in = inp.shape
    kH, kW = weight.shape[2], weight.shape[3]
    assert kH == kW == 3, "Kernel must be 3x3 for this implementation."
    assert padding == 0, "Padding not supported in this kernel."
    stride = stride

    h_out = (h_in + 2 * padding - kH) // stride + 1
    w_out = (w_in + 2 * padding - kW) // stride + 1

    out = torch.empty((batch, channels, h_out, w_out), device=inp.device, dtype=inp.dtype)

    # Flatten kernel weights: (channels, 9)
    weight_flat = weight.view(channels, -1).contiguous()

    BLOCK_SIZE = 128
    grid = lambda meta: ((batch * channels * h_out * w_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    depthwise_conv2d_kernel[grid](
        inp_ptr=inp.reshape(-1),
        weight_ptr=weight_flat.reshape(-1),
        out_ptr=out.reshape(-1),
        batch=batch,
        channels=channels,
        h_in=h_in,
        w_in=w_in,
        h_out=h_out,
        w_out=w_out,
        stride=stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with a 3×3 kernel using a custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        assert kernel_size == 3, "Only 3×3 kernel is supported in the Triton implementation."
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv2d(x, self.conv.weight, self.stride, self.padding)