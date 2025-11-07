import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weights pointer
    out_ptr,  # Output tensor pointer
    x_batch, x_in_channels, x_height, x_width,
    w_out_channels, w_in_channels, w_kernel_size,
    out_batch, out_out_channels, out_height, out_width,
    divisor,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
):
    # Calculate indices
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_oc = tl.program_id(3)
    pid_ic = tl.program_id(4)

    # Block offsets for output
    h_offset = pid_h * BLOCK_SIZE_H
    w_offset = pid_w * BLOCK_SIZE_W
    oc_offset = pid_oc * BLOCK_SIZE_OC

    # Input indices
    ic_start = pid_ic * BLOCK_SIZE_IC
    ic_end = tl.minimum(ic_start + BLOCK_SIZE_IC, w_in_channels)

    # Loop over input channels and kernel
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    for ic in range(ic_start, ic_end, BLOCK_SIZE_IC):
        # Load input tile
        x_block = tl.load(
            x_ptr + pid_batch * x_in_channels * x_height * x_width +
            ic * x_height * x_width +
            h_offset * x_width + w_offset,
            mask=tl.arange(0, BLOCK_SIZE_H)[:, None] < x_height,
            other=0.0
        )

        # Load weights
        w_block = tl.load(
            w_ptr + pid_oc * w_in_channels * w_kernel_size * w_kernel_size +
            ic * w_kernel_size * w_kernel_size +
            tl.arange(0, w_kernel_size)[:, None] * w_kernel_size +
            tl.arange(0, w_kernel_size)[None, :],
            mask=tl.arange(0, w_kernel_size)[:, None] < w_kernel_size,
            other=0.0
        )

        # Perform convolution
        for i in range(w_kernel_size):
            for j in range(w_kernel_size):
                acc += x_block[i:i + BLOCK_SIZE_H, j:j + BLOCK_SIZE_W] * w_block[i, j]

    # Load output
    out_tile = tl.load(
        out_ptr + pid_batch * out_out_channels * out_height * out_width +
        pid_oc * out_height * out_width +
        h_offset * out_width + w_offset,
        mask=tl.arange(0, BLOCK_SIZE_H)[:, None] < out_height,
        other=0.0
    )

    # Write back
    tl.store(
        out_ptr + pid_batch * out_out_channels * out_height * out_width +
        pid_oc * out_height * out_width +
        h_offset * out_width + w_offset,
        acc,
        mask=tl.arange(0, BLOCK_SIZE_H)[:, None] < out_height,
    )


@triton.jit
def leaky_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    negative_slope: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0, x, negative_slope * x)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv2d_leaky_relu(x, w, divisor):
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    batch, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = w.shape

    out_height = height - kernel_size + 1
    out_width = width - kernel_size + 1

    # Output tensor
    out = torch.empty(batch, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)

    # Grid setup
    grid_h = (out_height + 15) // 16
    grid_w = (out_width + 15) // 16
    grid_oc = (out_channels + 15) // 16
    grid_ic = (in_channels + 15) // 16

    # Kernel launch
    conv2d_kernel[
        (batch, grid_h, grid_w, grid_oc, grid_ic),
    ](
        x, w, out,
        batch, in_channels, height, width,
        out_channels, in_channels, kernel_size,
        batch, out_channels, out_height, out_width,
        divisor,
        BLOCK_SIZE_H=16,
        BLOCK_SIZE_W=16,
        BLOCK_SIZE_OC=16,
        BLOCK_SIZE_IC=16,
    )

    # Apply LeakyReLU via Triton
    out = out.contiguous()
    out = torch.empty_like(out)
    n_elements = out.numel()

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    leaky_relu_kernel[grid](
        out, out, n_elements, 0.01, BLOCK_SIZE=128
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor

    def forward(self, x):
        return triton_conv2d_leaky_relu(x, self.conv.weight, self.divisor)