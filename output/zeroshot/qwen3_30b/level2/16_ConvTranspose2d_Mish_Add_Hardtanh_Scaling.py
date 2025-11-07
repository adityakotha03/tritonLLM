import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, stride, padding, output_padding,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_OC: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    # Block indices
    pid = tl.program_id(0)
    pid_h = pid // (BLOCK_W // TILE_SIZE)
    pid_w = (pid % (BLOCK_W // TILE_SIZE)) // (BLOCK_H // TILE_SIZE)
    pid_c = pid % (BLOCK_H // TILE_SIZE)

    # Calculate output channel block
    out_c_start = pid_c * BLOCK_OC
    out_c_end = tl.minimum(out_c_start + BLOCK_OC, out_channels)

    # Calculate output spatial block
    h_start = pid_h * TILE_SIZE
    h_end = h_start + TILE_SIZE
    w_start = pid_w * TILE_SIZE
    w_end = w_start + TILE_SIZE

    # Output dimensions
    out_h = (height - 1) * stride + kernel_size - 2 * padding + output_padding
    out_w = (width - 1) * stride + kernel_size - 2 * padding + output_padding

    # Initialize accumulator
    acc = tl.zeros((TILE_SIZE, TILE_SIZE), dtype=tl.float32)

    # Loop over input channels and kernel
    for c in range(0, in_channels, BLOCK_OC):
        c_start = c
        c_end = tl.minimum(c_start + BLOCK_OC, in_channels)

        # Load weights (out_channels, in_channels, kH, kW) -> (oc, ic, kH, kW)
        # We tile the weights in chunks of BLOCK_OC in in_channels
        w_ptrs = w_ptr + c_start * kernel_size * kernel_size * out_channels
        w_ptrs += out_c_start * kernel_size * kernel_size
        w = tl.load(
            w_ptrs + (tl.arange(0, BLOCK_OC)[:, None, None] * kernel_size * kernel_size +
                      tl.arange(0, kernel_size)[None, :, None] * kernel_size +
                      tl.arange(0, kernel_size)[None, None, :]),
            mask=(tl.arange(0, BLOCK_OC)[:, None, None] < out_c_end - out_c_start) &
                 (tl.arange(0, kernel_size)[None, :, None] < kernel_size) &
                 (tl.arange(0, kernel_size)[None, None, :] < kernel_size),
            other=0.0
        )

        # Load input data (batch_size, in_channels, height, width)
        # We need to load input at positions (h_start - padding, w_start - padding) and go from there
        x_ptrs = x_ptr + (tl.arange(0, batch_size)[:, None, None, None] * in_channels * height * width +
                          c_start * height * width +
                          (tl.arange(0, TILE_SIZE)[:, None, None] + h_start - padding) * width +
                          (tl.arange(0, TILE_SIZE)[None, :, None] + w_start - padding) * 1)

        # Clip to valid input bounds
        valid_h = (tl.arange(0, TILE_SIZE)[:, None, None] + h_start - padding) >= 0
        valid_w = (tl.arange(0, TILE_SIZE)[None, :, None] + w_start - padding) >= 0
        valid_h = valid_h & (tl.arange(0, TILE_SIZE)[:, None, None] + h_start - padding) < height
        valid_w = valid_w & (tl.arange(0, TILE_SIZE)[None, :, None] + w_start - padding) < width

        mask = valid_h & valid_w
        x = tl.load(
            x_ptrs,
            mask=mask[:, :, None] & (tl.arange(0, c_end - c_start)[:, None, None] < c_end - c_start),
            other=0.0
        )

        # Compute convolution: x * w
        # (TILE_SIZE, TILE_SIZE, BLOCK_OC) * (BLOCK_OC, kernel_size, kernel_size) -> (TILE_SIZE, TILE_SIZE, kernel_size, kernel_size)
        # Then reduce over kernel_size, kernel_size
        for k in range(kernel_size):
            for l in range(kernel_size):
                x_kl = x[:, :, k, l]
                w_kl = w[:, k, l]
                acc += tl.dot(x_kl, w_kl.T)

    # Write output
    out_ptrs = out_ptr + (tl.arange(0, batch_size)[:, None, None, None] * out_channels * out_h * out_w +
                          out_c_start * out_h * out_w +
                          (tl.arange(0, TILE_SIZE)[:, None, None] + h_start) * out_w +
                          (tl.arange(0, TILE_SIZE)[None, :, None] + w_start) * 1)

    # Ensure output bounds
    out_valid_h = (tl.arange(0, TILE_SIZE)[:, None, None] + h_start) < out_h
    out_valid_w = (tl.arange(0, TILE_SIZE)[None, :, None] + w_start) < out_w
    out_mask = out_valid_h & out_valid_w
    tl.store(out_ptrs, acc, mask=out_mask[:, :, None] & (tl.arange(0, out_c_end - out_c_start)[:, None, None] < out_c_end - out_c_start))


@triton.jit
def mish_hardtanh_scale_kernel(
    x_ptr, out_ptr,
    batch_size, out_channels, out_h, out_w,
    add_value, scale,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles BLOCK_SIZE elements
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * out_channels * out_h * out_w

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Apply Mish: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x))
    # To avoid overflow, use log(1 + exp(x)) = x + log(1 + exp(-x)) for x < 0, else log(1 + exp(x))
    # We use online computation: compute tanh(softplus(x)) without storing intermediate results
    # Compute softplus(x)
    neg_x = -x
    softplus_x = tl.where(x > 0, x + tl.log1p(tl.exp(neg_x)), tl.log1p(tl.exp(x)))
    # Compute tanh(softplus_x)
    tanh_softplus = tl.tanh(softplus_x)
    # Compute Mish
    mish = x * tanh_softplus

    # Add value
    added = mish + add_value

    # Apply Hardtanh: clamp to [-1, 1]
    clamped = tl.clamp(added, -1.0, 1.0)

    # Scale
    scaled = clamped * scale

    tl.store(out_ptr + offsets, scaled, mask=mask)


def triton_conv_transpose(x: torch.Tensor, weight: torch.Tensor, stride, padding, output_padding, kernel_size, in_channels, out_channels):
    # Ensure contiguous tensors
    x = x.contiguous()
    weight = weight.contiguous()

    # Output dimensions
    batch_size, _, height, width = x.shape
    out_h = (height - 1) * stride + kernel_size - 2 * padding + output_padding
    out_w = (width - 1) * stride + kernel_size - 2 * padding + output_padding

    # Output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)

    # Set block sizes
    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_OC = 16
    TILE_SIZE = 16

    # Grid size
    grid = lambda meta: (
        (out_h + meta["BLOCK_H"] - 1) // meta["BLOCK_H"] *
        (out_w + meta["BLOCK_W"] - 1) // meta["BLOCK_W"] *
        (out_channels + meta["BLOCK_OC"] - 1) // meta["BLOCK_OC"],
    )

    # Launch kernel
    conv_transpose_kernel[grid](
        x, weight, out,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding, output_padding,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_OC=BLOCK_OC,
        TILE_SIZE=TILE_SIZE
    )

    return out


def triton_mish_hardtanh_scale(x: torch.Tensor, add_value, scale):
    x = x.contiguous()
    out = torch.empty_like(x)

    # Use a BLOCK_SIZE that is a power of 2 and fits the output size
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Power of 2, fits memory and warp size

    # Grid
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    mish_hardtanh_scale_kernel[grid](
        x, out, x.shape[0], x.shape[1], x.shape[2], x.shape[3],
        add_value, scale, BLOCK_SIZE=BLOCK_SIZE
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale

        # Use BFloat16 for better tensor core performance
        self.use_bfloat16 = True

    def forward(self, x):
        # Convert to BFloat16 if available
        if self.use_bfloat16 and x.dtype == torch.float32:
            x = x.to(torch.bfloat16)

        # Perform convolution transpose using Triton
        x = triton_conv_transpose(
            x, self.conv_transpose.weight, self.conv_transpose.stride[0], self.conv_transpose.padding[0],
            self.conv_transpose.output_padding[0], self.conv_transpose.kernel_size[0],
            self.conv_transpose.in_channels, self.conv_transpose.out_channels
        )

        # Fuse Mish, Add, Hardtanh, Scale into one Triton kernel
        x = triton_mish_hardtanh_scale(x, self.add_value, self.scale)

        # Convert back to float32 if needed
        if self.use_bfloat16 and x.dtype == torch.bfloat16:
            x = x.to(torch.float32)

        return x