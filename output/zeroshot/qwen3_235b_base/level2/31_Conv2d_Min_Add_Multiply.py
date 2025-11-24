import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_min_bias_scale_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, out_channels, out_height, out_width, in_channels, in_height, in_width,
    kernel_size, padding, stride,
    constant_value, scaling_factor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 2D block ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute starting indices
    m_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_range = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load bias values (per-channel, broadcasted)
    bias_offset = m_range[:, None]
    bias_mask = bias_offset < out_channels
    bias = tl.load(bias_ptr + bias_offset, mask=bias_mask, other=0.0)  # (BLOCK_M, 1)

    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 3x3 convolution with implicit gemm (no unrolling for simplicity)
    for c in range(in_channels):
        for ky in range(kernel_size):
            for kx in range(kernel_size):
                # Compute input pixel coordinates
                iy = ky - padding + tl.arange(0, BLOCK_N) // out_width * stride
                ix = kx - padding + tl.arange(0, BLOCK_N) % out_width * stride
                in_mask = (iy >= 0) & (iy < in_height) & (ix >= 0) & (ix < in_width)

                # Input offset: (batch, c, iy, ix) -> (B, C, H, W)
                in_offset = (
                    (tl.arange(0, BLOCK_N) // out_width) * in_channels * in_height * in_width +
                    c * in_height * in_width +
                    iy * in_width +
                    ix
                )
                in_val = tl.load(
                    x_ptr + in_offset,
                    mask=in_mask[None, :] & (m_range[:, None] < batch_size),
                    other=0.0
                )  # (BLOCK_M, BLOCK_N)

                # Weight offset: (out_c, in_c, ky, kx)
                w_offset = m_range[:, None] * in_channels * kernel_size * kernel_size + \
                           c * kernel_size * kernel_size + ky * kernel_size + kx
                w_mask = (m_range[:, None] < out_channels) & (c < in_channels)
                weight = tl.load(weight_ptr + w_offset, mask=w_mask, other=0.0)  # (BLOCK_M, 1)

                # Outer product: weight[:, None] @ in_val -> (BLOCK_M, BLOCK_N)
                acc += weight * in_val

    # Add bias (broadcasted)
    acc += bias

    # Apply min(clamp) with constant_value
    acc = tl.minimum(acc, constant_value)

    # Scale
    acc = acc * scaling_factor

    # Output index and mask
    out_offset = m_range[:, None] * out_height * out_width * out_channels + \
                 n_range[None, :]
    out_mask = (m_range[:, None] < out_channels) & (n_range[None, :] < out_height * out_width)
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


def triton_conv_min_bias_scale(x, weight, bias, constant_value, scaling_factor):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    padding = (kernel_size - 1) // 2
    stride = 1
    out_height = (in_height + 2 * padding - kernel_size) // stride + 1
    out_width = (in_width + 2 * padding - kernel_size) // stride + 1

    out = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(out_channels, meta['BLOCK_M']),
            triton.cdiv(out_height * out_width, meta['BLOCK_N'])
        )

    # Heuristics for block sizes
    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = 32

    conv_min_bias_scale_kernel[grid](
        x, weight, bias, out,
        batch_size, out_channels, out_height, out_width,
        in_channels, in_height, in_width,
        kernel_size, padding, stride,
        constant_value, scaling_factor,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using a fused Triton kernel for conv + min + bias + scale.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        # Fused convolution + min + bias + scale via Triton
        return triton_conv_min_bias_scale(x, self.weight, self.bias, self.constant_value, self.scaling_factor)