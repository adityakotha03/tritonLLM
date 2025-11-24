import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv2d_bn_scale_kernel(
    x_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
    gamma_ptr, beta_ptr, out_ptr,
    batch_size, in_channels, out_channels, height, width, out_h, out_w,
    kernel_size, stride, padding,
    eps,
    scaling_factor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 2D convolution with batch norm and scale fusion: Conv2d -> BatchNorm2d -> Scale
    # We use a tiled GEMM-like approach with implicit padding

    # Program IDs
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Handle output spatial dimensions (out_h, out_w)
    hw_pid = pid_b
    batch = hw_pid // (out_h * out_w)
    hw_rem = hw_pid % (out_h * out_w)
    out_row = hw_rem // out_w
    out_col = hw_rem % out_w

    # Input spatial start (due to stride and padding)
    in_row_start = out_row * stride - padding
    in_col_start = out_col * stride - padding

    # Pointers into input x for this output location (with implicit padding handling)
    x_offsets_base = batch * in_channels * height * width
    x_mask_base = (batch < batch_size) & (out_row < out_h) & (out_col < out_w)

    # Weight layout: [out_channels, in_channels, k, k]
    weight_offsets = tl.arange(0, BLOCK_M)[:, None] * in_channels * kernel_size * kernel_size + \
                     tl.arange(0, BLOCK_K)[None, :] * kernel_size * kernel_size + \
                     tl.arange(0, kernel_size)[:, None] * kernel_size + \
                     tl.arange(0, kernel_size)[None, :]
    weight_ptrs = weight_ptr + weight_offsets
    weight_mask = (tl.arange(0, BLOCK_M) < out_channels)[:, None] & (tl.arange(0, BLOCK_K) < in_channels * kernel_size * kernel_size)[None, :]

    # Initialize accumulator for output channel block
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Iterate over input channel tiles
    for ic in range(0, in_channels, BLOCK_K):
        # Compute input patch pointers
        x_ptrs = x_ptr + x_offsets_base
        x_mask = x_mask_base
        val = tl.zeros((kernel_size, kernel_size), dtype=tl.float32)
        for i in range(kernel_size):
            for j in range(kernel_size):
                in_i = in_row_start + i
                in_j = in_col_start + j
                in_mask = (in_i >= 0) & (in_i < height) & (in_j >= 0) & (in_j < width) & x_mask
                offset = ic * height * width + in_i * width + in_j
                val[i, j] = tl.load(x_ptr + offset, mask=in_mask, other=0.0)

        # Flatten val to [k*k] and tile to [BLOCK_K]
        val_flat = val.reshape((kernel_size * kernel_size,))
        val_tile = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for i in range(kernel_size * kernel_size):
            if ic + i < in_channels * kernel_size * kernel_size:
                val_tile = tl.where(tl.arange(0, BLOCK_K) == i, val_flat[i], val_tile)
        
        # Load weights for current input channel tile
        w = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

        # Accumulate GEMM: acc += w @ val_tile
        acc += tl.sum(w * val_tile[None, :], axis=1)

        # Update pointers
        weight_ptrs += BLOCK_K
        weight_mask = (tl.arange(0, BLOCK_M) < out_channels)[:, None] & \
                      (tl.arange(0, BLOCK_K) < in_channels * kernel_size * kernel_size - ic - BLOCK_K)[None, :]

    # Add bias
    bias_ptrs = bias_ptr + tl.arange(0, BLOCK_M)
    bias_mask = tl.arange(0, BLOCK_M) < out_channels
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    acc += bias

    # BatchNorm: (acc - running_mean) / sqrt(running_var + eps) * gamma + beta
    mean = tl.load(running_mean_ptr + tl.arange(0, BLOCK_M), mask=bias_mask, other=0.0)
    var = tl.load(running_var_ptr + tl.arange(0, BLOCK_M), mask=bias_mask, other=0.0)
    inv_std = 1.0 / tl.sqrt(var + eps)
    gamma = tl.load(gamma_ptr + tl.arange(0, BLOCK_M), mask=bias_mask, other=1.0)
    beta = tl.load(beta_ptr + tl.arange(0, BLOCK_M), mask=bias_mask, other=0.0)
    bn_out = (acc - mean) * inv_std * gamma + beta

    # Scale
    bn_out = bn_out * scaling_factor

    # Store output
    out_batch_offset = batch * out_channels * out_h * out_w
    out_hw_offset = out_row * out_w + out_col
    out_channel_offset = pid_m * BLOCK_M
    out_ptrs = out_ptr + out_batch_offset + out_hw_offset * out_channels + out_channel_offset + tl.arange(0, BLOCK_M)
    out_mask = (tl.arange(0, BLOCK_M) < out_channels) & (batch < batch_size) & (out_row < out_h) & (out_col < out_w)
    tl.store(out_ptrs, bn_out, mask=out_mask)


def triton_conv2d_bn_scale(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
    bn_gamma: torch.Tensor,
    bn_beta: torch.Tensor,
    eps: float,
    scaling_factor: float
):
    assert x.is_cuda and conv_weight.is_cuda
    x = x.contiguous()
    conv_weight = conv_weight.contiguous()
    conv_bias = conv_bias.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = conv_weight.shape
    stride = 1
    padding = kernel_size // 2
    out_h = (height + 2 * padding - kernel_size) // stride + 1
    out_w = (width + 2 * padding - kernel_size) // stride + 1

    # Output tensor
    out = torch.empty((batch_size, out_channels, out_h, out_w), dtype=x.dtype, device=x.device)

    # 1D grid over spatial locations (batch * out_h * out_w)
    # 2D grid over output channels (blocks in M dimension)
    BLOCK_M = 16
    BLOCK_N = 32
    BLOCK_K = 32

    def grid(META):
        return (
            batch_size * out_h * out_w,
            triton.cdiv(out_channels, META['BLOCK_M']),
            1
        )

    _conv2d_bn_scale_kernel[grid](
        x, conv_weight, conv_bias,
        bn_running_mean, bn_running_var, bn_gamma, bn_beta,
        out,
        batch_size, in_channels, out_channels, height, width, out_h, out_w,
        kernel_size, stride, padding,
        eps,
        scaling_factor,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using fused Triton kernel for Conv2d + BatchNorm2d + Scale.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.eps = self.bn.eps

    def forward(self, x):
        # Fused Conv2d -> BatchNorm2d -> Scale using Triton
        return triton_conv2d_bn_scale(
            x,
            self.conv.weight,
            self.conv.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.weight,
            self.bn.bias,
            self.eps,
            self.scaling_factor
        )