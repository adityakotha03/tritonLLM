import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_batch_norm_relu_conv_kernel(
    x_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
    gamma_ptr, beta_ptr, out_ptr,
    batch_size, height, width, in_channels, out_channels,
    input_stride, output_stride, weight_stride_h, weight_stride_w,
    eps: tl.constexpr, BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_HW: tl.constexpr, BLOCK_SIZE_C_IN: tl.constexpr, BLOCK_SIZE_C_OUT: tl.constexpr
):
    # Compute program ids
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_out_ch = tl.program_id(2)

    # Compute offsets
    batch_start = pid_b * BLOCK_SIZE_BATCH
    hw_start = pid_hw * BLOCK_SIZE_HW
    out_ch_start = pid_out_ch * BLOCK_SIZE_C_OUT

    # Batch, height, width indices
    b_offsets = batch_start + tl.arange(0, BLOCK_SIZE_BATCH)
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE_HW)
    out_ch_offsets = out_ch_start + tl.arange(0, BLOCK_SIZE_C_OUT)

    # Mask for valid batch and spatial indices
    valid_b = b_offsets < batch_size
    valid_hw = hw_offsets < height * width
    valid_out_ch = out_ch_offsets < out_channels

    # Broadcast to full shape
    b_mask = valid_b[:, None, None]
    hw_mask = valid_hw[None, :, None]
    out_ch_mask = valid_out_ch[None, None, :]

    # Flatten spatial dimensions
    flat_hw = hw_offsets // width
    flat_w = hw_offsets % width
    flat_hw = flat_hw * width + flat_w

    # Load input x: shape (batch, in_channels, height, width)
    # We iterate over input channels in a loop
    acc = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_HW, BLOCK_SIZE_C_OUT), dtype=tl.float32)

    for ch_in_start in range(0, in_channels, BLOCK_SIZE_C_IN):
        ch_in_offsets = ch_in_start + tl.arange(0, BLOCK_SIZE_C_IN)
        valid_ch_in = ch_in_offsets < in_channels
        ch_in_mask = valid_ch_in[None, None, :]

        # Load input: (BLOCK_SIZE_BATCH, BLOCK_SIZE_C_IN, BLOCK_SIZE_HW)
        x_offsets = b_offsets[:, None, None] * input_stride + ch_in_offsets[None, :, None] * height * width + flat_hw[None, None, :]
        x_mask = b_mask and ch_in_mask and hw_mask
        x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

        # BatchNorm: (x - running_mean) / sqrt(running_var + eps) * gamma + beta
        running_mean = tl.load(running_mean_ptr + ch_in_offsets, mask=valid_ch_in, other=0.0)
        running_var = tl.load(running_var_ptr + ch_in_offsets, mask=valid_ch_in, other=0.0)
        gamma = tl.load(gamma_ptr + ch_in_offsets, mask=valid_ch_in, other=1.0)
        beta = tl.load(beta_ptr + ch_in_offsets, mask=valid_ch_in, other=0.0)

        inv_std = 1.0 / tl.sqrt(running_var + eps)
        x_bn = (x - running_mean[None, :, None]) * inv_std[None, :, None] * gamma[None, :, None] + beta[None, :, None]

        # ReLU
        x_relu = tl.where(x_bn > 0, x_bn, 0.0)

        # Conv: 3x3, so we need to handle neighbors
        # For simplicity, assume padding=1 and kernel=3x3
        h = flat_hw // width
        w = flat_hw % width

        # Compute padded coordinates
        h_pad = h + 1
        w_pad = w + 1

        # We'll unfold the 3x3 kernel
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                h_k = h + dy
                w_k = w + dx
                valid_hw_k = (h_k >= 0) & (h_k < height) & (w_k >= 0) & (w_k < width)
                hw_k_flat = h_k * width + w_k
                x_k_offsets = b_offsets[:, None, None] * input_stride + ch_in_offsets[None, :, None] * height * width + hw_k_flat[None, None, :]
                x_k_mask = b_mask and ch_in_mask and valid_hw_k[None, None, :]
                x_k = tl.load(x_ptr + x_k_offsets, mask=x_k_mask, other=0.0)

                # Load weights: (out_ch, in_ch, 3, 3)
                w_yx = (dy + 1) * 3 + (dx + 1)
                weight_offset = out_ch_offsets[:, None] * weight_stride_h + ch_in_offsets[None, :] * weight_stride_w + w_yx
                weight_mask = valid_out_ch[:, None] and valid_ch_in[None, :]
                weight = tl.load(weight_ptr + weight_offset, mask=weight_mask, other=0.0)

                # Multiply and accumulate
                # (BLOCK_SIZE_BATCH, BLOCK_SIZE_C_IN, BLOCK_SIZE_HW) -> (BLOCK_SIZE_BATCH, BLOCK_SIZE_HW, BLOCK_SIZE_C_IN)
                x_k = tl.trans(x_k)
                # (BLOCK_SIZE_C_OUT, BLOCK_SIZE_C_IN)
                weight = weight
                # acc += x_k @ weight.T
                for i in range(BLOCK_SIZE_BATCH):
                    for j in range(BLOCK_SIZE_HW):
                        for oc in range(BLOCK_SIZE_C_OUT):
                            if valid_out_ch[oc] and valid_b[i] and valid_hw[j]:
                                acc_val = 0.0
                                for ic in range(BLOCK_SIZE_C_IN):
                                    if valid_ch_in[ic]:
                                        acc_val += x_k[i, j, ic] * weight[oc, ic]
                                acc[i, j, oc] += acc_val

    # Store output
    out_offsets = b_offsets[:, None, None] * output_stride + out_ch_offsets[None, None, :] * height * width + flat_hw[None, None, :]
    out_mask = b_mask and out_ch_mask and hw_mask
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)


def fused_batch_norm_relu_conv(x, bn_weight, bn_bias, bn_running_mean, bn_running_var, conv_weight, conv_bias, eps=1e-5):
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_weight.shape[0]
    assert conv_weight.shape[1] == in_channels
    assert conv_weight.shape[2] == 3 and conv_weight.shape[3] == 3

    out = torch.empty((batch_size, out_channels, height, width), device=x.device, dtype=x.dtype)

    # Compute strides
    input_stride = in_channels * height * width
    output_stride = out_channels * height * width
    weight_stride_h = out_channels * 3 * 3
    weight_stride_w = 3 * 3

    # 1D grid over batch, spatial, and output channels
    BLOCK_SIZE_BATCH = 4
    BLOCK_SIZE_HW = 32
    BLOCK_SIZE_C_IN = 32
    BLOCK_SIZE_C_OUT = 16

    grid = (
        triton.cdiv(batch_size, BLOCK_SIZE_BATCH),
        triton.cdiv(height * width, BLOCK_SIZE_HW),
        triton.cdiv(out_channels, BLOCK_SIZE_C_OUT)
    )

    fused_batch_norm_relu_conv_kernel[grid](
        x, conv_weight, conv_bias, bn_running_mean, bn_running_var,
        bn_weight, bn_bias, out,
        batch_size, height, width, in_channels, out_channels,
        input_stride, output_stride, weight_stride_h, weight_stride_w,
        eps=eps,
        BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
        BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT
    )
    return out


class FusedBatchNormReLUConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return fused_batch_norm_relu_conv(
            x,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.conv.weight,
            self.conv.bias
        )


class ModelNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(ModelNew, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(FusedBatchNormReLUConv(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, dim=1)
        return x