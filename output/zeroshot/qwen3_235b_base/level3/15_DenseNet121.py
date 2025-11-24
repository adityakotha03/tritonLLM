import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_batch_norm_relu_kernel(
    x_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
    out_ptr, n_channels, height, width, num_elements,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr
):
    pid_c = tl.program_id(0)
    pid_hw = tl.program_id(1)

    channels_offset = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_offset = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    mask_c = channels_offset < n_channels
    mask_hw = hw_offset < height * width

    offset = channels_offset[:, None] * (height * width) + hw_offset[None, :]
    mask = mask_c[:, None] & mask_hw[None, :]

    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    mean = tl.load(running_mean_ptr + channels_offset, mask=mask_c, other=0.0)
    inv_var = tl.load(running_var_ptr + channels_offset, mask=mask_c, other=0.0)
    inv_std = tl.rsqrt(inv_var + eps)
    weight = tl.load(weight_ptr + channels_offset, mask=mask_c, other=1.0)
    bias = tl.load(bias_ptr + channels_offset, mask=mask_c, other=0.0)

    x_hat = (x - mean[:, None]) * inv_std[:, None]
    out = x_hat * weight[:, None] + bias[:, None]
    out = tl.where(out > 0, out, 0.0)  # ReLU

    tl.store(out_ptr + offset, out, mask=mask)


def triton_fused_batch_norm_relu(
    x: torch.Tensor,
    bn: nn.BatchNorm2d
):
    assert x.is_cuda and bn.weight.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)

    n_channels = x.shape[1]
    height, width = x.shape[2], x.shape[3]
    num_elements = x.numel()

    BLOCK_C = 16
    BLOCK_HW = 512
    grid = (triton.cdiv(n_channels, BLOCK_C), triton.cdiv(height * width, BLOCK_HW))

    fused_batch_norm_relu_kernel[grid](
        x, bn.weight, bn.bias, bn.running_mean, bn.running_var,
        out, n_channels, height, width, num_elements,
        eps=bn.eps, BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW
    )
    return out


@triton.jit
def fused_conv2d_relu_kernel(
    x_ptr, weight_ptr, out_ptr,
    batch, in_channels, in_height, in_width,
    out_channels, out_height, out_width,
    kernel_size, stride, padding,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OUT_CH: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_IN_CH: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    hw_id = tl.program_id(2)

    # Define offsets
    b_offset = batch_id * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    oc_offset = out_ch_id * BLOCK_OUT_CH + tl.arange(0, BLOCK_OUT_CH)
    hw_offset = hw_id * BLOCK_HW + tl.arange(0, BLOCK_HW)

    # Masks
    b_mask = b_offset < batch
    oc_mask = oc_offset < out_channels
    hw_mask = hw_offset < out_height * out_width

    # Output spatial indices
    h_out = (hw_offset // out_width).to(tl.int32)
    w_out = (hw_offset % out_width).to(tl.int32)

    # Input spatial start (with stride and padding)
    h_in_start = h_out * stride - padding
    w_in_start = w_out * stride - padding

    # Input patch: [BLOCK_BATCH, BLOCK_HW, KERNEL, KERNEL, BLOCK_IN_CH]
    acc = tl.zeros((BLOCK_BATCH, BLOCK_HW, BLOCK_OUT_CH), dtype=tl.float32)

    # Loop over input channels in blocks
    for ic_base in range(0, in_channels, BLOCK_IN_CH):
        ic_offset = ic_base + tl.arange(0, BLOCK_IN_CH)
        ic_mask = ic_offset < in_channels

        # Load input patch: [BLOCK_BATCH, BLOCK_HW, BLOCK_IN_CH, KERNEL, KERNEL]
        x_patches = tl.load(
            x_ptr +
            b_offset[:, None, None, None, None] * in_channels * in_height * in_width +
            ic_offset[None, None, :, None, None] * in_height * in_width +
            (h_in_start[:, None, None] + tl.arange(0, kernel_size)[None, :, None]) * in_width +
            (w_in_start[:, None, None] + tl.arange(0, kernel_size)[None, None, :]),
            mask=b_mask[:, None, None, None, None] &
                ic_mask[None, None, :, None, None] &
                ((h_in_start[:, None, None] + tl.arange(0, kernel_size)[None, :, None]) >= 0) &
                ((h_in_start[:, None, None] + tl.arange(0, kernel_size)[None, :, None]) < in_height) &
                ((w_in_start[:, None, None] + tl.arange(0, kernel_size)[None, None, :]) >= 0) &
                ((w_in_start[:, None, None] + tl.arange(0, kernel_size)[None, None, :]) < in_width),
            other=0.0
        )  # Shape: [B, K, K, C_in]

        # Reshape to [BLOCK_BATCH, BLOCK_HW, BLOCK_IN_CH * KERNEL * KERNEL]
        x_flat = tl.reshape(x_patches, (BLOCK_BATCH, BLOCK_HW, -1))

        # Load weights: [BLOCK_OUT_CH, BLOCK_IN_CH, K, K]
        w = tl.load(
            weight_ptr +
            oc_offset[:, None, None, None] * in_channels * kernel_size * kernel_size +
            ic_offset[None, :, None, None] * kernel_size * kernel_size +
            tl.arange(0, kernel_size)[None, None, :, None] * kernel_size +
            tl.arange(0, kernel_size)[None, None, None, :],
            mask=oc_mask[:, None, None, None] & ic_mask[None, :, None, None],
            other=0.0
        )
        w_flat = tl.reshape(w, (BLOCK_OUT_CH, -1))  # [OUT_CH, IN_CH * K * K]

        # GEMM: [B, HW, OUT_CH] = [B, HW, IN_CH*K*K] @ [OUT_CH, IN_CH*K*K].T
        acc += tl.dot(x_flat, w_flat.T)

    # ReLU
    acc = tl.where(acc > 0, acc, 0.0)

    # Store output
    output_offset = (
        b_offset[:, None, None] * out_channels * out_height * out_width +
        oc_offset[None, :, None] * out_height * out_width +
        h_out[:, None, None] * out_width +
        w_out[:, None, None]
    )
    output_mask = b_mask[:, None, None] & oc_mask[None, :, None] & hw_mask[None, None, :]
    tl.store(out_ptr + output_offset, acc, mask=output_mask)


def triton_fused_conv2d_relu(
    x: torch.Tensor,
    conv: nn.Conv2d
):
    assert x.is_cuda and conv.weight.is_cuda
    x = x.contiguous()
    batch, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_size, _ = conv.weight.shape
    out_height = (in_height + 2 * conv.padding[0] - kernel_size) // conv.stride[0] + 1
    out_width = (in_width + 2 * conv.padding[1] - kernel_size) // conv.stride[1] + 1

    out = torch.empty((batch, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)

    BLOCK_BATCH = 1
    BLOCK_OUT_CH = 16
    BLOCK_HW = 64
    BLOCK_IN_CH = 16
    BLOCK_K = 3

    grid = (
        triton.cdiv(batch, BLOCK_BATCH),
        triton.cdiv(out_channels, BLOCK_OUT_CH),
        triton.cdiv(out_height * out_width, BLOCK_HW)
    )

    fused_conv2d_relu_kernel[grid](
        x, conv.weight, out,
        batch, in_channels, in_height, in_width,
        out_channels, out_height, out_width,
        kernel_size, conv.stride[0], conv.padding[0],
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_OUT_CH=BLOCK_OUT_CH,
        BLOCK_HW=BLOCK_HW,
        BLOCK_IN_CH=BLOCK_IN_CH,
        BLOCK_K=BLOCK_K
    )
    return out


class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            bn = layer[0]
            conv = layer[2]
            # Fuse BatchNorm + ReLU + Conv2d + ReLU into one kernel
            x_bn_relu = triton_fused_batch_norm_relu(x, bn)
            new_feature = triton_fused_conv2d_relu(x_bn_relu, conv)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = triton_fused_batch_norm_relu(x, self.bn)
        x = self.conv(x)
        x = self.pool(x)
        return x


class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_features = 64
        block_layers = [6, 12, 24, 16]

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = triton_fused_batch_norm_relu(x, self.bn1)
        x = self.pool1(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = triton_fused_batch_norm_relu(x, self.final_bn)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x