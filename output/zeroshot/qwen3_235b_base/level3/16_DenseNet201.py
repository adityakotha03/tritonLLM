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

    channel_offset = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_offset = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    mask_c = channel_offset < n_channels
    mask_hw = hw_offset < height * width

    offset = channel_offset[:, None] * (height * width) + hw_offset[None, :]
    mask = mask_c[:, None] & mask_hw[None, :]

    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    mean = tl.load(running_mean_ptr + channel_offset, mask=mask_c, other=0.0)
    var = tl.load(running_var_ptr + channel_offset, mask=mask_c, other=0.0)
    gamma = tl.load(weight_ptr + channel_offset, mask=mask_c, other=1.0)
    beta = tl.load(bias_ptr + channel_offset, mask=mask_c, other=0.0)

    inv_std = 1.0 / tl.sqrt(var + eps)
    x_hat = (x - mean[:, None]) * inv_std[:, None]
    out = gamma[:, None] * x_hat + beta[:, None]
    out = tl.maximum(0.0, out)

    tl.store(out_ptr + offset, out, mask=mask)


def triton_fused_batch_norm_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    n, c, h, w = x.shape
    num_elements = n * c * h * w

    out = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(c, meta['BLOCK_C']), triton.cdiv(h * w, meta['BLOCK_HW']))

    fused_batch_norm_relu_kernel[grid](
        x, weight, bias, running_mean, running_var, out,
        c, h, w, num_elements,
        eps=eps,
        BLOCK_C=16, BLOCK_HW=256
    )
    return out


@triton.jit
def fused_conv2d_relu_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch, out_channels, in_channels, out_h, out_w, in_h, in_w,
    kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles = batch * out_h * out_w
    if pid >= num_tiles:
        return

    batch_idx = pid // (out_h * out_w)
    remaining = pid % (out_h * out_w)
    out_y = remaining // out_w
    out_x = remaining % out_w

    out_offset = pid * out_channels + tl.arange(0, BLOCK_SIZE)
    mask_out = out_offset < out_channels

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    bias = tl.load(bias_ptr + out_offset, mask=mask_out, other=0.0)

    for c in range(in_channels):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                in_y = out_y * stride_h - pad_h + kh
                in_x = out_x * stride_w - pad_w + kw

                mask_in = (in_y >= 0) & (in_y < in_h) & (in_x >= 0) & (in_x < in_w)
                in_offset = batch_idx * in_channels * in_h * in_w + c * in_h * in_w + in_y * in_w + in_x
                x_val = tl.load(x_ptr + in_offset, mask=mask_in, other=0.0)

                w_offset = out_offset * in_channels * kernel_h * kernel_w + c * kernel_h * kernel_w + kh * kernel_w + kw
                w_val = tl.load(weight_ptr + w_offset, mask=mask_out, other=0.0)

                acc += x_val.to(tl.float32) * w_val

    acc += bias
    out = tl.maximum(0.0, acc)
    out = out.to(x_ptr.dtype.element_ty)

    tl.store(out_ptr + out_offset, out, mask=mask_out)


def triton_fused_conv2d_relu(x, weight, bias, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1):
    assert groups == 1
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    b, c_in, h_in, w_in = x.shape
    c_out, c_in_g, k_h, k_w = weight.shape
    assert c_in == c_in_g
    s_h, s_w = stride
    p_h, p_w = padding
    d_h, d_w = dilation

    h_out = (h_in + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
    w_out = (w_in + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1

    out = torch.empty((b, c_out, h_out, w_out), device=x.device, dtype=x.dtype)

    grid = lambda meta: (b * h_out * w_out,)

    fused_conv2d_relu_kernel[grid](
        x, weight, bias, out,
        b, c_out, c_in, h_out, w_out, h_in, w_in,
        k_h, k_w, s_h, s_w, p_h, p_w,
        BLOCK_SIZE=64
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
            relu = layer[1]
            conv = layer[2]
            x_in = x
            x_bn = bn(x_in)
            x_relu = relu(x_bn)
            x_conv = conv(x_relu)
            new_feature = x_conv
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = triton_fused_batch_norm_relu(
            x, self.bn.weight, self.bn.bias, self.bn.running_mean, self.bn.running_var
        )
        x = triton_fused_conv2d_relu(x, self.conv.weight, self.conv.bias, stride=1, padding=0)
        x = self.pool(x)
        return x


class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_features = 64
        block_layers = [6, 12, 48, 32]

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
        x = triton_fused_batch_norm_relu(
            x, self.bn1.weight, self.bn1.bias, self.bn1.running_mean, self.bn1.running_var
        )
        x = self.pool1(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = triton_fused_batch_norm_relu(
            x, self.final_bn.weight, self.final_bn.bias, self.final_bn.running_mean, self.final_bn.running_var
        )
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x