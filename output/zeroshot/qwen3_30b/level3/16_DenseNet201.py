import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton kernel for fused BatchNorm + ReLU + Conv2d + Dropout (fused)
@triton.jit
def fused_conv_bn_relu_dropout_kernel(
    x_ptr,  # Input tensor pointer
    weight_ptr,  # Weight tensor pointer
    bias_ptr,  # Bias pointer
    running_mean_ptr,  # Running mean
    running_var_ptr,  # Running var
    out_ptr,  # Output pointer
    batch_size,  # Batch size
    in_channels,  # Input channels
    out_channels,  # Output channels
    height,  # Height of feature map
    width,  # Width of feature map
    kernel_size,  # Kernel size (assumed square)
    stride,  # Stride
    padding,  # Padding
    BLOCK_SIZE: tl.constexpr,
    USE_CUDNN: tl.constexpr = True,
):
    # Block-wide offsets
    pid = tl.program_id(0)  # 1D grid: each block handles one spatial region
    block_idx = pid // (out_channels * (height // stride) * (width // stride))
    c = pid % (out_channels * (height // stride) * (width // stride))

    # Channel, height, width decomposition
    c_out = c // ((height // stride) * (width // stride))
    h_out = (c // (width // stride)) % (height // stride)
    w_out = c % (width // stride)

    # Base output index
    out_offset = block_idx * out_channels * (height // stride) * (width // stride) + c

    # Load input and weights using local memory (shared) if needed
    # We will use tiling to reduce global memory accesses
    tile_size = 16
    tile_h = tl.min(tl.load(tl.arange(0, tile_size, dtype=tl.int32) + h_out * stride), height - 1)
    tile_w = tl.min(tl.load(tl.arange(0, tile_size, dtype=tl.int32) + w_out * stride), width - 1)

    # Compute spatial offset for input
    input_offset = (block_idx * in_channels * height * width +
                    (c_out // (kernel_size * kernel_size)) * height * width +
                    tile_h * width + tile_w)

    # Load weights for this channel
    weight_offset = c_out * kernel_size * kernel_size
    weight = tl.load(weight_ptr + weight_offset, mask=tl.arange(0, kernel_size * kernel_size) < kernel_size * kernel_size, other=0.0)

    # Load input tiles
    x_tile = tl.load(x_ptr + input_offset, mask=tl.arange(0, tile_size * tile_size) < tile_size * tile_size, other=0.0)

    # Compute output (Conv2d)
    conv_val = tl.sum(x_tile * weight, axis=0)

    # Load running mean and var for this output channel
    mean = tl.load(running_mean_ptr + c_out, mask=c_out < out_channels, other=0.0)
    var = tl.load(running_var_ptr + c_out, mask=c_out < out_channels, other=0.0)

    # BatchNorm: (x - mean) / sqrt(var + eps)
    eps = 1e-5
    inv_var = 1.0 / tl.sqrt(var + eps)
    norm = (conv_val - mean) * inv_var

    # Bias
    bias = tl.load(bias_ptr + c_out, mask=c_out < out_channels, other=0.0)
    norm += bias

    # ReLU
    output = tl.max(norm, 0.0)

    # Dropout: probabilistic zeroing (inference mode only)
    dropout_mask = tl.load(tl.arange(0, 1, dtype=tl.int32), mask=tl.arange(0, 1) == 1, other=1.0)  # Dummy: use 1.0 for inference
    output = output * dropout_mask

    # Store output
    tl.store(out_ptr + out_offset, output, mask=tl.arange(0, 1) == 1)


# Triton kernel for fused Conv2d + BatchNorm + ReLU (used in transition layer)
@triton.jit
def fused_conv_bn_relu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % out_channels
    h = (pid // out_channels) % (height // stride)
    w = (pid // out_channels) // (height // stride)

    # Output offset
    out_offset = (pid // out_channels) * (width // stride) + w

    # Compute input region
    h_start = h * stride - padding
    w_start = w * stride - padding

    # Load weights
    weight = tl.load(weight_ptr + c * kernel_size * kernel_size, mask=tl.arange(0, kernel_size * kernel_size) < kernel_size * kernel_size, other=0.0)

    # Load input
    x_val = tl.zeros((kernel_size, kernel_size), dtype=tl.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            h_idx = h_start + i
            w_idx = w_start + j
            valid = (h_idx >= 0) & (h_idx < height) & (w_idx >= 0) & (w_idx < width)
            offset = (pid // out_channels) * in_channels * height * width + (c // (kernel_size * kernel_size)) * height * width + h_idx * width + w_idx
            x_val[i, j] = tl.load(x_ptr + offset, mask=valid, other=0.0)

    # Conv
    conv_val = tl.sum(x_val * weight, axis=(0, 1))

    # BN
    mean = tl.load(running_mean_ptr + c, mask=c < out_channels, other=0.0)
    var = tl.load(running_var_ptr + c, mask=c < out_channels, other=0.0)
    eps = 1e-5
    inv_var = 1.0 / tl.sqrt(var + eps)
    norm = (conv_val - mean) * inv_var

    # Bias + ReLU
    bias = tl.load(bias_ptr + c, mask=c < out_channels, other=0.0)
    output = tl.max(norm + bias, 0.0)

    tl.store(out_ptr + out_offset, output, mask=tl.arange(0, 1) == 1)


# Triton kernel for efficient spatial concatenation (cat along channel dim)
@triton.jit
def concat_channels_kernel(
    x_ptrs,  # List of pointers to input tensors
    out_ptr,
    batch_size,
    num_tensors,
    h,
    w,
    total_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (h * w * total_channels)
    hw_idx = pid % (h * w * total_channels)
    h_idx = (hw_idx // w) % h
    w_idx = hw_idx % w

    out_offset = b * total_channels * h * w + hw_idx
    c = hw_idx // (h * w)

    # Determine which input tensor to read from
    accumulated = 0
    for i in range(num_tensors):
        c_size = tl.load(x_ptrs[i] + 1, mask=tl.arange(0, 1) == 1)  # Placeholder
        if c < accumulated + c_size:
            in_c = c - accumulated
            in_ptr = x_ptrs[i] + 2  # Skip metadata
            in_offset = b * c_size * h * w + in_c * h * w + h_idx * w + w_idx
            value = tl.load(in_ptr + in_offset, mask=tl.arange(0, 1) == 1, other=0.0)
            tl.store(out_ptr + out_offset, value, mask=tl.arange(0, 1) == 1)
            break
        accumulated += c_size


# Triton kernel for online softmax (used in AdaptiveAvgPool2d + classification)
@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (channels * height * width)
    c = (pid // (height * width)) % channels
    hw = pid % (height * width)

    # Load input
    input_offset = b * channels * height * width + c * height * width + hw
    x = tl.load(x_ptr + input_offset, mask=tl.arange(0, 1) == 1, other=0.0)

    # Online softmax: subtract max, then exp
    # We'll use shared memory for max reduction
    shared_mem = tl.load(tl.arange(0, 16), mask=tl.arange(0, 1) == 1, other=0.0)  # Placeholder
    max_val = x
    tl.store(shared_mem + 0, max_val, mask=tl.arange(0, 1) == 1)

    # Reduce max across all elements in this block
    # Simplified: assume we have enough shared memory and parallelism
    # In practice, we would use reduction primitives
    max_val = tl.load(shared_mem + 0, mask=tl.arange(0, 1) == 1)

    # Compute exp
    exp_val = tl.exp(x - max_val)

    # Store
    tl.store(out_ptr + input_offset, exp_val, mask=tl.arange(0, 1) == 1)


# Triton kernel for adaptive_avg_pool2d (online)
@triton.jit
def adaptive_avg_pool_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    out_h,
    out_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (channels * out_h * out_w)
    c = (pid // (out_h * out_w)) % channels
    oh = (pid // out_w) % out_h
    ow = pid % out_w

    # Compute input region
    h_start = oh * height // out_h
    h_end = (oh + 1) * height // out_h
    w_start = ow * width // out_w
    w_end = (ow + 1) * width // out_w

    # Compute sum
    total = 0.0
    count = 0.0
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            offset = b * channels * height * width + c * height * width + h * width + w
            total += tl.load(x_ptr + offset, mask=tl.arange(0, 1) == 1, other=0.0)
            count += 1.0

    # Average
    avg = total / count
    tl.store(out_ptr + pid, avg, mask=tl.arange(0, 1) == 1)


# Triton kernel for Linear layer (fused with bias)
@triton.jit
def linear_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    in_features,
    out_features,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // out_features
    c = pid % out_features

    # Load weight row
    w_row = tl.load(w_ptr + c * in_features, mask=tl.arange(0, in_features) < in_features, other=0.0)

    # Load input
    x_vec = tl.load(x_ptr + b * in_features, mask=tl.arange(0, in_features) < in_features, other=0.0)

    # Matmul + bias
    out = tl.dot(x_vec, w_row) + tl.load(bias_ptr + c, mask=tl.arange(0, 1) == 1, other=0.0)

    tl.store(out_ptr + pid, out, mask=tl.arange(0, 1) == 1)


# Custom Triton-based DenseBlock (fused layers)
class TritonDenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, device="cuda"):
        super().__init__()
        self.num_layers = num_layers
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate
        self.device = device
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_c = num_input_features + i * growth_rate
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_c).to(device),
                    nn.Conv2d(in_c, growth_rate, kernel_size=3, padding=1, bias=False).to(device),
                    nn.Dropout(0.0).to(device)
                )
            )
        self.to(device)

    def forward(self, x):
        features = [x]
        for i, layer in enumerate(self.layers):
            # Fused: BN + Conv + ReLU + Dropout → custom Triton kernel
            in_features = self.num_input_features + i * self.growth_rate
            out_features = self.growth_rate
            h, w = x.shape[2], x.shape[3]
            batch_size = x.shape[0]
            kernel_size = 3
            stride = 1
            padding = 1

            # Prepare pointers
            x_ptr = x.data_ptr()
            weight_ptr = layer[1].weight.data_ptr()
            bias_ptr = layer[1].bias.data_ptr()
            running_mean_ptr = layer[0].running_mean.data_ptr()
            running_var_ptr = layer[0].running_var.data_ptr()

            # Output
            out = torch.empty_like(x, size=(batch_size, out_features, h, w))

            # Grid and block
            num_elements = batch_size * out_features * h * w
            BLOCK_SIZE = 256
            grid = lambda meta: (num_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]

            # Launch kernel
            fused_conv_bn_relu_dropout_kernel[grid](
                x_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
                out.data_ptr(),
                batch_size, in_features, out_features, h, w,
                kernel_size, stride, padding,
                BLOCK_SIZE=BLOCK_SIZE
            )

            features.append(out)
            x = torch.cat(features, dim=1)
        return x


# Custom Triton-based TransitionLayer
class TritonTransitionLayer(nn.Module):
    def __init__(self, num_input_features, num_output_features, device="cuda"):
        super().__init__()
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self.device = device
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False).to(device)
        self.bn = nn.BatchNorm2d(num_output_features).to(device)
        self.relu = nn.ReLU(inplace=True).to(device)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2).to(device)

    def forward(self, x):
        batch_size, in_c, h, w = x.shape
        out_h, out_w = h // 2, w // 2

        # Fused: BN + Conv + ReLU + AvgPool → Triton kernel
        out = torch.empty(batch_size, self.num_output_features, out_h, out_w, device=x.device)

        # Load pointers
        x_ptr = x.data_ptr()
        weight_ptr = self.conv.weight.data_ptr()
        bias_ptr = self.conv.bias.data_ptr()
        running_mean_ptr = self.bn.running_mean.data_ptr()
        running_var_ptr = self.bn.running_var.data_ptr()

        # Grid
        num_elements = batch_size * self.num_output_features * out_h * out_w
        BLOCK_SIZE = 256
        grid = lambda meta: (num_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]

        # Launch kernel
        fused_conv_bn_relu_kernel[grid](
            x_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
            out.data_ptr(),
            batch_size, in_c, self.num_output_features, h, w,
            kernel_size=1, stride=1, padding=0,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # Final pool
        out = F.avg_pool2d(out, kernel_size=2, stride=2)
        return out


# Final model with Triton kernels
class ModelNew(nn.Module):
    def __init__(self, growth_rate=32, num_classes=1000, device="cuda"):
        super().__init__()
        self.device = device

        # Initial conv
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device),
            nn.BatchNorm2d(64).to(device),
            nn.ReLU(inplace=True).to(device),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)
        )

        # Dense blocks and transition layers (with Triton)
        num_features = 64
        block_layers = [6, 12, 48, 32]

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = TritonDenseBlock(num_layers, num_features, growth_rate, device)
            self.dense_blocks.append(block)
            num_features += num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TritonTransitionLayer(num_features, num_features // 2, device)
                self.transition_layers.append(transition)
                num_features //= 2

        # Final layers
        self.final_bn = nn.BatchNorm2d(num_features).to(device)
        self.classifier = nn.Linear(num_features, num_classes).to(device)

    def forward(self, x):
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn(x)
        x = F.relu(x, inplace=True)
        x = adaptive_avg_pool_kernel[x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]](
            x.data_ptr(), x.data_ptr(), x.shape[0], x.shape[1], x.shape[2], x.shape[3], 1, 1
        )
        x = x.view(x.shape[0], -1)
        x = linear_kernel[
            x.shape[0] * self.classifier.out_features
        ](
            x.data_ptr(),
            self.classifier.weight.data_ptr(),
            self.classifier.bias.data_ptr(),
            x.data_ptr(),
            x.shape[0], x.shape[1], self.classifier.out_features
        )
        return x