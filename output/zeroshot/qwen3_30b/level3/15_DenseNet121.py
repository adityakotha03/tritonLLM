import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton kernels for custom operations

@triton.jit
def batch_norm_fwd_kernel(
    x_ptr,  # Input pointer
    weight_ptr,  # Weight pointer
    bias_ptr,  # Bias pointer
    mean_ptr,  # Mean pointer
    var_ptr,  # Variance pointer
    out_ptr,  # Output pointer
    batch_size,  # Batch size
    channels,  # Number of channels
    height,  # Height
    width,  # Width
    BLOCK_SIZE: tl.constexpr,
):
    # Block index
    pid = tl.program_id(0)
    # Each block processes one channel
    channel = pid * BLOCK_SIZE
    # Thread index within the block
    tid = tl.arange(0, BLOCK_SIZE)
    # Mask for valid threads
    mask = tid < channels

    # Load weight, bias, mean, and variance for this channel
    weight = tl.load(weight_ptr + channel + tid, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + channel + tid, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + channel + tid, mask=mask, other=0.0)
    var = tl.load(var_ptr + channel + tid, mask=mask, other=0.0)

    # Compute effective variance (var + eps)
    var = var + 1e-5  # eps
    inv_var = 1.0 / tl.sqrt(var)

    # Iterate over spatial dimensions and batch
    for b in range(batch_size):
        for h in range(height):
            for w in range(width):
                # Compute offset for this spatial location
                offset = (b * channels * height * width) + (channel * height * width) + (h * width) + w
                # Load input
                x = tl.load(x_ptr + offset, mask=mask, other=0.0)
                # Normalize
                x = (x - mean) * inv_var
                # Scale and shift
                x = x * weight + bias
                # Store output
                tl.store(out_ptr + offset, x, mask=mask)

    # Synchronize
    tl.static_print("BatchNorm forward done")


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input pointer
    w_ptr,  # Weight pointer
    out_ptr,  # Output pointer
    batch_size,  # Batch size
    in_channels,  # Input channels
    out_channels,  # Output channels
    height,  # Height
    width,  # Width
    kernel_size,  # Kernel size (3x3)
    BLOCK_SIZE: tl.constexpr,
):
    # Each block processes one output channel
    pid = tl.program_id(0)
    out_channel = pid * BLOCK_SIZE
    # Thread index
    tid = tl.arange(0, BLOCK_SIZE)
    mask = tid < out_channels

    # Load weights
    w_offset = (out_channel + tid) * in_channels * kernel_size * kernel_size
    w = tl.load(w_ptr + w_offset, mask=mask, other=0.0)
    w = tl.reshape(w, (out_channels, in_channels, kernel_size, kernel_size))
    
    # Iterate over batch, spatial dimensions
    for b in range(batch_size):
        for h in range(height):
            for w in range(width):
                # Output offset
                out_offset = (b * out_channels * height * width) + (out_channel * height * width) + (h * width) + w
                # Accumulate output
                acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
                for ic in range(in_channels):
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            # Input offset
                            in_offset = (b * in_channels * height * width) + (ic * height * width) + ((h + kh) * width) + (w + kw)
                            x = tl.load(x_ptr + in_offset, mask=mask, other=0.0)
                            # Accumulate
                            acc += x * w[pid, ic, kh, kw]
                # Store output
                tl.store(out_ptr + out_offset, acc, mask=mask)


@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.max(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def cat_kernel(
    inputs_ptr,  # Pointer to first input
    out_ptr,  # Output pointer
    num_inputs,  # Number of inputs to concatenate
    batch_size,  # Batch size
    height,  # Height
    width,  # Width
    total_channels,  # Total number of channels
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_channels * batch_size * height * width

    # Compute channel offset for this block
    channel_offset = (pid % num_inputs) * (batch_size * height * width)
    base_offset = pid // num_inputs * (total_channels * batch_size * height * width)
    src_offset = base_offset + channel_offset

    # Load input
    x = tl.load(inputs_ptr + src_offset + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def avg_pool2d_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Pool size is 2x2
    pool_size = 2
    out_h = height // pool_size
    out_w = width // pool_size

    pid = tl.program_id(0)
    # Each block handles one spatial location in the output
    out_h_idx = pid // out_w
    out_w_idx = pid % out_w

    # Output offset
    out_offset = (pid * channels)
    # Iterate over input 2x2 patch
    acc = tl.zeros((channels,), dtype=tl.float32)
    for i in range(pool_size):
        for j in range(pool_size):
            in_h = out_h_idx * pool_size + i
            in_w = out_w_idx * pool_size + j
            in_offset = (in_h * width + in_w) * channels
            x = tl.load(x_ptr + in_offset, mask=tl.arange(0, channels) < channels, other=0.0)
            acc += x
    # Average
    acc /= 4.0
    tl.store(out_ptr + out_offset, acc, mask=tl.arange(0, channels) < channels)


@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    target_h,
    target_w,
    BLOCK_SIZE: tl.constexpr,
):
    # For adaptive pooling to (1,1), target_h = target_w = 1
    pid = tl.program_id(0)
    out_offset = pid * channels
    acc = tl.zeros((channels,), dtype=tl.float32)
    # Calculate pooling window size
    stride_h = height // target_h
    stride_w = width // target_w
    # Pool each window
    for h in range(target_h):
        for w in range(target_w):
            start_h = h * stride_h
            start_w = w * stride_w
            for i in range(stride_h):
                for j in range(stride_w):
                    in_h = start_h + i
                    in_w = start_w + j
                    offset = (in_h * width + in_w) * channels
                    x = tl.load(x_ptr + offset, mask=tl.arange(0, channels) < channels, other=0.0)
                    acc += x
    acc /= (stride_h * stride_w)
    tl.store(out_ptr + out_offset, acc, mask=tl.arange(0, channels) < channels)


@triton.jit
def linear_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    batch_size,
    in_features,
    out_features,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each block processes one row of output
    row = pid * BLOCK_SIZE
    tid = tl.arange(0, BLOCK_SIZE)
    mask = row + tid < out_features

    # Load weights for this row
    w_offset = (row + tid) * in_features
    w = tl.load(w_ptr + w_offset, mask=mask, other=0.0)

    # Compute output
    acc = tl.zeros((in_features,), dtype=tl.float32)
    for b in range(batch_size):
        x_offset = b * in_features
        x = tl.load(x_ptr + x_offset, mask=tl.arange(0, in_features) < in_features, other=0.0)
        acc += x * w

    # Store output
    out_offset = pid * BLOCK_SIZE * in_features
    tl.store(out_ptr + out_offset, acc, mask=mask)


# Triton wrappers

def triton_batch_norm(x, weight, bias, running_mean, running_var, training=True):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)

    batch_size, channels, height, width = x.shape
    BLOCK_SIZE = 128
    grid = lambda meta: (channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]

    batch_norm_fwd_kernel[grid](
        x, weight, bias, running_mean, running_var, out,
        batch_size, channels, height, width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_conv2d(x, weight, bias=None, stride=1, padding=1):
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    out_channels, in_channels, kernel_h, kernel_w = weight.shape
    batch_size, _, height, width = x.shape
    out_height = height
    out_width = width

    out = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 128
    grid = lambda meta: (out_channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]

    conv2d_kernel[grid](
        x, weight, out, batch_size, in_channels, out_channels,
        height, width, kernel_h, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_relu(x):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_cat(inputs, dim=1):
    assert all(x.is_cuda for x in inputs)
    batch_size = inputs[0].shape[0]
    height = inputs[0].shape[2]
    width = inputs[0].shape[3]
    total_channels = sum(x.shape[1] for x in inputs)
    out = torch.empty(batch_size, total_channels, height, width, device=inputs[0].device, dtype=inputs[0].dtype)
    inputs_ptr = torch.cat([x.contiguous().view(-1) for x in inputs], dim=0).contiguous().data_ptr()
    out_ptr = out.data_ptr()

    BLOCK_SIZE = 128
    grid = lambda meta: (total_channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]

    cat_kernel[grid](
        inputs_ptr, out_ptr, len(inputs), batch_size, height, width, total_channels,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_avg_pool2d(x, kernel_size=2, stride=2):
    assert x.is_cuda
    x = x.contiguous()
    batch_size, channels, height, width = x.shape
    out_h = height // kernel_size
    out_w = width // kernel_size
    out = torch.empty(batch_size, channels, out_h, out_w, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 128
    grid = lambda meta: (out_h * out_w + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    avg_pool2d_kernel[grid](
        x, out, batch_size, channels, height, width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_adaptive_avg_pool2d(x, output_size=(1, 1)):
    assert x.is_cuda
    x = x.contiguous()
    batch_size, channels, height, width = x.shape
    out_h, out_w = output_size
    out = torch.empty(batch_size, channels, out_h, out_w, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 128
    grid = lambda meta: (out_h * out_w + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    adaptive_avg_pool2d_kernel[grid](
        x, out, batch_size, channels, height, width, out_h, out_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_linear(x, weight, bias=None):
    assert x.is_cuda and weight.is_cuda
    batch_size, in_features = x.shape
    out_features, _ = weight.shape
    out = torch.empty(batch_size, out_features, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 128
    grid = lambda meta: (out_features + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    linear_kernel[grid](
        x, weight, out, batch_size, in_features, out_features,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


# New model with Triton kernels

class DenseBlockNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super().__init__()
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
            # Apply BN, ReLU, Conv, Dropout via Triton
            x = layer[0](x)
            x = triton_batch_norm(x, layer[0].weight, layer[0].bias, layer[0].running_mean, layer[0].running_var)
            x = triton_relu(x)
            x = triton_conv2d(x, layer[2].weight, layer[2].bias)
            x = layer[3](x)  # Dropout
            features.append(x)
            x = triton_cat(features, dim=1)
        return x


class TransitionLayerNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features

    def forward(self, x):
        # Apply BN, ReLU, Conv, AvgPool via Triton
        x = triton_batch_norm(x, self.bn.weight, self.bn.bias, self.bn.running_mean, self.bn.running_var)
        x = triton_relu(x)
        x = triton_conv2d(x, self.conv.weight, self.conv.bias)
        x = triton_avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super().__init__()
        self.growth_rate = growth_rate
        self.num_classes = num_classes

        # Initial convolution and pooling
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Dense blocks and transition layers
        num_features = 64
        block_layers = [6, 12, 24, 16]
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlockNew(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayerNew(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        # Final batch norm and classifier
        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        # Final BN and classifier
        x = triton_batch_norm(x, self.final_bn.weight, self.final_bn.bias, self.final_bn.running_mean, self.final_bn.running_var)
        x = triton_relu(x)
        x = triton_adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = triton_linear(x, self.classifier.weight, self.classifier.bias)
        return x