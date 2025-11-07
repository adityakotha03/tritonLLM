import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_channels, out_channels, height, width, kernel_size,
    stride_h, stride_w, pad_h, pad_w,
    input_batch_stride, weight_batch_stride, output_batch_stride,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
    TILE_H: tl.constexpr, TILE_W: tl.constexpr, TILE_C: tl.constexpr,
):
    # Get program ID for the output block
    pid = tl.program_id(0)
    pid_h = pid // (TILE_W // BLOCK_W)
    pid_w = (pid // (TILE_H // BLOCK_H)) % (TILE_W // BLOCK_W)
    pid_c = pid % (TILE_H // BLOCK_H) if TILE_H // BLOCK_H > 1 else 0

    # Block offsets
    block_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    block_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    block_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    # Mask for valid indices
    h_mask = block_h < height
    w_mask = block_w < width
    c_mask = block_c < out_channels

    # Output offsets
    out_h = block_h // stride_h
    out_w = block_w // stride_w
    out_c = block_c

    # Load output offsets
    out_offset = (out_h * width + out_w) * out_channels + out_c
    out_ptr += out_offset

    # Input offsets
    input_batch = tl.program_id(1)
    input_offset = input_batch * input_batch_stride
    input_ptr += input_offset

    # Weight offsets
    weight_offset = (out_c * in_channels * kernel_size * kernel_size) + (tl.arange(0, in_channels) * kernel_size * kernel_size)
    weight_ptr += weight_offset

    # Accumulate output
    acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_C), dtype=tl.float32)
    
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Compute input offsets for current kernel position
            ih = out_h * stride_h + kh - pad_h
            iw = out_w * stride_w + kw - pad_w

            # Valid input indices
            valid = (ih >= 0) & (ih < height) & (iw >= 0) & (iw < width)
            valid_h = ih < height
            valid_w = iw < width

            # Load input
            input_offset = (ih * width + iw) * in_channels + tl.arange(0, in_channels)
            input_ptr_local = input_ptr + input_offset
            input_vals = tl.load(input_ptr_local, mask=valid, other=0.0)

            # Load weight
            weight_offset = (kh * kernel_size + kw) * in_channels
            weight_ptr_local = weight_ptr + weight_offset
            weight_vals = tl.load(weight_ptr_local, mask=tl.arange(0, in_channels) < in_channels, other=0.0)

            # Compute dot product
            input_expanded = input_vals[None, :]  # (1, in_channels)
            weight_expanded = weight_vals[:, None]  # (in_channels, 1)
            dot_prod = tl.dot(input_expanded, weight_expanded)
            acc += dot_prod

    # Load bias
    bias_ptr += out_c
    bias_vals = tl.load(bias_ptr, mask=c_mask, other=0.0)
    acc += bias_vals[None, None, :]

    # Store output
    out_mask = h_mask[:, None, None] & w_mask[None, :, None] & c_mask[None, None, :]
    tl.store(out_ptr, acc, mask=out_mask)


@triton.jit
def batch_norm_kernel(
    input_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
    output_ptr,
    batch_size, channels, height, width,
    eps,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    block_c = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = block_c < channels

    # Load input
    input_offset = block_c
    input_ptr += input_offset
    input_vals = tl.load(input_ptr, mask=c_mask, other=0.0)

    # Load statistics
    mean = tl.load(running_mean_ptr + block_c, mask=c_mask, other=0.0)
    var = tl.load(running_var_ptr + block_c, mask=c_mask, other=0.0)

    # Normalize
    input_norm = (input_vals - mean) / tl.sqrt(var + eps)
    output = input_norm * weight_ptr + bias_ptr

    # Store
    output_ptr += block_c
    tl.store(output_ptr, output, mask=c_mask)


@triton.jit
def relu6_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.minimum(tl.maximum(x, 0.0), 6.0)
    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def adaptive_avg_pool_kernel(
    input_ptr, output_ptr,
    batch_size, channels, height, width,
    out_h, out_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_idx < batch_size * channels

    # Input and output offsets
    input_batch = block_idx // channels
    input_channel = block_idx % channels

    # Calculate input stride
    input_offset = input_batch * channels * height * width + input_channel * height * width
    output_offset = input_batch * channels * out_h * out_w + input_channel * out_h * out_w

    # Pooling size
    pool_h = height // out_h
    pool_w = width // out_w

    # Accumulate
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for h in range(out_h):
        for w in range(out_w):
            i_h = h * pool_h
            i_w = w * pool_w
            for ph in range(pool_h):
                for pw in range(pool_w):
                    idx = (i_h + ph) * width + (i_w + pw)
                    acc += tl.load(input_ptr + input_offset + idx, mask=mask, other=0.0)

    # Average
    acc /= (pool_h * pool_w)
    tl.store(output_ptr + output_offset, acc, mask=mask)


@triton.jit
def linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_features, out_features,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size

    # Input
    input_ptr += offsets * in_features
    input_vals = tl.load(input_ptr, mask=mask[:, None], other=0.0)

    # Weight
    weight_ptr += tl.arange(0, out_features)
    weight_vals = tl.load(weight_ptr, mask=tl.arange(0, out_features) < out_features, other=0.0)

    # Compute matmul + bias
    output = tl.dot(input_vals, weight_vals)
    if bias_ptr is not None:
        bias_vals = tl.load(bias_ptr, mask=tl.arange(0, out_features) < out_features, other=0.0)
        output += bias_vals[None, :]

    # Output
    output_ptr += offsets * out_features
    tl.store(output_ptr, output, mask=mask[:, None])


class TritonConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride_h = stride if isinstance(stride, int) else stride[0]
        self.stride_w = stride if isinstance(stride, int) else stride[1]
        self.pad_h = padding if isinstance(padding, int) else padding[0]
        self.pad_w = padding if isinstance(padding, int) else padding[1]
        self.bias = bias

        # Initialize weight
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Ensure contiguous
        x = x.contiguous()

        # Prepare output
        out_height = (height + 2 * self.pad_h - self.kernel_size) // self.stride_h + 1
        out_width = (width + 2 * self.pad_w - self.kernel_size) // self.stride_w + 1
        out = torch.empty(batch_size, self.out_channels, out_height, out_width, device=x.device, dtype=x.dtype)

        # Define grid
        BLOCK_H, BLOCK_W, BLOCK_C = 16, 16, 16
        TILE_H, TILE_W, TILE_C = 64, 64, 16
        grid_h = (out_height + TILE_H - 1) // TILE_H
        grid_w = (out_width + TILE_W - 1) // TILE_W
        grid_c = (self.out_channels + TILE_C - 1) // TILE_C
        grid = (grid_h * grid_w * grid_c, batch_size)

        # Launch kernel
        conv2d_kernel[grid](
            x, self.weight, self.bias, out,
            self.in_channels, self.out_channels, height, width, self.kernel_size,
            self.stride_h, self.stride_w, self.pad_h, self.pad_w,
            x.stride(0), self.weight.stride(0), out.stride(0),
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C,
            TILE_H=TILE_H, TILE_W=TILE_W, TILE_C=TILE_C,
        )

        return out


class TritonBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))

    def forward(self, x):
        x = x.contiguous()
        out = torch.empty_like(x)
        BLOCK_C = 16
        grid = (x.shape[1] // BLOCK_C + 1, x.shape[0])
        batch_norm_kernel[grid](
            x, self.weight, self.bias, self.running_mean, self.running_var,
            out, x.shape[0], x.shape[1], x.shape[2], x.shape[3],
            self.eps, BLOCK_C=BLOCK_C
        )
        return out


class TritonReLU6(nn.Module):
    def forward(self, x):
        x = x.contiguous()
        out = torch.empty_like(x)
        BLOCK_SIZE = 1024
        grid = (x.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
        relu6_kernel[grid](x, out, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
        return out


class TritonAdaptiveAvgPool2d(nn.Module):
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        out_h, out_w = 1, 1
        out = torch.empty(batch_size, channels, out_h, out_w, device=x.device, dtype=x.dtype)
        BLOCK_SIZE = 64
        grid = (batch_size * channels + BLOCK_SIZE - 1) // BLOCK_SIZE
        adaptive_avg_pool_kernel[grid](
            x, out, batch_size, channels, height, width,
            out_h, out_w, BLOCK_SIZE=BLOCK_SIZE
        )
        return out


class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        batch_size, _ = x.shape
        out = torch.empty(batch_size, self.out_features, device=x.device, dtype=x.dtype)
        BLOCK_SIZE = 256
        grid = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
        linear_kernel[grid](
            x, self.weight, self.bias, out,
            batch_size, self.in_features, self.out_features,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # Replace initial conv + BN
        self.conv1 = TritonConv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = TritonBatchNorm2d(32)
        
        # MBConv blocks
        self.mbconv1 = self._make_mbconv_block(32, 16, 1, 1)
        self.mbconv2 = self._make_mbconv_block(16, 24, 2, 6)
        self.mbconv3 = self._make_mbconv_block(24, 40, 2, 6)
        self.mbconv4 = self._make_mbconv_block(40, 80, 2, 6)
        self.mbconv5 = self._make_mbconv_block(80, 112, 1, 6)
        self.mbconv6 = self._make_mbconv_block(112, 192, 2, 6)
        self.mbconv7 = self._make_mbconv_block(192, 320, 1, 6)
        
        # Final conv + BN
        self.conv2 = TritonConv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = TritonBatchNorm2d(1280)
        
        # Final FC layer
        self.fc = TritonLinear(1280, num_classes)
        
        # Activation
        self.relu6 = TritonReLU6()
        self.adaptive_avg_pool = TritonAdaptiveAvgPool2d()

    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        hidden_dim = round(in_channels * expand_ratio)
        return nn.Sequential(
            TritonConv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            TritonBatchNorm2d(hidden_dim),
            self.relu6,
            TritonConv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            TritonBatchNorm2d(hidden_dim),
            self.relu6,
            TritonConv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            TritonBatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        x = self.relu6(self.bn1(self.conv1(x)))
        
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)
        
        x = self.relu6(self.bn2(self.conv2(x)))
        x = self.adaptive_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x