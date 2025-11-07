import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weight tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    in_channels,  # Input channels
    out_channels,  # Output channels
    D, H, W,  # Spatial dimensions
    kernel_size,  # Size of the kernel
    groups,  # Number of groups
    stride_d, stride_h, stride_w,  # Strides
    padding_d, padding_h, padding_w,  # Padding
    output_padding_d, output_padding_h, output_padding_w,  # Output padding
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Thread indices
    pid_b = tl.program_id(0)  # Batch index
    pid_c = tl.program_id(1)  # Output channel index
    pid_d = tl.program_id(2)  # D dimension index
    pid_h = tl.program_id(3)  # H dimension index
    pid_w = tl.program_id(4)  # W dimension index

    # Block sizes for tiling
    grid_d = (D + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D
    grid_h = (H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (W + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W

    # Block offsets
    block_d_offset = pid_d * BLOCK_SIZE_D
    block_h_offset = pid_h * BLOCK_SIZE_H
    block_w_offset = pid_w * BLOCK_SIZE_W

    # Channel offset for grouped convolution
    group_size = out_channels // groups
    group_id = pid_c // group_size
    c_offset = (pid_c % group_size) * (in_channels // groups)

    # Shared memory for weight and input tiles
    # We tile the kernel and input spatial dimensions
    x_tile = tl.load(
        x_ptr + pid_b * in_channels * D * H * W + c_offset * D * H * W,
        mask=tl.arange(0, BLOCK_SIZE_D)[:, None, None] < D,
        other=0.0
    ).to(tl.float32)

    w_tile = tl.load(
        w_ptr + (pid_c // group_size) * (in_channels // groups) * kernel_size * kernel_size * kernel_size +
        (pid_c % group_size) * kernel_size * kernel_size * kernel_size,
        mask=tl.arange(0, kernel_size)[:, None, None] < kernel_size,
        other=0.0
    ).to(tl.float32)

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Loop over kernel spatial dimensions
    for kd in range(kernel_size):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Compute input indices
                input_d = block_d_offset + kd - padding_d
                input_h = block_h_offset + kh - padding_h
                input_w = block_w_offset + kw - padding_w

                # Check bounds
                valid_d = (input_d >= 0) & (input_d < D)
                valid_h = (input_h >= 0) & (input_h < H)
                valid_w = (input_w >= 0) & (input_w < W)

                # Load input tile
                input_val = tl.load(
                    x_ptr + pid_b * in_channels * D * H * W + c_offset * D * H * W + 
                    input_d * H * W + input_h * W + input_w,
                    mask=valid_d & valid_h & valid_w,
                    other=0.0
                ).to(tl.float32)

                # Weight value
                weight_val = w_tile[kd, kh, kw]

                # Accumulate
                acc += input_val * weight_val

    # Apply stride and output padding
    out_d = block_d_offset * stride_d + output_padding_d
    out_h = block_h_offset * stride_h + output_padding_h
    out_w = block_w_offset * stride_w + output_padding_w

    # Store output
    out_ptrs = out_ptr + pid_b * out_channels * D * H * W + pid_c * D * H * W + out_d * H * W + out_h * W + out_w
    tl.store(out_ptrs, acc, mask=tl.arange(0, BLOCK_SIZE_D)[:, None, None] < D)


@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def group_norm_kernel(
    x_ptr,
    mean_ptr,
    invstd_ptr,
    out_ptr,
    batch_size,
    channels,
    D, H, W,
    num_groups,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Thread indices
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    # Compute global shape
    total_elements = batch_size * channels * D * H * W
    group_size = channels // num_groups
    group_id = pid_c // group_size

    # Group mean and variance
    group_mean = tl.load(mean_ptr + group_id * D * H * W + pid_d * H * W + pid_h * W + pid_w)
    group_invstd = tl.load(invstd_ptr + group_id * D * H * W + pid_d * H * W + pid_h * W + pid_w)

    # Load input
    input_offset = pid_b * channels * D * H * W + pid_c * D * H * W + pid_d * H * W + pid_h * W + pid_w
    x = tl.load(x_ptr + input_offset)
    # Normalize
    out = (x - group_mean) * group_invstd
    # Store output
    tl.store(out_ptr + input_offset, out)


def triton_conv_transpose3d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    batch_size, in_channels, D, H, W = x.shape
    out_channels, _, kernel_size, _, _ = weight.shape

    # Ensure input is contiguous
    x = x.contiguous()
    weight = weight.contiguous()

    # Allocate output
    out = torch.empty(batch_size, out_channels, D, H, W, dtype=x.dtype, device=x.device)

    # Kernel parameters
    stride_d = stride_h = stride_w = stride if isinstance(stride, int) else stride[0]
    padding_d = padding_h = padding_w = padding if isinstance(padding, int) else padding[0]
    output_padding_d = output_padding_h = output_padding_w = output_padding if isinstance(output_padding, int) else output_padding[0]

    # Grid configuration
    grid_d = (D + 127) // 128
    grid_h = (H + 127) // 128
    grid_w = (W + 127) // 128
    grid_c = (out_channels + 127) // 128
    grid_b = batch_size

    # Block sizes
    BLOCK_SIZE_D = 128
    BLOCK_SIZE_H = 128
    BLOCK_SIZE_W = 128
    BLOCK_SIZE_C = 128

    # Launch kernel
    conv_transpose3d_kernel[
        (grid_b, grid_c, grid_d, grid_h, grid_w)
    ](
        x, weight, out, batch_size, in_channels, out_channels,
        D, H, W, kernel_size, groups, stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w, output_padding_d, output_padding_h, output_padding_w,
        BLOCK_SIZE_D=BLOCK_SIZE_D, BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W, BLOCK_SIZE_C=BLOCK_SIZE_C
    )

    # Add bias if provided
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1)

    return out


def triton_relu(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    BLOCK_SIZE = 128
    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_group_norm(x, weight=None, bias=None, num_groups=8, eps=1e-5):
    batch_size, channels, D, H, W = x.shape
    x = x.contiguous()

    # Compute group mean and inverse std
    group_size = channels // num_groups
    x_reshaped = x.view(batch_size, num_groups, group_size, D, H, W)
    mean = x_reshaped.mean(dim=(2, 3, 4), keepdim=True)
    var = x_reshaped.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
    invstd = 1.0 / torch.sqrt(var + eps)

    # Reshape for Triton
    mean = mean.view(batch_size, channels, D, H, W)
    invstd = invstd.view(batch_size, channels, D, H, W)

    # Allocate output
    out = torch.empty_like(x)

    # Grid configuration
    grid_d = (D + 127) // 128
    grid_h = (H + 127) // 128
    grid_w = (W + 127) // 128
    grid_c = (channels + 127) // 128
    grid_b = batch_size

    # Launch kernel
    group_norm_kernel[
        (grid_b, grid_c, grid_d, grid_h, grid_w)
    ](
        x, mean, invstd, out, batch_size, channels, D, H, W, num_groups, eps,
        BLOCK_SIZE=128
    )

    # Scale and shift
    if weight is not None:
        out = out * weight.view(1, channels, 1, 1, 1)
    if bias is not None:
        out = out + bias.view(1, channels, 1, 1, 1)

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

    def forward(self, x):
        x = triton_conv_transpose3d(x, self.conv_transpose.weight, self.conv_transpose.bias, stride=self.conv_transpose.stride, padding=self.conv_transpose.padding, output_padding=self.conv_transpose.output_padding, groups=self.conv_transpose.groups)
        x = triton_relu(x)
        x = triton_group_norm(x, self.group_norm.weight, self.group_norm.bias, num_groups=self.group_norm.num_groups, eps=self.group_norm.eps)
        return x