import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    x_stride_0, x_stride_1, x_stride_2, x_stride_3,
    w_stride_0, w_stride_1, w_stride_2, w_stride_3,
    out_stride_0, out_stride_1, out_stride_2, out_stride_3,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, stride, padding,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Compute the indices for this block
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    # Compute output indices
    h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    # Compute output shape
    out_h = (height - 1) * stride + kernel_size - 2 * padding
    out_w = (width - 1) * stride + kernel_size - 2 * padding

    # Create mask for valid output indices
    h_mask = (h < out_h) & (h >= 0)
    w_mask = (w < out_w) & (w >= 0)
    c_mask = (c < out_channels) & (c >= 0)

    # Create indices for input and kernel
    h_in = h // stride
    w_in = w // stride
    h_kernel = h % stride
    w_kernel = w % stride

    # Mask for valid input indices
    h_in_mask = (h_in < height) & (h_in >= 0)
    w_in_mask = (w_in < width) & (w_in >= 0)
    h_kernel_mask = (h_kernel < kernel_size) & (h_kernel >= 0)
    w_kernel_mask = (w_kernel < kernel_size) & (w_kernel >= 0)

    # Combine masks
    h_mask = h_mask & h_in_mask & h_kernel_mask
    w_mask = w_mask & w_in_mask & w_kernel_mask

    # Create output and input indices
    out_indices = (pid_batch * out_stride_0 +
                   c * out_stride_1 +
                   h * out_stride_2 +
                   w * out_stride_3)

    # Load input and weights
    x = tl.load(x_ptr +
                pid_batch * x_stride_0 +
                (h_in * x_stride_2 + w_in * x_stride_3) +
                c * x_stride_1,
                mask=h_in_mask & w_in_mask & c_mask,
                other=0.0)

    w = tl.load(w_ptr +
                (c * w_stride_1 +
                 h_kernel * w_stride_2 +
                 w_kernel * w_stride_3),
                mask=h_kernel_mask & w_kernel_mask & c_mask,
                other=0.0)

    # Compute output
    # Perform convolution
    out = x * w

    # Store output
    tl.store(out_ptr + out_indices, out, mask=h_mask & w_mask & c_mask)


@triton.jit
def maxpool_kernel(
    x_ptr,
    out_ptr,
    x_stride_0, x_stride_1, x_stride_2, x_stride_3,
    out_stride_0, out_stride_1, out_stride_2, out_stride_3,
    batch_size, in_channels, height, width,
    kernel_size, stride,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Compute the indices for this block
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    # Compute output indices
    h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    # Compute output shape
    out_h = (height - kernel_size) // stride + 1
    out_w = (width - kernel_size) // stride + 1

    # Create mask for valid output indices
    h_mask = (h < out_h) & (h >= 0)
    w_mask = (w < out_w) & (w >= 0)
    c_mask = (c < in_channels) & (c >= 0)

    # Compute input indices
    h_in = h * stride
    w_in = w * stride

    # Create mask for valid input indices
    h_in_mask = (h_in < height) & (h_in >= 0)
    w_in_mask = (w_in < width) & (w_in >= 0)

    # Combine masks
    h_mask = h_mask & h_in_mask
    w_mask = w_mask & w_in_mask

    # Compute input indices
    x_indices = (pid_batch * x_stride_0 +
                 c * x_stride_1 +
                 (h_in * x_stride_2 + w_in * x_stride_3))

    # Load input data
    x = tl.load(x_ptr + x_indices,
                mask=h_in_mask & w_in_mask & c_mask,
                other=-float('inf'))

    # Compute maxpool
    out = tl.max(x, axis=0)

    # Store output
    out_indices = (pid_batch * out_stride_0 +
                   c * out_stride_1 +
                   h * out_stride_2 +
                   w * out_stride_3)
    tl.store(out_ptr + out_indices, out, mask=h_mask & w_mask & c_mask)


@triton.jit
def hardtanh_kernel(
    x_ptr,
    out_ptr,
    x_stride_0, x_stride_1, x_stride_2, x_stride_3,
    out_stride_0, out_stride_1, out_stride_2, out_stride_3,
    batch_size, in_channels, height, width,
    min_val, max_val,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Compute the indices for this block
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    # Compute output indices
    h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    # Create mask for valid output indices
    h_mask = (h < height) & (h >= 0)
    w_mask = (w < width) & (w >= 0)
    c_mask = (c < in_channels) & (c >= 0)

    # Compute input indices
    x_indices = (pid_batch * x_stride_0 +
                 c * x_stride_1 +
                 h * x_stride_2 +
                 w * x_stride_3)

    # Load input data
    x = tl.load(x_ptr + x_indices,
                mask=h_mask & w_mask & c_mask,
                other=0.0)

    # Clamp the values
    out = tl.clamp(x, min_val, max_val)

    # Store output
    out_indices = (pid_batch * out_stride_0 +
                   c * out_stride_1 +
                   h * out_stride_2 +
                   w * out_stride_3)
    tl.store(out_ptr + out_indices, out, mask=h_mask & w_mask & c_mask)


@triton.jit
def mean_kernel(
    x_ptr,
    out_ptr,
    x_stride_0, x_stride_1, x_stride_2, x_stride_3,
    out_stride_0, out_stride_1, out_stride_2, out_stride_3,
    batch_size, in_channels, height, width,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Compute the indices for this block
    pid_batch = tl.program_id(0)
    pid_c = tl.program_id(1)

    # Compute output indices
    c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    # Create mask for valid output indices
    c_mask = (c < in_channels) & (c >= 0)

    # Initialize output
    sum_val = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)

    # Load input data and accumulate
    for h in range(0, height, BLOCK_SIZE_H):
        h_idx = h + tl.arange(0, BLOCK_SIZE_H)
        h_mask = (h_idx < height) & (h_idx >= 0)

        for w in range(0, width, BLOCK_SIZE_W):
            w_idx = w + tl.arange(0, BLOCK_SIZE_W)
            w_mask = (w_idx < width) & (w_idx >= 0)

            # Compute input indices
            x_indices = (pid_batch * x_stride_0 +
                         c * x_stride_1 +
                         h_idx * x_stride_2 +
                         w_idx * x_stride_3)

            # Load input data
            x = tl.load(x_ptr + x_indices,
                        mask=h_mask[:, None] & w_mask[None, :] & c_mask[None, :],
                        other=0.0)

            sum_val += tl.sum(x, axis=(0, 1))

    # Compute mean
    out = sum_val / (height * width)

    # Store output
    out_indices = (pid_batch * out_stride_0 +
                   c * out_stride_1 +
                   0 * out_stride_2 +
                   0 * out_stride_3)
    tl.store(out_ptr + out_indices, out, mask=c_mask)


@triton.jit
def tanh_kernel(
    x_ptr,
    out_ptr,
    x_stride_0, x_stride_1, x_stride_2, x_stride_3,
    out_stride_0, out_stride_1, out_stride_2, out_stride_3,
    batch_size, in_channels, height, width,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Compute the indices for this block
    pid_batch = tl.program_id(0)
    pid_c = tl.program_id(1)

    # Compute output indices
    c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    # Create mask for valid output indices
    c_mask = (c < in_channels) & (c >= 0)

    # Compute input indices
    x_indices = (pid_batch * x_stride_0 +
                 c * x_stride_1 +
                 0 * x_stride_2 +
                 0 * x_stride_3)

    # Load input data
    x = tl.load(x_ptr + x_indices,
                mask=c_mask,
                other=0.0)

    # Compute tanh
    out = tl.tanh(x)

    # Store output
    out_indices = (pid_batch * out_stride_0 +
                   c * out_stride_1 +
                   0 * out_stride_2 +
                   0 * out_stride_3)
    tl.store(out_ptr + out_indices, out, mask=c_mask)


def triton_conv_transpose(x, w, out, height, width, kernel_size, stride, padding):
    # Ensure inputs are contiguous
    x = x.contiguous()
    w = w.contiguous()

    # Determine block size and grid
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 16

    # Grid dimensions
    grid = lambda meta: (x.shape[0],  # batch
                         (meta["out_h"] + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],  # height
                         (meta["out_w"] + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],  # width
                         (meta["out_c"] + meta["BLOCK_SIZE_C"] - 1) // meta["BLOCK_SIZE_C"])  # channels

    # Compute output shape
    out_h = (height - 1) * stride + kernel_size - 2 * padding
    out_w = (width - 1) * stride + kernel_size - 2 * padding

    # Launch kernel
    conv_transpose_kernel[grid](
        x, w, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        x.shape[0], x.shape[1], w.shape[0], height, width,
        kernel_size, stride, padding,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        out_h=out_h,
        out_w=out_w,
        out_c=out.shape[1],
    )
    return out


def triton_maxpool(x, out, kernel_size, stride):
    # Ensure inputs are contiguous
    x = x.contiguous()

    # Determine block size and grid
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    BLOCK_SIZE_C = 16

    # Grid dimensions
    grid = lambda meta: (x.shape[0],  # batch
                         (meta["out_h"] + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],  # height
                         (meta["out_w"] + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],  # width
                         (meta["in_c"] + meta["BLOCK_SIZE_C"] - 1) // meta["BLOCK_SIZE_C"])  # channels

    # Compute output shape
    out_h = (x.shape[2] - kernel_size) // stride + 1
    out_w = (x.shape[3] - kernel_size) // stride + 1

    # Launch kernel
    maxpool_kernel[grid](
        x, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        x.shape[0], x.shape[1], x.shape[2], x.shape[3],
        kernel_size, stride,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        out_h=out_h,
        out_w=out_w,
        in_c=x.shape[1],
    )
    return out


def triton_hardtanh(x, out, min_val, max_val):
    # Ensure inputs are contiguous
    x = x.contiguous()

    # Determine block size and grid
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 16

    # Grid dimensions
    grid = lambda meta: (x.shape[0],  # batch
                         (meta["out_h"] + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],  # height
                         (meta["out_w"] + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],  # width
                         (meta["out_c"] + meta["BLOCK_SIZE_C"] - 1) // meta["BLOCK_SIZE_C"])  # channels

    # Launch kernel
    hardtanh_kernel[grid](
        x, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        x.shape[0], x.shape[1], x.shape[2], x.shape[3],
        min_val, max_val,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        out_h=x.shape[2],
        out_w=x.shape[3],
        out_c=x.shape[1],
    )
    return out


def triton_mean(x, out, height, width):
    # Ensure inputs are contiguous
    x = x.contiguous()

    # Determine block size and grid
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 16

    # Grid dimensions
    grid = lambda meta: (x.shape[0],  # batch
                         (meta["out_c"] + meta["BLOCK_SIZE_C"] - 1) // meta["BLOCK_SIZE_C"])  # channels

    # Launch kernel
    mean_kernel[grid](
        x, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        x.shape[0], x.shape[1], height, width,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    return out


def triton_tanh(x, out):
    # Ensure inputs are contiguous
    x = x.contiguous()

    # Determine block size and grid
    BLOCK_SIZE_C = 32

    # Grid dimensions
    grid = lambda meta: (x.shape[0],  # batch
                         (meta["out_c"] + meta["BLOCK_SIZE_C"] - 1) // meta["BLOCK_SIZE_C"])  # channels

    # Launch kernel
    tanh_kernel[grid](
        x, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        x.shape[0], x.shape[1], x.shape[2], x.shape[3],
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        # Initialize conv transpose weights
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        # Get input shape
        batch_size, in_channels, height, width = x.shape

        # Compute output shapes
        out_height = (height - 1) * self.conv_transpose.stride[0] + self.conv_transpose.kernel_size[0] - 2 * self.conv_transpose.padding[0]
        out_width = (width - 1) * self.conv_transpose.stride[1] + self.conv_transpose.kernel_size[1] - 2 * self.conv_transpose.padding[1]

        # Initialize output tensors
        x = self.conv_transpose(x)  # Conv transpose
        x = triton_maxpool(x, torch.empty_like(x), self.maxpool_kernel_size, self.maxpool_stride)
        x = triton_hardtanh(x, torch.empty_like(x), self.hardtanh_min, self.hardtanh_max)
        x = triton_mean(x, torch.empty_like(x), height, width)
        x = triton_tanh(x, torch.empty_like(x))
        return x