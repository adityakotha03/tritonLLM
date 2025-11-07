import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, in_channels, out_channels,
    depth, height, width,
    kernel_size, stride, padding,
    BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr, BLOCK_SIZE_IN: tl.constexpr,
    TILE_SIZE_D: tl.constexpr, TILE_SIZE_H: tl.constexpr, TILE_SIZE_W: tl.constexpr,
    TILE_SIZE_OUT: tl.constexpr, TILE_SIZE_IN: tl.constexpr,
):
    # Block indices
    pid_d = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c_out = tl.program_id(3)
    pid_c_in = tl.program_id(4)

    # Grid dimensions
    num_pid_d = tl.cdiv(depth, BLOCK_SIZE_D)
    num_pid_h = tl.cdiv(height, BLOCK_SIZE_H)
    num_pid_w = tl.cdiv(width, BLOCK_SIZE_W)
    num_pid_c_out = tl.cdiv(out_channels, BLOCK_SIZE_OUT)
    num_pid_c_in = tl.cdiv(in_channels, BLOCK_SIZE_IN)

    # Compute offsets
    offset_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    offset_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offset_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    offset_c_out = pid_c_out * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    offset_c_in = pid_c_in * BLOCK_SIZE_IN + tl.arange(0, BLOCK_SIZE_IN)

    # Mask for out-of-bounds
    mask_d = offset_d < depth
    mask_h = offset_h < height
    mask_w = offset_w < width
    mask_c_out = offset_c_out < out_channels
    mask_c_in = offset_c_in < in_channels

    # Tile indices
    tile_d = offset_d // TILE_SIZE_D
    tile_h = offset_h // TILE_SIZE_H
    tile_w = offset_w // TILE_SIZE_W
    tile_c_out = offset_c_out // TILE_SIZE_OUT
    tile_c_in = offset_c_in // TILE_SIZE_IN

    # Global offset
    global_offset = (
        (pid_d * num_pid_h * num_pid_w * num_pid_c_out * num_pid_c_in + 
         pid_h * num_pid_w * num_pid_c_out * num_pid_c_in + 
         pid_w * num_pid_c_out * num_pid_c_in + 
         pid_c_out * num_pid_c_in + 
         pid_c_in) * BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W * BLOCK_SIZE_OUT * BLOCK_SIZE_IN
    )

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_OUT), dtype=tl.float32)

    # Load input tile
    for kd in range(kernel_size):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Compute input indices
                d_in = offset_d + kd - padding
                h_in = offset_h + kh - padding
                w_in = offset_w + kw - padding

                # Mask valid input indices
                mask_in_d = (d_in >= 0) & (d_in < depth)
                mask_in_h = (h_in >= 0) & (h_in < height)
                mask_in_w = (w_in >= 0) & (w_in < width)

                # Combine masks
                valid = mask_in_d & mask_in_h & mask_in_w

                # Load input data
                idx_in = (d_in * height * width + h_in * width + w_in) * in_channels + offset_c_in
                x = tl.load(x_ptr + idx_in, mask=valid[:, None] & mask_c_in[None, :], other=0.0)

                # Load kernel data
                idx_k = (kd * kernel_size * kernel_size + kh * kernel_size + kw) * out_channels * in_channels + \
                        (offset_c_out[:, None] * in_channels + offset_c_in[None, :])
                w = tl.load(w_ptr + idx_k, mask=mask_c_out[:, None] & mask_c_in[None, :], other=0.0)

                # Compute partial product
                acc += tl.dot(x, w, allow_tf32=True)

    # Store output
    idx_out = (offset_d * height * width + offset_h * width + offset_w) * out_channels + offset_c_out
    tl.store(out_ptr + idx_out, acc, mask=mask_d[:, None, None, None] & mask_h[None, :, None, None] & 
             mask_w[None, None, :, None] & mask_c_out[None, None, None, :], other=0.0)


@triton.jit
def max_pool3d_kernel(
    x_ptr, out_ptr,
    batch_size, in_channels, depth, height, width,
    kernel_size, stride,
    BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    TILE_SIZE_D: tl.constexpr, TILE_SIZE_H: tl.constexpr, TILE_SIZE_W: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    offset_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    offset_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offset_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    offset_c = pid_c * tl.arange(0, 1)  # Broadcast over channels

    mask_d = offset_d < depth
    mask_h = offset_h < height
    mask_w = offset_w < width

    # Pool output indices
    out_d = offset_d // stride
    out_h = offset_h // stride
    out_w = offset_w // stride

    # Bounds check for output
    mask_out_d = out_d < (depth + padding) // stride
    mask_out_h = out_h < (height + padding) // stride
    mask_out_w = out_w < (width + padding) // stride

    # Global index
    idx = (out_d * (height // stride) * (width // stride) + out_h * (width // stride) + out_w) * in_channels + offset_c
    out_ptr = out_ptr + idx

    # Local index
    idx_local = (offset_d * height * width + offset_h * width + offset_w) * in_channels + offset_c
    x_ptr = x_ptr + idx_local

    # Initialize output with min
    out = tl.full((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), -float("inf"), dtype=tl.float32)

    # Iterate over kernel window
    for kd in range(kernel_size):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                d = offset_d + kd
                h = offset_h + kh
                w = offset_w + kw

                mask_valid = (d < depth) & (h < height) & (w < width)
                x = tl.load(x_ptr + (d * height * width + h * width + w) * in_channels + offset_c, 
                            mask=mask_valid[:, None] & mask_c[None, :], other=-float("inf"))

                out = tl.max(out, x)

    # Store output
    tl.store(out_ptr, out, mask=mask_out_d[:, None, None] & mask_out_h[None, :, None] & mask_out_w[None, None, :])


@triton.jit
def logsumexp_kernel(
    x_ptr, out_ptr,
    batch_size, in_channels, depth, height, width,
    BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    TILE_SIZE_D: tl.constexpr, TILE_SIZE_H: tl.constexpr, TILE_SIZE_W: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    offset_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    offset_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offset_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    mask_d = offset_d < depth
    mask_h = offset_h < height
    mask_w = offset_w < width

    # Compute output indices
    idx_out = (offset_d * height * width + offset_h * width + offset_w) * in_channels

    # Load input
    x = tl.load(x_ptr + idx_out, mask=mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :], other=-float("inf"))

    # Compute logsumexp
    max_val = tl.max(x, axis=0)
    x_centered = x - max_val
    exp_sum = tl.sum(tl.exp(x_centered), axis=0)
    out = max_val + tl.log(exp_sum)

    # Store result
    tl.store(out_ptr + idx_out, out, mask=mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :])


@triton.jit
def relu_kernel(
    x_ptr, out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv3d(x, w, stride, padding, kernel_size):
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA"
    x = x.contiguous()
    w = w.contiguous()

    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, _, _, _ = w.shape

    out = torch.empty(batch_size, out_channels, depth, height, width, device=x.device, dtype=x.dtype)

    # Grid
    BLOCK_SIZE_D = 16
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_OUT = 16
    BLOCK_SIZE_IN = 16
    TILE_SIZE_D = 4
    TILE_SIZE_H = 4
    TILE_SIZE_W = 4
    TILE_SIZE_OUT = 4
    TILE_SIZE_IN = 4

    grid = (
        tl.cdiv(depth, BLOCK_SIZE_D),
        tl.cdiv(height, BLOCK_SIZE_H),
        tl.cdiv(width, BLOCK_SIZE_W),
        tl.cdiv(out_channels, BLOCK_SIZE_OUT),
        tl.cdiv(in_channels, BLOCK_SIZE_IN)
    )

    conv3d_kernel[grid](
        x, w, out,
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_size, stride, padding,
        BLOCK_SIZE_D=BLOCK_SIZE_D, BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_OUT=BLOCK_SIZE_OUT, BLOCK_SIZE_IN=BLOCK_SIZE_IN,
        TILE_SIZE_D=TILE_SIZE_D, TILE_SIZE_H=TILE_SIZE_H, TILE_SIZE_W=TILE_SIZE_W,
        TILE_SIZE_OUT=TILE_SIZE_OUT, TILE_SIZE_IN=TILE_SIZE_IN,
    )
    return out


def triton_max_pool3d(x, kernel_size, stride):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()

    batch_size, in_channels, depth, height, width = x.shape

    out_depth = (depth + padding) // stride
    out_height = (height + padding) // stride
    out_width = (width + padding) // stride

    out = torch.empty(batch_size, in_channels, out_depth, out_height, out_width, device=x.device, dtype=x.dtype)

    BLOCK_SIZE_D = 16
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    TILE_SIZE_D = 4
    TILE_SIZE_H = 4
    TILE_SIZE_W = 4

    grid = (
        tl.cdiv(out_depth, BLOCK_SIZE_D),
        tl.cdiv(out_height, BLOCK_SIZE_H),
        tl.cdiv(out_width, BLOCK_SIZE_W),
        in_channels
    )

    max_pool3d_kernel[grid](
        x, out,
        batch_size, in_channels, depth, height, width,
        kernel_size, stride,
        BLOCK_SIZE_D=BLOCK_SIZE_D, BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W,
        TILE_SIZE_D=TILE_SIZE_D, TILE_SIZE_H=TILE_SIZE_H, TILE_SIZE_W=TILE_SIZE_W,
    )
    return out


def triton_logsumexp(x, dim=1, keepdim=True):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()

    if keepdim:
        out = torch.empty_like(x)
    else:
        out = torch.empty(x.shape[0], x.shape[2], x.shape[3], x.shape[4], device=x.device, dtype=x.dtype)

    BLOCK_SIZE_D = 16
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    TILE_SIZE_D = 4
    TILE_SIZE_H = 4
    TILE_SIZE_W = 4

    grid = (
        tl.cdiv(x.shape[2], BLOCK_SIZE_D),
        tl.cdiv(x.shape[3], BLOCK_SIZE_H),
        tl.cdiv(x.shape[4], BLOCK_SIZE_W),
    )

    logsumexp_kernel[grid](
        x, out,
        x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4],
        BLOCK_SIZE_D=BLOCK_SIZE_D, BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W,
        TILE_SIZE_D=TILE_SIZE_D, TILE_SIZE_H=TILE_SIZE_H, TILE_SIZE_W=TILE_SIZE_W,
    )
    return out


def triton_relu(x):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()

    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]

    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))

    def forward(self, x):
        # Conv3d
        x = triton_conv3d(x, self.weight, self.stride, self.padding, self.kernel_size)

        # Max Pooling
        x = triton_max_pool3d(x, 2, 2)

        # LogSumExp
        x = triton_logsumexp(x, dim=1, keepdim=True)

        # ReLU
        x = triton_relu(x)

        return x