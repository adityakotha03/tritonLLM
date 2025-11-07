import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, stride, padding,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
    TILE_H: tl.constexpr, TILE_W: tl.constexpr,
    TILE_C: tl.constexpr,
):
    # Thread indices
    pid_batch = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Calculate output spatial dimensions
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1

    # Check bounds
    if pid_batch >= batch_size or pid_c >= out_channels or pid_h >= out_height or pid_w >= out_width:
        return

    # Output pointer offset
    out_offset = pid_batch * out_channels * out_height * out_width + \
                 pid_c * out_height * out_width + \
                 pid_h * out_width + pid_w

    # Shared memory for input tile and kernel tile
    x_shared = tl.load(tl.make_block_ptr(x_ptr, shape=(batch_size, in_channels, height, width), strides=(in_channels * height * width, height * width, width, 1), offsets=(pid_batch, 0, pid_h * stride - padding, pid_w * stride - padding), block_shape=(1, in_channels, TILE_H, TILE_W), order=(0, 1, 2, 3)), mask=(1, in_channels, TILE_H, TILE_W), other=0.0)
    w_shared = tl.load(tl.make_block_ptr(w_ptr, shape=(out_channels, in_channels, kernel_size, kernel_size), strides=(in_channels * kernel_size * kernel_size, kernel_size * kernel_size, kernel_size, 1), offsets=(pid_c, 0, 0, 0), block_shape=(1, in_channels, kernel_size, kernel_size), order=(0, 1, 2, 3)), mask=(1, in_channels, kernel_size, kernel_size), other=0.0)

    # Accumulate output
    acc = tl.zeros((TILE_H, TILE_W), dtype=tl.float32)
    for c in range(0, in_channels, BLOCK_C):
        for h in range(0, kernel_size, BLOCK_H):
            for w in range(0, kernel_size, BLOCK_W):
                # Load input patch
                x_tile = tl.load(tl.make_block_ptr(x_ptr, shape=(batch_size, in_channels, height, width), strides=(in_channels * height * width, height * width, width, 1), offsets=(pid_batch, c, pid_h * stride - padding + h, pid_w * stride - padding + w), block_shape=(1, BLOCK_C, BLOCK_H, BLOCK_W), order=(0, 1, 2, 3)), mask=(1, BLOCK_C, BLOCK_H, BLOCK_W), other=0.0)
                # Load kernel patch
                w_tile = tl.load(tl.make_block_ptr(w_ptr, shape=(out_channels, in_channels, kernel_size, kernel_size), strides=(in_channels * kernel_size * kernel_size, kernel_size * kernel_size, kernel_size, 1), offsets=(pid_c, c, h, w), block_shape=(1, BLOCK_C, BLOCK_H, BLOCK_W), order=(0, 1, 2, 3)), mask=(1, BLOCK_C, BLOCK_H, BLOCK_W), other=0.0)

                # Compute dot product
                acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    # Store output
    tl.store(tl.make_block_ptr(out_ptr, shape=(batch_size, out_channels, out_height, out_width), strides=(out_channels * out_height * out_width, out_height * out_width, out_width, 1), offsets=(pid_batch, pid_c, pid_h, pid_w), block_shape=(1, 1, 1, 1), order=(0, 1, 2, 3)), acc, mask=(1, 1, 1, 1))


@triton.jit
def avg_pool_kernel(
    x_ptr, out_ptr,
    batch_size, channels, height, width,
    pool_size, stride,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    out_height = (height - pool_size) // stride + 1
    out_width = (width - pool_size) // stride + 1

    if pid_batch >= batch_size or pid_c >= channels or pid_h >= out_height or pid_w >= out_width:
        return

    # Calculate input offsets
    start_h = pid_h * stride
    start_w = pid_w * stride
    end_h = start_h + pool_size
    end_w = start_w + pool_size

    # Use shared memory for accumulation
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    for h in range(0, pool_size, BLOCK_H):
        for w in range(0, pool_size, BLOCK_W):
            h_idx = start_h + h
            w_idx = start_w + w
            # Load tile
            tile = tl.load(tl.make_block_ptr(x_ptr, shape=(batch_size, channels, height, width), strides=(channels * height * width, height * width, width, 1), offsets=(pid_batch, pid_c, h_idx, w_idx), block_shape=(1, 1, BLOCK_H, BLOCK_W), order=(0, 1, 2, 3)), mask=(1, 1, BLOCK_H, BLOCK_W), other=0.0)
            acc += tile

    # Compute average
    acc /= (pool_size * pool_size)

    # Store output
    tl.store(tl.make_block_ptr(out_ptr, shape=(batch_size, channels, out_height, out_width), strides=(channels * out_height * out_width, out_height * out_width, out_width, 1), offsets=(pid_batch, pid_c, pid_h, pid_w), block_shape=(1, 1, 1, 1), order=(0, 1, 2, 3)), acc, mask=(1, 1, 1, 1))


@triton.jit
def sigmoid_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def sum_kernel(
    x_ptr, out_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * channels * height * width
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sum_val = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, sum_val, mask=pid < batch_size)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16, 'BLOCK_C': 8, 'TILE_H': 16, 'TILE_W': 16, 'TILE_C': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32, 'BLOCK_C': 8, 'TILE_H': 32, 'TILE_W': 32, 'TILE_C': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16, 'BLOCK_C': 16, 'TILE_H': 16, 'TILE_W': 16, 'TILE_C': 16}, num_stages=4, num_warps=4),
    ],
    key=['height', 'width', 'out_channels', 'in_channels', 'kernel_size'],
)
def launch_conv2d_kernel(x, w, out, batch_size, in_channels, out_channels, height, width, kernel_size, stride, padding):
    grid = (batch_size, out_channels, (height + 2 * padding - kernel_size) // stride + 1, (width + 2 * padding - kernel_size) // stride + 1)
    conv2d_kernel[grid](
        x, w, out,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding,
        BLOCK_H=16, BLOCK_W=16, BLOCK_C=8, TILE_H=16, TILE_W=16, TILE_C=8
    )


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32}, num_stages=4, num_warps=8),
    ],
    key=['height', 'width', 'channels', 'pool_size']
)
def launch_avg_pool_kernel(x, out, batch_size, channels, height, width, pool_size, stride):
    out_height = (height - pool_size) // stride + 1
    out_width = (width - pool_size) // stride + 1
    grid = (batch_size, channels, out_height, out_width)
    avg_pool_kernel[grid](
        x, out,
        batch_size, channels, height, width,
        pool_size, stride,
        BLOCK_H=16, BLOCK_W=16
    )


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=8),
    ],
    key=['n_elements']
)
def launch_sigmoid_kernel(x, out, n_elements):
    grid = (triton.cdiv(n_elements, 1024),)
    sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=8),
    ],
    key=['n_elements']
)
def launch_sum_kernel(x, out, n_elements):
    grid = (triton.cdiv(n_elements, 1024),)
    sum_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)


def triton_conv2d(x, w, stride=1, padding=1):
    # Assume x is [B, C_in, H, W], w is [C_out, C_in, K, K]
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = w.shape
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    out = torch.empty(batch_size, out_channels, out_height, out_width, dtype=torch.bfloat16, device=x.device)

    # Launch kernel
    launch_conv2d_kernel(x, w, out, batch_size, in_channels, out_channels, height, width, kernel_size, stride, padding)
    return out


def triton_avg_pool(x, kernel_size=4, stride=4):
    batch_size, channels, height, width = x.shape
    out_height = (height - kernel_size) // stride + 1
    out_width = (width - kernel_size) // stride + 1
    out = torch.empty(batch_size, channels, out_height, out_width, dtype=torch.bfloat16, device=x.device)

    launch_avg_pool_kernel(x, out, batch_size, channels, height, width, kernel_size, stride)
    return out


def triton_sigmoid(x):
    out = torch.empty_like(x)
    launch_sigmoid_kernel(x, out, x.numel())
    return out


def triton_sum(x):
    # Sum over all dimensions
    sum_out = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    launch_sum_kernel(x, sum_out, x.numel())
    return sum_out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        # Initialize weights in bfloat16 for tensor core efficiency
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.bfloat16).cuda())
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        # Input to bfloat16 for computation
        x = x.to(torch.bfloat16)
        
        # Conv2d + AvgPool + Sigmoid + Sum fusion in Triton
        x = triton_conv2d(x, self.weight, stride=1, padding=1)
        x = triton_avg_pool(x, kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size)
        x = triton_sigmoid(x)
        x = triton_sum(x)
        
        return x