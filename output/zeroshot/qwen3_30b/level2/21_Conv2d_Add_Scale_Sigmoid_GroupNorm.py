import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Pointer to input tensor (batch, in_channels, height, width)
    w_ptr,  # Pointer to kernel weights (out_channels, in_channels, kernel_size, kernel_size)
    y_ptr,  # Pointer to output tensor (batch, out_channels, height, width)
    bias_ptr,  # Pointer to bias (out_channels,)
    scale_ptr,  # Pointer to scale (out_channels,)
    batch, in_channels, out_channels, height, width, kernel_size, 
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Block indices
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    # Compute the starting indices
    start_h = pid_h * BLOCK_SIZE_H
    start_w = pid_w * BLOCK_SIZE_W
    start_c = pid_c * BLOCK_SIZE_C

    # Define offsets for current block
    h_offsets = start_h + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = start_w + tl.arange(0, BLOCK_SIZE_W)
    c_offsets = start_c + tl.arange(0, BLOCK_SIZE_C)

    # Mask for valid h, w, c indices
    h_mask = h_offsets < height
    w_mask = w_offsets < width
    c_mask = c_offsets < out_channels

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C), dtype=tl.float32)

    # Loop over input channels and kernel size
    for c_in in range(0, in_channels, 8):
        # Load kernel weights: (out_channels, in_channels, kernel_size, kernel_size)
        # We use 8x8 tiling and load in batches of 8 channels
        c_in_end = min(c_in + 8, in_channels)
        k_h = tl.arange(0, kernel_size)
        k_w = tl.arange(0, kernel_size)
        w_ptrs = w_ptr + (
            (tl.broadcast_to(c_offsets[:, None, None], (BLOCK_SIZE_C, kernel_size, kernel_size)) * in_channels + 
             tl.broadcast_to(c_in, (BLOCK_SIZE_C, kernel_size, kernel_size))) * kernel_size * kernel_size + 
            (tl.broadcast_to(k_h[None, :, None], (BLOCK_SIZE_C, kernel_size, kernel_size)) * kernel_size + 
             tl.broadcast_to(k_w[None, None, :], (BLOCK_SIZE_C, kernel_size, kernel_size)))
        )
        w_vals = tl.load(w_ptrs, mask=(c_mask[:, None, None] & (k_h[None, :, None] < kernel_size) & (k_w[None, None, :] < kernel_size)), other=0.0)

        # Load input patch: (in_channels, kernel_size, kernel_size) centered at (h, w)
        x_ptrs = x_ptr + (
            (pid_b * in_channels * height * width + 
             tl.broadcast_to(c_in, (BLOCK_SIZE_H, BLOCK_SIZE_W, kernel_size, kernel_size)) * height * width + 
             tl.broadcast_to(h_offsets[:, None, None], (BLOCK_SIZE_H, BLOCK_SIZE_W, kernel_size, kernel_size)) * width + 
             tl.broadcast_to(w_offsets[None, :, None], (BLOCK_SIZE_H, BLOCK_SIZE_W, kernel_size, kernel_size)))
        )
        x_vals = tl.load(x_ptrs, mask=(h_mask[:, None, None] & w_mask[None, :, None] & (k_h[None, None, :] < kernel_size) & (k_w[None, None, :] < kernel_size)), other=0.0)

        # Compute dot product: (in_channels, kernel_size, kernel_size) x (in_channels, kernel_size, kernel_size)
        # We use matrix multiplication in a tiled way
        for k in range(kernel_size * kernel_size):
            k_h_idx = k // kernel_size
            k_w_idx = k % kernel_size
            x_kernel = x_vals[:, :, k_h_idx, k_w_idx]
            w_kernel = w_vals[:, :, k_h_idx, k_w_idx]
            acc += tl.dot(x_kernel, w_kernel.T)  # shape (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C)

    # Handle bias and scale
    bias_vals = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0)
    scale_vals = tl.load(scale_ptr + c_offsets, mask=c_mask, other=0.0)

    # Apply bias and scale
    acc = acc + bias_vals[None, None, :]
    acc = acc * scale_vals[None, None, :]

    # Apply sigmoid in a fused, stable way
    # Use log_sigmoid to avoid overflow
    # f(x) = sigmoid(x) = 1 / (1 + exp(-x))
    # We use: sigmoid(x) = 0.5 * (1 + tanh(0.5 * x)) for numerical stability
    acc = tl.tanh(0.5 * acc) * 0.5 + 0.5

    # Store output
    out_ptrs = y_ptr + (
        pid_b * out_channels * height * width + 
        tl.broadcast_to(c_offsets, (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C)) * height * width + 
        tl.broadcast_to(h_offsets[:, None, None], (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C)) * width + 
        tl.broadcast_to(w_offsets[None, :, None], (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C))
    )
    tl.store(out_ptrs, acc, mask=(h_mask[:, None, None] & w_mask[None, :, None] & c_mask[None, None, :]))


@triton.jit
def group_norm_kernel(
    x_ptr,  # Input (batch, out_channels, height, width)
    out_ptr,  # Output (batch, out_channels, height, width)
    mean_ptr,  # Mean per group (batch, out_channels // num_groups, height, width)
    inv_std_ptr,  # Inverse standard deviation per group
    batch, out_channels, height, width, num_groups, 
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    start_h = pid_h * BLOCK_SIZE_H
    start_w = pid_w * BLOCK_SIZE_W
    start_c = pid_c * BLOCK_SIZE_C

    h_offsets = start_h + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = start_w + tl.arange(0, BLOCK_SIZE_W)
    c_offsets = start_c + tl.arange(0, BLOCK_SIZE_C)

    h_mask = h_offsets < height
    w_mask = w_offsets < width
    c_mask = c_offsets < out_channels

    # Group index for this channel
    group_id = c_offsets // (out_channels // num_groups)

    # Load input
    x_ptrs = x_ptr + (
        pid_b * out_channels * height * width + 
        tl.broadcast_to(c_offsets, (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C)) * height * width + 
        tl.broadcast_to(h_offsets[:, None, None], (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C)) * width + 
        tl.broadcast_to(w_offsets[None, :, None], (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C))
    )
    x_vals = tl.load(x_ptrs, mask=(h_mask[:, None, None] & w_mask[None, :, None] & c_mask[None, None, :]))

    # Load mean and inverse std for this group
    mean_ptrs = mean_ptr + (
        pid_b * (out_channels // num_groups) * height * width + 
        tl.broadcast_to(group_id, (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C)) * height * width + 
        tl.broadcast_to(h_offsets[:, None, None], (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C)) * width + 
        tl.broadcast_to(w_offsets[None, :, None], (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C))
    )
    inv_std_ptrs = inv_std_ptr + (
        pid_b * (out_channels // num_groups) * height * width + 
        tl.broadcast_to(group_id, (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C)) * height * width + 
        tl.broadcast_to(h_offsets[:, None, None], (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C)) * width + 
        tl.broadcast_to(w_offsets[None, :, None], (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C))
    )

    mean_vals = tl.load(mean_ptrs, mask=(h_mask[:, None, None] & w_mask[None, :, None] & c_mask[None, None, :]))
    inv_std_vals = tl.load(inv_std_ptrs, mask=(h_mask[:, None, None] & w_mask[None, :, None] & c_mask[None, None, :]))

    # Normalize
    out_vals = (x_vals - mean_vals) * inv_std_vals

    # Store output
    out_ptrs = out_ptr + (
        pid_b * out_channels * height * width + 
        tl.broadcast_to(c_offsets, (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C)) * height * width + 
        tl.broadcast_to(h_offsets[:, None, None], (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C)) * width + 
        tl.broadcast_to(w_offsets[None, :, None], (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C))
    )
    tl.store(out_ptrs, out_vals, mask=(h_mask[:, None, None] & w_mask[None, :, None] & c_mask[None, None, :]))


def triton_conv2d(x, w, bias, scale, kernel_size, height, width, in_channels, out_channels):
    # Ensure contiguous on GPU
    x = x.contiguous()
    w = w.contiguous()
    bias = bias.contiguous()
    scale = scale.contiguous()

    # Allocate output
    out = torch.empty_like(x, dtype=torch.bfloat16)

    # Define block sizes
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    BLOCK_SIZE_C = 32

    # Grid definition
    num_blocks_h = (height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    num_blocks_w = (width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    num_blocks_c = (out_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C

    grid = lambda meta: (x.size(0), num_blocks_h, num_blocks_w, num_blocks_c)

    # Launch kernel
    conv2d_kernel[grid](
        x, w, out, bias, scale,
        x.size(0), in_channels, out_channels, height, width, kernel_size,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W, BLOCK_SIZE_C=BLOCK_SIZE_C
    )

    return out


def triton_group_norm(x, mean, inv_std, num_groups):
    x = x.contiguous()
    mean = mean.contiguous()
    inv_std = inv_std.contiguous()

    out = torch.empty_like(x, dtype=torch.bfloat16)

    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    BLOCK_SIZE_C = 32

    num_blocks_h = (x.size(2) + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    num_blocks_w = (x.size(3) + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    num_blocks_c = (x.size(1) + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C

    grid = lambda meta: (x.size(0), num_blocks_h, num_blocks_w, num_blocks_c)

    group_norm_kernel[grid](
        x, out, mean, inv_std,
        x.size(0), x.size(1), x.size(2), x.size(3), num_groups,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W, BLOCK_SIZE_C=BLOCK_SIZE_C
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super().__init__()
        # Use same conv layer as in original
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.num_groups = num_groups
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = width = 256

    def forward(self, x):
        # Move to bfloat16 for better tensor core utilization
        x = x.to(torch.bfloat16)

        # 1. Conv2d + bias + scale + sigmoid fused in Triton
        out = triton_conv2d(
            x, self.conv.weight, self.bias, self.scale,
            self.kernel_size, self.height, self.width, self.in_channels, self.out_channels
        )

        # 2. GroupNorm: Compute mean and std, then normalize
        # Compute mean: (batch, out_channels // num_groups, height, width)
        group_size = self.out_channels // self.num_groups
        x_reshaped = out.view(out.size(0), self.num_groups, group_size, out.size(2), out.size(3))
        mean = x_reshaped.mean(dim=2, keepdim=True)
        std = x_reshaped.var(dim=2, keepdim=True, unbiased=False).sqrt() + 1e-5
        inv_std = 1.0 / std

        # Fuse normalization with Triton
        out_norm = triton_group_norm(out, mean, inv_std, self.num_groups)

        return out_norm