import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_reduction_kernel(
    x_ptr,  # input: (B, C, D, H, W), in BF16
    out_ptr,  # output: (B, C, H, W), in BF16
    B: tl.constexpr, C: tl.constexpr, D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Global index for the (b, h, w) position
    global_idx = tl.program_id(0) * BLOCK_SIZE + tl.thread_id()
    total_positions = B * H * W
    if global_idx >= total_positions:
        return

    # Convert global_idx to (b, h, w)
    b = global_idx // (H * W)
    h = (global_idx // W) % H
    w = global_idx % W

    # For each channel c, reduce over d
    for c in range(C):
        # Initialize min_val to a large value
        min_val = tl.full((1,), 1e10, dtype=tl.bfloat16)
        for d in range(D):
            # Load x[b, c, d, h, w]
            x_val = tl.load(x_ptr + b * C * D * H * W + c * D * H * W + d * H * W + h * W + w, 
                           mask=1, other=1e10)
            min_val = tl.minimum(min_val, x_val)
        # Store min_val to out[b, c, h, w]
        tl.store(out_ptr + b * C * H * W + c * H * W + h * W + w, min_val)


@triton.jit
def softmax_kernel(
    x_ptr,  # input: (B, C, H, W), in BF16
    out_ptr,  # output: (B, C, H, W), in BF16
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Global index for the (b, h, w) position
    global_idx = tl.program_id(0) * BLOCK_SIZE + tl.thread_id()
    total_positions = B * H * W
    if global_idx >= total_positions:
        return

    # Convert global_idx to (b, h, w)
    b = global_idx // (H * W)
    h = (global_idx // W) % H
    w = global_idx % W

    # For this (b, h, w), compute softmax over c
    # Step 1: find the maximum along c
    max_val = tl.full((1,), -1e10, dtype=tl.bfloat16)
    for c in range(C):
        x_val = tl.load(x_ptr + b * C * H * W + c * H * W + h * W + w, 
                       mask=1, other=-1e10)
        max_val = tl.maximum(max_val, x_val)

    # Step 2: compute the sum of exp(x - max)
    sum_exp = tl.full((1,), 0.0, dtype=tl.bfloat16)
    for c in range(C):
        x_val = tl.load(x_ptr + b * C * H * W + c * H * W + h * W + w, 
                       mask=1, other=0.0)
        # exp(x - max)
        exp_val = tl.exp(x_val - max_val)
        sum_exp = sum_exp + exp_val

    # Step 3: compute output = exp(x - max) / sum_exp
    for c in range(C):
        x_val = tl.load(x_ptr + b * C * H * W + c * H * W + h * W + w, 
                       mask=1, other=0.0)
        exp_val = tl.exp(x_val - max_val)
        out_val = exp_val / sum_exp
        tl.store(out_ptr + b * C * H * W + c * H * W + h * W + w, out_val)


def triton_min_reduction(x: torch.Tensor):
    """Apply min reduction along dim=2 (D) using Triton kernel."""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    B, C, D, H, W = x.shape
    # Convert to BF16
    x_bf16 = x.to(torch.bfloat16)

    out = torch.empty(B, C, H, W, dtype=torch.bfloat16, device=x.device)

    # Grid: one block per 256 (b,h,w) positions
    # Number of blocks
    total_positions = B * H * W
    num_blocks = (total_positions + 255) // 256  # ceil division
    grid = lambda meta: (num_blocks,)

    # Launch kernel
    min_reduction_kernel[grid](
        x_bf16, out, B, C, D, H, W, BLOCK_SIZE=256
    )

    return out.to(torch.float32)


def triton_softmax(x: torch.Tensor):
    """Apply softmax along dim=1 (C) using Triton kernel."""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    B, C, H, W = x.shape
    # Convert to BF16
    x_bf16 = x.to(torch.bfloat16)

    out = torch.empty(B, C, H, W, dtype=torch.bfloat16, device=x.device)

    # Grid: one block per 256 (b,h,w) positions
    total_positions = B * H * W
    num_blocks = (total_positions + 255) // 256  # ceil division
    grid = lambda meta: (num_blocks,)

    # Launch kernel
    softmax_kernel[grid](
        x_bf16, out, B, C, H, W, BLOCK_SIZE=256
    )

    return out.to(torch.float32)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    def forward(self, x):
        x = self.conv(x)
        x = triton_min_reduction(x)
        x = triton_softmax(x)
        return x