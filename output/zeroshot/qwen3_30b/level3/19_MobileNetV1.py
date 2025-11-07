import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 16, 'BLOCK_SIZE_C': 1}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 16, 'BLOCK_SIZE_C': 4}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 8, 'BLOCK_SIZE_C': 1}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 8, 'BLOCK_SIZE_W': 32, 'BLOCK_SIZE_C': 1}, num_stages=3, num_warps=4),
    ],
    key=['H', 'W', 'C', 'K', 'stride'],
)
@triton.jit
def depthwise_conv_bn_relu_kernel(
    x_ptr,  # Input tensor (B, C, H, W)
    w_ptr,  # Weights tensor (C, 1, K, K) -> stored as (C, K, K) since groups=inp
    bn_weight_ptr,  # BatchNorm weight (C,)
    bn_bias_ptr,  # BatchNorm bias (C,)
    out_ptr,  # Output tensor (B, C, H_out, W_out)
    B, C, H, W, H_out, W_out, K, stride,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Each program handles a block of output spatial elements
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)

    # Calculate the starting position in output
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W
    c_start = pid_c * BLOCK_SIZE_C

    # Calculate the output indices
    h_offset = h_start + tl.arange(0, BLOCK_SIZE_H)[:, None]
    w_offset = w_start + tl.arange(0, BLOCK_SIZE_W)[None, :]
    c_offset = c_start + tl.arange(0, BLOCK_SIZE_C)

    # Clamp to valid indices for output
    h_out = tl.where(h_offset < H_out, h_offset, 0)
    w_out = tl.where(w_offset < W_out, w_offset, 0)
    c_out = tl.where(c_offset < C, c_offset, 0)

    # Calculate input indices
    # For a depthwise conv, each input channel has its own kernel
    # So we only use the input channel c_out for output channel c_out
    # The input indices: h_in = h_out * stride + dh, w_in = w_out * stride + dw
    # where dh, dw in [0, K-1]
    dh = tl.arange(0, K)[:, None, None]
    dw = tl.arange(0, K)[None, :, None]
    h_in = h_out[:, :, None] * stride + dh
    w_in = w_out[:, :, None] * stride + dw

    # Create masks for output and input
    mask_out = (h_out < H_out)[:, :, None] & (w_out < W_out)[None, :, None] & (c_out < C)[None, None, :]
    mask_in = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)

    # Load input values
    # We need to load from x_ptr with indices (B, c_out, h_in, w_in)
    # We'll loop over the batch dimension
    # We can use a loop over batch or use the fact that we are using 2D grid and batch is the first dimension
    # But in Triton, we can use the program_id for batch? No, we have only 3D grid: (H_out, W_out, C)
    # So we need to handle batch dimension in the loop.
    # We'll use a different approach: we'll have a separate program_id for batch.

    # We need to modify the grid to include batch.
    # So we change the grid to (num_h_blocks, num_w_blocks, C, B) -> but that's too many blocks.

    # Alternative: we'll assume batch=1 for now and later vectorize over batch.
    # But we can't.

    # We need to redesign.

    # Let's change the kernel to have a 4D grid: (H_out, W_out, C, B) but that's too large.

    # Instead, we'll have the kernel process one batch at a time, but then we can't use the full parallelism.

    # We'll do: the kernel is launched with grid=(num_h_blocks, num_w_blocks, C) and the batch dimension is handled by the caller.
    # But then we have to loop over batch in Python.

    # That's not efficient.

    # We'll redesign the kernel to have a 4D grid: (B, H_out, W_out, C) but then we need to index.

    # We'll change the program_id:
    # pid_b = tl.program_id(0)
    # pid_h = tl.program_id(1)
    # pid_w = tl.program_id(2)
    # pid_c = tl.program_id(3)

    # But we have to change the kernel.

    # Let's rewrite the kernel with 4D grid.

    # We'll restructure.

    # But this is getting too complex.

Given the time and complexity, and the fact that we must provide a working solution, we will instead choose a different approach.

We'll use the following: we will write a kernel that handles one output channel and one spatial block at a time, and then we'll call it for each batch independently.

But that would be slow.

Alternatively, we'll use a different strategy: we'll write a kernel that processes a tile of the spatial dimensions and a tile of the channels, but only one batch at a time.

But the grid will be (H_out, W_out, C) and the batch dimension is handled by the caller's loop.

This is not efficient.

Given the complexity of writing a full depthwise convolution kernel with shared memory and 4D indexing in Triton, and the time constraints, we will instead provide a working solution that uses a simpler kernel for the depthwise conv that processes one output element at a time, but with a block size of 128.

We'll use a 1D grid with BLOCK_SIZE=128 for the output spatial dimensions.

But we need to handle the batch and channel dimensions.

We'll flatten the output to (B * C * H_out * W_out) and then process with a 1D grid.

Then in the kernel, we will have:

- pid = tl.program_id(0)
- offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
- then map this offset to (b, c, h, w)

Then we can compute the input indices.

Let's do that.

We'll write a 1D kernel for depthwise conv.

But we'll do it for one batch, one channel, one spatial location.

Let's write a kernel for one element.

Then we can use BLOCK_SIZE to process multiple elements.

But we want to use shared memory for the input tile.

We'll do a 1D grid with BLOCK_SIZE=128.

We'll tile the input.

Let's do it.

We'll assume that we are processing a batch of size 1, and then we'll loop over batch in the wrapper.

But we can't.

We'll include the batch in the offset.

So the total output size is B * C * H_out * W_out.

We'll flatten and then process.

So the kernel will be:
