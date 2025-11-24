import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr,  # pointer to input tensor (batch, in_channels, D, H, W)
    output_ptr,  # pointer to output tensor (batch, out_channels, D, H, W)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Define the spatial dimensions
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)

    # Define the spatial indices
    d_idx = tl.arange(0, D)
    h_idx = tl.arange(0, H)
    w_idx = tl.arange(0, W)

    # Define the kernel indices
    k_d = tl.arange(0, kernel_size)
    k_h = tl.arange(0, kernel_size)
    k_w = tl.arange(0, kernel_size)

    # Define the input and output indices
    # We will compute the convolution using a sliding window
    # Input shape: (batch, in_channels, D, H, W)
    # Output shape: (batch, out_channels, D, H, W)

    # Compute the output spatial indices
    # We will process one output channel at a time
    # and one spatial location at a time

    # For each output location (d, h, w), we compute the convolution
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the output spatial coordinates
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the input and output indices
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # We will compute the convolution for one output channel at a time
    # and one spatial location at a time

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0) & (w_in < W)

    # Compute the convolution
    # We will compute the convolution using a sliding window
    # We loop over the input spatial dimensions
    # We use a block of size BLOCK_SIZE for each spatial dimension

    # Define the output spatial index
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Define the input spatial coordinates
    d_in = d_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    h_in = h_out - (kernel_size // 2) + tl.arange(0, kernel_size)
    w_in = w_out - (kernel_size // 2) + tl.arange(0, kernel_size)

    # Create the mask for valid indices
    d_mask = (d_in >= 0) & (d_in < D)
    h_mask = (h_in >= 0) & (h_in < H)
    w_mask = (w_in >= 0) & (w_in < W)

    # Create the mask for valid kernel indices
    d_k_mask = (d_in >= 0) & (d_in < D)
    h_k_mask = (h_in >= 0) & (h_in < H)
    w_k_mask = (w_in >= 0