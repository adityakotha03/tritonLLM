import torch
import torch.nn as nn
import triton
import triton.language as tl

# ---------- Triton kernel for bias + scaling + tanh fusion ----------
@triton.jit
def bias_scale_tanh_kernel(
    in_ptr,          # pointer to input tensor (after conv)
    bias_ptr,        # pointer to bias tensor (broadcastable)
    out_ptr,         # pointer to output tensor
    stride_h, stride_w,  # stride of conv output
    channels,        # number of channels
    H, W,            # spatial dimensions of conv output
    scaling_factor,  # scalar to multiply after tanh
    BLOCK_W: tl.constexpr,  # block width
    BLOCK_H: tl.constexpr,  # block height
    CHANNELS_PER_BLOCK: tl.constexpr,
):
    # Compute 2D program id
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Compute start indices
    w_start = pid_w * BLOCK_W
    h_start = pid_h * BLOCK_H

    # Load bias (single element per channel, broadcast)
    bias = tl.load(bias_ptr, mask=tl.arange(0, channels) < channels, other=0.0)

    # Iterate over blocks of channels
    for c in range(0, channels, CHANNELS_PER_BLOCK):
        c_idx = c + tl.arange(0, CHANNELS_PER_BLOCK)
        # Broadcast mask
        mask_c = c_idx < channels

        # Compute offsets for current block
        offsets = (
            (h_start * stride_h + tl.arange(0, BLOCK_H))[:, None] * stride_w * H
            + (w_start * stride_w + tl.arange(0, BLOCK_W))[None, :]
            + c_idx[None, None] * H * W
        )
        # Flatten offsets
        offsets = tl.reshape(offsets, -1)

        # Mask for boundaries
        mask_hw = (h_start < H) & (w_start < W)
        mask_hw = mask_hw & (tl.arange(0, BLOCK_H * BLOCK_W) < (H - h_start) * (W - w_start))

        # Load input block
        inp = tl.load(in_ptr + offsets, mask=mask_hw, other=0.0)
        # Apply tanh and scaling
        inp = tl.math.tanh(inp) * scaling_factor
        # Add bias
        inp += bias[mask_c][:, None, None]

        # Store
        tl.store(out_ptr + offsets, inp, mask=mask_hw)

# ---------- Helper to launch the kernel ----------
def bias_scale_tanh(in_tensor: torch.Tensor, bias: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    """
    in_tensor: (N, C, H, W) after conv
    bias: (C, 1, 1)
    scaling_factor: scalar
    """
    assert in_tensor.is_cuda and bias.is_cuda
    N, C, H, W = in_tensor.shape
    out = torch.empty_like(in_tensor)

    # Parameters for kernel
    BLOCK_W = 32
    BLOCK_H = 32
    CHANNELS_PER_BLOCK = 32

    grid = ( ( (W + BLOCK_W - 1) // BLOCK_W ),  # grid_x
             ( (H + BLOCK_H - 1) // BLOCK_H ),  # grid_y
             N )  # grid_z for batch

    # Flatten pointers
    in_ptr = in_tensor.view(-1).contiguous().ptr()
    out_ptr = out.view(-1).contiguous().ptr()
    bias_ptr = bias.view(-1).contiguous().ptr()

    # Launch
    bias_scale_tanh_kernel[grid](
        in_ptr, bias_ptr, out_ptr,
        stride_h=W, stride_w=1,
        channels=C, H=H, W=W,
        scaling_factor=scaling_factor,
        BLOCK_W=BLOCK_W, BLOCK_H=BLOCK_H,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK
    )
    return out

# ---------- Optimized model ----------
class ModelNew(nn.Module):
    """
    A model that performs a convolution, then fuses tanh, scaling, and bias addition
    into a single custom Triton kernel, and finally max-pools.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        # Convolution
        x = self.conv(x)  # shape: (N, C, H, W)
        # Fused tanh + scaling + bias addition
        x = bias_scale_tanh(x, self.bias, self.scaling_factor)
        # Max-pooling
        x = self.max_pool(x)
        return x