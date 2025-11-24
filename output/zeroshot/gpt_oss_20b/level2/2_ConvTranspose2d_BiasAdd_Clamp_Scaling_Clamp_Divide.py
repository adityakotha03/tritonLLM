import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------
# Triton kernel: bias + clamping + scaling
# --------------------------------------
@triton.jit
def bias_scale_clamp_kernel(
    out_ptr,        # pointer to conv_transpose output (NHWC)
    bias_ptr,       # pointer to bias (C)
    n_elements,     # total number of elements in out
    n_channels,     # number of output channels
    H, W,           # spatial dimensions
    stride_H, stride_W,  # strides for channel indexing
    scaling_factor: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    out = tl.load(out_ptr + offsets, mask=mask, other=0.0)

    # Compute channel index
    # Assuming out_ptr is NHWC: stride_W for W, stride_H for H, n_channels for C
    # offset = n + h*W + w + c*n_channels*H*W
    # We recover c by modulo with n_channels after dividing by H*W
    idx = offsets
    # compute (idx // (H*W)) % n_channels
    idx_ch = (idx // (H * W)) % n_channels
    bias = tl.load(bias_ptr + idx_ch, mask=mask, other=0.0)

    # Bias add
    out = out + bias

    # Clamp [0,1]
    out = tl.math.max(out, 0.0)
    out = tl.math.min(out, 1.0)

    # Scale
    out = out * scaling_factor

    # Clamp again
    out = tl.math.max(out, 0.0)
    out = tl.math.min(out, 1.0)

    # Divide by scaling factor
    out = out / scaling_factor

    tl.store(out_ptr + offsets, out, mask=mask)

# --------------------------------------
# Wrapper function for Triton kernel
# --------------------------------------
def bias_scale_clamp(out: torch.Tensor, bias: torch.Tensor, scaling_factor: float):
    """
    Apply bias, clamping, scaling, second clamping, and division in a single Triton kernel.
    Assumes out is the result of conv_transpose and is contiguous.
    bias shape: (C,)
    """
    assert out.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    out = out.contiguous()
    bias = bias.contiguous()

    C, H, W = out.shape[1], out.shape[2], out.shape[3]
    n_elements = out.numel()

    BLOCK_SIZE = 256  # tuned block size

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    bias_scale_clamp_kernel[grid](
        out,            # out_ptr
        bias,           # bias_ptr
        n_elements,
        C,
        H, W,
        W, H,           # stride_W, stride_H (not used, kept for clarity)
        scaling_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

# --------------------------------------
# Optimized model
# --------------------------------------
class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, adds a bias term, clamps, scales,
    clamps again, and divides using a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding,
                 bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        # bias is a learnable parameter of shape (out_channels, 1, 1)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard transposed convolution
        out = self.conv_transpose(x)

        # Broadcast bias to match out shape (N, C, H, W)
        # bias shape: (C, 1, 1) -> (C,)
        bias_flat = self.bias.squeeze(-1).squeeze(-1)

        # Apply custom Triton kernel for bias + clamping + scaling
        out = bias_scale_clamp(out, bias_flat, self.scaling_factor)
        return out