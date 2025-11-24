import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Triton kernel that fuses add, clamp, and GELU
# ------------------------------------------------------------------
@triton.jit
def fused_add_clamp_gelu_kernel(
    x_ptr,              # Input tensor after Conv3d + LeakyReLU
    sum_tensor_ptr,     # Broadcasted sum tensor of shape (C,1,1,1)
    out_ptr,            # Output tensor
    n_elements,         # Total number of elements (N*C*D*H*W)
    C: tl.constexpr,    # Number of channels
    DHW: tl.constexpr,  # D * H * W
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute channel index for each element
    # channel = (offset // (DHW)) % C
    channel_idx = ((offsets // DHW) % C).to(tl.int32)

    # Load broadcasted sum tensor
    # sum_tensor shape: (C,1,1,1) -> stored as contiguous 1D of length C
    sum_val = tl.load(sum_tensor_ptr + channel_idx, mask=mask, other=0.0)

    # Add
    y = x + sum_val

    # Clamp to [-1, 1]
    y = tl.where(y > 1.0, 1.0, tl.where(y < -1.0, -1.0, y))

    # GELU approximation: 0.5 * y * (1 + tanh( sqrt(2/pi) * (y + 0.044715 * y^3) ))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    y_cubed = y * y * y
    inner = sqrt_2_over_pi * (y + 0.044715 * y_cubed)
    gelu = 0.5 * y * (1.0 + tl.tanh(inner))

    # Store result
    tl.store(out_ptr + offsets, gelu, mask=mask)

# ------------------------------------------------------------------
# Wrapper function that calls the kernel
# ------------------------------------------------------------------
def fused_add_clamp_gelu(x: torch.Tensor, sum_tensor: torch.Tensor) -> torch.Tensor:
    """
    x: Tensor of shape (N, C, D, H, W) after Conv3d + LeakyReLU
    sum_tensor: Parameter of shape (C, 1, 1, 1) broadcasted
    """
    assert x.is_cuda and sum_tensor.is_cuda, "Tensors must be on CUDA."

    # Ensure contiguous memory layout
    x = x.contiguous()
    sum_tensor = sum_tensor.contiguous()

    N, C, D, H, W = x.shape
    n_elements = N * C * D * H * W
    DHW = D * H * W

    out = torch.empty_like(x)

    BLOCK_SIZE = 1024  # Tune this if needed

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # sum_tensor is stored as (C,1,1,1) -> flatten to 1D of length C
    sum_tensor_flat = sum_tensor.view(-1)

    fused_add_clamp_gelu_kernel[grid](
        x_ptr=x.view(-1).data_ptr(),
        sum_tensor_ptr=sum_tensor_flat.data_ptr(),
        out_ptr=out.view(-1).data_ptr(),
        n_elements=n_elements,
        C=C,
        DHW=DHW,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

# ------------------------------------------------------------------
# Optimized Model
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies LeakyReLU, adds a broadcasted tensor,
    clamps, and applies GELU activation using a fused Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = self.conv(x)                                 # Conv3d
        x = F.leaky_relu(x, negative_slope=0.2)          # LeakyReLU
        x = fused_add_clamp_gelu(x, self.sum_tensor)     # Add + clamp + GELU (fused)
        return x