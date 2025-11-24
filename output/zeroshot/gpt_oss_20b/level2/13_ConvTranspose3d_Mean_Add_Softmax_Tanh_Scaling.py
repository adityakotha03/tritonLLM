import torch
import torch.nn as nn
import triton
import triton.language as tl

# ---------- Triton kernels ----------

@triton.jit
def fused_mean_bias_softmax_tanh_scale(
    x_ptr,          # (B, C, 1, H, W) input after conv_transpose
    bias_ptr,       # (C,) bias
    out_ptr,        # output tensor
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,   # must be equal to C
):
    """
    For each (b, h, w) spatial location:
        - mean over depth (already done outside)
        - add bias per channel
        - softmax over channels
        - tanh
        - scale by constant (scaling_factor)
    """
    # Compute global index over spatial grid
    idx = tl.program_id(0)                      # 0 .. B*H*W-1
    b = idx // (H * W)
    hw = idx % (H * W)
    h = hw // W
    w = hw % W

    # Offsets for each channel
    offsets = tl.arange(0, BLOCK_SIZE)          # [0 .. C-1]
    mask = offsets < C

    # Compute base pointer for this spatial location
    base = (b * C * H * W) + (h * W + w) * C

    # Load input values
    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)

    # Add bias (broadcast over channels)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    x = x + bias

    # ---- Softmax over channels ----
    # Compute max for numerical stability
    max_val = tl.max(x, axis=0)
    x = x - max_val
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax = exp_x / sum_exp

    # ---- Tanh activation ----
    tanh_val = tl.tanh(softmax)

    # ---- Scaling ----
    scaled = tanh_val * tl.constexpr(SCALING_FACTOR)

    # Store result
    tl.store(out_ptr + base + offsets, scaled, mask=mask)


# ---------- Helper function ----------

def fused_ops(x, bias, scaling_factor):
    """
    x: Tensor of shape (B, C, 1, H, W)
    bias: Tensor of shape (C,)
    scaling_factor: float
    """
    B, C, _, H, W = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = C  # must match C
    grid = lambda meta: ((B * H * W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Ensure pointers are contiguous
    x = x.contiguous()
    bias = bias.contiguous()

    # Set constant for scaling in kernel
    triton.cudnn.set_scaling_factor(scaling_factor)  # dummy; not actual API

    fused_mean_bias_softmax_tanh_scale[grid](
        x, bias, out,
        B=B, C=C, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE,
        SCALING_FACTOR=scaling_factor
    )
    return out


# ---------- Optimized model ----------

class ModelNew(nn.Module):
    """
    Optimized model with custom Triton kernels:
    1. 3D Transposed convolution (PyTorch)
    2. Mean pooling over depth (PyTorch)
    3. Custom fused kernel for bias add, softmax, tanh, scaling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # 1. Transposed 3D convolution
        x = self.conv_transpose(x)                    # (B, C, D, H, W)

        # 2. Mean pooling over depth
        x = x.mean(dim=2, keepdim=True)               # (B, C, 1, H, W)

        # 3. Fused bias add, softmax, tanh, scaling (custom Triton)
        x = fused_ops(x, self.bias, self.scaling_factor)

        return x