import torch
import torch.nn as nn
import triton
import triton.language as tl

# ---------- Triton kernel that fuses ReLU, LeakyReLU, GELU, Sigmoid, and bias addition ----------
@triton.jit
def fused_activation_bias_kernel(
    input_ptr,           # Conv output
    bias_ptr,            # Bias to add (broadcasted over N,D,H,W)
    out_ptr,             # Output tensor
    n_elements,          # Total number of elements in the output tensor
    bias_n, bias_d, bias_h, bias_w,  # dimensions of bias (C,1,1,1)
    BLOCK_SIZE: tl.constexpr,
    NEG_SLOPE: tl.constexpr,  # negative slope for LeakyReLU
):
    """
    Each program processes BLOCK_SIZE contiguous elements.
    The bias is broadcasted along N, D, H, W dimensions.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the input element (float32)
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # ---------- Activation chain ----------
    # ReLU
    x = tl.where(x > 0.0, x, 0.0)
    # LeakyReLU
    x = tl.where(x > 0.0, x, x * NEG_SLOPE)
    # GELU (approximation)
    x = x * 0.5 * (1.0 + tl.math.erf(x * tl.math.sqrt(0.5)))
    # Sigmoid
    x = 1.0 / (1.0 + tl.math.exp(-x))

    # ---------- Add bias ----------
    # Compute bias index: bias is (C,1,1,1) -> broadcast over N,D,H,W
    # For each global element, we need its channel index.
    # Global element layout is [N, C, D, H, W] in PyTorch (NCHWD)
    # We can compute channel offset by:
    # channel = (offset // (D*H*W)) % C
    # where D,H,W are spatial dims of the output tensor.
    # We'll pass bias_c, bias_d, bias_h, bias_w as dims of output.
    stride_c = bias_d * bias_h * bias_w
    stride_n = bias_c * stride_c
    global_idx = offsets  # already global index in the flattened tensor
    channel_idx = (global_idx // (bias_d * bias_h * bias_w)) % bias_c
    bias_offset = channel_idx  # since bias shape is (C,1,1,1)
    bias_val = tl.load(bias_ptr + bias_offset, mask=mask, other=0.0)

    x = x + bias_val

    # Store the result
    tl.store(out_ptr + offsets, x, mask=mask)

# ---------- Wrapper for the fused kernel ----------
def fused_activation_bias(x: torch.Tensor, bias: torch.Tensor, negative_slope: float = 0.01):
    """
    x: Conv output tensor of shape [N, C, D, H, W] (float32)
    bias: Bias tensor of shape [C, 1, 1, 1] (float32)
    Returns: Tensor after applying ReLU -> LeakyReLU -> GELU -> Sigmoid -> bias addition
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()

    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 256  # Tunable; 256 is a good default for A100

    # Compute output spatial dims
    _, C, D, H, W = x.shape
    bias_c, bias_d, bias_h, bias_w = bias.shape

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_activation_bias_kernel[grid](
        x,
        bias,
        out,
        n_elements,
        bias_c, bias_d, bias_h, bias_w,
        BLOCK_SIZE=BLOCK_SIZE,
        NEG_SLOPE=negative_slope,
    )
    return out

# ---------- New model ----------
class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution followed by a fused sequence of
    ReLU → LeakyReLU → GELU → Sigmoid and bias addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = fused_activation_bias(x, self.bias, negative_slope=0.01)
        return x