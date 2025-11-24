import torch
import torch.nn as nn
import triton
import triton.language as tl

# ---------- Triton kernel for bias subtraction + tanh ----------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16),
    ],
    key=["N_ELEMENTS"],
)
@triton.jit
def bias_tanh_kernel(
    out_ptr,          # pointer to convolution output
    bias_ptr,         # pointer to bias (C x 1 x 1)
    C,                # number of channels
    N_ELEMENTS,       # total number of elements in output (B*C*H*W)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    # Load output element
    out = tl.load(out_ptr + offsets, mask=mask, other=0.0)

    # Compute channel index for this element
    channel = (offsets // (N_ELEMENTS // C)) % C
    bias_offset = channel
    bias_val = tl.load(bias_ptr + bias_offset, mask=mask, other=0.0)

    # Subtract bias
    out = out - bias_val

    # Apply tanh (using triton.tanh)
    out = tl.tanh(out)

    tl.store(out_ptr + offsets, out, mask=mask)

# ---------- Helper function to launch the kernel ----------

def bias_tanh(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x: Tensor of shape (B, C, H, W), must be contiguous and on CUDA
    bias: Tensor of shape (C, 1, 1), must be contiguous and on CUDA
    Returns a new tensor with bias subtracted and tanh applied.
    """
    assert x.is_cuda and bias.is_cuda, "Both tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()

    B, C, H, W = x.shape
    N_ELEMENTS = B * C * H * W

    out = torch.empty_like(x)

    grid = lambda meta: ((N_ELEMENTS + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    bias_tanh_kernel[grid](out, bias.squeeze(), C, N_ELEMENTS, BLOCK_SIZE=grid.meta["BLOCK_SIZE"])
    return out

# ---------- Optimized model ----------

class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    Bias subtraction and tanh are fused into a single Triton kernel for better performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape,
                 stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        # Ensure bias is a learnable parameter with shape (C,1,1)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = bias_tanh(x, self.bias)
        return x