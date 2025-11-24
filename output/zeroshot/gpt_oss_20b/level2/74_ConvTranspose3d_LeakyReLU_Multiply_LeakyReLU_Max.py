import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
# Triton kernel:  relu(x) -> x * m -> relu
# --------------------------------------------------------------------------- #
@triton.jit
def relu_mul_relu_kernel(
    x_ptr,      # input tensor
    m_ptr,      # multiplier tensor (broadcasted over spatial dim)
    out_ptr,    # output tensor
    n_elements, # number of elements in the input/output tensors
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # load input and multiplier (broadcast over channel dim)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # multiplier shape is (C,1,1,1) -> broadcast over spatial dimensions
    m = tl.load(m_ptr + offsets, mask=mask, other=1.0)

    # first relu
    x = tl.where(x > 0, x, x * 0.2)

    # multiply
    y = x * m

    # second relu
    y = tl.where(y > 0, y, y * 0.2)

    tl.store(out_ptr + offsets, y, mask=mask)

def triton_relu_mul_relu(x: torch.Tensor, m: torch.Tensor):
    """
    x: shape (N, C, D, H, W)
    m: shape (C, 1, 1, 1) broadcasted over N, D, H, W
    """
    assert x.is_cuda and m.is_cuda
    x = x.contiguous()
    m = m.contiguous()

    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 256

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    relu_mul_relu_kernel[grid](x, m, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

# --------------------------------------------------------------------------- #
# Optimized Model
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies LeakyReLU, multiplies by a learnable parameter,
    applies LeakyReLU again, and performs a max pooling operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        # conv transpose
        x = self.conv_transpose(x)

        # fused relu -> mul -> relu
        x = triton_relu_mul_relu(x, self.multiplier)

        # max pooling
        x = self.max_pool(x)
        return x