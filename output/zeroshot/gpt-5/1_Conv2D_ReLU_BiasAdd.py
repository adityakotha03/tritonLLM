import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_add_bias_kernel(
    x_ptr,            # *pointer* to input tensor
    bias_ptr,         # *pointer* to bias tensor (shape [C])
    out_ptr,          # *pointer* to output tensor
    HW: tl.constexpr, # H * W per channel
    C: tl.constexpr,  # number of channels
    NC,               # N * C
    stride_nc,        # stride between consecutive (n, c) planes in elements (typically HW)
    stride_hw,        # stride between consecutive spatial elements in elements (typically 1)
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(0)  # which (n, c) we are processing
    pid_hw = tl.program_id(1)  # which block along spatial dimension

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs_hw < HW

    # Base pointers for this (n,c) plane
    x_offs = pid_nc * stride_nc + offs_hw * stride_hw

    # Load input
    x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)

    # Determine channel index (constant for this program instance)
    c_idx = pid_nc % C
    b = tl.load(bias_ptr + c_idx)

    # relu then add bias: y = relu(x) + b
    y = tl.maximum(x, 0) + b

    # Store result
    tl.store(out_ptr + x_offs, y, mask=mask)


def triton_relu_add_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused ReLU + bias add for NCHW tensors with bias broadcast over (C, 1, 1).
    Falls back to PyTorch ops on CPU.
    """
    if not x.is_cuda:
        return torch.relu(x) + bias

    assert x.ndim == 4, "Input must be NCHW"
    N, C, H, W = x.shape
    x = x.contiguous()
    bias_vec = bias.contiguous().view(C)
    out = torch.empty_like(x)

    HW = H * W
    NC = N * C
    stride_nc = HW
    stride_hw = 1

    # Choose a reasonable block size
    BLOCK_HW = 256
    grid = (NC, triton.cdiv(HW, BLOCK_HW))

    relu_add_bias_kernel[grid](
        x, bias_vec, out,
        HW, C, NC,
        stride_nc, stride_hw,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
        num_stages=2
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that keeps cuDNN convolution and uses a fused Triton kernel
    for ReLU + bias add.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        # Fuse ReLU + bias add with Triton when on CUDA; fall back on CPU.
        return triton_relu_add_bias(x, self.bias)

batch_size = 128
in_channels  = 64
out_channels = 128
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]