import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def add_relu_kernel(
    x_ptr,  # Pointer to first input (residual output)
    y_ptr,  # Pointer to second input (identity)
    out_ptr,  # Pointer to output (final ReLU result)
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that performs elementwise addition (residual + identity)
    followed by a ReLU in a single pass. Loads both tensors, adds them,
    applies ReLU, and writes the result back.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    out = tl.maximum(out, 0.0)  # ReLU
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_add_relu(x: torch.Tensor, y: torch.Tensor):
    """
    Wrapper for the Triton kernel that performs addition and ReLU
    on two contiguous tensors of the same shape.
    """
    assert x.is_cuda and y.is_cuda, "Both tensors must be on CUDA."
    assert x.shape == y.shape, "Inputs must have the same shape."
    x = x.contiguous()
    y = y.contiguous()

    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 256  # Chosen for good occupancy and coalescing

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    add_relu_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Triton replaces the final ReLU (after residual addition)
        out = triton_add_relu(out, identity)
        return out