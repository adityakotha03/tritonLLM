import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ------------------ Triton kernels --------------------------------

@triton.jit
def relu_kernel(
    inp_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0.0, x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["in_height", "in_width", "kernel_size", "stride"],
)
@triton.jit
def maxpool2d_kernel(
    inp_ptr,
    out_ptr,
    in_height,
    in_width,
    in_channels,
    out_height,
    out_width,
    kernel_size,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a contiguous block of output elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Compute 3D output indices
    n_elements = out_height * out_width * in_channels
    mask = offsets < n_elements

    out_h = offsets // (out_width * in_channels)
    rem = offsets % (out_width * in_channels)
    out_w = rem // in_channels
    c = rem % in_channels

    h_start = out_h * stride
    w_start = out_w * stride

    # Compute max over kernel window
    max_val = tl.full([BLOCK_SIZE], float("-inf"))
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            h_idx = h_start + kh
            w_idx = w_start + kw
            inp_idx = ((h_idx * in_width + w_idx) * in_channels + c)
            val = tl.load(inp_ptr + inp_idx, mask=mask, other=float("-inf"))
            max_val = tl.maximum(max_val, val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


# ------------------ Helper wrappers --------------------------------

def triton_relu(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=128)
    return out


def triton_maxpool2d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
) -> torch.Tensor:
    # Assume NCHW format
    N, C, H, W = x.shape
    # Compute output dims
    out_h = (H + 2 * padding - kernel_size) // stride + 1
    out_w = (W + 2 * padding - kernel_size) // stride + 1

    # Pad input if needed
    if padding > 0:
        x = F.pad(x, (padding, padding, padding, padding))

    x = x.contiguous()
    out = torch.empty((N, C, out_h, out_w), device=x.device, dtype=x.dtype)

    # Launch kernel for each batch channel
    for n in range(N):
        inp_ptr = x[n].data_ptr()
        out_ptr = out[n].data_ptr()
        grid = lambda meta: ((out_h * out_w * C + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        maxpool2d_kernel[grid](
            inp_ptr,
            out_ptr,
            H,
            W,
            C,
            out_h,
            out_w,
            kernel_size,
            stride,
            BLOCK_SIZE=128,
        )

    return out


def triton_flatten(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


# ------------------ Model with Triton ops ----------------------------

class ModelNew(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(3, stride=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(3, stride=2)

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1 + ReLU + MaxPool
        x = self.conv1(x)
        x = triton_relu(x)
        x = self.maxpool1(x)

        # Conv2 + ReLU + MaxPool
        x = self.conv2(x)
        x = triton_relu(x)
        x = self.maxpool2(x)

        # Conv3 + ReLU
        x = self.conv3(x)
        x = triton_relu(x)

        # Conv4 + ReLU
        x = self.conv4(x)
        x = triton_relu(x)

        # Conv5 + ReLU + MaxPool
        x = self.conv5(x)
        x = triton_relu(x)
        x = self.maxpool3(x)

        # Flatten
        x = triton_flatten(x)

        # Fully connected layers
        x = self.fc1(x)
        x = triton_relu(x)
        x = self.fc2(x)
        x = triton_relu(x)
        x = self.fc3(x)

        return x