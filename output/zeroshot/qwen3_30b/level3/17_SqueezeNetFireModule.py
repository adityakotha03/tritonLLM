import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel_1x1(
    x_ptr,
    w_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    # Compute global offsets
    h_offset = pid_h * BLOCK_SIZE_H
    w_offset = pid_w * BLOCK_SIZE_W
    c_offset = pid_c * BLOCK_SIZE_C

    # Compute indices for input, weight, and output
    h_indices = h_offset + tl.arange(0, BLOCK_SIZE_H)
    w_indices = w_offset + tl.arange(0, BLOCK_SIZE_W)
    c_indices = c_offset + tl.arange(0, BLOCK_SIZE_C)

    # Create masks for bounds
    h_mask = h_indices < height
    w_mask = w_indices < width
    c_mask = c_indices < out_channels

    # Compute global input and output offsets
    batch_offset = pid_batch * in_channels * height * width
    out_batch_offset = pid_batch * out_channels * height * width

    # Load input tensor (N, C_in, H, W)
    x_ptrs = x_ptr + batch_offset + c_indices[:, None, None] * height * width + h_indices[None, :, None] * width + w_indices[None, None, :]
    x = tl.load(x_ptrs, mask=(c_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]), other=0.0)

    # Load weights (C_out, C_in, 1, 1)
    w_ptrs = w_ptr + c_indices[:, None, None, None] * in_channels + c_indices[None, :, None, None]
    w = tl.load(w_ptrs, mask=(c_mask[:, None, None, None] & c_mask[None, :, None, None]), other=0.0)

    # Compute output: sum over in_channels
    out = tl.dot(x, w, allow_tf32=True)

    # Write output (N, C_out, H, W)
    out_ptrs = out_ptr + out_batch_offset + c_indices[:, None, None] * height * width + h_indices[None, :, None] * width + w_indices[None, None, :]
    tl.store(out_ptrs, out, mask=(c_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]))


@triton.jit
def conv2d_kernel_3x3(
    x_ptr,
    w_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    # Compute global offsets
    h_offset = pid_h * BLOCK_SIZE_H
    w_offset = pid_w * BLOCK_SIZE_W
    c_offset = pid_c * BLOCK_SIZE_C

    # Compute indices for input, weight, and output
    h_indices = h_offset + tl.arange(0, BLOCK_SIZE_H)
    w_indices = w_offset + tl.arange(0, BLOCK_SIZE_W)
    c_indices = c_offset + tl.arange(0, BLOCK_SIZE_C)

    # Create masks for bounds
    h_mask = h_indices < height
    w_mask = w_indices < width
    c_mask = c_indices < out_channels

    # Compute global input and output offsets
    batch_offset = pid_batch * in_channels * height * width
    out_batch_offset = pid_batch * out_channels * height * width

    # Define kernel size
    kernel_size = 3
    kh_indices = tl.arange(0, kernel_size)
    kw_indices = tl.arange(0, kernel_size)

    # Load input tensor (N, C_in, H, W)
    x_ptrs = x_ptr + batch_offset + c_indices[:, None, None, None] * height * width + h_indices[None, :, None, None] * width + w_indices[None, None, :, None]
    x = tl.load(x_ptrs, mask=(c_mask[:, None, None, None] & h_mask[None, :, None, None] & w_mask[None, None, :, None]), other=0.0)

    # Load weights (C_out, C_in, 3, 3)
    w_ptrs = w_ptr + c_indices[:, None, None, None] * in_channels * kernel_size * kernel_size + c_indices[None, :, None, None] * kernel_size * kernel_size + kh_indices[None, None, :, None] * kernel_size + kw_indices[None, None, None, :]
    w = tl.load(w_ptrs, mask=(c_mask[:, None, None, None] & c_mask[None, :, None, None]), other=0.0)

    # Compute output: sum over in_channels and kernel size
    out = tl.dot(x, w, allow_tf32=True)

    # Write output (N, C_out, H, W)
    out_ptrs = out_ptr + out_batch_offset + c_indices[:, None, None] * height * width + h_indices[None, :, None] * width + w_indices[None, None, :] * 1
    tl.store(out_ptrs, out, mask=(c_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]))


@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv2d_1x1(x: torch.Tensor, w: torch.Tensor, out_channels: int, height: int, width: int):
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    # Prepare output tensor
    out = torch.empty(x.size(0), out_channels, height, width, device=x.device, dtype=x.dtype)

    # Grid setup
    grid = lambda meta: (
        x.size(0),  # batch
        (height + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (width + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
        (out_channels + meta["BLOCK_SIZE_C"] - 1) // meta["BLOCK_SIZE_C"],
    )

    # Launch kernel
    conv2d_kernel_1x1[grid](
        x,
        w,
        out,
        x.size(0),
        x.size(1),
        out_channels,
        height,
        width,
        1, 1, 0, 0,
        BLOCK_SIZE_H=64,
        BLOCK_SIZE_W=64,
        BLOCK_SIZE_C=32,
    )
    return out


def triton_conv2d_3x3(x: torch.Tensor, w: torch.Tensor, out_channels: int, height: int, width: int):
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    # Prepare output tensor
    out = torch.empty(x.size(0), out_channels, height, width, device=x.device, dtype=x.dtype)

    # Grid setup
    grid = lambda meta: (
        x.size(0),  # batch
        (height + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (width + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
        (out_channels + meta["BLOCK_SIZE_C"] - 1) // meta["BLOCK_SIZE_C"],
    )

    # Launch kernel
    conv2d_kernel_3x3[grid](
        x,
        w,
        out,
        x.size(0),
        x.size(1),
        out_channels,
        height,
        width,
        1, 1, 1, 1,
        BLOCK_SIZE_H=64,
        BLOCK_SIZE_W=64,
        BLOCK_SIZE_C=32,
    )
    return out


def triton_relu(x: torch.Tensor):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super().__init__()
        self.squeeze_conv = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, bias=False)
        self.expand1x1_conv = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1, bias=False)
        self.expand3x3_conv = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1, bias=False)

        # Initialize weights
        nn.init.kaiming_uniform_(self.squeeze_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.expand1x1_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.expand3x3_conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # Squeeze layer + ReLU
        x = triton_conv2d_1x1(x, self.squeeze_conv.weight, self.squeeze_conv.out_channels, x.size(2), x.size(3))
        x = triton_relu(x)

        # Expand1x1 + ReLU
        expand1x1_out = triton_conv2d_1x1(x, self.expand1x1_conv.weight, self.expand1x1_conv.out_channels, x.size(2), x.size(3))
        expand1x1_out = triton_relu(expand1x1_out)

        # Expand3x3 + ReLU
        expand3x3_out = triton_conv2d_3x3(x, self.expand3x3_conv.weight, self.expand3x3_conv.out_channels, x.size(2), x.size(3))
        expand3x3_out = triton_relu(expand3x3_out)

        # Concatenate along channel dimension
        return torch.cat([expand1x1_out, expand3x3_out], dim=1)