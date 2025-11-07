import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr, 
    weight_ptr, 
    output_ptr,
    batch_size, 
    in_channels, 
    out_channels,
    height, 
    width,
    kernel_size,
    input_stride_0,  # batch
    input_stride_1,  # in_channels
    input_stride_2,  # height
    input_stride_3,  # width
    weight_stride_0,  # out_channels
    weight_stride_1,  # in_channels
    weight_stride_2,  # kernel_h
    weight_stride_3,  # kernel_w
    output_stride_0,  # batch
    output_stride_1,  # out_channels
    output_stride_2,  # out_h
    output_stride_3,  # out_w
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Thread block indices
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Compute output height and width
    out_height = height - kernel_size + 1
    out_width = width - kernel_size + 1

    # Compute output position
    out_h_start = pid_h * BLOCK_SIZE_H
    out_w_start = pid_w * BLOCK_SIZE_W

    # Block offsets for input and output
    out_h_offsets = out_h_start + tl.arange(0, BLOCK_SIZE_H)
    out_w_offsets = out_w_start + tl.arange(0, BLOCK_SIZE_W)

    # Mask for valid output positions
    mask_h = out_h_offsets < out_height
    mask_w = out_w_offsets < out_width
    mask_hw = mask_h[:, None] & mask_w[None, :]

    # Compute output index in output tensor
    out_idx = (pid_b * output_stride_0 +
               pid_c * output_stride_1 +
               out_h_offsets[:, None] * output_stride_2 +
               out_w_offsets[None, :] * output_stride_3)

    # Shared memory for input tiles and weights
    input_tile = tl.load(
        input_ptr + (
            pid_b * input_stride_0 +
            tl.arange(0, in_channels)[:, None, None] * input_stride_1 +
            (out_h_offsets[:, None, None] + tl.arange(0, kernel_size)[:, None, None]) * input_stride_2 +
            (out_w_offsets[None, :, None] + tl.arange(0, kernel_size)[None, :, None]) * input_stride_3
        ),
        mask=(tl.arange(0, in_channels)[:, None, None] < in_channels) &
              (tl.arange(0, kernel_size)[:, None, None] < kernel_size) &
              (tl.arange(0, kernel_size)[None, :, None] < kernel_size) &
              mask_hw,
        other=0.0
    )

    # Load weight tile
    weight_tile = tl.load(
        weight_ptr + (
            pid_c * weight_stride_0 +
            tl.arange(0, in_channels)[:, None, None] * weight_stride_1 +
            tl.arange(0, kernel_size)[:, None, None] * weight_stride_2 +
            tl.arange(0, kernel_size)[None, :, None] * weight_stride_3
        ),
        mask=(tl.arange(0, in_channels)[:, None, None] < in_channels) &
              (tl.arange(0, kernel_size)[:, None, None] < kernel_size) &
              (tl.arange(0, kernel_size)[None, :, None] < kernel_size),
        other=0.0
    )

    # Compute convolution: batched dot product across channels and kernel
    # (in_channels, kernel_h, kernel_w) @ (in_channels, kernel_h, kernel_w)
    # Output: (out_h, out_w)
    conv = tl.dot(input_tile, weight_tile, allow_tf32=True)

    # Store result with masking
    tl.store(output_ptr + out_idx, conv, mask=mask_hw)


@triton.jit
def mish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Mish: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x))
    # We compute it in a numerically stable way
    exp_x = tl.exp(-tl.abs(x))
    softplus_x = tl.log1p(tl.exp(-tl.abs(x))) + tl.where(x > 0, x, 0)
    
    # tanh(softplus(x))
    tanh_val = tl.tanh(softplus_x)
    
    # x * tanh(softplus(x))
    out = x * tanh_val
    
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def subtract_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    subtract_val,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x - subtract_val
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def fused_subtract_mish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    subtract_val_1,
    subtract_val_2,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # First subtract
    x = x - subtract_val_1
    # Second subtract
    x = x - subtract_val_2

    # Mish: x * tanh(softplus(x))
    exp_x_neg = tl.exp(-tl.abs(x))
    softplus_x = tl.log1p(exp_x_neg) + tl.where(x > 0, x, 0)
    tanh_val = tl.tanh(softplus_x)
    out = x * tanh_val

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, kernel_size: int):
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    x = x.contiguous()
    weight = weight.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, _, _ = weight.shape
    out_height = height - kernel_size + 1
    out_width = width - kernel_size + 1

    out = torch.empty(batch_size, out_channels, out_height, out_width, dtype=x.dtype, device=x.device)

    # Strides
    input_stride = (x.stride(0), x.stride(1), x.stride(2), x.stride(3))
    weight_stride = (weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3))
    output_stride = (out.stride(0), out.stride(1), out.stride(2), out.stride(3))

    # Tune block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 8

    # Grid dimensions: (batch, out_channels, out_h//block_h, out_w//block_w)
    grid = lambda meta: (
        batch_size,
        out_channels,
        (out_height + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (out_width + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
    )

    # Launch kernel
    conv2d_kernel[grid](
        x, weight, out,
        batch_size, in_channels, out_channels, height, width, kernel_size,
        *input_stride, *weight_stride, *output_stride,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W, BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    return out


def triton_subtract(x: torch.Tensor, subtract_val: float):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]

    subtract_kernel[grid](x, out, n_elements, subtract_val, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_fused_subtract_mish(x: torch.Tensor, subtract_val_1: float, subtract_val_2: float):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]

    fused_subtract_mish_kernel[grid](
        x, out, n_elements, subtract_val_1, subtract_val_2, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        # Register weight as a parameter to ensure it's on GPU
        self.register_buffer('weight', self.conv.weight.data)

    def forward(self, x):
        # Replace Conv2d + two subtractions + Mish with fused Triton kernel
        x = triton_conv2d(x, self.weight, kernel_size=self.conv.kernel_size[0])
        x = triton_fused_subtract_mish(x, self.subtract_value_1, self.subtract_value_2)
        return x