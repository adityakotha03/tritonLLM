import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mul_scalar_kernel(
    x_ptr, out_ptr, multiplier, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    output = x * multiplier
    tl.store(out_ptr + offsets, output, mask=mask)


@triton.jit
def global_avg_pool_2d_kernel(
    x_ptr, out_ptr, batch, channels, height, width, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_channels = channels
    batch_channels = batch * channels
    total_batches = batch_channels

    # Each program (block) processes one (batch, channel) slice
    if pid >= total_batches:
        return

    batch_idx = pid // num_channels
    channel_idx = pid % num_channels

    # Compute offsets for this (batch, channel)
    offset = (batch_idx * num_channels + channel_idx) * (height * width)
    x_block_ptr = x_ptr + offset + tl.arange(0, BLOCK_SIZE)
    
    # Load all spatial elements for this (batch, channel)
    mask = tl.arange(0, BLOCK_SIZE) < (height * width)
    values = tl.load(x_block_ptr, mask=mask, other=0.0)

    # Compute sum and mean
    sum_val = tl.sum(values, axis=0)
    mean_val = sum_val / (height * width)

    # Output is (batch, channels, 1, 1)
    out_offset = pid
    tl.store(out_ptr + out_offset, mean_val)


def triton_mul_scalar(x: torch.Tensor, multiplier: float):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    mul_scalar_kernel[grid](x, out, multiplier, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_global_avg_pool_2d(x: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    batch, channels, height, width = x.shape
    out = torch.empty((batch, channels, 1, 1), dtype=x.dtype, device=x.device)
    n_elements = batch * channels
    BLOCK_SIZE = height * width  # Load entire spatial dim per block if fits
    if BLOCK_SIZE > 4096:
        BLOCK_SIZE = 4096
    grid = lambda meta: (batch * channels,)
    global_avg_pool_2d_kernel[grid](
        x, out, batch, channels, height, width, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for scalar multiplication and global average pooling.
    The transposed convolution remains as-is since it's already highly optimized in PyTorch/CuDNN.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = multiplier

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_mul_scalar(x, self.multiplier)
        x = triton_global_avg_pool_2d(x)
        x = triton_global_avg_pool_2d(x)
        return x