import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def mul_clamp_mul_max_kernel(
    x_ptr,               # Pointer to input tensor (after conv + instance norm)
    multiplier_ptr,      # Pointer to multiplier parameter
    out_ptr,             # Pointer to output tensor (after max over channel)
    n_batches,           # Batch size
    out_channels,        # Number of channels (C)
    depth,               # D
    height,              # H
    width,               # W
    numel_per_batch,     # Number of elements per batch (C * D * H * W)
    clamp_min,           # Clamping minimum
    clamp_max,           # Clamping maximum
    BLOCK_SIZE_DHW: tl.constexpr,
):
    # Compute program ids
    pid_b = tl.program_id(0)  # batch index
    pid_c = tl.program_id(1)  # channel index
    pid_dhw = tl.program_id(2)  # spatial block index

    # Compute offsets
    batch_offset = pid_b * numel_per_batch
    channel_offset = pid_c * (depth * height * width)
    
    # Compute start and mask for DHW block
    dhw_offset = pid_dhw * BLOCK_SIZE_DHW
    offsets_dhw = dhw_offset + tl.arange(0, BLOCK_SIZE_DHW)
    mask_dhw = offsets_dhw < (depth * height * width)
    
    # Load multiplier for this channel
    multiplier = tl.load(multiplier_ptr + pid_c)
    
    # First multiply: x * multiplier
    x_ptrs = x_ptr + batch_offset + channel_offset + offsets_dhw
    x = tl.load(x_ptrs, mask=mask_dhw, other=0.0)
    x = x * multiplier
    
    # Clamp
    x = tl.maximum(x, clamp_min)
    x = tl.minimum(x, clamp_max)
    
    # Second multiply: x * multiplier again
    x = x * multiplier
    
    # Store back
    tl.store(x_ptrs, x, mask=mask_dhw)

    # Now handle max over channel (only in first channel block)
    if pid_c == 0:
        # Load all channels for this spatial location to compute max
        max_vals = tl.full([BLOCK_SIZE_DHW], -float('inf'), dtype=tl.float32)
        for c in range(out_channels):
            c_offset = c * (depth * height * width)
            x_c_ptrs = x_ptr + batch_offset + c_offset + offsets_dhw
            x_c = tl.load(x_c_ptrs, mask=mask_dhw, other=-float('inf'))
            max_vals = tl.maximum(max_vals, x_c)
        
        # Store max result into output
        out_ptrs = out_ptr + pid_b * (depth * height * width) + offsets_dhw
        tl.store(out_ptrs, max_vals, mask=mask_dhw)


def triton_mul_clamp_mul_max(x: torch.Tensor, multiplier: torch.nn.Parameter):
    """
    Custom Triton kernel that fuses:
    x = x * multiplier
    x = clamp(x, -1.0, 1.0)
    x = x * multiplier
    x = torch.max(x, dim=1)[0]
    """
    assert x.is_cuda and multiplier.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    multiplier = multiplier.contiguous()

    batch_size, out_channels, depth, height, width = x.shape
    numel_per_batch = out_channels * depth * height * width
    out_shape = (batch_size, depth, height, width)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    # 1D block size for spatial dimensions
    BLOCK_SIZE_DHW = 512

    # Grid: (batch, channels, DHW_blocks)
    grid = lambda meta: (
        batch_size,
        out_channels,
        triton.cdiv(depth * height * width, meta['BLOCK_SIZE_DHW'])
    )

    mul_clamp_mul_max_kernel[grid](
        x_ptr=x,
        multiplier_ptr=multiplier,
        out_ptr=out,
        n_batches=batch_size,
        out_channels=out_channels,
        depth=depth,
        height=height,
        width=width,
        numel_per_batch=numel_per_batch,
        clamp_min=-1.0,
        clamp_max=1.0,
        BLOCK_SIZE_DHW=BLOCK_SIZE_DHW,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel fusion for elementwise operations and max reduction.
    The 3D convolution and instance norm are kept as PyTorch ops (heavily optimized already),
    but the sequence: mul -> norm -> clamp -> mul -> max is fused into one kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        x = triton_mul_clamp_mul_max(x, self.multiplier)
        return x