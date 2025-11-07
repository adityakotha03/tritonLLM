import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton kernel for fused conv2d + mish activation
@triton.jit
def conv2d_mish_kernel(
    x_ptr,       # Input tensor pointer
    w_ptr,       # Weight tensor pointer
    out_ptr,     # Output tensor pointer
    bias_ptr,    # Bias tensor pointer
    x_stride_0,  # Stride for batch dimension
    x_stride_1,  # Stride for channel dimension
    x_stride_2,  # Stride for height dimension
    x_stride_3,  # Stride for width dimension
    w_stride_0,  # Stride for output channel
    w_stride_1,  # Stride for input channel
    w_stride_2,  # Stride for kernel height
    w_stride_3,  # Stride for kernel width
    out_stride_0, # Stride for batch
    out_stride_1, # Stride for output channel
    out_stride_2, # Stride for height
    out_stride_3, # Stride for width
    bias_stride,  # Stride for bias
    batch_size,   # Number of batches
    in_channels,  # Number of input channels
    out_channels, # Number of output channels
    height,       # Input height
    width,        # Input width
    kernel_h,     # Kernel height
    kernel_w,     # Kernel width
    out_h,        # Output height
    out_w,        # Output width
    BLOCK_H: tl.constexpr,   # Tile size for height
    BLOCK_W: tl.constexpr,   # Tile size for width
    BLOCK_C: tl.constexpr,   # Tile size for channels (output)
):
    # Get thread block index
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)

    # Calculate starting position for this block
    block_start_h = pid_h * BLOCK_H
    block_start_w = pid_w * BLOCK_W
    block_start_c = pid_c * BLOCK_C

    # Create offsets for output region
    offs_h = block_start_h + tl.arange(0, BLOCK_H)
    offs_w = block_start_w + tl.arange(0, BLOCK_W)
    offs_c = block_start_c + tl.arange(0, BLOCK_C)

    # Create mask for bounds checking
    mask_h = offs_h < out_h
    mask_w = offs_w < out_w
    mask_c = offs_c < out_channels
    mask = mask_h[:, None, None] & mask_w[None, :, None] & mask_c[None, None, :]

    # Compute output shape
    output_offset = (
        pid_h * BLOCK_H * out_stride_2 +
        pid_w * BLOCK_W * out_stride_3 +
        pid_c * BLOCK_C * out_stride_1
    )

    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_C), dtype=tl.float32)

    # Loop over input channels and kernel dimensions
    for c in range(0, in_channels, BLOCK_C):
        # Load input tile
        offs_c_in = c + tl.arange(0, BLOCK_C)
        mask_in = offs_c_in < in_channels
        mask_in = mask_in[None, None, :]
        input_offset = (
            0 * x_stride_0 +  # batch dim
            offs_c_in * x_stride_1 +  # input channel
            tl.arange(0, BLOCK_H)[:, None, None] * x_stride_2 +  # height
            tl.arange(0, BLOCK_W)[None, :, None] * x_stride_3  # width
        )
        x_tile = tl.load(
            x_ptr + input_offset,
            mask=mask_in & mask,
            other=0.0
        )

        # Load weight tile
        offs_c_out = tl.arange(0, BLOCK_C)[:, None, None]
        offs_kh = tl.arange(0, kernel_h)[:, None, None]
        offs_kw = tl.arange(0, kernel_w)[None, :, None]
        weight_offset = (
            offs_c_out * w_stride_0 +
            offs_c_in[None, :, None] * w_stride_1 +
            offs_kh * w_stride_2 +
            offs_kw * w_stride_3
        )
        w_tile = tl.load(
            w_ptr + weight_offset,
            mask=mask_in & mask,
            other=0.0
        )

        # Perform convolution (batched matmul over kernel dims)
        # (BLOCK_H, BLOCK_W, BLOCK_C) @ (BLOCK_C, kernel_h, kernel_w) -> (BLOCK_H, BLOCK_W, kernel_h, kernel_w)
        # Then sum over kernel dims
        # Using tensor core matmul with fp16/bf16
        acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    # Apply bias
    bias_offset = pid_c * BLOCK_C * bias_stride + tl.arange(0, BLOCK_C)
    bias_mask = bias_offset < out_channels
    bias = tl.load(bias_ptr + bias_offset, mask=bias_mask, other=0.0)
    acc = acc + bias[None, None, :]

    # Apply Mish activation (fused)
    # Mish: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x))
    # We approximate with a numerically stable version
    # Use online softmax-like approach for numerical stability
    # Avoid exp and tanh blowup
    # Use log1p(exp(x))
    # We can compute it safely in float32
    # Then use tanh
    # We apply in a fused way

    # First, compute softplus: log(1 + exp(x))
    x = acc
    # Avoid overflow: clamp input to prevent exp overflow
    x = tl.clamp(x, -30.0, 30.0)
    # Compute softplus
    softplus = tl.math.log1p(tl.math.exp(x))
    # Compute tanh(softplus)
    tanh_val = tl.math.tanh(softplus)
    # Final output
    out = x * tanh_val

    # Store result
    tl.store(
        out_ptr + output_offset,
        out,
        mask=mask
    )


# Triton kernel for second Mish activation (separate since fused is not needed)
@triton.jit
def mish_kernel(
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

    # Mish: x * tanh(softplus(x))
    # Clamp to prevent overflow in exp
    x = tl.clamp(x, -30.0, 30.0)
    softplus = tl.math.log1p(tl.math.exp(x))
    tanh_val = tl.math.tanh(softplus)
    out = x * tanh_val

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv2d_mish(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    # Ensure contiguous on GPU and fp32
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Input dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    out_h = height - kernel_h + 1
    out_w = width - kernel_w + 1

    # Output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)

    # Strides for indexing
    x_stride_0, x_stride_1, x_stride_2, x_stride_3 = x.stride()
    w_stride_0, w_stride_1, w_stride_2, w_stride_3 = weight.stride()
    out_stride_0, out_stride_1, out_stride_2, out_stride_3 = out.stride()
    bias_stride = bias.stride(0)

    # Define block sizes based on A100 architecture
    BLOCK_H = 32
    BLOCK_W = 32
    BLOCK_C = 16

    # Grid dimensions
    grid_h = triton.cdiv(out_h, BLOCK_H)
    grid_w = triton.cdiv(out_w, BLOCK_W)
    grid_c = triton.cdiv(out_channels, BLOCK_C)

    grid = (grid_h, grid_w, grid_c)

    # Launch kernel
    conv2d_mish_kernel[grid](
        x_ptr=x,
        w_ptr=weight,
        out_ptr=out,
        bias_ptr=bias,
        x_stride_0=x_stride_0,
        x_stride_1=x_stride_1,
        x_stride_2=x_stride_2,
        x_stride_3=x_stride_3,
        w_stride_0=w_stride_0,
        w_stride_1=w_stride_1,
        w_stride_2=w_stride_2,
        w_stride_3=w_stride_3,
        out_stride_0=out_stride_0,
        out_stride_1=out_stride_1,
        out_stride_2=out_stride_2,
        out_stride_3=out_stride_3,
        bias_stride=bias_stride,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        out_h=out_h,
        out_w=out_w,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_C=BLOCK_C,
    )
    return out


def triton_mish(x: torch.Tensor):
    # Ensure contiguous
    x = x.contiguous()
    out = torch.empty_like(x)

    # Grid size
    n_elements = x.numel()
    BLOCK_SIZE = 128

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    mish_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)
        # Set requires_grad to False to avoid retraining
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Fused conv2d + mish
        x = triton_conv2d_mish(x, self.conv.weight, self.conv.bias)
        # Second mish
        x = triton_mish(x)
        return x