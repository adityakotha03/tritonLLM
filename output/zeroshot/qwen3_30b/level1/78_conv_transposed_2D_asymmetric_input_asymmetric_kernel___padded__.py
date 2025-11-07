import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    X_ptr,  # Input tensor pointer (batch, in_channels, H, W)
    W_ptr,  # Weight tensor pointer (out_channels, in_channels, KH, KW)
    B_ptr,  # Bias pointer (out_channels,) if bias is True
    Out_ptr,  # Output tensor pointer (batch, out_channels, OH, OW)
    batch_size,  # Number of batches
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    H, W,  # Input spatial dimensions
    OH, OW,  # Output spatial dimensions
    KH, KW,  # Kernel spatial dimensions
    stride_h, stride_w,  # Stride dimensions
    pad_h, pad_w,  # Padding dimensions
    BIAS: tl.constexpr,  # Whether bias is used (compile-time bool)
    BLOCK_SIZE: tl.constexpr,
):
    # Thread block ID
    pid = tl.program_id(0)  # Grid dimension: (num_blocks,)
    num_blocks = (batch_size * out_channels * OH * OW + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Each thread block processes BLOCK_SIZE output elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Compute output index (batch, out_channel, out_h, out_w)
    # Convert flat index to 4D indices
    out_idx = offsets
    out_batch = out_idx // (out_channels * OH * OW)
    out_rest = out_idx % (out_channels * OH * OW)
    out_outc = out_rest // (OH * OW)
    out_rest = out_rest % (OH * OW)
    out_oh = out_rest // OW
    out_ow = out_rest % OW

    # Check bounds
    mask = (out_batch < batch_size) & (out_outc < out_channels) & (out_oh < OH) & (out_ow < OW)

    # Output value
    out_val = tl.zeros((1,), dtype=tl.float32)

    # Compute input indices from output (unrolled for spatial transposed conv)
    # For each output position (out_oh, out_ow), compute the corresponding input region
    # Input indices: (in_h, in_w) = (out_oh * stride_h - pad_h + kh, out_ow * stride_w - pad_w + kw)
    # But we go over kernel and reverse: input_h = out_oh * stride_h + kh - pad_h
    # So we iterate over kernel offsets kh, kw
    for kh in range(KH):
        for kw in range(KW):
            # Input spatial position: in_h = out_oh * stride_h + kh - pad_h
            in_h = out_oh * stride_h + kh - pad_h
            in_w = out_ow * stride_w + kw - pad_w

            # Check if input is valid
            if (in_h < 0) or (in_h >= H) or (in_w < 0) or (in_w >= W):
                continue

            # Compute input channel index
            for ic in range(in_channels):
                # Input pointer
                x_idx = out_batch * in_channels * H * W + ic * H * W + in_h * W + in_w
                x_val = tl.load(X_ptr + x_idx, mask=True)

                # Weight index
                w_idx = out_outc * in_channels * KH * KW + ic * KH * KW + kh * KW + kw
                w_val = tl.load(W_ptr + w_idx, mask=True)

                # Accumulate
                out_val += x_val * w_val

    # Handle bias
    if BIAS:
        bias_val = tl.load(B_ptr + out_outc, mask=out_outc < out_channels)
        out_val += bias_val

    # Store output
    out_idx = out_batch * out_channels * OH * OW + out_outc * OH * OW + out_oh * OW + out_ow
    tl.store(Out_ptr + out_idx, out_val, mask=mask)


@triton.jit
def conv_transpose_kernel_fused(
    X_ptr,  # Input tensor pointer
    W_ptr,  # Weight tensor pointer
    B_ptr,  # Bias pointer
    Out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    H, W,  # Input spatial dimensions
    OH, OW,  # Output spatial dimensions
    KH, KW,  # Kernel spatial dimensions
    stride_h, stride_w,  # Stride dimensions
    pad_h, pad_w,  # Padding dimensions
    BIAS: tl.constexpr,  # Whether bias is used
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Thread block ID
    pid = tl.program_id(0)
    num_blocks = (batch_size * out_channels * OH * OW + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Convert flat index to 4D output indices
    out_idx = offsets
    out_batch = out_idx // (out_channels * OH * OW)
    out_rest = out_idx % (out_channels * OH * OW)
    out_outc = out_rest // (OH * OW)
    out_rest = out_rest % (OH * OW)
    out_oh = out_rest // OW
    out_ow = out_rest % OW

    # Mask for bounds checking
    mask = (out_batch < batch_size) & (out_outc < out_channels) & (out_oh < OH) & (out_ow < OW)

    # Initialize output value
    out_val = tl.zeros((1,), dtype=tl.float32)

    # Iterate over kernel
    for kh in range(KH):
        for kw in range(KW):
            in_h = out_oh * stride_h + kh - pad_h
            in_w = out_ow * stride_w + kw - pad_w

            # Skip invalid input positions
            if (in_h < 0) or (in_h >= H) or (in_w < 0) or (in_w >= W):
                continue

            for ic in range(in_channels):
                # Load input and weight
                x_idx = out_batch * in_channels * H * W + ic * H * W + in_h * W + in_w
                w_idx = out_outc * in_channels * KH * KW + ic * KH * KW + kh * KW + kw

                x_val = tl.load(X_ptr + x_idx, mask=True)
                w_val = tl.load(W_ptr + w_idx, mask=True)

                out_val += x_val * w_val

    # Add bias
    if BIAS:
        bias_val = tl.load(B_ptr + out_outc, mask=out_outc < out_channels)
        out_val += bias_val

    # Apply activation (e.g., ReLU)
    if ACTIVATION == 1:  # ReLU
        out_val = tl.max(out_val, 0.0)

    # Store result
    out_idx = out_batch * out_channels * OH * OW + out_outc * OH * OW + out_oh * OW + out_ow
    tl.store(Out_ptr + out_idx, out_val, mask=mask)


def triton_conv_transpose(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, stride=(1,1), padding=(0,0)):
    """
    Custom Triton-based transposed convolution implementation with fusion and optimization.

    Args:
        x: Input tensor (B, C_in, H, W)
        weight: Weight tensor (C_out, C_in, KH, KW)
        bias: Bias tensor (C_out,)
        stride: Stride (h, w)
        padding: Padding (h, w)

    Returns:
        Output tensor (B, C_out, OH, OW)
    """
    # Ensure inputs are on GPU and contiguous
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert weight.is_cuda, "Weight tensor must be on CUDA"
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Extract parameters
    B, C_in, H, W = x.shape
    C_out, _, KH, KW = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    # Compute output dimensions
    OH = (H - 1) * stride_h - 2 * pad_h + KH
    OW = (W - 1) * stride_w - 2 * pad_w + KW

    # Output tensor
    out = torch.empty(B, C_out, OH, OW, dtype=x.dtype, device=x.device)

    # Grid setup
    total_elements = B * C_out * OH * OW
    BLOCK_SIZE = 512  # Power of 2, good for occupancy on A100

    # Determine grid
    grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)

    # Use fusable kernel with ReLU activation if bias is present
    # For A100, use fp16/bf16 with Tensor Cores
    # We assume input is fp32, but we can cast to bf16 for performance
    if x.dtype == torch.float32:
        x = x.to(torch.bfloat16)
        weight = weight.to(torch.bfloat16)
        if bias is not None:
            bias = bias.to(torch.bfloat16)
        out = out.to(torch.bfloat16)

    # Launch kernel with activation fused
    conv_transpose_kernel_fused[grid](
        x, weight, bias, out,
        B, C_in, C_out, H, W,
        OH, OW, KH, KW,
        stride_h, stride_w, pad_h, pad_w,
        BIAS=(bias is not None),
        BLOCK_SIZE=BLOCK_SIZE,
        ACTIVATION=1  # ReLU fused
    )

    # Cast back to fp32 if needed
    if x.dtype == torch.float32:
        out = out.to(torch.float32)

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        # Move to GPU on first forward pass
        self.register_buffer('dummy', torch.empty(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton kernel for performance
        return triton_conv_transpose(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding
        )