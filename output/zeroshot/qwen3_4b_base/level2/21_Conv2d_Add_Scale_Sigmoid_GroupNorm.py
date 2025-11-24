import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # pointer to input tensor (B, C_in, H, W)
    output_ptr,  # pointer to output tensor (B, C_out, H_out, W_out)
    weight_ptr,  # pointer to convolution weights (C_out, C_in, K, K)
    bias_ptr,  # pointer to bias (C_out, 1, 1)
    scale_ptr,  # pointer to scale (C_out, 1, 1)
    B: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
):
    # Get the program ID for batch, channel, and spatial dimensions
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    row_id = tl.program_id(2)
    col_id = tl.program_id(3)

    # Compute the output spatial dimensions
    H_out = (H + 2 * pad_h - K) // stride_h + 1
    W_out = (W + 2 * pad_w - K) // stride_w + 1

    # Compute the current output position
    h_out = row_id * BLOCK_SIZE_H
    w_out = col_id * BLOCK_SIZE_W

    # Define the range of input positions to process
    h_in = tl.arange(0, K)
    w_in = tl.arange(0, K)

    # Load weights and bias
    weights = tl.load(weight_ptr + channel_id * C_in * K * K + tl.arange(0, C_in) * K * K + h_in[:, None] * K + w_in[None, :], mask=(h_in < K) & (w_in < K), other=0.0)
    bias = tl.load(bias_ptr + channel_id * 1 * 1 + tl.arange(0, 1), mask=(channel_id < C_out), other=0.0)
    scale = tl.load(scale_ptr + channel_id * 1 * 1 + tl.arange(0, 1), mask=(channel_id < C_out), other=1.0)

    # Compute output spatial indices
    h_start = h_out * stride_h
    w_start = w_out * stride_w
    h_end = h_start + K
    w_end = w_start + K

    # Define input and output indices
    h_idx = tl.arange(0, H)
    w_idx = tl.arange(0, W)

    # Create mask for valid input positions
    h_mask = (h_idx >= h_start) & (h_idx < h_end)
    w_mask = (w_idx >= w_start) & (w_idx < w_end)

    # Load input features for all channels
    input_features = tl.zeros((C_in, H, W), dtype=tl.float16)
    for c in tl.arange(0, C_in):
        input_data = tl.load(input_ptr + batch_id * C_in * H * W + c * H * W + h_idx[:, None] * W + w_idx[None, :], mask=h_mask[:, None] & w_mask[None, :], other=0.0)
        input_features = input_features + input_data

    # Compute convolution via inner product
    output = tl.zeros((C_out, H_out, W_out), dtype=tl.float16)
    for c_out in tl.arange(0, C_out):
        # Load weight slice
        weight_slice = tl.load(weight_ptr + c_out * C_in * K * K + tl.arange(0, C_in) * K * K + h_in[:, None] * K + w_in[None, :], mask=(h_in < K) & (w_in < K), other=0.0)
        # Compute convolution
        conv = tl.dot(input_features, weight_slice, 1)
        # Apply bias and scale
        conv = conv + bias[c_out]
        conv = conv * scale[c_out]
        # Apply sigmoid
        sig = tl.sigmoid(conv)
        # Store to output
        output = output + sig

    # Store output
    output_ptr_idx = batch_id * C_out * H_out * W_out + channel_id * H_out * W_out + h_out * W_out + w_out
    tl.store(output_ptr + output_ptr_idx, output, mask=(h_out < H_out) & (w_out < W_out))


@triton.jit
def group_norm_kernel(
    x_ptr,  # pointer to input (B, C, H, W)
    g_ptr,  # pointer to group norm groups
    C: tl.constexpr,
    G: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    h_id = tl.program_id(2)
    w_id = tl.program_id(3)

    # Compute output spatial indices
    h_out = h_id * BLOCK_SIZE
    w_out = w_id * BLOCK_SIZE

    # Load input data
    x = tl.load(x_ptr + batch_id * C * H * W + channel_id * H * W + h_out * W + w_out, mask=(h_out < H) & (w_out < W), other=0.0)

    # Compute group index
    group_idx = channel_id // (C // G)
    group_offset = channel_id % (C // G)

    # Load group statistics (mean and variance) from shared memory
    # Note: In practice, group norm requires per-group mean and variance, which we compute in the kernel
    # For simplicity, we simulate a fused group norm with mean and variance computed on the fly
    # In real implementation, this would be precomputed or cached

    # Compute mean and variance across spatial dimensions
    mean = tl.sum(x, axes=[1, 2, 3]) / (H * W)
    var = tl.sum((x - mean) ** 2, axes=[1, 2, 3]) / (H * W)

    # Normalize
    x_norm = (x - mean) / tl.sqrt(var + 1e-5)

    # Store result
    tl.store(x_ptr + batch_id * C * H * W + channel_id * H * W + h_out * W + w_out, x_norm, mask=(h_out < H) & (w_out < W))


def triton_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    pad_h: int = 1,
    pad_w: int = 1,
    stride_h: int = 1,
    stride_w: int = 1,
    kernel_size: int = 3,
):
    """
    Custom convolution kernel using Triton.
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda and scale.is_cuda, "All tensors must be on CUDA."
    assert input.dtype == torch.float16 or input.dtype == torch.bfloat16, "Input must be FP16 or BF16."

    B, C_in, H, W = input.shape
    C_out, _, K, K = weight.shape
    H_out = (H + 2 * pad_h - K) // stride_h + 1
    W_out = (W + 2 * pad_w - K) // stride_w + 1

    # Ensure contiguous memory
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    scale = scale.contiguous()

    # Allocate output
    output = torch.empty((B, C_out, H_out, W_out), dtype=input.dtype, device=input.device)

    # Define block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

    # Grid dimensions
    grid = lambda meta: (
        (B + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (C_out + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
        (H_out + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (W_out + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
    )

    # Launch kernel
    conv2d_kernel[grid](
        input.data_ptr(),
        output.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        scale.data_ptr(),
        B, C_in, C_out, H, W, K,
        pad_h, pad_w,
        BLOCK_SIZE_H, BLOCK_SIZE_W,
        stride_h, stride_w,
        BLOCK_SIZE=BLOCK_SIZE_H,
    )

    return output


def triton_group_norm(
    x: torch.Tensor,
    num_groups: int,
):
    """
    Custom group normalization kernel using Triton.
    """
    assert x.is_cuda, "Input must be on CUDA."
    B, C, H, W = x.shape

    # Ensure contiguous memory
    x = x.contiguous()

    # Allocate output
    out = torch.empty_like(x)

    # Define block size
    BLOCK_SIZE = 16

    # Grid dimensions
    grid = lambda meta: ((B + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], (C + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], (H + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], (W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"])

    # Launch kernel
    group_norm_kernel[grid](
        x.data_ptr(),
        None,  # dummy pointer for group norm (in practice, we'd pass group info)
        C, num_groups, H, W, BLOCK_SIZE,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super().__init__()
        # Use custom kernels instead of standard PyTorch operators
        self.conv = None  # will be replaced in forward
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = None  # will be replaced in forward

    def forward(self, x):
        # Step 1: Convolution with bias and scale
        # We use a custom Triton kernel for convolution
        conv_weight = torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size).to(x.device)
        conv_output = triton_conv2d(
            x,
            conv_weight,
            self.bias,
            self.scale,
            pad_h=1,
            pad_w=1,
            stride_h=1,
            stride_w=1,
            kernel_size=self.kernel_size,
        )
        
        # Step 2: Add bias and scale
        conv_output = conv_output + self.bias
        conv_output = conv_output * self.scale
        
        # Step 3: Apply sigmoid
        sig_output = torch.sigmoid(conv_output)
        
        # Step 4: Apply group normalization
        norm_output = triton_group_norm(sig_output, num_groups=self.num_groups)
        
        return norm_output