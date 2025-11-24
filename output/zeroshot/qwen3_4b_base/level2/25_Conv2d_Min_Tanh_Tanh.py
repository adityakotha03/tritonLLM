import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # pointer to input tensor (batch, in_channels, H, W)
    output_ptr,  # pointer to output tensor (batch, out_channels, H, W)
    input_shape,  # (batch, in_channels, H, W)
    output_shape,  # (batch, out_channels, H, W)
    kernel,  # (out_channels, in_channels, kernel_size, kernel_size)
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
):
    # Define block dimensions
    batch = tl.program_id(0)
    out_channel = tl.program_id(1)
    out_h = tl.program_id(2)
    out_w = tl.program_id(3)

    # Compute output spatial indices
    out_h_idx = out_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_w_idx = out_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_h_idx = out_h_idx % output_shape[3]  # height
    out_w_idx = out_w_idx % output_shape[4]  # width

    # Compute input spatial indices with padding
    pad_h = padding
    pad_w = padding
    in_h_idx = out_h_idx - pad_h
    in_w_idx = out_w_idx - pad_w

    # Mask to avoid out-of-bounds
    in_h_mask = (in_h_idx >= 0) & (in_h_idx < input_shape[2])
    in_w_mask = (in_w_idx >= 0) & (in_w_idx < input_shape[3])
    valid_mask = in_h_mask & in_w_mask

    # Load kernel weights
    kernel_offset = out_channel * kernel_size * kernel_size * GROUPS + tl.arange(0, GROUPS)
    kernel_weights = tl.load(kernel + kernel_offset, mask=tl.arange(0, GROUPS) < GROUPS, other=0.0)

    # Compute input features
    input_features = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            h = in_h_idx[i]
            w = in_w_idx[j]
            if h < 0 or h >= input_shape[2] or w < 0 or w >= input_shape[3]:
                continue
            # Load input feature
            input_val = tl.load(input_ptr + batch * input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] +
                                out_channel * input_shape[1] * input_shape[2] * input_shape[3] +
                                h * input_shape[1] * input_shape[3] + w, mask=valid_mask, other=0.0)
            # Accumulate weighted sum
            input_features[i, j] = input_val

    # Convolution output
    output = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            h = in_h_idx[i]
            w = in_w_idx[j]
            if not valid_mask[i, j]:
                continue
            # Compute convolution sum
            conv_sum = 0.0
            for k in range(kernel_size):
                for l in range(kernel_size):
                    h_idx = h + k - padding
                    w_idx = w + l - padding
                    if h_idx >= 0 and h_idx < input_shape[2] and w_idx >= 0 and w_idx < input_shape[3]:
                        input_val = tl.load(input_ptr + batch * input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] +
                                            out_channel * input_shape[1] * input_shape[2] * input_shape[3] +
                                            h_idx * input_shape[1] * input_shape[3] + w_idx, mask=valid_mask, other=0.0)
                        kernel_val = tl.load(kernel + (out_channel * kernel_size * kernel_size + k * kernel_size + l), mask=valid_mask, other=0.0)
                        conv_sum += input_val * kernel_val
            output[i, j] = conv_sum

    # Store output
    output_ptr_offset = batch * output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] + \
                        out_channel * output_shape[2] * output_shape[3] + \
                        out_h_idx * output_shape[3] + out_w_idx
    tl.store(output_ptr + output_ptr_offset, output, mask=valid_mask)


@triton.jit
def min_tanh_kernel(
    x_ptr,  # pointer to input tensor (batch, C, H, W)
    out_ptr,  # pointer to output tensor (batch, 1, H, W)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of spatial indices
    batch = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)

    # Define spatial offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    h_offsets = h * BLOCK_SIZE + offsets
    w_offsets = w * BLOCK_SIZE + offsets

    # Mask to ensure valid bounds
    h_mask = (h_offsets < height)
    w_mask = (w_offsets < width)
    valid_mask = h_mask & w_mask

    # Load input values
    x = tl.load(x_ptr + batch * in_channels * height * width + tl.arange(0, BLOCK_SIZE), mask=valid_mask, other=0.0)

    # Reduce over channel dimension (dim=1)
    min_val = tl.min(x, axis=0)
    # Apply tanh
    tanh_val = tl.tanh(min_val)
    # Store result
    tl.store(out_ptr + batch * height * width + h_offsets, tanh_val, mask=valid_mask)


def triton_conv2d(input_tensor, kernel):
    """
    Custom Triton kernel for 2D convolution.
    """
    batch, in_channels, H, W = input_tensor.shape
    out_channels, _, kernel_size, _ = kernel.shape
    stride = 1
    padding = 1

    # Ensure contiguous
    input_tensor = input_tensor.contiguous()
    output_tensor = torch.empty((batch, out_channels, H, W), dtype=torch.float32, device=input_tensor.device)

    # Define block size
    BLOCK_SIZE = 16

    # Grid dimensions
    grid = lambda meta: (
        (batch + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (out_channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (H + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    # Launch kernel
    conv2d_kernel[grid](
        input_tensor.data_ptr(),
        output_tensor.data_ptr(),
        input_tensor.shape,
        output_tensor.shape,
        kernel.data_ptr(),
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE,
        1,
    )
    return output_tensor


def triton_min_tanh(x):
    """
    Custom Triton kernel for min along channel and tanh.
    """
    batch, in_channels, H, W = x.shape
    out = torch.empty((batch, 1, H, W), dtype=torch.float32, device=x.device)

    BLOCK_SIZE = 16

    grid = lambda meta: (
        (batch + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (H + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    min_tanh_kernel[grid](
        x.data_ptr(),
        out.data_ptr(),
        batch,
        in_channels,
        H,
        W,
        BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # Initialize kernel weights
        self.kernel = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32, device="cuda")

    def forward(self, x):
        # Step 1: Convolution
        x = triton_conv2d(x, self.kernel)
        # Step 2: Min over channel dimension
        x = triton_min_tanh(x)
        # Step 3: Tanh activation
        x = torch.tanh(x)
        # Step 4: Final Tanh
        x = torch.tanh(x)
        return x