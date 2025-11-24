import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # pointer to input tensor (batch, in_channels, H, W)
    weight_ptr,  # pointer to convolution weights (out_channels, in_channels, kernel_size, kernel_size)
    bias_ptr,  # pointer to bias (out_channels)
    output_ptr,  # pointer to output tensor (batch, out_channels, H_out, W_out)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_size: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Compute grid dimensions
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)

    # Compute output spatial dimensions
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W

    # Define the range of indices in the output spatial dimensions
    h_range = tl.arange(0, BLOCK_SIZE_H)
    w_range = tl.arange(0, BLOCK_SIZE_W)

    # Compute output indices
    h_idx = h_start + h_range
    w_idx = w_start + w_range

    # Check bounds
    h_mask = (h_idx < height)
    w_mask = (w_idx < width)
    mask = h_mask & w_mask

    # Compute input and output indices
    # Input: (batch, in_channels, H, W)
    # Output: (batch, out_channels, H_out, W_out)
    # Output spatial size: (H - kernel_size + 2*pad_h, W - kernel_size + 2*pad_w)
    # We assume zero padding
    pad_h = (kernel_size - 1) // 2
    pad_w = (kernel_size - 1) // 2
    h_offset = h_idx - pad_h
    w_offset = w_idx - pad_w

    # Compute valid input indices
    h_input = h_offset + tl.arange(0, kernel_size)
    w_input = w_offset + tl.arange(0, kernel_size)

    # Create valid mask for input indices
    h_input_mask = (h_input >= 0) & (h_input < height)
    w_input_mask = (w_input >= 0) & (w_input < width)
    valid_mask = h_input_mask[:, :, None, None] & w_input_mask[:, :, None, None]

    # Load input data (batch, in_channels, H, W)
    # We use a tile-based approach to avoid full memory load
    # For simplicity, we assume input is contiguous and we load in a 2D block
    # We will use a 2D loop over output and input spatial dimensions
    # Instead, we restructure to compute output (b, oc, h_out, w_out) with kernel
    # We use a different approach: we compute the output at each (h_idx, w_idx)

    # We recompute the output value at each (h_idx, w_idx)
    # We use a loop over in_channels and kernel_size
    # We use shared memory to cache weights and input patches

    # Shared memory for input patch (in_channels, kernel_size, kernel_size)
    input_patch = tl.zeros((in_channels, kernel_size, kernel_size), dtype=tl.float32)

    # Load input patch into shared memory
    # We compute input indices for the current output position
    # input_idx: (in_channels, kernel_size, kernel_size)
    # We compute input indices as: (h_input, w_input)
    # We use a 2D loop over kernel
    # We will do this in a nested loop over kernel
    # We compute the output value for each output position

    # We instead use a more efficient approach: compute output value per output pixel
    # We use a 3D loop over in_channels, kernel_h, kernel_w
    # We will load input and weights in a fused way

    # We restructure: compute output value at (h_idx, w_idx)
    # We loop over output channels
    oc = tl.arange(0, out_channels)
    # Loop over in_channels
    ic = tl.arange(0, in_channels)

    # We will compute the output value for each output channel
    # We use a fused computation: output[oc] = sum over ic and kernel_h, kernel_w of input[ic] * weight[oc, ic, kh, kw]
    # We load weights into shared memory
    # We use a 2D kernel loop

    # Load weights into shared memory (out_channels, in_channels, kernel_size, kernel_size)
    # We do this in a way that avoids redundant loads
    # We use a 2D loop over kernel
    # We assume weights are contiguous

    # We will compute output for one output pixel at a time
    # We use a 3D loop over kernel and in_channels
    # We compute the output value for each output pixel

    # We compute output value at (h_idx, w_idx) for each output channel
    # We use a loop over in_channels and kernel indices
    # We load input and weights in a tiled fashion

    # We define the output value
    output_val = tl.zeros((out_channels), dtype=tl.float32)

    # We loop over in_channels and kernel indices
    # We use a 2D loop over kernel
    # We compute the convolution sum
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Compute input indices
            h_in = h_idx + kh - pad_h
            w_in = w_idx + kw - pad_w
            # Check bounds
            h_in_mask = (h_in >= 0) & (h_in < height)
            w_in_mask = (w_in >= 0) & (w_in < width)
            valid_in = h_in_mask & w_in_mask
            # Load input
            # We load input in a 2D fashion
            # We use a single load per input pixel
            # We will use a different approach: tile the input and weights

            # Instead, we use a more efficient approach: we pre-load the input patch
            # We compute input patch for current output position
            # We load input into shared memory
            # We do this in a separate loop

            # We will instead use a different kernel design: we loop over in_channels and kernel
            # We compute the sum over kernel and in_channels
            pass

    # We restructure to use a more efficient kernel
    # We compute the output value at (h_idx, w_idx) for each output channel
    # We loop over in_channels and kernel indices
    # We use a 2D loop over kernel
    # We compute the convolution sum

    # We use a 3D loop over in_channels, kh, kw
    # We load input and weights in a fused way
    # We use a single loop over in_channels and kernel indices

    # We define the output value
    output_val = tl.zeros((out_channels), dtype=tl.float32)

    # We loop over in_channels and kernel indices
    # We compute the convolution sum
    # We use a 2D loop over kernel
    # We compute the output value at (h_idx, w_idx)
    for ic in tl.arange(0, in_channels):
        for kh in tl.arange(0, kernel_size):
            for kw in tl.arange(0, kernel_size):
                # Compute input index
                h_in = h_idx + kh - pad_h
                w_in = w_idx + kw - pad_w
                # Check bounds
                h_in_mask = (h_in >= 0) & (h_in < height)
                w_in_mask = (w_in >= 0) & (w_in < width)
                valid_in = h_in_mask & w_in_mask
                # Load input
                input_val = tl.load(
                    input_ptr + (batch_size * in_channels * height * width + 
                                ic * height * width + h_in * width + w_in),
                    mask=valid_in,
                    other=0.0
                )
                # Load weight
                weight_val = tl.load(
                    weight_ptr + (out_channels * in_channels * kernel_size * kernel_size +
                                ic * kernel_size * kernel_size + kh * kernel_size + kw),
                    mask=valid_in,
                    other=0.0
                )
                # Accumulate
                output_val += input_val * weight_val

    # Add bias
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + oc, mask=oc < out_channels, other=0.0)
        output_val += bias_val

    # Store output
    tl.store(output_ptr + (batch_size * out_channels * height * width + 
                          oc * height * width + h_idx * width + w_idx),
             output_val, mask=mask)


@triton.jit
def batch_norm_kernel(
    x_ptr,  # pointer to input (batch, C, H, W)
    running_mean_ptr,  # pointer to running mean (C)
    running_var_ptr,  # pointer to running variance (C)
    gamma_ptr,  # pointer to gamma (C)
    beta_ptr,  # pointer to beta (C)
    output_ptr,  # pointer to output (batch, C, H, W)
    batch_size: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes a block of output
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * C * H * W)

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Load parameters
    mean = tl.load(running_mean_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C, other=0.0)
    var = tl.load(running_var_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C, other=1.0)
    gamma = tl.load(gamma_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C, other=1.0)
    beta = tl.load(beta_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C, other=0.0)

    # Compute normalization
    # x_norm = (x - mean) / sqrt(var + epsilon)
    # output = gamma * x_norm + beta
    # We compute this per channel
    # We loop over channels
    C_idx = tl.arange(0, C)
    # Compute per-channel normalization
    x_norm = (x - mean[C_idx]) / tl.sqrt(var[C_idx] + epsilon)
    output = gamma[C_idx] * x_norm + beta[C_idx]

    # Store output
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    Performs 2D convolution using a custom Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None

    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    pad_h = (kernel_size - 1) // 2
    pad_w = (kernel_size - 1) // 2
    out_height = height - kernel_size + 2 * pad_h
    out_width = width - kernel_size + 2 * pad_w

    # Output shape
    output = torch.empty((batch_size, out_channels, out_height, out_width), dtype=x.dtype, device=x.device)

    # Define kernel parameters
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

    # Grid
    grid = lambda meta: (
        (out_height + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (out_width + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
    )

    # Launch kernel
    conv2d_kernel[
        grid,
        (BLOCK_SIZE_H, BLOCK_SIZE_W)
    ](
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr() if bias is not None else None,
        output.data_ptr(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        pad_h,
        pad_w,
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
    )
    return output


def triton_batch_norm(x: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
    """
    Performs batch normalization using a custom Triton kernel.
    """
    assert x.is_cuda and running_mean.is_cuda and running_var.is_cuda and gamma.is_cuda and beta.is_cuda, "All inputs must be on CUDA."
    x = x.contiguous()
    running_mean = running_mean.contiguous()
    running_var = running_var.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    batch_size, C, H, W = x.shape
    epsilon = 1e-5

    # Output tensor
    output = torch.empty_like(x)

    # Grid
    BLOCK_SIZE = 128
    grid = lambda meta: ((batch_size * C * H * W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    batch_norm_kernel[
        grid,
        (BLOCK_SIZE,)
    ](
        x.data_ptr(),
        running_mean.data_ptr(),
        running_var.data_ptr(),
        gamma.data_ptr(),
        beta.data_ptr(),
        output.data_ptr(),
        batch_size,
        C,
        H,
        W,
        epsilon,
        BLOCK_SIZE,
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super().__init__()
        # Define convolution weights and bias
        self.conv_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float16, device='cuda')
        self.conv_bias = torch.zeros(out_channels, dtype=torch.float16, device='cuda')
        # Define batch norm parameters
        self.bn_running_mean = torch.zeros(out_channels, dtype=torch.float32, device='cuda')
        self.bn_running_var = torch.ones(out_channels, dtype=torch.float32, device='cuda')
        self.bn_gamma = torch.ones(out_channels, dtype=torch.float32, device='cuda')
        self.bn_beta = torch.zeros(out_channels, dtype=torch.float32, device='cuda')
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Step 1: Convolution
        x = triton_conv2d(x, self.conv_weight, self.conv_bias)
        # Step 2: Batch Normalization
        x = triton_batch_norm(x, self.bn_running_mean, self.bn_running_var, self.bn_gamma, self.bn_beta)
        # Step 3: Scale
        x = x * self.scaling_factor
        return x