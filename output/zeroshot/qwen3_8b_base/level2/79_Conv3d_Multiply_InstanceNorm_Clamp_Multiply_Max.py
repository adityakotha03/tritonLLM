import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C_in, D, H, W)
    weight_shape,  # (C_out, C_in, Kd, Kh, Kw)
    output_shape,  # (N, C_out, D, H, W)
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID (block index)
    pid = tl.program_id(0)
    # Get the thread ID within the block
    tid = tl.program_id(1)

    # Compute the output index
    out_idx = pid * BLOCK_SIZE + tid
    # Compute the output coordinates (N, C_out, D, H, W)
    N, C_out, D, H, W = output_shape
    out_n = out_idx // (C_out * D * H * W)
    out_c = (out_idx // (D * H * W)) % C_out
    out_d = (out_idx // (H * W)) % D
    out_h = (out_idx // W) % H
    out_w = out_idx % W

    # Compute the input coordinates (N, C_in, D, H, W)
    input_n = out_n
    input_c = tl.load(weight_ptr + out_c * weight_shape[1] + tid, mask=tl.arange(0, weight_shape[1]) < weight_shape[1], other=0)
    input_d = out_d - padding_d
    input_h = out_h - padding_h
    input_w = out_w - padding_w

    # Compute the weight indices
    weight_c_in = tl.arange(0, weight_shape[1])
    weight_kd = tl.arange(0, weight_shape[2])
    weight_kh = tl.arange(0, weight_shape[3])
    weight_kw = tl.arange(0, weight_shape[4])

    # Compute the input indices
    input_d_start = input_d - weight_kd + stride_d
    input_h_start = input_h - weight_kh + stride_h
    input_w_start = input_w - weight_kw + stride_w

    # Compute the input offset
    input_offset = input_n * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]
    input_offset += input_c * input_shape[2] * input_shape[3] * input_shape[4]
    input_offset += input_d_start * input_shape[3] * input_shape[4]
    input_offset += input_h_start * input_shape[4]
    input_offset += input_w_start

    # Load input values
    input_val = tl.load(input_ptr + input_offset, mask=tl.arange(0, weight_shape[1]) < weight_shape[1], other=0.0)
    input_val = input_val * tl.load(weight_ptr + out_c * weight_shape[1] + tid, mask=tl.arange(0, weight_shape[1]) < weight_shape[1], other=0.0)

    # Accumulate the result
    tl.atomic_add(output_ptr + out_n * output_shape[1] * output_shape[2] * output_shape[3] * output_shape[4] + out_c * output_shape[2] * output_shape[3] * output_shape[4] + out_d * output_shape[3] * output_shape[4] + out_h * output_shape[4] + out_w, input_val)


@triton.jit
def instance_norm_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    mean_ptr,  # Pointer to mean tensor
    var_ptr,  # Pointer to variance tensor
    eps,  # Epsilon for numerical stability
    input_shape,  # (N, C, D, H, W)
    output_shape,  # (N, C, D, H, W)
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID (block index)
    pid = tl.program_id(0)
    # Get the thread ID within the block
    tid = tl.program_id(1)

    # Compute the output index
    out_idx = pid * BLOCK_SIZE + tid
    # Compute the output coordinates (N, C, D, H, W)
    N, C, D, H, W = output_shape
    out_n = out_idx // (C * D * H * W)
    out_c = (out_idx // (D * H * W)) % C
    out_d = (out_idx // (H * W)) % D
    out_h = (out_idx // W) % H
    out_w = out_idx % W

    # Compute the input offset
    input_offset = out_n * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] + out_c * input_shape[2] * input_shape[3] * input_shape[4] + out_d * input_shape[3] * input_shape[4] + out_h * input_shape[4] + out_w
    input_val = tl.load(input_ptr + input_offset, mask=tl.arange(0, 1) < 1, other=0.0)

    # Compute mean and variance
    mean = tl.load(mean_ptr + out_c, mask=tl.arange(0, 1) < 1, other=0.0)
    var = tl.load(var_ptr + out_c, mask=tl.arange(0, 1) < 1, other=0.0)

    # Normalize
    normalized_val = (input_val - mean) / tl.sqrt(var + eps)

    # Store result
    tl.store(output_ptr + input_offset, normalized_val, mask=tl.arange(0, 1) < 1)


@triton.jit
def clamp_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    clamp_min,  # Minimum value
    clamp_max,  # Maximum value
    input_shape,  # (N, C, D, H, W)
    output_shape,  # (N, C, D, H, W)
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID (block index)
    pid = tl.program_id(0)
    # Get the thread ID within the block
    tid = tl.program_id(1)

    # Compute the output index
    out_idx = pid * BLOCK_SIZE + tid
    # Compute the output coordinates (N, C, D, H, W)
    N, C, D, H, W = output_shape
    out_n = out_idx // (C * D * H * W)
    out_c = (out_idx // (D * H * W)) % C
    out_d = (out_idx // (H * W)) % D
    out_h = (out_idx // W) % H
    out_w = out_idx % W

    # Compute the input offset
    input_offset = out_n * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] + out_c * input_shape[2] * input_shape[3] * input_shape[4] + out_d * input_shape[3] * input_shape[4] + out_h * input_shape[4] + out_w
    input_val = tl.load(input_ptr + input_offset, mask=tl.arange(0, 1) < 1, other=0.0)

    # Clamp
    clamped_val = tl.maximum(tl.minimum(input_val, clamp_max), clamp_min)

    # Store result
    tl.store(output_ptr + input_offset, clamped_val, mask=tl.arange(0, 1) < 1)


@triton.jit
def multiply_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, D, H, W)
    weight_shape,  # (C, 1, 1, 1)
    output_shape,  # (N, C, D, H, W)
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID (block index)
    pid = tl.program_id(0)
    # Get the thread ID within the block
    tid = tl.program_id(1)

    # Compute the output index
    out_idx = pid * BLOCK_SIZE + tid
    # Compute the output coordinates (N, C, D, H, W)
    N, C, D, H, W = output_shape
    out_n = out_idx // (C * D * H * W)
    out_c = (out_idx // (D * H * W)) % C
    out_d = (out_idx // (H * W)) % D
    out_h = (out_idx // W) % H
    out_w = out_idx % W

    # Compute the input offset
    input_offset = out_n * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] + out_c * input_shape[2] * input_shape[3] * input_shape[4] + out_d * input_shape[3] * input_shape[4] + out_h * input_shape[4] + out_w
    input_val = tl.load(input_ptr + input_offset, mask=tl.arange(0, 1) < 1, other=0.0)

    # Compute weight
    weight_val = tl.load(weight_ptr + out_c, mask=tl.arange(0, 1) < 1, other=0.0)

    # Multiply
    output_val = input_val * weight_val

    # Store result
    tl.store(output_ptr + input_offset, output_val, mask=tl.arange(0, 1) < 1)


@triton.jit
def max_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, D, H, W)
    output_shape,  # (N, 1, D, H, W)
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID (block index)
    pid = tl.program_id(0)
    # Get the thread ID within the block
    tid = tl.program_id(1)

    # Compute the output index
    out_idx = pid * BLOCK_SIZE + tid
    # Compute the output coordinates (N, 1, D, H, W)
    N, C, D, H, W = input_shape
    out_n = out_idx // (D * H * W)
    out_d = (out_idx // (H * W)) % D
    out_h = (out_idx // W) % H
    out_w = out_idx % W

    # Compute the input offset
    input_offset = out_n * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] + tid * input_shape[2] * input_shape[3] * input_shape[4] + out_d * input_shape[3] * input_shape[4] + out_h * input_shape[4] + out_w
    input_val = tl.load(input_ptr + input_offset, mask=tl.arange(0, 1) < 1, other=0.0)

    # Compute max
    max_val = tl.max(input_val)

    # Store result
    tl.store(output_ptr + out_n * output_shape[1] * output_shape[2] * output_shape[3] * output_shape[4] + out_d * output_shape[3] * output_shape[4] + out_h * output_shape[4] + out_w, max_val, mask=tl.arange(0, 1) < 1)


def triton_conv3d(input, weight, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w):
    N, C_in, D, H, W = input.shape
    C_out, _, Kd, Kh, Kw = weight.shape
    output_shape = (N, C_out, D - Kd + 2 * padding_d, H - Kh + 2 * padding_h, W - Kw + 2 * padding_w)
    output = torch.empty(output_shape, device=input.device, dtype=input.dtype)

    # Calculate grid size
    grid = lambda meta: ((output.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    conv3d_kernel[grid](input, weight, output, (N, C_in, D, H, W), (C_out, C_in, Kd, Kh, Kw), (N, C_out, D - Kd + 2 * padding_d, H - Kh + 2 * padding_h, W - Kw + 2 * padding_w), stride_d, stride_h, stride_w, padding_d, padding_h, padding_w, BLOCK_SIZE=128)

    return output


def triton_instance_norm(input, mean, var, eps):
    N, C, D, H, W = input.shape
    output = torch.empty_like(input)
    output_shape = (N, C, D, H, W)

    # Calculate grid size
    grid = lambda meta: ((output.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    instance_norm_kernel[grid](input, output, mean, var, eps, (N, C, D, H, W), output_shape, BLOCK_SIZE=128)

    return output


def triton_clamp(input, clamp_min, clamp_max):
    N, C, D, H, W = input.shape
    output = torch.empty_like(input)

    # Calculate grid size
    grid = lambda meta: ((output.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    clamp_kernel[grid](input, output, clamp_min, clamp_max, (N, C, D, H, W), (N, C, D, H, W), BLOCK_SIZE=128)

    return output


def triton_multiply(input, weight):
    N, C, D, H, W = input.shape
    C_weight, _, _, _, _ = weight.shape
    output = torch.empty_like(input)

    # Calculate grid size
    grid = lambda meta: ((output.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    multiply_kernel[grid](input, weight, output, (N, C, D, H, W), (C_weight, 1, 1, 1), (N, C, D, H, W), BLOCK_SIZE=128)

    return output


def triton_max(input):
    N, C, D, H, W = input.shape
    output_shape = (N, 1, D, H, W)
    output = torch.empty(output_shape, device=input.device, dtype=input.dtype)

    # Calculate grid size
    grid = lambda meta: ((output.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    max_kernel[grid](input, output, (N, C, D, H, W), output_shape, BLOCK_SIZE=128)

    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = None
        self.multiplier = None
        self.instance_norm = None
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        # Conv3d with Triton kernel
        x = triton_conv3d(x, self.multiplier, 1, 1, 1, 1, 1, 1)
        x = triton_multiply(x, self.multiplier)
        x = triton_instance_norm(x, self.mean, self.var, 1e-5)
        x = triton_clamp(x, self.clamp_min, self.clamp_max)
        x = triton_multiply(x, self.multiplier)
        x = triton_max(x)
        return x