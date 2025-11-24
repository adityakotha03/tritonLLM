import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, D, H, W)
    kernel_size,  # (kD, kH, kW)
    stride,  # (sD, sH, sW)
    padding,  # (pD, pH, pW)
    output_padding,  # (oD, oH, oW)
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the 3D index of the current thread
    pid = tl.program_id(0)
    # Compute the 3D offset for the current thread
    offset_d = pid // (BLOCK_SIZE * BLOCK_SIZE)
    offset_h = (pid // BLOCK_SIZE) % BLOCK_SIZE
    offset_w = pid % BLOCK_SIZE

    # Compute the input and output dimensions
    N, C, D, H, W = input_shape
    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    pD, pH, pW = padding
    oD, oH, oW = output_padding

    # Compute the output dimensions
    out_D = (D - 1) * sD + kD - 2 * pD + oD
    out_H = (H - 1) * sH + kH - 2 * pH + oH
    out_W = (W - 1) * sW + kW - 2 * pW + oW

    # Compute the output index for this thread
    out_d = offset_d
    out_h = offset_h
    out_w = offset_w

    # Compute the input indices for this thread
    in_d_start = out_d * sD - pD
    in_h_start = out_h * sH - pH
    in_w_start = out_w * sW - pW

    # Compute the input indices for the kernel
    in_d = in_d_start + tl.arange(0, kD)
    in_h = in_h_start + tl.arange(0, kH)
    in_w = in_w_start + tl.arange(0, kW)

    # Flatten the input indices
    in_idx = (in_d * H * W + in_h * W + in_w) + tl.arange(0, C) * H * W * D
    in_idx = in_idx + tl.arange(0, N) * C * D * H * W

    # Load input values
    input_val = tl.load(input_ptr + in_idx, mask=in_idx < N * C * D * H * W, other=0.0)

    # Compute the weight indices
    weight_idx = (tl.arange(0, kD) * kH * kW + tl.arange(0, kH) * kW + tl.arange(0, kW)) + tl.arange(0, C) * kD * kH * kW
    weight_idx = weight_idx + tl.arange(0, out_channels) * C * kD * kH * kW

    # Load weight values
    weight_val = tl.load(weight_ptr + weight_idx, mask=weight_idx < out_channels * C * kD * kH * kW, other=0.0)

    # Compute the output value
    output_val = tl.dot(input_val, weight_val)

    # Store the output value
    out_idx = (out_d * out_H * out_W + out_h * out_W + out_w) + tl.arange(0, N) * out_channels * out_H * out_W
    tl.store(output_ptr + out_idx, output_val)


@triton.jit
def max_pool_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, D, H, W)
    kernel_size,  # (kD, kH, kW)
    stride,  # (sD, sH, sW)
    padding,  # (pD, pH, pW)
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the 3D index of the current thread
    pid = tl.program_id(0)
    # Compute the 3D offset for the current thread
    offset_d = pid // (BLOCK_SIZE * BLOCK_SIZE)
    offset_h = (pid // BLOCK_SIZE) % BLOCK_SIZE
    offset_w = pid % BLOCK_SIZE

    # Compute the input and output dimensions
    N, C, D, H, W = input_shape
    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    pD, pH, pW = padding

    # Compute the output dimensions
    out_D = (D - 1) * sD + kD - 2 * pD
    out_H = (H - 1) * sH + kH - 2 * pH
    out_W = (W - 1) * sW + kW - 2 * pW

    # Compute the output index for this thread
    out_d = offset_d
    out_h = offset_h
    out_w = offset_w

    # Compute the input indices for this thread
    in_d_start = out_d * sD - pD
    in_h_start = out_h * sH - pH
    in_w_start = out_w * sW - pW

    # Compute the input indices for the kernel
    in_d = in_d_start + tl.arange(0, kD)
    in_h = in_h_start + tl.arange(0, kH)
    in_w = in_w_start + tl.arange(0, kW)

    # Flatten the input indices
    in_idx = (in_d * H * W + in_h * W + in_w) + tl.arange(0, C) * H * W * D
    in_idx = in_idx + tl.arange(0, N) * C * D * H * W

    # Load input values
    input_val = tl.load(input_ptr + in_idx, mask=in_idx < N * C * D * H * W, other=-float('inf'))

    # Compute the max value
    max_val = tl.max(input_val)

    # Store the max value
    out_idx = (out_d * out_H * out_W + out_h * out_W + out_w) + tl.arange(0, N) * C * out_H * out_W
    tl.store(output_ptr + out_idx, max_val)


@triton.jit
def softmax_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, D, H, W)
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the 3D index of the current thread
    pid = tl.program_id(0)
    # Compute the 3D offset for the current thread
    offset_d = pid // (BLOCK_SIZE * BLOCK_SIZE)
    offset_h = (pid // BLOCK_SIZE) % BLOCK_SIZE
    offset_w = pid % BLOCK_SIZE

    # Compute the input and output dimensions
    N, C, D, H, W = input_shape

    # Compute the output index for this thread
    out_d = offset_d
    out_h = offset_h
    out_w = offset_w

    # Compute the input indices for this thread
    in_idx = (out_d * H * W + out_h * W + out_w) + tl.arange(0, N) * C * D * H * W
    in_idx = in_idx + tl.arange(0, C) * D * H * W

    # Load input values
    input_val = tl.load(input_ptr + in_idx, mask=in_idx < N * C * D * H * W, other=0.0)

    # Compute the max value
    max_val = tl.max(input_val)

    # Subtract the max value
    input_val -= max_val

    # Compute the exponential
    exp_val = tl.exp(input_val)

    # Compute the sum
    sum_val = tl.sum(exp_val)

    # Compute the softmax
    output_val = exp_val / sum_val

    # Store the output value
    out_idx = (out_d * H * W + out_h * W + out_w) + tl.arange(0, N) * C * D * H * W
    out_idx = out_idx + tl.arange(0, C) * D * H * W
    tl.store(output_ptr + out_idx, output_val)


@triton.jit
def subtract_kernel(
    input_ptr,  # Pointer to input tensor
    subtract_ptr,  # Pointer to subtract tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, D, H, W)
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the 3D index of the current thread
    pid = tl.program_id(0)
    # Compute the 3D offset for the current thread
    offset_d = pid // (BLOCK_SIZE * BLOCK_SIZE)
    offset_h = (pid // BLOCK_SIZE) % BLOCK_SIZE
    offset_w = pid % BLOCK_SIZE

    # Compute the input and output dimensions
    N, C, D, H, W = input_shape

    # Compute the output index for this thread
    out_d = offset_d
    out_h = offset_h
    out_w = offset_w

    # Compute the input indices for this thread
    in_idx = (out_d * H * W + out_h * W + out_w) + tl.arange(0, N) * C * D * H * W
    in_idx = in_idx + tl.arange(0, C) * D * H * W

    # Load input values
    input_val = tl.load(input_ptr + in_idx, mask=in_idx < N * C * D * H * W, other=0.0)

    # Load subtract values
    subtract_val = tl.load(subtract_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C, other=0.0)

    # Subtract the values
    output_val = input_val - subtract_val

    # Store the output value
    out_idx = (out_d * H * W + out_h * W + out_w) + tl.arange(0, N) * C * D * H * W
    out_idx = out_idx + tl.arange(0, C) * D * H * W
    tl.store(output_ptr + out_idx, output_val)


@triton.jit
def swish_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, D, H, W)
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the 3D index of the current thread
    pid = tl.program_id(0)
    # Compute the 3D offset for the current thread
    offset_d = pid // (BLOCK_SIZE * BLOCK_SIZE)
    offset_h = (pid // BLOCK_SIZE) % BLOCK_SIZE
    offset_w = pid % BLOCK_SIZE

    # Compute the input and output dimensions
    N, C, D, H, W = input_shape

    # Compute the output index for this thread
    out_d = offset_d
    out_h = offset_h
    out_w = offset_w

    # Compute the input indices for this thread
    in_idx = (out_d * H * W + out_h * W + out_w) + tl.arange(0, N) * C * D * H * W
    in_idx = in_idx + tl.arange(0, C) * D * H * W

    # Load input values
    input_val = tl.load(input_ptr + in_idx, mask=in_idx < N * C * D * H * W, other=0.0)

    # Compute the sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-input_val))

    # Compute the swish
    output_val = input_val * sigmoid_val

    # Store the output value
    out_idx = (out_d * H * W + out_h * W + out_w) + tl.arange(0, N) * C * D * H * W
    out_idx = out_idx + tl.arange(0, C) * D * H * W
    tl.store(output_ptr + out_idx, output_val)


def triton_conv_transpose(x, weight, in_shape, kernel_size, stride, padding, output_padding):
    out_channels = weight.shape[0]
    N, C, D, H, W = in_shape
    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    pD, pH, pW = padding
    oD, oH, oW = output_padding

    out_D = (D - 1) * sD + kD - 2 * pD + oD
    out_H = (H - 1) * sH + kH - 2 * pH + oH
    out_W = (W - 1) * sW + kW - 2 * pW + oW

    out_shape = (N, out_channels, out_D, out_H, out_W)

    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    grid = lambda meta: ((out_D * out_H * out_W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    conv_transpose_kernel[grid](x, weight, out, in_shape, kernel_size, stride, padding, output_padding, BLOCK_SIZE=128)
    return out


def triton_max_pool(x, in_shape, kernel_size, stride, padding):
    N, C, D, H, W = in_shape
    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    pD, pH, pW = padding

    out_D = (D - 1) * sD + kD - 2 * pD
    out_H = (H - 1) * sH + kH - 2 * pH
    out_W = (W - 1) * sW + kW - 2 * pW

    out_shape = (N, C, out_D, out_H, out_W)

    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    grid = lambda meta: ((out_D * out_H * out_W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    max_pool_kernel[grid](x, out, in_shape, kernel_size, stride, padding, BLOCK_SIZE=128)
    return out


def triton_softmax(x, in_shape):
    N, C, D, H, W = in_shape
    out_shape = (N, C, D, H, W)

    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    grid = lambda meta: ((D * H * W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    softmax_kernel[grid](x, out, in_shape, BLOCK_SIZE=128)
    return out


def triton_subtract(x, subtract, in_shape):
    N, C, D, H, W = in_shape
    out_shape = (N, C, D, H, W)

    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    grid = lambda meta: ((D * H * W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    subtract_kernel[grid](x, subtract, out, in_shape, BLOCK_SIZE=128)
    return out


def triton_swish(x, in_shape):
    N, C, D, H, W = in_shape
    out_shape = (N, C, D, H, W)

    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    grid = lambda meta: ((D * H * W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    swish_kernel[grid](x, out, in_shape, BLOCK_SIZE=128)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.subtract = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        x = triton_conv_transpose(x, self.weight, x.shape, self.kernel_size, self.stride, self.padding, self.output_padding)
        x = triton_max_pool(x, x.shape, self.pool_kernel_size, self.pool_stride, self.pool_padding)
        x = triton_softmax(x, x.shape)
        x = triton_subtract(x, self.subtract, x.shape)
        x = triton_swish(x, x.shape)
        x = torch.max(x, dim=1)[0]
        return x