import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_stride0, input_stride1, input_stride2, input_stride3, input_stride4,
    weight_stride0, weight_stride1, weight_stride2, weight_stride3, weight_stride4,
    output_stride0, output_stride1, output_stride2, output_stride3, output_stride4,
    batch_size, in_channels, out_channels, depth, height, width,
    kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w,
    output_pad_d, output_pad_h, output_pad_w,
    BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Thread block indices
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    # Output spatial dimensions
    out_depth = (depth - 1) * stride_d + kernel_d - 2 * pad_d + output_pad_d
    out_height = (height - 1) * stride_h + kernel_h - 2 * pad_h + output_pad_h
    out_width = (width - 1) * stride_w + kernel_w - 2 * pad_w + output_pad_w

    # Block offsets
    block_d = pid_d * BLOCK_SIZE_D
    block_h = pid_h * BLOCK_SIZE_H
    block_w = pid_w * BLOCK_SIZE_W

    # Load input and weight
    input_offset = pid_b * input_stride0 + pid_c * input_stride1 + \
                   tl.arange(0, BLOCK_SIZE_D)[:, None, None] * input_stride2 + \
                   tl.arange(0, BLOCK_SIZE_H)[None, :, None] * input_stride3 + \
                   tl.arange(0, BLOCK_SIZE_W)[None, None, :] * input_stride4
    input_mask = (tl.arange(0, BLOCK_SIZE_D)[:, None, None] < depth) & \
                 (tl.arange(0, BLOCK_SIZE_H)[None, :, None] < height) & \
                 (tl.arange(0, BLOCK_SIZE_W)[None, None, :] < width)
    input_vals = tl.load(input_ptr + input_offset, mask=input_mask, other=0.0)

    weight_offset = pid_c * weight_stride0 + pid_b * weight_stride1 + \
                    tl.arange(0, kernel_d)[:, None, None] * weight_stride2 + \
                    tl.arange(0, kernel_h)[None, :, None] * weight_stride3 + \
                    tl.arange(0, kernel_w)[None, None, :] * weight_stride4
    weight_vals = tl.load(weight_ptr + weight_offset, mask=(tl.arange(0, kernel_d)[:, None, None] < kernel_d) &
                                                    (tl.arange(0, kernel_h)[None, :, None] < kernel_h) &
                                                    (tl.arange(0, kernel_w)[None, None, :] < kernel_w), other=0.0)

    # Compute output
    output = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    for kd in range(kernel_d):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Compute spatial indices in output
                out_d = block_d + kd * stride_d
                out_h = block_h + kh * stride_h
                out_w = block_w + kw * stride_w

                # Compute input indices
                in_d = out_d - kd + pad_d
                in_h = out_h - kh + pad_h
                in_w = out_w - kw + pad_w

                # Check bounds
                valid_in = (in_d >= 0) & (in_d < depth) & (in_h >= 0) & (in_h < height) & (in_w >= 0) & (in_w < width)
                valid_out = (out_d < out_depth) & (out_h < out_height) & (out_w < out_width)

                # Mask input
                input_idx = in_d * input_stride2 + in_h * input_stride3 + in_w * input_stride4
                output_idx = out_d * output_stride2 + out_h * output_stride3 + out_w * output_stride4

                # Add to output
                mask = valid_in & valid_out
                tl.atomic_add(output_ptr + pid_b * output_stride0 + pid_c * output_stride1 + output_idx, 
                              input_vals * weight_vals[kd, kh, kw] * tl.cast(mask, tl.float32))

    # Store output
    output_offset = pid_b * output_stride0 + pid_c * output_stride1 + \
                    tl.arange(0, BLOCK_SIZE_D)[:, None, None] * output_stride2 + \
                    tl.arange(0, BLOCK_SIZE_H)[None, :, None] * output_stride3 + \
                    tl.arange(0, BLOCK_SIZE_W)[None, None, :] * output_stride4
    output_mask = (tl.arange(0, BLOCK_SIZE_D)[:, None, None] < out_depth) & \
                  (tl.arange(0, BLOCK_SIZE_H)[None, :, None] < out_height) & \
                  (tl.arange(0, BLOCK_SIZE_W)[None, None, :] < out_width)
    tl.store(output_ptr + output_offset, output, mask=output_mask)


@triton.jit
def sum_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    sum_weight,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x + sum_weight
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_elements
    var = tl.sum((x - mean) ** 2, axis=0) / n_elements
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * inv_std
    weight = tl.load(weight_ptr, mask=mask, other=1.0)
    bias = tl.load(bias_ptr, mask=mask, other=0.0)
    out = x_norm * weight + bias
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def avg_pool_kernel(
    x_ptr,
    out_ptr,
    x_stride0, x_stride1, x_stride2, x_stride3, x_stride4,
    out_stride0, out_stride1, out_stride2, out_stride3, out_stride4,
    batch_size, channels, in_depth, in_height, in_width,
    pool_d, pool_h, pool_w,
    BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    block_d = pid_d * BLOCK_SIZE_D
    block_h = pid_h * BLOCK_SIZE_H
    block_w = pid_w * BLOCK_SIZE_W

    # Pooling indices
    out_d = block_d + tl.arange(0, BLOCK_SIZE_D)[:, None, None]
    out_h = block_h + tl.arange(0, BLOCK_SIZE_H)[None, :, None]
    out_w = block_w + tl.arange(0, BLOCK_SIZE_W)[None, None, :]

    # Input indices
    in_d = out_d * pool_d
    in_h = out_h * pool_h
    in_w = out_w * pool_w

    # Bounds
    valid_out = (out_d < (in_depth + pool_d - 1) // pool_d) & \
                (out_h < (in_height + pool_h - 1) // pool_h) & \
                (out_w < (in_width + pool_w - 1) // pool_w)
    valid_in_d = (in_d < in_depth) & (in_d + pool_d <= in_depth)
    valid_in_h = (in_h < in_height) & (in_h + pool_h <= in_height)
    valid_in_w = (in_w < in_width) & (in_w + pool_w <= in_width)

    mask = valid_out & valid_in_d & valid_in_h & valid_in_w
    x_offsets = pid_b * x_stride0 + pid_c * x_stride1 + \
                (in_d + tl.arange(0, pool_d)[:, None, None]) * x_stride2 + \
                (in_h + tl.arange(0, pool_h)[None, :, None]) * x_stride3 + \
                (in_w + tl.arange(0, pool_w)[None, None, :]) * x_stride4
    x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    pool_size = pool_d * pool_h * pool_w
    pooled_vals = tl.sum(x_vals, axis=(0, 1, 2)) / pool_size

    out_offsets = pid_b * out_stride0 + pid_c * out_stride1 + \
                  out_d * out_stride2 + out_h * out_stride3 + out_w * out_stride4
    tl.store(out_ptr + out_offsets, pooled_vals, mask=valid_out)


@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Approximate GELU
    pi = 3.141592653589793
    cdf = 0.5 * (1.0 + tl.tanh((x * (2.0 / pi) ** 0.5) * (1.0 + 0.044715 * x ** 2)))
    out = x * cdf
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv_transpose(x, weight, bias=None, stride=(1,1,1), padding=(0,0,0), output_padding=(0,0,0), groups=1):
    if bias is not None:
        raise NotImplementedError("Bias not supported in this Triton version")
    if groups != 1:
        raise NotImplementedError("Grouped convolutions not supported")

    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, kernel_d, kernel_h, kernel_w = weight.shape
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    output_pad_d, output_pad_h, output_pad_w = output_padding

    # Output shape
    out_depth = (depth - 1) * stride_d + kernel_d - 2 * pad_d + output_pad_d
    out_height = (height - 1) * stride_h + kernel_h - 2 * pad_h + output_pad_h
    out_width = (width - 1) * stride_w + kernel_w - 2 * pad_w + output_pad_w

    # Allocate output
    output = torch.empty(batch_size, out_channels, out_depth, out_height, out_width, device=x.device, dtype=x.dtype)

    # Strides
    input_stride0, input_stride1, input_stride2, input_stride3, input_stride4 = x.stride()
    weight_stride0, weight_stride1, weight_stride2, weight_stride3, weight_stride4 = weight.stride()
    output_stride0, output_stride1, output_stride2, output_stride3, output_stride4 = output.stride()

    # Grid
    BLOCK_SIZE_D = 16
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_K = 16

    grid = (batch_size, out_channels, 
            (out_depth + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D,
            (out_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
            (out_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)

    conv_transpose_kernel[grid](
        x, weight, output,
        input_stride0, input_stride1, input_stride2, input_stride3, input_stride4,
        weight_stride0, weight_stride1, weight_stride2, weight_stride3, weight_stride4,
        output_stride0, output_stride1, output_stride2, output_stride3, output_stride4,
        batch_size, in_channels, out_channels, depth, height, width,
        kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w,
        output_pad_d, output_pad_h, output_pad_w,
        BLOCK_SIZE_D=BLOCK_SIZE_D, BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    return output


def triton_sum(x, sum_weight, output_shape):
    out = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    sum_kernel[grid](x, out, n_elements, sum_weight, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_layer_norm(x, weight, bias, eps=1e-5):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    layer_norm_kernel[grid](x, weight, bias, out, n_elements, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_avg_pool(x, pool_kernel_size):
    batch_size, channels, depth, height, width = x.shape
    pool_d, pool_h, pool_w = pool_kernel_size
    out_depth = (depth + pool_d - 1) // pool_d
    out_height = (height + pool_h - 1) // pool_h
    out_width = (width + pool_w - 1) // pool_w

    out = torch.empty(batch_size, channels, out_depth, out_height, out_width, device=x.device, dtype=x.dtype)

    input_stride0, input_stride1, input_stride2, input_stride3, input_stride4 = x.stride()
    output_stride0, output_stride1, output_stride2, output_stride3, output_stride4 = out.stride()

    BLOCK_SIZE_D = 16
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

    grid = (batch_size, channels,
            (out_depth + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D,
            (out_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
            (out_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)

    avg_pool_kernel[grid](
        x, out,
        input_stride0, input_stride1, input_stride2, input_stride3, input_stride4,
        output_stride0, output_stride1, output_stride2, output_stride3, output_stride4,
        batch_size, channels, depth, height, width,
        pool_d, pool_h, pool_w,
        BLOCK_SIZE_D=BLOCK_SIZE_D, BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W
    )

    return out


def triton_gelu(x):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        # ConvTranspose3D
        x = triton_conv_transpose(x, self.conv_transpose.weight, stride=tuple(self.conv_transpose.stride), padding=tuple(self.conv_transpose.padding), output_padding=tuple(self.conv_transpose.output_padding))
        
        # Sum
        x = triton_sum(x, self.sum_weight.item(), x.shape)
        
        # LayerNorm
        x = triton_layer_norm(x, self.norm.weight, self.norm.bias, eps=self.norm.eps)
        
        # AvgPool3d
        x = triton_avg_pool(x, self.avg_pool.kernel_size)
        
        # GELU
        x = triton_gelu(x)
        
        return x