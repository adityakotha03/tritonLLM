import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, stride, padding,
    x_stride0, x_stride1, x_stride2, x_stride3,
    w_stride0, w_stride1, w_stride2, w_stride3,
    out_stride0, out_stride1, out_stride2, out_stride3,
    BLOCK_SIZE: tl.constexpr,
):
    # Define block dimensions
    pid = tl.program_id(0)
    block_height = pid // (out_channels * (height // stride) * (width // stride))
    block_out_c = (pid // ((height // stride) * (width // stride))) % out_channels
    block_h = (pid // (width // stride)) % (height // stride)
    block_w = pid % (width // stride)

    # Calculate output position
    out_h = block_h * stride
    out_w = block_w * stride

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over input channels and kernel elements
    for ic in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Calculate input position
                h = out_h + kh - padding
                w = out_w + kw - padding
                # Load input value with mask
                h_mask = (h >= 0) & (h < height)
                w_mask = (w >= 0) & (w < width)
                mask = h_mask & w_mask
                x_val = tl.load(
                    x_ptr + ic * x_stride1 + h * x_stride2 + w * x_stride3,
                    mask=mask,
                    other=0.0
                )
                # Load weight value
                w_val = tl.load(
                    w_ptr + block_out_c * w_stride0 + ic * w_stride1 + kh * w_stride2 + kw * w_stride3
                )
                # Accumulate
                acc += x_val * w_val

    # Write output
    out_ptr = out_ptr + block_out_c * out_stride1 + block_h * out_stride2 + block_w * out_stride3
    tl.store(out_ptr, acc, mask=tl.arange(0, BLOCK_SIZE) < out_channels)


@triton.jit
def gelu_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = tl.constant(0.7978845608028654)
    x3 = x * x * x
    x = 0.5 * x * (1.0 + tl.tanh(sqrt_2_over_pi * (x + 0.044715 * x3)))
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def avg_pool2d_kernel(
    x_ptr, out_ptr,
    batch_size, in_channels, height, width,
    out_h, out_w,
    x_stride0, x_stride1, x_stride2, x_stride3,
    out_stride0, out_stride1, out_stride2, out_stride3,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch = pid // (in_channels * out_h * out_w)
    out_c = (pid // (out_h * out_w)) % in_channels
    out_h_idx = (pid // out_w) % out_h
    out_w_idx = pid % out_w

    # Calculate the pooling window
    h_start = out_h_idx * (height // out_h)
    h_end = h_start + (height // out_h)
    w_start = out_w_idx * (width // out_w)
    w_end = w_start + (width // out_w)

    # Accumulate sum
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            x_val = tl.load(
                x_ptr + batch * x_stride0 + out_c * x_stride1 + h * x_stride2 + w * x_stride3,
                mask=(h < height) & (w < width),
                other=0.0
            )
            acc += x_val

    # Average
    count = (h_end - h_start) * (w_end - w_start)
    acc = acc / count

    # Store output
    out_ptr = out_ptr + batch * out_stride0 + out_c * out_stride1 + out_h_idx * out_stride2 + out_w_idx * out_stride3
    tl.store(out_ptr, acc, mask=tl.arange(0, BLOCK_SIZE) < in_channels)


def triton_conv2d(x, weight, stride=1, padding=0):
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    out_h = (height + 2 * padding - kernel_size) // stride + 1
    out_w = (width + 2 * padding - kernel_size) // stride + 1

    out = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)

    # Calculate strides
    x_stride0, x_stride1, x_stride2, x_stride3 = x.stride()
    w_stride0, w_stride1, w_stride2, w_stride3 = weight.stride()
    out_stride0, out_stride1, out_stride2, out_stride3 = out.stride()

    # Configure kernel launch
    grid = lambda meta: (batch_size * out_channels * out_h * out_w,)
    BLOCK_SIZE = 32  # Tunable block size

    # Launch kernel
    conv2d_kernel[grid](
        x, weight, out,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding,
        x_stride0, x_stride1, x_stride2, x_stride3,
        w_stride0, w_stride1, w_stride2, w_stride3,
        out_stride0, out_stride1, out_stride2, out_stride3,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


def triton_gelu(x):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    n_elements = x.numel()

    # Create output
    out = torch.empty_like(x)

    # Configure kernel launch
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    BLOCK_SIZE = 256  # Tunable

    # Launch kernel
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_avg_pool2d(x, output_size):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_h, out_w = output_size

    # Calculate pooling window size
    kernel_h = height // out_h
    kernel_w = width // out_w

    out = torch.empty(batch_size, in_channels, out_h, out_w, device=x.device, dtype=x.dtype)

    # Calculate strides
    x_stride0, x_stride1, x_stride2, x_stride3 = x.stride()
    out_stride0, out_stride1, out_stride2, out_stride3 = out.stride()

    # Configure kernel launch
    grid = lambda meta: (batch_size * in_channels * out_h * out_w,)
    BLOCK_SIZE = 128  # Tunable

    # Launch kernel
    avg_pool2d_kernel[grid](
        x, out,
        batch_size, in_channels, height, width,
        out_h, out_w,
        x_stride0, x_stride1, x_stride2, x_stride3,
        out_stride0, out_stride1, out_stride2, out_stride3,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # Initialize the weight
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.stride = 1
        self.padding = (kernel_size - 1) // 2

    def forward(self, x):
        # Apply convolution with Triton kernel
        x = triton_conv2d(x, self.weight, stride=self.stride, padding=self.padding)

        # Apply GELU with Triton kernel
        x = triton_gelu(x)

        # Apply adaptive avg pooling with Triton kernel
        x = triton_avg_pool2d(x, output_size=(1, 1))

        # Squeeze to (batch_size, out_channels)
        return x.squeeze(-1).squeeze(-1)