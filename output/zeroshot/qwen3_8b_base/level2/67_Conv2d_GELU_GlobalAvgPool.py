import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_gelu_avg_pool_kernel(
    x_ptr,  # Pointer to input tensor
    w_ptr,  # Pointer to weight tensor
    b_ptr,  # Pointer to bias tensor
    out_ptr,  # Pointer to output tensor
    stride_h, stride_w,  # Strides for input
    kernel_h, kernel_w,  # Kernel size
    out_h, out_w,  # Output dimensions
    num_channels,  # Number of input channels
    num_output_channels,  # Number of output channels
    BLOCK_H: tl.constexpr,  # Block size for height
    BLOCK_W: tl.constexpr,  # Block size for width
    BLOCK_C: tl.constexpr,  # Block size for channels
):
    # Get the thread index
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)

    # Compute the position in the output
    oh = pid_h * BLOCK_H
    ow = pid_w * BLOCK_W
    oc = pid_c * BLOCK_C

    # Compute the offset in the input
    offset_h = oh * stride_h
    offset_w = ow * stride_w

    # Initialize accumulator
    acc = tl.zeros((BLOCK_C, BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Iterate over the kernel
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Compute input offset
            input_offset = offset_h + kh * stride_h + offset_w + kw * stride_w
            # Load weights
            weight = tl.load(w_ptr + (kh * kernel_w + kw) * num_output_channels * num_channels + oc * num_channels + tl.arange(0, BLOCK_C), mask=tl.arange(0, BLOCK_C) < num_channels, other=0.0)
            # Load input
            input_data = tl.load(x_ptr + input_offset + tl.arange(0, BLOCK_C), mask=tl.arange(0, BLOCK_C) < num_channels, other=0.0)
            # Multiply and accumulate
            acc += input_data * weight

    # Add bias
    bias = tl.load(b_ptr + oc, other=0.0)
    acc += bias

    # Apply GELU
    acc = tl.where(acc > 0, acc, acc * (1.702 * acc + 0.6766 * acc * acc))

    # Store output
    out_offset = oc * out_h * out_w + oh * out_w + ow
    tl.store(out_ptr + out_offset + tl.arange(0, BLOCK_C), acc, mask=tl.arange(0, BLOCK_C) < num_output_channels)


def triton_conv_gelu_avg_pool(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, stride_h, stride_w, kernel_h, kernel_w):
    """
    This function wraps the Triton kernel call for convolution + GELU + adaptive average pooling.
    """
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()

    # Compute output dimensions
    out_h = (x.shape[2] - kernel_h) // stride_h + 1
    out_w = (x.shape[3] - kernel_w) // stride_w + 1

    # Prepare output tensor
    out = torch.empty((x.shape[0], w.shape[0], out_h, out_w), dtype=x.dtype, device=x.device)

    # Determine block sizes
    BLOCK_H = 8
    BLOCK_W = 8
    BLOCK_C = 16

    # Determine grid size
    grid = lambda meta: (out_h // meta["BLOCK_H"], out_w // meta["BLOCK_W"], w.shape[0] // meta["BLOCK_C"])

    # Launch the Triton kernel
    conv_gelu_avg_pool_kernel[grid](x, w, b, out, stride_h, stride_w, kernel_h, kernel_w, out_h, out_w, x.shape[1], w.shape[0], BLOCK_H, BLOCK_W, BLOCK_C)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        # Perform convolution + GELU + adaptive average pooling
        x = triton_conv_gelu_avg_pool(x, self.weight, self.bias, 1, 1, self.kernel_size, self.kernel_size)
        return x