import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_mish_tanh_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    output_ptr,  # Pointer to output tensor
    stride_d, stride_h, stride_w,  # Strides for input
    padding_d, padding_h, padding_w,  # Padding for input
    kernel_d, kernel_h, kernel_w,  # Kernel size
    out_channels, in_channels,  # Output and input channels
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a single output element
    # Compute the output position (out_d, out_h, out_w, out_c)
    out_d = tl.program_id(0)
    out_h = tl.program_id(1)
    out_w = tl.program_id(2)
    out_c = tl.program_id(3)

    # Compute the input position (in_d, in_h, in_w, in_c)
    in_d = out_d + padding_d
    in_h = out_h + padding_h
    in_w = out_w + padding_w
    in_c = out_c

    # Initialize output
    out = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over the input channels
    for in_c_idx in range(in_channels):
        # Load weight for this input channel
        weight = tl.load(weight_ptr + (in_c_idx * out_channels * kernel_d * kernel_h * kernel_w) + (out_c * kernel_d * kernel_h * kernel_w) + (tl.arange(0, kernel_d) * kernel_h * kernel_w) + (tl.arange(0, kernel_h) * kernel_w) + tl.arange(0, kernel_w), mask=(tl.arange(0, kernel_d) < kernel_d) & (tl.arange(0, kernel_h) < kernel_h) & (tl.arange(0, kernel_w) < kernel_w), other=0.0)

        # Compute the input indices for this weight
        in_d_idx = tl.arange(0, kernel_d)
        in_h_idx = tl.arange(0, kernel_h)
        in_w_idx = tl.arange(0, kernel_w)

        # Compute the input positions for this weight
        in_d = out_d + in_d_idx
        in_h = out_h + in_h_idx
        in_w = out_w + in_w_idx

        # Load input values
        input_val = tl.load(input_ptr + (in_channels * (in_d * stride_h * stride_w + in_h * stride_w + in_w) + in_c_idx), mask=(in_d < input.shape[2]) & (in_h < input.shape[3]) & (in_w < input.shape[4]), other=0.0)

        # Multiply with weight and accumulate
        out += input_val * weight

    # Add bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + out_c, other=0.0)
        out += bias

    # Apply Mish activation
    out = out * (tl.exp(out) / (tl.exp(out) + 1.0)) * (1.0 + 0.5 * out)

    # Apply Tanh activation
    out = (2.0 / (1.0 + tl.exp(-2.0 * out)) - 1.0)

    # Store the result
    tl.store(output_ptr + (out_c * output.shape[2] * output.shape[3] * output.shape[4] + out_d * output.shape[3] * output.shape[4] + out_h * output.shape[4] + out_w), out, mask=(out_d < output.shape[2]) & (out_h < output.shape[3]) & (out_w < output.shape[4]))

def triton_conv3d_mish_tanh(input, weight, bias, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w, kernel_d, kernel_h, kernel_w, out_channels, in_channels):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda), "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Prepare output tensor
    output_shape = (input.shape[0], out_channels, (input.shape[2] + 2 * padding_d - kernel_d) // stride_d + 1, (input.shape[3] + 2 * padding_h - kernel_h) // stride_h + 1, (input.shape[4] + 2 * padding_w - kernel_w) // stride_w + 1)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Number of elements in the output
    n_elements = output.numel()
    BLOCK_SIZE = 1

    # Determine the number of blocks needed
    grid = lambda meta: (output.shape[2], output.shape[3], output.shape[4], output.shape[1],)

    # Launch the Triton kernel
    conv3d_mish_tanh_kernel[grid](input, weight, bias, output, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w, kernel_d, kernel_h, kernel_w, out_channels, in_channels, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        # Compute output dimensions
        D_out = (x.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        H_out = (x.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (x.shape[4] + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Perform convolution, Mish, and Tanh using Triton kernel
        x = triton_conv3d_mish_tanh(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.padding,
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
            self.out_channels,
            self.in_channels
        )
        return x