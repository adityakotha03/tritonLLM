import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_relu_leaky_relu_gelu_sigmoid_bias_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    output_ptr,  # Pointer to output tensor
    stride_d, stride_h, stride_w,  # Strides for input
    kernel_d, kernel_h, kernel_w,  # Kernel size
    out_channels, in_channels,  # Output and input channels
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID (block index)
    pid = tl.program_id(0)
    # Compute the 3D index (d, h, w) for the output
    d = pid // (stride_h * stride_w)
    rem = pid % (stride_h * stride_w)
    h = rem // stride_w
    w = rem % stride_w

    # Compute the starting position in the input tensor
    input_offset = d * stride_d * in_channels * height * width + \
                   h * stride_h * in_channels * width + \
                   w * stride_w * in_channels

    # Initialize the output value
    output = tl.zeros((out_channels,), dtype=tl.float32)

    # Iterate over all output channels
    for c in range(out_channels):
        # Compute the weight offset for this channel
        weight_offset = c * in_channels * kernel_d * kernel_h * kernel_w
        # Initialize the accumulated value for this channel
        acc = tl.zeros((1,), dtype=tl.float32)

        # Iterate over all input channels
        for ic in range(in_channels):
            # Compute the input channel offset
            input_channel_offset = input_offset + ic
            # Iterate over the kernel in depth
            for kd in range(kernel_d):
                # Compute the input depth offset
                input_depth_offset = input_channel_offset + kd * height * width
                # Iterate over the kernel in height
                for kh in range(kernel_h):
                    # Compute the input height offset
                    input_height_offset = input_depth_offset + kh * width
                    # Iterate over the kernel in width
                    for kw in range(kernel_w):
                        # Compute the input offset for this kernel element
                        input_pos = input_height_offset + kw
                        # Load the input value
                        input_val = tl.load(input_ptr + input_pos, mask=input_pos < input_ptr.shape[0], other=0.0)
                        # Load the weight value
                        weight_val = tl.load(weight_ptr + weight_offset + ic * kernel_d * kernel_h * kernel_w + kd * kernel_h * kernel_w + kh * kernel_w + kw, mask=weight_offset + ic * kernel_d * kernel_h * kernel_w + kd * kernel_h * kernel_w + kh * kernel_w + kw < weight_ptr.shape[0], other=0.0)
                        # Multiply and accumulate
                        acc += input_val * weight_val

            # Apply ReLU
            acc = tl.maximum(acc, 0.0)
            # Apply Leaky ReLU
            acc = tl.where(acc > 0, acc, acc * 0.01)
            # Apply GELU
            acc = 0.5 * acc * (1 + tl.erf(acc / tl.sqrt(2.0)))
            # Apply Sigmoid
            acc = 1.0 / (1.0 + tl.exp(-acc))
            # Add bias
            bias_val = tl.load(bias_ptr + c, mask=c < bias_ptr.shape[0], other=0.0)
            acc += bias_val

        # Store the accumulated value for this output channel
        output[c] = acc

    # Store the output values
    output_offset = d * out_channels * height * width + h * out_channels * width + w * out_channels
    tl.store(output_ptr + output_offset, output, mask=output_offset < output_ptr.shape[0])


def triton_conv3d_relu_leaky_relu_gelu_sigmoid_bias(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    kernel_d: int,
    kernel_h: int,
    kernel_w: int,
    out_channels: int,
    in_channels: int,
):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Compute output shape
    output_shape = (
        input.shape[0],
        out_channels,
        (input.shape[2] - kernel_d + 1),
        (input.shape[3] - kernel_h + 1),
        (input.shape[4] - kernel_w + 1)
    )

    # Prepare output tensor
    output = torch.empty(output_shape, device=input.device, dtype=input.dtype)

    # Determine the number of blocks needed
    num_blocks = (input.shape[2] * input.shape[3] * input.shape[4]) // (BLOCK_SIZE) + 1

    # Launch the Triton kernel
    grid = lambda meta: (num_blocks,)
    conv3d_relu_leaky_relu_gelu_sigmoid_bias_kernel[grid](input, weight, bias, output, stride_d, stride_h, stride_w, kernel_d, kernel_h, kernel_w, out_channels, in_channels, BLOCK_SIZE=128)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias_shape = bias_shape

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Perform convolution
        x = triton_conv3d_relu_leaky_relu_gelu_sigmoid_bias(
            x,
            self.weight,
            self.bias,
            stride_d=1,
            stride_h=1,
            stride_w=1,
            kernel_d=self.kernel_size,
            kernel_h=self.kernel_size,
            kernel_w=self.kernel_size,
            out_channels=self.out_channels,
            in_channels=self.in_channels,
        )
        return x