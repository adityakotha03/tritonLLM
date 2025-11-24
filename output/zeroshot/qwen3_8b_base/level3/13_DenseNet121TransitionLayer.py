import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (batch, in_channels, height, width)
    output_shape,  # (batch, out_channels, height, width)
    kernel_size,  # Kernel size (assumed square)
    stride,  # Stride
    padding,  # Padding
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch, input channel, output channel, height, width
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)

    # Calculate the input channel indices
    in_channels = input_shape[1]
    out_channels = output_shape[1]
    height = input_shape[2]
    width = input_shape[3]
    out_height = output_shape[2]
    out_width = output_shape[3]

    # Compute the input height and width for this output position
    in_h_start = out_h_idx * stride - padding
    in_w_start = out_w_idx * stride - padding
    in_h_end = in_h_start + kernel_size
    in_w_end = in_w_start + kernel_size

    # Initialize the output value
    out_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over the output channel
    for in_channel_idx in range(in_channels):
        # Compute the weight offset
        weight_offset = out_channel_idx * in_channels + in_channel_idx
        weight_ptr = weight_ptr + weight_offset * kernel_size * kernel_size

        # Iterate over the kernel
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Compute the input offset
                in_h = in_h_start + kh
                in_w = in_w_start + kw
                if in_h < 0 or in_h >= height or in_w < 0 or in_w >= width:
                    continue
                input_offset = batch_idx * in_channels * height * width + in_channel_idx * height * width + in_h * width + in_w
                input_val = tl.load(input_ptr + input_offset, eviction_policy="evict_last")
                weight_val = tl.load(weight_ptr + kh * kernel_size + kw, eviction_policy="evict_last")
                out_val += input_val * weight_val

    # Store the result
    out_offset = batch_idx * out_channels * out_height * out_width + out_channel_idx * out_height * out_width + out_h_idx * out_width + out_w_idx
    tl.store(output_ptr + out_offset, out_val)


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, stride: int, padding: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()

    # Prepare output tensor
    batch, in_channels, in_height, in_width = input.shape
    out_channels = weight.shape[0]
    out_height = (in_height + 2 * padding - kernel_size) // stride + 1
    out_width = (in_width + 2 * padding - kernel_size) // stride + 1
    output = torch.empty((batch, out_channels, out_height, out_width), dtype=input.dtype, device=input.device)

    kernel_size = weight.shape[1]
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: (batch, out_channels, out_height, out_width)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, input.shape, output.shape, kernel_size, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def batch_norm_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    mean_ptr,  # Pointer to mean tensor
    var_ptr,  # Pointer to variance tensor
    gamma_ptr,  # Pointer to gamma tensor
    beta_ptr,  # Pointer to beta tensor
    eps: tl.constexpr,
    num_channels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch, input channel, height, width
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    h_idx = tl.program_id(2)
    w_idx = tl.program_id(3)

    # Compute the input offset
    input_offset = batch_idx * num_channels * height * width + channel_idx * height * width + h_idx * width + w_idx
    input_val = tl.load(input_ptr + input_offset, eviction_policy="evict_last")

    # Compute the mean and variance
    mean_val = tl.load(mean_ptr + channel_idx, eviction_policy="evict_last")
    var_val = tl.load(var_ptr + channel_idx, eviction_policy="evict_last")

    # Compute the gamma and beta
    gamma_val = tl.load(gamma_ptr + channel_idx, eviction_policy="evict_last")
    beta_val = tl.load(beta_ptr + channel_idx, eviction_policy="evict_last")

    # Normalize
    normalized_val = (input_val - mean_val) / tl.sqrt(var_val + eps)

    # Scale and shift
    output_val = normalized_val * gamma_val + beta_val

    # Store the result
    output_offset = batch_idx * num_channels * height * width + channel_idx * height * width + h_idx * width + w_idx
    tl.store(output_ptr + output_offset, output_val)


def triton_batch_norm(input: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps=1e-5):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and mean.is_cuda and var.is_cuda and gamma.is_cuda and beta.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    # Prepare output tensor
    output = torch.empty_like(input)

    num_channels = input.shape[1]
    height = input.shape[2]
    width = input.shape[3]
    batch = input.shape[0]

    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: (batch, num_channels, height, width)

    # Launch the Triton kernel
    batch_norm_kernel[grid](input, output, mean, var, gamma, beta, eps, num_channels, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(ModelNew, self).__init__()
        self.transition = nn.Sequential(
            # Custom BatchNorm2d
            self._create_custom_batchnorm(num_input_features),
            # Custom ReLU
            self._create_custom_relu(),
            # Custom Conv2d
            self._create_custom_conv2d(num_input_features, num_output_features),
            # Custom AvgPool2d
            self._create_custom_avgpool()
        )

    def _create_custom_batchnorm(self, num_channels):
        class CustomBatchNorm2d(nn.Module):
            def __init__(self, num_channels):
                super(CustomBatchNorm2d, self).__init__()
                self.register_buffer('mean', torch.tensor([0.0] * num_channels))
                self.register_buffer('var', torch.tensor([1.0] * num_channels))
                self.gamma = nn.Parameter(torch.ones(num_channels))
                self.beta = nn.Parameter(torch.zeros(num_channels))
                self.eps = 1e-5

            def forward(self, x):
                return triton_batch_norm(x, self.mean, self.var, self.gamma, self.beta, self.eps)
        return CustomBatchNorm2d(num_channels)

    def _create_custom_relu(self):
        class CustomReLU(nn.Module):
            def forward(self, x):
                return F.relu(x)
        return CustomReLU()

    def _create_custom_conv2d(self, in_channels, out_channels):
        class CustomConv2d(nn.Module):
            def __init__(self, in_channels, out_channels):
                super(CustomConv2d, self).__init__()
                self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
                self.stride = 1
                self.padding = 0

            def forward(self, x):
                return triton_conv2d(x, self.weight, self.stride, self.padding)
        return CustomConv2d(in_channels, out_channels)

    def _create_custom_avgpool(self):
        class CustomAvgPool2d(nn.Module):
            def __init__(self, kernel_size=2, stride=2):
                super(CustomAvgPool2d, self).__init__()
                self.kernel_size = kernel_size
                self.stride = stride

            def forward(self, x):
                return F.avg_pool2d(x, self.kernel_size, self.stride)
        return CustomAvgPool2d()