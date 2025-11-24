import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C_in, H, W)
    weight_shape,  # (C_out, C_in, K, K)
    output_shape,  # (N, C_out, H_out, W_out)
    stride,  # Stride of the convolution
    padding,  # Padding added to both sides
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the 2D index in the output
    pid = tl.program_id(0)
    n, c_out, h_out, w_out = output_shape
    output_idx = pid
    # Compute the corresponding input position
    n_out = output_idx // (c_out * h_out * w_out)
    c_out_idx = (output_idx // (h_out * w_out)) % c_out
    h_idx = (output_idx // w_out) % h_out
    w_idx = output_idx % w_out

    # Compute the input position with padding
    h_in = h_idx + padding
    w_in = w_idx + padding
    # Compute the starting input channel
    c_in_start = 0
    # Compute the starting output channel
    c_out_start = 0
    # Compute the starting kernel position
    k_start = 0

    # Loop over the kernel
    for k in range(weight_shape[2]):
        for l in range(weight_shape[3]):
            # Compute the input offset
            input_offset = (n_out * input_shape[1] + c_in_start + k) * input_shape[2] * input_shape[3] + h_in + k * input_shape[2] + w_in + l
            # Load the input value
            input_val = tl.load(input_ptr + input_offset, mask=input_offset < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], other=0.0)
            # Load the weight value
            weight_val = tl.load(weight_ptr + (c_out_start * weight_shape[1] + c_in_start) * weight_shape[2] * weight_shape[3] + k * weight_shape[3] + l, mask=(c_out_start * weight_shape[1] + c_in_start) < weight_shape[1], other=0.0)
            # Compute the output value
            output_val = input_val * weight_val
            # Accumulate the output value
            output_ptr[output_idx] += output_val

    # Compute the output value
    output_val = output_ptr[output_idx]
    # Store the output value
    tl.store(output_ptr + output_idx, output_val, mask=output_idx < n * c_out * h_out * w_out)


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
    output = torch.empty_like(input)

    # Shape of input (N, C_in, H, W)
    input_shape = (input.shape[0], input.shape[1], input.shape[2], input.shape[3])
    # Shape of weight (C_out, C_in, K, K)
    weight_shape = (weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3])
    # Shape of output (N, C_out, H_out, W_out)
    output_shape = (input.shape[0], weight.shape[0], input.shape[2] - weight.shape[2] + 2 * padding, input.shape[3] - weight.shape[3] + 2 * padding)
    # Number of elements in the output
    n_elements = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]
    # Choose block size
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, input_shape, weight_shape, output_shape, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def batch_norm_kernel(
    input_ptr,  # Pointer to input tensor
    mean_ptr,  # Pointer to mean tensor
    var_ptr,  # Pointer to variance tensor
    gamma_ptr,  # Pointer to gamma tensor
    beta_ptr,  # Pointer to beta tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the 2D index in the output
    pid = tl.program_id(0)
    n, c, h, w = input_shape
    input_idx = pid
    # Compute the corresponding input position
    n_idx = input_idx // (c * h * w)
    c_idx = (input_idx // (h * w)) % c
    h_idx = (input_idx // w) % h
    w_idx = input_idx % w

    # Compute the mean and variance for the current channel
    mean = tl.load(mean_ptr + c_idx, mask=c_idx < c, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=c_idx < c, other=0.0)

    # Compute the normalized value
    x = tl.load(input_ptr + input_idx, mask=input_idx < n * c * h * w, other=0.0)
    x_norm = (x - mean) / tl.sqrt(var + eps)

    # Compute the scaled and shifted value
    gamma = tl.load(gamma_ptr + c_idx, mask=c_idx < c, other=0.0)
    beta = tl.load(beta_ptr + c_idx, mask=c_idx < c, other=0.0)
    output_val = x_norm * gamma + beta

    # Store the result
    tl.store(output_ptr + input_idx, output_val, mask=input_idx < n * c * h * w)


def triton_batch_norm(input: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5):
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

    # Shape of input (N, C, H, W)
    input_shape = (input.shape[0], input.shape[1], input.shape[2], input.shape[3])
    # Number of elements in the input
    n_elements = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
    # Choose block size
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    batch_norm_kernel[grid](input, mean, var, gamma, beta, output, input_shape, eps, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Custom Conv2D kernel
        weight = torch.nn.Parameter(torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)).cuda()
        x = triton_conv2d(x, weight, stride=1, padding=(self.kernel_size - 1) // 2)
        # Custom BatchNorm kernel
        mean = torch.nn.Parameter(torch.randn(self.out_channels)).cuda()
        var = torch.nn.Parameter(torch.randn(self.out_channels)).cuda()
        gamma = torch.nn.Parameter(torch.randn(self.out_channels)).cuda()
        beta = torch.nn.Parameter(torch.randn(self.out_channels)).cuda()
        x = triton_batch_norm(x, mean, var, gamma, beta, eps=1e-5)
        # Scaling
        x = x * self.scaling_factor
        return x