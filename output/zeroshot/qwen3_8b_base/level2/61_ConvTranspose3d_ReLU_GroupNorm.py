import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_relu_groupnorm_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    gamma_ptr,  # Pointer to gamma tensor for group norm
    beta_ptr,   # Pointer to beta tensor for group norm
    batch_size,  # Batch size
    in_channels,  # Input channels
    out_channels,  # Output channels
    kernel_size,  # Kernel size
    groups,  # Number of groups
    D, H, W,  # Input spatial dimensions
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Compute the 3D index for the current thread
    pid = tl.program_id(0)
    # Compute the batch index
    batch_idx = pid // (D * H * W)
    # Compute the spatial index (D, H, W) in the input
    spatial_idx = pid % (D * H * W)
    # Compute the spatial coordinates
    d = spatial_idx // (H * W)
    h = (spatial_idx % (H * W)) // W
    w = spatial_idx % W

    # Compute the output spatial dimensions
    out_d = d + kernel_size - 1
    out_h = h + kernel_size - 1
    out_w = w + kernel_size - 1

    # Compute the input channel index
    in_ch = tl.program_id(1) % in_channels
    # Compute the output channel index
    out_ch = tl.program_id(1) // in_channels

    # Compute the offset in the input tensor
    input_offset = batch_idx * in_channels * D * H * W + d * H * W + h * W + w
    input_offset += in_ch * D * H * W

    # Compute the offset in the weight tensor
    weight_offset = out_ch * in_channels // groups * kernel_size * kernel_size * kernel_size
    weight_offset += in_ch * kernel_size * kernel_size * kernel_size

    # Compute the offset in the output tensor
    output_offset = batch_idx * out_channels * D * H * W + out_d * H * W + out_h * W + out_w
    output_offset += out_ch * D * H * W

    # Compute the offset in the gamma and beta tensors
    gamma_offset = out_ch * groups
    beta_offset = gamma_offset

    # Load input value
    x = tl.load(input_ptr + input_offset, mask=tl.program_id(1) < in_channels, other=0.0)

    # Load weight value
    w = tl.load(weight_ptr + weight_offset, mask=tl.program_id(1) < in_channels, other=0.0)

    # Perform convolution
    conv = x * w

    # Apply ReLU
    conv = tl.maximum(conv, 0.0)

    # Compute mean and variance for group normalization
    mean = 0.0
    var = 0.0
    for i in range(GROUP_SIZE):
        mean += conv[i]
        var += conv[i] * conv[i]
    mean /= GROUP_SIZE
    var = var / GROUP_SIZE - mean * mean

    # Normalize
    inv_std = tl.rsqrt(var + 1e-5)
    norm = (conv - mean) * inv_std

    # Apply gamma and beta
    gamma = tl.load(gamma_ptr + gamma_offset, mask=tl.program_id(1) < groups, other=0.0)
    beta = tl.load(beta_ptr + beta_offset, mask=tl.program_id(1) < groups, other=0.0)
    norm = norm * gamma + beta

    # Store output
    tl.store(output_ptr + output_offset, norm)


def triton_conv_transpose3d_relu_groupnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    groups: int,
    D: int,
    H: int,
    W: int,
):
    assert input.is_cuda and weight.is_cuda and gamma.is_cuda and beta.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    # Output tensor shape
    out_shape = (batch_size, out_channels, D + kernel_size - 1, H + kernel_size - 1, W + kernel_size - 1)
    output = torch.empty(out_shape, dtype=input.dtype, device=input.device)

    # Compute the grid size
    num_elements = D * H * W * in_channels
    num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks, in_channels)

    # Launch the Triton kernel
    conv_transpose3d_relu_groupnorm_kernel[grid](
        input, weight, output, gamma, beta,
        batch_size, in_channels, out_channels, kernel_size, groups,
        D, H, W, BLOCK_SIZE=128, GROUP_SIZE=16
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.bias = bias

        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Perform transposed 3D convolution, ReLU, and group normalization using Triton
        x = triton_conv_transpose3d_relu_groupnorm(
            x, self.weight, self.gamma, self.beta,
            x.size(0), self.in_channels, self.out_channels,
            self.kernel_size, self.groups,
            x.size(2), x.size(3), x.size(4)
        )
        return x