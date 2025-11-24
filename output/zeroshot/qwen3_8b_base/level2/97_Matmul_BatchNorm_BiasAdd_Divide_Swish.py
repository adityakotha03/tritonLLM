import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_bn_bias_div_swish_kernel(
    x_ptr,  # pointer to input tensor
    weight_ptr,  # pointer to weight tensor
    bias_ptr,  # pointer to bias tensor
    bn_mean_ptr,  # pointer to batch norm mean
    bn_var_ptr,  # pointer to batch norm variance
    bn_eps,  # epsilon for batch norm
    out_ptr,  # pointer to output tensor
    batch_size,  # batch size
    in_features,  # input features
    out_features,  # output features
    divide_value,  # division value
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block start index
    block_start = pid * BLOCK_SIZE
    # Compute the offset for each thread in the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < batch_size * in_features

    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Load weight data
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    # Compute matmul
    x = tl.dot(x, weight)

    # Batch norm
    mean = tl.load(bn_mean_ptr, mask=mask, other=0.0)
    var = tl.load(bn_var_ptr, mask=mask, other=0.0)
    x = (x - mean) * tl.rsqrt(var + bn_eps)

    # Bias addition
    bias = tl.load(bias_ptr, mask=mask, other=0.0)
    x = x + bias

    # Division
    x = x / divide_value

    # Swish activation
    x = x * tl.sigmoid(x)

    # Store output
    tl.store(out_ptr + offsets, x, mask=mask)


def triton_matmul_bn_bias_div_swish(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, bn_mean: torch.Tensor, bn_var: torch.Tensor, bn_eps, bn_momentum, divide_value):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda and bn_mean.is_cuda and bn_var.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    bn_mean = bn_mean.contiguous()
    bn_var = bn_var.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    matmul_bn_bn_bias_div_swish_kernel[grid](x, weight, bias, bn_mean, bn_var, bn_eps, out, x.size(0), x.size(1), weight.size(0), divide_value, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a matrix multiplication, batch normalization, bias addition, division, and Swish activation using Triton kernels.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.bn_mean = nn.Parameter(torch.zeros(out_features))
        self.bn_var = nn.Parameter(torch.ones(out_features))
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.divide_value = divide_value

    def forward(self, x):
        # Use Triton kernel for optimized matmul, bn, bias, division, and swish
        x = triton_matmul_bn_bias_div_swish(x, self.weight, self.bias, self.bn_mean, self.bn_var, self.bn_eps, self.bn_momentum, self.divide_value)
        return x