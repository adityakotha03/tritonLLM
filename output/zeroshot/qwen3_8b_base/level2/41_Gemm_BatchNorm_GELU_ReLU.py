import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_batchnorm_gelu_relu_kernel(
    x_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    mean_ptr,  # Pointer to mean tensor
    rstd_ptr,  # Pointer to running std tensor
    gamma_ptr,  # Pointer to gamma tensor
    beta_ptr,  # Pointer to beta tensor
    out_ptr,  # Pointer to output tensor
    batch_size,  # Number of samples in batch
    in_features,  # Number of input features
    out_features,  # Number of output features
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    # Get the block index within the batch
    block_idx = tl.program_id(1)
    # Compute the offset for the current block
    offset = batch_idx * in_features + block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offset < batch_idx * in_features + out_features

    # Load input
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    # Load weight
    weight = tl.load(weight_ptr + block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    # Compute GEMM (x @ weight)
    x_weight = tl.dot(x, weight)
    # Add bias
    x_weight = tl.load(bias_ptr + block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0) + x_weight

    # BatchNorm
    mean = tl.load(mean_ptr + block_idx, mask=mask, other=0.0)
    rstd = tl.load(rstd_ptr + block_idx, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    beta = tl.load(beta_ptr + block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    x_weight = (x_weight - mean) * rstd * gamma + beta

    # GELU
    x_weight = 0.5 * x_weight * (1.0 + tl.math.erf(x_weight * 1.41421356 / 2.0))

    # ReLU
    x_weight = tl.maximum(x_weight, 0.0)

    # Store output
    tl.store(out_ptr + offset, x_weight, mask=mask)


def triton_gemm_batchnorm_gelu_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda and mean.is_cuda and rstd.is_cuda and gamma.is_cuda and beta.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    mean = mean.contiguous()
    rstd = rstd.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Determine the number of blocks needed
    BLOCK_SIZE = 128
    num_blocks = (in_features + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = lambda meta: (batch_size, num_blocks,)

    # Launch the Triton kernel
    gemm_batchnorm_gelu_relu_kernel[grid](x, weight, bias, mean, rstd, gamma, beta, out, batch_size, in_features, out_features, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.mean = nn.Parameter(torch.randn(out_features))
        self.rstd = nn.Parameter(torch.randn(out_features))
        self.gamma = nn.Parameter(torch.randn(out_features))
        self.beta = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return triton_gemm_batchnorm_gelu_relu(x, self.weight, self.bias, self.mean, self.rstd, self.gamma, self.beta)