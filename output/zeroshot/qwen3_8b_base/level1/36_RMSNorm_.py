import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,  # Pointer to input tensor
    scale_ptr,  # Pointer to scale tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in the tensor
    num_features,  # Number of features
    eps: tl.constexpr,  # Epsilon value for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute squared values
    x2 = x * x
    # Compute mean along feature dimension (sum across features and divide by num_features)
    sum_x2 = tl.sum(x2, axis=0)
    mean_x2 = sum_x2 / num_features
    # Add epsilon and take square root
    rms = tl.sqrt(mean_x2 + eps)
    # Compute reciprocal of RMS
    inv_rms = 1.0 / rms
    # Scale the input by inv_rms
    out = x * inv_rms
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_rmsnorm(x: torch.Tensor, eps: float = 1e-5):
    """
    Applies RMS normalization using a Triton kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    # Compute scale tensor
    scale = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)
    scale = scale.contiguous()
    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    num_features = x.shape[1]
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    rmsnorm_kernel[grid](x, scale, out, n_elements, num_features, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rmsnorm(x, self.eps)