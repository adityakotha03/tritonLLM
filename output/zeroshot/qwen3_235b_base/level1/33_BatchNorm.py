import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batch_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
    out_ptr,
    num_features,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Each feature channel has its own mean, var, weight, bias
    # We assume input shape: (batch, features, H, W), so feature index is (offsets // (H * W)) % num_features
    # But we don't have H and W here — instead, we know total size and num_features
    # So feature_id = (offsets // spatial_elements) % num_features
    # However, we can compute spatial_elements = n_elements // (batch_size * num_features)
    # But we don't have batch_size. Instead, we use modulo arithmetic: feature_id = (offsets // (n_elements // num_features)) % num_features
    spatial_elements = n_elements // num_features
    feature_ids = (offsets // spatial_elements) % num_features

    # Gather per-channel running mean and var
    mean = tl.load(running_mean_ptr + feature_ids, mask=mask, other=0.0)
    var = tl.load(running_var_ptr + feature_ids, mask=mask, other=0.0)

    # Normalize
    x_hat = (x - mean) / tl.sqrt(var + eps)

    # Load weight and bias if provided
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + feature_ids, mask=mask, other=1.0)
        x_hat = x_hat * weight
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + feature_ids, mask=mask, other=0.0)
        x_hat = x_hat + bias

    # Store output
    tl.store(out_ptr + offsets, x_hat, mask=mask)


def triton_batch_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()

    # Heuristic for block size
    BLOCK_SIZE = 1024
    if n_elements < 2048:
        BLOCK_SIZE = 64
    elif n_elements < 32768:
        BLOCK_SIZE = 256
    elif n_elements < 1048576:
        BLOCK_SIZE = 512

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    batch_norm_kernel[grid](
        x,
        weight,
        bias,
        running_mean,
        running_var,
        out,
        num_features=running_mean.numel(),
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom Triton kernel for Batch Normalization.
    """
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton-based batch norm during forward pass
        if x.is_cuda:
            return triton_batch_norm(
                x,
                self.bn.weight,
                self.bn.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps,
            )
        else:
            # Fall back to PyTorch implementation if not on CUDA
            return self.bn(x)