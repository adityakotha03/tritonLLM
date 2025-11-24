import torch
import torch.nn as nn
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: fused batch‑norm, bias add, division and Swish
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_bn_swish_kernel(
    y_ptr,                # input from matmul
    out_ptr,              # output
    bn_weight_ptr,        # gamma
    bn_bias_ptr,          # beta
    running_mean_ptr,     # running mean
    running_var_ptr,      # running var
    bias_ptr,             # additional bias
    divide_value,         # scalar divide
    eps,                  # epsilon
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load
    y   = tl.load(y_ptr + offset, mask=mask, other=0.0)
    gamma = tl.load(bn_weight_ptr + offset, mask=mask, other=0.0)
    beta  = tl.load(bn_bias_ptr + offset, mask=mask, other=0.0)
    mean  = tl.load(running_mean_ptr + offset, mask=mask, other=0.0)
    var   = tl.load(running_var_ptr + offset, mask=mask, other=0.0)
    bias_add = tl.load(bias_ptr + offset, mask=mask, other=0.0)

    # Batch‑norm
    denom = tl.sqrt(var + eps)
    norm = (y - mean) / denom
    y = norm * gamma + beta

    # Bias, division and Swish
    y = y + bias_add
    y = y / divide_value
    sigmoid = tl.math.sigmoid(y)
    y = y * sigmoid

    # Store
    tl.store(out_ptr + offset, y, mask=mask)


def fused_bn_swish(y, bn_weight, bn_bias, running_mean, running_var, bias, divide_value, eps):
    """
    Wrapper that launches the fused Triton kernel.
    """
    assert y.is_cuda
    n_elements = y.numel()
    BLOCK_SIZE = 1024  # will be autotuned
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    out = torch.empty_like(y)
    fused_bn_swish_kernel[grid](
        y,
        out,
        bn_weight,
        bn_bias,
        running_mean,
        running_var,
        bias,
        divide_value,
        eps,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ----------------------------------------------------------------------
# Optimised model
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, batch normalization, bias addition,
    division, and Swish activation, with the latter operations fused into a single
    Triton kernel for maximum throughput.
    """
    def __init__(self, in_features, out_features,
                 bn_eps=1e-5, bn_momentum=0.1,
                 bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        # Custom bias added after batchnorm
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value

    def forward(self, x):
        # Matrix multiplication
        y = self.matmul(x)

        # Prepare tensors for the fused kernel
        # bn.weight == gamma, bn.bias == beta
        bn_weight = self.bn.weight
        bn_bias = self.bn.bias
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var

        # Fused batch‑norm, bias add, division and Swish
        y = fused_bn_swish(
            y,
            bn_weight,
            bn_bias,
            running_mean,
            running_var,
            self.bias,
            self.divide_value,
            self.bn.eps,
        )
        return y