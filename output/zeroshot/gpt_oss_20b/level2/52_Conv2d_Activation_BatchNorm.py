import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -----------------------------
# Fused activation kernel
# y = tanh(log(1+exp(x))) * x
# -----------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def activation_fused_kernel(
    x_ptr,  # Input pointer
    out_ptr,  # Output pointer
    N,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sp = tl.math.log1p(tl.math.exp(x))          # softplus
    t = tl.math.tanh(sp)                        # tanh
    y = t * x
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_fused_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Apply tanh(softplus(x)) * x using a fused Triton kernel.
    """
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    activation_fused_kernel[grid](x, out, N, BLOCK_SIZE=256)
    return out


# -----------------------------
# BatchNorm2d kernel
# y = weight * (x - mean) / sqrt(var + eps) + bias
# -----------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def batchnorm2d_kernel(
    x_ptr,          # Input pointer
    out_ptr,        # Output pointer
    mean_ptr,       # Running mean pointer
    var_ptr,        # Running var pointer
    weight_ptr,     # Weight (gamma) pointer
    bias_ptr,       # Bias (beta) pointer
    N,              # Total number of elements
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute channel index for each element
    # stride: channel stride is 1 because input is flattened as (N*H*W, C)
    # But easier: broadcast mean/var/weight/bias via offset // (H*W)
    # We assume contiguous NHWC? No, input is (B, C, H, W)
    # Flattened index order: (B*C*H*W). Channel stride = H*W
    # We'll compute channel index by division
    # Since Triton does not support division efficiently, we precompute channel offsets
    # Here we assume block size is small enough that we can compute channel via modulo
    # We use tl.div which is integer division
    stride_hw = tl.constexpr(tl.var("stride_hw"))  # placeholder

# The above approach is complex; instead we will compute in a simpler way:
# We will launch grid over (batch, channel, height*width) dimensions to keep channel as a separate axis.

# Reimplement batchnorm kernel with three-dimensional grid
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def batchnorm2d_kernel_3d(
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    M: tl.constexpr,  # Number of elements per channel (H*W)
    N: tl.constexpr,  # Number of channels
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for n in tl.arange(0, BLOCK_SIZE_N):
        if mask_n[n]:
            # Load per-channel parameters
            mean = tl.load(mean_ptr + offs_n[n])
            var = tl.load(var_ptr + offs_n[n])
            weight = tl.load(weight_ptr + offs_n[n])
            bias = tl.load(bias_ptr + offs_n[n])
            denom = tl.math.rsqrt(var + eps)
            for m in tl.arange(0, BLOCK_SIZE_M):
                if mask_m[m]:
                    idx = offs_n[n] * M + offs_m[m]
                    x_val = tl.load(x_ptr + idx)
                    y = weight * (x_val - mean) * denom + bias
                    tl.store(out_ptr + idx, y)

def triton_batchnorm2d(x: torch.Tensor,
                       running_mean: torch.Tensor,
                       running_var: torch.Tensor,
                       weight: torch.Tensor,
                       bias: torch.Tensor,
                       eps: float = 1e-5) -> torch.Tensor:
    """
    Apply BatchNorm2d using a Triton kernel.
    x shape: (B, C, H, W)
    running_mean, running_var, weight, bias shape: (C,)
    """
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    B, C, H, W = x.shape
    M = H * W
    N = C
    out = torch.empty_like(x)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    batchnorm2d_kernel_3d[grid](
        x,
        out,
        running_mean,
        running_var,
        weight,
        bias,
        M,
        N,
        eps,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
    )
    return out


# -----------------------------
# Optimized Model
# -----------------------------
class ModelNew(nn.Module):
    """
    Optimized model using Triton fused activation and batchnorm kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        # Convolution (PyTorch)
        x = self.conv(x)

        # Fused activation
        x = triton_fused_activation(x)

        # BatchNorm using Triton kernel
        x = triton_batchnorm2d(
            x,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.weight,
            self.bn.bias,
            eps=self.bn.eps,
        )
        return x