import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_bn_softmax_kernel(
    x_ptr,  # Input tensor pointer (batch_size, in_features)
    w_ptr,  # Weight tensor pointer (out_features, in_features)
    b_ptr,  # Bias pointer (out_features,)
    running_mean_ptr,  # Running mean pointer (out_features,)
    running_var_ptr,  # Running var pointer (out_features,)
    scale_ptr,  # Scale parameter pointer (1,)
    out_ptr,  # Output tensor pointer (batch_size, out_features)
    batch_size: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    TILE_SIZE_M: tl.constexpr,
    TILE_SIZE_N: tl.constexpr,
):
    # Block index for batch
    batch_id = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Thread indices
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Load input row (batch_id, in_features)
    x_ptrs = x_ptr + batch_id * in_features + offs_k
    x = tl.load(x_ptrs, mask=offs_k < in_features, other=0.0)

    # Load weight block (out_features, in_features)
    w_ptrs = w_ptr + offs_n[:, None] * in_features + offs_k[None, :]
    w = tl.load(w_ptrs, mask=(offs_n[:, None] < out_features) & (offs_k[None, :] < in_features), other=0.0)

    # Compute gemm: out = x @ w.T
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, in_features, BLOCK_SIZE_K):
        x = tl.load(x_ptrs + k, mask=(offs_k < in_features - k), other=0.0)
        w = tl.load(w_ptrs + k * out_features, mask=(offs_n[:, None] < out_features) & (offs_k[None, :] < in_features - k), other=0.0)
        acc += tl.dot(x, w)
    acc = acc.to(tl.float16)

    # Load bias and batch norm params (only one row per block)
    mean = tl.load(running_mean_ptr + offs_n, mask=offs_n < out_features, other=0.0)
    var = tl.load(running_var_ptr + offs_n, mask=offs_n < out_features, other=0.0)
    inv_std = tl.rsqrt(var + eps)

    # Apply batch norm: (acc - mean) / std
    acc = acc - mean[None, :]
    acc = acc * inv_std[None, :]

    # Apply scaling
    scale = tl.load(scale_ptr)
    acc = acc * scale

    # Online softmax: logsumexp trick for numerical stability
    # Find max value along N dimension
    max_val = tl.max(acc, axis=1)
    max_val = tl.broadcast_to(max_val[:, None], acc.shape)
    shifted = acc - max_val
    exp = tl.exp(shifted)
    sum_exp = tl.sum(exp, axis=1)
    sum_exp = tl.broadcast_to(sum_exp[:, None], acc.shape)
    out = exp / sum_exp

    # Store output
    out_ptrs = out_ptr + batch_id * out_features + offs_n
    tl.store(out_ptrs, out, mask=offs_n < out_features)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'TILE_SIZE_M': 128, 'TILE_SIZE_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'TILE_SIZE_M': 256, 'TILE_SIZE_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'TILE_SIZE_M': 64, 'TILE_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'TILE_SIZE_M': 128, 'TILE_SIZE_N': 64}, num_stages=4, num_warps=4),
    ],
    key=['batch_size', 'in_features', 'out_features'],
)
def launch_gemm_bn_softmax_kernel(x, w, b, running_mean, running_var, scale, out, batch_size, in_features, out_features, eps):
    # Grid definition: (batch_size, num_m_tiles, num_n_tiles)
    num_m_tiles = triton.cdiv(out_features, 128)
    num_n_tiles = triton.cdiv(out_features, 128)

    grid = (batch_size, num_m_tiles, num_n_tiles)

    # Launch kernel
    gemm_bn_softmax_kernel[grid](
        x, w, b, running_mean, running_var, scale, out,
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        eps=eps,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
        TILE_SIZE_M=128,
        TILE_SIZE_N=128,
    )


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.bfloat16).cuda())
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16).cuda())
        self.running_mean = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16).cuda(), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(out_features, dtype=torch.bfloat16).cuda(), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(scale_shape, dtype=torch.bfloat16).cuda())

        # Set batch norm momentum
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps

    def forward(self, x):
        # Ensure input is bfloat16
        x = x.to(torch.bfloat16)

        # Create output tensor
        out = torch.empty(x.shape[0], self.weight.shape[0], dtype=torch.bfloat16, device=x.device)

        # Launch Triton kernel
        launch_gemm_bn_softmax_kernel(
            x, self.weight, self.bias, self.running_mean, self.running_var, self.scale,
            out, x.shape[0], x.shape[1], self.weight.shape[0], self.bn_eps
        )

        return out

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            # In eval mode, set running stats to freeze
            self.running_mean.requires_grad = False
            self.running_var.requires_grad = False
        else:
            # In train mode, allow gradients
            self.running_mean.requires_grad = True
            self.running_var.requires_grad = True