import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ------------------------------------------------------------------
# Triton kernels
# ------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_K": 128}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 512, "BLOCK_SIZE_K": 128}, num_warps=16),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_gemm_bn_scale_softmax(
    # Pointers
    A_ptr,  # (M, K)
    B_ptr,  # (K, N)
    bias_ptr,  # (N,)
    gamma_ptr,  # (N,)
    beta_ptr,  # (N,)
    running_mean_ptr,  # (N,)
    running_var_ptr,  # (N,)
    scale_ptr,  # (1,)
    # Sizes
    M, N, K,
    # BatchNorm params
    eps,
    momentum,
    # Scale param
    scale,
    # Output
    out_ptr,
    # Triton params
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Each program instance processes a block of rows of the output matrix
    (size BLOCK_SIZE_M). The kernel computes
        Y = (A @ B + bias)
        Y = gamma * (Y - running_mean) / sqrt(running_var + eps) + beta
        Y = scale * Y
        Y = softmax(Y, dim=1)
    """
    pid = tl.program_id(0)

    # Row offset for this block
    row_offset = pid * BLOCK_SIZE_M
    # Loop over rows in the block
    for i in range(row_offset, row_offset + BLOCK_SIZE_M):
        # Guard against out-of-bounds rows
        if i >= M:
            break

        # Accumulate dot product for this row
        acc = tl.zeros([N], dtype=tl.float32)

        # Iterate over K in tiles
        for k in range(0, K, BLOCK_SIZE_K):
            # Load tiles of A and B
            a = tl.load(A_ptr + i * K + k + tl.arange(0, BLOCK_SIZE_K), mask=(k + tl.arange(0, BLOCK_SIZE_K) < K), other=0.0)  # (K_tile,)
            b = tl.load(B_ptr + (k + tl.arange(0, BLOCK_SIZE_K)) * N,  # (K_tile, N)
                        mask=(k + tl.arange(0, BLOCK_SIZE_K) < K), other=0.0)  # (K_tile, N)
            # Broadcast a to shape (K_tile, N)
            a_exp = tl.broadcast_to(a[:, None], (BLOCK_SIZE_K, N))
            acc += tl.sum(a_exp * b, axis=0)  # (N,)

        # Add bias
        bias = tl.load(bias_ptr + tl.arange(0, N))
        acc = acc + bias

        # BatchNorm (affine)
        gamma = tl.load(gamma_ptr + tl.arange(0, N))
        beta = tl.load(beta_ptr + tl.arange(0, N))
        mean = tl.load(running_mean_ptr + tl.arange(0, N))
        var = tl.load(running_var_ptr + tl.arange(0, N))
        acc = gamma * (acc - mean) * tl.rsqrt(var + eps) + beta

        # Scale
        acc = acc * scale

        # Softmax over N
        # First find max for numerical stability
        max_val = tl.max(acc, axis=0)
        acc = tl.exp(acc - max_val)
        sum_exp = tl.sum(acc, axis=0)
        acc = acc / sum_exp

        # Store result
        tl.store(out_ptr + i * N + tl.arange(0, N), acc, mask=tl.arange(0, N) < N)


# ------------------------------------------------------------------
# Wrapper function
# ------------------------------------------------------------------

def matmul_gemm_bn_scale_softmax(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-5,
    momentum: float = 0.1,
) -> torch.Tensor:
    """
    Wrapper around the Triton kernel. Works only on CUDA.
    """
    assert A.is_cuda and B.is_cuda and bias.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    out = torch.empty((M, N), dtype=A.dtype, device=A.device)

    grid = lambda meta: ((M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],)

    _matmul_gemm_bn_scale_softmax[grid](
        A, B, bias, gamma, beta,
        running_mean, running_var, scale,
        M, N, K,
        eps, momentum,
        scale.item(),
        out_ptr=out,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_K=128,
    )
    return out


# ------------------------------------------------------------------
# Optimized model
# ------------------------------------------------------------------

class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), Batch Normalization,
    scaling, and Softmax using a single fused Triton kernel.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum

        # Linear layer (weights + bias)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.empty(out_features))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        # BatchNorm parameters
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        # The BatchNorm layer will create running_mean and running_var as buffers
        # We will expose them to the kernel directly.

        # Scale parameter
        self.scale = nn.Parameter(torch.ones(scale_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_features)
        # Perform fused kernel
        out = matmul_gemm_bn_scale_softmax(
            A=x,
            B=self.weight.t(),
            bias=self.bias,
            gamma=self.bn.weight,
            beta=self.bn.bias,
            running_mean=self.bn.running_mean,
            running_var=self.bn.running_var,
            scale=self.scale,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
        )
        return out