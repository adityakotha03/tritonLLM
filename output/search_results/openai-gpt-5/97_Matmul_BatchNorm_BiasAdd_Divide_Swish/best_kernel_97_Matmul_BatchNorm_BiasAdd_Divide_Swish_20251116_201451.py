# Best Kernel for 97_Matmul_BatchNorm_BiasAdd_Divide_Swish
# Generated: 20251116_201451
# Speedup: 1.05x
# Runtime: 8.1400 ms
# Round: 1
# Idea: Use Tensor Cores with mixed precision + fold BN/bias/divide into an affine epilogue, then Swish - What: Run the GEMM in BF16 (or TF32) on Tensor Cores with FP32 accumulation using Triton’s tl.dot (allow_tf32=True or bf16 path). Precompute per-output-channel affine coefficients to fold BatchNorm, extra bias, and divide into a single scale/shift in the epilogue: - a = gamma / sqrt(running_var + eps) / divide_value - b = beta + extra_bias - running_mean * a - Epilogue: y = a * (XW^T + bias_linear) + b, then Swish y = y * sigmoid(y) - Keep Swish in FP32 for accuracy; cast back as needed. - Why it helps on A100: A100’s Tensor Cores deliver up to 312 TFLOPS (BF16/FP16) or 156 TFLOPS (TF32). Moving the GEMM to Tensor Cores with FP32 accumulation yields large speedups vs FP32 CUDA cores, while folding BN/bias/div reduces extra memory reads/writes and arithmetic in separate kernels. Keeping the nonlinear in FP32 preserves numerical stability. - Targets: Compute throughput (Tensor Cores), memory traffic (fusion), instruction count (epilogue folding).

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_affine_swish_kernel(
    A_ptr,  # bf16 [M, K]
    B_ptr,  # bf16 [K, N]
    C_ptr,  # fp32 [M, N]
    scale_ptr,  # fp32 [N]
    bias_ptr,   # fp32 [N]
    M: tl.constexpr,  # rows of A / C
    N: tl.constexpr,  # cols of B / C
    K: tl.constexpr,  # cols of A / rows of B
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_iter = tl.cdiv(K, BLOCK_K)
    for k in range(0, k_iter):
        k_mask = offs_k[None, :] + k * BLOCK_K < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask, other=0.0, eviction_policy="evict_last")
        b = tl.load(b_ptrs, mask=k_mask.T & (offs_n[None, :] < N), other=0.0, eviction_policy="evict_last")
        # a, b are bf16 -> tl.dot uses tensor cores; accumulates to fp32
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Load per-column scale and bias
    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)

    # Apply affine: y = acc * scale + bias
    y = acc * scale[None, :] + bias[None, :]

    # Swish: y * sigmoid(y)
    # sigmoid(y) = 1 / (1 + exp(-y))
    sig = 1.0 / (1.0 + tl.exp(-y))
    y = y * sig

    # Store
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask_store = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, y, mask=mask_store)


def triton_matmul_affine_swish(a_bf16: torch.Tensor,
                               b_bf16: torch.Tensor,
                               scale_fp32: torch.Tensor,
                               bias_fp32: torch.Tensor,
                               out_fp32: torch.Tensor):
    assert a_bf16.is_cuda and b_bf16.is_cuda and out_fp32.is_cuda
    M, K = a_bf16.shape
    KB, N = b_bf16.shape
    assert KB == K, "K dimensions must match"
    # Strides in elements
    stride_am, stride_ak = a_bf16.stride()
    stride_bk, stride_bn = b_bf16.stride()
    stride_cm, stride_cn = out_fp32.stride()

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    matmul_affine_swish_kernel[grid](
        a_bf16, b_bf16, out_fp32, scale_fp32, bias_fp32,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )
    return out_fp32


class ModelNew(nn.Module):
    """
    Optimized model: GEMM on Tensor Cores in BF16 with FP32 accumulation.
    BatchNorm (inference), extra bias, and division are folded into a fused affine epilogue.
    Applies Swish in-kernel in FP32. Falls back to PyTorch path in training mode.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = float(divide_value)
        self.in_features = in_features
        self.out_features = out_features

    def _compute_affine_params(self, device, dtype=torch.float32):
        # Use BN running stats and affine parameters
        gamma = self.bn.weight.detach().to(device=device, dtype=dtype) if self.bn.weight is not None else torch.ones(self.out_features, device=device, dtype=dtype)
        beta = self.bn.bias.detach().to(device=device, dtype=dtype) if self.bn.bias is not None else torch.zeros(self.out_features, device=device, dtype=dtype)
        running_mean = self.bn.running_mean.detach().to(device=device, dtype=dtype)
        running_var = self.bn.running_var.detach().to(device=device, dtype=dtype)
        eps = torch.tensor(self.bn.eps, device=device, dtype=dtype)

        a = gamma / torch.sqrt(running_var + eps) / self.divide_value  # [N]
        b = beta - running_mean * a  # [N]

        # Fold extra bias (may be scalar or per-channel)
        if self.bias.numel() == 1:
            b = b + self.bias.to(device=device, dtype=dtype).reshape(()).expand_as(b)
        elif self.bias.numel() == self.out_features:
            b = b + self.bias.to(device=device, dtype=dtype)
        else:
            # Generic broadcast to [N]
            b = b + self.bias.to(device=device, dtype=dtype).view(-1).expand_as(b)

        # Fold linear bias: a * bias_linear
        if self.linear.bias is not None:
            b = b + a * self.linear.bias.detach().to(device=device, dtype=dtype)

        return a, b  # scale, bias

    def forward(self, x: torch.Tensor):
        # Fast path: inference/eval mode on CUDA with 2D input
        if (not self.training) and x.is_cuda and x.dim() == 2 and x.shape[1] == self.in_features:
            M, K = x.shape
            N = self.out_features
            device = x.device

            # Prepare inputs for Tensor Cores: BF16
            A_bf16 = x.contiguous().to(torch.bfloat16)

            # Weight: use W^T (K, N), BF16
            Wt_bf16 = self.linear.weight.t().contiguous().to(torch.bfloat16)

            # Affine params (scale, bias) in FP32
            scale_fp32, bias_fp32 = self._compute_affine_params(device=device, dtype=torch.float32)

            # Output tensor in FP32
            out = torch.empty((M, N), device=device, dtype=torch.float32)

            # Launch Triton kernel
            triton_matmul_affine_swish(A_bf16, Wt_bf16, scale_fp32.contiguous(), bias_fp32.contiguous(), out)

            return out

        # Fallback (training or unsupported shapes): original computation
        x = self.linear(x)
        x = self.bn(x)
        x = x + self.bias
        x = x / self.divide_value
        x = x * torch.sigmoid(x)
        return x