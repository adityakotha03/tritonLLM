import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_k = pid_k * BLOCK_SIZE_K

    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)

    # Create masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    offs_am = offs_m[:, None] * stride_am
    offs_ak = offs_k[None, :] * stride_ak
    offs_bk = offs_k[:, None] * stride_bk
    offs_bn = offs_n[None, :] * stride_bn

    # Load A and B blocks
    a = tl.load(a_ptr + offs_am + offs_ak, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    b = tl.load(b_ptr + offs_bk + offs_bn, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

    # Perform matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc += tl.dot(a, b, allow_tf32=True)

    # Apply GELU activation directly in the kernel
    if ACTIVATION == "gelu":
        acc = acc * 0.5 * (1.0 + tl.erf(acc / tl.sqrt(2.0)))
    elif ACTIVATION == "relu":
        acc = tl.maximum(acc, 0.0)

    # Store output
    offs_om = offs_m[:, None] * stride_om
    offs_on = offs_n[None, :] * stride_on
    tl.store(out_ptr + offs_om + offs_on, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def batch_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, mean_ptr, var_ptr,
    out_ptr,
    N, C,
    stride_xn, stride_xc,
    stride_wn, stride_bn,
    stride_on, stride_oc,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < C

    # Load inputs
    x = tl.load(x_ptr + offs * stride_xc, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offs * stride_wn, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offs * stride_bn, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + offs * stride_wn, mask=mask, other=0.0)
    var = tl.load(var_ptr + offs * stride_wn, mask=mask, other=0.0)

    # Normalize
    x_norm = (x - mean) / tl.sqrt(var + eps)

    # Scale and shift
    out = x_norm * weight + bias

    # Apply ReLU
    out = tl.maximum(out, 0.0)

    # Store output
    tl.store(out_ptr + offs * stride_oc, out, mask=mask)


def triton_matmul_with_batchnorm_gelu_relu(x, weight, bias, running_mean, running_var, weight_bn, bias_bn, eps=1e-5):
    M, K = x.shape
    K, N = weight.shape

    # Ensure contiguous tensors
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    running_mean = running_mean.contiguous()
    running_var = running_var.contiguous()
    weight_bn = weight_bn.contiguous()
    bias_bn = bias_bn.contiguous()

    # Output tensor
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)

    # Set up grid
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    num_blocks_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_blocks_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_blocks_k = triton.cdiv(K, BLOCK_SIZE_K)

    grid = (num_blocks_m, num_blocks_n, num_blocks_k)

    # Launch matmul kernel with GELU and ReLU fused
    matmul_kernel[grid](
        x, weight, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        "gelu"
    )

    # BatchNorm and ReLU fused
    out = out.contiguous()
    batch_norm_kernel[(M + 127) // 128, ](
        out, weight_bn, bias_bn, running_mean, running_var,
        out,
        M, N,
        out.stride(0), out.stride(1),
        weight_bn.stride(0), bias_bn.stride(0),
        out.stride(0), out.stride(1),
        eps,
        128
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.register_buffer("running_mean", torch.zeros(out_features))
        self.register_buffer("running_var", torch.ones(out_features))
        self.register_buffer("weight_bn", torch.ones(out_features))
        self.register_buffer("bias_bn", torch.zeros(out_features))

    def forward(self, x):
        return triton_matmul_with_batchnorm_gelu_relu(
            x, self.weight, self.bias,
            self.running_mean, self.running_var,
            self.weight_bn, self.bias_bn,
            eps=1e-5
        )