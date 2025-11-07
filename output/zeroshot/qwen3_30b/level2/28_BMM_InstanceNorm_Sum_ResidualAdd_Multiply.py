import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_BIAS: tl.constexpr,
    BIAS_PTR,
    EPS: tl.constexpr
):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offset for current block
    block_start_m = pid * BLOCK_M
    block_start_n = pid_n * BLOCK_N
    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize pointers for current block
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K blocks
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load A and B
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        
        # Perform matmul
        acc += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Apply activation
    if ACTIVATION == "relu":
        acc = tl.relu(acc)
    
    # Handle bias
    if USE_BIAS:
        bias = tl.load(BIAS_PTR + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    
    # Store result
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn), acc, mask=c_mask)


@triton.jit
def instance_norm_kernel(
    x_ptr, weight_ptr, bias_ptr,
    M, N, C,
    stride_xm, stride_xn, stride_xc,
    stride_w, stride_b,
    eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr
):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offset for current block
    block_start_m = pid * BLOCK_M
    block_start_n = pid_n * BLOCK_N
    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator for mean and var
    mean = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    var = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over C blocks
    for c in range(0, tl.cdiv(C, BLOCK_C)):
        # Load data
        offs_c = c * BLOCK_C + tl.arange(0, BLOCK_C)
        x_ptrs = x_ptr + (offs_m[:, None, None] * stride_xm + offs_n[None, :, None] * stride_xn + offs_c[None, None, :] * stride_xc)
        x = tl.load(x_ptrs, mask=offs_c[None, None, :] < C, other=0.0)
        
        # Update mean and var
        mean += x
        var += x * x
        
    # Reduce across C dimension
    mean = tl.sum(mean, axis=2) / C
    var = tl.sum(var, axis=2) / C
    var = var - mean * mean
    
    # Normalize
    mean = mean[:, None]
    var = var[:, None]
    x_norm = x - mean
    x_norm = x_norm / tl.sqrt(var + eps)
    
    # Apply scale and shift
    weight = tl.load(weight_ptr + offs_n, mask=offs_n < N, other=0.0)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    weight = weight[None, :]
    bias = bias[None, :]
    
    # Compute output
    out = x_norm * weight + bias
    
    # Store result
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(x_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn), out, mask=c_mask)


@triton.jit
def add_mul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N,
    stride_am, stride_an, stride_bm, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offset for current block
    block_start_m = pid * BLOCK_M
    block_start_n = pid_n * BLOCK_N
    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_N)
    
    # Load A and B
    a = tl.load(a_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    b = tl.load(b_ptr + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    
    # Perform a + b
    add = a + b
    
    # Perform (a + b) * b
    out = add * b
    
    # Store result
    tl.store(c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn), out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_matmul_with_bias(x, w, b, activation="none"):
    assert x.is_cuda and w.is_cuda and (b is None or b.is_cuda), "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()
    
    M, K = x.shape
    K, N = w.shape
    C = N
    
    # Prepare output tensor
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    
    # Set block sizes
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    
    # Grid configuration
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    
    # Launch kernel
    matmul_kernel[grid](
        x, w, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        ACTIVATION=activation,
        USE_BIAS=(b is not None),
        BIAS_PTR=b,
        EPS=1e-5
    )
    return out


def triton_instance_norm(x, weight, bias):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    M, N, C = x.shape
    out = torch.empty_like(x)
    
    BLOCK_M, BLOCK_N, BLOCK_C = 128, 128, 64
    
    # Grid configuration
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    
    # Launch kernel
    instance_norm_kernel[grid](
        out, weight, bias,
        M, N, C,
        out.stride(0), out.stride(1), out.stride(2),
        weight.stride(0), bias.stride(0),
        1e-5,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C
    )
    return out


def triton_add_mul(a, b):
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()
    
    M, N = a.shape
    out = torch.empty_like(a)
    
    BLOCK_M, BLOCK_N = 128, 128
    
    # Grid configuration
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    
    # Launch kernel
    add_mul_kernel[grid](
        a, b, out,
        M, N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if out_features > 0 else None
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        self.weight.data = self.weight.data * (1 / (in_features ** 0.5))
        self.bias.data = self.bias.data * (1 / (in_features ** 0.5))
        self.eps = eps
        self.momentum = momentum
        
    def forward(self, x, y):
        # Linear layer with Triton kernel
        x = triton_matmul_with_bias(x, self.weight, self.bias, activation="relu")
        
        # Reshape for instance norm
        x = x.view(x.size(0), 1, 1, x.size(1))
        
        # Instance normalization with Triton kernel
        x = triton_instance_norm(x, self.weight, self.bias)
        
        # Reshape back
        x = x.view(x.size(0), x.size(3))
        
        # Add and multiply with Triton kernel
        x = triton_add_mul(x, y)
        
        return x