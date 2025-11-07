import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_sigmoid_kernel(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak, stride_bk, stride_bn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    pid = tl.program_id(0)
    pid_m = pid // (n // BLOCK_SIZE_N)
    pid_n = pid % (n // BLOCK_SIZE_N)

    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, k, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k + BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k + BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ACTIVATION == 1:
        accumulator = tl.sigmoid(accumulator)

    c = accumulator.to(tl.float16)

    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)

    mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(c_ptr + offs_m[:, None] * n + offs_n[None, :], c, mask=mask)


@triton.jit
def logsumexp_kernel(
    x_ptr, y_ptr,
    m, n,
    stride_xm, stride_xn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    block_start_m = pid * BLOCK_SIZE_M
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < m
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask_m[:, None], other=-float("inf"))

    # Find max value in each row
    x_max = tl.max(x, axis=1)
    x_max = tl.broadcast_to(x_max[:, None], (BLOCK_SIZE_M, BLOCK_SIZE_N))

    # Compute exp(x - max)
    x_shifted = x - x_max
    x_exp = tl.exp(x_shifted)
    x_exp = tl.where(x > -float("inf"), x_exp, 0.0)

    # Sum the exponentials
    x_sum = tl.sum(x_exp, axis=1)

    # Log of the sum
    result = x_max[:, 0] + tl.log(x_sum)

    # Write output
    out_offsets = offs_m
    tl.store(y_ptr + out_offsets, result, mask=mask_m)


def triton_matmul_sigmoid(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, activation: bool):
    assert x.is_cuda and w1.is_cuda and w2.is_cuda, "All inputs must be on CUDA"
    assert x.dtype == torch.float16 and w1.dtype == torch.float16 and w2.dtype == torch.float16, "Only float16 supported"
    assert x.is_contiguous() and w1.is_contiguous() and w2.is_contiguous(), "Inputs must be contiguous"

    m, k = x.shape
    k2, n = w1.shape
    assert k == k2, "Input size mismatch"
    assert w1.shape[0] == k and w2.shape[0] == n, "Weight shape mismatch"

    # Output shape: (m, n)
    out = torch.empty(m, n, dtype=torch.float16, device=x.device)

    # Determine grid size
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),)

    # Launch kernel
    matmul_sigmoid_kernel[grid](
        x, w1, out,
        m, n, k,
        x.stride(0), x.stride(1), w1.stride(0), w1.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        ACTIVATION=1 if activation else 0
    )

    # Second matmul: out * w2
    out2 = torch.empty(m, w2.shape[1], dtype=torch.float16, device=x.device)
    out2_ptr = out2.data_ptr()
    out_ptr = out.data_ptr()

    # Second matmul kernel with fused sigmoid
    matmul_sigmoid_kernel[grid](
        out, w2, out2,
        m, w2.shape[1], n,
        out.stride(0), out.stride(1), w2.stride(0), w2.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        ACTIVATION=0  # No activation after second matmul
    )

    return out2


def triton_logsumexp(x: torch.Tensor):
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dtype == torch.float16, "Only float16 supported"
    assert x.is_contiguous(), "Input must be contiguous"

    m, n = x.shape

    # Output shape: (m,)
    out = torch.empty(m, dtype=torch.float16, device=x.device)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128

    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]),)

    logsumexp_kernel[grid](
        x, out,
        m, n,
        x.stride(0), x.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
    )

    # Convert to float32 for precision
    return out.to(torch.float32)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        # Initialize weights with proper dtype
        self.weight1 = nn.Parameter(torch.randn(input_size, hidden_size, dtype=torch.float16, device="cuda"))
        self.weight2 = nn.Parameter(torch.randn(hidden_size, output_size, dtype=torch.float16, device="cuda"))

    def forward(self, x):
        # Ensure input is float16 and contiguous
        x = x.to(torch.float16)
        x = x.contiguous()

        # First matmul + sigmoid
        x = triton_matmul_sigmoid(x, self.weight1, self.weight2, activation=True)

        # LogSumExp over features
        x = triton_logsumexp(x)

        return x