import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_mish_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_FC: tl.constexpr,
    MAX_N: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_m = pid // (M // BLOCK_M)
    pid_n = pid % (M // BLOCK_M)

    # offsets
    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N

    # offsets for the A matrix
    offs_am = block_start_m + tl.arange(0, BLOCK_M)
    offs_ak = tl.arange(0, BLOCK_K)
    # offsets for the B matrix
    offs_bk = tl.arange(0, BLOCK_K)
    offs_bn = block_start_n + tl.arange(0, BLOCK_N)

    # pointers to the start of the blocks
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # load A
        a = tl.load(a_ptrs, mask=offs_ak[None, :] < K - k * BLOCK_K, other=0.0)
        # load B
        b = tl.load(b_ptrs, mask=offs_bk[:, None] < K - k * BLOCK_K, other=0.0)

        # matmul
        accumulator += tl.dot(a, b)

        # update pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # clamp for gradient
    accumulator = tl.where(accumulator < 20.0, accumulator, 20.0)
    accumulator = tl.where(accumulator > -20.0, accumulator, -20.0)

    # Mish activation: x * tanh(softplus(x))
    # first compute softplus: log(1 + exp(x))
    softplus = tl.log1p(tl.exp(accumulator))
    # then compute tanh(softplus)
    tanh_softplus = tl.tanh(softplus)
    # then multiply by input
    out = accumulator * tanh_softplus

    # second Mish activation
    # same computation: x * tanh(softplus(x))
    softplus = tl.log1p(tl.exp(out))
    tanh_softplus = tl.tanh(softplus)
    out = out * tanh_softplus

    # write to output
    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
def matmul_mish_triton(a, b, out, M, N, K):
    # assuming that a and b are contiguous
    assert a.is_contiguous() and b.is_contiguous(), "Tensors must be contiguous"
    assert a.dtype == torch.float32 and b.dtype == torch.float32, "Only FP32 supported"

    # determine grid size
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']), )

    # launch the kernel
    matmul_mish_kernel[grid](
        a, b, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=256, BLOCK_N=256, BLOCK_K=64,
        GROUP_SIZE_M=8,
        ACTIVATION="mish",
        USE_FC=False,
        MAX_N=2048,
        TILE_SIZE=128,
    )


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Perform matrix multiplication and two Mish activations in a single fused Triton kernel
        out = torch.empty_like(self.linear.weight)
        matmul_mish_triton(x, self.linear.weight, out, x.size(0), self.linear.out_features, self.linear.in_features)
        return out