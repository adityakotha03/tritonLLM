import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Triton kernel: batched matrix‑multiply, scalar multiply, bias add and LeakyReLU
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128},
                      num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128},
                      num_warps=16),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256},
                      num_warps=16),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_gemm_kernel(
    a_ptr,          # [B, K]
    w_ptr,          # [N, K]  (weight matrix, not transposed)
    b_ptr,          # [N]      (bias vector)
    out_ptr,        # [B, N]
    M, N, K,        # matrix dimensions
    multiplier,     # scalar multiplier
    neg_slope,      # negative slope for LeakyReLU
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Block indices
    block_m = tl.program_id(0)
    block_n = tl.program_id(1)

    # Offsets for this block
    row = block_m * BLOCK_SIZE_M
    col = block_n * BLOCK_SIZE_N

    # Accumulator for the dot product (bf16 → fp32)
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    # Iterate over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        k0 = k + tl.arange(0, BLOCK_SIZE_K)

        # Load A tile: [M, K]
        a_offsets = (row + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)[:, None] * K + k0[None, :]
        a = tl.load(a_ptr + a_offsets, mask=(a_offsets < M * K), other=0.0).to(tl.bfloat16)

        # Load W tile: [N, K] -> we need the transpose for dot
        w_offsets = (col + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64)[:, None] * K + k0[None, :]
        w = tl.load(w_ptr + w_offsets, mask=(w_offsets < N * K), other=0.0).to(tl.bfloat16)

        # Compute partial product (bf16 → fp32)
        acc += tl.dot(a, w)

    # Apply scalar multiplier and bias
    out = acc * multiplier

    # Broadcast bias to [M, N]
    bias = tl.load(b_ptr + (col + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64))
    out += bias[None, :]

    # LeakyReLU
    neg_out = out * neg_slope
    out = tl.where(out > 0, out, neg_out)

    # Store result
    out_offsets = (row + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)[:, None] * N + col + tl.arange(0, BLOCK_SIZE_N)
    tl.store(out_ptr + out_offsets, out.to(tl.bfloat16), mask=(out_offsets < M * N))


def triton_fused_gemm(a: torch.Tensor,
                      weight: torch.Tensor,
                      bias: torch.Tensor,
                      multiplier: float,
                      neg_slope: float):
    """
    Wrapper for the fused Triton kernel.
    a:      [B, K]  (bf16)
    weight: [N, K]  (bf16)
    bias:   [N]
    """
    assert a.is_cuda and weight.is_cuda and bias.is_cuda
    a = a.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    B, K = a.shape
    N = weight.shape[0]
    out = torch.empty((B, N), dtype=a.dtype, device=a.device)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128

    grid = lambda meta: (
        (B + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
    )

    fused_gemm_kernel[grid](
        a_ptr=a.data_ptr(),
        w_ptr=weight.data_ptr(),
        b_ptr=bias.data_ptr(),
        out_ptr=out.data_ptr(),
        M=B,
        N=N,
        K=K,
        multiplier=multiplier,
        neg_slope=neg_slope,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return out


# --------------------------------------------------------------------------- #
# Optimized model
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Same functional model as the original, but using a fused Triton kernel for
    the linear layer + scalar multiply + LeakyReLU. The weight and bias are stored
    in bfloat16 to fully exploit Tensor Core performance on the A100.
    """
    def __init__(self, in_features: int, out_features: int,
                 multiplier: float, negative_slope: float):
        super().__init__()
        # Linear layer without bias (bias handled separately)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.data = self.linear.weight.data.to(torch.bfloat16)
        self.bias = nn.Parameter(torch.zeros(out_features, device="cuda", dtype=torch.bfloat16))
        self.multiplier = multiplier
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor):
        # Ensure inputs are bfloat16
        x = x.to(torch.bfloat16)

        # Compute the fused GEMM + scalar multiply + bias + LeakyReLU
        out = triton_fused_gemm(
            a=x,
            weight=self.linear.weight,
            bias=self.bias,
            multiplier=self.multiplier,
            neg_slope=self.negative_slope,
        )
        return out