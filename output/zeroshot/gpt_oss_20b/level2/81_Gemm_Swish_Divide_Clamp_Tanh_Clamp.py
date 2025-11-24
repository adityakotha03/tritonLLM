import torch
import torch.nn as nn
import triton
import triton.language as tl

# ---------- Triton kernel ---------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 512, "BLOCK_SIZE_N": 512, "BLOCK_SIZE_K": 64}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_swish_tanh_clamp_kernel(
    A_ptr, B_ptr, bias_ptr, out_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N

    # Allocate accumulators in registers
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over tiles of K
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tiles of A and B
        A_offsets = (row_start[:, None] + tl.arange(0, BLOCK_SIZE_M)[:, None],
                     k + tl.arange(0, BLOCK_SIZE_K)[None, :])
        B_offsets = (k + tl.arange(0, BLOCK_SIZE_K)[:, None],
                     col_start[:, None] + tl.arange(0, BLOCK_SIZE_N)[None, :])

        a_mask = (A_offsets[0] < M) & (A_offsets[1] < K)
        b_mask = (B_offsets[0] < K) & (B_offsets[1] < N)

        a = tl.load(A_ptr + A_offsets[0] * K + A_offsets[1], mask=a_mask, other=0.0, dtype=tl.float16).to(tl.float32)
        b = tl.load(B_ptr + B_offsets[0] * N + B_offsets[1], mask=b_mask, other=0.0, dtype=tl.float16).to(tl.float32)

        acc += tl.dot(a, b)

    # Add bias
    bias_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)
    bias_mask = bias_offsets < N
    bias = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0, dtype=tl.float32)
    acc += bias[None, :]

    # Apply swish: y * sigmoid(y)
    acc_swish = acc * tl.sigmoid(acc)

    # Divide by 2
    acc_swish_div2 = acc_swish * 0.5

    # First clamp between -1 and 1
    acc_clamped1 = tl.clip(acc_swish_div2, -1.0, 1.0)

    # Tanh
    acc_tanh = tl.tanh(acc_clamped1)

    # Final clamp
    acc_final = tl.clip(acc_tanh, -1.0, 1.0)

    # Store result
    out_offsets = (row_start[:, None] + tl.arange(0, BLOCK_SIZE_M)[:, None],
                   col_start[:, None] + tl.arange(0, BLOCK_SIZE_N)[None, :])
    out_mask = (out_offsets[0] < M) & (out_offsets[1] < N)
    tl.store(out_ptr + out_offsets[0] * N + out_offsets[1],
             acc_final.to(tl.float16),
             mask=out_mask)

# ---------- Triton helper ----------------------------------------------------
def matmul_swish_tanh_clamp(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor):
    """
    A: (M, K)  float16
    B: (K, N)  float16
    bias: (N,)  float16
    returns: (M, N) float16
    """
    assert A.is_cuda and B.is_cuda and bias.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    out = torch.empty((M, N), dtype=A.dtype, device=A.device)

    grid = lambda meta: (
        (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
    )

    matmul_swish_tanh_clamp_kernel[grid](
        A, B, bias, out,
        M, N, K,
        BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64,
    )
    return out

# ---------- Optimized model --------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # Prepare weight and bias as parameters (float16)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.float16, device="cuda")
        )
        if bias:
            self.bias_param = nn.Parameter(
                torch.randn(out_features, dtype=torch.float16, device="cuda")
            )
        else:
            self.bias_param = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are float16
        x = x.to(torch.float16)
        # Transpose weight to match A (M, K) shape
        out = matmul_swish_tanh_clamp(
            x, self.weight.t(), self.bias_param if self.bias_param is not None else torch.zeros(self.out_features, dtype=torch.float16, device="cuda")
        )
        return out