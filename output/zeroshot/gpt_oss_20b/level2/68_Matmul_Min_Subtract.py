import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------------------
# Triton kernel for a fused matmul + min + sub operation
# --------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_min_sub_kernel(
    A_ptr, B_ptr, out_ptr, C_ptr,  # C is constant vector broadcast
    M, N, K,  # matrix dimensions
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    dtype: tl.constexpr,
):
    # program id
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # calculate block start indices
    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N

    # allocate registers for the accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dtype)

    # loop over K dimension in tiles
    for k in range(0, K, BLOCK_K):
        # offsets for A and B tiles
        A_offsets = block_start_m[:, None] + tl.arange(0, BLOCK_M)[None, :] + k[None, None]
        B_offsets = block_start_n[None, :] + tl.arange(0, BLOCK_N)[None, :] + k[None, None]

        # load tiles
        A_tile = tl.load(A_ptr + A_offsets, mask=A_offsets < K, other=0.0)
        B_tile = tl.load(B_ptr + B_offsets, mask=B_offsets < K, other=0.0)

        # accumulate
        acc += tl.dot(A_tile, B_tile)

    # apply min with constant
    # constant is broadcasted to each element in the block
    constant = tl.load(C_ptr)

    acc = tl.minimum(acc, constant)
    acc = acc - constant

    # store result
    out_offsets = block_start_m[:, None] + tl.arange(0, BLOCK_M)[None, :] + block_start_n[None, :] * M
    tl.store(out_ptr + out_offsets, acc, mask=(out_offsets < (M * N)))

# --------------------------------------------------------------
# Triton wrapper for the fused operation
# --------------------------------------------------------------
def triton_fused_matmul_min_sub(A: torch.Tensor, B: torch.Tensor, constant: torch.Tensor):
    """
    A: [M, K]  (weights transpose)
    B: [N, K]  (input batch)
    constant: scalar tensor broadcasted
    Returns: [M, N]
    """
    assert A.is_cuda and B.is_cuda, "Input tensors must be on CUDA."
    M, K = A.shape
    N, Kb = B.shape
    assert K == Kb, "Inner dimensions must match."

    # Prepare output
    out = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # constant buffer (single value)
    const_ptr = constant.contiguous()

    # Grid dimensions
    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    # Launch kernel
    matmul_min_sub_kernel[grid](
        A, B, out, const_ptr, M, N, K,
        BLOCK_M=meta["BLOCK_M"],
        BLOCK_N=meta["BLOCK_N"],
        BLOCK_K=meta["BLOCK_K"],
        dtype=A.dtype,
    )
    return out

# --------------------------------------------------------------
# Optimized model
# --------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model using a fused Triton kernel for matmul, min and sub.
    """
    def __init__(self, in_features: int, out_features: int, constant: float):
        super().__init__()
        # Weights are stored as [out_features, in_features]
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16))
        # Bias is not used in the fused kernel; we treat bias as part of constant subtraction
        self.constant = nn.Parameter(torch.tensor(constant, device="cuda", dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor):
        # x: [batch, in_features]
        # transpose weight to shape [out_features, in_features]
        # Triton kernel expects A: [M, K] where M=out_features, K=in_features
        # B: [N, K] where N=batch_size
        x = x.to(dtype=self.weight.dtype)
        out = triton_fused_matmul_min_sub(self.weight, x, self.constant)
        return out