import torch
import torch.nn as nn
import triton
import triton.language as tl

# ------------------------------------------------------------
# Triton kernel: fused linear -> (matmul + bias + (1+scale)*output)
# ------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_fused_kernel(
    X_ptr,          # [M, K]
    W_ptr,          # [K, N]
    B_ptr,          # [N]
    out_ptr,        # [M, N]
    scaling,        # float32
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    stride_x_m: tl.constexpr,
    stride_x_k: tl.constexpr,
    stride_w_k: tl.constexpr,
    stride_w_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
):
    # Grid coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block start indices
    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N

    # Allocate accumulator in registers
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

    # Compute tile boundaries
    row_offsets = block_start_m + tl.arange(0, BLOCK_M)
    col_offsets = block_start_n + tl.arange(0, BLOCK_N)

    # Masks for boundaries
    mask_m = row_offsets[:, None] < M
    mask_n = col_offsets[None, :] < N

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)

        # Load tiles of X and W
        X_tile = tl.load(
            X_ptr + row_offsets[:, None] * stride_x_m + k_offsets[None, :] * stride_x_k,
            mask=mask_m[:, None] & (k_offsets[None, :] < K),
            other=0.0,
        )
        W_tile = tl.load(
            W_ptr + k_offsets[:, None] * stride_w_k + col_offsets[None, :] * stride_w_n,
            mask=(k_offsets[:, None] < K) & mask_n[None, :],
            other=0.0,
        )

        # Accumulate
        acc += tl.dot(X_tile, W_tile)

    # Add bias
    B_tile = tl.load(B_ptr + col_offsets, mask=mask_n, other=0.0)
    acc = acc + B_tile[None, :]

    # Apply scaling factor: out = (1 + scaling) * acc
    alpha = 1.0 + scaling
    acc = acc * alpha

    # Store result
    tl.store(
        out_ptr + row_offsets[:, None] * stride_out_m + col_offsets[None, :] * stride_out_n,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )

# ------------------------------------------------------------
# Triton wrapper
# ------------------------------------------------------------
def triton_linear_fused(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, scaling: float):
    """
    x:     (M, K)  float16
    weight:(K, N)  float16
    bias:  (N,)    float16
    scaling: float32
    """
    M, K = x.shape
    K_, N = weight.shape
    assert K_ == K

    # Output
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Strides
    stride_x_m = x.stride(0)
    stride_x_k = x.stride(1)
    stride_w_k = weight.stride(0)
    stride_w_n = weight.stride(1)
    stride_out_m = out.stride(0)
    stride_out_n = out.stride(1)

    # Grid
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))

    linear_fused_kernel[grid](
        x, weight, bias, out,
        scaling,
        M, N, K,
        stride_x_m=stride_x_m,
        stride_x_k=stride_x_k,
        stride_w_k=stride_w_k,
        stride_w_n=stride_w_n,
        stride_out_m=stride_out_m,
        stride_out_n=stride_out_n,
    )
    return out

# ------------------------------------------------------------
# Optimized model
# ------------------------------------------------------------
class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, scaling, and residual addition
    using a custom Triton kernel for fusion.
    """
    def __init__(self, in_features: int, out_features: int, scaling_factor: float):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float16 for TensorCore acceleration
        x_fp16 = x.to(torch.float16)
        weight_fp16 = self.linear.weight.to(torch.float16)
        bias_fp16 = self.linear.bias.to(torch.float16)

        out_fp16 = triton_linear_fused(
            x_fp16,
            weight_fp16,
            bias_fp16,
            self.scaling_factor,
        )
        # Cast back to original dtype
        return out_fp16.to(x.dtype)