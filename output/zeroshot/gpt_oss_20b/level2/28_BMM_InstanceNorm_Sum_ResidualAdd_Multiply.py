import torch
import torch.nn as nn
import triton
import triton.language as tl

# ----------------------------------------------------
# Triton kernels
# ----------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_fwd(
    a_ptr,  # (M, K)
    b_ptr,  # (K, N)
    out_ptr,  # (M, N)
    M, N, K,
    stride_a_m: tl.constexpr,
    stride_a_k: tl.constexpr,
    stride_b_k: tl.constexpr,
    stride_b_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    row_indices = row_start + tl.arange(0, BLOCK_M)
    col_indices = col_start + tl.arange(0, BLOCK_N)

    C = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + row_indices[:, None] * stride_a_m + (k + tl.arange(0, BLOCK_K)) * stride_a_k,
            mask=row_indices[:, None] < M,
            other=0.0,
        )
        b = tl.load(
            b_ptr + (k + tl.arange(0, BLOCK_K)) * stride_b_k + col_indices[None, :] * stride_b_n,
            mask=col_indices[None, :] < N,
            other=0.0,
        )
        C += tl.dot(a, b)

    mask = (row_indices[:, None] < M) & (col_indices[None, :] < N)
    tl.store(
        out_ptr + row_indices[:, None] * stride_out_m + col_indices[None, :] * stride_out_n,
        C,
        mask=mask,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def add_mul_fwd(
    x_ptr,  # (N,)
    y_ptr,  # (N,)
    out_ptr,  # (N,)
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    y = tl.load(y_ptr + offset, mask=mask, other=0.0)

    # fused: (x + y) * y
    out = (x + y) * y
    tl.store(out_ptr + offset, out, mask=mask)


# ----------------------------------------------------
# Model definition
# ----------------------------------------------------

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for matrix multiplication and fused add/mul.
    Instance normalization remains in PyTorch as it is highly optimized.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Linear layer parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device='cuda'))
        self.bias = nn.Parameter(torch.randn(out_features, device='cuda'))

        # InstanceNorm2d uses the output feature dimension
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): (batch_size, in_features)
            y (torch.Tensor): (batch_size, out_features)

        Returns:
            torch.Tensor: (batch_size, out_features)
        """
        batch_size, in_features = x.shape
        out_features = self.weight.shape[0]

        # ----------------------------------------------------
        # 1. Triton matmul: x @ weight.T
        # ----------------------------------------------------
        a = x  # (B, K)
        b = self.weight  # (N, K)
        # We need (B, N) output
        out_matmul = torch.empty((batch_size, out_features), device='cuda', dtype=torch.float32)

        grid = lambda meta: (
            (batch_size + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
            (out_features + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
        )

        matmul_fwd[grid](
            a_ptr=a,
            b_ptr=b,
            out_ptr=out_matmul,
            M=batch_size,
            N=out_features,
            K=in_features,
            stride_a_m=1,
            stride_a_k=batch_size,
            stride_b_k=1,
            stride_b_n=in_features,
            stride_out_m=1,
            stride_out_n=batch_size,
        )

        # Add bias
        out_matmul += self.bias

        # ----------------------------------------------------
        # 2. Instance Normalization
        # ----------------------------------------------------
        # InstanceNorm2d expects (N, C, H, W). We use 1x1 spatial dims.
        out_norm = self.instance_norm(out_matmul.unsqueeze(1).unsqueeze(1))
        out_norm = out_norm.squeeze(1).squeeze(1)

        # ----------------------------------------------------
        # 3. Triton fused add and mul: (x + y) * y
        # ----------------------------------------------------
        out_final = torch.empty_like(out_norm)
        N = out_norm.numel()
        BLOCK_SIZE = 256
        grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

        add_mul_fwd[grid](
            x_ptr=out_norm.reshape(-1),
            y_ptr=y.reshape(-1),
            out_ptr=out_final.reshape(-1),
            N=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return out_final