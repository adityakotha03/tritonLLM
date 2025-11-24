import torch
import triton
import triton.language as tl

# -------------------------------------------------------------
# Triton kernel for matrix‑multiplication (C = A @ B)
# -------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 8}, num_warps=1),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 8}, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 8}, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 8}, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Thread block coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Starting indices for this block
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N

    # Accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # K‑loop
    for k_start in range(0, K, BLOCK_SIZE_K):
        # 2‑D indices
        k = k_start + tl.arange(0, BLOCK_SIZE_K)[None, :]
        m = m_start + tl.arange(0, BLOCK_SIZE_M)[:, None]
        n = n_start + tl.arange(0, BLOCK_SIZE_N)[None, :]

        # Masks for out‑of‑bounds
        a_mask = (m < M) & (k < K)
        b_mask = (k < K) & (n < N)

        # Load tiles
        A = tl.load(A_ptr + m * stride_am + k * stride_ak, mask=a_mask, other=0.0)
        B = tl.load(B_ptr + k * stride_bk + n * stride_bn, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A, B)

    # Store result (with boundary mask)
    m_mask = (m_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < M)
    n_mask = (n_start + tl.arange(0, BLOCK_SIZE_N)[None, :] < N)
    out_mask = m_mask & n_mask
    tl.store(
        C_ptr + (m_start + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_cm
        + (n_start + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_cn,
        acc,
        mask=out_mask,
    )

# -------------------------------------------------------------
# Helper to invoke the Triton kernel
# -------------------------------------------------------------
def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication using a custom Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    assert A.is_contiguous() and B.is_contiguous(), "Inputs must be contiguous."
    assert A.shape[1] == B.shape[0], "Inner dimensions must match."

    M, K = A.shape
    _, N = B.shape
    dtype = A.dtype
    C = torch.empty((M, N), dtype=dtype, device=A.device)

    # Strides for contiguous tensors
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Grid definition
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    # Launch the kernel (autotune handles meta)
    _matmul_kernel[grid](
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
    )

    return C

# -------------------------------------------------------------
# New model using the Triton matmul
# -------------------------------------------------------------
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)