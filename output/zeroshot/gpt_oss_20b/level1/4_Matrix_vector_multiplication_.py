import torch
import triton
import triton.language as tl


# Kernel for matrix–vector multiplication  (C = A * B)
# A: (M, K)  B: (K, 1)  C: (M, 1)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 256}, name="small"),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_K": 256}, name="medium"),
        triton.Config({"BLOCK_SIZE_M": 512, "BLOCK_SIZE_K": 256}, name="large"),
    ],
    key=["M", "K"],
)
@triton.jit
def _matmul_vec_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_cm,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID corresponds to a block of rows of A
    pid = tl.program_id(0)
    start_m = pid * BLOCK_SIZE_M
    m_idx = start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = m_idx < M

    # Accumulator for the dot product of each row with the vector B
    acc = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)

    # Loop over K in tiles
    for k in range(0, K, BLOCK_SIZE_K):
        # Load a tile of A: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_tile = tl.load(
            A_ptr
            + m_idx[:, None] * stride_am
            + (k + tl.arange(0, BLOCK_SIZE_K)) * stride_ak,
            mask=mask_m[:, None] & (k + tl.arange(0, BLOCK_SIZE_K) < K),
            other=0.0,
        )

        # Load the corresponding tile of B: shape (BLOCK_SIZE_K,)
        b_tile = tl.load(
            B_ptr + (k + tl.arange(0, BLOCK_SIZE_K)) * stride_bk,
            mask=(k + tl.arange(0, BLOCK_SIZE_K) < K),
            other=0.0,
        )

        # Accumulate the dot product
        acc += tl.dot(a_tile, b_tile)

    # Store the result
    tl.store(C_ptr + m_idx * stride_cm, acc, mask=mask_m)


def triton_matmul_vec(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix-vector multiplication using a custom Triton kernel.
    """
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    out = torch.empty((M, 1), device=A.device, dtype=A.dtype)

    # Define grid size: one program per BLOCK_SIZE_M rows
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]),)

    # Launch kernel
    _matmul_vec_kernel[grid](
        A, B, out,
        M, K,
        stride_am=K,   # stride from row i to row i+1 in A
        stride_ak=1,   # A is contiguous in columns
        stride_bk=1,   # B is contiguous
        stride_cm=1,   # C is a vector
        BLOCK_SIZE_M=grid.__closure__[0].cell_contents["BLOCK_SIZE_M"],  # noqa: E501
        BLOCK_SIZE_K=grid.__closure__[0].cell_contents["BLOCK_SIZE_K"],  # noqa: E501
    )

    return out


class ModelNew(torch.nn.Module):
    """
    Optimized model that uses a custom Triton kernel for matrix-vector multiplication.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul_vec(A, B)