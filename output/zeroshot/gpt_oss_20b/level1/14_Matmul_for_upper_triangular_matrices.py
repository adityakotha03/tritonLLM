import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def _matmul_upper_kernel(
    A_ptr: tl.tensor,
    B_ptr: tl.tensor,
    C_ptr: tl.tensor,
    N: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute C = triu(A @ B) for upper triangular A and B.
    """
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    row_start = pid_x * BLOCK_SIZE
    col_start = pid_y * BLOCK_SIZE

    row_offset = row_start + tl.arange(0, BLOCK_SIZE)[:, None]  # shape (B,1)
    col_offset = col_start + tl.arange(0, BLOCK_SIZE)[None, :]  # shape (1,B)

    row_mask = row_offset < N
    col_mask = col_offset < N

    acc = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)

    for k in range(0, N, BLOCK_SIZE):
        k_offset = k + tl.arange(0, BLOCK_SIZE)[None, :]  # shape (1,B)
        k_mask = k_offset < N

        # A block: rows=row_offset, cols=k_offset
        a_ptr = A_ptr + (row_offset * N + k_offset)
        a_mask = row_mask[:, None] & k_mask[None, :]
        a = tl.load(a_ptr, mask=a_mask, other=0.0)

        # B block: rows=k_offset, cols=col_offset
        b_ptr = B_ptr + (k_offset * N + col_offset)
        b_mask = k_mask[:, None] & col_mask[None, :]
        b = tl.load(b_ptr, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    upper_mask = row_offset <= col_offset
    final_mask = row_mask[:, None] & col_mask[None, :] & upper_mask

    out_ptr = C_ptr + (row_offset * N + col_offset)
    tl.store(out_ptr, acc, mask=final_mask)


def triton_matmul_upper(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper that launches the Triton kernel for upper‑triangular matrix multiplication.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    assert A.shape == B.shape, "Input shapes must match."
    assert A.dim() == 2 and A.shape[0] == A.shape[1], "Inputs must be square matrices."

    A = A.contiguous()
    B = B.contiguous()

    N = A.shape[0]
    out = torch.empty_like(A)

    grid = lambda meta: (
        (N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    _matmul_upper_kernel[grid](A, B, out, N, BLOCK_SIZE=128)
    return out


class ModelNew(torch.nn.Module):
    """
    Model that performs matrix multiplication for upper triangular matrices
    using a custom Triton kernel.
    """

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul_upper(A, B)