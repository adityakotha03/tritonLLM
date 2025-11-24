import torch
import torch.nn as nn
import triton
import triton.language as tl

# =======================
# Triton kernel
# =======================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def mul_scalar_kernel(
    A_ptr,          # pointer to input matrix A
    s_ptr,          # pointer to scalar s
    out_ptr,        # pointer to output matrix C
    n_elements,     # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Multiply each element of A by scalar s and store the result in out.
    """
    # Starting index for this program (thread block)
    block_start = tl.program_id(0) * BLOCK_SIZE

    # Create a vector of offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to avoid out-of-bounds access at the tail
    mask = offsets < n_elements

    # Load elements from A
    a_vals = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # Load scalar once (broadcast)
    s = tl.load(s_ptr)

    # Compute
    out_vals = a_vals * s

    # Store result
    tl.store(out_ptr + offsets, out_vals, mask=mask)


# =======================
# Triton wrapper
# =======================
def triton_mul_scalar(A: torch.Tensor, s: float) -> torch.Tensor:
    """
    Wrapper that launches the Triton kernel for matrix‑scalar multiplication.
    """
    assert A.is_cuda, "Input tensor must be on CUDA device."
    A = A.contiguous()
    out = torch.empty_like(A)

    n_elements = A.numel()

    # Create a single‑element tensor for the scalar
    s_tensor = torch.tensor([s], dtype=A.dtype, device=A.device)

    # Define grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel
    mul_scalar_kernel[grid](
        A, s_tensor, out, n_elements, BLOCK_SIZE=grid(1)["BLOCK_SIZE"]
    )
    return out


# =======================
# Optimized model
# =======================
class ModelNew(nn.Module):
    """
    Optimized model that performs matrix‑scalar multiplication using a custom
    Triton kernel for maximum throughput on NVIDIA A100 GPUs.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return triton_mul_scalar(A, s)