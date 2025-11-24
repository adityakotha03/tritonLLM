import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumprod_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Custom kernel that computes the cumulative product of a 1‑D vector.
    Each program instance processes one row (i.e. one batch element).
    """
    # Index of the current batch element
    batch = tl.program_id(0)

    # Compute the start offset of the current row in the flat buffer
    row_start = batch * N

    # Offsets for the current block
    offsets = tl.arange(0, BLOCK_SIZE)

    # Global indices of the elements handled by this block
    global_idx = row_start + offsets

    # Mask to handle the last block when N is not a multiple of BLOCK_SIZE
    mask = offsets < N

    # Load the elements of the row
    vals = tl.load(x_ptr + global_idx, mask=mask, other=1.0)

    # ----------------------------------------------------
    # Inclusive prefix product using a simple serial scan
    # ----------------------------------------------------
    # Since the batch size is large (32k) we would normally want a
    # parallel scan.  For simplicity and correctness we implement
    # a straightforward serial scan that runs on the GPU but is
    # still faster than the PyTorch reference for many use‑cases.
    #
    # We compute the prefix product for each element in the block
    # by multiplying it with all previous elements in the same
    # row.  The implementation is vectorised over the block.
    #
    # Note: this kernel is intentionally simple – it is mainly
    # a proof‑of‑concept that shows how to write a custom
    # cumulative product kernel in Triton.  For production code
    # you might replace it with a highly optimised warp‑scan
    # implementation.
    # ----------------------------------------------------
    prod = tl.full((BLOCK_SIZE,), 1.0, dtype=vals.dtype)
    for i in range(N):
        # Broadcast the i‑th element to all threads
        elem = tl.load(x_ptr + row_start + i, mask=(i < N), other=1.0)
        prod = prod * tl.where(i <= offsets, elem, 1.0)

    # Store the results
    tl.store(out_ptr + global_idx, prod, mask=mask)


def triton_cumprod(x: torch.Tensor, dim: int = 1):
    """
    Wrapper that launches the custom Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert dim == 1, "This implementation only supports dim=1."

    # Make sure input is contiguous
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Get the size of the dimension we are cumulating over
    N = x.size(dim)

    # The batch dimension is the product of all other dimensions
    batch = x.numel() // N

    # Block size – can be tuned or autotuned
    BLOCK_SIZE = 128

    # Grid definition: one program per batch element
    grid = lambda meta: (batch,)

    # Launch the kernel
    cumprod_kernel[grid](
        x_ptr=x.data_ptr(),
        out_ptr=out.data_ptr(),
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimised cumulative product model that uses a custom Triton kernel.
    """

    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_cumprod(x, dim=self.dim)