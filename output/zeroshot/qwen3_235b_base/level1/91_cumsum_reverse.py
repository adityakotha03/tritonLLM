import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumsum_rev_kernel(
    x_ptr,        # pointer to input tensor
    out_ptr,      # pointer to output tensor
    n_elements,   # total number of elements in the tensor
    seq_len,      # length of the dimension we're cumsumming over (dim)
    stride,       # stride along the cumsum dimension
    num_seqs,     # number of independent sequences (total elements / seq_len)
    BLOCK_SIZE: tl.constexpr,
):
    # Each block handles one sequence (i.e., one row in the flattened view)
    pid = tl.program_id(0)
    if pid >= num_seqs:
        return

    # Calculate the starting index for this sequence
    seq_start = pid * seq_len
    offset = seq_start + (seq_len - 1) * stride  # Start from the last element in the sequence

    # Reverse cumulative sum: traverse from last to first element in the sequence
    cumsum_val = tl.load(x_ptr + offset)

    # Write the first (last element in original order)
    tl.store(out_ptr + offset, cumsum_val)

    # Iterate backward through the sequence (in memory: subtract stride each time)
    for i in range(1, seq_len):
        offset -= stride
        x_val = tl.load(x_ptr + offset)
        cumsum_val += x_val
        tl.store(out_ptr + offset, cumsum_val)


def triton_cumsum_rev(x: torch.Tensor, dim: int):
    """
    Custom Triton kernel to compute reverse cumulative sum along a given dimension.
    Equivalent to: torch.cumsum(x.flip(dim), dim=dim).flip(dim)
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    # Handle negative dim
    if dim < 0:
        dim = x.ndim + dim

    # Reshape to flatten all dimensions except `dim` into a single batch
    seq_len = x.size(dim)
    other_dims = x.size()[:dim] + x.size()[dim+1:]
    num_seqs = 1
    for d in other_dims:
        num_seqs *= d

    # Reshape to [num_seqs, seq_len] with appropriate stride
    x_reshaped = x.view(-1, seq_len) if dim == x.ndim - 1 else x.movedim(dim, -1).reshape(-1, seq_len)
    stride = x.stride(dim)

    # Prepare output
    out = torch.empty_like(x_reshaped)

    # Launch kernel
    n_elements = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    grid = lambda meta: (num_seqs,)

    cumsum_rev_kernel[grid](
        x_reshaped, out, n_elements, seq_len, stride, num_seqs, BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape back to original shape and move dim back if needed
    out = out.view(*other_dims, seq_len)
    if dim != x.ndim - 1:
        out = out.movedim(-1, dim)

    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom Triton kernel for reverse cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumsum_rev(x, self.dim)