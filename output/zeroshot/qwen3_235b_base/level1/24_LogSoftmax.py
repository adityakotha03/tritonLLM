import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def logsoftmax_kernel(
    input_ptr, output_ptr,
    stride,
    batch_size, dim,
    BLOCK_SIZE: tl.constexpr,
):
    batch_pid = tl.program_id(0)
    row_start = batch_pid * stride
    offset = row_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offset < (row_start + dim)
    row = tl.load(input_ptr + offset, mask=mask, other=-float('inf'))
    
    # Online softmax: subtract max for numerical stability
    row_max = tl.max(row, axis=0)
    row_minus_max = row - row_max
    exp_row = tl.exp(row_minus_max)
    
    exp_sum = tl.sum(exp_row, axis=0)
    log_sum_exp = tl.log(exp_sum)
    
    result = row_minus_max - log_sum_exp
    tl.store(output_ptr + offset, result, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized version of LogSoftmax using a custom Triton kernel.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input tensor must be on GPU."
        x = x.contiguous()
        batch_size, dim = x.shape
        out = torch.empty_like(x)

        # Choose block size as power of 2, at least as large as dim
        BLOCK_SIZE = triton.next_power_of_2(dim)

        # Grid spans over batch dimension
        grid = lambda meta: (batch_size,)

        logsoftmax_kernel[grid](
            x,
            out,
            x.stride(0),
            batch_size,
            dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out