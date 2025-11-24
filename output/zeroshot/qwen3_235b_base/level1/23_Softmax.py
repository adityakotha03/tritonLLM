import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    input_row_ptr = input_ptr + row_idx * stride + col_offsets
    row = tl.load(input_row_ptr, mask=mask, other=-float('inf'))
    
    # Online softmax: subtract max for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    exp_row = tl.exp(row_minus_max)
    sum_exp = tl.sum(exp_row, axis=0)
    softmax_output = exp_row / sum_exp

    output_row_ptr = output_ptr + row_idx * stride + col_offsets
    tl.store(output_row_ptr, softmax_output, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom Triton kernel to perform Softmax activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        assert x.is_cuda, "Input tensor must be on CUDA."
        x = x.contiguous()
        batch_size, n_cols = x.shape
        out = torch.empty_like(x)

        # Choose block size as power of 2 covering the feature dimension
        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        # Grid size = number of rows (batch_size)
        grid = (batch_size,)

        softmax_kernel[grid](
            x,
            out,
            n_cols=n_cols,
            stride=x.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out