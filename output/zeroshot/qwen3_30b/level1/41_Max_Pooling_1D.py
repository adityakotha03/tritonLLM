import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_pool1d_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    indices_ptr,  # Pointer to indices tensor (only if return_indices=True)
    batch_size,
    num_features,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    dilation,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    pid_output = tl.program_id(2)

    # Calculate output position
    output_start = pid_output * BLOCK_SIZE
    output_end = min(output_start + BLOCK_SIZE, output_length)

    # Iterate over output positions in this block
    for output_idx in range(output_start, output_end):
        # Calculate the start and end indices in the input
        input_start = output_idx * stride - padding
        input_end = input_start + kernel_size * dilation

        # Clamp the input indices to valid range
        input_start = tl.max(tl.tensor(0), input_start)
        input_end = tl.min(tl.tensor(input_length), input_end)

        # Compute the effective kernel size for this output position
        effective_kernel_size = (input_end - input_start + dilation - 1) // dilation
        effective_kernel_size = tl.min(tl.tensor(kernel_size), effective_kernel_size)

        # Find max value and its index
        max_val = tl.load(x_ptr + pid_batch * num_features * input_length + pid_feature * input_length + input_start)
        max_idx = input_start

        # Loop through the input window
        for i in range(1, effective_kernel_size):
            idx = input_start + i * dilation
            val = tl.load(x_ptr + pid_batch * num_features * input_length + pid_feature * input_length + idx, mask=idx < input_length, other=-float('inf'))
            # Use comparison to update max
            mask = val > max_val
            max_val = tl.where(mask, val, max_val)
            max_idx = tl.where(mask, idx, max_idx)

        # Store the max value
        out_ptr += pid_batch * num_features * output_length + pid_feature * output_length + output_idx
        tl.store(out_ptr, max_val)

        # Store indices if requested
        if indices_ptr is not None:
            indices_ptr += pid_batch * num_features * output_length + pid_feature * output_length + output_idx
            tl.store(indices_ptr, max_idx)


def triton_max_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int, return_indices: bool) -> tuple:
    """
    Custom Triton-based Max Pooling 1D implementation.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, L_in)
        kernel_size (int): Size of the pooling window
        stride (int): Stride of the pooling window
        padding (int): Padding applied to the input
        dilation (int): Spacing between kernel elements
        return_indices (bool): Whether to return indices of maximum values

    Returns:
        torch.Tensor: Pooled output tensor (B, C, L_out)
        torch.Tensor: Indices of maximum values (if return_indices=True)
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    B, C, L_in = x.shape
    L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    assert L_out > 0, "Output length must be positive"

    # Allocate output
    out = torch.empty(B, C, L_out, dtype=x.dtype, device=x.device)

    # Allocate indices if needed
    indices = None
    if return_indices:
        indices = torch.empty(B, C, L_out, dtype=torch.int64, device=x.device)

    # Define grid
    grid = lambda meta: (B, C, (L_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"])

    # Launch kernel
    max_pool1d_kernel[grid](
        x,
        out,
        indices,
        B,
        C,
        L_in,
        L_out,
        kernel_size,
        stride,
        padding,
        dilation,
        BLOCK_SIZE=128,
        TILE_SIZE=16
    )

    if return_indices:
        return out, indices
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Directly call Triton kernel for full optimization
        out = triton_max_pool1d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            return_indices=self.return_indices
        )
        return out if not self.return_indices else out[0]  # Return only output when not returning indices