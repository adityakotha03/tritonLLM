import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool1d_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    in_channels,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of output elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_length

    # Calculate output batch and channel indices
    out_idx = offsets
    out_batch = pid // (output_length // in_channels)
    out_channel = (pid % (output_length // in_channels)) // output_length
    out_batch_idx = out_batch * in_channels * output_length
    out_channel_idx = out_channel * output_length

    # Calculate the starting position in the input for this output element
    start_input = (out_idx // stride) * stride
    start_input = start_input - padding
    start_input = tl.max(tl.zeros((), tl.int32), start_input)
    
    # Compute the number of elements to sum over in the input
    num_elements = tl.min(kernel_size, input_length - start_input)
    
    # Compute input indices
    input_offsets = start_input + tl.arange(0, kernel_size)
    input_mask = input_offsets < input_length
    
    # Read the input values and sum them up
    x = tl.load(x_ptr + (out_batch_idx + out_channel_idx) * input_length + input_offsets, mask=input_mask, other=0.0)
    sum_val = tl.sum(x, axis=0)
    
    # Divide by the number of valid elements
    avg_val = sum_val / num_elements
    tl.store(out_ptr + (out_batch_idx + out_channel_idx) * output_length + offsets, avg_val, mask=mask)

def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    # Ensure inputs are on GPU and contiguous
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    batch_size, in_channels, input_length = x.shape
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1

    # Create output tensor
    out = torch.empty(batch_size, in_channels, output_length, device=x.device, dtype=x.dtype)

    # Define grid and block size
    n_elements = batch_size * in_channels * output_length
    BLOCK_SIZE = 128  # Power of 2 for optimal performance

    # Grid: one block per output element
    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],

    # Launch Triton kernel
    avg_pool1d_kernel[grid](
        x_ptr=x, 
        out_ptr=out, 
        batch_size=batch_size,
        in_channels=in_channels,
        input_length=input_length,
        output_length=output_length,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)