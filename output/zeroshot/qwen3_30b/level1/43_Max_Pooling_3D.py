import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_pool3d_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    indices_ptr,  # Pointer to indices tensor (if return_indices is True)
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    in_dim1: tl.constexpr,
    in_dim2: tl.constexpr,
    in_dim3: tl.constexpr,
    out_dim1: tl.constexpr,
    out_dim2: tl.constexpr,
    out_dim3: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    return_indices: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data of size BLOCK_SIZE
    pid = tl.program_id(0)  # Global thread block ID
    total_elements = out_dim1 * out_dim2 * out_dim3 * channels * batch_size
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Calculate the 5D indices for the output tensor (batch, channel, out_dim1, out_dim2, out_dim3)
    out_idx = offsets
    out_batch = out_idx // (channels * out_dim1 * out_dim2 * out_dim3)
    out_idx = out_idx % (channels * out_dim1 * out_dim2 * out_dim3)
    out_channel = out_idx // (out_dim1 * out_dim2 * out_dim3)
    out_idx = out_idx % (out_dim1 * out_dim2 * out_dim3)
    out_d1 = out_idx // (out_dim2 * out_dim3)
    out_idx = out_idx % (out_dim2 * out_dim3)
    out_d2 = out_idx // out_dim3
    out_d3 = out_idx % out_dim3

    # Map output indices to input indices
    in_d1 = out_d1 * stride - padding
    in_d2 = out_d2 * stride - padding
    in_d3 = out_d3 * stride - padding

    # Initialize output and indices
    out_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    out_indices = tl.full((BLOCK_SIZE,), -1, dtype=tl.int32)

    # Iterate over kernel size with dilation
    for k1 in range(kernel_size):
        for k2 in range(kernel_size):
            for k3 in range(kernel_size):
                in_k1 = in_d1 + k1 * dilation
                in_k2 = in_d2 + k2 * dilation
                in_k3 = in_d3 + k3 * dilation

                # Check bounds
                valid = (in_k1 >= 0) & (in_k1 < in_dim1) & (in_k2 >= 0) & (in_k2 < in_dim2) & (in_k3 >= 0) & (in_k3 < in_dim3)

                # Calculate input index
                in_idx = (out_batch * channels + out_channel) * in_dim1 * in_dim2 * in_dim3 + \
                         in_k1 * in_dim2 * in_dim3 + \
                         in_k2 * in_dim3 + \
                         in_k3

                # Load input value
                val = tl.load(x_ptr + in_idx, mask=valid, other=float('-inf'))

                # Update max value and index
                new_mask = valid & (val > out_val)
                out_val = tl.where(new_mask, val, out_val)
                new_indices = tl.where(new_mask, in_idx, out_indices)
                out_indices = tl.where(new_mask, new_indices, out_indices)

    # Store output
    out_ptr_offset = (out_batch * channels + out_channel) * out_dim1 * out_dim2 * out_dim3 + \
                     out_d1 * out_dim2 * out_dim3 + \
                     out_d2 * out_dim3 + \
                     out_d3
    tl.store(out_ptr + out_ptr_offset, out_val, mask=mask)

    # Store indices if requested
    if return_indices:
        indices_ptr_offset = (out_batch * channels + out_channel) * out_dim1 * out_dim2 * out_dim3 + \
                             out_d1 * out_dim2 * out_dim3 + \
                             out_d2 * out_dim3 + \
                             out_d3
        tl.store(indices_ptr + indices_ptr_offset, out_indices, mask=mask)


def triton_max_pool3d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int, return_indices: bool = False) -> torch.Tensor:
    """
    Custom Triton-based 3D Max Pooling implementation.
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()

    # Dimensions
    batch_size, channels, in_dim1, in_dim2, in_dim3 = x.shape
    out_dim1 = (in_dim1 + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_dim2 = (in_dim2 + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_dim3 = (in_dim3 + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    # Output tensor
    out = torch.empty(batch_size, channels, out_dim1, out_dim2, out_dim3, dtype=x.dtype, device=x.device)

    # Indices tensor (if needed)
    indices = torch.empty(batch_size, channels, out_dim1, out_dim2, out_dim3, dtype=torch.int32, device=x.device) if return_indices else None

    # Grid size
    total_elements = out.numel()
    BLOCK_SIZE = 128  # Tunable parameter, power of 2
    grid = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    max_pool3d_kernel[grid](
        x, out, indices,
        batch_size, channels, in_dim1, in_dim2, in_dim3,
        out_dim1, out_dim2, out_dim3,
        kernel_size, stride, padding, dilation,
        return_indices,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out if not return_indices else (out, indices)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton kernel instead of PyTorch's MaxPool3d
        return triton_max_pool3d(
            x, self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices
        )


def get_inputs():
    x = torch.rand(16, 32, 128, 128, 128).cuda()
    return [x]

def get_init_inputs():
    return [3, 2, 1, 3]