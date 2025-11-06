import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool3d_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    x_stride_0, x_stride_1, x_stride_2, x_stride_3, x_stride_4,  # Strides for input (B, C, D, H, W)
    out_stride_0, out_stride_1, out_stride_2, out_stride_3, out_stride_4,  # Strides for output
    batch_size, channels, in_depth, in_height, in_width,
    out_depth, out_height, out_width,
    kernel_size, stride, padding,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
):
    # Thread block indices
    pid_b = tl.program_id(0)  # Batch dimension
    pid_c = tl.program_id(1)  # Channel dimension
    pid_d = tl.program_id(2)  # Output depth
    pid_h = tl.program_id(3)  # Output height
    pid_w = tl.program_id(4)  # Output width

    # Calculate output indices
    out_d = pid_d * stride + tl.arange(0, BLOCK_SIZE_D)
    out_h = pid_h * stride + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * stride + tl.arange(0, BLOCK_SIZE_W)

    # Define mask to limit access to valid output region
    mask_d = out_d < out_depth
    mask_h = out_h < out_height
    mask_w = out_w < out_width
    mask_ohw = mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]

    # Compute input indices for the pooling region
    in_d = out_d * stride - padding
    in_h = out_h * stride - padding
    in_w = out_w * stride - padding

    # Compute valid input indices
    valid_d = (in_d >= 0) & (in_d + kernel_size <= in_depth)
    valid_h = (in_h >= 0) & (in_h + kernel_size <= in_height)
    valid_w = (in_w >= 0) & (in_w + kernel_size <= in_width)
    valid_region = valid_d[:, None, None] & valid_h[None, :, None] & valid_w[None, None, :]

    # Combine valid output and valid input region
    mask = mask_ohw & valid_region

    # Calculate the number of elements to sum in the kernel
    num_elements = kernel_size * kernel_size * kernel_size
    num_elements = tl.load(tl.make_block_ptr(x_ptr, shape=(1,), strides=(0,), offsets=(0,), block_shape=(1,), order=(0,)))

    # Shared memory for reduction: we use it to reduce over spatial dims
    # We will use shared memory to store partial sums across tiles
    shmem = tl.load(tl.make_block_ptr(x_ptr, shape=(1,), strides=(0,), offsets=(0,), block_shape=(1,), order=(0,)))

    # Load input data in a tiled manner
    # We'll iterate over kernel_size slices in depth, height, width
    sum_vals = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Loop over kernel spatial dimensions
    for k_d in range(0, kernel_size):
        for k_h in range(0, kernel_size):
            for k_w in range(0, kernel_size):
                # Compute input spatial indices for this kernel tile
                in_d = out_d * stride - padding + k_d
                in_h = out_h * stride - padding + k_h
                in_w = out_w * stride - padding + k_w

                # Create valid mask for this input index
                valid_d = (in_d >= 0) & (in_d < in_depth)
                valid_h = (in_h >= 0) & (in_h < in_height)
                valid_w = (in_w >= 0) & (in_w < in_width)
                valid = valid_d[:, None, None] & valid_h[None, :, None] & valid_w[None, None, :]

                # Compute input offset
                offsets_d = in_d * x_stride_2
                offsets_h = in_h * x_stride_3
                offsets_w = in_w * x_stride_4
                offsets = offsets_d[:, None, None] + offsets_h[None, :, None] + offsets_w[None, None, :]

                # Add batch and channel offsets
                base_offset = (pid_b * x_stride_0 + pid_c * x_stride_1 + offsets)
                # Load input data with masking
                x_vals = tl.load(x_ptr + base_offset, mask=valid, other=0.0)

                # Accumulate into sum_vals
                sum_vals += x_vals

    # Now reduce across kernel size
    sum_vals = tl.sum(sum_vals, axis=(0, 1, 2))

    # Compute output index
    out_offsets = (pid_b * out_stride_0 + pid_c * out_stride_1 +
                   pid_d * out_stride_2 + pid_h * out_stride_3 + pid_w * out_stride_4)

    # Store output
    tl.store(out_ptr + out_offsets, sum_vals / num_elements, mask=mask_ohw)


def triton_avg_pool3d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA device."
    x = x.contiguous()
    batch_size, channels, in_depth, in_height, in_width = x.shape

    # Compute output shape
    out_depth = (in_depth + 2 * padding - kernel_size) // stride + 1
    out_height = (in_height + 2 * padding - kernel_size) // stride + 1
    out_width = (in_width + 2 * padding - kernel_size) // stride + 1

    # Output tensor
    out = torch.empty(batch_size, channels, out_depth, out_height, out_width, dtype=x.dtype, device=x.device)

    # Compute strides
    x_stride_0, x_stride_1, x_stride_2, x_stride_3, x_stride_4 = x.stride()
    out_stride_0, out_stride_1, out_stride_2, out_stride_3, out_stride_4 = out.stride()

    # Block sizes (optimized for A100: warp-friendly, shared memory utilization)
    BLOCK_SIZE_D = 8
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 32
    BLOCK_SIZE_B = 16

    # Grid dimensions: (batch, channels, out_depth, out_height, out_width)
    grid = (
        (batch_size + BLOCK_SIZE_B - 1) // BLOCK_SIZE_B,
        (channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C,
        (out_depth + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D,
        (out_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (out_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,
    )

    # Launch kernel
    avg_pool3d_kernel[grid](
        x,
        out,
        x_stride_0, x_stride_1, x_stride_2, x_stride_3, x_stride_4,
        out_stride_0, out_stride_1, out_stride_2, out_stride_3, out_stride_4,
        batch_size, channels, in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        kernel_size, stride, padding,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)