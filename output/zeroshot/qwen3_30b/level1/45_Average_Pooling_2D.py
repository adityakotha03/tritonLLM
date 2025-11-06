import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    out_height,
    out_width,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
    TILE_HEIGHT: tl.constexpr,
    TILE_WIDTH: tl.constexpr,
):
    # Compute the output indices for this block
    pid = tl.program_id(0)
    pid_batch = pid // (out_height * out_width)
    pid_h = (pid % (out_height * out_width)) // out_width
    pid_w = (pid % (out_height * out_width)) % out_width

    # Compute the start position in the input tensor for this output position
    h_start = pid_h * stride - padding
    w_start = pid_w * stride - padding

    # Define the offsets for the current output position
    out_offset = pid_batch * channels * out_height * out_width + \
                 pid_batch * channels * out_height * out_width + \
                 pid_h * out_width + pid_w

    # Allocate shared memory for the current tile of the pooling window
    shared_x = tl.load(tl.make_block_ptr(x_ptr, shape=(1, 1, kernel_size, kernel_size),
                                         strides=(channels * height * width, 1, height * width, width),
                                         offsets=(pid_batch * channels, 0, h_start, w_start),
                                         block_shape=(1, 1, kernel_size, kernel_size),
                                         order=(0, 1, 2, 3)),
                       mask=(pid_batch < batch_size) & (tl.arange(0, kernel_size)[:, None] < kernel_size) & 
                            (tl.arange(0, kernel_size)[None, :] < kernel_size),
                       other=0.0)

    # Compute the sum over the kernel window
    sum_val = tl.zeros((1, 1), dtype=tl.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            h = h_start + i
            w = w_start + j
            valid = (h >= 0) & (h < height) & (w >= 0) & (w < width)
            val = tl.load(
                x_ptr + pid_batch * channels * height * width +
                tl.arange(0, channels)[:, None] * height * width +
                h * width + w,
                mask=valid & (tl.arange(0, channels)[:, None] < channels),
                other=0.0
            )
            sum_val += val

    # Compute the effective number of elements in the window (for averaging)
    count = 0.0
    for i in range(kernel_size):
        for j in range(kernel_size):
            h = h_start + i
            w = w_start + j
            if h >= 0 and h < height and w >= 0 and w < width:
                count += 1.0

    # Divide by the count
    avg_val = sum_val / count

    # Store the result
    tl.store(out_ptr + out_offset, avg_val, mask=(pid_batch < batch_size) & (pid_h < out_height) & (pid_w < out_width))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'TILE_HEIGHT': 16, 'TILE_WIDTH': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'TILE_HEIGHT': 16, 'TILE_WIDTH': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512, 'TILE_HEIGHT': 32, 'TILE_WIDTH': 32}, num_stages=3, num_warps=4),
    ],
    key=['batch_size', 'channels', 'height', 'width', 'out_height', 'out_width', 'kernel_size', 'stride', 'padding']
)
def triton_avg_pool(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    # Ensure input is on CUDA
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()

    # Extract shape
    batch_size, channels, height, width = x.shape
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1

    # Create output tensor
    out = torch.empty(batch_size, channels, out_height, out_width, dtype=x.dtype, device=x.device)

    # Define grid size: one block per output element
    n_output_elements = batch_size * channels * out_height * out_width
    grid = lambda meta: (n_output_elements // meta['BLOCK_SIZE'],)

    # Launch kernel
    avg_pool_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        out_height=out_height,
        out_width=out_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        BLOCK_SIZE=128,
        TILE_HEIGHT=16,
        TILE_WIDTH=16,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)


# Inputs and init inputs for testing (not part of the required output)
batch_size = 16
channels = 64
height = 2048
width = 2048
kernel_size = 11

def get_inputs():
    x = torch.rand(batch_size, channels, height, width, device='cuda')
    return [x]

def get_init_inputs():
    return [kernel_size]