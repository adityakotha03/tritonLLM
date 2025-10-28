import torch
import torch.nn as nn
import triton
import triton.language as tl


# Fused kernel for element-wise (x - value) and hardswish activation.
# This kernel is autotuned for different block sizes to find the most optimal one.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def subtract_hardswish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    subtract_value,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that fuses two element-wise operations:
    1. Subtraction of a scalar value.
    2. Hardswish activation.
    This reduces memory I/O by performing both operations in a single pass.
    """
    # Each program instance computes a block of outputs.
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the last block, which may not be full.
    mask = offsets < n_elements
    
    # Load a block of data from the input tensor.
    x = tl.load(x_ptr + offsets, mask=mask)

    # --- Fused Operation 1: Subtract ---
    x = x - subtract_value
    
    # --- Fused Operation 2: Hardswish ---
    # hardswish(x) = x * relu6(x + 3) / 6
    # relu6(y) = min(max(0, y), 6)
    x_plus_3 = x + 3.0
    relu_x_plus_3 = tl.maximum(0.0, x_plus_3)
    relu6_x_plus_3 = tl.minimum(relu_x_plus_3, 6.0)
    out = x * relu6_x_plus_3 / 6.0

    # Store the result block back to global memory.
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_subtract_hardswish(x: torch.Tensor, subtract_value: float):
    """
    Wrapper function for the subtract_hardswish_kernel.
    """
    out = torch.empty_like(x)
    assert x.is_contiguous() and out.is_contiguous()
    n_elements = x.numel()
    
    # The grid is 1D, with each program handling a block of elements.
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    subtract_hardswish_kernel[grid](x, out, n_elements, subtract_value)
    return out


# Fused kernel for MaxPool2d and Mish activation.
@triton.jit
def maxpool2d_mish_kernel(
    x_ptr,
    out_ptr,
    N, C, H, W,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oc, stride_oh, stride_ow,
    KERNEL_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
):
    """
    Triton kernel that fuses MaxPool2d and Mish activation.
    Each program instance computes a single output pixel. This naive implementation
    demonstrates fusion; for optimal performance, a tiled approach using shared
    memory would be better.
    """
    # The grid is 1D, flattened over all output pixels.
    pid = tl.program_id(axis=0)

    # Calculate the (n, c, h_out, w_out) coordinates from the program ID.
    w_out = pid % W_out
    h_out = (pid // W_out) % H_out
    c = (pid // (W_out * H_out)) % C
    n = pid // (W_out * H_out * C)

    # Determine the top-left corner of the pooling window in the input tensor.
    h_start = h_out * STRIDE
    w_start = w_out * STRIDE

    # Initialize max value to a very small number.
    max_val = -float('inf')

    # Iterate over the pooling window. This loop is unrolled by the JIT compiler.
    for kh in range(0, KERNEL_SIZE):
        for kw in range(0, KERNEL_SIZE):
            h_in = h_start + kh
            w_in = w_start + kw
            in_offset = n * stride_xn + c * stride_xc + h_in * stride_xh + w_in * stride_xw
            val = tl.load(x_ptr + in_offset)
            max_val = tl.maximum(max_val, val)

    # --- Fused Operation: Mish activation on the max value ---
    # mish(x) = x * tanh(softplus(x)) where softplus(x) = log(1 + exp(x))
    softplus_of_max = tl.log(1.0 + tl.exp(max_val))
    exp_2y = tl.exp(2.0 * softplus_of_max)
    tanh_of_softplus = (exp_2y - 1.0) / (exp_2y + 1.0)
    mish_val = max_val * tanh_of_softplus
    
    # Calculate output offset and store the final result.
    out_offset = n * stride_on + c * stride_oc + h_out * stride_oh + w_out * stride_ow
    tl.store(out_ptr + out_offset, mish_val)


def triton_maxpool2d_mish(x: torch.Tensor, kernel_size: int, stride: int):
    """
    Wrapper function for the maxpool2d_mish_kernel.
    NOTE: This implementation assumes padding=0 and that input dimensions are
    divisible by the stride.
    """
    N, C, H, W = x.shape
    
    # Calculate output dimensions.
    H_out = H // stride
    W_out = W // stride
    
    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    assert x.is_contiguous()
    out = out.contiguous()

    # The grid size is the total number of output pixels.
    grid = (N * C * H_out * W_out,)
    
    maxpool2d_mish_kernel[grid](
        x, out,
        N, C, H, W,
        H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        KERNEL_SIZE=kernel_size,
        STRIDE=stride
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels to fuse operations.
    - Fuses (subtract + hardswish).
    - Fuses (MaxPool2d + mish).
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        # The Conv2d layer is kept as is, since it's highly optimized in cuDNN.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        
        # Store pooling parameters for our custom Triton kernel.
        self.pool_kernel_size = pool_kernel_size
        # The original model's nn.MaxPool2d(k) implies a stride of k.
        self.pool_stride = pool_kernel_size

    def forward(self, x):
        # 1. Standard Convolution
        x = self.conv(x)
        
        # 2. Fused (subtract + hardswish)
        x = triton_subtract_hardswish(x, self.subtract_value)
        
        # 3. Fused (MaxPool2d + Mish)
        x = triton_maxpool2d_mish(x, self.pool_kernel_size, self.pool_stride)
        
        return x