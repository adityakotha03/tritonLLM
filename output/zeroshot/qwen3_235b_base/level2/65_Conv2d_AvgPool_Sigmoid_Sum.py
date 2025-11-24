import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Sigmoid: 1 / (1 + exp(-x))
    exp_neg_x = tl.exp(-x)
    sigmoid = 1.0 / (1.0 + exp_neg_x)
    tl.store(out_ptr + offsets, sigmoid, mask=mask)


def triton_sigmoid(x):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


@triton.jit
def _sum_kernel(x_ptr, out_ptr, n_rows, row_size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_offset < n_rows
    output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for col in range(0, row_size):
        col_offset = col
        data_offset = row_offset * row_size + col_offset
        mask = row_mask
        x = tl.load(x_ptr + data_offset, mask=mask, other=0.0)
        output += x
    tl.store(out_ptr + row_offset, output, mask=row_mask)


def triton_sum(x, dim):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    dim = [d % x.ndim for d in dim]
    # We sum over last contiguous dims
    if dim != list(range(x.ndim - len(dim), x.ndim)):
        permute_dims = [i for i in range(x.ndim) if i not in dim] + dim
        x = x.permute(permute_dims)
    shape_before = x.shape[:-len(dim)]
    x = x.reshape(-1, x.shape[-1])
    n_rows, row_size = x.shape
    out = torch.zeros(n_rows, device=x.device, dtype=x.dtype)
    grid = lambda meta: ((n_rows + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _sum_kernel[grid](x, out, n_rows, row_size, BLOCK_SIZE=128)
    return out.reshape(*shape_before)


# Fusion of Conv2d + AvgPool2d + Sigmoid using Triton
# We do not implement full Conv2d in Triton here due to complexity,
# but we can fuse AvgPool2d + Sigmoid after PyTorch Conv2d, or replace Conv+AvgPool with a strided conv-like avg pool
# However, since the model applies Conv2d -> AvgPool2d -> Sigmoid -> Sum,
# we can fuse AvgPool2d + Sigmoid + Sum into one kernel for significant memory bandwidth savings.

@triton.jit
def _avg_pool_sigmoid_sum_kernel(
    x_ptr, out_ptr,
    input_n, input_h, input_w, input_c,
    output_h, output_w,
    pool_h, pool_w,
    scale,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr
):
    # Each block handles a subset of N and HW
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    # Compute block ranges
    n_offset = pid_n * BLOCK_SIZE_N
    hw_offset = pid_hw * BLOCK_SIZE_HW

    n_range = tl.arange(0, BLOCK_SIZE_N)
    hw_range = hw_offset + tl.arange(0, BLOCK_SIZE_HW)

    n_mask = n_offset + n_range < input_n
    hw_mask = hw_range < output_h * output_w
    n_idx = n_offset + n_range[:, None]
    hw_idx = hw_range[None, :]

    # Convert hw_idx to h and w
    out_h_idx = hw_idx // output_w
    out_w_idx = hw_idx % output_w

    # Input region start
    in_h_start = out_h_idx * pool_h
    in_w_start = out_w_idx * pool_w

    # Initialize sum
    pool_sum = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_HW), dtype=tl.float32)

    # Iterate over pool kernel
    for ph in range(0, pool_h):
        for pw in range(0, pool_w):
            h_idx = in_h_start + ph
            w_idx = in_w_start + pw
            # Bounds check
            h_valid = (h_idx < input_h)
            w_valid = (w_idx < input_w)
            valid = h_valid & w_valid
            # Compute input offset
            in_offset = n_idx * input_h * input_w * input_c + \
                        h_idx * input_w * input_c + \
                        w_idx * input_c
            # Load values (C is contiguous)
            for c in range(0, input_c):
                mask = n_mask[:, None] & valid & (c < input_c)
                x = tl.load(x_ptr + in_offset + c, mask=mask, other=0.0)
                pool_sum += x

    # Average
    pool_avg = pool_sum * scale

    # Sigmoid
    sigmoid = 1.0 / (1.0 + tl.exp(-pool_avg))

    # Sum over spatial and channel dimensions
    # Here we sum over output_h * output_w * input_c implicitly
    total_sum = tl.sum(sigmoid, axis=1)  # Sum over HW dim

    # Store result
    tl.store(out_ptr + n_idx[:, 0], total_sum, mask=n_mask)


def fused_avg_pool_sigmoid_sum(x, pool_size, dim):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    input_n, input_c, input_h, input_w = x.shape
    pool_h, pool_w = pool_size, pool_size
    output_h, output_w = input_h // pool_h, input_w // pool_w
    scale = 1.0 / (pool_h * pool_w)

    out = torch.zeros(input_n, device=x.device, dtype=x.dtype)

    # Transpose to N H W C for better memory access in kernel
    x = x.permute(0, 2, 3, 1).contiguous()

    grid = (triton.cdiv(input_n, 16), triton.cdiv(output_h * output_w, 512))
    _avg_pool_sigmoid_sum_kernel[grid](
        x, out,
        input_n, input_h, input_w, input_c,
        output_h, output_w,
        pool_h, pool_w,
        scale,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_HW=512
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model with fused AvgPool + Sigmoid + Sum using Triton.
    Conv2d is kept as PyTorch op for simplicity and efficiency (uses cuDNN),
    but the rest are fused into a single Triton kernel to reduce memory traffic.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        x = fused_avg_pool_sigmoid_sum(x, self.pool_kernel_size, dim=[1,2,3])
        return x