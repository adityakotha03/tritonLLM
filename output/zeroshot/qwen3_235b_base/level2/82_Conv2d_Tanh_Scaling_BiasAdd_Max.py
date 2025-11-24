import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_tanh_scale_bias_kernel(
    x_ptr,          # pointer to input image (NHWC: batch, height, width, in_channels)
    weight_ptr,     # pointer to weights (out_channels, kernel, kernel, in_channels)
    bias_ptr,       # pointer to bias (out_channels,)
    out_ptr,        # pointer to output (after tanh, scale, bias)
    scaling_factor, # scalar scaling factor
    stride,         # stride of conv
    padding,        # padding
    dilation,       # dilation
    batch,          # N
    out_h,          # output height
    out_w,          # output width
    in_c,           # input channels
    out_c,          # output channels
    k_h,            # kernel height
    k_w,            # kernel width
    inp_h,          # input height
    inp_w,          # input width
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # 2D block ID for output spatial location
    pid = tl.program_id(0)
    batch_id = pid // (out_h * out_w)
    hw_id = pid % (out_h * out_w)
    out_y = hw_id // out_w
    out_x = hw_id % out_w

    # Pointers into this batch
    x_batch_ptr = x_ptr + batch_id * inp_h * inp_w * in_c
    out_batch_ptr = out_ptr + batch_id * out_h * out_w * out_c

    # Compute input region for this output location
    # Use dilation and stride
    center_y = out_y * stride - padding + dilation * (k_h // 2)
    center_x = out_x * stride - padding + dilation * (k_w // 2)

    # Load bias once per output channel
    bias_offsets = tl.arange(0, BLOCK_SIZE_N)
    bias_mask = bias_offsets < out_c
    bias = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0)

    # Iterate over input channels in blocks
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for ic in range(0, in_c, BLOCK_SIZE_K):
        # Load input patch (k_h, k_w, block_k)
        for ky in range(k_h):
            for kx in range(k_w):
                in_y = center_y + ky * dilation
                in_x = center_x + kx * dilation
                # Bounds check
                in_y_valid = (in_y >= 0) and (in_y < inp_h)
                in_x_valid = (in_x >= 0) and (in_x < inp_w)
                valid = in_y_valid and in_x_valid
                if valid:
                    input_offset = in_y * inp_w * in_c + in_x * in_c + ic
                    x_ptrs = x_batch_ptr + input_offset + tl.arange(0, BLOCK_SIZE_K)
                    mask = (ic + tl.arange(0, BLOCK_SIZE_K)) < in_c
                    x = tl.load(x_ptrs, mask=mask, other=0.0)
                else:
                    x = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)

                # Load weights (out_c, ky, kx, ic:ic+block_k)
                w_offset = (0 + ky * k_w * in_c * out_c + kx * in_c * out_c + ic * out_c)
                w_ptrs = weight_ptr + w_offset + bias_offsets[:, None] * 1 + tl.arange(0, BLOCK_SIZE_K)[None, :] * out_c
                w_mask = bias_mask[:, None] & ((ic + tl.arange(0, BLOCK_SIZE_K)[None, :]) < in_c)
                w = tl.load(w_ptrs, mask=w_mask, other=0.0)

                # Accumulate GEMM
                acc += tl.dot(x[None, :], w, out_dtype=tl.float32)

        # End of kernel loop

    # Add bias and apply tanh + scale
    acc = acc + bias
    acc = tl.tanh(acc)
    acc = acc * scaling_factor

    # Store output
    out_offset = out_y * out_w * out_c + out_x * out_c
    out_ptrs = out_batch_ptr + out_offset + bias_offsets
    out_mask = bias_mask
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def maxpool_kernel(
    x_ptr,       # input pointer (NHWC)
    out_ptr,     # output pointer (NHWC)
    batch,       # N
    inp_h,       # input height
    inp_w,       # input width
    out_c,       # channels
    pool_size,   # pooling kernel size (assume square)
    out_h,       # output height
    out_w,       # output width
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // (out_h * out_w)
    hw_id = pid % (out_h * out_w)
    out_y = hw_id // out_w
    out_x = hw_id % out_w

    # Input spatial start
    start_y = out_y * pool_size
    start_x = out_x * pool_size

    # Pointers
    x_batch_ptr = x_ptr + batch_id * inp_h * inp_w * out_c
    out_batch_ptr = out_ptr + batch_id * out_h * out_w * out_c

    # Initialize max value
    max_val = tl.full((BLOCK_SIZE_C,), value=float("-inf"), dtype=tl.float32)

    for py in range(pool_size):
        for px in range(pool_size):
            cur_y = start_y + py
            cur_x = start_x + px
            valid = (cur_y < inp_h) & (cur_x < inp_w)
            if valid:
                offset = cur_y * inp_w * out_c + cur_x * out_c + tl.arange(0, BLOCK_SIZE_C)
                mask = tl.arange(0, BLOCK_SIZE_C) < out_c
                x = tl.load(x_batch_ptr + offset, mask=mask, other=float("-inf"))
                max_val = tl.maximum(max_val, x)

    # Store result
    out_offset = out_y * out_w * out_c + out_x * out_c + tl.arange(0, BLOCK_SIZE_C)
    out_mask = tl.arange(0, BLOCK_SIZE_C) < out_c
    tl.store(out_batch_ptr + out_offset, max_val, mask=out_mask)


class ModelNew(nn.Module):
    """
    Optimized version of Model using fused Triton kernels for conv+tanh*scale+bias and maxpool.
    Input is expected in NHWC format for better memory coalescing in Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        # Standard Conv2d for weight initialization and access
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size

        # Store shapes for kernel launch
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        # Convert to NHWC for better Triton performance
        x = x.to(memory_format=torch.channels_last)
        x = x.contiguous()

        batch, _, height, width = x.shape

        # Conv output spatial size
        pad = self.kernel_size // 2
        stride = 1
        dilation = 1
        out_h = (height + 2 * pad - dilation * (self.kernel_size - 1) - 1) // stride + 1
        out_w = (width + 2 * pad - dilation * (self.kernel_size - 1) - 1) // stride + 1

        # Output after conv+tanh*scale+bias
        out = torch.empty((batch, out_h, out_w, self.out_channels), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)

        # Launch fused conv + tanh + scale + bias
        # Weight in (out_c, in_c, k, k) -> rearrange to (out_c, k, k, in_c)
        weight = self.conv.weight.permute(0, 2, 3, 1).contiguous()
        # Bias: (out_c, 1, 1) -> (out_c,)
        bias = self.bias.view(-1)

        grid = lambda meta: (batch * out_h * out_w,)

        # Heuristic block sizes
        BLOCK_SIZE_M = 1
        BLOCK_SIZE_N = triton.next_power_of_2(self.out_channels)
        BLOCK_SIZE_K = triton.next_power_of_2(self.in_channels)

        conv_tanh_scale_bias_kernel[grid](
            x, weight, bias, out,
            self.scaling_factor,
            stride, pad, dilation,
            batch, out_h, out_w,
            self.in_channels, self.out_channels,
            self.kernel_size, self.kernel_size,
            height, width,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )

        # Now apply max pooling
        pool_out_h = out_h // self.pool_kernel_size
        pool_out_w = out_w // self.pool_kernel_size
        pool_out = torch.empty((batch, pool_out_h, pool_out_w, self.out_channels), device=out.device, dtype=out.dtype, memory_format=torch.channels_last)

        pool_grid = lambda meta: (batch * pool_out_h * pool_out_w,)

        maxpool_kernel[pool_grid](
            out, pool_out,
            batch, out_h, out_w, self.out_channels,
            self.pool_kernel_size,
            pool_out_h, pool_out_w,
            BLOCK_SIZE_C=triton.next_power_of_2(self.out_channels),
        )

        # Convert back to NCHW if needed (but keep NHWC for downstream ops)
        pool_out = pool_out.permute(0, 3, 1, 2).contiguous()
        return pool_out