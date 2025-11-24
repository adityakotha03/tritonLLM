import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # GELU approximation using tanh method
    x_sq = x * x
    x_cube = x_sq * x
    inner = 0.044715 * x_cube + x
    tanh_inner = tl.tanh(0.79788456 * inner)
    gelu = 0.5 * x * (1.0 + tanh_inner)

    tl.store(out_ptr + offsets, gelu, mask=mask)


def triton_gelu(x):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    BLOCK_SIZE = 1024
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def conv_2d_nhwc_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    batch, height, width, in_channels,
    out_channels, kernel_h, kernel_w,
    out_height, out_width,
    stride_h, stride_w,
    pad_h, pad_w,
    load_K: tl.constexpr,  # Tiling size for in_channels
    store_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    # Compute program ids
    pid_batch = tl.program_id(axis=0)
    pid_out_h = tl.program_id(axis=1)
    pid_out_w = tl.program_id(axis=2)
    pid_oc = tl.program_id(axis=3)

    # Calculate output spatial indices
    out_h_offset = pid_out_h * BLOCK_M
    out_w_offset = pid_out_w * BLOCK_N

    # Pointers to output
    output_offsets = (
        pid_batch * out_channels * out_height * out_width +
        pid_oc * out_height * out_width +
        out_h_offset * out_width + out_w_offset
    )
    output_mask = (
        (tl.arange(0, BLOCK_M)[:, None] < out_height - out_h_offset) &
        (tl.arange(0, BLOCK_N)[None, :] < out_width - out_w_offset)
    )
    output_ptrs = out_ptr + output_offsets + tl.arange(0, BLOCK_M)[:, None] * out_width + tl.arange(0, BLOCK_N)[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over input channels in tiles
    for ic in range(0, in_channels, load_K):
        current_load_K = min(load_K, in_channels - ic)

        # Load input tile (BLOCK_M x BLOCK_N x load_K) with padding
        input_tiles = []
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                h_offset = pid_out_h * stride_h * BLOCK_M + kh - pad_h
                w_offset = pid_out_w * stride_w * BLOCK_N + kw - pad_w
                h_mask = (h_offset >= 0) & (h_offset < height) & (tl.arange(0, BLOCK_M)[:, None] < out_height)
                w_mask = (w_offset >= 0) & (w_offset < width) & (tl.arange(0, BLOCK_N)[None, :] < out_width)
                global_h = h_offset
                global_w = w_offset
                input_ptrs = x_ptr + \
                    pid_batch * height * width * in_channels + \
                    global_h * width * in_channels + \
                    global_w * in_channels + \
                    ic + tl.arange(0, BLOCK_M)[:, None] * width * in_channels + tl.arange(0, BLOCK_N)[None, :] * in_channels
                input_tile = tl.load(
                    input_ptrs,
                    mask=h_mask & w_mask & (tl.arange(0, current_load_K)[None, None, :] < in_channels - ic),
                    other=0.0
                )
                input_tiles.append(input_tile)
        
        # Stack and reshape input tiles
        input_tile_cat = tl.concatenate([tl.expand_dims(t, 2) for t in input_tiles], 2)  # (BLOCK_M, BLOCK_N, K*K)
        input_tile_cat = tl.reshape(input_tile_cat, (BLOCK_M * BLOCK_N, kernel_h * kernel_w))

        # Load weights (out_channels, in_channels, k, k) -> (oc, ic, k*k)
        weight_ptrs = w_ptr + \
            pid_oc * in_channels * kernel_h * kernel_w + \
            ic * kernel_h * kernel_w + tl.arange(0, current_load_K)[:, None] * kernel_h * kernel_w + tl.arange(0, kernel_h * kernel_w)[None, :]
        weight_mask = (tl.arange(0, current_load_K)[:, None] < in_channels - ic) & (tl.arange(0, kernel_h * kernel_w)[None, :] < kernel_h * kernel_w)
        weight_tile = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        weight_tile = tl.reshape(weight_tile, (current_load_K, kernel_h * kernel_w))
        weight_tile = tl.trans(weight_tile)  # (k*k, current_load_K)

        # Compute: (BLOCK_M*BLOCK_N, k*k) @ (k*k, current_load_K) -> (BLOCK_M*BLOCK_N, current_load_K)
        input_weight_dot = tl.dot(input_tile_cat.to(tl.float16), weight_tile.to(tl.float16), out_dtype=tl.float32)
        input_weight_dot = tl.reshape(input_weight_dot, (BLOCK_M, BLOCK_N, current_load_K))

        # Sum over spatial kernel dims
        acc += tl.sum(input_weight_dot, axis=2)

    # Add bias if needed
    if USE_BIAS:
        bias = tl.load(bias_ptr + pid_oc)
        acc += bias

    # Store output
    tl.store(output_ptrs, acc, mask=output_mask)


def triton_conv2d_nhwc(x, weight, bias=None, stride=1, padding=1, dilation=1):
    assert x.is_cuda and weight.is_cuda
    if bias is not None:
        assert bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch, height, width, in_channels = x.shape
    out_channels, in_c, kernel_h, kernel_w = weight.shape
    assert in_channels == in_c

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    out_height = (height + 2 * pad_h - dilation * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (width + 2 * pad_w - dilation * (kernel_w - 1) - 1) // stride_w + 1

    out = torch.empty((batch, out_height, out_width, out_channels), device=x.device, dtype=x.dtype)

    # Grid: (batch, grid_h, grid_w, out_channels)
    grid = (
        batch,
        triton.cdiv(out_height, 16),
        triton.cdiv(out_width, 16),
        out_channels
    )

    # Launch kernel
    conv_2d_nhwc_kernel[grid](
        x, weight, bias, out,
        batch, height, width, in_channels,
        out_channels, kernel_h, kernel_w,
        out_height, out_width,
        stride_h, stride_w,
        pad_h, pad_w,
        load_K=16,
        store_K=16,
        BLOCK_M=16,
        BLOCK_N=16,
        USE_BIAS=bias is not None
    )
    return out


@triton.jit
def global_avg_pool_2d_kernel(
    x_ptr,
    out_ptr,
    batch,
    height,
    width,
    channels,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_ch = tl.program_id(1)

    # Batch and channel offsets
    batch_offset = pid_batch * BLOCK_SIZE_BATCH
    ch_offset = pid_ch * BLOCK_SIZE_HW

    # Loop over batch block
    for b in range(BLOCK_SIZE_BATCH):
        curr_batch = batch_offset + b
        if curr_batch >= batch:
            continue

        # Loop over channel block
        for c in range(BLOCK_SIZE_HW):
            curr_ch = ch_offset + c
            if curr_ch >= channels:
                continue

            # Compute sum over H and W
            acc = tl.zeros((1,), dtype=tl.float32)
            for h in range(height):
                for w in range(width):
                    offset = curr_batch * height * width * channels + h * width * channels + w * channels + curr_ch
                    val = tl.load(x_ptr + offset)
                    acc += val.to(tl.float32)

            # Store average
            avg = acc / (height * width)
            out_offset = curr_batch * channels + curr_ch
            tl.store(out_ptr + out_offset, avg)


def triton_global_avg_pool2d_nhwc(x):
    assert x.is_cuda
    x = x.contiguous()
    batch, height, width, channels = x.shape
    out = torch.empty((batch, channels), device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(batch, 8), triton.cdiv(channels, 16))
    global_avg_pool_2d_kernel[grid](
        x, out,
        batch, height, width, channels,
        BLOCK_SIZE_BATCH=8,
        BLOCK_SIZE_HW=16
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Conv2d, GELU, and Global Average Pooling.
    Input is converted to NHWC for better memory access.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Initialize weights and bias manually
        k = 1.0 / in_channels / kernel_size / kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * k**0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        batch, in_ch, height, width = x.shape

        # Convert from NCHW to NHWC
        x = x.permute(0, 2, 3, 1).contiguous()

        # Conv2d using Triton kernel
        x = triton_conv2d_nhwc(x, self.weight, self.bias, stride=1, padding=self.kernel_size//2)

        # GELU activation
        x = triton_gelu(x)

        # Global average pooling
        x = triton_global_avg_pool2d_nhwc(x)

        return x