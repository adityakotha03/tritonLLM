import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hardswish_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # ReLU(x) * clamp((x + 3) / 6, 0, 1) -> HardSwish
    x_relu = tl.where(x > 0, x, 0.0)
    shift = x + 3.0
    clamp_low = tl.where(shift < 0, 0.0, shift)
    clamp_val = tl.where(clamp_low > 6, 6.0, clamp_low)
    div = clamp_val / 6.0
    out = x_relu * div
    tl.store(output_ptr + offsets, out, mask=mask)


def triton_hardswish(x):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    hardswish_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


@triton.jit
def conv2d_relu_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    out_channels,
    out_height,
    out_width,
    in_channels,
    kernel_size,
    in_height,
    in_width,
    stride,
    padding,
    load_C: tl.constexpr,
    store_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(out_channels, BLOCK_M)
    num_blocks_n = tl.cdiv(out_height * out_width, BLOCK_N)
    m_block_id = pid // num_blocks_n
    n_block_id = pid % num_blocks_n

    # Pointers for output tile
    offs_m = m_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n_h = (offs_n // out_width) % out_height
    offs_n_w = offs_n % out_width

    # Input spatial indices
    h_start = offs_n_h * stride - padding
    w_start = offs_n_w * stride - padding
    ih = h_start[:, None] + tl.arange(0, kernel_size)[None, :]
    iw = w_start[:, None] + tl.arange(0, kernel_size)[None, :]
    valid_hw = (ih >= 0) & (iw >= 0) & (ih < in_height) & (iw < in_width)

    # Initialize pointers for weight and input
    w_ptrs = weight_ptr + offs_m[:, None] * (in_channels * kernel_size * kernel_size) + \
             tl.arange(0, in_channels)[None, :] * (kernel_size * kernel_size) + \
             tl.arange(0, kernel_size * kernel_size)[None, :]
    mask_w = (offs_m < out_channels)[:, None] & (tl.arange(0, in_channels)[None, :] < in_channels)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for ic in range(0, tl.cdiv(in_channels, BLOCK_K)):
        # Input block pointer
        k_block_start = ic * BLOCK_K
        offs_k = k_block_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < in_channels

        # Load input tile: [BLOCK_K, kernel_size, kernel_size, BATCH, out_H, out_W]
        input_batch_offsets = offs_k[:, None, None, None] * in_height * in_width * 1 + \
                              ih[None, :, :, None] * in_width + iw[None, :, :, None]
        input_mask = mask_k[:, None, None, None] & valid_hw[None, :, :, None]
        input_ptrs = input_ptr + input_batch_offsets
        x = tl.load(input_ptrs, mask=input_mask, other=0.0)  # Shape: (BLOCK_K, kernel_size, kernel_size, BATCH, H, W)
        x = x.permute(3, 0, 1, 2, 4)  # Reshape to (BATCH, BLOCK_K, kernel_size, kernel_size, H, W) -> not directly supported

        # Instead, we do grouped load over spatial and channel dims
        # We reshape spatial convolution into GEMM via implicit loops
        # Here we do direct tiling: (BLOCK_M, BLOCK_K*K*K) x (BLOCK_K*K*K, BLOCK_N)
        # We loop over input channels and spatial kernel

        # Reshape weight: (out_channels, in_channels, k, k) -> (out_channels, in_channels * k * k)
        w = tl.load(w_ptrs + offs_k[None, :] * (kernel_size * kernel_size), mask=mask_w & mask_k[None, :], other=0.0)

        # Reshape x: (BLOCK_K, K, K, BATCH, H, W) -> (BLOCK_K * K * K, BLOCK_N)
        x_flat = tl.reshape(x, (BLOCK_K * kernel_size * kernel_size, BLOCK_N))

        # GEMM step
        acc += tl.dot(w, x_flat.to(tl.float32), out_dtype=tl.float32)

    # Add bias
    bias = tl.load(bias_ptr + offs_m, mask=offs_m < out_channels, other=0.0)
    acc = acc + bias[:, None]

    # ReLU
    acc = tl.where(acc > 0, acc, 0.0)

    # Store result
    output_offsets = m_block_id * BLOCK_M * store_C + n_block_id * BLOCK_N
    output_ptrs = output_ptr + output_offsets + \
                  offs_m[:, None] * store_C + tl.arange(0, BLOCK_N)[None, :]
    mask_o = (offs_m < out_channels)[:, None] & (tl.arange(0, BLOCK_N)[None, :] < (out_height * out_width))
    tl.store(output_ptrs, acc, mask=mask_o)


def triton_conv2d_relu(x, weight, bias, stride=1, padding=1):
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    out_height = (in_height + 2 * padding - kernel_size) // stride + 1
    out_width = (in_width + 2 * padding - kernel_size) // stride + 1

    output = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    output = output.contiguous()

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Reshape output to 2D: (out_channels, out_height * out_width)
    output_reshaped = output.view(out_channels, -1)

    # Launch kernel
    def grid(meta):
        return (triton.cdiv(out_channels, meta['BLOCK_M']) * triton.cdiv(out_height * out_width, meta['BLOCK_N']),)

    # Use autotuning
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        ],
        key=['out_channels', 'in_channels', 'kernel_size', 'in_height', 'in_width'],
    )
    @triton.jit
    def _conv2d_relu_kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        bias_ptr,
        batch_size,
        out_channels,
        out_height,
        out_width,
        in_channels,
        kernel_size,
        in_height,
        in_width,
        stride,
        padding,
        load_C: tl.constexpr,
        store_C: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_blocks_m = tl.cdiv(out_channels, BLOCK_M)
        num_blocks_n = tl.cdiv(out_height * out_width, BLOCK_N)
        m_block_id = pid // num_blocks_n
        n_block_id = pid % num_blocks_n

        offs_m = m_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = n_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_n_h = (offs_n // out_width) % out_height
        offs_n_w = offs_n % out_width

        h_start = offs_n_h * stride - padding
        w_start = offs_n_w * stride - padding
        ih = h_start[:, None] + tl.arange(0, kernel_size)[None, :]
        iw = w_start[:, None] + tl.arange(0, kernel_size)[None, :]
        valid_hw = (ih >= 0) & (iw >= 0) & (ih < in_height) & (iw < in_width)

        w_ptrs = weight_ptr + offs_m[:, None] * (in_channels * kernel_size * kernel_size) + \
                 tl.arange(0, in_channels)[None, :] * (kernel_size * kernel_size) + \
                 tl.arange(0, kernel_size * kernel_size)[None, :]

        mask_w = (offs_m < out_channels)[:, None] & (tl.arange(0, in_channels)[None, :] < in_channels)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for ic in range(0, tl.cdiv(in_channels, BLOCK_K)):
            k_block_start = ic * BLOCK_K
            offs_k = k_block_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < in_channels

            input_batch_offsets = offs_k[:, None, None, None] * in_height * in_width + \
                                  ih[None, :, :, None] * in_width + iw[None, :, :, None]
            input_mask = mask_k[:, None, None, None] & valid_hw[None, :, :, None]
            input_ptrs = input_ptr + input_batch_offsets
            x = tl.load(input_ptrs, mask=input_mask, other=0.0)

            # Reshape x: (BLOCK_K, kernel_size, kernel_size, BLOCK_N) -> (BLOCK_K * kernel_size * kernel_size, BLOCK_N)
            x_flat = tl.reshape(x, (BLOCK_K * kernel_size * kernel_size, BLOCK_N))

            # Load weights: (BLOCK_M, BLOCK_K, kernel_size, kernel_size) -> (BLOCK_M, BLOCK_K * kernel_size * kernel_size)
            w = tl.load(w_ptrs + offs_k[None, :] * (kernel_size * kernel_size), mask=mask_w & mask_k[None, :], other=0.0)
            w = tl.reshape(w, (BLOCK_M, BLOCK_K * kernel_size * kernel_size))

            acc += tl.dot(w, x_flat.to(tl.float32), out_dtype=tl.float32)

        bias = tl.load(bias_ptr + offs_m, mask=offs_m < out_channels, other=0.0)
        acc = acc + bias[:, None]
        acc = tl.where(acc > 0, acc, 0.0)

        output_offsets = m_block_id * BLOCK_M * out_height * out_width + n_block_id * BLOCK_N
        output_ptrs = output_ptr + output_offsets + \
                      offs_m[:, None] * out_height * out_width + tl.arange(0, BLOCK_N)[None, :]
        mask_o = (offs_m < out_channels)[:, None] & (tl.arange(0, BLOCK_N)[None, :] < (out_height * out_width))
        tl.store(output_ptrs, acc, mask=mask_o)

    # Launch fused kernel
    load_C = in_height * in_width
    store_C = out_height * out_width
    _conv2d_relu_kernel[grid](
        x,
        weight,
        output_reshaped,
        bias,
        batch_size,
        out_channels,
        out_height,
        out_width,
        in_channels,
        kernel_size,
        in_height,
        in_width,
        stride,
        padding,
        load_C,
        store_C,
    )

    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.bias)

    def forward(self, x):
        x = triton_conv2d_relu(x, self.weight, self.bias, stride=1, padding=1)
        x = triton_hardswish(x)
        return x