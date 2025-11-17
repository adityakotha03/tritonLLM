# Best Kernel for 1_Conv2D_ReLU_BiasAdd
# Generated: 20251116_181053
# Speedup: 1.12x
# Runtime: 6.5200 ms
# Round: 0
# Idea: Given the PyTorch code, replace the operation with a custom Triton kernel

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_PIX': 64, 'BLOCK_CO': 64, 'K_TILE': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_PIX': 128, 'BLOCK_CO': 64, 'K_TILE': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_PIX': 64, 'BLOCK_CO': 128, 'K_TILE': 32}, num_warps=8, num_stages=2),
    ],
    key=['N', 'C_IN', 'H', 'W', 'C_OUT', 'K', 'H_OUT', 'W_OUT'],
)
@triton.jit
def conv2d_relu_addbias_kernel(
    x_ptr,                          # *float* [N, C_IN, H, W]
    w2d_ptr,                        # *float* [K_TOTAL, C_OUT] (packed weights)
    b_conv_ptr,                     # *float* [C_OUT]
    b_extra_ptr,                    # *float* [C_OUT]
    y_ptr,                          # *float* [N, C_OUT, H_OUT, W_OUT]
    N: tl.constexpr,                # batch size
    C_IN: tl.constexpr,             # input channels
    H: tl.constexpr,                # input height
    W: tl.constexpr,                # input width
    C_OUT: tl.constexpr,            # output channels
    K: tl.constexpr,                # kernel size (assume square)
    H_OUT: tl.constexpr,            # output height
    W_OUT: tl.constexpr,            # output width
    x_sN, x_sC, x_sH, x_sW,         # strides for x
    w2d_sR, w2d_sC,                 # strides for packed weight 2D (rows, cols)
    y_sN, y_sC, y_sH, y_sW,         # strides for y
    BLOCK_PIX: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    K_TILE: tl.constexpr,
):
    # Program IDs
    pid_pix = tl.program_id(0)  # tiles over N*H_OUT*W_OUT
    pid_co = tl.program_id(1)   # tiles over C_OUT

    # Offsets for pixels (flattened n*h*w)
    pix_offs = pid_pix * BLOCK_PIX + tl.arange(0, BLOCK_PIX)
    co_offs = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    pix_mask = pix_offs < (N * H_OUT * W_OUT)
    co_mask = co_offs < C_OUT

    # Map flattened pixel indices to (n, h, w)
    HW_OUT = H_OUT * W_OUT
    n = pix_offs // HW_OUT
    rem = pix_offs % HW_OUT
    h = rem // W_OUT
    w = rem % W_OUT

    # Base pointer for each pixel (top-left of receptive field)
    x_base = x_ptr + n[:, None] * x_sN + h[:, None] * x_sH + w[:, None] * x_sW  # [BLOCK_PIX, 1]

    # Accumulator
    acc = tl.zeros((BLOCK_PIX, BLOCK_CO), dtype=tl.float32)

    K_TOTAL = C_IN * K * K
    # Reduction loop over K_TOTAL in tiles
    for k0 in range(0, K_TOTAL, K_TILE):
        r_idx = k0 + tl.arange(0, K_TILE)  # [K_TILE]
        r_mask = r_idx < K_TOTAL

        # Convert reduction index to (ci, kh, kw)
        KK = K * K
        ci = r_idx // KK
        rem2 = r_idx % KK
        kh = rem2 // K
        kw = rem2 % K

        # Load A: [BLOCK_PIX, K_TILE] from input patches
        a_ptrs = x_base + ci[None, :] * x_sC + kh[None, :] * x_sH + kw[None, :] * x_sW
        A = tl.load(a_ptrs, mask=(pix_mask[:, None] & r_mask[None, :]), other=0.0)

        # Load B: [K_TILE, BLOCK_CO] from packed weights
        b_ptrs = w2d_ptr + r_idx[:, None] * w2d_sR + co_offs[None, :] * w2d_sC
        B = tl.load(b_ptrs, mask=(r_mask[:, None] & co_mask[None, :]), other=0.0)

        # Matmul accumulate
        acc += tl.dot(A, B)

    # Add conv bias
    b_conv = tl.load(b_conv_ptr + co_offs, mask=co_mask, other=0.0)
    acc = acc + b_conv[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Add extra bias (broadcast over spatial)
    b_extra = tl.load(b_extra_ptr + co_offs, mask=co_mask, other=0.0)
    acc = acc + b_extra[None, :]

    # Store output
    y_ptrs = (
        y_ptr
        + n[:, None] * y_sN
        + co_offs[None, :] * y_sC
        + h[:, None] * y_sH
        + w[:, None] * y_sW
    )
    tl.store(y_ptrs, acc, mask=(pix_mask[:, None] & co_mask[None, :]))


def triton_conv2d_relu_addbias(x: torch.Tensor, w: torch.Tensor, b_conv: torch.Tensor, b_extra: torch.Tensor):
    """
    Fused conv2d (stride=1, padding=0, dilation=1, groups=1) + ReLU + add extra bias.
    x: [N, C_IN, H, W]
    w: [C_OUT, C_IN, K, K]
    b_conv: [C_OUT]
    b_extra: [C_OUT] or [C_OUT, 1, 1]
    """
    assert x.is_cuda and w.is_cuda and b_extra.is_cuda, "All tensors must be CUDA"
    assert x.dim() == 4 and w.dim() == 4, "x and w must be 4D (NCHW and OIHW)"
    N, C_IN, H, W = x.shape
    C_OUT, C_IN_w, K, K_w = w.shape
    assert C_IN == C_IN_w and K == K_w, "Incompatible conv weight shape"
    if b_conv is None:
        b_conv = torch.zeros(C_OUT, device=x.device, dtype=x.dtype)
    else:
        assert b_conv.numel() == C_OUT, "conv bias shape mismatch"
    b_extra_flat = b_extra.view(C_OUT).contiguous()

    # Output dims for padding=0, stride=1
    H_OUT = H - K + 1
    W_OUT = W - K + 1
    assert H_OUT > 0 and W_OUT > 0, "Invalid output size; check kernel size and padding"

    # Pack weights to 2D [K_TOTAL, C_OUT] with contiguous columns (coalesced loads across co)
    # Original w is [C_OUT, C_IN, K, K] (OIHW)
    w2d = w.reshape(C_OUT, C_IN * K * K).transpose(0, 1).contiguous()  # [K_TOTAL, C_OUT]

    # Allocate output
    y = torch.empty((N, C_OUT, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    # Ensure contiguity for expected strides
    x_c = x.contiguous()
    y_c = y  # will fill directly
    # Strides in elements
    x_sN, x_sC, x_sH, x_sW = x_c.stride()
    y_sN, y_sC, y_sH, y_sW = y_c.stride()
    w2d_sR, w2d_sC = w2d.stride()

    grid = lambda meta: (
        triton.cdiv(N * H_OUT * W_OUT, meta['BLOCK_PIX']),
        triton.cdiv(C_OUT, meta['BLOCK_CO']),
    )

    conv2d_relu_addbias_kernel[grid](
        x_c, w2d, b_conv.contiguous(), b_extra_flat,
        y_c,
        N, C_IN, H, W, C_OUT, K, H_OUT, W_OUT,
        x_sN, x_sC, x_sH, x_sW,
        w2d_sR, w2d_sC,
        y_sN, y_sC, y_sH, y_sW,
    )
    return y_c


class ModelNew(nn.Module):
    """
    Optimized model that fuses Conv2d + ReLU + bias add using a custom Triton kernel.
    Assumes stride=1, padding=0, dilation=1, groups=1 as in the provided Model.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # Fallback zero bias in case conv.bias is None (shouldn't happen since bias=True)
        self.register_buffer("zero_bias", torch.zeros(out_channels), persistent=False)

    def forward(self, x):
        assert x.is_cuda, "Input must be on CUDA for Triton kernels"
        b_conv = self.conv.bias if self.conv.bias is not None else self.zero_bias
        # Fuse conv + relu + add bias
        y = triton_conv2d_relu_addbias(x, self.conv.weight, b_conv, self.bias)
        return y


# Default configuration and helpers to match the original interface
batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]