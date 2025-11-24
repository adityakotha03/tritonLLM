import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------------------------
# Triton kernel that performs a 3×3 convolution (stride 1, padding 1),
# instance‑normalisation and division by a constant in one pass.
# --------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64, "BLOCK_X": 8, "BLOCK_Y": 8}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_X": 16, "BLOCK_Y": 16}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def conv_norm_div_kernel(
    A_ptr,          # Input: [B, C_in, H, W]
    B_ptr,          # Weights: [C_out, C_in, 3, 3]
    gamma_ptr,      # InstanceNorm gamma: [C_out]
    beta_ptr,       # InstanceNorm beta: [C_out]
    out_ptr,        # Output: [B, C_out, H, W]
    stride: tl.constexpr,
    pad: tl.constexpr,
    B: tl.constexpr,   # batch size
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    DIV: tl.constexpr,
):
    """
    Each program computes a BLOCK_X×BLOCK_Y tile of the output feature map
    for a single batch element and channel.  The tile is produced by
    accumulating over the input channels using a 3×3 kernel.  After
    convolution, a per‑channel mean/variance is computed and the
    instance‑normalisation is applied, followed by division by a constant.
    """

    # ------------------------------------------------------------
    # Compute the (x, y) coordinates of the current tile in the output
    # ------------------------------------------------------------
    block_x = tl.program_id(0) * BLOCK_X
    block_y = tl.program_id(1) * BLOCK_Y
    channel = tl.program_id(2)

    # ------------------------------------------------------------
    # Helper to load a tile of the input
    # ------------------------------------------------------------
    def load_A(b, c, h, w):
        # Convert to a linear index
        stride_b = C_in * H * W
        stride_c = H * W
        stride_h = W
        return tl.load(
            A_ptr + b * stride_b + c * stride_c + h * stride_h + w,
            mask=(h < H) & (w < W) & (b < B) & (c < C_in),
            other=0.0,
        )

    # ------------------------------------------------------------
    # Helper to load a tile of the weights
    # ------------------------------------------------------------
    def load_B(co, ci, kh, kw):
        stride_co = C_in * 3 * 3
        stride_ci = 3 * 3
        stride_kh = 3
        return tl.load(
            B_ptr + co * stride_co + ci * stride_ci + kh * stride_kh + kw,
            mask=(co < C_out) & (ci < C_in) & (kh < 3) & (kw < 3),
            other=0.0,
        )

    # ------------------------------------------------------------
    # Accumulator for the convolution result
    # ------------------------------------------------------------
    acc = tl.zeros([BLOCK_X, BLOCK_Y], dtype=tl.float32)

    # ------------------------------------------------------------
    # Convolution over input channels
    # ------------------------------------------------------------
    for ci in range(0, C_in, BLOCK_K):
        # Load a small tile of the input channel
        for kh in range(3):
            for kw in range(3):
                h_offset = block_y + kh - pad
                w_offset = block_x + kw - pad
                a = load_A(tl.arange(0, BLOCK_X), ci, h_offset, w_offset)
                b = load_B(channel, ci, kh, kw)
                acc += a * b

    # ------------------------------------------------------------
    # Instance normalisation over the spatial tile
    # ------------------------------------------------------------
    mean = tl.mean(acc, axis=(0, 1))
    var  = tl.var(acc, axis=(0, 1), unbiased=False)

    gamma = tl.load(gamma_ptr + channel, mask=(channel < C_out), other=1.0)
    beta  = tl.load(beta_ptr  + channel, mask=(channel < C_out), other=0.0)

    acc = gamma * (acc - mean) / tl.sqrt(var + 1e-5) + beta

    # ------------------------------------------------------------
    # Division by a constant
    # ------------------------------------------------------------
    acc = acc / DIV

    # ------------------------------------------------------------
    # Store the result
    # ------------------------------------------------------------
    for i in range(BLOCK_X):
        for j in range(BLOCK_Y):
            h_out = block_y + i
            w_out = block_x + j
            if (h_out < H) & (w_out < W):
                idx = h_out * W + w_out
                tl.store(
                    out_ptr + channel * H * W + idx,
                    acc[i, j],
                    mask=True
                )


# --------------------------------------------------------------------
# Helper wrapper that launches the kernel
# --------------------------------------------------------------------
def conv_norm_div(a: torch.Tensor,
                  weight: torch.Tensor,
                  gamma: torch.Tensor,
                  beta: torch.Tensor,
                  divide_by: float) -> torch.Tensor:
    B, C_in, H, W = a.shape
    C_out, _, kh, kw = weight.shape
    assert kh == kw == 3 and a.device == weight.device
    out = torch.empty((B, C_out, H, W), dtype=a.dtype, device=a.device)

    stride = 1
    pad = 1

    grid = lambda meta: (
        (H + meta["BLOCK_X"] - 1) // meta["BLOCK_X"],
        (W + meta["BLOCK_Y"] - 1) // meta["BLOCK_Y"],
        C_out,
    )

    conv_norm_div_kernel[grid](
        a,
        weight,
        gamma,
        beta,
        out,
        stride,
        pad,
        B,
        C_in,
        C_out,
        H,
        W,
        meta["BLOCK_M"],
        meta["BLOCK_N"],
        meta["BLOCK_K"],
        meta["BLOCK_X"],
        meta["BLOCK_Y"],
        divide_by,
    )
    return out


# --------------------------------------------------------------------
# ModelNew that uses the custom Triton kernel
# --------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised model that performs the same computation as the original
    Model but replaces the conv, instance‑normalisation and division
    with a fused Triton kernel for better throughput.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        assert kernel_size == 3, "Only 3×3 kernels are supported by the Triton kernel."
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size).cuda()
        )
        self.gamma = nn.Parameter(torch.ones(out_channels).cuda())
        self.beta  = nn.Parameter(torch.zeros(out_channels).cuda())
        self.divide_by = divide_by

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # All tensors are assumed to be on the same CUDA device.
        return conv_norm_div(x, self.weight, self.gamma, self.beta, self.divide_by)