import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------
# Custom Triton kernel that fuses
#   1) Global average pooling over H,W
#   2) Bias addition
#   3) Log-sum-exp over channel dimension
#   4) Final multiplication by 10
# --------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16, "BLOCK_C": 32}, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32, "BLOCK_C": 64}, num_warps=8),
    ],
    key=["N", "C", "H", "W"],
)
@triton.jit
def fused_ops_kernel(
    input_ptr,   # [N, C, H, W]
    bias_ptr,    # [C, 1, 1]
    out_ptr,     # [N, 1]
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Thread mapping: one thread processes one element of a tile
    n = tl.program_id(0)                     # batch index
    c = tl.program_id(1) * BLOCK_C           # channel start
    h = tl.program_id(2) * BLOCK_H           # height start
    w = tl.program_id(3) * BLOCK_W           # width start

    # Compute tile indices
    c_offsets = c + tl.arange(0, BLOCK_C)
    h_offsets = h + tl.arange(0, BLOCK_H)
    w_offsets = w + tl.arange(0, BLOCK_W)

    # Masks for bounds
    c_mask = c_offsets < C
    h_mask = h_offsets < H
    w_mask = w_offsets < W

    # Accumulate sum over spatial dimensions for each channel
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for hi in range(0, H, BLOCK_H):
        for wi in range(0, W, BLOCK_W):
            h_off = hi + tl.arange(0, BLOCK_H)
            w_off = wi + tl.arange(0, BLOCK_W)
            h_mask = h_off < H
            w_mask = w_off < W

            # Load input block
            inp = tl.load(
                input_ptr + n * C * H * W
                + c_offsets[:, None, None] * H * W
                + h_off[None, :, None] * W
                + w_off[None, None, :],
                mask=c_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :],
                other=0.0,
            )

            acc += tl.sum(inp, dim=(1, 2))

    # Broadcast bias and add
    bias = tl.load(bias_ptr + c_offsets[:, None, None], mask=c_mask[:, None, None], other=0.0)
    acc += bias.squeeze(-1).squeeze(-1)

    # Reduce across channels with log-sum-exp
    # First compute max for numerical stability
    max_val = tl.reduce_max(acc, axis=0)
    # Compute exp(acc - max)
    exp_vals = tl.exp(acc - max_val)
    sum_exp = tl.reduce_sum(exp_vals, axis=0)
    log_sum_exp = max_val + tl.log(sum_exp)

    # Write output (scalar per batch)
    if n == 0:  # Only one thread per batch writes result
        tl.store(out_ptr, log_sum_exp * 10.0)

# --------------------------------------------------
# Wrapper function that prepares the kernel launch
# --------------------------------------------------
def fused_ops(input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    input : [N, C, H, W]
    bias  : [C, 1, 1]
    returns : [N]
    """
    N, C, H, W = input.shape
    out = torch.empty((N,), device=input.device, dtype=input.dtype)

    grid = lambda meta: (
        (N,  # batch grid
         (C + meta["BLOCK_C"] - 1) // meta["BLOCK_C"],
         (H + meta["BLOCK_H"] - 1) // meta["BLOCK_H"],
         (W + meta["BLOCK_W"] - 1) // meta["BLOCK_W"]),
    )

    fused_ops_kernel[grid](
        input,
        bias,
        out,
        N=N,
        C=C,
        H=H,
        W=W,
        BLOCK_H=meta["BLOCK_H"],
        BLOCK_W=meta["BLOCK_W"],
        BLOCK_C=meta["BLOCK_C"],
    )
    return out

# --------------------------------------------------
# Optimized model using the custom kernel
# --------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed convolution,
    then uses a fused Triton kernel for the remaining ops.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Transposed convolution
        x = self.conv_transpose(x)                     # [B, C, H, W]
        # Fused operations: avg pool + bias + logsumexp + sum + mul
        out = fused_ops(x, self.bias)                  # [B]
        return out