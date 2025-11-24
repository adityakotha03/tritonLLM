import torch
import torch.nn as nn
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Triton kernel for 3‑D transposed convolution (kernel size 3)
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 4, 'BLOCK_H': 4, 'BLOCK_W': 4}, num_warps=4),
        triton.Config({'BLOCK_D': 8, 'BLOCK_H': 8, 'BLOCK_W': 8}, num_warps=8),
        triton.Config({'BLOCK_D': 16, 'BLOCK_H': 16, 'BLOCK_W': 16}, num_warps=8),
    ],
    key=['N', 'C_in', 'C_out', 'D_out', 'H_out', 'W_out', 'K', 'S', 'P', 'O', 'D', 'H', 'W'],
)
@triton.jit
def conv_transpose3d_kernel(
    # Pointers
    input_ptr,  # (N, C_in, D_in, H_in, W_in)
    weight_ptr, # (C_in, C_out, K, K, K)
    bias_ptr,   # (C_out,)  or None
    output_ptr, # (N, C_out, D_out, H_out, W_out)

    # Input sizes
    N: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    D_in: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,

    # Output sizes
    D_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,

    # Convolution parameters
    K: tl.constexpr,   # kernel size (assumed cubic)
    S: tl.constexpr,   # stride
    P: tl.constexpr,   # padding
    O: tl.constexpr,   # output padding
    D: tl.constexpr,   # dilation
    H: tl.constexpr,   # dilation (same for H)
    W: tl.constexpr,   # dilation (same for W)

    # Block dimensions
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Compute global indices for the output voxel
    d_out = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    h_out = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out = tl.program_id(2) * BLOCK_W + tl.arange(0, BLOCK_W)

    # Flatten indices
    d_out_idx = d_out[:, None, None]
    h_out_idx = h_out[None, :, None]
    w_out_idx = w_out[None, None, :]

    # Create masks for boundaries
    mask_d = d_out_idx < D_out
    mask_h = h_out_idx < H_out
    mask_w = w_out_idx < W_out
    mask = mask_d & mask_h & mask_w

    # Allocate accumulators
    acc = tl.zeros([BLOCK_D, BLOCK_H, BLOCK_W, C_out], dtype=tl.float32)

    # Iterate over input batch
    for n in range(N):
        # Iterate over kernel
        for kd in range(K):
            for kh in range(K):
                for kw in range(K):
                    # Compute corresponding input coordinates
                    d_in = d_out_idx * S - P + kd * D
                    h_in = h_out_idx * S - P + kh * D
                    w_in = w_out_idx * S - P + kw * D

                    # Mask for valid input coordinates
                    mask_d_in = (d_in >= 0) & (d_in < D_in)
                    mask_h_in = (h_in >= 0) & (h_in < H_in)
                    mask_w_in = (w_in >= 0) & (w_in < W_in)
                    mask_in = mask_d_in & mask_h_in & mask_w_in & mask

                    if tl.any(mask_in):
                        # Load input slice: shape [BLOCK_D, BLOCK_H, BLOCK_W, C_in]
                        in_offsets = (
                            n * (C_in * D_in * H_in * W_in)  # batch
                            + tl.arange(0, BLOCK_D)[:, None, None] * (C_in * H_in * W_in)
                            + tl.arange(0, BLOCK_H)[None, :, None] * (C_in * W_in)
                            + tl.arange(0, BLOCK_W)[None, None, :] * C_in
                        )
                        in_offsets = in_offsets + d_in[:, None, None] * (C_in * H_in * W_in) \
                                              + h_in[None, :, None] * (C_in * W_in) \
                                              + w_in[None, None, :] * C_in

                        in_vals = tl.load(
                            input_ptr + in_offsets,
                            mask=mask_in,
                            other=0.0,
                        )  # [D,H,W,C_in]

                        # Load kernel slice: shape [C_in, C_out]
                        k_offs = (
                            tl.arange(0, C_in)[:, None]
                            + kd * C_in * C_out
                            + kh * C_in * C_out * K
                            + kw * C_in * C_out * K * K
                        )
                        w_vals = tl.load(
                            weight_ptr + k_offs,
                            mask=None,
                            other=0.0,
                        )  # [C_in, C_out]

                        # Compute contribution: in_vals [D,H,W,C_in] @ w_vals [C_in,C_out]
                        # Broadcast to output
                        in_vals = tl.transpose(in_vals, 3, 0)  # [C_in,D,H,W]
                        w_vals = tl.transpose(w_vals, 0, 1)    # [C_in,C_out]
                        prod = tl.dot(in_vals, w_vals)  # [D,H,W,C_out]
                        prod = tl.transpose(prod, 2, 3)  # [D,H,W,C_out]
                        acc = acc + prod

    # Add bias if present
    if bias_ptr is not None:
        bias_vals = tl.load(bias_ptr)
        acc = acc + bias_vals[None, None, None, :]

    # Store results
    out_offsets = (
        tl.arange(0, BLOCK_D)[:, None, None] * (C_out * H_out * W_out)
        + tl.arange(0, BLOCK_H)[None, :, None] * (C_out * W_out)
        + tl.arange(0, BLOCK_W)[None, None, :] * C_out
    )
    out_offsets = out_offsets + d_out_idx[:, None, None] * (C_out * H_out * W_out) \
                         + h_out_idx[None, :, None] * (C_out * W_out) \
                         + w_out_idx[None, None, :] * C_out

    tl.store(output_ptr + out_offsets, acc, mask=mask)

# ------------------------------------------------------------------
# Wrapper that calls the Triton kernel
# ------------------------------------------------------------------
def triton_conv_transpose3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
    output_padding: tuple[int, int, int] = (0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
):
    N, C_in, D_in, H_in, W_in = input.shape
    C_in_w, C_out, K, K_h, K_w = weight.shape
    assert C_in_w == C_in and K == K_h == K_w
    S = stride
    P = padding
    O = output_padding
    D = dilation

    # Compute output shape
    D_out = (D_in - 1) * S[0] - 2 * P[0] + D[0] * (K - 1) + O[0] + 1
    H_out = (H_in - 1) * S[1] - 2 * P[1] + D[1] * (K - 1) + O[1] + 1
    W_out = (W_in - 1) * S[2] - 2 * P[2] + D[2] * (K - 1) + O[2] + 1

    out = torch.empty((N, C_out, D_out, H_out, W_out), device=input.device, dtype=input.dtype)

    # Launch kernel
    grid = (
        (D_out + 15) // 16,
        (H_out + 15) // 16,
        (W_out + 15) // 16,
    )
    conv_transpose3d_kernel[grid](
        input,
        weight,
        bias if bias is not None else tl.constexpr(None),
        out,
        N,
        C_in,
        C_out,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        K,
        S[0],
        P[0],
        O[0],
        D[0],
        D[1],
        D[2],
        BLOCK_D=16,
        BLOCK_H=16,
        BLOCK_W=16,
    )
    return out

# ------------------------------------------------------------------
# Optimised model using the Triton kernel
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution using a custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, output_padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride, stride)
        self.padding = (padding, padding, padding)
        self.output_padding = (output_padding, output_padding, output_padding)
        self.dilation = (dilation, dilation, dilation)
        self.groups = groups
        self.bias_flag = bias

        # Weight and bias parameters
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
        )