import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_3d_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    B,  # batch size
    C_in,
    C_out,
    D,
    H,
    W,
    K_d,
    K_h,
    K_w,
    S_d,
    S_h,
    S_w,
    P_d,
    P_h,
    P_w,
    O_d,
    O_h,
    O_w,
    O_pad_d,
    O_pad_h,
    O_pad_w,
    stride_x,
    stride_w,
    stride_bias,
    stride_out,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
    TILE_K_D: tl.constexpr,
    TILE_K_H: tl.constexpr,
    TILE_K_W: tl.constexpr,
):
    # Thread indices
    pid_batch = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    # Compute output dimensions
    d_offset = pid_d * BLOCK_D
    h_offset = pid_h * BLOCK_H
    w_offset = pid_w * BLOCK_W

    c_out_offset = pid_c_out * BLOCK_C_OUT

    # Load weights (C_out x C_in x K_d x K_h x K_w)
    w_ptrs = w_ptr + (pid_c_out * C_in * K_d * K_h * K_w) + tl.arange(0, BLOCK_C_IN)[:, None, None, None] * (K_d * K_h * K_w) + \
             tl.arange(0, K_d)[None, :, None, None] * (K_h * K_w) + \
             tl.arange(0, K_h)[None, None, :, None] * (K_w) + \
             tl.arange(0, K_w)[None, None, None, :]  # (BLOCK_C_IN, K_d, K_h, K_w)
    w = tl.load(w_ptrs, mask=(tl.arange(0, BLOCK_C_IN)[:, None, None, None] < C_in) & 
                            (tl.arange(0, K_d)[None, :, None, None] < K_d) &
                            (tl.arange(0, K_h)[None, None, :, None] < K_h) &
                            (tl.arange(0, K_w)[None, None, None, :] < K_w), other=0.0)

    # Load bias (C_out,)
    bias = tl.load(bias_ptr + c_out_offset, mask=pid_c_out < C_out, other=0.0)

    # Output accumulators
    acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Loop over input channels and kernel positions
    for c_in in range(0, C_in, BLOCK_C_IN):
        # Compute input offset
        c_in_offset = c_in
        x_ptrs = x_ptr + (pid_batch * C_in * D * H * W) + \
                 (c_in_offset * D * H * W) + \
                 (tl.arange(0, BLOCK_D)[:, None, None] * (H * W)) + \
                 (tl.arange(0, BLOCK_H)[None, :, None] * W) + \
                 (tl.arange(0, BLOCK_W)[None, None, :])

        # Load input block
        x = tl.load(x_ptrs, mask=(tl.arange(0, BLOCK_D)[:, None, None] < D) &
                                 (tl.arange(0, BLOCK_H)[None, :, None] < H) &
                                 (tl.arange(0, BLOCK_W)[None, None, :] < W), other=0.0)

        # Compute output positions in the input space
        for k_d in range(0, K_d, TILE_K_D):
            for k_h in range(0, K_h, TILE_K_H):
                for k_w in range(0, K_w, TILE_K_W):
                    d_in = d_offset + k_d
                    h_in = h_offset + k_h
                    w_in = w_offset + k_w

                    # Compute output indices
                    d_out = d_in * S_d - P_d
                    h_out = h_in * S_h - P_h
                    w_out = w_in * S_w - P_w

                    # Check bounds
                    valid_d = (d_out >= 0) & (d_out < D)
                    valid_h = (h_out >= 0) & (h_out < H)
                    valid_w = (w_out >= 0) & (w_out < W)

                    # Only if in bounds
                    if valid_d and valid_h and valid_w:
                        # Accumulate
                        acc += tl.dot(x, w, allow_tf32=False)

    # Scale and add bias
    acc += bias

    # Store output
    out_ptrs = out_ptr + (pid_batch * C_out * O_d * O_h * O_w) + \
               (c_out_offset * O_d * O_h * O_w) + \
               (tl.arange(0, BLOCK_D)[:, None, None] * (O_h * O_w)) + \
               (tl.arange(0, BLOCK_H)[None, :, None] * O_w) + \
               (tl.arange(0, BLOCK_W)[None, None, :])

    mask = (tl.arange(0, BLOCK_D)[:, None, None] < O_d) & \
           (tl.arange(0, BLOCK_H)[None, :, None] < O_h) & \
           (tl.arange(0, BLOCK_W)[None, None, :] < O_w)

    tl.store(out_ptrs, acc, mask=mask)


@triton.jit
def residual_add_mul_kernel(
    x_ptr,
    out_ptr,
    B,
    C,
    D,
    H,
    W,
    stride_x,
    stride_out,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    c_offset = pid_c * BLOCK_C
    d_offset = pid_d * BLOCK_D
    h_offset = pid_h * BLOCK_H
    w_offset = pid_w * BLOCK_W

    # Load original input
    x_ptrs = x_ptr + (pid_batch * C * D * H * W) + \
             (c_offset * D * H * W) + \
             (tl.arange(0, BLOCK_D)[:, None, None] * (H * W)) + \
             (tl.arange(0, BLOCK_H)[None, :, None] * W) + \
             (tl.arange(0, BLOCK_W)[None, None, :])
    x = tl.load(x_ptrs, mask=(tl.arange(0, BLOCK_D)[:, None, None] < D) &
                             (tl.arange(0, BLOCK_H)[None, :, None] < H) &
                             (tl.arange(0, BLOCK_W)[None, None, :] < W), other=0.0)

    # Perform fused operations:
    # x = x + x  # residual add
    # x = x * x  # multiplication
    # x = x + x  # residual add
    # → x = ((x + x) * x) + x = (2x * x) + x = 2x^2 + x
    acc = 2.0 * x * x + x

    # Store result
    out_ptrs = out_ptr + (pid_batch * C * D * H * W) + \
               (c_offset * D * H * W) + \
               (tl.arange(0, BLOCK_D)[:, None, None] * (H * W)) + \
               (tl.arange(0, BLOCK_H)[None, :, None] * W) + \
               (tl.arange(0, BLOCK_W)[None, None, :])
    mask = (tl.arange(0, BLOCK_D)[:, None, None] < D) & \
           (tl.arange(0, BLOCK_H)[None, :, None] < H) & \
           (tl.arange(0, BLOCK_W)[None, None, :] < W)

    tl.store(out_ptrs, acc, mask=mask)


def triton_conv_transpose_3d(x, weight, bias, kernel_size, stride, padding, output_padding):
    B, C_in, D, H, W = x.shape
    C_out, C_in, K_d, K_h, K_w = weight.shape
    O_d = (D - 1) * stride[0] - 2 * padding[0] + kernel_size[0] + output_padding[0]
    O_h = (H - 1) * stride[1] - 2 * padding[1] + kernel_size[1] + output_padding[1]
    O_w = (W - 1) * stride[2] - 2 * padding[2] + kernel_size[2] + output_padding[2]

    # Allocate output
    out = torch.empty(B, C_out, O_d, O_h, O_w, device=x.device, dtype=x.dtype)

    # Grid setup
    grid = lambda meta: (
        B,
        (C_out + meta["BLOCK_C_OUT"] - 1) // meta["BLOCK_C_OUT"],
        (O_d + meta["BLOCK_D"] - 1) // meta["BLOCK_D"],
        (O_h + meta["BLOCK_H"] - 1) // meta["BLOCK_H"],
        (O_w + meta["BLOCK_W"] - 1) // meta["BLOCK_W"],
    )

    # Launch kernel
    conv_transpose_3d_kernel[grid](
        x,
        weight,
        bias,
        out,
        B,
        C_in,
        C_out,
        D,
        H,
        W,
        K_d,
        K_h,
        K_w,
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        O_d,
        O_h,
        O_w,
        output_padding[0],
        output_padding[1],
        output_padding[2],
        x.stride(0),
        weight.stride(0),
        bias.stride(0),
        out.stride(0),
        BLOCK_D=64,
        BLOCK_H=64,
        BLOCK_W=64,
        BLOCK_C_OUT=16,
        BLOCK_C_IN=8,
        TILE_K_D=3,
        TILE_K_H=3,
        TILE_K_W=3,
    )

    return out


def triton_residual_add_mul(x):
    B, C, D, H, W = x.shape

    # Allocate output
    out = torch.empty_like(x)

    # Grid setup
    grid = lambda meta: (
        B,
        (C + meta["BLOCK_C"] - 1) // meta["BLOCK_C"],
        (D + meta["BLOCK_D"] - 1) // meta["BLOCK_D"],
        (H + meta["BLOCK_H"] - 1) // meta["BLOCK_H"],
        (W + meta["BLOCK_W"] - 1) // meta["BLOCK_W"],
    )

    # Launch kernel
    residual_add_mul_kernel[grid](
        x,
        out,
        B,
        C,
        D,
        H,
        W,
        x.stride(0),
        out.stride(0),
        BLOCK_D=64,
        BLOCK_H=64,
        BLOCK_W=64,
        BLOCK_C=16,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Perform conv_transpose with Triton
        x = triton_conv_transpose_3d(
            x,
            self.conv_transpose.weight,
            self.bias,
            kernel_size=self.conv_transpose.kernel_size,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding
        )

        # Fuse residual additions and multiplication
        x = triton_residual_add_mul(x)

        return x