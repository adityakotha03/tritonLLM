import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _dw_conv_kernel(
    x_ptr, w_ptr, y_ptr,
    B, H, W, C,
    K, stride, padding, dilation,
    input_height_stride, input_width_stride, input_channel_stride,
    weight_height_stride, weight_width_stride, weight_channel_stride,
    output_height_stride, output_width_stride, output_channel_stride,
    eps,  # used for padding value (0.0)
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_w = tl.program_id(axis=2)
    pid_c = tl.program_id(axis=3)

    # Offset for channel blocks
    c_offset = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    mask_c = c_offset < C

    # Input/output spatial offsets
    h_offset = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_offset = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Output spatial bounds
    h_mask = h_offset < (H + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    w_mask = w_offset < (W + 2 * padding - dilation * (K - 1) - 1) // stride + 1

    # Input spatial coordinates
    h_input = h_offset * stride - padding + dilation * 0
    w_input = w_offset * stride - padding + dilation * 0

    # Load weights (K, K, C)
    k1 = tl.arange(0, BLOCK_SIZE_K)
    k2 = tl.arange(0, BLOCK_SIZE_K)
    k1_mask = k1 < K
    k2_mask = k2 < K
    weight_offsets = (
        k1[:, None] * weight_height_stride +
        k2[None, :] * weight_width_stride +
        c_offset[None, None, :] * weight_channel_stride
    )
    weight_mask = k1_mask[:, None, None] & k2_mask[None, :, None] & mask_c[None, None, :]
    weights = tl.load(w_ptr + weight_offsets, mask=weight_mask, other=0.0)

    # Accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C), dtype=tl.float32)

    # Convolution loop over kernel
    for ki in range(0, K):
        for kj in range(0, K):
            h_cur = h_input + dilation * ki
            w_cur = w_input + dilation * kj

            # Bounds checking with padding
            h_valid = (h_cur >= 0) & (h_cur < H)
            w_valid = (w_cur >= 0) & (w_cur < W)
            valid_mask = h_valid[:, :, None] & w_valid[:, :, None] & mask_c[None, None, :]

            # Input offsets
            x_offsets = (
                pid_b * input_channel_stride * C +
                h_cur[:, :, None] * input_height_stride +
                w_cur[:, :, None] * input_width_stride +
                c_offset[None, None, :] * input_channel_stride
            )
            x_vals = tl.load(x_ptr + x_offsets, mask=valid_mask, other=eps)

            # Multiply-accumulate
            acc += x_vals * weights[ki, kj, :]

    # Store output
    y_offsets = (
        pid_b * output_channel_stride * C +
        h_offset[:, :, None] * output_height_stride +
        w_offset[:, :, None] * output_width_stride +
        c_offset[None, None, :] * output_channel_stride
    )
    y_mask = h_mask[:, :, None] & w_mask[:, :, None] & mask_c[None, None, :]
    tl.store(y_ptr + y_offsets, acc, mask=y_mask)


@triton.jit
def _pw_conv_kernel(
    x_ptr, w_ptr, y_ptr,
    B, H, W, IN_C, OUT_C,
    input_height_stride, input_width_stride, input_channel_stride,
    weight_channel_stride, weight_output_stride,
    output_height_stride, output_width_stride, output_channel_stride,
    BLOCK_SIZE_CIN: tl.constexpr,
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_cout = tl.program_id(3)

    # Output channel block
    cout_start = pid_cout * BLOCK_SIZE_COUT
    cout = cout_start + tl.arange(0, BLOCK_SIZE_COUT)
    cout_mask = cout < OUT_C

    # Spatial block
    h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    h_mask = h < H
    w_mask = w < W

    # Input channel block (full reduction)
    cin_blocks = tl.cdiv(IN_C, BLOCK_SIZE_CIN)
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_COUT), dtype=tl.float32)

    for cid in range(0, cin_blocks):
        cin_start = cid * BLOCK_SIZE_CIN
        cin = cin_start + tl.arange(0, BLOCK_SIZE_CIN)
        cin_mask = cin < IN_C
        mask_hw = h_mask[:, None] & w_mask[:, None]

        # Load input: (B, H, W, IN_C)
        x_offsets = (
            pid_b * input_channel_stride * IN_C +
            h[:, None] * input_height_stride +
            w[:, None] * input_width_stride +
            cin[None, :] * input_channel_stride
        )
        x_mask = mask_hw[:, :, None] & cin_mask[None, None, :]
        x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

        # Load weights: (IN_C, OUT_C)
        w_offsets = (
            cin[:, None] * weight_channel_stride +
            cout[None, :] * weight_output_stride
        )
        w_mask = cin_mask[:, None] & cout_mask[None, :]
        w = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

        # GEMM: [H*W, Cin] @ [Cin, Cout] -> [H*W, Cout]
        acc += tl.dot(x, w)

    # Store output
    y_offsets = (
        pid_b * output_channel_stride * OUT_C +
        h[:, None] * output_height_stride +
        w[:, None] * output_width_stride +
        cout[None, :] * output_channel_stride
    )
    y_mask = h_mask[:, :, None] & w_mask[:, :, None] & cout_mask[None, None, :]
    tl.store(y_ptr + y_offsets, acc, mask=y_mask)


def triton_depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
):
    B, C, H, W = x.shape
    K = weight.shape[2]
    assert weight.shape == (C, 1, K, K), "Depthwise weight must be (C, 1, K, K)"

    # Output spatial size
    H_out = (H + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (K - 1) - 1) // stride + 1

    # Create output
    out = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)

    # Strides
    x_stride = x.stride()
    w_stride = weight.stride()
    out_stride = out.stride()

    # Reshape to (B, H, W, C) for Triton
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    weight_nhwc = weight.squeeze(1).permute(1, 2, 0).contiguous()  # (K, K, C)
    out_nhwc = out.permute(0, 2, 3, 1).contiguous()

    # Grid
    grid = (B, triton.cdiv(H_out, 16), triton.cdiv(W_out, 16), triton.cdiv(C, 16))

    # Launch kernel
    _dw_conv_kernel[grid](
        x_nhwc, weight_nhwc, out_nhwc,
        B, H, W, C, K, stride, padding, dilation,
        x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        weight_nhwc.stride(0), weight_nhwc.stride(1), weight_nhwc.stride(2),
        out_nhwc.stride(1), out_nhwc.stride(2), out_nhwc.stride(3),
        eps=0.0,
        BLOCK_SIZE_C=16,
        BLOCK_SIZE_H=16,
        BLOCK_SIZE_W=16,
        BLOCK_SIZE_K=triton.cdiv(K, 1),
    )

    # Back to NCHW
    out = out_nhwc.permute(0, 3, 1, 2).contiguous()

    # Add bias
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)

    return out


def triton_pointwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
):
    B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    assert weight.shape == (C_out, C_in, 1, 1), "Pointwise weight must be (C_out, C_in, 1, 1)"

    out = torch.empty((B, C_out, H, W), device=x.device, dtype=x.dtype)

    # Reshape to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    weight_flat = weight.squeeze(-1).squeeze(-1).t().contiguous()  # (C_in, C_out)
    out_nhwc = out.permute(0, 2, 3, 1).contiguous()

    # Grid
    grid = (B, triton.cdiv(H, 16), triton.cdiv(W, 16), triton.cdiv(C_out, 16))

    # Launch kernel
    _pw_conv_kernel[grid](
        x_nhwc, weight_flat, out_nhwc,
        B, H, W, C_in, C_out,
        x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        weight_flat.stride(0), weight_flat.stride(1),
        out_nhwc.stride(1), out_nhwc.stride(2), out_nhwc.stride(3),
        BLOCK_SIZE_CIN=32,
        BLOCK_SIZE_COUT=16,
        BLOCK_SIZE_H=16,
        BLOCK_SIZE_W=16,
    )

    out = out_nhwc.permute(0, 3, 1, 2).contiguous()

    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)

    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise-separable 2D convolution using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Depthwise weight
        self.weight_dw = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        # Pointwise weight
        self.weight_pw = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))
        
        if bias:
            self.bias_dw = nn.Parameter(torch.empty(in_channels))
            self.bias_pw = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias_dw = None
            self.bias_pw = None

        # Init parameters
        nn.init.kaiming_uniform_(self.weight_dw, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.weight_pw, a=0, mode='fan_in', nonlinearity='relu')
        if self.bias_dw is not None:
            nn.init.zeros_(self.bias_dw)
            nn.init.zeros_(self.bias_pw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = triton_depthwise_conv2d(
            x, self.weight_dw, self.bias_dw,
            stride=self.stride, padding=self.padding, dilation=self.dilation
        )
        x = triton_pointwise_conv2d(x, self.weight_pw, self.bias_pw)
        return x