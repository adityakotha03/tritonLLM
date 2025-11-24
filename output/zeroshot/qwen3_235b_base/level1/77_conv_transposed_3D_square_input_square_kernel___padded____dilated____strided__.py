import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_transpose3d_kernel(
    input_ptr, weight_ptr, output_ptr,
    bias_ptr,
    batch_size, in_channels, out_channels,
    input_depth, input_height, input_width,
    output_depth, output_height, output_width,
    kernel_size_d, kernel_size_h, kernel_size_w,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    dilation_d, dilation_h, dilation_w,
    output_padding_d, output_padding_h, output_padding_w,
    groups: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # 3D transposed convolution via implicit gemm formulation: (P, Q, R) x (K, K, K) -> (T, H, W)
    # We iterate over output spatial locations and compute the contribution from input and kernel.

    # Program IDs
    pid_b = tl.program_id(axis=0)  # batch
    pid_m = tl.program_id(axis=1)  # output channel (out_channels)
    pid_n = tl.program_id(axis=2)  # output spatial position (output_depth * output_height * output_width)

    # Offset for output spatial dimensions
    od = (pid_n // (output_height * output_width))
    oh = (pid_n % (output_height * output_width)) // output_width
    ow = (pid_n % output_width)

    # Initialize accumulator for output value
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over input channels and kernel
    for ic in range(0, in_channels, BLOCK_SIZE_K):
        # Load input tile: [batch, ic:ic+BLOCK_SIZE_K, ...]
        for kd in range(0, kernel_size_d):
            for kh in range(0, kernel_size_h):
                for kw in range(0, kernel_size_w):
                    # Compute input spatial location
                    id_val = od * stride_d - padding_d + kd * dilation_d
                    ih_val = oh * stride_h - padding_h + kh * dilation_h
                    iw_val = ow * stride_w - padding_w + kw * dilation_w

                    # Check bounds
                    valid = (id_val >= 0) & (id_val < input_depth) & \
                            (ih_val >= 0) & (ih_val < input_height) & \
                            (iw_val >= 0) & (iw_val < input_width)

                    # Input offset
                    input_offset = pid_b * in_channels * input_depth * input_height * input_width + \
                                   ic * input_depth * input_height * input_width + \
                                   id_val * input_height * input_width + \
                                   ih_val * input_width + iw_val
                    input_mask = (valid & (ic < in_channels))[:, None]

                    # Weight offset: [out_ch, in_ch, kd, kh, kw]
                    weight_offset = pid_m * in_channels * kernel_size_d * kernel_size_h * kernel_size_w + \
                                    ic * kernel_size_d * kernel_size_h * kernel_size_w + \
                                    kd * kernel_size_h * kernel_size_w + kh * kernel_size_w + kw
                    weight_mask = (ic < in_channels)[:, None]

                    # Load input and weight tiles
                    x = tl.load(input_ptr + input_offset, mask=input_mask, other=0.0)
                    w = tl.load(weight_ptr + weight_offset, mask=weight_mask, other=0.0)

                    # Accumulate
                    acc += tl.dot(w[None, :], x[:, None], out_dtype=tl.float32)

    # Add bias if present
    if has_bias:
        b = tl.load(bias_ptr + pid_m)
        acc += b

    # Store output
    output_offset = pid_b * out_channels * output_depth * output_height * output_width + \
                    pid_m * output_depth * output_height * output_width + \
                    od * output_height * output_width + oh * output_width + ow
    tl.store(output_ptr + output_offset, acc, mask=(pid_m < out_channels)[None, :])


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    output_padding: tuple,
    dilation: tuple,
):
    B, C_in, D_in, H_in, W_in = x.shape
    C_out, _, Kd, Kh, Kw = weight.shape
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    dil_d, dil_h, dil_w = dilation
    out_pad_d, out_pad_h, out_pad_w = output_padding

    # Compute output spatial dimensions
    D_out = (D_in - 1) * stride_d - 2 * pad_d + dil_d * (Kd - 1) + out_pad_d + 1
    H_out = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (Kh - 1) + out_pad_h + 1
    W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (Kw - 1) + out_pad_w + 1

    # Output tensor
    out = torch.empty((B, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Launch kernel
    def grid(META):
        return (B, triton.cdiv(C_out, META['BLOCK_SIZE_M']), D_out * H_out * W_out)

    # Heuristics for block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16

    _conv_transpose3d_kernel[grid](
        x, weight, out, bias,
        B, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        Kd, Kh, Kw,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        out_pad_d, out_pad_h, out_pad_w,
        groups=1,
        has_bias=(bias is not None),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias

        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Weight initialization (same as ConvTranspose3d)
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='leaky_relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose weight to match expected format: (out_channels, in_channels, K, K, K)
        weight_t = self.weight.permute(1, 0, 2, 3, 4).contiguous()

        # Use Triton kernel
        return triton_conv_transpose3d(
            x,
            weight_t,
            self.bias,
            stride=(self.stride, self.stride, self.stride),
            padding=(self.padding, self.padding, self.padding),
            output_padding=(0, 0, 0),
            dilation=(self.dilation, self.dilation, self.dilation),
        )