import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch_size, out_channels, in_channels, depth_out, height_out, width_out, depth_in, height_in, width_in,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    output_pad_d, output_pad_h, output_pad_w,
    acc_dtype: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUPS: tl.constexpr
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    # Calculate output block start
    d_start = pid_d * BLOCK_D
    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W

    # Stride for output
    output_d_stride = height_out * width_out
    output_h_stride = width_out
    output_w_stride = 1

    # Stride for input
    input_d_stride = height_in * width_in
    input_h_stride = width_in
    input_w_stride = 1

    # Stride for weight
    weight_d_stride = kernel_h * kernel_w
    weight_h_stride = kernel_w
    weight_w_stride = 1
    weight_k_stride = in_channels // GROUPS * kernel_d * kernel_h * kernel_w
    weight_c_stride = out_channels // GROUPS * in_channels // GROUPS * kernel_d * kernel_h * kernel_w

    # Group computation
    group_id = pid_c_out // (out_channels // GROUPS)
    in_channel_group_start = group_id * (in_channels // GROUPS)
    out_channel_group_start = group_id * (out_channels // GROUPS)

    # Offset for output channel
    weight_channel_offset = (pid_c_out - out_channel_group_start) * weight_k_stride

    # Initialize accumulator for output block
    acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=acc_dtype)

    # Loop over input channels
    for c_in_base in range(0, in_channels // GROUPS, BLOCK_K):
        c_in_block = min(BLOCK_K, in_channels // GROUPS - c_in_base)

        # Load input block (batch, c_in, depth_in, height_in, width_in)
        for d in range(BLOCK_D):
            for h in range(BLOCK_H):
                for w in range(BLOCK_W):
                    o_d = d_start + d
                    o_h = h_start + h
                    o_w = w_start + w

                    if o_d < depth_out and o_h < height_out and o_w < width_out:
                        # Compute corresponding input indices
                        i_d_start = o_d - output_pad_d - pad_d
                        i_h_start = o_h - output_pad_h - pad_h
                        i_w_start = o_w - output_pad_w - pad_w

                        # Accumulate over kernel
                        for k_d in range(kernel_d):
                            for k_h in range(kernel_h):
                                for k_w in range(kernel_w):
                                    i_d = i_d_start + k_d * stride_d
                                    i_h = i_h_start + k_h * stride_h
                                    i_w = i_w_start + k_w * stride_w

                                    # Check bounds
                                    if (0 <= i_d < depth_in and 0 <= i_h < height_in and 0 <= i_w < width_in):
                                        for c_in in range(c_in_block):
                                            c_in_idx = c_in_base + c_in
                                            input_offset = pid_b * in_channels * depth_in * height_in * width_in + \
                                                           (in_channel_group_start + c_in_idx) * depth_in * height_in * width_in + \
                                                           i_d * input_d_stride + i_h * input_h_stride + i_w * input_w_stride
                                            weight_offset = weight_channel_offset + \
                                                            c_in_idx * weight_d_stride + \
                                                            k_d * weight_d_stride + k_h * weight_h_stride + k_w * weight_w_stride
                                            input_val = tl.load(input_ptr + input_offset)
                                            weight_val = tl.load(weight_ptr + weight_offset)
                                            acc[d, h, w] += input_val.to(acc_dtype) * weight_val.to(acc_dtype)
                    else:
                        acc[d, h, w] = 0.0

    # Store output
    for d in range(BLOCK_D):
        for h in range(BLOCK_H):
            for w in range(BLOCK_W):
                o_d = d_start + d
                o_h = h_start + h
                o_w = w_start + w
                if o_d < depth_out and o_h < height_out and o_w < width_out:
                    output_offset = pid_b * out_channels * depth_out * height_out * width_out + \
                                    pid_c_out * depth_out * height_out * width_out + \
                                    o_d * output_d_stride + o_h * output_h_stride + o_w * output_w_stride
                    tl.store(output_ptr + output_offset, acc[d, h, w])


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    output_padding: tuple,
    groups: int
):
    B, C_in, D_in, H_in, W_in = x.shape
    C_out, C_in_g, K_d, K_h, K_w = weight.shape
    C_in_g_total = C_in // groups

    # Compute output spatial dimensions
    D_out = (D_in - 1) * stride[0] - 2 * padding[0] + kernel_size[0] + output_padding[0]
    H_out = (H_in - 1) * stride[1] - 2 * padding[1] + kernel_size[1] + output_padding[1]
    W_out = (W_in - 1) * stride[2] - 2 * padding[2] + kernel_size[2] + output_padding[2]

    # Output tensor
    out = torch.zeros(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)

    # Define block sizes
    BLOCK_D = 4
    BLOCK_H = 8
    BLOCK_W = 8
    BLOCK_K = 16

    # Grid
    grid = (B, C_out, triton.cdiv(D_out, BLOCK_D), triton.cdiv(H_out, BLOCK_H), triton.cdiv(W_out, BLOCK_W))

    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, out,
        B, C_out, C_in, D_out, H_out, W_out, D_in, H_in, W_in,
        K_d, K_h, K_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        groups,
        acc_dtype=tl.float32,
        BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        BLOCK_K=BLOCK_K, GROUPS=groups
    )

    # Add bias if present
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1)

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='leaky_relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose weight to match expected format: (out_channels, in_channels // groups, k_d, k_h, k_w)
        weight_t = self.weight.permute(1, 0, 2, 3, 4).contiguous()
        return triton_conv_transpose3d(
            x, weight_t, self.bias,
            self.stride, self.padding, self.output_padding, self.groups
        )