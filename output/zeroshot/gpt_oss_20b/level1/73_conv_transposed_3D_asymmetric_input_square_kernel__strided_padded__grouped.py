import torch
import torch.nn as nn
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel for 3d transposed convolution (single‑kernel, stride=2)
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 4, 'BLOCK_H': 4, 'BLOCK_W': 8, 'BLOCK_C': 8}, num_warps=4),
        triton.Config({'BLOCK_D': 8, 'BLOCK_H': 8, 'BLOCK_W': 16, 'BLOCK_C': 16}, num_warps=8),
    ],
    key=['N', 'C', 'D', 'H', 'W'],
)
@triton.jit
def conv_transpose3d_kernel(
    input_ptr,          # [N, C, D_in, H_in, W_in]
    weight_ptr,         # [C_out, C_in/groups, kD, kH, kW]
    bias_ptr,           # [C_out]
    output_ptr,         # [N, C_out, D_out, H_out, W_out]
    N, C_in, C_out, D_in, H_in, W_in,
    D_out, H_out, W_out,
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # program ids for batch, output channel block, depth block, height block, width block
    n = tl.program_id(0)
    c_out_b = tl.program_id(1)
    d_out_b = tl.program_id(2)
    h_out_b = tl.program_id(3)
    w_out_b = tl.program_id(4)

    # compute ranges
    d_start = d_out_b * BLOCK_D
    h_start = h_out_b * BLOCK_H
    w_start = w_out_b * BLOCK_W
    c_start = c_out_b * BLOCK_C

    d_offsets = d_start + tl.arange(0, BLOCK_D)
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    c_offsets = c_start + tl.arange(0, BLOCK_C)

    # masks for output boundaries
    mask_d = d_offsets < D_out
    mask_h = h_offsets < H_out
    mask_w = w_offsets < W_out
    mask_c = c_offsets < C_out

    # iterate over kernel
    for kd in range(kD):
        for kh in range(kH):
            for kw in range(kW):
                # corresponding input coords
                d_in = d_offsets * stride_d + kd - pad_d
                h_in = h_offsets * stride_h + kh - pad_h
                w_in = w_offsets * stride_w + kw - pad_w

                # masks for input bounds
                in_mask_d = (d_in >= 0) & (d_in < D_in)
                in_mask_h = (h_in >= 0) & (h_in < H_in)
                in_mask_w = (w_in >= 0) & (w_in < W_in)

                # combine all masks
                full_mask = mask_d & mask_h & mask_w & mask_c & in_mask_d & in_mask_h & in_mask_w

                if not full_mask.any():
                    continue

                # load input slice
                input_idx = (
                    n * (C_in * D_in * H_in * W_in)
                    + tl.arange(0, BLOCK_C)[:, None, None, None] * (D_in * H_in * W_in)
                    + d_in[None, :, None, None] * (H_in * W_in)
                    + h_in[None, None, :, None] * W_in
                    + w_in[None, None, None, :]
                )
                inp = tl.load(input_ptr + input_idx, mask=full_mask, other=0.0)

                # load weight slice
                weight_idx = (
                    c_offsets[:, None, None, None] * (C_in * kD * kH * kW)
                    + kd * (C_in * kH * kW)
                    + kh * (C_in * kW)
                    + kw * C_in
                )
                wgt = tl.load(weight_ptr + weight_idx, mask=full_mask, other=0.0)

                # accumulate
                out_idx = (
                    n * (C_out * D_out * H_out * W_out)
                    + c_offsets[:, None, None, None] * (D_out * H_out * W_out)
                    + d_offsets[None, :, None, None] * (H_out * W_out)
                    + h_offsets[None, None, :, None] * W_out
                    + w_offsets[None, None, None, :]
                )
                out = tl.load(output_ptr + out_idx, mask=full_mask, other=0.0)
                out = out + inp * wgt
                tl.store(output_ptr + out_idx, out, mask=full_mask)

    # add bias
    if bias_ptr is not None:
        bias_idx = c_offsets[:, None, None, None]
        bias = tl.load(bias_ptr + bias_idx, mask=mask_c, other=0.0)
        bias = bias[None, :, None, None, None]
        out_idx = (
            n * (C_out * D_out * H_out * W_out)
            + c_offsets[:, None, None, None] * (D_out * H_out * W_out)
            + d_offsets[None, :, None, None] * (H_out * W_out)
            + h_offsets[None, None, :, None] * W_out
            + w_offsets[None, None, None, :]
        )
        out = tl.load(output_ptr + out_idx, mask=full_mask, other=0.0)
        out = out + bias
        tl.store(output_ptr + out_idx, out, mask=full_mask)

# ----------------------------------------------------------------------
# Wrapper function that launches the kernel
# ----------------------------------------------------------------------
def conv_transpose3d_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
) -> torch.Tensor:
    N, C_in, D_in, H_in, W_in = input.shape
    C_out, C_in_per_g, kD, kH, kW = weight.shape
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding

    # output size
    D_out = (D_in - 1) * stride_d - 2 * pad_d + kD
    H_out = (H_in - 1) * stride_h - 2 * pad_h + kH
    W_out = (W_in - 1) * stride_w - 2 * pad_w + kW

    output = torch.empty((N, C_out, D_out, H_out, W_out), dtype=input.dtype, device=input.device)

    # grid dims
    grid = (
        N,
        (C_out + 15) // 16,
        (D_out + 3) // 4,
        (H_out + 3) // 4,
        (W_out + 7) // 8,
    )

    conv_transpose3d_kernel[grid](
        input,
        weight,
        bias,
        output,
        N, C_in, C_out, D_in, H_in, W_in,
        D_out, H_out, W_out,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        BLOCK_D=4, BLOCK_H=4, BLOCK_W=8, BLOCK_C=16,
    )
    return output

# ----------------------------------------------------------------------
# New model that uses the custom Triton kernel
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Weight and bias are stored as parameters
        weight_shape = (
            out_channels,
            in_channels // groups,
            kernel_size,
            kernel_size,
            kernel_size,
        )
        self.weight = nn.Parameter(
            torch.randn(weight_shape, dtype=torch.float32, device="cuda")
        )
        if bias:
            self.bias_param = nn.Parameter(
                torch.randn(out_channels, dtype=torch.float32, device="cuda")
            )
        else:
            self.register_parameter("bias_param", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous
        x = x.contiguous()
        stride = (self.stride, self.stride, self.stride)
        padding = (self.padding, self.padding, self.padding)
        return conv_transpose3d_triton(
            x, self.weight, self.bias_param, stride, padding
        )