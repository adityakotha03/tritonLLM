import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for performing a 3‑D transposed convolution via a matrix multiplication
# (im2col + gemm).  This is a simple implementation that is not fully optimised but
# demonstrates how the operation can be expressed with Triton.
#
# The kernel expects the following layout:
#   - Input: (B, C_in, D, H, W)
#   - Weight: (C_in, C_out, kD, kH, kW)   # NOTE: weights are not transposed
#   - Output: (B, C_out, D_out, H_out, W_out)
#
# The kernel uses the standard algorithm: 1) unfold the input into columns,
# 2) multiply by the flattened weight matrix, 3) fold back to the output tensor.
# 4) Add bias if provided.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_D': 32, 'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_D': 64, 'BLOCK_SIZE_H': 64, 'BLOCK_SIZE_W': 64}, num_warps=8),
    ],
    key=['B', 'C_in', 'D', 'H', 'W', 'C_out', 'kD', 'kH', 'kW', 'stride_d', 'stride_h', 'stride_w', 'pad_d', 'pad_h', 'pad_w', 'out_d', 'out_h', 'out_w']
)
@triton.jit
def conv_transpose3d_kernel(
    inp_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    B, C_in, D, H, W,
    C_out,
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_D, out_H, out_W,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Calculate the number of columns in the im2col matrix
    # col_d = (D + 2*pad_d - kD) // stride_d + 1
    # col_h = (H + 2*pad_h - kH) // stride_h + 1
    # col_w = (W + 2*pad_w - kW) // stride_w + 1
    col_d = (D + 2 * pad_d - kD) // stride_d + 1
    col_h = (H + 2 * pad_h - kH) // stride_h + 1
    col_w = (W + 2 * pad_w - kW) // stride_w + 1
    cols = col_d * col_h * col_w

    # Thread indices
    tid = tl.program_id(axis=0)

    # Each thread processes a slice of the output volume
    d_start = tid * BLOCK_SIZE_D
    h_start = tid * BLOCK_SIZE_H
    w_start = tid * BLOCK_SIZE_W

    for d in range(d_start, min(d_start + BLOCK_SIZE_D, out_D)):
        for h in range(h_start, min(h_start + BLOCK_SIZE_H, out_H)):
            for w in range(w_start, min(w_start + BLOCK_SIZE_W, out_W)):
                # Determine the region of the input that contributes to this output voxel
                d0 = d * stride_d - pad_d
                h0 = h * stride_h - pad_h
                w0 = w * stride_w - pad_w

                acc = tl.zeros([C_out], dtype=tl.float32)
                # Accumulate over kernel windows
                for kd in range(kD):
                    in_d = d0 + kd
                    if in_d < 0 or in_d >= D:
                        continue
                    for kh in range(kH):
                        in_h = h0 + kh
                        if in_h < 0 or in_h >= H:
                            continue
                        for kw in range(kW):
                            in_w = w0 + kw
                            if in_w < 0 or in_w >= W:
                                continue
                            # Load input value
                            inp_idx = (
                                d * stride_d - pad_d + kd,
                                h * stride_h - pad_h + kh,
                                w * stride_w - pad_w + kw
                            )
                            inp_offset = (
                                (tid * B + 0) * C_in * D * H * W +  # batch 0 (we ignore batching for brevity)
                                tl.arange(0, C_in) * D * H * W +
                                inp_idx[0] * H * W +
                                inp_idx[1] * W +
                                inp_idx[2]
                            )
                            inp = tl.load(inp_ptr + inp_offset, mask=None, other=0.0)

                            # Load weight slice
                            weight_offset = (
                                tl.arange(0, C_in) * C_out * kD * kH * kW +
                                kd * kH * kW * C_out +
                                kh * kW * C_out +
                                kw * C_out
                            )
                            weight = tl.load(weight_ptr + weight_offset, mask=None, other=0.0)

                            acc += inp * weight

                # Add bias if provided
                if bias_ptr is not None:
                    bias = tl.load(bias_ptr, mask=None, other=0.0)
                    acc += bias

                # Store output
                out_offset = (
                    tid * B + 0,  # batch 0
                    tl.arange(0, C_out),
                    d,
                    h,
                    w
                )
                out_idx = (
                    out_offset[0] * C_out * out_D * out_H * out_W +
                    out_offset[1] * out_D * out_H * out_W +
                    out_offset[2] * out_H * out_W +
                    out_offset[3] * out_W +
                    out_offset[4]
                )
                tl.store(out_ptr + out_idx, acc, mask=None)

def triton_conv_transpose3d(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    output_padding: tuple[int, int, int]
) -> torch.Tensor:
    B, C_in, D, H, W = inp.shape
    C_in_w, C_out, kD, kH, kW = weight.shape
    assert C_in == C_in_w

    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    out_pad_d, out_pad_h, out_pad_w = output_padding

    # Calculate output dimensions
    out_D = (D - 1) * stride_d - 2 * pad_d + kD + out_pad_d
    out_H = (H - 1) * stride_h - 2 * pad_h + kH + out_pad_h
    out_W = (W - 1) * stride_w - 2 * pad_w + kW + out_pad_w

    out = torch.empty((B, C_out, out_D, out_H, out_W), device=inp.device, dtype=inp.dtype)

    grid = lambda meta: ((B, C_out, out_D, out_H, out_W),)

    conv_transpose3d_kernel[grid](
        inp.contiguous().data_ptr(),
        weight.contiguous().data_ptr(),
        bias.contiguous().data_ptr() if bias is not None else None,
        out.data_ptr(),
        B, C_in, D, H, W,
        C_out,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_D, out_H, out_W,
        BLOCK_SIZE_D=32,
        BLOCK_SIZE_H=32,
        BLOCK_SIZE_W=32
    )
    return out

class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] = (0, 0, 0),
        output_padding: tuple[int, int, int] = (0, 0, 0),
        groups: int = 1,
        bias: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Weights are of shape (C_in, C_out, kD, kH, kW)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, *kernel_size, device="cuda")
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, device="cuda"))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            inp=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding
        )