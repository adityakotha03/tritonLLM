import torch
import torch.nn as nn
import triton
import triton.language as tl

# ------------------------------------
# Triton kernel for a generic 3D conv
# ------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=16),
    ],
    key=["batch", "out_c", "in_c", "kernel_d", "kernel_h", "kernel_w", "pad_d", "pad_h", "pad_w", "stride_d", "stride_h", "stride_w"],
)
@triton.jit
def conv3d_kernel(
    input_ptr,          # [B, Cin, D, H, W]
    weight_ptr,         # [Cout, Cin, KD, KH, KW]
    bias_ptr,           # [Cout] or None
    output_ptr,         # [B, Cout, D_out, H_out, W_out]
    B: tl.constexpr,
    Cin: tl.constexpr,
    Cout: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    pad_d: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    D_in: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    D_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Linear index of the current program (output element)
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (B * Cout * D_out * H_out * W_out)

    # Decode offset to multidimensional indices
    b_idx = offsets // (Cout * D_out * H_out * W_out)
    rest = offsets % (Cout * D_out * H_out * W_out)

    oc_idx = rest // (D_out * H_out * W_out)
    rest = rest % (D_out * H_out * W_out)

    od_idx = rest // (H_out * W_out)
    rest = rest % (H_out * W_out)

    oh_idx = rest // W_out
    ow_idx = rest % W_out

    # Compute the corresponding input coordinates
    in_d_start = od_idx * stride_d - pad_d
    in_h_start = oh_idx * stride_h - pad_h
    in_w_start = ow_idx * stride_w - pad_w

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over input channels and kernel dims
    for ic in range(Cin):
        for kd in range(KD):
            id = in_d_start + kd
            for kh in range(KH):
                ih = in_h_start + kh
                for kw in range(KW):
                    iw = in_w_start + kw

                    # Check bounds
                    valid = (id >= 0) & (id < D_in) & \
                            (ih >= 0) & (ih < H_in) & \
                            (iw >= 0) & (iw < W_in)

                    # Load input value
                    inp_offset = (
                        b_idx * Cin * D_in * H_in * W_in
                        + ic * D_in * H_in * W_in
                        + id * H_in * W_in
                        + ih * W_in
                        + iw
                    )
                    inp = tl.load(input_ptr + inp_offset, mask=valid, other=0.0)

                    # Load weight
                    w_offset = (
                        oc_idx * Cin * KD * KH * KW
                        + ic * KD * KH * KW
                        + kd * KH * KW
                        + kh * KW
                        + kw
                    )
                    w = tl.load(weight_ptr + w_offset, mask=mask, other=0.0)

                    acc += inp * w

    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + oc_idx, mask=mask, other=0.0)
        acc += bias

    # Store the result
    out_offset = (
        b_idx * Cout * D_out * H_out * W_out
        + oc_idx * D_out * H_out * W_out
        + od_idx * H_out * W_out
        + oh_idx * W_out
        + ow_idx
    )
    tl.store(output_ptr + out_offset, acc, mask=mask)


# ------------------------------------
# Triton wrapper
# ------------------------------------
def triton_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
):
    B, Cin, D_in, H_in, W_in = x.shape
    Cout, _, KD, KH, KW = weight.shape
    pad_d, pad_h, pad_w = padding
    stride_d, stride_h, stride_w = stride

    # Compute output shape
    D_out = (D_in + 2 * pad_d - KD) // stride_d + 1
    H_out = (H_in + 2 * pad_h - KH) // stride_h + 1
    W_out = (W_in + 2 * pad_w - KW) // stride_w + 1

    out = torch.empty((B, Cout, D_out, H_out, W_out), dtype=x.dtype, device=x.device)

    # Launch kernel
    grid = lambda meta: ((B * Cout * D_out * H_out * W_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    conv3d_kernel[grid](
        x, weight, bias,
        out,
        B, Cin, Cout,
        KD, KH, KW,
        pad_d, pad_h, pad_w,
        stride_d, stride_h, stride_w,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        BLOCK_SIZE=128,
    )
    return out


# ------------------------------------
# Optimized model
# ------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int, int],
                 stride: tuple[int, int, int] = (1, 1, 1),
                 padding: tuple[int, int, int] = (0, 0, 0),
                 dilation: tuple[int, int, int] = (1, 1, 1),
                 groups: int = 1,
                 bias: bool = False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        # copy weights to Triton-friendly layout
        self.weight = self.conv.weight
        if bias:
            self.bias = self.conv.bias
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv3d(
            x,
            self.weight,
            self.bias,
            self.conv.stride,
            self.conv.padding,
        )