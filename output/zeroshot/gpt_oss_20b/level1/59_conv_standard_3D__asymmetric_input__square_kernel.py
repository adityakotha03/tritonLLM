import torch
import torch.nn as nn
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel for 3‑D convolution (kernel size (kH, kW, kD), stride=1, dilation=1, padding may be >0)
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    ],
    key=['n_out', 'kD', 'kH', 'kW', 'C_in_per_group'],
)
@triton.jit
def conv3d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_out,
    H, W, D,
    C_out, C_in_per_group, kD, kH, kW,
    pad_h, pad_w, pad_d,
    stride_h, stride_w, stride_d,
    dilation_h, dilation_w, dilation_d,
    groups,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_out

    # Compute multi‑dimensional indices for each output element
    # (batch, out_c, out_h, out_w, out_d)
    batch = offsets // (C_out * H * W * D)
    rem = offsets % (C_out * H * W * D)
    out_c = rem // (H * W * D)
    rem2 = rem % (H * W * D)
    out_h = rem2 // (W * D)
    rem3 = rem2 % (W * D)
    out_w = rem3 // D
    out_d = rem3 % D

    # Determine the group for the current output channel
    out_c_per_group = C_out // groups
    group = out_c // out_c_per_group
    c_in_per_group = C_in_per_group

    # Accumulator for the convolution result
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over all kernel dimensions and input channels
    for ic_local in range(c_in_per_group):
        # input channel index
        ic = group * c_in_per_group + ic_local

        for kd in range(kD):
            in_d = out_d + kd * dilation_d - pad_d
            # mask for valid depth index
            mask_d = (in_d >= 0) & (in_d < D)

            for kh in range(kH):
                in_h = out_h + kh * dilation_h - pad_h
                mask_h = (in_h >= 0) & (in_h < H)

                for kw in range(kW):
                    in_w = out_w + kw * dilation_w - pad_w
                    mask_w = (in_w >= 0) & (in_w < W)

                    # Combine all masks
                    mask_ = mask_d & mask_h & mask_w & mask

                    if tl.any(mask_):
                        # Compute flat offsets into weight and input tensors
                        w_offset = (
                            ((out_c * c_in_per_group + ic_local) * kD + kd) * kH * kW
                            + kh * kW
                            + kw
                        )
                        w = tl.load(weight_ptr + w_offset, mask=mask_, other=0.0)

                        inp_offset = (
                            ((batch * C_out + ic) * H + in_h) * W * D
                            + in_w * D
                            + in_d
                        )
                        x = tl.load(input_ptr + inp_offset, mask=mask_, other=0.0)

                        acc += w * x

    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + out_c, mask=mask, other=0.0)
        acc += bias

    tl.store(out_ptr + offsets, acc, mask=mask)


# ----------------------------------------------------------------------
# Wrapper function that launches the Triton kernel
# ----------------------------------------------------------------------
def triton_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
    groups: int,
):
    # x: (B, C_in, H, W, D)
    B, C_in, H, W, D = x.shape
    C_out, _, kD, kH, kW = weight.shape

    stride_h, stride_w, stride_d = stride
    pad_h, pad_w, pad_d = padding
    dilation_h, dilation_w, dilation_d = dilation

    # Compute output spatial dimensions
    H_out = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) // stride_w + 1
    D_out = (D + 2 * pad_d - dilation_d * (kD - 1) - 1) // stride_d + 1

    # Allocate output tensor
    out = torch.empty((B, C_out, H_out, W_out, D_out), dtype=x.dtype, device=x.device)

    # Flatten input, weight, bias for linear addressing
    x_flat = x.reshape(-1)
    weight_flat = weight.reshape(-1)
    bias_flat = bias.reshape(-1) if bias is not None else None

    # Number of output elements
    n_out = B * C_out * H_out * W_out * D_out

    # Determine number of input channels per group
    c_in_per_group = C_in // groups

    # Launch kernel
    conv3d_kernel[
        lambda meta: (n_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    ](
        x_flat,
        weight_flat,
        bias_flat,
        out.reshape(-1),
        n_out,
        H,
        W,
        D,
        C_out,
        c_in_per_group,
        kD,
        kH,
        kW,
        pad_h,
        pad_w,
        pad_d,
        stride_h,
        stride_w,
        stride_d,
        dilation_h,
        dilation_w,
        dilation_d,
        groups,
    )

    return out


# ----------------------------------------------------------------------
# Custom Model implementation that uses the Triton kernel
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with an asymmetric input and a square kernel
    implemented using a custom Triton kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super(ModelNew, self).__init__()

        # Store conv parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

        # Weight parameter shape: (out_channels, in_channels // groups, kD, kH, kW)
        weight_shape = (
            out_channels,
            in_channels // groups,
            1,  # kD (depth dimension is 1)
            kernel_size,
            kernel_size,
        )
        self.weight = nn.Parameter(torch.randn(weight_shape, dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform 3D convolution using the custom Triton kernel.
        """
        stride = (self.stride, self.stride, 1)
        padding = (self.padding, self.padding, 0)
        dilation = (self.dilation, self.dilation, 1)

        return triton_conv3d(
            x,
            self.weight,
            self.bias,
            stride,
            padding,
            dilation,
            self.groups,
        )