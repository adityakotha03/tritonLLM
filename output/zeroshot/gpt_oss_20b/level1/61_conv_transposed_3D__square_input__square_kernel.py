import torch
import torch.nn as nn
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel for a generic ConvTranspose3d (stride=1, padding=0, output_padding=0)
# ----------------------------------------------------------------------
@triton.jit
def convtranspose3d_kernel(
    input,
    weight,
    output,
    bias,
    N: tl.constexpr,
    C_in: tl.constexpr,
    D_in: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    C_out: tl.constexpr,
    D_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    kD: tl.constexpr,
    kH: tl.constexpr,
    kW: tl.constexpr,
    padding: tl.constexpr,
    stride: tl.constexpr,
    output_padding: tl.constexpr,
    use_bias: tl.constexpr,
):
    """
    Computes the output of a ConvTranspose3d layer.
    Assumes stride=1, padding=0, output_padding=0 for simplicity.
    """
    # Number of elements per output tensor
    total_out = N * C_out * D_out * H_out * W_out

    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_out

    # Convert flat index to multi‑dimensional coordinates
    batch = offsets // (C_out * D_out * H_out * W_out)
    rem1 = offsets % (C_out * D_out * H_out * W_out)

    oc = rem1 // (D_out * H_out * W_out)
    rem2 = rem1 % (D_out * H_out * W_out)

    d_out = rem2 // (H_out * W_out)
    rem3 = rem2 % (H_out * W_out)

    h_out = rem3 // W_out
    w_out = rem3 % W_out

    # Accumulator
    val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over input channels
    for ic in range(C_in):
        # Loop over kernel depth
        for kd in tl.static_range(kD):
            d_in = d_out - (kd - padding)
            mask_d = (d_in >= 0) & (d_in < D_in)

            # Loop over kernel height
            for kh in tl.static_range(kH):
                h_in = h_out - (kh - padding)
                mask_h = (h_in >= 0) & (h_in < H_in)

                # Loop over kernel width
                for kw in tl.static_range(kW):
                    w_in = w_out - (kw - padding)
                    mask_w = (w_in >= 0) & (w_in < W_in)

                    mask_in = mask & mask_d & mask_h & mask_w

                    if tl.any(mask_in):
                        # Compute linear offsets into input and weight
                        in_offset = (
                            ((batch * C_in + ic) * D_in + d_in)
                            * H_in * W_in
                            + h_in * W_in
                            + w_in
                        )
                        w_offset = (
                            ((ic * C_out + oc) * kD + kd)
                            * kH * kW
                            + kh * kW
                            + kw
                        )

                        val += (
                            tl.load(input + in_offset, mask=mask_in, other=0.0)
                            * tl.load(weight + w_offset, mask=mask_in, other=0.0)
                        )

    if use_bias:
        val += tl.load(bias + oc, mask=mask, other=0.0)

    tl.store(output + offsets, val, mask=mask)


# ----------------------------------------------------------------------
# Wrapper that prepares the kernel launch
# ----------------------------------------------------------------------
def convtranspose3d_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
) -> torch.Tensor:
    """
    Computes a transposed 3‑D convolution using the custom Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda, "All tensors must be on GPU"
    if bias is not None:
        assert bias.is_cuda, "Bias must be on GPU"
        use_bias = True
    else:
        use_bias = False

    # Input and weight shapes
    N, C_in, D_in, H_in, W_in = input.shape
    C_out, _, kD, kH, kW = weight.shape  # weight shape: (C_in, C_out, kD, kH, kW)

    # Output dimensions (stride=1, padding=0, output_padding=0)
    D_out = (D_in - 1) * stride - 2 * padding + kD + output_padding
    H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + kW + output_padding

    output = torch.empty(
        (N, C_out, D_out, H_out, W_out),
        device=input.device,
        dtype=input.dtype,
    )

    BLOCK_SIZE = 256

    grid = lambda meta: ((N * C_out * D_out * H_out * W_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    convtranspose3d_kernel[grid](
        input,
        weight,
        output,
        bias,
        N,
        C_in,
        D_in,
        H_in,
        W_in,
        C_out,
        D_out,
        H_out,
        W_out,
        BLOCK_SIZE=BLOCK_SIZE,
        kD=kD,
        kH=kH,
        kW=kW,
        padding=padding,
        stride=stride,
        output_padding=output_padding,
        use_bias=use_bias,
    )

    return output


# ----------------------------------------------------------------------
# Model that uses the custom Triton kernel
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Transposed 3D convolution implemented with a custom Triton kernel.
    """

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
    ):
        super().__init__()
        assert groups == 1, "Only groups=1 is supported in this implementation"

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias_flag = bias

        # Weight: shape (in_channels, out_channels, kD, kH, kW)
        self.weight = nn.Parameter(
            torch.randn(
                in_channels,
                out_channels,
                kernel_size,
                kernel_size,
                kernel_size,
                device="cuda",
                dtype=torch.float32,
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.randn(out_channels, device="cuda", dtype=torch.float32)
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return convtranspose3d_triton(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )