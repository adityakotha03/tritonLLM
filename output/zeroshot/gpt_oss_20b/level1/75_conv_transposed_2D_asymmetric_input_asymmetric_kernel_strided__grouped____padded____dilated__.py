import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
#                      Custom Triton kernel for ConvTranspose2d                #
# --------------------------------------------------------------------------- #

@triton.jit
def conv_transpose2d_kernel(
    in_ptr,          # [batch, in_ch, h_in, w_in]
    weight_ptr,      # [out_ch, in_ch/groups, k_h, k_w]
    bias_ptr,        # [out_ch] (or None)
    out_ptr,         # [batch, out_ch, h_out, w_out]
    batch_size: tl.constexpr,
    in_ch: tl.constexpr,
    out_ch: tl.constexpr,
    h_in: tl.constexpr,
    w_in: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    h_out: tl.constexpr,
    w_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Naive Triton implementation of a grouped 2‑D transposed convolution.
    The kernel tiles the output feature map and accumulates contributions
    from the input tensor and filter weights.
    """

    # Global linear index for the output element to compute
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < (batch_size * out_ch * h_out * w_out)

    if tl.any(mask):
        # Convert linear index to 4‑D coordinates
        out_n = idx // (out_ch * h_out * w_out)
        rem = idx % (out_ch * h_out * w_out)
        oc = rem // (h_out * w_out)
        rem = rem % (h_out * w_out)
        oh = rem // w_out
        ow = rem % w_out

        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        # Compute the corresponding input channel group
        group_idx = oc // (out_ch // groups)

        # Iterate over the kernel spatial dimensions
        for kh in range(0, weight_ptr.shape[2]):
            for kw in range(0, weight_ptr.shape[3]):
                # Compute the corresponding input position
                h_in_pos = oh * stride_h - pad_h + kh * dilation_h
                w_in_pos = ow * stride_w - pad_w + kw * dilation_w

                # Load input slice (all batches)
                # Skip if out of bounds
                in_mask = (h_in_pos >= 0) & (h_in_pos < h_in) & (w_in_pos >= 0) & (w_in_pos < w_in)

                if tl.any(in_mask):
                    # For each input channel in the group
                    for ic_in_group in range(in_ch // groups):
                        ic = group_idx * (in_ch // groups) + ic_in_group

                        # Load input value
                        in_val = tl.load(
                            in_ptr
                            + (out_n * in_ch * h_in * w_in)
                            + (ic * h_in * w_in)
                            + (h_in_pos * w_in)
                            + w_in_pos,
                            mask=in_mask,
                            other=0.0,
                        )

                        # Load weight value
                        w_val = tl.load(
                            weight_ptr
                            + (oc * (in_ch // groups) * weight_ptr.shape[2] * weight_ptr.shape[3])
                            + (ic_in_group * weight_ptr.shape[2] * weight_ptr.shape[3])
                            + (kh * weight_ptr.shape[3])
                            + kw
                        )

                        acc += in_val * w_val

        # Add bias if present
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + oc)
            acc += bias_val

        # Store the result
        tl.store(
            out_ptr
            + (out_n * out_ch * h_out * w_out)
            + (oc * h_out * w_out)
            + (oh * w_out)
            + ow,
            acc,
            mask=mask,
        )


# --------------------------------------------------------------------------- #
#                          Helper function to call the kernel                #
# --------------------------------------------------------------------------- #

def triton_conv_transpose2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
):
    """
    Wrapper that prepares the kernel launch configuration and calls the Triton
    kernel for a 2‑D transposed convolution.
    """
    batch_size, in_ch, h_in, w_in = input.shape
    out_ch, _, k_h, k_w = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    # Compute output size
    h_out = (h_in - 1) * stride_h - 2 * pad_h + (k_h - 1) * dilation_h + 1
    w_out = (w_in - 1) * stride_w - 2 * pad_w + (k_w - 1) * dilation_w + 1

    # Allocate output tensor
    out = torch.empty(
        (batch_size, out_ch, h_out, w_out),
        dtype=input.dtype,
        device=input.device,
    )

    BLOCK_SIZE = 256  # Tune this for best performance

    # Compute number of programs (blocks)
    grid = lambda meta: (
        ((batch_size * out_ch * h_out * w_out + meta["BLOCK_SIZE"] - 1)
         // meta["BLOCK_SIZE"]),
    )

    conv_transpose2d_kernel[grid](
        input,
        weight,
        bias if bias is not None else None,
        out,
        batch_size=batch_size,
        in_ch=in_ch,
        out_ch=out_ch,
        h_in=h_in,
        w_in=w_in,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        groups=groups,
        h_out=h_out,
        w_out=w_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


# --------------------------------------------------------------------------- #
#                                 ModelNew class                              #
# --------------------------------------------------------------------------- #

class ModelNew(nn.Module):
    """
    Optimized model that replaces the PyTorch ConvTranspose2d operator with a
    custom Triton kernel. The interface is identical to the original
    architecture.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels // groups,
                *kernel_size,
                device="cuda",
                dtype=torch.float32,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, device="cuda", dtype=torch.float32))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )