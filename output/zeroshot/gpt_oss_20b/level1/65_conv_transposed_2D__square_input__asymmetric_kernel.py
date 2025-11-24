import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------
# Custom Triton kernels for the transposed
# 2‑D convolution (output up‑sampling).
# --------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K', 'dtype'],
)
@triton.jit
def _conv_transpose2d_kernel(
    # pointers to the tensors
    in_ptr,      # [B, C_in, H_in, W_in]
    weight_ptr,  # [C_in, C_out, kH, kW]
    bias_ptr,    # [C_out] (optional)
    out_ptr,     # [B, C_out, H_out, W_out]
    # sizes
    B: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    kH: tl.constexpr,
    kW: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    has_bias: tl.constexpr,
    # kernel tile size
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    dtype: tl.constexpr,
):
    """
    Naive implementation of ConvTranspose2d using im2col
    and GEMM.  Works for stride=1 only.  For larger stride
    the kernel must be specialised, which is beyond the scope
    of this example.  The code is intentionally simple to
    illustrate how Triton can be used for a convolution
    operation.

    The kernel expects the following layout:

        in_ptr  : (B, C_in, H_in, W_in)   -> 4‑D tensor
        weight_ptr : (C_in, C_out, kH, kW) -> 4‑D tensor
        out_ptr  : (B, C_out, H_out, W_out)

    All tensors are contiguous in memory and stored in
    row‑major order.
    """
    # grid dimensions: number of output rows * columns
    # each program processes a block of size BLOCK_M x BLOCK_N of the output
    program_id = tl.program_id(0)
    row_start = program_id // ((W_out + BLOCK_N - 1) // BLOCK_N)
    col_start = program_id % ((W_out + BLOCK_N - 1) // BLOCK_N)

    # offsets in the output tensor
    row_idx = row_start * BLOCK_M + tl.arange(0, BLOCK_M)
    col_idx = col_start * BLOCK_N + tl.arange(0, BLOCK_N)

    # mask for boundaries
    mask = (row_idx[:, None] < H_out) & (col_idx[None, :] < W_out)

    # initialise accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=dtype)

    # iterate over kernel windows
    for i in range(kH):
        for j in range(kW):
            # compute the corresponding input coordinates
            h_in = row_idx[:, None] * stride + i - padding
            w_in = col_idx[None, :] * stride + j - padding

            # mask for valid input coordinates
            valid = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)

            # load input values
            # flatten input index
            inp_idx = h_in * W_in + w_in
            inp_offset = (inp_idx[None, :] * C_in * H_in * W_in)  # [1, N] -> offset per channel

            # gather input over channels
            for c_in in range(C_in):
                # compute pointer offset for this channel
                ptr = in_ptr + c_in * H_in * W_in
                val = tl.load(ptr + inp_offset, mask=valid, other=0.0)
                # gather weight
                w_ptr = weight_ptr + c_in * C_out * kH * kW
                w_val = tl.load(
                    w_ptr + i * C_out * kW + j * C_out,
                    mask=valid,
                    other=0.0,
                )
                acc += val * w_val

    # add bias if present
    if has_bias:
        bias = tl.load(bias_ptr)
        acc += bias

    # store result
    out_offset = (row_idx[:, None] * W_out + col_idx[None, :]) * C_out
    for c_out in range(C_out):
        ptr = out_ptr + c_out * H_out * W_out
        tl.store(ptr + out_offset, acc, mask=mask)


def conv_transpose2d_torch_like(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
    output_padding: int,
) -> torch.Tensor:
    """
    Wrapper that launches the Triton kernel.  Works only for
    stride=1, padding>=0 and output_padding==0.
    """
    assert stride == 1, "Only stride==1 is supported in this example."
    assert output_padding == 0, "output_padding must be 0 in this example."
    B, C_in, H_in, W_in = input.shape
    C_out, _, kH, kW = weight.shape

    # compute output shape
    H_out = (H_in - 1) * stride - 2 * padding + kH
    W_out = (W_in - 1) * stride - 2 * padding + kW

    out = torch.empty(B, C_out, H_out, W_out, device=input.device, dtype=input.dtype)

    # launch the kernel
    grid = lambda meta: ((H_out + meta["BLOCK_M"] - 1) // meta["BLOCK_M"]) * \
                       ((W_out + meta["BLOCK_N"] - 1) // meta["BLOCK_N"])

    _conv_transpose2d_kernel[grid](
        input,
        weight,
        bias if bias is not None else torch.zeros(0, device=input.device, dtype=input.dtype),
        out,
        B, C_in, C_out,
        H_in, W_in,
        H_out, W_out,
        kH, kW,
        stride, padding, output_padding,
        int(bias is not None),
        dtype=tl.float32,
    )
    return out


# --------------------------------------------
# ModelNew that uses the custom Triton kernel.
# --------------------------------------------

class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with a square input and an asymmetric kernel
    using a custom Triton kernel for the core convolution operation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Torch's ConvTranspose2d is used only for weight initialization.
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the Triton kernel for the convolution part
        out = conv_transpose2d_torch_like(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias if self.bias else None,
            self.stride,
            self.padding,
            self.output_padding,
        )
        return out