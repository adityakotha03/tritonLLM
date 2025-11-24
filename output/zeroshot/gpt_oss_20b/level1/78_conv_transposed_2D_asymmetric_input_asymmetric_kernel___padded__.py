import torch
import torch.nn as nn
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton kernel for 2D transposed convolution
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def conv_transpose2d_kernel(
    X_ptr,          # input tensor: (N, C_in, H_in, W_in)
    W_ptr,          # weight tensor: (C_out, C_in, KH, KW)
    B_ptr,          # bias tensor: (C_out,) or None
    OUT_ptr,        # output tensor: (N, C_out, H_out, W_out)
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Transposed convolution over a batch of inputs.
    The kernel is written for a single batch element and
    is launched with a grid covering all output spatial positions.
    """
    pid = tl.program_id(axis=0)  # linear index over all output pixels
    # Total number of output pixels per image
    pixels_per_img = H_out * W_out
    # Compute which image in the batch this program handles
    img_idx = pid // pixels_per_img
    pixel_idx = pid % pixels_per_img
    # Map pixel_idx to (y_out, x_out)
    y_out = pixel_idx // W_out
    x_out = pixel_idx % W_out

    # Load pointers for this image
    x_img = X_ptr + img_idx * (C_in * H_in * W_in)
    out_img = OUT_ptr + img_idx * (C_out * H_out * W_out)

    # Prepare accumulation buffer for each output channel
    acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)

    # Iterate over kernel height
    for ky in range(KH):
        # Corresponding input y position (before stride)
        y_in = y_out + pad_h - ky
        # Must be divisible by stride_h
        if y_in % stride_h != 0:
            continue
        y_in //= stride_h
        if (y_in < 0) or (y_in >= H_in):
            continue

        # Iterate over kernel width
        for kx in range(KW):
            x_in = x_out + pad_w - kx
            if x_in % stride_w != 0:
                continue
            x_in //= stride_w
            if (x_in < 0) or (x_in >= W_in):
                continue

            # Load input patch: (C_in)
            in_offset = (y_in * W_in + x_in) * C_in
            inp = tl.load(x_img + in_offset, mask=True, other=0.0)

            # Load weight slice for all output channels
            # We treat the weights as (C_out, C_in, KH, KW)
            # Flattened as: [C_out * C_in * KH * KW]
            # For a given ky, kx we need the block over C_out and C_in
            for co in range(0, C_out, BLOCK_SIZE_N):
                # Load weights for this co block
                w_off = (co * C_in * KH * KW) + (ky * C_in * KW) + (kx * C_in)
                w = tl.load(W_ptr + w_off, mask=True, other=0.0, num_elements=BLOCK_SIZE_N * C_in)
                w = w.to(tl.float32).reshape(BLOCK_SIZE_N, C_in)
                # Accumulate dot product: sum over C_in
                acc[co:co+BLOCK_SIZE_N] += tl.sum(inp * w, axis=1)

    # Write the result to the output tensor
    out_offset = (y_out * W_out + x_out) * C_out
    for co in range(0, C_out, BLOCK_SIZE_N):
        out_slice = acc[co:co+BLOCK_SIZE_N]
        # Add bias if present
        if B_ptr is not None:
            bias = tl.load(B_ptr + co, mask=True, other=0.0, num_elements=BLOCK_SIZE_N)
            out_slice += bias
        tl.store(out_img + out_offset + co, out_slice, mask=True)


# ----------------------------------------------------------------------
# Triton wrapper
# ----------------------------------------------------------------------
def triton_conv_transpose2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
):
    """
    A Triton implementation of 2D transposed convolution.
    Supports only the exact combination used in the original model.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    N, C_in, H_in, W_in = input.shape
    C_out, _, KH, KW = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    H_out = (H_in - 1) * stride_h - 2 * pad_h + KH
    W_out = (W_in - 1) * stride_w - 2 * pad_w + KW

    out = torch.empty((N, C_out, H_out, W_out), dtype=input.dtype, device=input.device)

    # Grid: one program per output pixel per image
    pixels_per_img = H_out * W_out
    grid = lambda meta: (
        (N * pixels_per_img + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
    )

    conv_transpose2d_kernel[grid](
        input,
        weight,
        bias,
        out,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        KH=KH,
        KW=KW,
        H_in=H_in,
        W_in=W_in,
        H_out=H_out,
        W_out=W_out,
        C_in=C_in,
        C_out=C_out,
        N=N,
    )
    return out


# ----------------------------------------------------------------------
# Optimized model
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, *kernel_size, dtype=torch.float32)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
        )