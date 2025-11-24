import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
#  Triton kernel that performs a 3‑D transposed convolution (de‑convolution)
#  The implementation follows the “im2col + matmul” strategy that is common
#  for transposed convolutions.  All the work is done in one kernel, so the
#  number of global memory accesses is minimal and the tensor‑core path is
#  used for the large matrix multiplication.
# --------------------------------------------------------------------------- #

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 256, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 32}, num_warps=8),
        triton.Config({"BLOCK_SIZE_N": 256, "BLOCK_SIZE_M": 256, "BLOCK_SIZE_K": 32}, num_warps=16),
    ],
    key=["N", "C", "H", "W", "D"],
)
@triton.jit
def conv_transpose3d_kernel(
    # pointers
    input_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    bias_ptr: tl.tensor,
    output_ptr: tl.tensor,

    # sizes
    N: tl.constexpr,  # batch
    C: tl.constexpr,  # in_channels
    D: tl.constexpr,  # depth
    H: tl.constexpr,  # height
    W: tl.constexpr,  # width
    K: tl.constexpr,  # out_channels
    KD: tl.constexpr, # kernel depth
    KH: tl.constexpr, # kernel height
    KW: tl.constexpr, # kernel width
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_d: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    has_bias: tl.constexpr,

    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute the transposed convolution for a single batch element.
    The kernel is written in a way that each thread processes a small
    2‑D tile of the output feature map and iterates over all input channels.
    """
    # ----------------------------------------------------------------------- #
    #  Compute the shape of the output tensor for a single batch element
    # ----------------------------------------------------------------------- #
    out_D = (D - 1) * stride_d - 2 * pad_d + dilation_d * (KD - 1) + 1
    out_H = (H - 1) * stride_h - 2 * pad_h + dilation_h * (KH - 1) + 1
    out_W = (W - 1) * stride_w - 2 * pad_w + dilation_w * (KW - 1) + 1

    # Each thread processes a tile of size BLOCK_SIZE_N x BLOCK_SIZE_M in
    # the (out_D, out_H, out_W) space.
    # 1st dimension: out_D * out_H (flattened)
    # 2nd dimension: out_W
    grid_DH = (out_D * out_H + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_W  = (out_W          + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    # The grid is created externally so we just use program_id to index
    # the 2‑D tile.
    tid = tl.program_id(0)
    tile_row = tid // grid_W
    tile_col = tid %  grid_W

    # Range of positions in the flattened (out_D * out_H) dimension
    row_offsets = tile_row * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    col_offsets = tile_col * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # Compute the corresponding out_D, out_H, out_W indices
    out_dh = row_offsets // out_H
    out_h  = row_offsets %  out_H
    out_w  = col_offsets

    # Mask to keep only valid output indices
    valid_dh = (out_dh < out_D) & (out_h < out_H)
    valid_w  = (out_w  < out_W)

    # ----------------------------------------------------------------------- #
    #  For every output position, iterate over all kernels (in_channels) and
    #  accumulate contributions from the corresponding input positions.
    # ----------------------------------------------------------------------- #
    for c in range(C):
        # Compute the range of kernel indices that can reach the current
        # output location.  Because we are doing a transposed convolution,
        # the kernel is applied *backwards* – i.e. we shift the kernel
        # over the output grid and map it to the input grid.
        # The mapping from output to input is:
        #
        #   in_d = (out_d - k_d * dilation_d + pad_d) // stride_d
        #   in_h = (out_h - k_h * dilation_h + pad_h) // stride_h
        #   in_w = (out_w - k_w * dilation_w + pad_w) // stride_w
        #
        # Only the kernel indices that satisfy these equations contribute.

        k_d_range = tl.arange(0, KD)
        k_h_range = tl.arange(0, KH)
        k_w_range = tl.arange(0, KW)

        # Expand kernel indices to match the tile size
        k_d_grid = tl.broadcast_to(k_d_range, (BLOCK_SIZE_N, 1))
        k_h_grid = tl.broadcast_to(k_h_range, (1, BLOCK_SIZE_M))
        k_w_grid = tl.broadcast_to(k_w_range, (1, BLOCK_SIZE_M))

        # Compute candidate input coordinates
        in_d = (out_dh - k_d_grid * dilation_d + pad_d) // stride_d
        in_h = (out_h  - k_h_grid * dilation_h + pad_h) // stride_h
        in_w = (out_w  - k_w_grid * dilation_w + pad_w) // stride_w

        # Validity mask for the kernel indices
        in_d_valid = (in_d >= 0) & (in_d < D)
        in_h_valid = (in_h >= 0) & (in_h < H)
        in_w_valid = (in_w >= 0) & (in_w < W)
        valid_mask = in_d_valid & in_h_valid & in_w_valid

        # Only process valid kernel positions
        if tl.any(valid_mask):
            # Gather input values
            input_idx = (
                c * D * H * W
                + in_d * H * W
                + in_h * W
                + in_w
            )
            input_vals = tl.load(
                input_ptr + input_idx,
                mask=valid_mask & valid_dh[:, None] & valid_w[None, :],
                other=0.0,
            )

            # Gather weights (transpose of the conv weight)
            # Weight shape: (C, K, KD, KH, KW)
            # We want W^T: (K, C, KD, KH, KW)
            weight_idx = (
                c * K * KD * KH * KW
                + k_d_grid * KH * KW
                + k_h_grid * KW
                + k_w_grid
            )
            weight_vals = tl.load(
                weight_ptr + weight_idx,
                mask=valid_mask,
                other=0.0,
            )

            # Accumulate into the output tile
            out_vals = tl.load(
                output_ptr + tl.arange(0, BLOCK_SIZE_N * BLOCK_SIZE_M),
                mask=valid_dh[:, None] & valid_w[None, :],
                other=0.0,
            )
            out_vals += tl.dot(input_vals, weight_vals)
            tl.store(
                output_ptr + tl.arange(0, BLOCK_SIZE_N * BLOCK_SIZE_M),
                out_vals,
                mask=valid_dh[:, None] & valid_w[None, :],
            )

    # ----------------------------------------------------------------------- #
    #  Add bias if needed (broadcast over spatial dimensions)
    # ----------------------------------------------------------------------- #
    if has_bias:
        for k in range(K):
            bias_idx = k
            bias_val = tl.load(bias_ptr + bias_idx)
            # Broadcast bias across the tile
            out_vals = tl.load(
                output_ptr + tl.arange(0, BLOCK_SIZE_N * BLOCK_SIZE_M),
                mask=valid_dh[:, None] & valid_w[None, :],
                other=0.0,
            )
            out_vals += bias_val
            tl.store(
                output_ptr + tl.arange(0, BLOCK_SIZE_N * BLOCK_SIZE_M),
                out_vals,
                mask=valid_dh[:, None] & valid_w[None, :],
            )

# --------------------------------------------------------------------------- #
#  Helper function that launches the kernel for a whole batch
# --------------------------------------------------------------------------- #
def conv_transpose3d_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int | tuple[int, int, int],
    padding: int | tuple[int, int, int],
    dilation: int | tuple[int, int, int],
) -> torch.Tensor:
    """
    Performs a 3‑D transposed convolution using a custom Triton kernel.
    The function is compatible with the torch.nn.ConvTranspose3d
    interface (except for the bias argument which may be None).

    Args:
        input (torch.Tensor): Input tensor of shape (N, C, D, H, W)
        weight (torch.Tensor): Weight tensor of shape (C, K, KD, KH, KW)
        bias (torch.Tensor | None): Bias tensor of shape (K,)
        stride (int | tuple[int, int, int])
        padding (int | tuple[int, int, int])
        dilation (int | tuple[int, int, int])

    Returns:
        torch.Tensor: Output tensor of shape (N, K, D_out, H_out, W_out)
    """
    N, C, D, H, W = input.shape
    K, _, KD, KH, KW = weight.shape

    stride = (stride, stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation, dilation) if isinstance(dilation, int) else dilation

    out_D = (D - 1) * stride[0] - 2 * padding[0] + dilation[0] * (KD - 1) + 1
    out_H = (H - 1) * stride[1] - 2 * padding[1] + dilation[1] * (KH - 1) + 1
    out_W = (W - 1) * stride[2] - 2 * padding[2] + dilation[2] * (KW - 1) + 1

    out = torch.empty((N, K, out_D, out_H, out_W), dtype=input.dtype, device=input.device)

    # Flatten weight to match the kernel's expectations
    # weight shape: (C, K, KD, KH, KW)
    weight_flat = weight.contiguous()
    bias_flat = bias.contiguous() if bias is not None else torch.zeros(K, device=input.device, dtype=input.dtype)

    # Determine grid size for the kernel
    # Each program handles a tile of size (BLOCK_SIZE_N, BLOCK_SIZE_M)
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_M = 128

    grid_DH = (out_D * out_H + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_W  = (out_W          + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid = (grid_DH * grid_W,)

    conv_transpose3d_kernel[grid](
        input_ptr=input.data_ptr(),
        weight_ptr=weight_flat.data_ptr(),
        bias_ptr=bias_flat.data_ptr(),
        output_ptr=out.data_ptr(),
        N=N,
        C=C,
        D=D,
        H=H,
        W=W,
        K=K,
        KD=KD,
        KH=KH,
        KW=KW,
        stride_d=stride[0],
        stride_h=stride[1],
        stride_w=stride[2],
        pad_d=padding[0],
        pad_h=padding[1],
        pad_w=padding[2],
        dilation_d=dilation[0],
        dilation_h=dilation[1],
        dilation_w=dilation[2],
        has_bias=bias is not None,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=32,
    )
    return out

# --------------------------------------------------------------------------- #
#  Optimized Model that uses the Triton kernel
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    3‑D transposed convolution using a custom Triton kernel.
    Parameters match torch.nn.ConvTranspose3d.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, *kernel_size, device="cuda")
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, device="cuda"))
        else:
            self.register_parameter("bias", None)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose3d_triton(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )