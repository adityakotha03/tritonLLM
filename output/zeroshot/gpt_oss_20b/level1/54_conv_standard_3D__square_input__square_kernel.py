import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
# Triton kernel for a 3‑D convolution (group = 1, no bias handling in the kernel)
# --------------------------------------------------------------------------- #
@triton.jit
def conv3d_kernel(
    x_ptr,            # Input tensor pointer
    w_ptr,            # Weight tensor pointer
    b_ptr,            # Bias tensor pointer (contiguous even if bias is None)
    out_ptr,          # Output tensor pointer

    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,

    depth_in: tl.constexpr,
    width_in: tl.constexpr,
    height_in: tl.constexpr,

    depth_out: tl.constexpr,
    width_out: tl.constexpr,
    height_out: tl.constexpr,

    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,

    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,

    pad_d: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,

    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,

    bias_present: tl.constexpr,     # 1 if bias is provided, 0 otherwise
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program processes a contiguous block of output elements.
    """
    total_out = batch_size * out_channels * depth_out * width_out * height_out

    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_out
    if not mask.any():
        return

    # ---- Decode offsets to indices --------------------------------------------------
    # Compute the indices of the output element for each thread
    b_idx = offsets // (out_channels * depth_out * width_out * height_out)
    rem = offsets % (out_channels * depth_out * width_out * height_out)
    oc_idx = rem // (depth_out * width_out * height_out)
    rem2 = rem % (depth_out * width_out * height_out)
    d_idx = rem2 // (width_out * height_out)
    rem3 = rem2 % (width_out * height_out)
    w_idx = rem3 // height_out
    h_idx = rem3 % height_out

    # Pre‑compute strides (constant across all threads)
    D_in = depth_in
    H_in = height_in
    W_in = width_in
    C_in = in_channels
    C_out = out_channels
    K_d = kernel_d
    K_h = kernel_h
    K_w = kernel_w

    # Input base offset for each batch
    input_base = b_idx * C_in * D_in * H_in * W_in
    # Weight base offset for each output channel
    weight_base = oc_idx * C_in * K_d * K_h * K_w

    out_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # ---- Convolution loop ---------------------------------------------------------
    for kd in range(K_d):
        for kh in range(K_h):
            for kw in range(K_w):
                # Compute input coordinates
                d_in = d_idx * stride_d + kd * dilation_d - pad_d
                h_in = h_idx * stride_h + kh * dilation_h - pad_h
                w_in = w_idx * stride_w + kw * dilation_w - pad_w

                # Mask to ignore out‑of‑bounds accesses
                in_mask = (d_in >= 0) & (d_in < D_in) & \
                          (h_in >= 0) & (h_in < H_in) & \
                          (w_in >= 0) & (w_in < W_in)

                # Load all channels
                for c in range(C_in):
                    # Offset in the input tensor
                    offset_in = input_base + c * D_in * H_in * W_in + \
                                d_in * H_in * W_in + h_in * W_in + w_in
                    x_val = tl.load(x_ptr + offset_in, mask=in_mask, other=0.0)

                    # Offset in the weight tensor
                    weight_offset = weight_base + c * K_d * K_h * K_w + \
                                    kd * K_h * K_w + kh * K_w + kw
                    w_val = tl.load(w_ptr + weight_offset)

                    out_val += x_val * w_val

    # ---- Add bias (if present) ----------------------------------------------------
    if bias_present:
        bias_offset = oc_idx
        b_val = tl.load(b_ptr + bias_offset)
        out_val += b_val

    # ---- Store --------------------------------------------------------------------
    tl.store(out_ptr + offsets, out_val, mask=mask)


# --------------------------------------------------------------------------- #
# Triton wrapper that prepares data, launches the kernel, and returns output
# --------------------------------------------------------------------------- #
def conv3d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    depth_in: int,
    width_in: int,
    height_in: int,
    depth_out: int,
    width_out: int,
    height_out: int,
    kernel_d: int,
    kernel_h: int,
    kernel_w: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    pad_d: int,
    pad_h: int,
    pad_w: int,
    dilation_d: int,
    dilation_h: int,
    dilation_w: int,
    bias_present: bool,
    BLOCK_SIZE: int = 128,
) -> torch.Tensor:
    """
    Wrapper that launches the Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    if bias_present:
        bias = bias.contiguous()
    else:
        bias = torch.zeros((out_channels,), dtype=weight.dtype, device=weight.device)

    out = torch.empty(
        (batch_size, out_channels, depth_out, width_out, height_out),
        dtype=weight.dtype,
        device=weight.device,
    )

    total_out = batch_size * out_channels * depth_out * width_out * height_out
    grid = lambda meta: ((total_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    conv3d_kernel[grid](
        x,
        weight,
        bias,
        out,
        batch_size,
        in_channels,
        out_channels,
        depth_in,
        width_in,
        height_in,
        depth_out,
        width_out,
        height_out,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dilation_d,
        dilation_h,
        dilation_w,
        int(bias_present),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# --------------------------------------------------------------------------- #
# Optimized model using the Triton convolution kernel
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Optimized 3D convolution model using a custom Triton kernel.
    The implementation assumes group = 1.
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
        super().__init__()
        assert groups == 1, "Only group = 1 is supported by the Triton kernel."
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that uses the Triton kernel.
        """
        # Parameters for computing output shape
        kernel_d, kernel_h, kernel_w = self.conv3d.kernel_size if isinstance(
            self.conv3d.kernel_size, tuple
        ) else (
            self.conv3d.kernel_size,
            self.conv3d.kernel_size,
            self.conv3d.kernel_size,
        )
        stride_d, stride_h, stride_w = self.conv3d.stride if isinstance(
            self.conv3d.stride, tuple
        ) else (
            self.conv3d.stride,
            self.conv3d.stride,
            self.conv3d.stride,
        )
        pad_d, pad_h, pad_w = self.conv3d.padding if isinstance(
            self.conv3d.padding, tuple
        ) else (
            self.conv3d.padding,
            self.conv3d.padding,
            self.conv3d.padding,
        )
        dil_d, dil_h, dil_w = self.conv3d.dilation if isinstance(
            self.conv3d.dilation, tuple
        ) else (
            self.conv3d.dilation,
            self.conv3d.dilation,
            self.conv3d.dilation,
        )

        # Helper to compute output dimension
        def calc_out_dim(in_dim, pad, dilation, kernel, stride):
            return (in_dim + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1

        B, C_in, D_in, H_in, W_in = x.shape
        D_out = calc_out_dim(D_in, pad_d, dil_d, kernel_d, stride_d)
        H_out = calc_out_dim(H_in, pad_h, dil_h, kernel_h, stride_h)
        W_out = calc_out_dim(W_in, pad_w, dil_w, kernel_w, stride_w)

        return conv3d_triton(
            x,
            self.conv3d.weight,
            self.conv3d.bias,
            batch_size=B,
            in_channels=C_in,
            out_channels=self.conv3d.out_channels,
            depth_in=D_in,
            width_in=W_in,
            height_in=H_in,
            depth_out=D_out,
            width_out=W_out,
            height_out=H_out,
            kernel_d=kernel_d,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_d=stride_d,
            stride_h=stride_h,
            stride_w=stride_w,
            pad_d=pad_d,
            pad_h=pad_h,
            pad_w=pad_w,
            dilation_d=dil_d,
            dilation_h=dil_h,
            dilation_w=dil_w,
            bias_present=self.conv3d.bias is not None,
        )