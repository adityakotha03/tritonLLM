import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Triton kernel for depth‑wise 2D convolution (groups == in_channels)
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def depthwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    out_ptr,
    H, W,
    K, C,              # K = kernel size, C = in_channels = groups
    stride_h_int, stride_w_int,
    padding_h_int, padding_w_int,
    dilation_h_int, dilation_w_int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # each program processes a patch of the output feature map
    program_id = tl.program_id(0)
    # compute output height and width
    out_H = (H + 2 * padding_h - dilation_h * (K - 1) - 1) // stride_h + 1
    out_W = (W + 2 * padding_w - dilation_w * (K - 1) - 1) // stride_w + 1

    # grid: each program works on a tile of (out_H*out_W) positions
    # program_id encodes (oh, ow)
    oh = program_id // out_W
    ow = program_id % out_W

    # load the channel index (since groups==in_channels)
    channel = tl.program_id(1)

    # compute input coordinate for the top-left of the kernel
    h_start = oh * stride_h + padding_h
    w_start = ow * stride_w + padding_w

    # accumulate
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # loop over kernel elements
    for ki in range(K):
        for kj in range(K):
            # input location
            h = h_start + ki * dilation_h
            w = w_start + kj * dilation_w
            mask = (h < H) & (w < W)
            inp_offset = channel * H * W + h * W + w
            inp = tl.load(input_ptr + inp_offset, mask=mask, other=0.0)
            w_offset = channel * K * K + ki * K + kj
            wt = tl.load(weight_ptr + w_offset)
            acc += inp * wt

    # store result
    out_offset = channel * out_H * out_W + oh * out_W + ow
    tl.store(out_ptr + out_offset, acc[0], mask=True)


# --------------------------------------------------------------------------- #
# Triton kernel for point‑wise 1×1 convolution (linear over channel dimension)
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def pointwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    M: tl.constexpr,      # number of spatial positions (H_out * W_out)
    K: tl.constexpr,      # in_channels
    N: tl.constexpr,      # out_channels
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    # compute row and column of the matrix product
    row = pid // BLOCK_N
    col = pid % BLOCK_N

    # offsets for matrix multiply
    offsets_m = row * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = col * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offsets_m < M
    mask_n = offsets_n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        x_offsets = k + tl.arange(0, BLOCK_K)
        w_offsets = k + tl.arange(0, BLOCK_K)

        x_mask = (x_offsets < K) & mask_m[:, None]
        w_mask = (w_offsets < K) & mask_n[None, :]

        x = tl.load(input_ptr + offsets_m[:, None] * K + x_offsets[None, :], mask=x_mask, other=0.0)
        w = tl.load(weight_ptr + offsets_n[None, :] * K + w_offsets[:, None], mask=w_mask, other=0.0)

        acc += tl.dot(x, w)

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offsets_n)
        acc += bias

    tl.store(tl.make_block_ptr(output_ptr, [M, N], [1, M], [0, 0]), acc, mask=mask_n)


# --------------------------------------------------------------------------- #
# Wrapper functions that launch the above kernels
# --------------------------------------------------------------------------- #
def triton_depthwise_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Performs a depth‑wise 2D convolution with Triton."""
    B, C, H, W = input.shape
    K = weight.shape[2]  # kernel size (assuming square)
    assert weight.shape[0] == C and weight.shape[1] == C, "groups must equal in_channels"
    stride_h, stride_w = stride, stride
    padding_h, padding_w = padding, padding
    dilation_h, dilation_w = dilation, dilation

    out_H = (H + 2 * padding_h - dilation_h * (K - 1) - 1) // stride_h + 1
    out_W = (W + 2 * padding_w - dilation_w * (K - 1) - 1) // stride_w + 1

    out = torch.empty((B, C, out_H, out_W), device=input.device, dtype=input.dtype)

    grid = lambda meta: (
        (B * out_H * out_W + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        C,
    )

    depthwise_conv2d_kernel[grid](
        input, weight,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        out,
        H, W,
        K, C,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        BLOCK_M=meta["BLOCK_M"],
        BLOCK_N=meta["BLOCK_N"],
        BLOCK_K=meta["BLOCK_K"],
    )
    return out


def triton_pointwise_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Performs a point‑wise (1×1) convolution with Triton."""
    B, C, H, W = input.shape
    out_C = weight.shape[0]
    M = B * H * W
    K = C
    N = out_C

    input_flat = input.permute(0, 2, 3, 1).reshape(M, K)
    output_flat = torch.empty((M, N), device=input.device, dtype=input.dtype)

    grid = lambda meta: ( (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"], )

    pointwise_conv2d_kernel[grid](
        input_flat,
        weight,
        bias,
        M=M, K=K, N=N,
        BLOCK_M=meta["BLOCK_M"],
        BLOCK_N=meta["BLOCK_N"],
        BLOCK_K=meta["BLOCK_K"],
    )

    out = output_flat.reshape(B, H, W, out_C).permute(0, 3, 1, 2)
    return out


# --------------------------------------------------------------------------- #
# Optimized Model with Triton kernels
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Depth‑wise‑separable 2D convolution with custom Triton kernels.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # depthwise weights: (C, C, K, K) with groups=C
        self.depthwise_weight = nn.Parameter(
            torch.randn(in_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.depthwise_bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.depthwise_bias = None

        # pointwise weights: (out_channels, in_channels, 1, 1)
        self.pointwise_weight = nn.Parameter(
            torch.randn(out_channels, in_channels, 1, 1)
        )
        if bias:
            self.pointwise_bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.pointwise_bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # depth‑wise conv
        dw = triton_depthwise_conv2d(
            x,
            self.depthwise_weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=self.depthwise_bias,
        )
        # point‑wise conv
        pw = triton_pointwise_conv2d(
            dw,
            self.pointwise_weight.squeeze(-1).squeeze(-1),
            bias=self.pointwise_bias,
        )
        return pw