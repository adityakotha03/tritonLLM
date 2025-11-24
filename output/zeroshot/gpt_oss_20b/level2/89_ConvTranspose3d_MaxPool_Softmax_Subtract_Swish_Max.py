import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------- Triton kernels ----------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=16),
    ],
    key=["N", "C"],
)
@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    N,            # number of spatial elements (depth*height*width)
    C,            # number of channels
    stride_x,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes BLOCK_SIZE spatial elements
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load the C channel values for each spatial location
    max_vals = tl.zeros([C], dtype=tl.float32)
    sum_exp = tl.zeros([C], dtype=tl.float32)

    for i in range(C):
        idx = offsets * stride_x + i * stride_out
        val = tl.load(x_ptr + idx, mask=mask, other=0.0)
        max_vals[i] = tl.maximum(max_vals[i], val)

    # Compute exp(x - max) and sum
    for i in range(C):
        idx = offsets * stride_x + i * stride_out
        val = tl.load(x_ptr + idx, mask=mask, other=0.0)
        exp_val = tl.exp(val - max_vals[i])
        sum_exp[i] = tl.sum(exp_val * mask)

    # Final softmax
    for i in range(C):
        idx = offsets * stride_x + i * stride_out
        val = tl.load(x_ptr + idx, mask=mask, other=0.0)
        exp_val = tl.exp(val - max_vals[i])
        out = exp_val / sum_exp[i]
        tl.store(out_ptr + idx, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["N", "C"],
)
@triton.jit
def swish_subtract_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    N,
    C,
    stride_x,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    for i in range(C):
        idx = offsets * stride_x + i * stride_out
        val = tl.load(x_ptr + idx, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + i)
        sub = val - bias
        swish = sub * tl.sigmoid(sub)
        tl.store(out_ptr + idx, swish, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["N", "C"],
)
@triton.jit
def max_channel_kernel(
    x_ptr,
    out_ptr,
    N,
    C,
    stride_x,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    max_vals = tl.full([C], -float("inf"), dtype=tl.float32)

    for i in range(C):
        idx = offsets * stride_x + i * stride_out
        val = tl.load(x_ptr + idx, mask=mask, other=-float("inf"))
        max_vals[i] = tl.maximum(max_vals[i], val)

    # Reduce across channels
    max_val = max_vals[0]
    for i in range(1, C):
        max_val = tl.maximum(max_val, max_vals[i])

    for i in range(C):
        idx = offsets * stride_x + i * stride_out
        tl.store(out_ptr + idx, max_val, mask=mask)


# ---------- Helper wrappers ----------

def triton_softmax(x: torch.Tensor):
    # x: (B, C, D, H, W)
    B, C, D, H, W = x.shape
    N = D * H * W
    out = torch.empty_like(x)
    stride_x = C * D * H * W
    stride_out = C * D * H * W
    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    softmax_kernel[grid](
        x, out, N, C, stride_x, stride_out, BLOCK_SIZE=128
    )
    return out


def triton_swish_subtract(x: torch.Tensor, bias: torch.Tensor):
    B, C, D, H, W = x.shape
    N = D * H * W
    out = torch.empty_like(x)
    stride_x = C * D * H * W
    stride_out = C * D * H * W
    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    swish_subtract_kernel[grid](
        x, bias, out, N, C, stride_x, stride_out, BLOCK_SIZE=128
    )
    return out


def triton_max_channel(x: torch.Tensor):
    B, C, D, H, W = x.shape
    N = D * H * W
    out = torch.empty_like(x)
    stride_x = C * D * H * W
    stride_out = C * D * H * W
    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    max_channel_kernel[grid](
        x, out, N, C, stride_x, stride_out, BLOCK_SIZE=128
    )
    return out


# ---------- Optimized model ----------

class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for:
        - Softmax across channels
        - Swish with channel-wise bias subtraction
        - Max over channels
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        pool_kernel_size,
        pool_stride,
        pool_padding,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.max_pool = nn.MaxPool3d(
            kernel_size=pool_kernel_size,
            stride=pool_stride,
            padding=pool_padding,
        )
        # Channel-wise bias for subtraction
        self.subtract = nn.Parameter(
            torch.randn(out_channels, device="cuda")
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        x = triton_softmax(x)
        x = triton_swish_subtract(x, self.subtract)
        x = triton_max_channel(x)
        return x