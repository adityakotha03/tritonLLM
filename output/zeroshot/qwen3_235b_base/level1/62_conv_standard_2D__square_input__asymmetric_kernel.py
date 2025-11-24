import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, in_channels, out_channels, height, width, out_height, out_width,
    kernel_h, kernel_w,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    groups,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program IDs
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    # Offsets for output tiles
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Pointers into input and output
    x_group = in_channels // groups
    w_group = out_channels // groups

    group_id = pid_b % groups
    batch_id = pid_b // groups

    off_x = batch_id * in_channels * height * width + group_id * x_group * height * width
    off_w = group_id * w_group * x_group * kernel_h * kernel_w
    off_out = batch_id * out_channels * out_height * out_width + group_id * w_group * out_height * out_width

    # Load input and weights
    x_ptrs = x_ptr + off_x + (
        (offs_m[:, None] // out_width) * width * x_group +
        (offs_m[:, None] % out_width) * x_group +
        (tl.arange(0, BLOCK_SIZE_K // x_group)[None, :] // kernel_w) * dilation_h * width +
        (tl.arange(0, BLOCK_SIZE_K // x_group)[None, :] % kernel_w) * dilation_w
    )
    w_ptrs = w_ptr + off_w + (
        (offs_n[None, :] // kernel_w) * kernel_w * x_group +
        (offs_n[None, :] % kernel_w) * x_group +
        tl.arange(0, BLOCK_SIZE_K // x_group)[:, None]
    )

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Convolution loop
    for k in range(0, x_group * kernel_h * kernel_w, BLOCK_SIZE_K):
        # Bounds for mask
        k_offs = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offs < x_group * kernel_h * kernel_w

        # Load input tile (BLOCK_SIZE_M x BLOCK_SIZE_K)
        x_mask = (offs_m[:, None] < out_height * out_width) & k_mask[None, :]
        x = tl.load(x_ptrs + k_offs[None, :], mask=x_mask, other=0.0)

        # Load weight tile (BLOCK_SIZE_K x BLOCK_SIZE_N)
        w_mask = k_mask[:, None] & (offs_n[None, :] < w_group * kernel_h * kernel_w)
        w = tl.load(w_ptrs + k_offs[:, None], mask=w_mask, other=0.0)

        # Matmul update
        acc += tl.dot(x, w)

        # Update pointers
        x_ptrs += BLOCK_SIZE_K
        w_ptrs += BLOCK_SIZE_K

    # Store result
    out_ptrs = out_ptr + off_out + offs_m[:, None] * out_width + offs_n[None, :]
    out_mask = (offs_m[:, None] < out_height * out_width) & (offs_n[None, :] < w_group * kernel_h * kernel_w)
    tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, dilation: int, groups: int):
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    out_height = (height + 2 * padding - dilation * (kernel_h - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1

    # Output tensor
    out = torch.zeros(batch_size, out_channels, out_height, out_width, device=x.device, dtype=torch.float16)

    # Tile sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    # Grid
    num_m_blocks = (out_height * out_width + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_n_blocks = ((out_channels // groups) * kernel_h * kernel_w + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_b_blocks = batch_size * groups

    grid = (num_b_blocks, num_m_blocks, num_n_blocks)

    conv2d_kernel[grid](
        x, weight, out,
        batch_size, in_channels, out_channels, height, width, out_height, out_width,
        kernel_h, kernel_w,
        stride, stride,
        padding, padding,
        dilation, dilation,
        groups,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    if bias is not None:
        out += bias.view(1, -1, 1, 1)

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

        # Initialize weight and bias parameters
        k_h, k_w = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, k_h, k_w))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Weight initialization
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )