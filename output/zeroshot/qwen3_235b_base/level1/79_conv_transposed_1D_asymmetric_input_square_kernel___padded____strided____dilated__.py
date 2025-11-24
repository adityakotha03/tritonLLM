import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_transpose1d_kernel(
    x_ptr,           # pointer to input tensor (batch, in_channels, length)
    weight_ptr,      # pointer to weight tensor (in_channels, out_channels, kernel_size)
    bias_ptr,        # pointer to bias (out_channels,)
    out_ptr,         # pointer to output tensor (batch, out_channels, length_out)
    batch: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    length: tl.constexpr,
    length_out: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    block_size_k: tl.constexpr,
    block_size_n: tl.constexpr,
    has_bias: tl.constexpr,
):
    # 2D block: each block handles a tile of (block_size_k, block_size_n) in (in_channels, out_channels)
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    # offsets for output channels and input channels
    k_offset = pid_k * block_size_k + tl.arange(0, block_size_k)
    n_offset = pid_n * block_size_n + tl.arange(0, block_size_n)
    k_mask = k_offset < in_channels
    n_mask = n_offset < out_channels

    # pre-compute base offsets for weight
    weight_base = weight_ptr + k_offset[:, None] * (out_channels * kernel_size) + n_offset[None, :] * kernel_size
    weight_mask = k_mask[:, None] & n_mask[None, :]

    # loop over batches
    for b in range(batch):
        # base pointer for input of this batch
        x_base = x_ptr + b * in_channels * length + k_offset[:, None] * length
        # base pointer for output of this batch
        out_base = out_ptr + b * out_channels * length_out + n_offset[None, :] * length_out

        # loop over output time steps (length_out)
        for o_idx in range(0, length_out):
            # compute the range of input indices that contribute to output position o_idx
            # input_idx = (o_idx - padding) // stride + dilation * (kernel_idx - 1)
            # but we reverse the logic: for each kernel position, compute where it lands
            val = tl.zeros((block_size_k, block_size_n), dtype=tl.float32)

            for ki in range(0, kernel_size):
                # compute input index
                i_idx = (o_idx - padding) + ki * dilation
                # check if this index is valid
                if 0 <= i_idx < length:
                    # load input: shape (block_size_k,)
                    x_data = tl.load(x_base + i_idx, mask=k_mask, other=0.0)
                    # load weights: shape (block_size_k, block_size_n)
                    w_data = tl.load(weight_base + ki, mask=weight_mask, other=0.0)
                    # fused multiply-add
                    val += x_data[:, None] * w_data

            # write output
            tl.store(out_base + o_idx, val.to(out_ptr.dtype.element_ty), mask=n_mask)

        # add bias if present
        if has_bias and pid_k == 0:
            bias_data = tl.load(bias_ptr + n_offset, mask=n_mask, other=0.0) if has_bias else 0.0
            # write bias to all output time steps
            for o_idx in range(length_out):
                tl.store(out_ptr + b * out_channels * length_out + n_offset * length_out + o_idx, 
                         bias_data, mask=n_mask)


def triton_conv_transpose1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int
):
    batch, in_channels, length = x.shape
    _, out_channels, kernel_size = weight.shape

    # compute output length
    length_out = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1

    # allocate output
    out = torch.zeros(batch, out_channels, length_out, dtype=x.dtype, device=x.device)

    # constants
    block_size_k = triton.next_power_of_2(in_channels)
    block_size_k = min(max(block_size_k, 16), 128)
    block_size_n = triton.next_power_of_2(out_channels)
    block_size_n = min(max(block_size_n, 16), 128)

    # grid
    grid = (triton.cdiv(in_channels, block_size_k), triton.cdiv(out_channels, block_size_n))

    # launch kernel
    _conv_transpose1d_kernel[grid](
        x, weight, bias, out,
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        length=length,
        length_out=length_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        block_size_k=block_size_k,
        block_size_n=block_size_n,
        has_bias=(bias is not None)
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized version of transposed 1D convolution using Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Initialize weight and optional bias
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)