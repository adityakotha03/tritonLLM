import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    batch_size, out_channels, out_height, out_width,
    in_channels, height, width, kernel_size,
    stride, padding, dilation,
    in_stride_b, in_stride_c, in_stride_h, in_stride_w,
    weight_stride_k, weight_stride_c, weight_stride_r, weight_stride_s,
    out_stride_b, out_stride_k, out_stride_h, out_stride_w,
    bias_stride,
    has_bias: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # 2D block ID
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # Compute output spatial index
    oh = pid_hw // out_width
    ow = pid_hw % out_width

    # Compute input start position (upper-left corner of receptive field)
    ih_start = oh * stride - padding
    iw_start = ow * stride - padding

    # Initialize accumulator for output channel block
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over input channel blocks
    for ic in range(0, in_channels, BLOCK_SIZE_K):
        # Load input patch (BLOCK_SIZE_M x BLOCK_SIZE_K x kh x kw)
        for ih_idx in range(0, kernel_size):
            for iw_idx in range(0, kernel_size):
                ih = ih_start + dilation * ih_idx
                iw = iw_start + dilation * iw_idx

                # Check bounds
                valid_h = (ih >= 0) and (ih < height)
                valid_w = (iw >= 0) and (iw < width)
                valid = valid_h and valid_w

                # Compute input offset
                offset_x = pid_b * in_stride_b + \
                           tl.arange(0, BLOCK_SIZE_M)[:, None] * in_stride_c + \
                           ic + tl.arange(0, BLOCK_SIZE_K)[None, :] + \
                           ih * in_stride_h + iw * in_stride_w
                mask_x = valid and (tl.arange(0, BLOCK_SIZE_M)[:, None] < batch_size) and \
                         (tl.arange(0, BLOCK_SIZE_K)[None, :] < in_channels - ic)
                x = tl.load(x_ptr + offset_x, mask=mask_x, other=0.0)

                # Compute weight offset
                offset_w = pid_k * weight_stride_k + \
                           (ic + tl.arange(0, BLOCK_SIZE_K)[None, :]) * weight_stride_c + \
                           ih_idx * weight_stride_r + iw_idx * weight_stride_s
                mask_w = (tl.arange(0, BLOCK_SIZE_K)[None, :] < in_channels - ic)
                w = tl.load(w_ptr + offset_w, mask=mask_w, other=0.0)

                # Outer product: x @ w.T -> (BLOCK_SIZE_M, BLOCK_SIZE_K) @ (BLOCK_SIZE_K,) -> (BLOCK_SIZE_M,)
                # But we want to accumulate over spatial dims too, so we do elementwise multiply and sum
                acc += tl.sum(x * w[None, :], axis=1)

    # Add bias if present
    if has_bias:
        bias = tl.load(bias_ptr + pid_k * bias_stride)
        acc += bias

    # Store output
    offset_out = pid_b * out_stride_b + pid_k * out_stride_k + oh * out_stride_h + ow * out_stride_w
    mask_out = (tl.arange(0, BLOCK_SIZE_M) < batch_size) and (pid_k < out_channels) and \
               (oh < out_height) and (ow < out_width)
    tl.store(out_ptr + offset_out, acc, mask=mask_out)


def triton_conv2d(x, weight, bias, stride, padding, dilation, groups):
    assert groups == 1, "Grouped convolution not supported in this kernel"
    assert x.is_cuda and weight.is_cuda
    if bias is not None:
        assert bias.is_cuda

    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size_h, kernel_size_w = weight.shape
    kernel_size = kernel_size_h
    assert kernel_size_h == kernel_size_w

    out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    out = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)

    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    # Grid: (batch_size, out_channels, out_height * out_width)
    grid = (batch_size, out_channels, out_height * out_width)

    conv2d_kernel[grid](
        x_ptr=x, w_ptr=weight, bias_ptr=bias, out_ptr=out,
        batch_size=batch_size, out_channels=out_channels, out_height=out_height, out_width=out_width,
        in_channels=in_channels, height=height, width=width, kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        in_stride_b=x.stride(0), in_stride_c=x.stride(1), in_stride_h=x.stride(2), in_stride_w=x.stride(3),
        weight_stride_k=weight.stride(0), weight_stride_c=weight.stride(1), weight_stride_r=weight.stride(2), weight_stride_s=weight.stride(3),
        out_stride_b=out.stride(0), out_stride_k=out.stride(1), out_stride_h=out.stride(2), out_stride_w=out.stride(3),
        bias_stride=bias.stride(0) if bias is not None else 0,
        has_bias=bias is not None,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
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
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)