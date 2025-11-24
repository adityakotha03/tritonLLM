import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, out_channels, out_height, out_width,
    in_channels, height, width,
    kernel_h, kernel_w,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_k, weight_stride_c, weight_stride_r, weight_stride_s,
    output_stride_b, output_stride_k, output_stride_h, output_stride_w,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_OUT_CH: tl.constexpr,
    BLOCK_SIZE_OUT_H: tl.constexpr,
    BLOCK_SIZE_OUT_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_w = tl.program_id(axis=3)

    # Compute starting indices and block sizes
    batch_start = pid_b * BLOCK_SIZE_BATCH
    ch_start = pid_k * BLOCK_SIZE_OUT_CH
    h_start = pid_h * BLOCK_SIZE_OUT_H
    w_start = pid_w * BLOCK_SIZE_OUT_W

    # Define offsets for output
    offsets_b = batch_start + tl.arange(0, BLOCK_SIZE_BATCH)
    offsets_k = ch_start + tl.arange(0, BLOCK_SIZE_OUT_CH)
    offsets_h = h_start + tl.arange(0, BLOCK_SIZE_OUT_H)
    offsets_w = w_start + tl.arange(0, BLOCK_SIZE_OUT_W)

    # Mask for valid batches and output channels
    mask_b = offsets_b < batch_size
    mask_k = offsets_k < out_channels
    mask_h = offsets_h < out_height
    mask_w = offsets_w < out_width

    # Full mask
    mask = mask_b[:, None, None, None] & mask_k[None, :, None, None] & \
           mask_h[None, None, :, None] & mask_w[None, None, None, :]

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_CH, BLOCK_SIZE_OUT_H, BLOCK_SIZE_OUT_W), dtype=tl.float32)

    # Loop over input channels and kernel dimensions
    for ic in range(0, in_channels):
        for r in range(0, kernel_h):
            for s in range(0, kernel_w):
                # Compute input spatial location
                h_im = offsets_h * stride_h - padding_h + r * dilation_h
                w_im = offsets_w * stride_w - padding_w + s * dilation_w

                # Bounds check
                h_mask = (h_im >= 0) & (h_im < height)
                w_mask = (w_im >= 0) & (w_im < width)
                im_mask = h_mask[None, None, :, :] & w_mask[None, None, :, :]

                # Load input: (BLOCK_SIZE_BATCH, in_channels, BLOCK_SIZE_OUT_H, BLOCK_SIZE_OUT_W)
                x_offsets = (offsets_b[:, None, None, None] * input_stride_b +
                             ic * input_stride_c +
                             h_im[None, None, :, None] * input_stride_h +
                             w_im[None, None, None, :] * input_stride_w)
                x = tl.load(x_ptr + x_offsets, mask=mask & im_mask[None, :, :, :], other=0.0)

                # Load weight: (out_channels, in_channels, kernel_h, kernel_w)
                w_offset = (offsets_k * weight_stride_k +
                            ic * weight_stride_c +
                            r * weight_stride_r +
                            s * weight_stride_s)
                w = tl.load(w_ptr + w_offset, mask=mask_k, other=0.0)

                # Multiply and accumulate
                acc += x * w[:, None, None, None]

    # Store output
    output_offsets = (offsets_b[:, None, None, None] * output_stride_b +
                      offsets_k[None, :, None, None] * output_stride_k +
                      offsets_h[None, None, :, None] * output_stride_h +
                      offsets_w[None, None, None, :] * output_stride_w)
    tl.store(out_ptr + output_offsets, acc, mask=mask)


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                  stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1)):
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else (stride[0], stride[1])
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    out_height = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    # Output tensor
    out = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)

    # Strides
    input_stride_b, input_stride_c, input_stride_h, input_stride_w = x.stride()
    weight_stride_k, weight_stride_c, weight_stride_r, weight_stride_s = weight.stride()
    output_stride_b, output_stride_k, output_stride_h, output_stride_w = out.stride()

    # Launch grid
    def grid(meta):
        return (
            triton.cdiv(batch_size, meta['BLOCK_SIZE_BATCH']),
            triton.cdiv(out_channels, meta['BLOCK_SIZE_OUT_CH']),
            triton.cdiv(out_height, meta['BLOCK_SIZE_OUT_H']),
            triton.cdiv(out_width, meta['BLOCK_SIZE_OUT_W']),
        )

    # Autotune block sizes
    conv2d_kernel[grid](
        x, weight, out,
        batch_size, out_channels, out_height, out_width,
        in_channels, height, width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        input_stride_b, input_stride_c, input_stride_h, input_stride_w,
        weight_stride_k, weight_stride_c, weight_stride_r, weight_stride_s,
        output_stride_b, output_stride_k, output_stride_h, output_stride_w,
        BLOCK_SIZE_BATCH=4,
        BLOCK_SIZE_OUT_CH=16,
        BLOCK_SIZE_OUT_H=16,
        BLOCK_SIZE_OUT_W=32,
        BLOCK_SIZE_K=32,
        BLOCK_SIZE_R=5,
    )

    if bias is not None:
        out += bias.view(1, -1, 1, 1)

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)