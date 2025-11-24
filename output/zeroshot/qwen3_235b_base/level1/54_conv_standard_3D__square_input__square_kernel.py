import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 2, 'BLOCK_H': 16, 'BLOCK_W': 16, 'BLOCK_K': 64, 'GROUP_H': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_T': 2, 'BLOCK_H': 16, 'BLOCK_W': 32, 'BLOCK_K': 64, 'GROUP_H': 4}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_T': 2, 'BLOCK_H': 32, 'BLOCK_W': 16, 'BLOCK_K': 64, 'GROUP_H': 4}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_T': 4, 'BLOCK_H': 16, 'BLOCK_W': 16, 'BLOCK_K': 64, 'GROUP_H': 8}, num_stages=3, num_warps=4),
    ],
    key=['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'depth', 'height', 'width'],
)
@triton.jit
def conv3d_kernel(
    input_ptr, weight_ptr, output_ptr,
    in_channels, out_channels, depth, height, width,
    kernel_size, stride, padding, dilation,
    batch_size, out_d, out_h, out_w,
    input_stride_b, input_stride_c, input_stride_t, input_stride_h, input_stride_w,
    weight_stride_k, weight_stride_c, weight_stride_t, weight_stride_h, weight_stride_w,
    output_stride_b, output_stride_k, output_stride_t, output_stride_h, output_stride_w,
    BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_H: tl.constexpr
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_out_t = tl.program_id(2)
    pid_out_h = tl.program_id(3) // GROUP_H
    pid_out_w = tl.program_id(3) % GROUP_H

    # Compute output spatial indices
    t = pid_out_t * BLOCK_T + tl.arange(0, BLOCK_T)
    h = pid_out_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w = pid_out_w * BLOCK_W + tl.arange(0, BLOCK_W)

    # Masks for spatial bounds
    t_mask = t < out_d
    h_mask = h < out_h
    w_mask = w < out_w
    mask_hw = h_mask[:, None] & w_mask[None, :]  # [BLOCK_H, BLOCK_W]

    # Initialize output accumulator
    acc = tl.zeros([BLOCK_T, BLOCK_H, BLOCK_W], dtype=tl.float32)

    # Loop over input channels and kernel space
    for ic in range(0, in_channels):
        for kt in range(0, kernel_size):
            for kh in range(0, kernel_size):
                for kw in range(0, kernel_size):
                    # Compute input spatial coordinates
                    in_t = t * stride - padding + kt * dilation
                    in_h = h * stride - padding + kh * dilation
                    in_w = w * stride - padding + kw * dilation

                    # Input coordinate masks
                    in_t_mask = (in_t >= 0) & (in_t < depth) & t_mask[:, None, None]
                    in_h_mask = (in_h >= 0) & (in_h < height)
                    in_w_mask = (in_w >= 0) & (in_w < width)
                    in_mask = in_t_mask & in_h_mask[:, None] & in_w_mask[None, :]

                    # Load input tile: [BLOCK_T, BLOCK_H, BLOCK_W]
                    input_offset = (
                        pid_b * input_stride_b +
                        ic * input_stride_c +
                        in_t * input_stride_t[:, None, None] +
                        in_h[None, :, None] * input_stride_h +
                        in_w[None, None, :] * input_stride_w
                    )
                    input_val = tl.load(input_ptr + input_offset, mask=in_mask, other=0.0)

                    # Load weight: scalar
                    weight_offset = (
                        pid_k * weight_stride_k +
                        ic * weight_stride_c +
                        kt * weight_stride_t +
                        kh * weight_stride_h +
                        kw * weight_stride_w
                    )
                    weight_val = tl.load(weight_ptr + weight_offset)

                    # Accumulate: outer product
                    acc += input_val.to(tl.float32) * weight_val.to(tl.float32)

    # Write back output
    output_offset = (
        pid_b * output_stride_b +
        pid_k * output_stride_k +
        t[:, None, None] * output_stride_t +
        h[None, :, None] * output_stride_h +
        w[None, None, :] * output_stride_w
    )
    output_mask = t_mask[:, None, None] & mask_hw[None, :, :]
    tl.store(output_ptr + output_offset, acc, mask=output_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        if groups != 1:
            raise NotImplementedError("Grouped 3D convolution not supported in this kernel")
        if bias:
            raise NotImplementedError("Bias not supported in this kernel")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Initialize weight
        k = 1.0 / (in_channels * kernel_size ** 3)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        nn.init.uniform_(self.weight, -k**0.5, k**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input dimensions
        batch_size, in_channels, depth, height, width = x.shape
        assert in_channels == self.in_channels, "Input channel count mismatch"

        # Output dimensions
        out_d = (depth + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_h = (height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_w = (width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        # Output tensor
        out = torch.empty(batch_size, self.out_channels, out_d, out_h, out_w, dtype=torch.float32, device=x.device)

        # Strides
        input_strides = x.stride()
        weight_strides = self.weight.stride()
        output_strides = out.stride()

        # Grid configuration
        grid = lambda meta: (
            batch_size,
            self.out_channels,
            triton.cdiv(out_d, meta['BLOCK_T']),
            triton.cdiv(out_h, meta['BLOCK_H']) * meta['GROUP_H'],
            triton.cdiv(out_w, meta['BLOCK_W']) * (meta['GROUP_H'] if meta['GROUP_H'] > 1 else 1)
        )

        # Launch kernel
        conv3d_kernel[grid](
            x, self.weight, out,
            in_channels, self.out_channels, depth, height, width,
            self.kernel_size, self.stride, self.padding, self.dilation,
            batch_size, out_d, out_h, out_w,
            *input_strides,
            *weight_strides,
            *output_strides
        )
        return out