import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_transpose2d_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch, in_channels, out_channels, input_height, input_width,
    output_height, output_width, kernel_size,
    stride, padding, output_padding,
    in_stride_b, in_stride_c, in_stride_h, in_stride_w,
    weight_stride_c, weight_stride_k, weight_stride_r, weight_stride_s,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Compute program indices
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # Calculate offsets for output spatial dimensions
    oh_start = pid_hw * BLOCK_SIZE_HW
    ow_start = 0
    oh_range = oh_start + tl.arange(0, BLOCK_SIZE_HW)
    ow_range = tl.arange(0, 1)

    # Output height and width bounds
    oh_mask = oh_range < output_height
    ow_mask = ow_range < output_width
    mask_hw = oh_mask and ow_mask

    # Loop over input channels in blocks
    for ic in range(0, in_channels, BLOCK_SIZE_IC):
        for b in range(0, batch, BLOCK_SIZE_BATCH):
            # Load input block (batch, in_channels, input_height, input_width)
            # We will compute output using implicit gemm: output[b, oc, oh, ow] += sum_{ic, kh, kw} input[b, ic, ih, iw] * weight[oc, ic, kh, kw]
            # where ih = (oh - kh + padding) // stride, iw = (ow - kw + padding) // stride

            # Iterate over kernel spatial dimensions
            for kh in range(0, kernel_size):
                for kw in range(0, kernel_size):
                    # Compute input spatial coordinates
                    ih = (oh_range - kh + padding) // stride
                    iw = (ow_range - kw + padding) // stride

                    # Check if input coordinates are valid
                    ih_valid = (ih >= 0) & (ih < input_height)
                    iw_valid = (iw >= 0) & (iw < input_width)
                    valid = ih_valid & iw_valid & mask_hw

                    # Load input values
                    input_offsets = (
                        (b + tl.arange(0, BLOCK_SIZE_BATCH))[:, None] * in_stride_b +
                        (ic + tl.arange(0, BLOCK_SIZE_IC)[None, :])[:, None] * in_stride_c +
                        ih[None, :] * in_stride_h +
                        iw[None, :]
                    )
                    input_vals = tl.load(input_ptr + input_offsets, mask=valid[None, :], other=0.0)

                    # Load weight values
                    weight_offsets = (
                        (pid_oc + tl.arange(0, BLOCK_SIZE_OC)[:, None]) * weight_stride_c +
                        (ic + tl.arange(0, BLOCK_SIZE_IC)[None, :])[:, None] * weight_stride_k +
                        kh * weight_stride_r +
                        kw * weight_stride_s
                    )
                    weight_vals = tl.load(weight_ptr + weight_offsets, mask=None, other=0.0)

                    # Perform outer product and accumulate
                    output_vals = tl.dot(weight_vals, input_vals, out_dtype=tl.float32)

                    # Accumulate into output
                    output_offsets = (
                        (b + tl.arange(0, BLOCK_SIZE_BATCH))[:, None] * out_stride_b +
                        (pid_oc + tl.arange(0, BLOCK_SIZE_OC)[:, None]) * out_stride_c +
                        oh_range[None, :] * out_stride_h +
                        ow_range[None, :]
                    )
                    tl.atomic_add(output_ptr + output_offsets, output_vals, mask=valid[None, :])


# Fused ConvTranspose2d + GELU (optional activation fusion)
@triton.jit
def _conv_transpose2d_gelu_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch, in_channels, out_channels, input_height, input_width,
    output_height, output_width, kernel_size,
    stride, padding, output_padding,
    in_stride_b, in_stride_c, in_stride_h, in_stride_w,
    weight_stride_c, weight_stride_k, weight_stride_r, weight_stride_s,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oh_start = pid_hw * BLOCK_SIZE_HW
    oh_range = oh_start + tl.arange(0, BLOCK_SIZE_HW)
    oh_mask = oh_range < output_height
    mask_hw = oh_mask

    for ic in range(0, in_channels, BLOCK_SIZE_IC):
        for b in range(0, batch, BLOCK_SIZE_BATCH):
            acc = tl.zeros((BLOCK_SIZE_OC, BLOCK_SIZE_BATCH * BLOCK_SIZE_HW), dtype=tl.float32)
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    ih = (oh_range - kh + padding) // stride
                    iw = (0 - kw + padding) // stride  # assuming width=1 for this tile

                    ih_valid = (ih >= 0) & (ih < input_height)
                    iw_valid = (iw >= 0) & (iw < input_width)
                    valid = ih_valid & iw_valid & mask_hw

                    input_offsets = (
                        (b + tl.arange(0, BLOCK_SIZE_BATCH))[:, None] * in_stride_b +
                        (ic + tl.arange(0, BLOCK_SIZE_IC)[None, :])[:, None] * in_stride_c +
                        ih[None, :] * in_stride_h +
                        iw
                    )
                    input_vals = tl.load(input_ptr + input_offsets, mask=valid[None, :], other=0.0)

                    weight_offsets = (
                        (pid_oc + tl.arange(0, BLOCK_SIZE_OC)[:, None]) * weight_stride_c +
                        (ic + tl.arange(0, BLOCK_SIZE_IC)[None, :])[:, None] * weight_stride_k +
                        kh * weight_stride_r +
                        kw * weight_stride_s
                    )
                    weight_vals = tl.load(weight_ptr + weight_offsets, mask=None, other=0.0)

                    # Compute contribution
                    contrib = tl.dot(weight_vals, input_vals, out_dtype=tl.float32)
                    acc += contrib

            # Apply GELU activation: x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            x = acc
            x_cubed = x * x * x
            inner = 0.7978845608028654 * (x + 0.044715 * x_cubed)
            tanh_inner = tl.tanh(inner)
            gelu_output = 0.5 * x * (1.0 + tanh_inner)

            output_offsets = (
                (b + tl.arange(0, BLOCK_SIZE_BATCH))[:, None] * out_stride_b +
                (pid_oc + tl.arange(0, BLOCK_SIZE_OC)[:, None]) * out_stride_c +
                oh_range[None, :] * out_stride_h +
                0 * out_stride_w
            )
            tl.store(output_ptr + output_offsets, gelu_output, mask=mask_hw[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Create weight tensor
        k = 1.0 / (in_channels * kernel_size * kernel_size)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * k)

        if bias:
            self.bias_tensor = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_buffer('bias_tensor', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        batch, in_channels, input_height, input_width = x.shape
        assert in_channels == self.in_channels

        # Compute output spatial dimensions
        output_height = (input_height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        output_width = (input_width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        out = torch.zeros(batch, self.out_channels, output_height, output_width, device=x.device, dtype=x.dtype)

        # Handle grouped convolutions
        assert self.groups == 1, "Grouped transposed convolution not supported in this kernel"
        assert self.output_padding == 0, "Non-zero output_padding not supported in this kernel"

        # Define block sizes
        BLOCK_SIZE_BATCH = 1
        BLOCK_SIZE_IC = 16
        BLOCK_SIZE_OC = 16
        BLOCK_SIZE_HW = 16

        # Compute number of blocks
        num_pid_b = triton.cdiv(batch, BLOCK_SIZE_BATCH)
        num_pid_oc = triton.cdiv(self.out_channels, BLOCK_SIZE_OC)
        num_pid_hw = triton.cdiv(output_height, BLOCK_SIZE_HW)

        # Define grid
        grid = (num_pid_b, num_pid_oc, num_pid_hw)

        # Launch kernel
        _conv_transpose2d_kernel[grid](
            x, self.weight, out,
            batch, in_channels, self.out_channels, input_height, input_width,
            output_height, output_width, self.kernel_size,
            self.stride, self.padding, self.output_padding,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.weight.stride(0), self.weight.stride(1), self.weight.stride(2), self.weight.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
            BLOCK_SIZE_IC=BLOCK_SIZE_IC,
            BLOCK_SIZE_OC=BLOCK_SIZE_OC,
            BLOCK_SIZE_HW=BLOCK_SIZE_HW,
        )

        # Add bias if present
        if self.bias:
            out = out + self.bias_tensor.view(1, -1, 1, 1)

        return out