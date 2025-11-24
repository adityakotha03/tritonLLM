import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch_size, out_channels, out_depth, out_height, out_width,
    in_channels, in_depth, in_height, in_width,
    kernel_depth, kernel_height, kernel_width,
    stride_depth, stride_height, stride_width,
    padding_depth, padding_height, padding_width,
    dilation_depth, dilation_height, dilation_width,
    groups,
    input_stride_b, input_stride_c, input_stride_d, input_stride_h, input_stride_w,
    weight_stride_k, weight_stride_g, weight_stride_r, weight_stride_s, weight_stride_t,
    output_stride_b, output_stride_k, output_stride_z, output_stride_h, output_stride_w,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_OUT_CHANNEL: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
    BLOCK_SIZE_IN_CHANNEL: tl.constexpr,
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_RS: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # Compute offsets for output spatial dimensions
    hw_per_block = BLOCK_SIZE_HW
    h_start = (pid_hw * hw_per_block) // out_width
    w_start = (pid_hw * hw_per_block) % out_width

    # Pointers for output
    output_offset_base = (
        pid_b * output_stride_b +
        pid_k * output_stride_k
    )
    output_offsets_zhw = (
        tl.arange(0, BLOCK_SIZE_KD)[:, None, None] * output_stride_z +
        tl.arange(0, BLOCK_SIZE_HW // out_width + 1)[None, :, None] * output_stride_h +
        tl.arange(0, min(BLOCK_SIZE_HW, out_width))[None, None, :]
    )
    output_mask_zhw = (
        (tl.arange(0, BLOCK_SIZE_KD)[:, None, None] < out_depth) &
        (tl.arange(0, BLOCK_SIZE_HW // out_width + 1)[None, :, None] < out_height) &
        (tl.arange(0, min(BLOCK_SIZE_HW, out_width))[None, None, :] < out_width)
    )

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_KD, BLOCK_SIZE_HW // out_width + 1, min(BLOCK_SIZE_HW, out_width)), dtype=tl.float32)

    # Loop over input channels and group
    group_id = pid_k // (out_channels // groups)
    out_channel_local = pid_k % (out_channels // groups)

    # Weight pointer for this group and output channel
    weight_offset_base = (
        pid_k * weight_stride_k +
        group_id * weight_stride_g
    )

    for ic_group in range(0, in_channels // groups, BLOCK_SIZE_IN_CHANNEL):
        # Load weights: [BLOCK_SIZE_KD, BLOCK_SIZE_RS, BLOCK_SIZE_IN_CHANNEL]
        r_offsets = tl.arange(0, kernel_depth)[:, None, None] * dilation_depth
        s_offsets = tl.arange(0, kernel_height)[None, :, None] * dilation_height
        t_offsets = tl.arange(0, kernel_width)[None, None, :] * dilation_width
        kdrt_offsets = (
            r_offsets * weight_stride_r +
            s_offsets * weight_stride_s +
            t_offsets * weight_stride_t
        )
        weight_mask = (
            (tl.arange(0, kernel_depth)[:, None, None] < kernel_depth) &
            (tl.arange(0, kernel_height)[None, :, None] < kernel_height) &
            (tl.arange(0, kernel_width)[None, None, :] < kernel_width)
        )
        weight_ptrs = weight_ptr + weight_offset_base + kdrt_offsets
        weight = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

        # Accumulate over input channels in this block
        for ic in range(ic_group, min(ic_group + BLOCK_SIZE_IN_CHANNEL, in_channels // groups)):
            # Compute input offsets
            input_offset_base = (
                pid_b * input_stride_b +
                (group_id * (in_channels // groups) + ic) * input_stride_c
            )
            for dz in range(kernel_depth):
                for dh in range(kernel_height):
                    for dw in range(kernel_width):
                        # Compute input spatial indices
                        iz = dz * dilation_depth - padding_depth + tl.arange(0, BLOCK_SIZE_KD)[:, None, None]
                        ih = dh * dilation_height - padding_height + tl.arange(0, BLOCK_SIZE_HW // out_width + 1)[None, :, None]
                        iw = dw * dilation_width - padding_width + tl.arange(0, min(BLOCK_SIZE_HW, out_width))[None, None, :]

                        # Input mask
                        input_mask = (
                            (iz >= 0) & (iz < in_depth) &
                            (ih >= 0) & (ih < in_height) &
                            (iw >= 0) & (iw < in_width)
                        )
                        input_offsets = (
                            input_offset_base +
                            iz * input_stride_d +
                            ih * input_stride_h +
                            iw * input_stride_w
                        )
                        input_ptrs = input_ptr + input_offsets
                        input_vals = tl.load(input_ptrs, mask=input_mask, other=0.0)

                        # Weight value
                        w_val = weight[dz, dh, dw]

                        # Accumulate
                        acc += input_vals * w_val

        # End of input channel block

    # Store output
    output_ptrs = output_ptr + output_offset_base + output_offsets_zhw
    tl.store(output_ptrs, acc, mask=output_mask_zhw)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        self.bias = bias

        # Initialize weight
        k_d, k_h, k_w = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, groups, k_d, k_h, k_w))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias_param', None)

        # Precompute strides for weight
        self.weight_stride_k = self.weight.stride(0)
        self.weight_stride_g = self.weight.stride(1)
        self.weight_stride_r = self.weight.stride(2)
        self.weight_stride_s = self.weight.stride(3)
        self.weight_stride_t = self.weight.stride(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on CUDA
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()

        batch_size, in_channels, in_depth, in_height, in_width = x.shape
        out_channels = self.out_channels
        k_d, k_h, k_w = self.kernel_size
        s_d, s_h, s_w = self.stride
        p_d, p_h, p_w = self.padding
        d_d, d_h, d_w = self.dilation
        groups = self.groups

        # Compute output spatial dimensions
        out_depth = (in_depth + 2 * p_d - d_d * (k_d - 1) - 1) // s_d + 1
        out_height = (in_height + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
        out_width = (in_width + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1

        # Output tensor
        output = torch.empty(batch_size, out_channels, out_depth, out_height, out_width, device=x.device, dtype=x.dtype)

        # Strides
        input_strides = x.stride()
        output_strides = output.stride()

        # Launch kernel
        def grid(meta):
            return (
                triton.cdiv(batch_size, meta['BLOCK_SIZE_BATCH']),
                triton.cdiv(out_channels, meta['BLOCK_SIZE_OUT_CHANNEL']),
                triton.cdiv(out_depth * out_height * out_width, meta['BLOCK_SIZE_HW'])
            )

        # Autotune blocking configurations
        @triton.autotune(
            configs=[
                triton.Config({'BLOCK_SIZE_BATCH': 1, 'BLOCK_SIZE_OUT_CHANNEL': 16, 'BLOCK_SIZE_HW': 64, 'BLOCK_SIZE_IN_CHANNEL': 16, 'BLOCK_SIZE_KD': 4, 'BLOCK_SIZE_RS': 16}, num_stages=3, num_warps=4),
                triton.Config({'BLOCK_SIZE_BATCH': 1, 'BLOCK_SIZE_OUT_CHANNEL': 32, 'BLOCK_SIZE_HW': 64, 'BLOCK_SIZE_IN_CHANNEL': 16, 'BLOCK_SIZE_KD': 4, 'BLOCK_SIZE_RS': 16}, num_stages=3, num_warps=4),
                triton.Config({'BLOCK_SIZE_BATCH': 1, 'BLOCK_SIZE_OUT_CHANNEL': 16, 'BLOCK_SIZE_HW': 128, 'BLOCK_SIZE_IN_CHANNEL': 16, 'BLOCK_SIZE_KD': 4, 'BLOCK_SIZE_RS': 16}, num_stages=3, num_warps=4),
            ],
            key=['in_channels', 'out_channels', 'out_depth', 'out_height', 'out_width'],
        )
        @triton.jit
        def _kernel(*args, **meta):
            conv3d_kernel(*args, **meta)

        _kernel[
            grid
        ](
            x, weight, output,
            batch_size, out_channels, out_depth, out_height, out_width,
            in_channels, in_depth, in_height, in_width,
            k_d, k_h, k_w,
            s_d, s_h, s_w,
            p_d, p_h, p_w,
            d_d, d_h, d_w,
            groups,
            input_strides[0], input_strides[1], input_strides[2], input_strides[3], input_strides[4],
            self.weight_stride_k, self.weight_stride_g, self.weight_stride_r, self.weight_stride_s, self.weight_stride_t,
            output_strides[0], output_strides[1], output_strides[2], output_strides[3], output_strides[4],
        )

        # Add bias if needed
        if self.bias:
            output += self.bias_param.view(1, -1, 1, 1, 1)

        return output