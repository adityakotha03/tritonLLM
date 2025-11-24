import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    batch_size, out_channels, length_out, in_channels, input_length,
    kernel_size, stride, padding, dilation,
    input_stride_b, input_stride_c, input_stride_l,
    weight_stride_c, weight_stride_k,
    output_stride_b, output_stride_c, output_stride_l,
    has_bias: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_OUT_CH: tl.constexpr,
    BLOCK_SIZE_LENGTH: tl.constexpr,
    BLOCK_SIZE_IN_CH: tl.constexpr,
    BLOCK_SIZE_KERNEL: tl.constexpr,
):
    # Compute program ids
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_l = tl.program_id(2)

    # Compute offsets for output tile
    batch_start = pid_b * BLOCK_SIZE_BATCH
    ch_start = pid_c * BLOCK_SIZE_OUT_CH
    length_start = pid_l * BLOCK_SIZE_LENGTH

    # Load bias if present
    bias_offset = ch_start + tl.arange(0, BLOCK_SIZE_OUT_CH)
    bias_mask = bias_offset < out_channels
    if has_bias:
        bias = tl.load(bias_ptr + bias_offset, mask=bias_mask, other=0.0)
    else:
        bias = tl.zeros([BLOCK_SIZE_OUT_CH], dtype=tl.float32)

    # Define output offsets
    output_offsets = (
        (batch_start + tl.arange(0, BLOCK_SIZE_BATCH))[:, None, None] * output_stride_b +
        (ch_start + tl.arange(0, BLOCK_SIZE_OUT_CH))[None, :, None] * output_stride_c +
        (length_start + tl.arange(0, BLOCK_SIZE_LENGTH))[None, None, :] * output_stride_l
    )
    output_mask = (
        (batch_start + tl.arange(0, BLOCK_SIZE_BATCH))[:, None, None] < batch_size
    ) & (
        (ch_start + tl.arange(0, BLOCK_SIZE_OUT_CH))[None, :, None] < out_channels
    ) & (
        (length_start + tl.arange(0, BLOCK_SIZE_LENGTH))[None, None, :] < length_out
    )

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_CH, BLOCK_SIZE_LENGTH), dtype=tl.float32)

    # Loop over input channels and kernel positions
    for ic in range(0, in_channels, BLOCK_SIZE_IN_CH):
        for k in range(0, kernel_size, BLOCK_SIZE_KERNEL):
            # Compute input and weight offsets
            input_ch_start = ic
            input_ch_end = min(ic + BLOCK_SIZE_IN_CH, in_channels)
            kernel_start = k
            kernel_end = min(k + BLOCK_SIZE_KERNEL, kernel_size)

            # Input offsets: (BLOCK_SIZE_BATCH, input_ch_block, kernel_block, BLOCK_SIZE_LENGTH)
            input_offsets = tl.zeros((BLOCK_SIZE_BATCH, input_ch_end - input_ch_start, kernel_end - kernel_start, BLOCK_SIZE_LENGTH), dtype=tl.int32)
            input_mask = tl.zeros((BLOCK_SIZE_BATCH, input_ch_end - input_ch_start, kernel_end - kernel_start, BLOCK_SIZE_LENGTH), dtype=tl.int1) == 1

            for i in range(input_ch_end - input_ch_start):
                for j in range(kernel_end - kernel_start):
                    # Compute input time index
                    out_time = length_start + tl.arange(0, BLOCK_SIZE_LENGTH)
                    in_time = out_time * stride - padding + (kernel_start + j) * dilation
                    in_time_mask = (in_time >= 0) & (in_time < input_length)
                    input_mask[:, i, j, :] = in_time_mask & (
                        (batch_start + tl.arange(0, BLOCK_SIZE_BATCH))[:, None] < batch_size
                    ) & (
                        (input_ch_start + i) < in_channels
                    )
                    input_offsets[:, i, j, :] = (
                        (batch_start + tl.arange(0, BLOCK_SIZE_BATCH))[:, None] * input_stride_b +
                        (input_ch_start + i) * input_stride_c +
                        in_time * input_stride_l
                    )

            # Flatten input and weight indices for loading
            input_flat_offsets = tl.reshape(input_offsets, (BLOCK_SIZE_BATCH * (input_ch_end - input_ch_start) * (kernel_end - kernel_start), BLOCK_SIZE_LENGTH))
            input_flat_mask = tl.reshape(input_mask, (BLOCK_SIZE_BATCH * (input_ch_end - input_ch_start) * (kernel_end - kernel_start), BLOCK_SIZE_LENGTH))

            # Load input tiles
            x_tile = tl.load(x_ptr + input_flat_offsets, mask=input_flat_mask, other=0.0)
            x_tile = tl.reshape(x_tile, (BLOCK_SIZE_BATCH, input_ch_end - input_ch_start, kernel_end - kernel_start, BLOCK_SIZE_LENGTH))

            # Load weight tiles
            weight_offsets = (
                (ch_start + tl.arange(0, BLOCK_SIZE_OUT_CH))[:, None] * weight_stride_c +
                (kernel_start + tl.arange(0, kernel_end - kernel_start))[None, :] * weight_stride_k
            )
            weight_mask = (
                (ch_start + tl.arange(0, BLOCK_SIZE_OUT_CH))[:, None] < out_channels
            ) & (
                (kernel_start + tl.arange(0, kernel_end - kernel_start))[None, :] < kernel_size
            )
            w_tile = tl.load(w_ptr + weight_offsets[:, None, :, None], mask=weight_mask[:, None, :, None], other=0.0)

            # Perform contraction: (B, OC, K, L) x (OC, IC, K) -> (B, OC, L)
            # Here we contract over IC and K
            w_tile = w_tile * 1.0  # Promote to float32
            x_tile = x_tile.to(tl.float32)
            acc += tl.sum(w_tile * x_tile, axis=[2, 1])

    # Add bias
    acc += bias[:, None]

    # Store output
    tl.store(out_ptr + output_offsets, acc, mask=output_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        if groups != 1:
            raise NotImplementedError("Grouped convolution not supported in this kernel")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias

        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, in_channels, length)
        batch_size, in_channels, input_length = x.shape
        assert in_channels == self.in_channels, "Input channel mismatch"

        # Compute output length
        length_out = (input_length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        output_shape = (batch_size, self.out_channels, length_out)

        # Allocate output
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)

        # Convert to contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous() if self.bias is not None else None

        # Define block sizes
        BLOCK_SIZE_BATCH = triton.next_power_of_2(batch_size)
        BLOCK_SIZE_BATCH = min(max(BLOCK_SIZE_BATCH, 1), 8)  # Limit batch block size

        # Heuristics for block sizes
        BLOCK_SIZE_OUT_CH = triton.next_power_of_2(self.out_channels)
        BLOCK_SIZE_OUT_CH = min(max(BLOCK_SIZE_OUT_CH, 1), 64)
        BLOCK_SIZE_LENGTH = triton.next_power_of_2(length_out)
        BLOCK_SIZE_LENGTH = min(max(BLOCK_SIZE_LENGTH, 1), 64)
        BLOCK_SIZE_IN_CH = triton.next_power_of_2(self.in_channels)
        BLOCK_SIZE_IN_CH = min(max(BLOCK_SIZE_IN_CH, 1), 64)
        BLOCK_SIZE_KERNEL = triton.next_power_of_2(self.kernel_size)
        BLOCK_SIZE_KERNEL = min(max(BLOCK_SIZE_KERNEL, 1), 16)

        # Grid configuration
        grid = (
            triton.cdiv(batch_size, BLOCK_SIZE_BATCH),
            triton.cdiv(self.out_channels, BLOCK_SIZE_OUT_CH),
            triton.cdiv(length_out, BLOCK_SIZE_LENGTH)
        )

        # Launch kernel
        conv1d_kernel[grid](
            x, weight, bias, out,
            batch_size, self.out_channels, length_out, self.in_channels, input_length,
            self.kernel_size, self.stride, self.padding, self.dilation,
            x.stride(0), x.stride(1), x.stride(2),
            self.weight.stride(0), self.weight.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            self.use_bias,
            BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
            BLOCK_SIZE_OUT_CH=BLOCK_SIZE_OUT_CH,
            BLOCK_SIZE_LENGTH=BLOCK_SIZE_LENGTH,
            BLOCK_SIZE_IN_CH=BLOCK_SIZE_IN_CH,
            BLOCK_SIZE_KERNEL=BLOCK_SIZE_KERNEL,
        )

        return out