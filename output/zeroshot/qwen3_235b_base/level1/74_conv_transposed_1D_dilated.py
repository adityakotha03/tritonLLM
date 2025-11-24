import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_transpose_kernel(
    x_ptr,           # pointer to input tensor (batch_size, in_channels, length)
    weight_ptr,      # pointer to weight tensor (in_channels, out_channels, kernel_size)
    bias_ptr,        # pointer to bias tensor (out_channels,)
    output_ptr,      # pointer to output tensor (batch_size, out_channels, length_out)
    batch_size,
    in_channels,
    out_channels,
    length,
    length_out,
    kernel_size,
    stride,
    padding,
    dilation,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # 2D block abstraction: each block handles a tile of output channels (M) and input spatial positions (N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for output channels and input positions
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N

    # Input and output sequence lengths
    length_dilated = (kernel_size - 1) * dilation + 1
    total_output_length = (length - 1) * stride + length_dilated - 2 * padding

    # Pointers into the weight matrix: (in_channels, out_channels, kernel_size) -> view as (in_channels * kernel_size, out_channels)
    weight_stride_0 = out_channels * kernel_size
    weight_stride_1 = out_channels
    weight_stride_2 = 1

    # Allocate local blocks in shared memory
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over in_channels and kernel_size (reduction over K dimension)
    for c in range(0, in_channels):
        # Load input slice: (batch, c, i) for all relevant i in BLOCK_SIZE_N
        for b in range(0, batch_size):
            input_base = x_ptr + b * in_channels * length + c * length
            for k in range(0, kernel_size):
                # Compute dilated position in output
                for i in range(0, length):
                    out_pos = i * stride + k * dilation - padding
                    if 0 <= out_pos < total_output_length:
                        # Accumulate contribution: input[b, c, i] * weight[c, :, k]
                        input_val = tl.load(input_base + i)
                        weight_base = weight_ptr + c * weight_stride_0 + k * weight_stride_2
                        weight_ptrs = weight_base + tl.arange(0, BLOCK_SIZE_M) + m_offset
                        mask = (tl.arange(0, BLOCK_SIZE_M) + m_offset) < out_channels
                        weight_vals = tl.load(weight_ptrs, mask=mask, other=0.0)
                        acc[:, out_pos - n_offset] += input_val * weight_vals

    # Handle bias
    if bias_ptr:
        bias_ptrs = bias_ptr + m_offset + tl.arange(0, BLOCK_SIZE_M)
        mask = (tl.arange(0, BLOCK_SIZE_M) + m_offset) < out_channels
        bias_vals = tl.load(bias_ptrs, mask=mask, other=0.0)
        acc += bias_vals[:, None]

    # Store output
    for b in range(0, batch_size):
        output_base = output_ptr + b * out_channels * total_output_length
        for m in range(0, BLOCK_SIZE_M):
            for n in range(0, BLOCK_SIZE_N):
                out_c = m_offset + m
                out_t = n_offset + n
                if out_c < out_channels and out_t < total_output_length:
                    tl.store(output_base + out_c * total_output_length + out_t, acc[m, n])


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Initialize the transposed convolution weights
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Weight initialization (same as ConvTranspose1d)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, in_channels, length)
        batch_size, in_channels, length = x.shape
        assert in_channels == self.in_channels

        # Compute output length
        length_dilated = (self.kernel_size - 1) * self.dilation + 1
        length_out = (length - 1) * self.stride + length_dilated - 2 * self.padding

        # Create output tensor
        output = torch.empty(batch_size, self.out_channels, length_out, device=x.device, dtype=x.dtype)

        # Ensure contiguous input and weight
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous() if self.bias is not None else None

        # Launch kernel
        def grid(meta):
            return (
                triton.cdiv(self.out_channels, meta['BLOCK_SIZE_M']),
                triton.cdiv(length_out, meta['BLOCK_SIZE_N']),
            )

        # Use autotuning for optimal block sizes
        @triton.autotune(
            configs=[
                triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
            ],
            key=['in_channels', 'out_channels', 'length', 'length_out', 'kernel_size'],
        )
        @triton.jit
        def _kernel(
            x_ptr, weight_ptr, bias_ptr, output_ptr,
            batch_size, in_channels, out_channels, length, length_out, kernel_size,
            stride, padding, dilation,
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
        ):
            conv1d_transpose_kernel(
                x_ptr, weight_ptr, bias_ptr, output_ptr,
                batch_size, in_channels, out_channels, length, length_out, kernel_size,
                stride, padding, dilation,
                BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
            )

        _kernel[grid](
            x, weight, bias, output,
            batch_size, self.in_channels, self.out_channels, length, length_out, self.kernel_size,
            self.stride, self.padding, self.dilation
        )

        return output