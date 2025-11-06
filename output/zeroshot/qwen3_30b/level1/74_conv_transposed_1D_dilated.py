import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, in_channels, out_channels, length, kernel_size,
    stride, padding, dilation,
    x_stride0, x_stride1, x_stride2,
    w_stride0, w_stride1, w_stride2,
    out_stride0, out_stride1, out_stride2,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_TENSOR_CORES: tl.constexpr
):
    # Thread block index
    pid = tl.program_id(0)
    pid_batch = pid // (out_channels * (length + padding * 2) // BLOCK_SIZE)
    pid_channel = (pid % (out_channels * (length + padding * 2) // BLOCK_SIZE)) // (length + padding * 2) // BLOCK_SIZE
    pid_seq = (pid % (out_channels * (length + padding * 2) // BLOCK_SIZE)) % ((length + padding * 2) // BLOCK_SIZE)

    # Grid dimensions
    total_elements = batch_size * out_channels * (length + padding * 2)
    num_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Block offsets
    block_start = pid * BLOCK_SIZE
    block_offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Output block range
    out_base = block_offsets // (length + padding * 2)
    out_channel = (block_offsets // (length + padding * 2)) % out_channels
    out_pos = block_offsets % (length + padding * 2)

    # Convert to output indices
    out_batch = out_base // out_channels
    out_channel_offset = out_channel
    out_seq_offset = out_pos

    # Output mask
    out_mask = (out_batch < batch_size) & (out_channel_offset < out_channels) & (out_seq_offset < (length + padding * 2))

    # Initialize output
    out_value = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Compute input start position
    input_start_pos = (out_seq_offset - padding) * stride
    input_end_pos = input_start_pos + kernel_size * dilation

    # Input position within the padded length
    input_seq = input_start_pos + tl.arange(0, kernel_size) * dilation

    # Input mask
    input_mask = (input_seq >= 0) & (input_seq < length)

    # Load weights (for this output channel, all input channels, kernel)
    w_ptr_base = w_ptr + out_channel_offset * in_channels * kernel_size
    w_ptrs = w_ptr_base + tl.arange(0, in_channels)[:, None] * w_stride1 + tl.arange(0, kernel_size)[None, :] * w_stride2

    w_vals = tl.load(w_ptrs, mask=tl.arange(0, in_channels)[:, None] < in_channels, other=0.0)
    w_vals = w_vals.to(tl.float16) if USE_BF16 else w_vals.to(tl.float32)

    # Iterate over input channels
    for c in range(in_channels):
        # Input pointer for this channel
        x_ptr_base = x_ptr + out_batch * x_stride0 + c * x_stride1 + input_seq * x_stride2
        x_ptrs = x_ptr_base

        # Load input data
        x_vals = tl.load(x_ptrs, mask=input_mask, other=0.0)
        x_vals = x_vals.to(tl.float16) if USE_BF16 else x_vals.to(tl.float32)

        # Multiply with weights and accumulate
        w_vals_c = w_vals[c, :]  # Shape: [kernel_size]
        prod = x_vals[:, None] * w_vals_c[None, :]  # [kernel_size, kernel_size] -> [kernel_size, 1]
        out_value = out_value + tl.sum(prod, axis=0)

    # Bias
    if HAS_BIAS:
        bias_ptr = w_ptr + out_channels * in_channels * kernel_size + out_channel_offset
        bias_val = tl.load(bias_ptr, mask=tl.arange(0, 1) < 1, other=0.0)
        out_value = out_value + bias_val

    # Store output
    out_ptr_base = out_ptr + out_batch * out_stride0 + out_channel_offset * out_stride1 + out_seq_offset * out_stride2
    tl.store(out_ptr_base, out_value, mask=out_mask)


@triton.jit
def conv_transpose1d_kernel_fused(
    x_ptr, w_ptr, out_ptr,
    batch_size, in_channels, out_channels, length, kernel_size,
    stride, padding, dilation,
    x_stride0, x_stride1, x_stride2,
    w_stride0, w_stride1, w_stride2,
    out_stride0, out_stride1, out_stride2,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_TENSOR_CORES: tl.constexpr,
    ACTIVATION: tl.constexpr  # 0: none, 1: relu, 2: gelu
):
    # Thread block index
    pid = tl.program_id(0)
    pid_batch = pid // (out_channels * (length + padding * 2) // BLOCK_SIZE)
    pid_channel = (pid % (out_channels * (length + padding * 2) // BLOCK_SIZE)) // (length + padding * 2) // BLOCK_SIZE
    pid_seq = (pid % (out_channels * (length + padding * 2) // BLOCK_SIZE)) % ((length + padding * 2) // BLOCK_SIZE)

    # Grid dimensions
    total_elements = batch_size * out_channels * (length + padding * 2)
    num_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Block offsets
    block_start = pid * BLOCK_SIZE
    block_offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Output block range
    out_base = block_offsets // (length + padding * 2)
    out_channel = (block_offsets // (length + padding * 2)) % out_channels
    out_pos = block_offsets % (length + padding * 2)

    # Convert to output indices
    out_batch = out_base // out_channels
    out_channel_offset = out_channel
    out_seq_offset = out_pos

    # Output mask
    out_mask = (out_batch < batch_size) & (out_channel_offset < out_channels) & (out_seq_offset < (length + padding * 2))

    # Initialize output
    out_value = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Compute input start position
    input_start_pos = (out_seq_offset - padding) * stride
    input_end_pos = input_start_pos + kernel_size * dilation

    # Input position within the padded length
    input_seq = input_start_pos + tl.arange(0, kernel_size) * dilation

    # Input mask
    input_mask = (input_seq >= 0) & (input_seq < length)

    # Load weights (for this output channel, all input channels, kernel)
    w_ptr_base = w_ptr + out_channel_offset * in_channels * kernel_size
    w_ptrs = w_ptr_base + tl.arange(0, in_channels)[:, None] * w_stride1 + tl.arange(0, kernel_size)[None, :] * w_stride2

    w_vals = tl.load(w_ptrs, mask=tl.arange(0, in_channels)[:, None] < in_channels, other=0.0)
    w_vals = w_vals.to(tl.bfloat16) if USE_BF16 else w_vals.to(tl.float32)

    # Iterate over input channels
    for c in range(in_channels):
        # Input pointer for this channel
        x_ptr_base = x_ptr + out_batch * x_stride0 + c * x_stride1 + input_seq * x_stride2
        x_ptrs = x_ptr_base

        # Load input data
        x_vals = tl.load(x_ptrs, mask=input_mask, other=0.0)
        x_vals = x_vals.to(tl.bfloat16) if USE_BF16 else x_vals.to(tl.float32)

        # Multiply with weights and accumulate
        w_vals_c = w_vals[c, :]  # Shape: [kernel_size]
        prod = x_vals[:, None] * w_vals_c[None, :]  # [kernel_size, kernel_size]
        out_value = out_value + tl.sum(prod, axis=0)

    # Bias
    if HAS_BIAS:
        bias_ptr = w_ptr + out_channels * in_channels * kernel_size + out_channel_offset
        bias_val = tl.load(bias_ptr, mask=tl.arange(0, 1) < 1, other=0.0)
        out_value = out_value + bias_val

    # Activation
    if ACTIVATION == 1:  # ReLU
        out_value = tl.maximum(out_value, 0.0)
    elif ACTIVATION == 2:  # GELU
        pi = 3.141592653589793
        sqrt_2_over_pi = 0.7978845608028654
        x = out_value
        tanh_val = tl.tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x))
        out_value = x * 0.5 * (1.0 + tanh_val)

    # Store output
    out_ptr_base = out_ptr + out_batch * out_stride0 + out_channel_offset * out_stride1 + out_seq_offset * out_stride2
    tl.store(out_ptr_base, out_value, mask=out_mask)


def triton_conv_transpose1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
    kernel_size: int,
    out_channels: int,
    in_channels: int,
    activation: str = None
):
    assert x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda), "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, _, length = x.shape
    out_length = (length - 1) * stride - 2 * padding + kernel_size * dilation

    out = torch.empty(batch_size, out_channels, out_length, device=x.device, dtype=x.dtype)

    # Strides
    x_stride0, x_stride1, x_stride2 = x.stride()
    w_stride0, w_stride1, w_stride2 = weight.stride()
    out_stride0, out_stride1, out_stride2 = out.stride()

    # Flags
    HAS_BIAS = bias is not None
    USE_BF16 = x.dtype == torch.bfloat16
    USE_TENSOR_CORES = x.dtype in (torch.bfloat16, torch.float16)
    ACTIVATION = 0
    if activation == "relu":
        ACTIVATION = 1
    elif activation == "gelu":
        ACTIVATION = 2

    # Grid
    total_elements = batch_size * out_channels * out_length
    BLOCK_SIZE = 128  # Tunable

    grid = lambda meta: (total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],

    # Launch kernel
    if activation is None:
        conv_transpose1d_kernel[grid](
            x, weight, out,
            batch_size, in_channels, out_channels, length, kernel_size,
            stride, padding, dilation,
            x_stride0, x_stride1, x_stride2,
            w_stride0, w_stride1, w_stride2,
            out_stride0, out_stride1, out_stride2,
            BLOCK_SIZE=BLOCK_SIZE,
            TILE_SIZE=16,
            HAS_BIAS=HAS_BIAS,
            USE_BF16=USE_BF16,
            USE_TENSOR_CORES=USE_TENSOR_CORES
        )
    else:
        conv_transpose1d_kernel_fused[grid](
            x, weight, out,
            batch_size, in_channels, out_channels, length, kernel_size,
            stride, padding, dilation,
            x_stride0, x_stride1, x_stride2,
            w_stride0, w_stride1, w_stride2,
            out_stride0, out_stride1, out_stride2,
            BLOCK_SIZE=BLOCK_SIZE,
            TILE_SIZE=16,
            HAS_BIAS=HAS_BIAS,
            USE_BF16=USE_BF16,
            USE_TENSOR_CORES=USE_TENSOR_CORES,
            ACTIVATION=ACTIVATION
        )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False, activation: str = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.activation = activation

        # Initialize weights
        weight = torch.randn(out_channels, in_channels, kernel_size, device="cuda")
        self.register_buffer("weight", weight)

        if bias:
            bias = torch.randn(out_channels, device="cuda")
            self.register_buffer("bias", bias)
        else:
            self.register_buffer("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose1d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            kernel_size=self.kernel_size,
            out_channels=self.out_channels,
            in_channels=self.in_channels,
            activation=self.activation
        )