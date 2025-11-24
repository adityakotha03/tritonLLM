import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_transpose_kernel(
    x_ptr,          # pointer to input tensor (batch_size, in_channels, length)
    weight_ptr,     # pointer to weight tensor (in_channels, out_channels_per_group, kernel_size)
    bias_ptr,       # pointer to bias (out_channels,) or None
    out_ptr,        # pointer to output tensor (batch_size, out_channels, length_out)
    batch_size,
    in_channels,
    out_channels,
    length,
    length_out,
    kernel_size,
    stride,
    padding,
    output_padding,
    groups,
    out_channels_per_group,
    headdim,
    # block sizes (compile-time constants)
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OUT_CH: tl.constexpr,
    BLOCK_IN_CH: tl.constexpr,
    BLOCK_LENGTH: tl.constexpr,
    BLOCK_KERNEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    # Compute program ids
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_l = tl.program_id(2)

    # Compute offsets for blocks
    batch_start = pid_b * BLOCK_BATCH
    oc_start = pid_oc * BLOCK_OUT_CH
    l_out_start = pid_l * BLOCK_LENGTH

    # Batch, output channel, and output length offsets
    b_offsets = batch_start + tl.arange(0, BLOCK_BATCH)
    oc_offsets = oc_start + tl.arange(0, BLOCK_OUT_CH)
    l_out_offsets = l_out_start + tl.arange(0, BLOCK_LENGTH)

    # Input length indices: for each output position, determine which input positions contribute
    # Reverse: output location l_out corresponds to input locations (l_out - padding + [0..kernel_size-1] - output_padding) // stride
    # Only valid if divisible by stride
    l_in_unscaled = l_out_offsets[:, None] - padding + tl.arange(0, BLOCK_KERNEL)[None, :] - output_padding
    l_in_valid = (l_in_unscaled >= 0) & (l_in_unscaled < length * stride)
    l_in_scaled = l_in_unscaled // stride
    l_in_mask = (l_in_unscaled % stride == 0) & l_in_valid

    # Clamp to valid range for indexing
    l_in_clamped = tl.where(l_in_mask, l_in_scaled, 0)

    # Broadcast across heads (channels within group)
    head_start = 0
    group_id = oc_start // out_channels_per_group
    ic_start = group_id * BLOCK_IN_CH
    ic_offsets = ic_start + tl.arange(0, BLOCK_IN_CH)

    # Load input: (BLOCK_BATCH, BLOCK_IN_CH, length) -> we need (BLOCK_BATCH, BLOCK_IN_CH, BLOCK_LENGTH, BLOCK_KERNEL)
    # We will iterate over kernel positions
    acc = tl.zeros((BLOCK_BATCH, BLOCK_OUT_CH, BLOCK_LENGTH), dtype=tl.float32)

    for k in range(0, kernel_size, BLOCK_KERNEL):
        cur_k_offsets = k + tl.arange(0, BLOCK_KERNEL)
        k_mask = cur_k_offsets < kernel_size

        # Input indices for this kernel block
        l_in_cur = tl.load(l_in_clamped, mask=l_in_mask, other=0)
        l_in_cur_mask = tl.load(l_in_mask, mask=None, other=0)

        # Input: (batch, in_ch, l_in_cur) -> (BLOCK_BATCH, BLOCK_IN_CH, BLOCK_LENGTH, BLOCK_KERNEL)
        # Weight: (in_ch, out_ch_per_group, k) -> (BLOCK_IN_CH, BLOCK_OUT_CH, BLOCK_KERNEL)
        # We need to index weight: [ic, oc_in_group, k]
        oc_in_group_offsets = oc_offsets % out_channels_per_group
        w = tl.load(
            weight_ptr +
            ic_offsets[:, None, None] * (out_channels_per_group * kernel_size) +
            oc_in_group_offsets[None, :, None] * kernel_size +
            cur_k_offsets[None, None, :],
            mask=(ic_offsets[:, None, None] < in_channels) &
                 (oc_in_group_offsets[None, :, None] < out_channels_per_group) &
                 k_mask[None, None, :],
            other=0.0
        )  # (BLOCK_IN_CH, BLOCK_OUT_CH, BLOCK_KERNEL)

        # Input x: (BLOCK_BATCH, BLOCK_IN_CH, BLOCK_LENGTH, BLOCK_KERNEL)
        x = tl.load(
            x_ptr +
            b_offsets[:, None, None, None] * (in_channels * length) +
            ic_offsets[None, :, None, None] * length +
            l_in_cur[None, None, :, :],
            mask=(b_offsets[:, None, None, None] < batch_size) &
                 (ic_offsets[None, :, None, None] < in_channels) &
                 l_in_cur_mask[None, None, :, :],
            other=0.0
        )  # (BLOCK_BATCH, BLOCK_IN_CH, BLOCK_LENGTH, BLOCK_KERNEL)

        # Contract over in_channels and kernel
        # x: (BLOCK_BATCH, BLOCK_IN_CH, BLOCK_LENGTH, BLOCK_KERNEL)
        # w: (BLOCK_IN_CH, BLOCK_OUT_CH, BLOCK_KERNEL)
        # We do: sum_{ic, k} x[b,ic,l,k] * w[ic,oc,k]
        # Reshape x to (BLOCK_BATCH, BLOCK_LENGTH, BLOCK_IN_CH * BLOCK_KERNEL)
        x_flat = tl.reshape(x, (BLOCK_BATCH, BLOCK_LENGTH, BLOCK_IN_CH * BLOCK_KERNEL))
        w_flat = tl.reshape(w, (BLOCK_IN_CH * BLOCK_KERNEL, BLOCK_OUT_CH))
        acc += tl.dot(x_flat, w_flat)  # (BLOCK_BATCH, BLOCK_LENGTH, BLOCK_OUT_CH)

    # Handle bias
    if bias_ptr is not None:
        b = tl.load(
            bias_ptr + oc_offsets,
            mask=oc_offsets < out_channels,
            other=0.0
        )  # (BLOCK_OUT_CH,)
        acc += b[None, None, :]

    # Write output
    mask_o = (b_offsets[:, None, None] < batch_size) & \
             (oc_offsets[None, :, None] < out_channels) & \
             (l_out_offsets[None, None, :] < length_out)
    tl.store(
        out_ptr +
        b_offsets[:, None, None] * (out_channels * length_out) +
        oc_offsets[None, :, None] * length_out +
        l_out_offsets[None, None, :],
        acc,
        mask=mask_o
    )


def triton_conv1d_transpose(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: int,
    output_padding: int,
    groups: int
):
    batch_size, in_channels, length = x.shape
    in_channels_weight, out_channels_per_group, kernel_size = weight.shape
    out_channels = groups * out_channels_per_group
    # Compute output length
    length_out = (length - 1) * stride - 2 * padding + kernel_size + output_padding

    # Output tensor
    out = torch.zeros(batch_size, out_channels, length_out, dtype=x.dtype, device=x.device)

    # Flatten weight: (in_channels, out_channels_per_group, kernel_size)
    weight = weight.view(in_channels, out_channels_per_group, kernel_size)

    # Block sizes (tuned for A100)
    BLOCK_BATCH = triton.next_power_of_2(batch_size)
    while BLOCK_BATCH > 16:
        BLOCK_BATCH //= 2
    BLOCK_BATCH = max(BLOCK_BATCH, 1)

    BLOCK_OUT_CH = 32
    BLOCK_IN_CH = 32
    BLOCK_LENGTH = 64
    BLOCK_KERNEL = triton.cdiv(kernel_size, 32) * 32  # pad kernel to multiple of 32

    grid = (
        triton.cdiv(batch_size, BLOCK_BATCH),
        triton.cdiv(out_channels, BLOCK_OUT_CH),
        triton.cdiv(length_out, BLOCK_LENGTH)
    )

    conv1d_transpose_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, length, length_out,
        kernel_size, stride, padding, output_padding, groups, out_channels_per_group,
        kernel_size,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_OUT_CH=BLOCK_OUT_CH,
        BLOCK_IN_CH=BLOCK_IN_CH,
        BLOCK_LENGTH=BLOCK_LENGTH,
        BLOCK_KERNEL=BLOCK_KERNEL,
        BLOCK_HEAD=32,
    )
    return out


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
        self.use_bias = bias

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Weight initialization (same as ConvTranspose1d)
        nn.init.kaiming_uniform_(self.weight, nonlinearity='leaky_relu', a=0.2)
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv1d_transpose(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding, self.groups
        )