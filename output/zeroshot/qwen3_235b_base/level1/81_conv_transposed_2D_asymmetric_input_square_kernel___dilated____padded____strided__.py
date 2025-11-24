import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,          # pointer to input tensor (batch_size, in_channels, height_in, width_in)
    weight_ptr,     # pointer to weight tensor (in_channels, out_channels, kernel_h, kernel_w)
    bias_ptr,       # pointer to bias tensor (out_channels,)
    out_ptr,        # pointer to output tensor (batch_size, out_channels, height_out, width_out)
    batch_size,
    out_channels,
    in_channels,
    height_in,
    width_in,
    height_out,
    width_out,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    has_bias: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_HO: tl.constexpr,
    BLOCK_WO: tl.constexpr,
):
    # Compute block indices
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_ho = tl.program_id(2)
    pid_wo = tl.program_id(3)

    # Compute starting indices for each block
    batch_start = pid_b * BLOCK_BATCH
    oc_start = pid_oc * BLOCK_OC
    ho_start = pid_ho * BLOCK_HO
    wo_start = pid_wo * BLOCK_WO

    # Define offsets within blocks
    batch_offsets = batch_start + tl.arange(0, BLOCK_BATCH)
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    ho_offsets = ho_start + tl.arange(0, BLOCK_HO)
    wo_offsets = wo_start + tl.arange(0, BLOCK_WO)

    # Masks to avoid out-of-bounds
    batch_mask = batch_offsets < batch_size
    oc_mask = oc_offsets < out_channels
    ho_mask = ho_offsets < height_out
    wo_mask = wo_offsets < width_out

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_BATCH, BLOCK_OC, BLOCK_HO, BLOCK_WO), dtype=tl.float32)

    # Iterate over input channels and kernel dimensions
    for ic in range(0, in_channels, BLOCK_IC):
        ic_end = min(ic + BLOCK_IC, in_channels)
        ic_block_size = ic_end - ic
        ic_offsets = ic + tl.arange(0, BLOCK_IC)
        ic_mask = ic_offsets < in_channels

        # Load input block: (BLOCK_BATCH, ic_block_size, height_in, width_in)
        x_offsets = (
            batch_offsets[:, None, None, None] * in_channels * height_in * width_in +
            ic_offsets[None, :, None, None] * height_in * width_in +
            tl.arange(0, height_in)[None, None, :, None] * width_in +
            tl.arange(0, width_in)[None, None, None, :]
        )
        x_mask = (
            batch_mask[:, None, None, None] &
            ic_mask[None, :, None, None] &
            (tl.arange(0, height_in)[None, None, :, None] < height_in) &
            (tl.arange(0, width_in)[None, None, None, :] < width_in)
        )
        x_block = tl.load(
            x_ptr + x_offsets,
            mask=x_mask,
            other=0.0
        )  # (BLOCK_BATCH, BLOCK_IC, height_in, width_in)

        # Iterate over kernel
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Compute output positions where this kernel element contributes
                # Output pixel (ho, wo) comes from input pixel ((ho - padding + kh * dilation) // stride)
                h_center = ho_offsets[:, None] + padding_h - kh * dilation_h
                w_center = wo_offsets[None, :] + padding_w - kw * dilation_w

                # Check if divisible by stride
                h_valid = (h_center >= 0) & ((h_center % stride_h) == 0)
                w_valid = (w_center >= 0) & ((w_center % stride_w) == 0)

                h_in = h_center // stride_h
                w_in = w_center // stride_w

                # Valid input locations
                h_in_valid = (h_in >= 0) & (h_in < height_in)
                w_in_valid = (w_in >= 0) & (w_in < width_in)
                valid = h_valid & w_valid & h_in_valid & w_in_valid

                # Load weights: (ic_block_size, BLOCK_OC)
                weight_offsets = (
                    ic_offsets[None, :] * out_channels * kernel_h * kernel_w +
                    oc_offsets[:, None] * kernel_h * kernel_w +
                    kh * kernel_w + kw
                )
                weight_mask = ic_mask[None, :] & oc_mask[:, None]
                weight = tl.load(
                    weight_ptr + weight_offsets,
                    mask=weight_mask,
                    other=0.0
                )  # (BLOCK_OC, ic_block_size)

                # Extract input values at valid positions
                x_vals = tl.load(
                    x_ptr +
                    batch_offsets[:, None, None] * in_channels * height_in * width_in +
                    ic_offsets[None, :, None] * height_in * width_in +
                    h_in[None, None, :] * width_in +
                    w_in[None, None, :],
                    mask=(
                        batch_mask[:, None, None] &
                        ic_mask[None, :, None] &
                        valid[None, None, :]
                    ),
                    other=0.0
                )  # (BLOCK_BATCH, ic_block_size, BLOCK_HO, BLOCK_WO)

                # Accumulate: acc += weight[:, ic] * x_vals[batch, ic, h_in, w_in]
                # Reshape weight to (BLOCK_OC, 1, 1, ic_block_size)
                weight = weight[:, None, None, :]
                # Reshape x_vals to (1, 1, BLOCK_HO, BLOCK_WO, ic_block_size)
                x_vals = x_vals[None, None, :, :, :]  # (1, 1, BLOCK_BATCH, BLOCK_HO, BLOCK_WO, ic_block_size)
                # Broadcast and multiply
                product = weight * x_vals
                # Sum over input channels
                product_sum = tl.sum(product, axis=-1)
                # Transpose to match accumulator layout
                product_sum = tl.trans(product_sum, (2, 0, 1, 3))  # (BLOCK_BATCH, BLOCK_OC, BLOCK_HO, BLOCK_WO)
                acc += product_sum

    # Add bias
    if has_bias:
        bias = tl.load(bias_ptr + oc_offsets, mask=oc_mask, other=0.0)
        acc += bias[:, None, None]

    # Store output
    out_offsets = (
        batch_offsets[:, None, None, None] * out_channels * height_out * width_out +
        oc_offsets[None, :, None, None] * height_out * width_out +
        ho_offsets[None, None, :, None] * width_out +
        wo_offsets[None, None, None, :]
    )
    out_mask = (
        batch_mask[:, None, None, None] &
        oc_mask[None, :, None, None] &
        ho_mask[None, None, :, None] &
        wo_mask[None, None, None, :]
    )
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)


def triton_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
):
    batch_size, in_channels, height_in, width_in = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    # Compute output spatial dimensions
    height_out = (height_in - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + 1
    width_out = (width_in - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + 1

    # Allocate output
    out = torch.zeros(batch_size, out_channels, height_out, width_out, device=x.device, dtype=x.dtype)

    # Constants
    has_bias = bias is not None
    if not has_bias:
        bias = torch.zeros(out_channels, device=x.device, dtype=weight.dtype)

    # Block sizes (tuned for A100)
    BLOCK_BATCH = triton.next_power_of_2(batch_size) if batch_size <= 32 else 16
    BLOCK_OC = 16
    BLOCK_IC = 16
    BLOCK_HO = 16
    BLOCK_WO = 16

    # Grid
    grid = (
        triton.cdiv(batch_size, BLOCK_BATCH),
        triton.cdiv(out_channels, BLOCK_OC),
        triton.cdiv(height_out, BLOCK_HO),
        triton.cdiv(width_out, BLOCK_WO),
    )

    # Launch kernel
    conv_transpose2d_kernel[grid](
        x_ptr=x.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias.data_ptr(),
        out_ptr=out.data_ptr(),
        batch_size=batch_size,
        out_channels=out_channels,
        in_channels=in_channels,
        height_in=height_in,
        width_in=width_in,
        height_out=height_out,
        width_out=width_out,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=stride,
        stride_w=stride,
        padding_h=padding,
        padding_w=padding,
        dilation_h=dilation,
        dilation_w=dilation,
        has_bias=has_bias,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_OC=BLOCK_OC,
        BLOCK_IC=BLOCK_IC,
        BLOCK_HO=BLOCK_HO,
        BLOCK_WO=BLOCK_WO,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Weight initialization (same as ConvTranspose2d)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )