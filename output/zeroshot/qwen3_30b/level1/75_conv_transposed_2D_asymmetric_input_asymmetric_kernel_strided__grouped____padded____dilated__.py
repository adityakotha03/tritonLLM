import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    X,  # Input tensor pointer
    W,  # Weight tensor pointer
    B,  # Bias pointer (if bias=True)
    Out,  # Output tensor pointer
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    groups,
    batch_size, in_channels, out_channels,
    height, width,
    out_h, out_w,
    kernel_h, kernel_w,
    X_stride_0, X_stride_1, X_stride_2, X_stride_3,
    W_stride_0, W_stride_1, W_stride_2, W_stride_3,
    B_stride,
    Out_stride_0, Out_stride_1, Out_stride_2, Out_stride_3,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    # Grid: one program per output spatial tile
    pid = tl.program_id(0)  # overall program ID
    pid_batch = pid // (out_h * out_w)
    pid_out_h = (pid % (out_h * out_w)) // out_w
    pid_out_w = (pid % (out_h * out_w)) % out_w

    # Compute output indices
    out_h_start = pid_out_h * BLOCK_SIZE_H
    out_w_start = pid_out_w * BLOCK_SIZE_W
    out_c_start = tl.program_id(1) * BLOCK_SIZE_C
    c = tl.program_id(2)

    # Thread indices
    tx = tl.arange(0, BLOCK_SIZE_W)
    ty = tl.arange(0, BLOCK_SIZE_H)
    tc = tl.arange(0, BLOCK_SIZE_C)

    # Output offsets
    out_offsets_h = out_h_start + ty
    out_offsets_w = out_w_start + tx
    out_offsets_c = out_c_start + tc

    # Mask for output bounds
    out_mask_h = out_offsets_h < out_h
    out_mask_w = out_offsets_w < out_w
    out_mask_c = out_offsets_c < out_channels

    # Compute input spatial offsets from output via transposed conv
    # (output_h, output_w) maps to (input_h, input_w) via:
    # input_h = output_h * stride_h - padding_h + (kernel_h - 1) * dilation_h - offset_h
    # input_w = output_w * stride_w - padding_w + (kernel_w - 1) * dilation_w - offset_w
    # But we iterate over the kernel for each output position
    # So we loop over kernel_h and kernel_w to compute the input positions
    # This is reversed from convolution: we accumulate input from kernel

    # Group offset
    group_size = out_channels // groups
    group_idx = c // group_size
    c_in = (c % group_size) * (in_channels // groups)

    # We use shared memory to load weight tiles and reuse in computation
    # Shared memory: [BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C] per group
    # We'll use tile-based blocking on output spatial dims
    # Load weights for current group and tile
    w_ptrs = W + (group_idx * out_channels * in_channels * kernel_h * kernel_w + c * in_channels * kernel_h * kernel_w + c_in * kernel_h * kernel_w + \
                  tl.arange(0, kernel_h)[:, None] * kernel_w + tl.arange(0, kernel_w)[None, :]) * tl.constexpr(1)

    # Load weights into shared memory
    # Shape: [kernel_h, kernel_w] for current input and output channel group
    w_shared = tl.load(w_ptrs, mask=(tl.arange(0, kernel_h)[:, None] < kernel_h) & (tl.arange(0, kernel_w)[None, :] < kernel_w), other=0.0)
    w_shared = tl.trans(w_shared)  # (kernel_w, kernel_h)

    # Accumulate output
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Loop over kernel dimensions
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Compute input spatial coordinates
            input_h = out_h_start + ty * stride_h - padding_h + kh * dilation_h
            input_w = out_w_start + tx * stride_w - padding_w + kw * dilation_w

            # Valid input bounds
            valid_h = (input_h >= 0) & (input_h < height)
            valid_w = (input_w >= 0) & (input_w < width)
            valid = valid_h & valid_w

            # Compute input offset
            input_offsets_h = input_h
            input_offsets_w = input_w

            # Input pointer (for this group)
            x_ptrs = X + (pid_batch * X_stride_0 + c_in * X_stride_1 + input_offsets_h * X_stride_2 + input_offsets_w * X_stride_3)
            x_vals = tl.load(x_ptrs, mask=valid, other=0.0)

            # Apply weight (broadcast across batch and spatial dims)
            w_val = w_shared[kw, kh]
            acc += x_vals * w_val

    # Apply bias if present
    if B is not None:
        b_val = tl.load(B + c, mask=out_mask_c)
        acc += b_val

    # Output pointer
    out_ptrs = Out + (pid_batch * Out_stride_0 + c * Out_stride_1 + out_offsets_h * Out_stride_2 + out_offsets_w * Out_stride_3)
    out_mask = out_mask_h[:, None] & out_mask_w[None, :] & out_mask_c[:, None, None]
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_conv_transpose(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: tuple, padding: tuple, dilation: tuple, groups: int):
    """
    Custom Triton-based 2D transposed convolution.
    Performs the operation: Out[i, o, h, w] = sum_{k, j} W[o, i, k, j] * X[i, j, h', w'] + b[o]
    where h' = h*stride_h - padding_h + k*dilation_h, etc.
    """
    assert x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda), "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Shape info
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation

    # Compute output dimensions
    out_h = (height - 1) * stride_h - 2 * padding_h + (kernel_h - 1) * dilation_h + 1
    out_w = (width - 1) * stride_w - 2 * padding_w + (kernel_w - 1) * dilation_w + 1

    # Output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)

    # Compute strides
    X_stride_0, X_stride_1, X_stride_2, X_stride_3 = x.stride()
    W_stride_0, W_stride_1, W_stride_2, W_stride_3 = weight.stride()
    if bias is not None:
        B_stride = bias.stride(0)
    else:
        B_stride = 0
    Out_stride_0, Out_stride_1, Out_stride_2, Out_stride_3 = out.stride()

    # Tune BLOCK_SIZE and TILE_SIZE for A100
    # We use 64x64 tiles for output, 32 for channel, and 32x32 for weight tile
    # Use 32x32, 32x64, 64x32, 64x64
    BLOCK_SIZE_H = 64
    BLOCK_SIZE_W = 64
    BLOCK_SIZE_C = 32

    # Grid: (num_output_tiles, num_output_channels_per_group, num_groups)
    num_output_tiles = (out_h + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H * (out_w + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    num_groups = groups
    num_output_channels_per_group = (out_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C

    grid = lambda meta: (num_output_tiles, num_output_channels_per_group, num_groups)

    # Launch kernel
    conv_transpose_kernel[grid](
        x, weight, bias, out,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups,
        batch_size, in_channels, out_channels,
        height, width,
        out_h, out_w,
        kernel_h, kernel_w,
        X_stride_0, X_stride_1, X_stride_2, X_stride_3,
        W_stride_0, W_stride_1, W_stride_2, W_stride_3,
        B_stride,
        Out_stride_0, Out_stride_1, Out_stride_2, Out_stride_3,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        TILE_SIZE=32,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )