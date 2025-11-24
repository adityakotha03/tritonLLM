import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    input_ptr, output_ptr,
    n_channels, spatial_size,
    stride_batch_chan, stride_channel,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    spatial_id = tl.program_id(1)

    offset = batch_id * stride_batch_chan + spatial_id * n_channels
    mask = offset + tl.arange(0, BLOCK_SIZE) < offset + n_channels

    buffer = tl.load(input_ptr + offset + tl.arange(0, BLOCK_SIZE), mask=mask, other=-float('inf'))
    cmax = tl.max(buffer, 0)
    rcmax = tl.exp(buffer - cmax)
    rsum = tl.sum(rcmax, 0)
    softmax_output = rcmax / rsum

    tl.store(output_ptr + offset + tl.arange(0, BLOCK_SIZE), softmax_output, mask=mask)


@triton.jit
def _sigmoid_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    idxs = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = idxs < n_elements
    x = tl.load(input_ptr + idxs, mask=mask, other=0.0)
    sig = tl.sigmoid(x)
    tl.store(output_ptr + idxs, sig, mask=mask)


@triton.jit
def _conv_transpose_3d_nhwc_kernel(
    input_ptr, weight_ptr, output_ptr, bias_ptr,
    batch, out_d, out_h, out_w, out_ch, in_d, in_h, in_w, in_ch,
    k_d, k_h, k_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    acc_dtype: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // (tl.cdiv(out_d * out_h * out_w, BLOCK_M))
    residual = pid % (tl.cdiv(out_d * out_h * out_w, BLOCK_M))
    out_z = residual // (tl.cdiv(out_h * out_w, BLOCK_M))
    residual = residual % (tl.cdiv(out_h * out_w, BLOCK_M))
    out_y = residual // (tl.cdiv(out_w, BLOCK_M))
    out_x = residual % (tl.cdiv(out_w, BLOCK_M))

    offs_m = out_z * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < out_d
    mask_n = offs_n < out_ch

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

    for k0 in range(0, k_d * k_h * k_w):
        kz = k0 // (k_h * k_w)
        k0_hw = k0 % (k_h * k_w)
        ky = k0_hw // k_w
        kx = k0_hw % k_w

        in_z = (offs_m - pad_d + kz * stride_d) // (1 + out_pad_d)
        in_y = (out_y * BLOCK_M + tl.arange(0, BLOCK_M) - pad_h + ky * stride_h) // (1 + out_pad_h)
        in_x = (out_x * BLOCK_M + tl.arange(0, BLOCK_M) - pad_w + kx * stride_w) // (1 + out_pad_w)

        mask_in = (in_z >= 0) & (in_z < in_d) & (in_y >= 0) & (in_y < in_h) & (in_x >= 0) & (in_x < in_w)
        in_offset = batch_id * in_d * in_h * in_w * in_ch + \
                    in_z[:, None] * in_h * in_w * in_ch + \
                    in_y[:, None] * in_w * in_ch + \
                    in_x[:, None] * in_ch + \
                    offs_k[None, :] * in_ch
        weight_offset = kz * k_h * k_w * out_ch * in_ch + ky * k_w * out_ch * in_ch + kx * out_ch * in_ch + \
                        offs_k[None, :] * out_ch + offs_n[:, None]

        a = tl.load(input_ptr + in_offset, mask=mask_in[:, None] & (offs_k[None, :] < in_ch), other=0.0)
        b = tl.load(weight_ptr + weight_offset, mask=(offs_k[None, :] < in_ch) & mask_n[:, None], other=0.0)
        accumulator += tl.dot(a, b, out_dtype=acc_dtype)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        accumulator += bias[None, :]

    out_offset = batch_id * out_d * out_h * out_w * out_ch + \
                 offs_m[:, None] * out_h * out_w * out_ch + \
                 (out_y * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * out_w * out_ch + \
                 (out_x * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * out_ch + \
                 offs_n[None, :]
    mask_out = (offs_m[:, None] < out_d) & (offs_n[None, :] < out_ch)
    tl.store(output_ptr + out_offset, accumulator, mask=mask_out)


def triton_conv_transpose3d_nhwc(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride,
    padding,
    output_padding,
    dilation=1
):
    B, D, H, W, C_in = x.shape
    kD, kH, kW, C_out, C_in_weight = weight.shape
    assert C_in == C_in_weight

    sD, sH, sW = stride
    pD, pH, pW = padding
    oD, oH, oW = output_padding

    out_d = (D - 1) * sD - 2 * pD + kD + oD
    out_h = (H - 1) * sH - 2 * pH + kH + oH
    out_w = (W - 1) * sW - 2 * pW + kW + oW

    out = torch.empty((B, out_d, out_h, out_w, C_out), dtype=x.dtype, device=x.device)

    def grid(META):
        return (B * triton.cdiv(out_d, META['BLOCK_M']) * triton.cdiv(out_h, META['BLOCK_M']) * triton.cdiv(out_w, META['BLOCK_M']),)

    _conv_transpose_3d_nhwc_kernel[grid](
        x, weight, out, bias if bias is not None else 0,
        B, out_d, out_h, out_w, C_out, D, H, W, C_in,
        kD, kH, kW,
        sD, sH, sW,
        pD, pH, pW,
        oD, oH, oW,
        acc_dtype=tl.float32,
        HAS_BIAS=bias is not None,
        BLOCK_M=16,
        BLOCK_N=32,
        BLOCK_K=32,
    )
    return out


def triton_softmax(x: torch.Tensor, dim: int):
    x = x.contiguous()
    out = torch.empty_like(x)
    dims = x.shape
    n_channels = dims[dim]
    spatial_size = x.numel() // dims[0] // n_channels
    grid = lambda meta: (dims[0], spatial_size)
    BLOCK_SIZE = triton.next_power_of_2(n_channels)
    _softmax_kernel[grid](
        x, out,
        n_channels, spatial_size,
        dims[1] * dims[2] * dims[3] * dims[4],  # NHW stride
        dims[4],  # C stride
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def triton_sigmoid(x: torch.Tensor):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    BLOCK_SIZE = 1024
    _sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for ConvTranspose3d, Softmax, and Sigmoid.
    The input is converted to NHWC for better memory access in transposed conv.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # Create weight and bias for transposed conv
        k = 1.0 / (in_channels * kernel_size ** 3)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        nn.init.uniform_(self.weight, -k**0.5, k**0.5)
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Convert from NCDHW to NDHWC
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        # Transpose weight to (kD, kH, kW, out_ch, in_ch)
        weight_t = self.weight.permute(2, 3, 4, 1, 0).contiguous()

        # Apply Triton-based transposed convolution
        x = triton_conv_transpose3d_nhwc(
            x, weight_t, self.bias,
            stride=self.stride, padding=self.padding, output_padding=self.output_padding
        )

        # Convert back to NCDHW for Softmax (dim=1 corresponds to channel)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        # Apply Triton-based Softmax
        x = triton_softmax(x, dim=1)

        # Apply Triton-based Sigmoid
        x = triton_sigmoid(x)

        return x