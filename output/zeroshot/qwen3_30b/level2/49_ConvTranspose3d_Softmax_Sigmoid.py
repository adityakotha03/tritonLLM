import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    D,
    H,
    W,
    kernel_size,
    stride,
    padding,
    output_padding,
    D_out,
    H_out,
    W_out,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
):
    # Shared memory for input and kernel tiles
    pid = tl.program_id(0)
    num_pid_d = tl.cdiv(D_out, BLOCK_SIZE_D)
    num_pid_h = tl.cdiv(H_out, BLOCK_SIZE_H)
    num_pid_w = tl.cdiv(W_out, BLOCK_SIZE_W)
    num_pid_oc = tl.cdiv(out_channels, BLOCK_SIZE_OC)
    num_pid_ic = tl.cdiv(in_channels, BLOCK_SIZE_IC)

    pid_d = pid // (num_pid_h * num_pid_w * num_pid_oc * num_pid_ic)
    pid_h = (pid // (num_pid_w * num_pid_oc * num_pid_ic)) % num_pid_h
    pid_w = (pid // (num_pid_oc * num_pid_ic)) % num_pid_w
    pid_oc = (pid // num_pid_ic) % num_pid_oc
    pid_ic = pid % num_pid_ic

    # Offsets for current tile
    offs_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    offs_oc = pid_oc * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC)
    offs_ic = pid_ic * BLOCK_SIZE_IC + tl.arange(0, BLOCK_SIZE_IC)

    # Calculate output indices
    out_d = offs_d
    out_h = offs_h
    out_w = offs_w
    out_batch = tl.arange(0, 1)

    # Calculate input indices for transpose
    in_d = (out_d - output_padding) * stride - padding
    in_h = (out_h - output_padding) * stride - padding
    in_w = (out_w - output_padding) * stride - padding

    # Bounds for input
    mask_d = (in_d >= 0) & (in_d < D)
    mask_h = (in_h >= 0) & (in_h < H)
    mask_w = (in_w >= 0) & (in_w < W)

    # Tile over input channel
    # Load input tile
    in_ptr = x_ptr + (out_batch[:, None, None, None] * in_channels * D * H * W +
                      offs_ic[None, :, None, None] * D * H * W +
                      in_d[None, None, :, None] * H * W +
                      in_h[None, None, None, :] * W +
                      in_w[None, None, None, :])
    in_tile = tl.load(in_ptr, mask=(mask_d[None, None, :, None] &
                                     mask_h[None, None, None, :] &
                                     mask_w[None, None, None, :]), other=0.0)

    # Load kernel tile
    kernel_ptr = w_ptr + (offs_oc[:, None, None, None] * in_channels * kernel_size * kernel_size * kernel_size +
                          offs_ic[None, :, None, None] * kernel_size * kernel_size * kernel_size +
                          tl.arange(0, kernel_size)[:, None, None, None] * kernel_size * kernel_size +
                          tl.arange(0, kernel_size)[None, :, None, None] * kernel_size +
                          tl.arange(0, kernel_size)[None, None, :, None])
    kernel_tile = tl.load(kernel_ptr, mask=(tl.arange(0, kernel_size)[:, None, None, None] >= 0), other=0.0)

    # Compute output tile
    # (in_d, in_h, in_w) -> (out_d, out_h, out_w)
    # Output is a reduction over input indices
    out_tile = tl.zeros((BLOCK_SIZE_OC, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                # Load input slice
                inp = in_tile[:, i, j, k]  # (BLOCK_SIZE_IC, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W)
                ker = kernel_tile[:, :, i, j, k]  # (BLOCK_SIZE_OC, BLOCK_SIZE_IC)

                # Compute dot product
                # (BLOCK_SIZE_IC, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W) x (BLOCK_SIZE_OC, BLOCK_SIZE_IC)
                # -> (BLOCK_SIZE_OC, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W)
                dot = tl.dot(ker, inp, allow_tf32=True)
                out_tile += dot

    # Store output
    out_ptr = out_ptr + (out_batch[:, None, None, None] * out_channels * D_out * H_out * W_out +
                         offs_oc[:, None, None, None] * D_out * H_out * W_out +
                         out_d[None, :, None, None] * H_out * W_out +
                         out_h[None, None, :, None] * W_out +
                         out_w[None, None, None, :])
    tl.store(out_ptr, out_tile, mask=(out_d[None, :, None, None] < D_out) &
                                 (out_h[None, None, :, None] < H_out) &
                                 (out_w[None, None, None, :] < W_out))


@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    D,
    H,
    W,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_d = tl.cdiv(D, BLOCK_SIZE_D)
    num_pid_h = tl.cdiv(H, BLOCK_SIZE_H)
    num_pid_w = tl.cdiv(W, BLOCK_SIZE_W)
    num_pid_c = tl.cdiv(channels, BLOCK_SIZE_C)

    pid_d = pid // (num_pid_h * num_pid_w * num_pid_c)
    pid_h = (pid // (num_pid_w * num_pid_c)) % num_pid_h
    pid_w = (pid // num_pid_c) % num_pid_w
    pid_c = pid % num_pid_c

    offs_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    # Calculate input indices
    in_ptr = x_ptr + (offs_c[None, None, None, :] * D * H * W +
                      offs_d[:, None, None, None] * H * W +
                      offs_h[None, :, None, None] * W +
                      offs_w[None, None, :, None])
    x_tile = tl.load(in_ptr, mask=(offs_d[:, None, None, None] < D) &
                                   (offs_h[None, :, None, None] < H) &
                                   (offs_w[None, None, :, None] < W) &
                                   (offs_c[None, None, None, :] < channels), other=-float('inf'))

    # Online softmax: subtract max and exponentiate
    x_max = tl.max(x_tile, axis=0)
    x_exp = tl.exp(x_tile - x_max[None, None, None, :])
    x_sum = tl.sum(x_exp, axis=0)
    out_tile = x_exp / x_sum[None, None, None, :]

    # Store output
    out_ptr = out_ptr + (offs_c[None, None, None, :] * D * H * W +
                         offs_d[:, None, None, None] * H * W +
                         offs_h[None, :, None, None] * W +
                         offs_w[None, None, :, None])
    tl.store(out_ptr, out_tile, mask=(offs_d[:, None, None, None] < D) &
                                   (offs_h[None, :, None, None] < H) &
                                   (offs_w[None, None, :, None] < W) &
                                   (offs_c[None, None, None, :] < channels))


@triton.jit
def sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv_transpose(x, w, bias, stride, padding, output_padding):
    assert x.is_cuda and w.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, in_channels, D, H, W = x.shape
    out_channels, _, kernel_size, _, _ = w.shape
    D_out = (D - 1) * stride - 2 * padding + kernel_size + output_padding
    H_out = (H - 1) * stride - 2 * padding + kernel_size + output_padding
    W_out = (W - 1) * stride - 2 * padding + kernel_size + output_padding

    out = torch.empty(batch_size, out_channels, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    BLOCK_SIZE_D = 16
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_OC = 32
    BLOCK_SIZE_IC = 32

    grid = lambda meta: (meta['num_pid_d'] * meta['num_pid_h'] * meta['num_pid_w'] * meta['num_pid_oc'] * meta['num_pid_ic'],)

    num_pid_d = tl.cdiv(D_out, BLOCK_SIZE_D)
    num_pid_h = tl.cdiv(H_out, BLOCK_SIZE_H)
    num_pid_w = tl.cdiv(W_out, BLOCK_SIZE_W)
    num_pid_oc = tl.cdiv(out_channels, BLOCK_SIZE_OC)
    num_pid_ic = tl.cdiv(in_channels, BLOCK_SIZE_IC)

    conv_transpose_kernel[grid](
        x, w, out, batch_size, in_channels, out_channels, D, H, W,
        kernel_size, stride, padding, output_padding, D_out, H_out, W_out,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_OC=BLOCK_SIZE_OC,
        BLOCK_SIZE_IC=BLOCK_SIZE_IC
    )

    if bias is not None:
        bias_ptr = bias
        out_ptr = out
        for i in range(batch_size):
            for j in range(D_out):
                for k in range(H_out):
                    for l in range(W_out):
                        out_ptr[i, :, j, k, l] += bias_ptr

    return out


def triton_softmax(x):
    assert x.is_cuda
    x = x.contiguous()
    batch_size, channels, D, H, W = x.shape
    BLOCK_SIZE_D = 16
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 32
    grid = lambda meta: (meta['num_pid_d'] * meta['num_pid_h'] * meta['num_pid_w'] * meta['num_pid_c'],)

    num_pid_d = tl.cdiv(D, BLOCK_SIZE_D)
    num_pid_h = tl.cdiv(H, BLOCK_SIZE_H)
    num_pid_w = tl.cdiv(W, BLOCK_SIZE_W)
    num_pid_c = tl.cdiv(channels, BLOCK_SIZE_C)

    out = torch.empty_like(x)
    softmax_kernel[grid](
        x, out, batch_size, channels, D, H, W,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    return out


def triton_sigmoid(x):
    assert x.is_cuda
    x = x.contiguous()
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    out = torch.empty_like(x)
    sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.register_buffer('weight', self.conv_transpose.weight.detach())
        if bias:
            self.register_buffer('bias', self.conv_transpose.bias.detach())
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        # Use Triton kernels for all operations
        x = triton_conv_transpose(x, self.weight, self.bias, stride=self.conv_transpose.stride[0], padding=self.conv_transpose.padding[0], output_padding=self.conv_transpose.output_padding[0])
        x = triton_softmax(x)
        x = triton_sigmoid(x)
        return x