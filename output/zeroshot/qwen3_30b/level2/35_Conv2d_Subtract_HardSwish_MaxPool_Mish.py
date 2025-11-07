import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, pad, stride,
    input_stride, weight_stride, output_stride,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_IC: tl.constexpr, BLOCK_OC: tl.constexpr,
    TILE_H: tl.constexpr, TILE_W: tl.constexpr,
    TILE_IC: tl.constexpr, TILE_OC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate block indices
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_ic = tl.program_id(3)
    pid_oc = tl.program_id(4)

    # Compute output spatial coordinates
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    h_mask = h < height
    w_mask = w < width
    mask = h_mask[:, None] & w_mask[None, :]

    # Input tile
    input_block = tl.load(
        input_ptr + (pid_batch * input_stride[0] + pid_ic * input_stride[1] + h[:, None] * input_stride[2] + w[None, :] * input_stride[3]),
        mask=mask[None, None, :, :],
        other=0.0
    )

    # Weight tile
    weight_block = tl.load(
        weight_ptr + (pid_oc * weight_stride[0] + pid_ic * weight_stride[1] + tl.arange(0, BLOCK_IC)[:, None] * weight_stride[2] + tl.arange(0, BLOCK_OC)[None, :] * weight_stride[3]),
        mask=(tl.arange(0, BLOCK_IC)[:, None] < in_channels) & (tl.arange(0, BLOCK_OC)[None, :] < out_channels),
        other=0.0
    )

    # Perform convolution via matrix multiplication
    output = tl.dot(input_block, weight_block)
    output = tl.broadcast(output, (BLOCK_H, BLOCK_W, BLOCK_OC))

    # Store output
    tl.store(
        output_ptr + (pid_batch * output_stride[0] + pid_oc * output_stride[1] + h[:, None] * output_stride[2] + w[None, :] * output_stride[3]),
        output,
        mask=mask[None, None, :, :]
    )


@triton.jit
def hardswish_kernel(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # HardSwish: x * relu6(x + 3) / 6
    x_clipped = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
    out = x * x_clipped / 6.0
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def mish_kernel(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Mish: x * tanh(softplus(x))
    softplus = tl.log1p(tl.exp(x))
    out = x * tl.tanh(softplus)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def maxpool_kernel(
    x_ptr, out_ptr, batch_size, in_channels, height, width, kernel_size,
    input_stride, output_stride, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < height * width * batch_size * in_channels

    # Compute output indices
    out_h = (offsets // (width * in_channels * batch_size)) // kernel_size
    out_w = (offsets // (in_channels * batch_size)) % width // kernel_size
    in_h = out_h * kernel_size
    in_w = out_w * kernel_size

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    out = x

    # Compute max over kernel
    for i in range(1, kernel_size):
        h = in_h + i
        w = in_w
        idx = (h * width + w) * in_channels * batch_size + offsets % (in_channels * batch_size)
        val = tl.load(x_ptr + idx, mask=mask, other=-float('inf'))
        out = tl.maximum(out, val)

        h = in_h
        w = in_w + i
        idx = (h * width + w) * in_channels * batch_size + offsets % (in_channels * batch_size)
        val = tl.load(x_ptr + idx, mask=mask, other=-float('inf'))
        out = tl.maximum(out, val)

    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        # Convert to bfloat16 to leverage Tensor Cores
        x = x.to(torch.bfloat16)

        # Convolution with Triton kernel
        # Use bfloat16 for compute and memory
        batch_size, in_channels, height, width = x.shape
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]
        pad = 1
        stride = 1

        # Compute output shape
        out_h = (height + 2 * pad - kernel_size) // stride + 1
        out_w = (width + 2 * pad - kernel_size) // stride + 1

        # Allocate output tensor
        output = torch.empty(batch_size, out_channels, out_h, out_w, dtype=torch.bfloat16, device=x.device)

        # Block sizes
        BLOCK_H = 8
        BLOCK_W = 8
        BLOCK_IC = 8
        BLOCK_OC = 8
        TILE_H = 16
        TILE_W = 16
        TILE_IC = 16
        TILE_OC = 16

        # Grid setup
        grid = lambda meta: (
            batch_size,
            (out_h + meta['BLOCK_H'] - 1) // meta['BLOCK_H'],
            (out_w + meta['BLOCK_W'] - 1) // meta['BLOCK_W'],
            (in_channels + meta['BLOCK_IC'] - 1) // meta['BLOCK_IC'],
            (out_channels + meta['BLOCK_OC'] - 1) // meta['BLOCK_OC']
        )

        # Launch convolution kernel
        conv2d_kernel[grid](
            x, self.conv.weight,
            output,
            batch_size, in_channels, out_channels, height, width,
            kernel_size, pad, stride,
            x.stride(), self.conv.weight.stride(),
            output.stride(),
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            BLOCK_IC=BLOCK_IC, BLOCK_OC=BLOCK_OC,
            TILE_H=TILE_H, TILE_W=TILE_W,
            TILE_IC=TILE_IC, TILE_OC=TILE_OC,
            BLOCK_SIZE=128
        )

        # Subtract
        output = output - self.subtract_value

        # HardSwish
        output = output.view(-1)
        output = torch.empty_like(output)
        grid = lambda meta: (output.numel() + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE']
        hardswish_kernel[grid](output, output, output.numel(), BLOCK_SIZE=128)

        # MaxPool
        output = output.view(batch_size, self.conv.out_channels, out_h, out_w)
        out_h = (out_h + 2 * (pool_kernel_size - 1)) // pool_kernel_size + 1
        out_w = (out_w + 2 * (pool_kernel_size - 1)) // pool_kernel_size + 1
        pool_output = torch.empty(batch_size, self.conv.out_channels, out_h, out_w, dtype=torch.bfloat16, device=x.device)
        grid = lambda meta: ((output.numel() + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
        maxpool_kernel[grid](
            output, pool_output,
            batch_size, self.conv.out_channels, out_h, out_w, pool_kernel_size,
            output.stride(), pool_output.stride(),
            BLOCK_SIZE=128
        )

        # Mish
        pool_output = pool_output.view(-1)
        output = torch.empty_like(pool_output)
        grid = lambda meta: (pool_output.numel() + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE']
        mish_kernel[grid](pool_output, output, pool_output.numel(), BLOCK_SIZE=128)

        # Reshape and return
        return output.view(batch_size, self.conv.out_channels, out_h, out_w).to(torch.float32)