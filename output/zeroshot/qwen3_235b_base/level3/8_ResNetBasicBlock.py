import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_bn_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
    gamma_ptr, beta_ptr, output_ptr,
    batch, height, width, in_channels, out_channels, kernel_size,
    input_height, input_width, output_height, output_width,
    stride, padding, dilation,
    eps,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr
):
    # 2D tiling over output channels (M), input channels (K), and output pixels (N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute ranges for output channels and pixel indices
    m_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_range < out_channels

    # Each block handles a tile of output spatial locations
    n_tiles = output_height * output_width
    n_range = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_range < n_tiles
    n_idx = n_range // output_width
    n_idy = n_range % output_width

    # Load batch norm params (gamma, beta, mean, var)
    bn_mean = tl.load(running_mean_ptr + m_range, mask=m_mask, other=0.0)
    bn_var = tl.load(running_var_ptr + m_range, mask=m_mask, other=1.0)
    bn_gamma = tl.load(gamma_ptr + m_range, mask=m_mask, other=1.0)
    bn_beta = tl.load(beta_ptr + m_range, mask=m_mask, other=0.0)
    bn_bias = tl.load(bias_ptr + m_range, mask=m_mask, other=0.0) if bias_ptr else tl.zeros_like(bn_mean)

    # Inverse std
    inv_std = 1.0 / tl.sqrt(bn_var + eps)

    # Precompute scale and offset for fused BN
    scale = bn_gamma * inv_std
    offset = bn_beta - bn_mean * inv_std * bn_gamma

    # Initialize accumulator for matmul
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Convolution loop over input channels and kernel
    for k in range(0, in_channels * kernel_size * kernel_size):
        k_c = k // (kernel_size * kernel_size)
        k_kx = (k % (kernel_size * kernel_size)) // kernel_size
        k_ky = (k % kernel_size)

        # Load input tile
        in_ch = k_c
        ih_start = n_idx * stride - padding + k_kx * dilation
        iw_start = n_idy * stride - padding + k_ky * dilation

        ih_mask = (ih_start >= 0) & (ih_start < input_height)
        iw_mask = (iw_start >= 0) & (iw_start < input_width)
        mask_h = ih_mask
        mask_w = iw_mask
        total_mask = mask_h[:, None] & mask_w[None, :] & m_mask[:, None]  # (BLOCK_M, BLOCK_N)

        input_offsets = (  # (batch, in_ch, h, w)
            tl.arange(0, batch)[:, None, None] * in_channels * input_height * input_width +
            in_ch[None, :, None] * input_height * input_width +
            ih_start[:, None] * input_width +
            iw_start[None, :]
        )
        input_vals = tl.load(
            input_ptr + input_offsets,
            mask=total_mask[None, :, :] & (tl.arange(0, batch)[:, None, None] < batch),
            other=0.0
        )  # (batch, BLOCK_M, BLOCK_N) -> reduce over batch and k
        input_vals = tl.sum(input_vals, axis=0)  # sum over batch

        # Load weight
        weight_offset = m_range[:, None] * (in_channels * kernel_size * kernel_size) + k
        weight_vals = tl.load(weight_ptr + weight_offset, mask=m_mask[:, None], other=0.0)  # (BLOCK_M, 1)

        # Outer product and accumulate
        acc += weight_vals * input_vals[None, :]

    # Add bias and apply fused BN + ReLU
    acc = acc + bn_bias[:, None]
    acc = acc * scale[:, None] + offset[:, None]
    acc = tl.where(acc > 0, acc, 0.0)  # ReLU

    # Store output
    output_offsets = (
        m_range[:, None] * (output_height * output_width) +
        n_idx[None, :] * output_width +
        n_idy[None, :]
    )
    tl.store(output_ptr + output_offsets, acc, mask=m_mask[:, None] & n_mask[None, :])


def triton_fused_conv_bn_relu(x, weight, bias, running_mean, running_var, gamma, beta,
                              stride, padding, dilation, eps=1e-5):
    batch, in_channels, height, width = x.shape
    out_channels = weight.shape[0]
    kernel_size = weight.shape[2]
    output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    output_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    out = torch.empty((batch, out_channels, output_height, output_width), device=x.device, dtype=x.dtype)

    # Define block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    # Grid
    grid = (triton.cdiv(out_channels, BLOCK_M), triton.cdiv(output_height * output_width, BLOCK_N))

    fused_conv_bn_relu_kernel[grid](
        x, weight, bias, running_mean, running_var, gamma, beta, out,
        batch, height, width, in_channels, out_channels, kernel_size,
        height, width, output_height, output_width,
        stride, padding, dilation, eps,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return out


@triton.jit
def add_relu_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    out = tl.where(out > 0, out, 0.0)  # ReLU
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_add_relu(x, y):
    assert x.shape == y.shape, "Input shapes must match"
    out = torch.empty_like(x)
    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_relu_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ModelNew, self).__init__()
        self.conv1_weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3))
        self.conv1_bn_weight = nn.Parameter(torch.empty(out_channels))
        self.conv1_bn_bias = nn.Parameter(torch.empty(out_channels))
        self.conv1_bn_running_mean = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        self.conv1_bn_running_var = nn.Parameter(torch.ones(out_channels), requires_grad=False)
        self.conv1_bn_num_batches_tracked = nn.Parameter(torch.tensor(0), requires_grad=False)

        self.conv2_weight = nn.Parameter(torch.empty(out_channels, out_channels, 3, 3))
        self.conv2_bn_weight = nn.Parameter(torch.empty(out_channels))
        self.conv2_bn_bias = nn.Parameter(torch.empty(out_channels))
        self.conv2_bn_running_mean = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        self.conv2_bn_running_var = nn.Parameter(torch.ones(out_channels), requires_grad=False)
        self.conv2_bn_num_batches_tracked = nn.Parameter(torch.tensor(0), requires_grad=False)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

        self.stride = stride
        self.eps = 1e-5
        self.momentum = 0.1

        # Initialize weights
        nn.init.kaiming_normal_(self.conv1_weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2_weight, mode='fan_out', nonlinearity='relu')
        nn.init.ones_(self.conv1_bn_weight)
        nn.init.zeros_(self.conv1_bn_bias)
        nn.init.ones_(self.conv2_bn_weight)
        nn.init.zeros_(self.conv2_bn_bias)

    def forward(self, x):
        identity = x

        # First fused conv-bn-relu
        out = triton_fused_conv_bn_relu(
            x, self.conv1_weight, self.conv1_bn_bias, self.conv1_bn_running_mean, self.conv1_bn_running_var,
            self.conv1_bn_weight, self.conv1_bn_bias, stride=self.stride, padding=1, dilation=1, eps=self.eps
        )

        # Second fused conv-bn (no ReLU yet)
        out = torch.nn.functional.conv2d(out, self.conv2_weight, padding=1)
        out = torch.nn.functional.batch_norm(
            out, self.conv2_bn_running_mean, self.conv2_bn_running_var,
            weight=self.conv2_bn_weight, bias=self.conv2_bn_bias,
            training=self.training, momentum=self.momentum, eps=self.eps
        )

        # Downsample shortcut if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Fused add + ReLU
        out = triton_add_relu(out, identity)

        return out