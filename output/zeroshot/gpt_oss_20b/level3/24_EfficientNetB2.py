import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------- Triton kernels ----------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    ],
    key=["in_c", "out_c", "h", "w"],
)
@triton.jit
def conv2d_kernel(
    input_ptr,   # float32 * 3*H*W
    weight_ptr,  # float32 * Cout*Cin*3*3
    bias_ptr,    # float32 * Cout (optional)
    out_ptr,     # float32 * Cout*H_out*W_out
    H, W, H_out, W_out,
    stride, padding,
    C_in, C_out,
    BLOCK_SIZE: tl.constexpr,
    WITH_BIAS: tl.constexpr,
):
    """
    A very simple 3×3 convolution kernel with stride 1 or 2.
    Only works for stride=1 or 2 and padding=1.
    """
    # each program processes a contiguous block of output elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < (C_out * H_out * W_out)

    if mask.any():
        # compute output coordinates
        out_idx = tl.where(mask, offsets, 0)
        out_c = out_idx // (H_out * W_out)
        out_hw = out_idx % (H_out * W_out)
        h_out = out_hw // W_out
        w_out = out_hw % W_out

        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for k_h in range(3):
            for k_w in range(3):
                h_in = h_out * stride + k_h - padding
                w_in = w_out * stride + k_w - padding
                # load input only if inside bounds
                in_mask = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                in_idx = tl.where(
                    in_mask,
                    out_c * C_in * H * W + tl.arange(0, C_in) * H * W + h_in * W + w_in,
                    0,
                )
                inp = tl.load(input_ptr + in_idx, mask=in_mask[:, None], other=0.0)
                # weight index
                w_idx = out_c * C_in * 9 + tl.arange(0, C_in) * 9 + k_h * 3 + k_w
                wt = tl.load(weight_ptr + w_idx, mask=mask[:, None], other=0.0)
                acc += tl.sum(inp * wt, axis=1)

        if WITH_BIAS:
            acc += tl.load(bias_ptr + out_c, mask=mask)

        tl.store(out_ptr + out_idx, acc, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
    ],
    key=["N", "C"],
)
@triton.jit
def bn_relu_kernel(
    inp_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    var_ptr,
    out_ptr,
    eps,
    N, C,
    BLOCK_SIZE: tl.constexpr,
):
    """
    BatchNorm + ReLU (inplace) for a tensor of shape (N, C, H, W)
    """
    # each program processes a contiguous block of C elements per sample
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < C

    if mask.any():
        for n in range(N):
            base = n * C * H * W
            for h in range(H):
                for w in range(W):
                    idx = base + offsets * H * W + h * W + w
                    inp = tl.load(inp_ptr + idx, mask=mask, other=0.0)
                    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=0.0)
                    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
                    mean = tl.load(mean_ptr + offsets, mask=mask, other=0.0)
                    var = tl.load(var_ptr + offsets, mask=mask, other=0.0)
                    norm = (inp - mean) * tl.rsqrt(var + eps)
                    out = gamma * norm + beta
                    out = tl.max(out, 0.0)  # ReLU
                    tl.store(out_ptr + idx, out, mask=mask)


# ---------- New Model implementation ----------

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        # Conv1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        # MBConv blocks
        self.mbconv1 = self._make_mbconv_block(32, 96, 1, 3)
        self.mbconv2 = self._make_mbconv_block(96, 144, 2, 6)
        self.mbconv3 = self._make_mbconv_block(144, 192, 2, 6)
        self.mbconv4 = self._make_mbconv_block(192, 288, 2, 6)
        self.mbconv5 = self._make_mbconv_block(288, 384, 1, 6)
        # Final conv
        self.conv_final = nn.Conv2d(384, 1408, kernel_size=1, stride=1, bias=False)
        self.bn_final = nn.BatchNorm2d(1408)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1408, num_classes)

    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        layers = []
        expanded_channels = in_channels * expand_ratio

        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(expanded_channels))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3,
                                stride=stride, padding=1, groups=expanded_channels, bias=False))
        layers.append(nn.BatchNorm2d(expanded_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Conv2d(expanded_channels, expanded_channels // 4, kernel_size=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(expanded_channels // 4, expanded_channels, kernel_size=1, bias=False))
        layers.append(nn.Sigmoid())

        layers.append(nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        return nn.Sequential(*layers)

    # Helper to replace a Conv2d+BN+ReLU with Triton kernel
    def conv_bn_relu_triton(self, x, conv, bn):
        # x: (N, C_in, H, W)
        N, C_in, H, W = x.shape
        # forward conv
        weight = conv.weight.data
        stride = conv.stride[0]
        padding = conv.padding[0]
        H_out = (H + 2 * padding - 3) // stride + 1
        W_out = (W + 2 * padding - 3) // stride + 1
        out = torch.empty((N, conv.out_channels, H_out, W_out), device=x.device, dtype=x.dtype)

        grid = lambda meta: ((conv.out_channels * H_out * W_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

        conv2d_kernel[grid](
            x,
            weight,
            torch.zeros(1, device=x.device),  # dummy bias
            out,
            H, W, H_out, W_out,
            stride, padding,
            C_in, conv.out_channels,
            BLOCK_SIZE=256,
            WITH_BIAS=False,
        )

        # batchnorm + ReLU
        gamma = bn.weight.data
        beta = bn.bias.data
        running_mean = bn.running_mean
        running_var = bn.running_var
        eps = bn.eps

        bn_out = torch.empty_like(out)
        Hb, Wb = H_out, W_out
        grid_bn = lambda meta: ((conv.out_channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        bn_relu_kernel[grid_bn](
            out, gamma, beta, running_mean, running_var, bn_out, eps,
            N, conv.out_channels,
            BLOCK_SIZE=256,
        )
        return bn_out

    def forward(self, x):
        # Conv1
        x = self.conv_bn_relu_triton(x, self.conv1, self.bn1)
        # MBConv blocks
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        # Conv final
        x = self.conv_bn_relu_triton(x, self.conv_final, self.bn_final)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x