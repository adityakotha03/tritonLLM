import torch
import torch.nn as nn
import triton
import triton.language as tl

# ----------------------------------
# Triton kernels
# ----------------------------------

# 1x1 convolution (channel reduction / expansion) using GEMM
@triton.autotune(
    configs=[
        triton.Config({}, 128),
        triton.Config({}, 256),
        triton.Config({}, 512),
    ],
    key=["M", "N", "K", "BLOCK_SIZE"],
)
@triton.jit
def conv1x1_kernel(
    weight_ptr,
    input_ptr,
    output_ptr,
    M,  # batch * H * W
    N,  # out_channels
    K,  # in_channels
    stride_M: tl.constexpr,
    stride_N: tl.constexpr,
    stride_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_m < M

    # load output accumulators
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE):
        offs_k = k + tl.arange(0, BLOCK_SIZE)
        # load weight: N x K
        w = tl.load(weight_ptr + (offs_k[:, None] * stride_N + tl.arange(0, BLOCK_SIZE)[None, :] * stride_K),
                    mask=offs_k[:, None] < K, other=0.0)
        # load input: M x K
        inp = tl.load(input_ptr + (offs_m[:, None] * stride_K + offs_k[None, :] * stride_K),
                      mask=mask[:, None] & (offs_k[None, :] < K), other=0.0)

        acc += tl.sum(inp * w, axis=1)

    tl.store(output_ptr + offs_m * stride_N, acc, mask=mask)


# 3x3 convolution with padding=1 (valid only for stride=1)
# We use im2col + GEMM approach
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_stages=3, num_warps=8),
    ],
    key=["out_h", "out_w", "K", "C", "BLOCK_SIZE"],
)
@triton.jit
def conv3x3_kernel(
    weight_ptr,  # [K, C, 3, 3]
    input_ptr,  # [B, C, H, W]
    output_ptr,  # [B, K, H, W]
    B, H, W, C, K,
    stride_B: tl.constexpr,
    stride_C: tl.constexpr,
    stride_H: tl.constexpr,
    stride_W: tl.constexpr,
    stride_K: tl.constexpr,
    stride_ker_C: tl.constexpr,
    stride_ker_H: tl.constexpr,
    stride_ker_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program processes one output element (b, k, h, w)
    total_out = B * K * H * W
    out_off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_off < total_out

    # decode out_off to (b, k, h, w)
    b = (out_off // (K * H * W)) % B
    k = (out_off // (H * W)) % K
    h = (out_off // W) % H
    w = out_off % W

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # iterate over channels
    for c in range(0, C, BLOCK_SIZE):
        c_off = c + tl.arange(0, BLOCK_SIZE)
        mask_c = c_off < C

        # load 3x3 patch for each channel
        for i in range(-1, 2):
            for j in range(-1, 2):
                in_h = tl.clamp(h + i, 0, H - 1)
                in_w = tl.clamp(w + j, 0, W - 1)
                inp = tl.load(
                    input_ptr
                    + (b * stride_B + c_off * stride_C + in_h * stride_H + in_w * stride_W),
                    mask=mask & mask_c,
                    other=0.0,
                )
                # weight index
                w_off = (k * stride_K + c_off * stride_ker_C + (i + 1) * stride_ker_H + (j + 1) * stride_ker_W)
                w_val = tl.load(weight_ptr + w_off, mask=mask_c, other=0.0)
                acc += inp * w_val

    tl.store(output_ptr + (b * stride_B + k * stride_K + h * stride_H + w * stride_W), acc, mask=mask)


# ----------------------------------
# Helper functions
# ----------------------------------

def conv1x1_torch(x, weight):
    """
    x: (B, C_in, H, W)
    weight: (C_out, C_in, 1, 1)
    """
    B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    # reshape to matrix: (B*H*W, C_in) * (C_in, C_out) -> (B*H*W, C_out)
    x_flat = x.reshape(B * H * W, C_in)
    weight_flat = weight.reshape(C_out, C_in).t()  # (C_in, C_out)
    out_flat = torch.empty((B * H * W, C_out), device=x.device, dtype=x.dtype)

    M = B * H * W
    N = C_out
    K = C_in
    grid = lambda meta: ((M + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    conv1x1_kernel[grid](
        weight_flat.data_ptr(),
        x_flat.data_ptr(),
        out_flat.data_ptr(),
        M, N, K,
        stride_M=x_flat.stride(0),
        stride_N=out_flat.stride(1),
        stride_K=x_flat.stride(1),
        BLOCK_SIZE=128,
    )
    return out_flat.reshape(B, C_out, H, W)


def conv3x3_torch(x, weight):
    """
    x: (B, C_in, H, W)
    weight: (C_out, C_in, 3, 3)
    """
    B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    out = torch.empty((B, C_out, H, W), device=x.device, dtype=x.dtype)

    total_out = B * C_out * H * W
    grid = lambda meta: ((total_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    conv3x3_kernel[grid](
        weight.data_ptr(),
        x.data_ptr(),
        out.data_ptr(),
        B, H, W, C_in, C_out,
        stride_B=x.stride(0),
        stride_C=x.stride(1),
        stride_H=x.stride(2),
        stride_W=x.stride(3),
        stride_K=out.stride(1),
        stride_ker_C=weight.stride(1),
        stride_ker_H=weight.stride(2),
        stride_ker_W=weight.stride(3),
        BLOCK_SIZE=128,
    )
    return out


# ----------------------------------
# New model
# ----------------------------------

class ModelNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(ModelNew, self).__init__()
        # store parameters
        self.in_channels = in_channels
        self.squeeze_channels = squeeze_channels
        self.expand1x1_channels = expand1x1_channels
        self.expand3x3_channels = expand3x3_channels

        # weights
        self.squeeze_weight = nn.Parameter(
            torch.randn(squeeze_channels, in_channels, 1, 1)
        )
        self.expand1x1_weight = nn.Parameter(
            torch.randn(expand1x1_channels, squeeze_channels, 1, 1)
        )
        self.expand3x3_weight = nn.Parameter(
            torch.randn(expand3x3_channels, squeeze_channels, 3, 3)
        )

    def forward(self, x):
        # squeeze
        x = conv1x1_torch(x, self.squeeze_weight)
        x = torch.relu(x, inplace=True)

        # expand1x1
        e1 = conv1x1_torch(x, self.expand1x1_weight)
        e1 = torch.relu(e1, inplace=True)

        # expand3x3
        e3 = conv3x3_torch(x, self.expand3x3_weight)
        e3 = torch.relu(e3, inplace=True)

        return torch.cat([e1, e3], dim=1)