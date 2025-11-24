import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------- Triton kernels ----------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def gemm_16f_kernel(
    A_ptr,  # [M, K]
    B_ptr,  # [K, N]
    C_ptr,  # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + (k + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_ak,
                    mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(B_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=offs_n[None, :] < N, other=0.0)
        acc += tl.dot(a, b)

    acc = acc.to(tl.float16)
    mask_m = offs_m < M
    mask_n = offs_n < N
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def bn_relu6_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    shift_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    inp = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    out = inp * scale + shift
    out = tl.where(out > 6.0, 6.0, tl.where(out > 0.0, out, 0.0))

    tl.store(output_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def add_kernel(
    a_ptr, b_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a + b, mask=mask)


# ---------- Helper wrappers ----------

def fused_bn_relu6(x, bn_layer: nn.BatchNorm2d):
    # precompute scale and shift
    eps = bn_layer.eps
    weight = bn_layer.weight
    bias = bn_layer.bias
    running_mean = bn_layer.running_mean
    running_var = bn_layer.running_var

    scale = weight / torch.sqrt(running_var + eps)
    shift = bias - weight * running_mean / torch.sqrt(running_var + eps)

    N = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    bn_relu6_kernel[grid](x.contiguous().view(-1),
                          out.contiguous().view(-1),
                          scale.contiguous().view(-1),
                          shift.contiguous().view(-1),
                          N,
                          BLOCK_SIZE=256)
    return out.view(x.shape)


def add_residual(x, residual):
    N = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x.contiguous().view(-1),
                     residual.contiguous().view(-1),
                     out.contiguous().view(-1),
                     N,
                     BLOCK_SIZE=256)
    return out.view(x.shape)


# ---------- Optimized model ----------

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(ModelNew, self).__init__()

        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=(kernel_size - 1) // 2,
                      groups=hidden_dim,
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x

        if hasattr(self, 'expand_conv'):
            # Conv1x1
            x = self.expand_conv[0](x)
            # fused BN+ReLU6
            x = fused_bn_relu6(x, self.expand_conv[1])

        # Depthwise conv
        x = self.depthwise_conv[0](x)
        x = fused_bn_relu6(x, self.depthwise_conv[1])

        # Project conv
        x = self.project_conv[0](x)
        x = fused_bn_relu6(x, self.project_conv[1])

        if self.use_residual:
            x = add_residual(x, identity)

        return x