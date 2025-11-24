import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _group_norm_kernel(
    X_ptr, gamma_ptr, beta_ptr, Y_ptr,
    rsqrt_var_ptr,
    M,  # batch * D * H * W
    N,  # channels
    G,
    num_groups: tl.constexpr,
    block_M: tl.constexpr,
    block_N: tl.constexpr,
    eps: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs_M = tl.cdiv(M, block_M)
    pid_m = pid % num_programs_M
    pid_g = pid // num_programs_M

    group_size = N // num_groups

    channels_per_block = tl.cdiv(num_groups, num_programs_M) * block_N
    channel_start = pid_g * group_size
    channel_end = min((pid_g + 1) * group_size, N)

    offset_m = pid_m * block_M + tl.arange(0, block_M)
    mask_m = offset_m < M
    mask_n = None

    X_block_ptr = tl.make_block_ptr(
        base=X_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * block_M, channel_start),
        block_shape=(block_M, channels_per_block),
        order=(1, 0)
    )

    gamma = tl.load(gamma_ptr + tl.arange(0, block_N), mask=channel_start + tl.arange(0, block_N) < channel_end, other=1.0)
    beta = tl.load(beta_ptr + tl.arange(0, block_N), mask=channel_start + tl.arange(0, block_N) < channel_end, other=0.0)

    mean = tl.zeros((block_M,), dtype=tl.float32)
    count = 0
    for c in range(channel_start, channel_end):
        x = tl.load(X_ptr + offset_m * N + c, mask=mask_m, other=0.0)
        mean += x
        count += 1
    mean = mean / count

    var = tl.zeros((block_M,), dtype=tl.float32)
    for c in range(channel_start, channel_end):
        x = tl.load(X_ptr + offset_m * N + c, mask=mask_m, other=0.0)
        x_centered = x - mean
        var += x_centered * x_centered
    var = var / count
    inv_var = tl.math.rsqrt(var + eps)

    tl.store(rsqrt_var_ptr + pid_m * block_M + tl.arange(0, block_M), inv_var, mask=mask_m)

    X_block = tl.load(X_block_ptr, boundary_check=(0,1), padding_option="zero")
    X_centered = X_block - mean[:, None]
    X_norm = X_centered * inv_var[:, None]

    output = X_norm * gamma[None, :] + beta[None, :]
    Y_block_ptr = tl.make_block_ptr(
        base=Y_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * block_M, channel_start),
        block_shape=(block_M, channels_per_block),
        order=(1, 0)
    )
    tl.store(Y_block_ptr, output, boundary_check=(0,1))


@triton.jit
def _mean_kernel(
    X_ptr, out_ptr,
    M, N,
    block_M: tl.constexpr,
    block_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offset_m = pid * block_M + tl.arange(0, block_M)
    mask_m = offset_m < M
    acc = tl.zeros((block_N,), dtype=tl.float32)
    for i in range(0, N, block_N):
        offset_n = i + tl.arange(0, block_N)
        mask_n = offset_n < N
        mask = mask_m[:, None] & mask_n[None, :]
        data = tl.load(X_ptr + offset_m[:, None] * N + offset_n[None, :], mask=mask, other=0.0)
        acc += tl.sum(data, axis=0)
    total = tl.sum(acc)
    if pid == 0:
        tl.store(out_ptr, total / (M * N))


def triton_group_norm(x, num_groups, weight, bias, eps=1e-5):
    M = x.shape[0] * x.shape[2] * x.shape[3] * x.shape[4]  # B * D * H * W
    N = x.shape[1]  # C
    assert N % num_groups == 0
    out = torch.empty_like(x)
    rsqrt_var = torch.empty((M,), dtype=torch.float32, device=x.device)

    def grid_meta(meta):
        num_programs_M = triton.cdiv(M, meta['block_M'])
        return (num_programs_M * num_groups,)

    _group_norm_kernel[grid_meta](
        x, weight, bias, out, rsqrt_var,
        M, N, num_groups,
        num_groups=num_groups,
        block_M=1024,
        block_N=32,
        eps=eps,
    )
    return out


def triton_mean(x, dims):
    keep_dims = [i for i in range(x.ndim) if i not in dims]
    if len(keep_dims) == 0:
        return torch.sum(x) / x.numel()
    x_reshaped = x.permute(keep_dims + dims).contiguous()
    shape_keep = x_reshaped.shape[:len(keep_dims)]
    x_flattened = x_reshaped.view(-1, x_reshaped.numel() // shape_keep.numel())
    M, N = x_flattened.shape
    out = torch.empty((1,), dtype=x.dtype, device=x.device)

    _mean_kernel[(triton.cdiv(M, 1024),)](
        x_flattened, out, M, N, block_M=1024, block_N=32
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = triton_group_norm(x, self.group_norm.num_groups, self.group_norm.weight, self.group_norm.bias, self.group_norm.eps)
        x = triton_mean(x, dims=[1, 2, 3, 4])
        return x