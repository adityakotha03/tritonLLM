import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr,
    bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    subtract_value,
    multiply_value,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k) & (offs_am[:, None] < M), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_bn[None, :] < N), other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    bias_ptrs = bias_ptr + offs_cn
    bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0)
    c += bias[None, :]
    c = c - subtract_value
    c = c * multiply_value
    c = tl.where(c > 0, c, 0.0)
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.linear_bias = nn.Parameter(torch.empty(out_features))
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.linear_weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear_weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.linear_bias, -bound, bound)

    def forward(self, x):
        M, K = x.shape
        N = self.out_features
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)

        def grid(META):
            return (
                triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            )

        matmul_relu_kernel[grid](
            x, self.linear_weight, y,
            self.linear_bias,
            M, N, K,
            x.stride(0), x.stride(1),
            self.linear_weight.stride(0), self.linear_weight.stride(1),
            y.stride(0), y.stride(1),
            self.subtract_value,
            self.multiply_value,
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=8,
        )
        return y