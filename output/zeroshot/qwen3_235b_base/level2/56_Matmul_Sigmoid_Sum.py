import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_bm,
    stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_k = tl.arange(0, BLOCK_K)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        accumulator = tl.dot(x, w, acc=accumulator)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    b_ptrs = b_ptr + offs_n * stride_bm
    mask_b = offs_n < N
    bias = tl.load(b_ptrs, mask=mask_b, other=0.0).expand_dims(0)
    accumulator = accumulator + bias

    sigmoid_output = tl.sigmoid(accumulator)

    sum_output = tl.sum(sigmoid_output, axis=1)[:, None]

    offs_out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_ptrs = out_ptr + offs_out_m * stride_om
    mask_out = offs_out_m < M
    tl.store(out_ptrs, sum_output, mask=mask_out)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        M, K = x.shape
        N = self.linear.out_features

        w = self.linear.weight
        b = self.linear.bias
        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)

        def grid(META):
            return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

        matmul_sigmoid_sum_kernel[grid](
            x, w, b, out,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            b.stride(0),
            out.stride(0),
            BLOCK_M=64, BLOCK_N=32, BLOCK_K=32,
            GROUP_M=8,
        )
        return out