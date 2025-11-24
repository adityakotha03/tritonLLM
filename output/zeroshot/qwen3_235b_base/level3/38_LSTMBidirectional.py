import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_linear_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, seq_len, in_features, out_features,
    stride_xb, stride_xs, stride_xi,
    stride_wi, stride_wo,
    stride_ob, stride_os, stride_oo,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (offs_m[:, None] // seq_len) * stride_xb + \
                    (offs_m[:, None] % seq_len) * stride_xs + \
                    offs_k[None, :] * stride_xi
    weight_ptrs = weight_ptr + offs_k[:, None] * stride_wi + offs_n[None, :] * stride_wo

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, in_features, BLOCK_SIZE_K):
        x_mask = (offs_m[:, None] < batch_size * seq_len) & (offs_k[None, :] < in_features)
        w_mask = (offs_k[:, None] < in_features) & (offs_n[None, :] < out_features)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(weight_ptrs, mask=w_mask, other=0.0)

        accumulator = tl.dot(x, w, acc=accumulator)

        x_ptrs += BLOCK_SIZE_K * stride_xi
        weight_ptrs += BLOCK_SIZE_K * stride_wi

    c = accumulator.to(tl.float16)

    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_n * stride_wo
        bias = tl.load(bias_ptrs, mask=offs_n < out_features, other=0.0)
        c += bias[None, :]

    if ACT == "relu":
        c = tl.maximum(0, c)
    elif ACT == "sigmoid":
        c = tl.sigmoid(c)
    elif ACT == "tanh":
        c = tl.tanh(c)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + (offs_m[:, None] // seq_len) * stride_ob + \
                              (offs_m[:, None] % seq_len) * stride_os + \
                              offs_n[None, :] * stride_oo
    mask = (offs_m[:, None] < batch_size * seq_len) & (offs_n[None, :] < out_features)
    tl.store(output_ptrs, c, mask=mask)


def triton_linear(x, weight, bias=None, activation=None):
    assert x.is_cuda and weight.is_cuda
    if bias is not None:
        assert bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, seq_len, in_features = x.shape
    out_features = weight.shape[0]
    out = torch.empty((batch_size, seq_len, out_features), device=x.device, dtype=torch.float16)

    def grid(META):
        return (
            triton.cdiv(batch_size * seq_len, META["BLOCK_SIZE_M"]),
            triton.cdiv(out_features, META["BLOCK_SIZE_N"]),
        )

    fused_linear_kernel[grid](
        x_ptr=x.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias.data_ptr() if bias is not None else 0,
        output_ptr=out.data_ptr(),
        batch_size=batch_size,
        seq_len=seq_len,
        in_features=in_features,
        out_features=out_features,
        stride_xb=x.stride(0),
        stride_xs=x.stride(1),
        stride_xi=x.stride(2),
        stride_wi=weight.stride(1),
        stride_wo=weight.stride(0),
        stride_ob=out.stride(0),
        stride_os=out.stride(1),
        stride_oo=out.stride(2),
        HAS_BIAS=bias is not None,
        ACT=activation,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['in_features', 'out_features'],
)
@triton.jit
def fused_matmul_add_relu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_ptrs = a_ptr + (offs_m[:, None] // 1) * stride_am + (offs_k[None, :] // 1) * stride_ak
        b_ptrs = b_ptr + (offs_k[:, None] // 1) * stride_bk + (offs_n[None, :] // 1) * stride_bn

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        accumulator = tl.dot(a, b, acc=accumulator)

        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    c = tl.maximum(0, c)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul_add_relu(a, b, c):
    assert a.is_cuda and b.is_cuda
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K, N = b.shape

    out = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    fused_matmul_add_relu_kernel[grid](
        a_ptr=a.data_ptr(),
        b_ptr=b.data_ptr(),
        c_ptr=out.data_ptr(),
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=out.stride(0), stride_cn=out.stride(1),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

    def forward(self, x, h0, c0):
        with torch.no_grad():
            out, hn = self.lstm(x, (h0, c0))

        batch_size, seq_len, _ = out.shape
        weight = self.fc.weight
        bias = self.fc.bias

        out = triton_linear(out[:, -1, :].unsqueeze(1), weight, bias)
        out = out.squeeze(1)
        return out