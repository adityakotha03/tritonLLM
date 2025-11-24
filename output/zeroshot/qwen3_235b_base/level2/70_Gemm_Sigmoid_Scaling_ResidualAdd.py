import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_gemm_sigmoid_scaling_residual_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    scaling_factor,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    weight_ptrs = weight_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k)
        w_mask = (offs_k[:, None] < K - k) & (offs_n[None, :] < N)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(weight_ptrs, mask=w_mask, other=0.0)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_SIZE_K * stride_xk
        weight_ptrs += BLOCK_SIZE_K * stride_wk

    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_n * 1
        bias_mask = offs_n < N
        bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
        accumulator += bias[None, :]

    original_x = accumulator
    sigmoid_x = tl.sigmoid(accumulator)
    scaled_sigmoid = sigmoid_x * scaling_factor
    output = scaled_sigmoid + original_x

    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, output, mask=output_mask)


def triton_fused_gemm_sigmoid_scaling_residual(x, weight, bias, scaling_factor):
    M, K = x.shape
    K, N = weight.shape
    assert x.is_cuda and weight.is_cuda
    if bias is not None:
        assert bias.is_cuda
    x = x.contiguous()
    weight = weight.t().contiguous()  # Transpose to row-major
    weight_t = weight
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_SIZE_M']),
            triton.cdiv(N, META['BLOCK_SIZE_N']),
        )

    has_bias = bias is not None

    fused_gemm_sigmoid_scaling_residual_kernel[grid](
        x_ptr=x,
        weight_ptr=weight_t,
        bias_ptr=bias,
        output_ptr=output,
        M=M, N=N, K=K,
        stride_xm=x.stride(0), stride_xk=x.stride(1),
        stride_wk=weight_t.stride(0), stride_wn=weight_t.stride(1),
        stride_om=output.stride(0), stride_on=output.stride(1),
        scaling_factor=scaling_factor,
        HAS_BIAS=has_bias,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
    )
    return output


class ModelNew(nn.Module):
    """
    Optimized version of Model using a fused Triton kernel for Gemm + Sigmoid + Scaling + ResidualAdd.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size, bias=True)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        return triton_fused_gemm_sigmoid_scaling_residual(
            x, self.linear.weight, self.linear.bias, self.scaling_factor
        )