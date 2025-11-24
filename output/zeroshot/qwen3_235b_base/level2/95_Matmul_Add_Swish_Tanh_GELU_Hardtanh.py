import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_matmul_add_swish_tanh_gelu_hardtanh_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, in_features, out_features,
    stride_xb, stride_xi,
    stride_wi, stride_wo,
    stride_bb,
    stride_ob, stride_oo,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    # 2D block IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Pointers for the tiles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Input and weight blocks
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xb + offs_k[None, :] * stride_xi)
    w_ptrs = weight_ptr + (offs_k[:, None] * stride_wi + offs_n[None, :] * stride_wo)

    # Accumulator for matmul
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Matrix multiplication loop
    for k in range(0, in_features, BLOCK_K):
        # Load tiles with masks
        x_mask = (offs_m[:, None] < batch_size) & (offs_k[None, :] < in_features)
        w_mask = (offs_k[:, None] < in_features) & (offs_n[None, :] < out_features)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Matmul with Tensor Core
        acc += tl.dot(x, w, out_dtype=tl.float32, allow_tf32=USE_TF32)

        # Update pointers
        x_ptrs += BLOCK_K * stride_xi
        w_ptrs += BLOCK_K * stride_wi

    # Add bias (add_value)
    bias_ptrs = bias_ptr + offs_n * stride_bb
    bias_mask = offs_n < out_features
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    acc += bias[None, :]

    # Swish: sigmoid(x) * x
    acc = acc * tl.sigmoid(acc)

    # Tanh
    acc = tl.tanh(acc)

    # GELU approximation using tanh
    # gelu(x) = x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_cubed = acc * acc * acc
    inner = 0.7978845608028654 * (acc + 0.044715 * x_cubed)  # sqrt(2/pi) ~ 0.79788
    gelu = acc * 0.5 * (1.0 + tl.tanh(inner))
    acc = gelu

    # Hardtanh: clamp between -1 and 1
    acc = tl.clamp(acc, -1.0, 1.0)

    # Store output
    out_ptrs = out_ptr + (offs_m[:, None] * stride_ob + offs_n[None, :] * stride_oo)
    out_mask = (offs_m[:, None] < batch_size) & (offs_n[None, :] < out_features)
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_fused_linear_activation(x, weight, bias):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch_size, in_features = x.shape
    out_features = weight.shape[0]

    # Output tensor
    out = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)

    # Launch grid
    def grid(META):
        return (
            triton.cdiv(batch_size, META['BLOCK_M']),
            triton.cdiv(out_features, META['BLOCK_N']),
        )

    # Autotune for best BLOCK sizes
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        ],
        key=['in_features', 'out_features'],
    )
    @triton.jit
    def kernel_caller(
        x_ptr, weight_ptr, bias_ptr, out_ptr,
        batch_size, in_features, out_features,
        stride_xb, stride_xi,
        stride_wi, stride_wo,
        stride_bb,
        stride_ob, stride_oo,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        fused_matmul_add_swish_tanh_gelu_hardtanh_kernel(
            x_ptr, weight_ptr, bias_ptr, out_ptr,
            batch_size, in_features, out_features,
            stride_xb, stride_xi,
            stride_wi, stride_wo,
            stride_bb,
            stride_ob, stride_oo,
            BLOCK_M, BLOCK_N, BLOCK_K,
            USE_TF32=True,
        )

    kernel_caller[grid](
        x, weight, bias, out,
        batch_size, in_features, out_features,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        bias.stride(0),
        out.stride(0), out.stride(1),
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using a fused Triton kernel for matmul + add + Swish + Tanh + GELU + Hardtanh.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(add_value_shape))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return triton_fused_linear_activation(x, self.weight, self.bias)