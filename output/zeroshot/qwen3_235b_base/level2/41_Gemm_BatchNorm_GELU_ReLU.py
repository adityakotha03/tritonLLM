import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_gemm_bn_gelu_relu_kernel(
    x_ptr, weight_ptr, bias_ptr,
    bn_weight_ptr, bn_bias_ptr,
    running_mean_ptr, running_var_ptr,
    eps,
    out_ptr,
    batch_size, in_features, out_features,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < batch_size
    mask_n = n_offsets < out_features

    # Load batch norm params
    mean = tl.load(running_mean_ptr + n_offsets, mask=mask_n, other=0.0)
    var = tl.load(running_var_ptr + n_offsets, mask=mask_n, other=1.0)
    gamma = tl.load(bn_weight_ptr + n_offsets, mask=mask_n, other=1.0)
    beta = tl.load(bn_bias_ptr + n_offsets, mask=mask_n, other=0.0)
    inv_std = 1.0 / tl.sqrt(var + eps)
    scale = gamma * inv_std
    bias_bn = beta - mean * scale

    # Matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, in_features, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < in_features
        x = tl.load(
            x_ptr + m_offsets[:, None] * in_features + k_offsets[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )
        w = tl.load(
            weight_ptr + n_offsets[:, None] * in_features + k_offsets[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0
        )
        acc += tl.dot(x, w.t(), out_dtype=tl.float32)

    # Add bias
    acc += tl.load(bias_ptr + n_offsets, mask=mask_n, other=0.0)[None, :]

    # Apply batch norm (affine transform)
    acc = acc * scale[None, :] + bias_bn[None, :]

    # GELU activation
    gelu_out = 0.5 * acc * (1.0 + tl.math.tanh(0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)))

    # ReLU
    relu_out = tl.where(gelu_out > 0, gelu_out, 0.0)

    # Store output
    tl.store(out_ptr + m_offsets[:, None] * out_features + n_offsets[None, :],
             relu_out, mask=mask_m[:, None] & mask_n[None, :])


class FusedGEMMBNGLUReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias,
                bn_weight, bn_bias,
                running_mean, running_var, eps):
        batch_size, in_features = x.shape
        out_features = weight.shape[0]

        # Output tensor
        out = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)

        # Launch kernel
        def grid(meta):
            return (
                triton.cdiv(batch_size, meta['BLOCK_M']),
                triton.cdiv(out_features, meta['BLOCK_N'])
            )

        # Autotune and launch
        fused_gemm_bn_gelu_relu_kernel[grid](
            x, weight, bias,
            bn_weight, bn_bias,
            running_mean, running_var,
            eps,
            out,
            batch_size, in_features, out_features,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )

        return out


def fused_gemm_bn_gelu_relu(x, gemm_layer, bn_layer):
    return FusedGEMMBNGLUReLU.apply(
        x,
        gemm_layer.weight,
        gemm_layer.bias,
        bn_layer.weight,
        bn_layer.bias,
        bn_layer.running_mean,
        bn_layer.running_var,
        bn_layer.eps
    )


class ModelNew(nn.Module):
    """
    Optimized model with fused GEMM, BatchNorm, GELU, and ReLU using Triton.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = fused_gemm_bn_gelu_relu(x, self.gemm, self.batch_norm)
        return x