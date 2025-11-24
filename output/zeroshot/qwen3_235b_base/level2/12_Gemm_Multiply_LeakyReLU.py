import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_gemm_mult_leaky_relu_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, in_features, out_features,
    multiplier, negative_slope,
    stride_xb, stride_xi,
    stride_wi, stride_wo,
    stride_ob, stride_oo,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Block IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for the tiles
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xb + offs_k[None, :] * stride_xi)
    weight_ptrs = weight_ptr + (offs_k[:, None] * stride_wi + offs_n[None, :] * stride_wo)
    output_ptrs = out_ptr + (offs_m[:, None] * stride_ob + offs_n[None, :] * stride_oo)

    # Accumulate matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over k blocks
    for k in range(0, in_features, BLOCK_K):
        # Load inputs
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < batch_size) & (offs_k[None, :] < in_features), other=0.0)
        w = tl.load(weight_ptrs, mask=(offs_k[:, None] < in_features) & (offs_n[None, :] < out_features), other=0.0)
        # Matmul
        acc += tl.dot(x, w)
        # Update pointers
        x_ptrs += BLOCK_K * stride_xi
        weight_ptrs += BLOCK_K * stride_wi

    # Add bias
    bias_ptrs = bias_ptr + offs_n * stride_wo
    bias = tl.load(bias_ptrs, mask=offs_n < out_features, other=0.0)
    acc += bias[None, :]

    # Multiply by scalar multiplier
    acc *= multiplier

    # Apply LeakyReLU: x >= 0 ? x : x * negative_slope
    acc = tl.where(acc >= 0, acc, acc * negative_slope)

    # Store result
    tl.store(output_ptrs, acc, mask=(offs_m[:, None] < batch_size) & (offs_n[None, :] < out_features))


class ModelNew(nn.Module):
    """
    Optimized version of Model using a fused Triton kernel for Gemm + Multiply + LeakyReLU.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        self.gemm = nn.Linear(in_features, out_features)

    def forward(self, x):
        assert x.is_cuda and self.gemm.weight.is_cuda and self.gemm.bias.is_cuda, "All tensors must be on CUDA."

        batch_size = x.shape[0]

        # Output tensor
        out = torch.empty((batch_size, self.out_features), device=x.device, dtype=x.dtype)

        # Launch kernel
        def grid(meta):
            return (
                triton.cdiv(batch_size, meta['BLOCK_M']),
                triton.cdiv(self.out_features, meta['BLOCK_N'])
            )

        # Autotune kernel for best performance
        fused_gemm_mult_leaky_relu_kernel[grid](
            x, self.gemm.weight, self.gemm.bias, out,
            batch_size, self.in_features, self.out_features,
            self.multiplier, self.negative_slope,
            x.stride(0), x.stride(1),
            self.gemm.weight.stride(0), self.gemm.weight.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )

        return out