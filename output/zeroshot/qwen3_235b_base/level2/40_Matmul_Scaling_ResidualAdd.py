import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_residual_scale_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    batch_size, in_features, out_features,
    scaling_factor,
    stride_xb, stride_xi,
    stride_wi, stride_wo,
    stride_ob, stride_oo,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Offsets for matrix multiplication
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % batch_size
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % out_features
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for input x and weight w
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xb + offs_k[None, :] * stride_xi)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wi + offs_n[None, :] * stride_wo)

    # Accumulator for matmul result
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Matrix multiplication loop
    for k in range(0, tl.cdiv(in_features, BLOCK_K)):
        # Load tiles of x and w
        x_mask = (offs_m[:, None] < batch_size) & (offs_k[None, :] < in_features)
        w_mask = (offs_k[:, None] < in_features) & (offs_n[None, :] < out_features)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # GEMM operation
        accumulator += tl.dot(x, w)

        # Update pointers
        x_ptrs += BLOCK_K * stride_xi
        w_ptrs += BLOCK_K * stride_wi

    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < out_features, other=0.0)
        accumulator += bias[None, :]

    # Residual connection: store original matmul output
    original_output = accumulator

    # Apply scaling: output = output * scaling_factor
    accumulator *= scaling_factor

    # Add residual: output = output + original_output
    accumulator += original_output

    # Output pointers
    offs_out_b = pid_b * stride_ob + offs_m * stride_ob
    offs_out_o = offs_n * stride_oo
    out_ptrs = out_ptr + (offs_out_b[:, None] + offs_out_o[None, :])

    # Output mask
    out_mask = (offs_m[:, None] < batch_size) & (offs_n[None, :] < out_features)

    # Store result
    tl.store(out_ptrs, accumulator, mask=out_mask)


class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom Triton kernel that fuses
    linear layer (matmul + bias) with scaling and residual addition.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        assert x.is_cuda, "Input tensor must be on GPU."
        assert self.weight.is_cuda and self.bias.is_cuda, "Model parameters must be on GPU."

        batch_size = x.shape[0]

        # Output tensor
        out = torch.empty((batch_size, self.out_features), device=x.device, dtype=x.dtype)

        # Launch kernel
        def grid(META):
            return (
                triton.cdiv(batch_size, META['BLOCK_M']),
                triton.cdiv(self.out_features, META['BLOCK_N']),
                triton.cdiv(self.in_features, META['BLOCK_K'])
            )

        matmul_residual_scale_kernel[grid](
            x, self.weight, self.bias, out,
            batch_size, self.in_features, self.out_features,
            self.scaling_factor,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )

        return out