import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_gemm_swish_div_clamp_tanh_clamp_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_features, out_features,
    stride_xb, stride_wi, stride_oi,
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for tiles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for loading
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xb + offs_k[None, :] * stride_oi)
    weight_ptrs = weight_ptr + (offs_k[:, None] * stride_wi + offs_n[None, :] * stride_oi)
    output_ptrs = output_ptr + (offs_m[:, None] * out_features + offs_n[None, :])

    # Block for K
    k_range = tl.cdiv(in_features, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # GEMM in blocks
    for k in range(k_range):
        # Load input and weights
        mask_x = (offs_m[:, None] < batch_size) & (offs_k[None, :] < in_features)
        mask_w = (offs_k[:, None] < in_features) & (offs_n[None, :] < out_features)
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w = tl.load(weight_ptrs, mask=mask_w, other=0.0)
        # Accumulate in float32
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_oi
        weight_ptrs += BLOCK_K * stride_wi

    # Add bias if present
    if has_bias:
        bias_ptrs = bias_ptr + offs_n
        mask_bias = offs_n < out_features
        bias = tl.load(bias_ptrs, mask=mask_bias, other=0.0)
        acc = acc + bias[None, :]

    # Swish: x * sigmoid(x)
    acc = acc * tl.sigmoid(acc)

    # Divide by 2
    acc = acc / 2.0

    # Clamp between -1 and 1
    acc = tl.clamp(acc, -1.0, 1.0)

    # Tanh
    acc = tl.tanh(acc)

    # Final clamp between -1 and 1 (redundant but kept as per original)
    acc = tl.clamp(acc, -1.0, 1.0)

    # Write output
    mask_o = (offs_m[:, None] < batch_size) & (offs_n[None, :] < out_features)
    tl.store(output_ptrs, acc, mask=mask_o)


def triton_fused_linear_swish_div_clamp_tanh_clamp(x, weight, bias=None):
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    if bias is not None:
        assert bias.is_cuda
    x = x.contiguous()
    weight = weight.t().contiguous()  # Transpose for efficient tiling
    has_bias = bias is not None
    if has_bias:
        bias = bias.contiguous()

    batch_size, in_features = x.shape
    out_features = weight.shape[0]

    # Output tensor
    output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)

    # Autotune block sizes
    def grid(META):
        return (
            triton.cdiv(batch_size, META['BLOCK_M']),
            triton.cdiv(out_features, META['BLOCK_N']),
        )

    # Launch kernel with autotuning
    triton_fused_linear_swish_div_clamp_tanh_clamp_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        stride_xb=x.stride(0),
        stride_wi=weight.stride(1),
        stride_oi=1,
        has_bias=has_bias,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
    )
    return output


class ModelNew(nn.Module):
    """
    Optimized version of Model using fused Triton kernel for GEMM + Swish + Divide + Clamp + Tanh + Clamp.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return triton_fused_linear_swish_div_clamp_tanh_clamp(x, self.weight, self.bias)