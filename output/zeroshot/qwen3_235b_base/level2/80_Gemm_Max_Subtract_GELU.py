import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_gemm_max_mean_gelu_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_features, out_features,
    max_dim, stride_xb, stride_xf,
    stride_wout, stride_wf,
    stride_ob, stride_om, stride_of,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_out = tl.program_id(1)

    # Pointers for this batch and output
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(batch_size, in_features),
        strides=(stride_xb, stride_xf),
        offsets=(pid_b, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0)
    )
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(out_features, in_features),
        strides=(stride_wout, stride_wf),
        offsets=(pid_out, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(1, 0)
    )

    # Load input and weight blocks
    x = tl.load(x_block_ptr, boundary_check=(0,1), padding_option="zero")
    w = tl.load(weight_block_ptr, boundary_check=(0,1), padding_option="zero")

    # Perform GEMM
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc = tl.dot(x, w.T, acc)

    # Add bias if exists
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid_out)
        acc += bias

    # Apply max reduction over max_dim (dim=1) -> (batch, 1)
    if max_dim == 1:
        row_max = tl.max(acc, axis=1)
        row_mean = tl.sum(acc, axis=1) / out_features
        acc = acc - row_mean[:, None]
        acc = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.70710678))  # GELU
        # Store max value separately
        output_max_ptr = output_ptr + pid_b * stride_ob + pid_out * stride_om
        tl.store(output_max_ptr, row_max)
    else:
        # This kernel assumes max_dim=1; otherwise fallback is needed
        pass


def triton_fused_gemm_max_mean_gelu(x, weight, bias, max_dim=1):
    assert max_dim == 1, "Only max_dim=1 supported"
    batch_size, in_features = x.shape
    out_features, _ = weight.shape

    # Output is (batch_size, out_features) after GEMM, then we reduce max over dim=1 -> (batch_size, 1)
    # But we need to return (batch_size, out_features) after GELU
    # So we keep full shape
    output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)

    def grid(META):
        return (batch_size, out_features)

    # Choose block sizes
    BLOCK_SIZE_M = min(64, batch_size)
    BLOCK_SIZE_N = min(64, out_features)
    BLOCK_SIZE_K = triton.next_power_of_2(in_features)

    fused_gemm_max_mean_gelu_kernel[grid](
        x, weight, bias, output,
        batch_size, in_features, out_features,
        max_dim,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), 1, output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return output


class ModelNew(nn.Module):
    """
    Optimized model with fused GEMM, max, mean subtraction, and GELU using Triton.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=True)
        self.max_dim = max_dim

    def forward(self, x):
        x = triton_fused_gemm_max_mean_gelu(x, self.gemm.weight, self.gemm.bias, self.max_dim)
        return x