import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_sum_max_mean_logsumexp_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_features, out_features,
    stride_xb, stride_wi, stride_wo, stride_ob,
    has_bias: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program IDs
    pid_b = tl.program_id(0)

    # Pointers for input batch row
    x_row_ptr = x_ptr + pid_b * stride_xb
    output_row_ptr = output_ptr + pid_b * stride_ob

    # Allocate shared memory for matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Matrix multiplication with bias
    for k in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        start_k = k * BLOCK_SIZE_K
        x_ptrs = x_row_ptr + start_k + tl.arange(0, BLOCK_SIZE_K)
        w_ptrs = weight_ptr + start_k * stride_wi + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_wi + tl.arange(0, BLOCK_SIZE_N)[None, :]

        x = tl.load(x_ptrs, mask=start_k + tl.arange(0, BLOCK_SIZE_K) < in_features, other=0.0)
        w = tl.load(w_ptrs, mask=start_k + tl.arange(0, BLOCK_SIZE_K)[:, None] < in_features, other=0.0)
        acc += tl.dot(x[None, :], w, out_dtype=tl.float32)

    # Add bias
    if has_bias:
        b_ptrs = bias_ptr + tl.arange(0, BLOCK_SIZE_N)
        bias = tl.load(b_ptrs, mask=tl.arange(0, BLOCK_SIZE_N) < out_features, other=0.0)
        acc += bias[None, :]

    # Reduce along output_features: sum, then max, then mean (all reduce to scalar per batch)
    # First: sum over out_features
    sum_val = tl.sum(acc, axis=1)[0]

    # Max over the same (though trivial since we have one value now)
    max_val = sum_val

    # Mean: divide by 1 (no-op), but conceptually it's there

    # First logsumexp: log(sum(exp(x))) but x is scalar -> log(exp(x)) = x
    lse1 = max_val

    # Second logsumexp: again, scalar -> identity
    lse2 = lse1

    # Store result
    tl.store(output_row_ptr, lse2)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        assert x.is_cuda, "Input tensor must be on GPU"
        batch_size = x.shape[0]

        # Output is (batch_size, 1)
        out = torch.empty((batch_size, 1), dtype=torch.float32, device=x.device)

        # Launch kernel
        def grid(meta):
            return (batch_size,)

        matmul_sum_max_mean_logsumexp_kernel[grid](
            x, self.weight, self.bias, out,
            batch_size, self.in_features, self.out_features,
            x.stride(0), self.weight.stride(1), self.weight.stride(0), out.stride(0),
            self.bias is not None,
            BLOCK_SIZE_M=1, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32
        )

        return out