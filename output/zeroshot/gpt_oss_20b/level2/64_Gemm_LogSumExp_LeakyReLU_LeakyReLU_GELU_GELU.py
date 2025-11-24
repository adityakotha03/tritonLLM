import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    ],
    key=["in_features"],
)
@triton.jit
def post_linear_kernel(
    in_ptr: tl.tensor,            # pointer to linear output
    out_ptr: tl.tensor,           # pointer to scalar output
    batch: tl.int32,              # batch size
    in_features: tl.int32,        # number of columns in linear output
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)

    # Load linear output row in tiles and compute logsumexp
    max_val = tl.float32(-float("inf"))
    sum_exp = tl.float32(0.0)

    # Process the row in tiles of BLOCK_SIZE
    for offset in range(0, in_features, BLOCK_SIZE):
        # Load a chunk of the row
        indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = indices < in_features
        vals = tl.load(in_ptr + batch_id * in_features + indices, mask=mask, other=0.0)

        # Compute max in this chunk
        chunk_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, chunk_max)

        # Temporarily store exp with current max (will subtract global max later)
        exp_vals = tl.exp(vals - chunk_max)
        sum_exp += tl.sum(exp_vals, axis=0)

    # Final log-sum-exp value for this row
    logsumexp_val = max_val + tl.log(sum_exp)

    # LeakyReLU (negative_slope=0.01) twice
    neg_slope = tl.float32(0.01)
    x = logsumexp_val
    x = tl.where(x > 0, x, x * neg_slope)
    x = tl.where(x > 0, x, x * neg_slope)

    # GELU (approximate version)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    sqrt2pi = tl.float32(0.7978845608028654)  # sqrt(2/pi)
    x_cubed = x * x * x
    inner = tl.tanh(sqrt2pi * (x + tl.float32(0.044715) * x_cubed))
    x = tl.float32(0.5) * x * (tl.float32(1.0) + inner)

    # GELU again
    x_cubed = x * x * x
    inner = tl.tanh(sqrt2pi * (x + tl.float32(0.044715) * x_cubed))
    x = tl.float32(0.5) * x * (tl.float32(1.0) + inner)

    # Store the result
    tl.store(out_ptr + batch_id, x)


def triton_post_linear(x: torch.Tensor) -> torch.Tensor:
    """
    Wraps the Triton kernel that performs LogSumExp followed by two
    LeakyReLU and two GELU activations on the output of a linear layer.
    """
    assert x.is_cuda, "Input must be on CUDA."

    batch, in_features = x.shape
    out = torch.empty(batch, 1, device=x.device, dtype=x.dtype)

    grid = lambda meta: (batch,)
    post_linear_kernel[grid](x, out, batch, in_features, BLOCK_SIZE=tl.constexpr)

    return out


class ModelNew(nn.Module):
    """
    Optimized model that replaces the post‑linear sequence of
    LogSumExp, LeakyReLU, LeakyReLU, GELU, GELU with a single
    fused Triton kernel.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gemm + bias
        x = self.linear(x)
        # Fused post‑linear ops
        return triton_post_linear(x)