import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _layernorm_partial_sums_kernel(
    x_ptr,           # *float*, flattened input [M, K]
    sum_ptr,         # *float32*, per-row sum [M]
    sumsq_ptr,       # *float32*, per-row sumsq [M]
    count_ptr,       # *float32*, per-row count [M]
    M,               # number of rows
    K,               # number of columns (normalized dimension)
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_t = tl.program_id(1)
    # Bounds
    row_offset = pid_m * K
    start_k = pid_t * BLOCK_SIZE
    offsets = start_k + tl.arange(0, BLOCK_SIZE)
    mask = offsets < K

    # Load a tile
    x = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
    x = x.to(tl.float32)

    # Compute local partial reductions
    m = tl.where(mask, 1.0, 0.0)
    s = tl.sum(x, axis=0)
    s2 = tl.sum(x * x, axis=0)
    n = tl.sum(m, axis=0)

    # Atomically accumulate into per-row accumulators
    tl.atomic_add(sum_ptr + pid_m, s)
    tl.atomic_add(sumsq_ptr + pid_m, s2)
    tl.atomic_add(count_ptr + pid_m, n)


@triton.jit
def _layernorm_mean_rstd_kernel(
    sum_ptr,      # *float32*, [M]
    sumsq_ptr,    # *float32*, [M]
    count_ptr,    # *float32*, [M]
    mean_ptr,     # *float32*, [M]
    rstd_ptr,     # *float32*, [M]
    M,
    eps,
):
    pid = tl.program_id(0)
    mask = pid < M
    s = tl.load(sum_ptr + pid, mask=mask, other=0.0)
    s2 = tl.load(sumsq_ptr + pid, mask=mask, other=0.0)
    n = tl.load(count_ptr + pid, mask=mask, other=1.0)
    mean = s / n
    var = tl.maximum(s2 / n - mean * mean, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + pid, mean, mask=mask)
    tl.store(rstd_ptr + pid, rstd, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["K"],
)
@triton.jit
def _layernorm_apply_kernel(
    x_ptr,        # input *any dtype*, flattened [M, K]
    y_ptr,        # output *same dtype as x*, flattened [M, K]
    mean_ptr,     # *float32*, [M]
    rstd_ptr,     # *float32*, [M]
    w_ptr,        # *any dtype*, flattened [K] or dummy if HAS_W = False
    b_ptr,        # *any dtype*, flattened [K] or dummy if HAS_B = False
    M,
    K,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_t = tl.program_id(1)

    start_k = pid_t * BLOCK_SIZE
    offsets = start_k + tl.arange(0, BLOCK_SIZE)
    mask = offsets < K
    row_offset = pid_m * K

    x = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + pid_m)
    rstd = tl.load(rstd_ptr + pid_m)

    y = (x - mean) * rstd

    if HAS_W:
        w = tl.load(w_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        y = y * w
    if HAS_B:
        b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = y + b

    tl.store(y_ptr + row_offset + offsets, y, mask=mask)


def triton_layer_norm(x: torch.Tensor, normalized_shape: tuple, weight: torch.Tensor = None, bias: torch.Tensor = None, eps: float = 1e-5):
    """
    Triton-optimized LayerNorm over the last len(normalized_shape) dimensions.
    Falls back to torch.nn.functional.layer_norm on non-CUDA tensors.
    """
    if not x.is_cuda:
        return F.layer_norm(x, normalized_shape, weight, bias, eps)

    # Ensure input is contiguous
    x_contig = x.contiguous()
    # Validate shape
    assert tuple(x_contig.shape[-len(normalized_shape):]) == tuple(normalized_shape), "Input shape must end with normalized_shape."

    # Compute M (rows) and K (normalized dimension size)
    K = 1
    for s in normalized_shape:
        K *= s
    M = x_contig.numel() // K

    # Flatten input to [M, K]
    x_2d = x_contig.view(M, K)

    # Prepare output
    y = torch.empty_like(x_contig)
    y_2d = y.view(M, K)

    # Prepare weight/bias flattened if provided
    HAS_W = weight is not None
    HAS_B = bias is not None
    if HAS_W:
        w_flat = weight.contiguous().view(K)
    else:
        w_flat = torch.empty(0, device=x.device, dtype=x.dtype)
    if HAS_B:
        b_flat = bias.contiguous().view(K)
    else:
        b_flat = torch.empty(0, device=x.device, dtype=x.dtype)

    # Accumulator buffers
    sum_buf = torch.zeros(M, device=x.device, dtype=torch.float32)
    sumsq_buf = torch.zeros(M, device=x.device, dtype=torch.float32)
    count_buf = torch.zeros(M, device=x.device, dtype=torch.float32)
    mean_buf = torch.empty(M, device=x.device, dtype=torch.float32)
    rstd_buf = torch.empty(M, device=x.device, dtype=torch.float32)

    # Kernel 1: compute partial sums and accumulate atomically
    BLOCK_SIZE_SUM = 4096
    grid1 = lambda meta: (M, triton.cdiv(K, meta["BLOCK_SIZE"]))
    _layernorm_partial_sums_kernel[grid1](
        x_2d,
        sum_buf,
        sumsq_buf,
        count_buf,
        M,
        K,
        BLOCK_SIZE=BLOCK_SIZE_SUM,
        num_warps=4,
    )

    # Kernel 2: compute mean and rstd per row
    grid2 = lambda meta: (M,)
    _layernorm_mean_rstd_kernel[grid2](
        sum_buf,
        sumsq_buf,
        count_buf,
        mean_buf,
        rstd_buf,
        M,
        eps,
    )

    # Kernel 3: apply normalization and affine
    grid3 = lambda meta: (M, triton.cdiv(K, meta["BLOCK_SIZE"]))
    _layernorm_apply_kernel[grid3](
        x_2d,
        y_2d,
        mean_buf,
        rstd_buf,
        w_flat,
        b_flat,
        M,
        K,
        HAS_W=HAS_W,
        HAS_B=HAS_B,
    )

    return y


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using Triton kernels.
    """
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return triton_layer_norm(x, self.ln.normalized_shape, self.ln.weight, self.ln.bias, self.ln.eps)
        else:
            # Fallback to PyTorch LayerNorm on CPU
            return self.ln(x)


# Example input generation functions mirrored from the original
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]