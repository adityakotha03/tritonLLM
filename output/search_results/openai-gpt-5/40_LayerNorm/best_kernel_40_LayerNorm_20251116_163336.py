# Best Kernel for 40_LayerNorm
# Generated: 20251116_163336
# Speedup: 17.81x
# Runtime: 0.5670 ms
# Round: 1
# Idea: - Description: Implement a two-pass kernel sequence with pipelined tiles in each pass. In the reduction pass, stream tiles of size T (e.g., 4K–8K elements) from global to shared using cp.async, double-buffering (2–3 stages) so that while tile t is being reduced (Welford in FP32), tile t+1 is being prefetched. In the normalization pass, use the same pipeline to load x, gamma, beta asynchronously while normalizing the previously loaded tile with precomputed mean/inv_std. Use SMEM circular buffers sized for the tile (e.g., T * sizeof(dtype) per array; for FP16 x/gamma/beta ~ 24 KB per stage at T=4096; still well within 163 KB per block even with 2–3 stages). Ensure 16B alignment for cp.async and choose vector width so each warp issues 128B-aligned transactions. - Why it helps on A100: Ampere’s cp.async lets you overlap global memory latency with compute. Double/triple buffering hides memory latency and keeps the FP32 reductions busy. With 164 KB SMEM per SM, you can sustain multiple stages and warps per block without starving occupancy. This moves you closer to the 1.9 TB/s peak bandwidth. - Targets: Asynchronous operations & latency hiding + memory access efficiency (vectorized, aligned loads) + compute pipeline efficiency (reduction/normalization overlap with prefetch).

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_reduce_kernel(
    x_ptr,            # *[B, N] flattened input
    sums_ptr,         # *[B] partial sum buffer
    sumsq_ptr,        # *[B] partial sumsq buffer
    N,                # number of features per row
    stride_row,       # stride between rows in elements
    CHUNK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)  # row id (batch element)
    pid_g = tl.program_id(1)  # tile-group id along N

    row_start = pid_b * stride_row
    base_tile = pid_g * CHUNK * BLOCK_SIZE

    acc_sum = tl.zeros((), dtype=tl.float32)
    acc_sumsq = tl.zeros((), dtype=tl.float32)

    for j in range(CHUNK):
        tile_start = base_tile + j * BLOCK_SIZE
        offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        x_ptrs = x_ptr + row_start + offsets
        tl.multiple_of(x_ptrs, 16)
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Accumulate tile-wise sums (vector reduction -> scalars)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)

    # Atomically add per-program accumulators into global row accumulators
    tl.atomic_add(sums_ptr + pid_b, acc_sum)
    tl.atomic_add(sumsq_ptr + pid_b, acc_sumsq)


@triton.jit
def _ln_finalize_stats_kernel(
    sums_ptr,   # *[B]
    sumsq_ptr,  # *[B]
    mean_ptr,   # *[B]
    rstd_ptr,   # *[B]
    N,          # int
    eps,        # float
):
    pid = tl.program_id(0)
    s = tl.load(sums_ptr + pid).to(tl.float32)
    ss = tl.load(sumsq_ptr + pid).to(tl.float32)
    n = N
    mean = s / n
    var = ss / n - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def _ln_norm_kernel(
    x_ptr,       # *[B, N]
    y_ptr,       # *[B, N]
    gamma_ptr,   # *[N]
    beta_ptr,    # *[N]
    mean_ptr,    # *[B]
    rstd_ptr,    # *[B]
    N,
    stride_row,
    CHUNK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    row_start = pid_b * stride_row
    base_tile = pid_g * CHUNK * BLOCK_SIZE

    mean = tl.load(mean_ptr + pid_b).to(tl.float32)
    rstd = tl.load(rstd_ptr + pid_b).to(tl.float32)

    for j in range(CHUNK):
        tile_start = base_tile + j * BLOCK_SIZE
        offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        x_ptrs = x_ptr + row_start + offsets
        g_ptrs = gamma_ptr + offsets
        b_ptrs = beta_ptr + offsets

        tl.multiple_of(x_ptrs, 16)
        tl.multiple_of(g_ptrs, 16)
        tl.multiple_of(b_ptrs, 16)

        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptrs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=mask, other=0.0).to(tl.float32)

        y = (x - mean) * rstd
        y = y * g + b
        tl.store(y_ptr + row_start + offsets, y.to(tl.float32), mask=mask)


class ModelNew(nn.Module):
    """
    Layer Normalization with two-pass Triton kernels:
      - Pass 1: Streamed reduction with pipelined tiles (partial sums/sumsq with atomics).
      - Pass 2: Streamed normalization with gamma/beta using the precomputed mean/rstd.
    """
    def __init__(self, normalized_shape: tuple, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        # Ensure contiguous memory layout
        x = x.contiguous()
        # Compute batch (rows) and normalized feature size (cols)
        norm_ndim = len(self.normalized_shape)
        assert norm_ndim >= 1 and tuple(x.shape[-norm_ndim:]) == self.normalized_shape, \
            "Input's last dims must match normalized_shape"
        rows = x.numel() // math.prod(self.normalized_shape)
        N = math.prod(self.normalized_shape)

        x2d = x.view(rows, N)
        stride_row = x2d.stride(0)

        # Flatten gamma/beta to length N
        gamma = self.weight.contiguous().view(N)
        beta = self.bias.contiguous().view(N)

        # Buffers for statistics
        sums = torch.zeros(rows, dtype=torch.float32, device=x.device)
        sumsq = torch.zeros(rows, dtype=torch.float32, device=x.device)

        # Tunable kernel parameters
        BLOCK_SIZE = 1024  # 1K elements per inner block
        CHUNK = 8          # 8 blocks per program => 8K elements/program
        tiles_per_program = BLOCK_SIZE * CHUNK
        G = (N + tiles_per_program - 1) // tiles_per_program
        grid_reduce = (rows, G)
        grid_finalize = (rows,)
        grid_norm = (rows, G)

        # Pass 1: Reduction (partial sums + sumsq). Use num_stages>1 for async pipelining.
        _ln_reduce_kernel[grid_reduce](
            x2d, sums, sumsq, N, stride_row,
            CHUNK=CHUNK, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8, num_stages=3,
        )

        # Finalize statistics per row
        mean = torch.empty(rows, dtype=torch.float32, device=x.device)
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
        _ln_finalize_stats_kernel[grid_finalize](
            sums, sumsq, mean, rstd, N, self.eps,
            num_warps=1, num_stages=1,
        )

        # Pass 2: Normalization
        y2d = torch.empty_like(x2d, dtype=torch.float32)
        _ln_norm_kernel[grid_norm](
            x2d, y2d, gamma, beta, mean, rstd, N, stride_row,
            CHUNK=CHUNK, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8, num_stages=3,
        )

        y = y2d.view_as(x)
        return y