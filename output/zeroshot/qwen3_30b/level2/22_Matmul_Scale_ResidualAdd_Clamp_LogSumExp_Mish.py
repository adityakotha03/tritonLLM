import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_add_scale_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, input_size, hidden_size,
    scale_factor,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    # Assume BIAS is a boolean indicating whether to add a bias (not applicable here)
    # ACTIVATION is a boolean indicating whether to apply activation (not applicable here)
    # But we will inline the operations: matmul + scale + add + clamp + logsumexp + mish

    # 2D Block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offset for the block
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Create offset ranges
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Masks for valid elements
    mask_m = offs_m < batch_size
    mask_n = offs_n < hidden_size
    mask = mask_m[:, None] & mask_n[None, :]

    # Accumulator for matmul
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension (input_size)
    for k in range(0, (input_size + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K):
        # Load X: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        offs_k_start = k * BLOCK_SIZE_K
        x_ptrs = x_ptr + offs_m[:, None] * input_size + offs_k_start + offs_k[None, :]
        x = tl.load(x_ptrs, mask=(offs_k_start + offs_k[None, :]) < input_size, other=0.0)

        # Load W: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        w_ptrs = w_ptr + offs_k_start[:, None] * hidden_size + offs_n[None, :]
        w = tl.load(w_ptrs, mask=(offs_k_start + offs_k[:, None]) < input_size, other=0.0)

        # Perform matrix multiplication: accumulator += x @ w
        accumulator += tl.dot(x, w)

    # Scale and add residual (x * scale_factor + x)
    accumulator *= scale_factor
    accumulator += accumulator  # equivalent to *= 2

    # Clamp
    accumulator = tl.clip(accumulator, -10.0, 10.0)

    # LogSumExp along dimension 1 (apply per row)
    # Use shared memory to reduce global memory accesses during reduction
    # We'll use a block-level reduction to compute logsumexp per row
    # Since we're processing batch_size, each row is a sequence of elements along hidden_size
    # We reduce along n dimension (dim=1)

    # We do a reduction over BLOCK_SIZE_N (n dimension) in a block, using shared memory
    # But we want to compute logsumexp per batch (each row), so we use a single reduction per row
    # We do it per block and combine via reduction

    # For simplicity and performance, we do this reduction inside the kernel per row
    # We use a single thread per row for reduction
    # We'll reduce along N axis, then store result

    # Start with first element of the block
    # But we need to do logsumexp over the full hidden_size for each row

    # We use shared memory to store partial reductions
    # Use a shared memory buffer to hold partial results for logsumexp
    # We only need one shared memory buffer per block

    # Let's do a reduction of accumulator over n dimension
    # We'll do a warp-reduction style loop
    # But we have to be careful: only one thread per row, but we're in a 2D grid

    # Instead, we perform a reduction for each row in the block
    # We will reduce across n dimension (hidden_size) and store per-row result

    # We assume BLOCK_SIZE_N >= hidden_size for simplicity (but we can't)
    # So we reduce over N in multiple steps

    # Let's restructure: first compute max over n dimension
    # Then compute logsumexp using max reduction and normalization

    # Shared memory for reduction
    shmem = tl.load(tl.static_range(0, 16384, 16384), tl.float32)  # Not directly usable, so we simulate

    # We need to recompute the reduction on the accumulator

    # Instead of full reduction here, we'll use a separate reduction kernel or inline it

    # Better: we split logsumexp into two parts: reduce_max and reduce_exp
    # But we can do it in one pass with shared memory

    # Let's do: max of accumulator per row (in the block), then subtract and sum exp
    # We do this using a loop over n

    # Since this is not trivial, we'll use a single reduction per row using warp-level ops

    # We use a loop to reduce along hidden_size dimension
    # We'll do this by gathering values from all blocks

    # But we can't do that in one kernel unless we reduce over all hidden_size

    # Alternative: compute logsumexp in a separate kernel, but we want fusion

    # Since logsumexp is a reduction over hidden_size, we can do it per block of rows
    # We'll do a reduction over n dimension for each m in the block
    # Using shared memory

    # Let's do a reduction for each row in the block

    # Initialize shared memory for reduction
    # We'll store the max and sum for each row in the block
    # We assume we have at most BLOCK_SIZE_M rows in the block

    # Shared memory for max and sum
    shmem_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    shmem_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # Loop over n dimension with blocking
    for k in range(0, (hidden_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N):
        start_n = k * BLOCK_SIZE_N
        end_n = start_n + BLOCK_SIZE_N
        mask_k = (start_n + offs_n) < hidden_size

        # Load accumulator for this block of n
        # Only for rows in the block and within range
        acc_ptrs = accumulator + offs_m[:, None] * hidden_size + start_n + offs_n[None, :]
        acc_block = tl.load(acc_ptrs, mask=mask_k & mask, other=-float('inf'))

        # Compute max
        current_max = tl.max(acc_block, axis=1)
        # Update global max
        shmem_max = tl.maximum(shmem_max, current_max)

        # Compute exp (shifted by max to avoid overflow)
        # Compute exp(x - max) for each row
        exp_val = tl.exp(acc_block - current_max[:, None])
        # Sum
        shmem_sum += tl.sum(exp_val, axis=1)

    # After reduction, global max and sum are in shmem_max and shmem_sum
    # But note: we're in a multi-block scenario, so we need to reduce across blocks

    # We cannot do that in this kernel — we need a two-pass approach

    # Instead, let's simplify: we’ll assume the entire hidden_size fits in shared memory
    # Or we do the logsumexp in a separate kernel? But we want fusion.

    # Alternatively, we can compute logsumexp for the entire row in a single kernel without sharing

    # We'll do: for each row, reduce over all hidden_size elements using a single thread per row
    # But we have to loop over n with a reduction

    # We'll do a warp-level reduction for each row in the block
    # Since we can't use shared memory easily, let's reframe

    # Actually, let's do logsumexp outside in a separate fused kernel

    # We need to restructure

    # Let's do: matmul + scale + add + clamp → store in out_ptr
    # Then call logsumexp in a separate kernel, then mish

    # But we want full fusion. So we do a three-kernel solution

    # Given complexity, we'll do:

    # 1. Matmul + scale + add + clamp: in this kernel
    # 2. Logsumexp: in a separate kernel
    # 3. Mish: in another kernel

    # But we can merge 2 and 3

    # Let’s refocus: do matmul + scale + add + clamp in this kernel

    # Save the result to out_ptr
    # Then we'll do logsumexp + mish in a subsequent kernel

    # So we write: only matmul, scale, add, clamp

    # We'll return the clamped output

    # But we want to avoid multiple kernels if possible

    # Given time, let's do matmul + scale + add + clamp in one kernel

    # Store the result
    out_ptrs = out_ptr + offs_m[:, None] * hidden_size + offs_n[None, :]
    tl.store(out_ptrs, accumulator, mask=mask)


@triton.jit
def logsumexp_mish_kernel(
    x_ptr, out_ptr,
    batch_size, hidden_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Compute logsumexp along dim=1 (hidden_size), then apply mish
    # Each program handles a block of batch_size elements

    pid_m = tl.program_id(0)
    block_start_m = pid_m * BLOCK_SIZE_M
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < batch_size

    # For each row, reduce along hidden_size dimension
    # Use shared memory for reduction
    shmem = tl.load(tl.static_range(0, 16384, 16384), tl.float32)  # dummy

    # We'll use shared memory to hold the reduction result for each row
    # But we need to do reduction over n

    # Initialize shared memory for reduction
    # We'll use a single value per row for max and sum
    # We'll do a loop over n blocks

    # We need a reduction across hidden_size for each row
    # Let's do in blocks of BLOCK_SIZE_N

    # Use shared memory for per-row max and sum
    shmem_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    shmem_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # Loop over n dimension
    for k in range(0, (hidden_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N):
        start_n = k * BLOCK_SIZE_N
        end_n = start_n + BLOCK_SIZE_N
        mask_k = (start_n + tl.arange(0, BLOCK_SIZE_N)) < hidden_size

        # Load x: (BLOCK_SIZE_M, BLOCK_SIZE_N)
        offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        x_ptrs = x_ptr + offs_m[:, None] * hidden_size + offs_n[None, :]
        x_block = tl.load(x_ptrs, mask=mask_k[None, :] & mask_m[:, None], other=-float('inf'))

        # Reduce: compute max
        current_max = tl.max(x_block, axis=1)
        shmem_max = tl.maximum(shmem_max, current_max)

        # Compute exp(x - max)
        exp_val = tl.exp(x_block - current_max[:, None])
        shmem_sum += tl.sum(exp_val, axis=1)

    # After reduction, compute logsumexp: log(sum(exp)) = log(sum) + max - max (but max is shifted)
    # Actually: log(sum(exp(x - max))) + max = logsumexp(x)
    logsumexp = tl.log(shmem_sum) + shmem_max

    # Now apply Mish: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x))
    # But we have logsumexp(x) as input

    # Apply mish: f(x) = x * tanh(softplus(x))
    # Let y = logsumexp(x)
    # softplus(y) = log(1 + exp(y))
    # tanh(softplus(y)) = tanh(log(1 + exp(y)))

    # But we can compute:
    # Let s = softplus(y) = log(1 + exp(y))
    # Then tanh(s) = (exp(s) - exp(-s)) / (exp(s) + exp(-s)) = ( (1+exp(y)) - 1/(1+exp(y)) ) / ( (1+exp(y)) + 1/(1+exp(y)) )
    # Instead, we use approximation: tanh(log(1 + exp(y))) = (exp(y) - 1) / (exp(y) + 1) ??? Not quite

    # Actually:
    # tanh(log(1 + exp(y))) = ( (1+exp(y)) - 1/(1+exp(y)) ) / ( (1+exp(y)) + 1/(1+exp(y)) )
    # Let z = exp(y), then:
    # = ( (1+z) - 1/(1+z) ) / ( (1+z) + 1/(1+z) ) = ( (1+z)^2 - 1 ) / ( (1+z)^2 + 1 ) = (1 + 2z + z^2 - 1) / (1 + 2z + z^2 + 1) = (2z + z^2) / (2 + 2z + z^2)

    # But it's messy. Instead, use: tanh(log(1+exp(y))) = (exp(y) - 1) / (exp(y) + 1) ? Let's check:
    # Let y = 0: tanh(log(2)) ≈ tanh(0.693) ≈ 0.6, (1-1)/(1+1) = 0 — no.

    # Better: use direct computation with safe exponentials

    # Instead, we use the standard formula:
    # mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))

    # We can compute it using:
    # softplus = tl.log1p(tl.exp(y))  # safe log(1+exp)
    # Then tanh = tl.tanh(softplus)

    # But we can't use tl.tanh on a value that might be large — but logsumexp is between -10 and 10 (due to clamp), so exp(y) is in [exp(-10), exp(10)] ~ [4.5e-5, 22026], so log(1+exp(y)) is safe

    # But we are in Triton, so we can use:
    # softplus = tl.log1p(tl.exp(y))

    # But there's a known stable version: for y > 0: log(1+exp(y)) = y + log(1+exp(-y)), for y<0: log(1+exp(y))

    # Triton has tl.log1p and tl.exp, so we use:

    # Compute softplus
    # We must be careful: if y is very large, exp(y) overflows
    # But we clamped to [-10,10], so exp(y) <= exp(10) ≈ 22026, which is < 1e5, so it's safe

    # Compute softplus
    # Use: softplus = tl.where(y > 0, y + tl.log1p(tl.exp(-y)), tl.log1p(tl.exp(y)))

    # But for simplicity, since y is in [-10,10], we can use tl.log1p(tl.exp(y))

    softplus = tl.log1p(tl.exp(logsumexp))

    # Compute tanh
    tanh_val = tl.tanh(softplus)

    # Apply mish
    mish = logsumexp * tanh_val

    # Store output: (batch_size, 1)
    out_ptrs = out_ptr + offs_m[:, None] * 1
    tl.store(out_ptrs, mish, mask=mask_m)


@triton.jit
def matmul_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, input_size, hidden_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Perform matrix multiplication: out = x @ w
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    mask_m = offs_m < batch_size
    mask_n = offs_n < hidden_size
    mask = mask_m[:, None] & mask_n[None, :]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, (input_size + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K):
        offs_k_start = k * BLOCK_SIZE_K
        x_ptrs = x_ptr + offs_m[:, None] * input_size + offs_k_start + offs_k[None, :]
        w_ptrs = w_ptr + offs_k_start[:, None] * hidden_size + offs_n[None, :]

        x = tl.load(x_ptrs, mask=(offs_k_start + offs_k[None, :]) < input_size, other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k_start + offs_k[:, None]) < input_size, other=0.0)

        accumulator += tl.dot(x, w)

    out_ptrs = out_ptr + offs_m[:, None] * hidden_size + offs_n[None, :]
    tl.store(out_ptrs, accumulator, mask=mask)


@triton.jit
def scale_add_kernel(
    x_ptr, out_ptr,
    batch_size, hidden_size,
    scale_factor,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # x = x * scale_factor + x
    # equivalent to x * (scale_factor + 1)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < batch_size
    mask_n = offs_n < hidden_size
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = x_ptr + offs_m[:, None] * hidden_size + offs_n[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # x = x * scale_factor + x
    out = x * (scale_factor + 1.0)

    out_ptrs = out_ptr + offs_m[:, None] * hidden_size + offs_n[None, :]
    tl.store(out_ptrs, out, mask=mask)


@triton.jit
def clamp_kernel(
    x_ptr, out_ptr,
    batch_size, hidden_size,
    clamp_min, clamp_max,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Clamp x to [clamp_min, clamp_max]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < batch_size
    mask_n = offs_n < hidden_size
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = x_ptr + offs_m[:, None] * hidden_size + offs_n[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    out = tl.clip(x, clamp_min, clamp_max)

    out_ptrs = out_ptr + offs_m[:, None] * hidden_size + offs_n[None, :]
    tl.store(out_ptrs, out, mask=mask)


@triton.jit
def logsumexp_kernel(
    x_ptr, out_ptr,
    batch_size, hidden_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Compute logsumexp along dim=1
    pid_m = tl.program_id(0)
    block_start_m = pid_m * BLOCK_SIZE_M
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < batch_size

    # Shared memory for max and sum reduction
    shmem_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    shmem_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # Reduce over hidden_size dimension
    for k in range(0, (hidden_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N):
        start_n = k * BLOCK_SIZE_N
        end_n = start_n + BLOCK_SIZE_N
        mask_k = (start_n + tl.arange(0, BLOCK_SIZE_N)) < hidden_size

        x_ptrs = x_ptr + offs_m[:, None] * hidden_size + (start_n + tl.arange(0, BLOCK_SIZE_N))[None, :]
        x_block = tl.load(x_ptrs, mask=mask_k[None, :] & mask_m[:, None], other=-float('inf'))

        current_max = tl.max(x_block, axis=1)
        shmem_max = tl.maximum(shmem_max, current_max)

        exp_val = tl.exp(x_block - current_max[:, None])
        shmem_sum += tl.sum(exp_val, axis=1)

    # Compute logsumexp
    logsumexp = tl.log(shmem_sum) + shmem_max

    # Store result: (batch_size, 1)
    out_ptrs = out_ptr + offs_m[:, None] * 1
    tl.store(out_ptrs, logsumexp, mask=mask_m)


@triton.jit
def mish_kernel(
    x_ptr, out_ptr,
    batch_size,
    BLOCK_SIZE: tl.constexpr
):
    # mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < batch_size

    x_ptrs = x_ptr + offs
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Compute softplus(x) = log(1 + exp(x))
    softplus = tl.log1p(tl.exp(x))

    # Compute tanh(softplus)
    tanh_val = tl.tanh(softplus)

    # Apply mish
    out = x * tanh_val

    out_ptrs = out_ptr + offs
    tl.store(out_ptrs, out, mask=mask)


def triton_matmul(x: torch.Tensor, w: torch.Tensor, block_size_m: int = 128, block_size_n: int = 128, block_size_k: int = 64):
    # Ensure contiguous on GPU
    x = x.contiguous()
    w = w.contiguous()
    batch_size, input_size = x.shape
    hidden_size = w.shape[1]

    out = torch.empty(batch_size, hidden_size, dtype=x.dtype, device=x.device)

    # Grid for matmul
    grid = lambda meta: (triton.cdiv(batch_size, meta["BLOCK_SIZE_M"]), triton.cdiv(hidden_size, meta["BLOCK_SIZE_N"]))

    matmul_kernel[grid](x, w, out, batch_size, input_size, hidden_size, BLOCK_SIZE_M=block_size_m, BLOCK_SIZE_N=block_size_n, BLOCK_SIZE_K=block_size_k)
    return out


def triton_scale_add(x: torch.Tensor, scale_factor: float, block_size_m: int = 128, block_size_n: int = 128):
    x = x.contiguous()
    batch_size, hidden_size = x.shape
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(batch_size, meta["BLOCK_SIZE_M"]), triton.cdiv(hidden_size, meta["BLOCK_SIZE_N"]))

    scale_add_kernel[grid](x, out, batch_size, hidden_size, scale_factor, BLOCK_SIZE_M=block_size_m, BLOCK_SIZE_N=block_size_n)
    return out


def triton_clamp(x: torch.Tensor, clamp_min: float, clamp_max: float, block_size_m: int = 128, block_size_n: int = 128):
    x = x.contiguous()
    batch_size, hidden_size = x.shape
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(batch_size, meta["BLOCK_SIZE_M"]), triton.cdiv(hidden_size, meta["BLOCK_SIZE_N"]))

    clamp_kernel[grid](x, out, batch_size, hidden_size, clamp_min, clamp_max, BLOCK_SIZE_M=block_size_m, BLOCK_SIZE_N=block_size_n)
    return out


def triton_logsumexp(x: torch.Tensor, block_size_m: int = 128, block_size_n: int = 128):
    x = x.contiguous()
    batch_size, hidden_size = x.shape
    out = torch.empty(batch_size, 1, dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(batch_size, meta["BLOCK_SIZE_M"]),)

    logsumexp_kernel[grid](x, out, batch_size, hidden_size, BLOCK_SIZE_M=block_size_m, BLOCK_SIZE_N=block_size_n)
    return out


def triton_mish(x: torch.Tensor, block_size: int = 128):
    x = x.contiguous()
    batch_size = x.shape[0]
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(batch_size, meta["BLOCK_SIZE"]),)

    mish_kernel[grid](x, out, batch_size, BLOCK_SIZE=block_size)
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size, bias=False)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Use A100-optimized block sizes
        self.block_size_m = 128
        self.block_size_n = 128
        self.block_size_k = 64
        self.block_size = 128

    def forward(self, x):
        # Matmul: x @ W
        out = triton_matmul(x, self.linear.weight, block_size_m=self.block_size_m, block_size_n=self.block_size_n, block_size_k=self.block_size_k)

        # Scale and add: out = out * scale_factor + out
        out = triton_scale_add(out, self.scale_factor, block_size_m=self.block_size_m, block_size_n=self.block_size_n)

        # Clamp
        out = triton_clamp(out, self.clamp_min, self.clamp_max, block_size_m=self.block_size_m, block_size_n=self.block_size_n)

        # LogSumExp
        out = triton_logsumexp(out, block_size_m=self.block_size_m, block_size_n=self.block_size_n)

        # Mish
        out = triton_mish(out, block_size=self.block_size)

        return out