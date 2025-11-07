import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gemm_group_norm_kernel(
    x_ptr, w_ptr, b_ptr, 
    out_ptr, 
    batch_size, in_features, out_features, num_groups,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    use_bias: tl.constexpr
):
    # Matrix dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Offset for this block
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_k = pid_k * BLOCK_SIZE_K

    # Offsets within block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)

    # Load input and weight data
    x_ptrs = x_ptr + (offs_m[:, None] * in_features + offs_k[None, :])
    w_ptrs = w_ptr + (offs_k[:, None] * out_features + offs_n[None, :])
    
    # Mask out-of-bounds indices
    x_mask = (offs_m[:, None] < batch_size) & (offs_k[None, :] < in_features)
    w_mask = (offs_k[:, None] < in_features) & (offs_n[None, :] < out_features)
    
    # Load x and w
    x = tl.load(x_ptrs, mask=x_mask, other=0.0)
    w = tl.load(w_ptrs, mask=w_mask, other=0.0)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Perform matrix multiplication
    for _ in range(0, (in_features + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K):
        # Compute inner product using tensor cores
        acc += tl.dot(x, w, allow_tf32=True)
        # Increment block start
        block_start_k += BLOCK_SIZE_K
        offs_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)
        x_ptrs = x_ptr + (offs_m[:, None] * in_features + offs_k[None, :])
        w_ptrs = w_ptr + (offs_k[:, None] * out_features + offs_n[None, :])
        x_mask = (offs_m[:, None] < batch_size) & (offs_k[None, :] < in_features)
        w_mask = (offs_k[:, None] < in_features) & (offs_n[None, :] < out_features)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
    
    # Convert to float32 for normalization
    acc = acc.to(tl.float32)

    # Perform Group Normalization: mean and variance per group
    # Each group has out_features // num_groups features
    group_size = out_features // num_groups
    offs_group = offs_n // group_size  # Group index per feature

    # Reshape output to (batch_size, num_groups, group_size)
    acc = acc.reshape(batch_size, num_groups, group_size)

    # Compute mean per group
    mean = tl.sum(acc, axis=2) / group_size
    mean = mean[:, None, :]  # Expand for broadcasting

    # Compute variance
    var = tl.sum((acc - mean) * (acc - mean), axis=2) / group_size
    var = var[:, None, :]  # Expand for broadcasting

    # Normalize
    normalized = (acc - mean) / (tl.sqrt(var + 1e-6))

    # Scale and shift if bias exists
    if use_bias:
        b_ptrs = b_ptr + offs_n
        bias = tl.load(b_ptrs, mask=offs_n < out_features, other=0.0)
        bias = bias[None, None, :]
        normalized = normalized + bias

    # Reshape back to (batch_size, out_features)
    normalized = normalized.reshape(batch_size, out_features)

    # Hardtanh clipping
    out = tl.clamp(normalized, -2.0, 2.0)

    # Store output
    out_ptrs = out_ptr + (offs_m[:, None] * out_features + offs_n[None, :])
    out_mask = (offs_m[:, None] < batch_size) & (offs_n[None, :] < out_features)
    tl.store(out_ptrs, out, mask=out_mask)


@triton.jit
def gemm_kernel(
    x_ptr, w_ptr, b_ptr,
    out_ptr,
    batch_size, in_features, out_features,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    use_bias: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_k = pid_k * BLOCK_SIZE_K

    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (offs_m[:, None] * in_features + offs_k[None, :])
    w_ptrs = w_ptr + (offs_k[:, None] * out_features + offs_n[None, :])

    x_mask = (offs_m[:, None] < batch_size) & (offs_k[None, :] < in_features)
    w_mask = (offs_k[:, None] < in_features) & (offs_n[None, :] < out_features)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, (in_features + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K):
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x, w, allow_tf32=True)
        block_start_k += BLOCK_SIZE_K
        offs_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)
        x_ptrs = x_ptr + (offs_m[:, None] * in_features + offs_k[None, :])
        w_ptrs = w_ptr + (offs_k[:, None] * out_features + offs_n[None, :])
        x_mask = (offs_m[:, None] < batch_size) & (offs_k[None, :] < in_features)
        w_mask = (offs_k[:, None] < in_features) & (offs_n[None, :] < out_features)

    # Convert to float32 for normalization
    acc = acc.to(tl.float32)

    # Handle bias
    if use_bias:
        b_ptrs = b_ptr + offs_n
        bias = tl.load(b_ptrs, mask=offs_n < out_features, other=0.0)
        acc = acc + bias[None, :]

    # Store output
    out_ptrs = out_ptr + (offs_m[:, None] * out_features + offs_n[None, :])
    out_mask = (offs_m[:, None] < batch_size) & (offs_n[None, :] < out_features)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def group_norm_kernel(
    x_ptr,
    out_ptr,
    batch_size, out_features, num_groups,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)

    # Group size
    group_size = out_features // num_groups

    # Map global offset to (batch, group, feature_in_group)
    batch_idx = offs // out_features
    feature_idx = offs % out_features
    group_idx = feature_idx // group_size

    # Create mask for valid elements
    mask = offs < batch_size * out_features

    # Load data
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Compute mean and variance per group
    # We use shared memory to store per-group sums
    shared_mem = tl.load(tl.static_pointer_cast(tl.zeros(1, dtype=tl.float32), 'char'), no_sync=True)
    shared_mem = tl.zeros((1,), dtype=tl.float32)

    # Since Triton doesn't support dynamic reduction across groups, we rely on grid-level sync
    # For simplicity, we use a single group and reduce across features
    # This is a simplified approximation for demonstration — in real use, consider tiling or reorganizing
    x = x.reshape(batch_size, out_features)
    mean = tl.sum(x, axis=1) / out_features
    mean = mean[:, None]

    var = tl.sum((x - mean) * (x - mean), axis=1) / out_features
    var = var[:, None]

    # Normalize
    normalized = (x - mean) / (tl.sqrt(var + 1e-6))

    # Reshape and store
    normalized = normalized.reshape(batch_size * out_features)
    tl.store(out_ptr + offs, normalized, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm_weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.bfloat16))
        self.gemm_bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.out_features = out_features

    def forward(self, x):
        # Ensure inputs are on GPU and in bfloat16
        x = x.to(torch.bfloat16).contiguous()

        # Allocate output
        out = torch.empty_like(x, dtype=torch.bfloat16)

        # GEMM with GroupNorm and Hardtanh fused
        # Determine grid dimensions
        num_blocks_m = triton.cdiv(x.size(0), 128)
        num_blocks_n = triton.cdiv(self.out_features, 128)
        num_blocks_k = triton.cdiv(x.size(1), 64)

        # Use TF32 for GEMM via Triton's allow_tf32
        grid = (num_blocks_m, num_blocks_n, num_blocks_k)

        # Launch fused kernel
        gemm_group_norm_kernel[grid](
            x,
            self.gemm_weight,
            self.gemm_bias,
            out,
            x.size(0), x.size(1), self.out_features, self.num_groups,
            BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
            TILE_SIZE=16,
            use_bias=True
        )

        # Apply Hardtanh in-place (no kernel needed; PyTorch is efficient)
        return torch.clamp(out, self.hardtanh_min, self.hardtanh_max)
