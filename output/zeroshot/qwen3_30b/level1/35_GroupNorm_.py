import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(
    x_ptr,
    mean_ptr,
    rstd_ptr,
    out_ptr,
    N,  # total number of elements per group
    C,  # number of channels (features)
    G,  # number of groups
    H,  # height
    W,  # width
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one block of elements along the channel dimension
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    block_start = pid * block_size

    # Calculate offset in the flattened tensor for this block
    # We process the data in chunks of `block_size` along the channel dimension
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < C

    # Number of elements per group (C // G)
    C_per_group = C // G

    # Determine which group this channel belongs to
    group_id = offset // C_per_group
    # Calculate the local channel index within the group
    local_ch = offset % C_per_group

    # Total elements in the spatial dimensions
    spatial_elements = H * W

    # Offset for current group's data in the tensor
    group_offset = group_id * C_per_group * spatial_elements

    # Each thread block handles one channel group, but we need to reduce across spatial dims
    # Load one spatial slice at a time (H x W), and accumulate mean/variance
    # Use shared memory to reduce memory traffic
    # We'll do reduction over H*W using a block-level reduction in shared memory

    # Shared memory for partial sum and sum of squares
    shared_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    shared_sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over spatial dimensions
    for i in range(0, spatial_elements, BLOCK_SIZE):
        # Calculate offset for spatial chunk
        spatial_offset = i + tl.arange(0, BLOCK_SIZE)
        spatial_mask = spatial_offset < spatial_elements

        # Load data: x[batch, ch, h, w] -> we are accessing all batches, same group, same ch
        # Shape: (batch, spatial_elements)
        # We need to extract the entire channel group across all spatial positions
        # But we're iterating per channel, so we must map correctly

        # For current channel group, get the data across all spatial positions
        # Offset for current group and channel
        base_offset = group_offset + (offset // C_per_group) * spatial_elements
        # Now, for this thread, get data at spatial offset
        # x_ptr[batch, ch, h, w] -> we need to traverse h, w
        # So: batch * C * H * W + ch * H * W + h * W + w
        # We will load chunk per spatial index
        x_ptr_offset = base_offset + spatial_offset
        x_data = tl.load(x_ptr + x_ptr_offset, mask=spatial_mask & mask, other=0.0)

        # Accumulate sum and sum of squares
        shared_sum += x_data
        shared_sum_sq += x_data * x_data

    # Reduce across spatial dims using shared memory (all threads in block)
    # Perform reduction in shared memory
    sum_val = tl.sum(shared_sum, axis=0)
    sum_sq_val = tl.sum(shared_sum_sq, axis=0)

    # Now, we do global reduction across all blocks via all-reduce
    # We use reduction with grid-level coordination: each block computes partial sum
    # But we must combine across all blocks in the group
    # Since we cannot do global reduction easily, we will instead use a single block per group
    # So: num_groups * (C // G) blocks, but we launch one block per channel group, not per channel

    # Wait — better idea: launch one block per group, not per channel. But we can't do that easily
    # Instead, we can launch one block per group, and handle all channels in the group in parallel

    # Let's reframe: we launch one block per group, and each thread handles one channel in the group
    # So we change our approach: we now launch `G` blocks (one per group), each block processes `C_per_group` channels
    # And we change the kernel to do per-group processing

    # So we need to restructure: block_id = group_id, and block_size = C_per_group

    # But we already used program_id as channel index. So we must change the kernel

    # Revised plan: we launch `G` blocks (each block handles one group), and each block has `C_per_group` threads
    # So we recompute the kernel to be per-group, and use BLOCK_SIZE = C_per_group

    # Let's rewrite the kernel accordingly
    # We'll change the function signature and logic

    # But since we're stuck in the middle, let's just rework it cleanly

    # Let's restart with correct design: one block per group, each block has C_per_group threads
    # So program_id(0) is the group index (0 to G-1)
    # Then, each thread handles one channel within the group

    # We'll define a new kernel
    pass


# Reimplementing correct group norm kernel: one block per group
@triton.jit
def group_norm_kernel_corrected(
    x_ptr,
    mean_ptr,
    rstd_ptr,
    out_ptr,
    N,  # total number of elements (B * C * H * W)
    C,  # channels
    G,  # groups
    H,  # height
    W,  # width
    BLOCK_SIZE: tl.constexpr,
):
    # Each block handles one group
    group_id = tl.program_id(0)  # 0 to G-1
    C_per_group = C // G

    # Each thread in the block handles one channel within the group
    ch_id = tl.program_id(1)  # 0 to C_per_group - 1
    thread_id = ch_id

    # Offset in the channel dimension
    ch_offset = group_id * C_per_group + ch_id

    # Total spatial elements
    spatial_elements = H * W

    # Load all spatial data for this channel
    x_ptr_offset = ch_offset * spatial_elements
    x_data = tl.load(x_ptr + x_ptr_offset + tl.arange(0, spatial_elements), mask=tl.arange(0, spatial_elements) < spatial_elements, other=0.0)

    # Reduce over spatial dimensions to compute mean and variance
    # We compute mean first
    mean = tl.sum(x_data, axis=0) / spatial_elements

    # Compute variance
    var = tl.sum((x_data - mean) * (x_data - mean), axis=0) / spatial_elements

    # Compute rstd = 1 / sqrt(var + eps)
    eps = 1e-6
    rstd = 1.0 / tl.sqrt(var + eps)

    # Store mean and rstd for this channel
    mean_ptr_offset = group_id * C_per_group + ch_id
    rstd_ptr_offset = group_id * C_per_group + ch_id

    tl.store(mean_ptr + mean_ptr_offset, mean)
    tl.store(rstd_ptr + rstd_ptr_offset, rstd)

    # Normalize and store output
    out_data = (x_data - mean) * rstd
    tl.store(out_ptr + x_ptr_offset, out_data)


@triton.jit
def group_norm_kernel_fused(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,  # total number of elements
    C,  # channels
    G,  # groups
    H,  # height
    W,  # width
    BLOCK_SIZE: tl.constexpr,
):
    # One block per group
    group_id = tl.program_id(0)
    C_per_group = C // G

    # Each thread in the block handles one channel in the group
    ch_id = tl.program_id(1)
    thread_id = ch_id

    # Channel index
    ch_offset = group_id * C_per_group + ch_id

    # Spatial elements
    spatial_elements = H * W

    # Load data for this channel
    x_ptr_offset = ch_offset * spatial_elements
    x_data = tl.load(x_ptr + x_ptr_offset + tl.arange(0, spatial_elements), mask=tl.arange(0, spatial_elements) < spatial_elements, other=0.0)

    # Compute mean and variance
    mean = tl.sum(x_data, axis=0) / spatial_elements
    var = tl.sum((x_data - mean) * (x_data - mean), axis=0) / spatial_elements
    eps = 1e-6
    rstd = 1.0 / tl.sqrt(var + eps)

    # Store mean and rstd for normalization
    mean_ptr_offset = group_id * C_per_group + ch_id
    rstd_ptr_offset = group_id * C_per_group + ch_id
    tl.store(mean_ptr + mean_ptr_offset, mean)
    tl.store(rstd_ptr + rstd_ptr_offset, rstd)

    # Apply scale and bias if provided
    scale = tl.load(weight_ptr + ch_offset, mask=ch_id < C_per_group, other=1.0)
    bias = tl.load(bias_ptr + ch_offset, mask=ch_id < C_per_group, other=0.0)

    # Normalize and scale
    out_data = (x_data - mean) * rstd * scale + bias

    # Store output
    tl.store(out_ptr + x_ptr_offset, out_data)


def triton_group_norm(x, weight=None, bias=None, num_groups=8, eps=1e-6):
    B, C, H, W = x.shape
    G = num_groups
    C_per_group = C // G

    # Create output tensor
    out = torch.empty_like(x)

    # Allocate memory for mean and rstd
    mean = torch.zeros(G * C_per_group, device=x.device, dtype=x.dtype)
    rstd = torch.zeros(G * C_per_group, device=x.device, dtype=x.dtype)

    # Convert to contiguous
    x = x.contiguous()
    out = out.contiguous()

    # Ensure weight and bias are contiguous if provided
    if weight is not None:
        weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Define grid: one block per group, and C_per_group threads per block
    # So grid is (G, C_per_group)
    grid = lambda meta: (G, C_per_group)

    # Launch kernel
    group_norm_kernel_fused[grid](
        x_ptr=x,
        weight_ptr=weight if weight is not None else tl.load,
        bias_ptr=bias if bias is not None else tl.load,
        out_ptr=out,
        N=x.numel(),
        C=C,
        G=G,
        H=H,
        W=W,
        BLOCK_SIZE=32  # Arbitrary, but small enough for shared memory
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.C_per_group = num_features // num_groups

        # Learnable parameters: scale and bias
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply Triton-based group normalization
        return triton_group_norm(x, weight=self.weight, bias=self.bias, num_groups=self.num_groups)
