import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------------------------------------------------
# Triton kernel: compute per‑channel mean and variance
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["N", "C", "D", "H", "W"],
)
@triton.jit
def batch_norm_stats_kernel(
    x_ptr,          # [N, C, D, H, W]
    mean_ptr,       # [C]
    var_ptr,        # [C]
    N: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Linear index of the current program
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Total number of elements per channel
    elements_per_channel = N * D * H * W

    # Compute channel index for each element
    channel_idx = idx // elements_per_channel
    # Mask to avoid out‑of‑bounds
    mask = channel_idx < C

    # Load elements, accumulate sums
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Stride across the whole tensor
    stride = C * D * H * W
    for i in range(0, N, BLOCK_SIZE):
        off = (i * C * D * H * W) + idx
        val = tl.load(x_ptr + off, mask=mask, other=0.0)
        sum_val += val
        sum_sq += val * val

    # Reduce within the block
    sum_val = tl.sum(sum_val, axis=0)
    sum_sq = tl.sum(sum_sq, axis=0)

    # Global reduction across blocks
    # Use Triton’s atomic operations
    tl.atomic_add(mean_ptr + channel_idx, sum_val)
    tl.atomic_add(var_ptr + channel_idx, sum_sq)


# ------------------------------------------------------------------
# Triton kernel: apply batch norm (after stats are ready)
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["N", "C", "D", "H", "W"],
)
@triton.jit
def batch_norm_apply_kernel(
    x_ptr,          # [N, C, D, H, W]
    y_ptr,          # output
    mean_ptr,       # [C]
    var_ptr,        # [C]
    gamma_ptr,      # [C]
    beta_ptr,       # [C]
    eps: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    elements_per_channel = N * D * H * W
    channel_idx = idx // elements_per_channel
    mask = channel_idx < C

    # Load data
    val = tl.load(x_ptr + idx, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + channel_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + channel_idx, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + channel_idx, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + channel_idx, mask=mask, other=0.0)

    # Normalize
    std = tl.sqrt(var / elements_per_channel + eps)
    norm = (val - mean) / std
    out = gamma * norm + beta

    tl.store(y_ptr + idx, out, mask=mask)


# ------------------------------------------------------------------
# Triton kernel: global average pooling over spatial dimensions
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["N", "C", "D", "H", "W"],
)
@triton.jit
def global_avg_pool_kernel(
    x_ptr,      # [N, C, D, H, W]
    y_ptr,      # [N, C, 1, 1, 1]
    N: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    elements = D * H * W
    channel_idx = idx // (N * elements)
    sample_idx = (idx // elements) % N
    mask = (channel_idx < C) & (sample_idx < N)

    # Sum over spatial dims
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, D * H * W, BLOCK_SIZE):
        off = (sample_idx * C * D * H * W) + (channel_idx * D * H * W) + i + idx
        val = tl.load(x_ptr + off, mask=mask, other=0.0)
        sum_val += val

    sum_val = tl.sum(sum_val, axis=0)
    avg = sum_val / elements

    out_off = sample_idx * C + channel_idx
    tl.store(y_ptr + out_off, avg, mask=mask)


# ------------------------------------------------------------------
# Wrapper functions
# ------------------------------------------------------------------
def triton_batch_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps=1e-5):
    N, C, D, H, W = x.shape
    mean = torch.zeros(C, device=x.device, dtype=torch.float32)
    var = torch.zeros(C, device=x.device, dtype=torch.float32)

    # Stats
    grid_stats = lambda meta: ((N * C * D * H * W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    batch_norm_stats_kernel[grid_stats](x, mean, var,
                                        N=N, C=C, D=D, H=H, W=W)

    # Apply
    out = torch.empty_like(x)
    grid_apply = lambda meta: ((N * C * D * H * W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    batch_norm_apply_kernel[grid_apply](x, out, mean, var, gamma, beta,
                                        eps=eps, N=N, C=C, D=D, H=H, W=W)
    return out


def triton_global_avg_pool(x: torch.Tensor):
    N, C, D, H, W = x.shape
    out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)

    grid = lambda meta: ((N * C + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    global_avg_pool_kernel[grid](x, out, N=N, C=C, D=D, H=H, W=W)
    return out


# ------------------------------------------------------------------
# New model using Triton kernels
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, scales the output,
    applies batch normalization, and then performs global average pooling.
    The batch normalization and pooling are implemented with custom Triton kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor,
                 eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

        # Learnable batch‑norm parameters
        self.gamma = nn.Parameter(torch.ones(out_channels, device='cuda'))
        self.beta = nn.Parameter(torch.zeros(out_channels, device='cuda'))

        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        # ConvTranspose3d
        x = self.conv_transpose(x)

        # Scale
        x = x * self.scale_factor

        # Batch norm via Triton
        x = triton_batch_norm(x, self.gamma, self.beta, eps=self.eps)

        # Global average pooling via Triton
        x = triton_global_avg_pool(x)

        return x