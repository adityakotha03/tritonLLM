import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
#            Triton kernels for Group Normalization (GroupNorm)             #
# --------------------------------------------------------------------------- #

@triton.jit
def gn_sum_kernel(
    x_ptr: tl.tensor,           # (B*C*H*W,) float32
    sum_ptr: tl.tensor,         # (B*G,) float32
    sumsq_ptr: tl.tensor,       # (B*G,) float32
    batch_size: tl.int32,
    channels: tl.int32,
    height: tl.int32,
    width: tl.int32,
    group_size: tl.int32,
    num_groups: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """First pass: accumulate sum and sum of squares per (batch, group)."""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * channels * height * width

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute multi‑dimensional indices from linear index
    batch_stride = channels * height * width
    idx = offsets
    batch = idx // batch_stride
    rem = idx % batch_stride
    channel_stride = height * width
    channel = rem // channel_stride

    # Determine group for this channel
    group = channel // group_size

    # Linear index into per‑batch‑group arrays
    linear_idx = batch * num_groups + group

    # Atomic adds to global memory
    tl.atomic_add(sum_ptr + linear_idx, x, mask=mask)
    tl.atomic_add(sumsq_ptr + linear_idx, x * x, mask=mask)


@triton.jit
def gn_mean_var_kernel(
    sum_ptr: tl.tensor,     # (B*G,) float32
    sumsq_ptr: tl.tensor,   # (B*G,) float32
    mean_ptr: tl.tensor,    # (B*G,) float32
    var_ptr: tl.tensor,     # (B*G,) float32
    batch_size: tl.int32,
    num_groups: tl.int32,
    group_size: tl.int32,
    height: tl.int32,
    width: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """Second pass: compute mean and variance per (batch, group)."""
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < batch_size * num_groups

    s = tl.load(sum_ptr + idx, mask=mask, other=0.0)
    ss = tl.load(sumsq_ptr + idx, mask=mask, other=0.0)

    count = group_size * height * width
    mean = s / count
    var = ss / count - mean * mean

    tl.store(mean_ptr + idx, mean, mask=mask)
    tl.store(var_ptr + idx, var, mask=mask)


@triton.jit
def gn_norm_kernel(
    x_ptr: tl.tensor,        # (B*C*H*W,) float32
    out_ptr: tl.tensor,      # (B*C*H*W,) float32
    gamma_ptr: tl.tensor,    # (C,) float32
    beta_ptr: tl.tensor,     # (C,) float32
    mean_ptr: tl.tensor,     # (B*G,) float32
    var_ptr: tl.tensor,      # (B*G,) float32
    batch_size: tl.int32,
    channels: tl.int32,
    height: tl.int32,
    width: tl.int32,
    group_size: tl.int32,
    num_groups: tl.int32,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    """Final pass: normalize and apply scale (gamma) and bias (beta)."""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * channels * height * width

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute indices as in gn_sum_kernel
    batch_stride = channels * height * width
    idx = offsets
    batch = idx // batch_stride
    rem = idx % batch_stride
    channel_stride = height * width
    channel = rem // channel_stride

    group = channel // group_size
    gv_idx = batch * num_groups + group

    mean = tl.load(mean_ptr + gv_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + gv_idx, mask=mask, other=0.0)
    inv_std = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(gamma_ptr + channel, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + channel, mask=mask, other=0.0)

    out = gamma * (x - mean) * inv_std + beta
    tl.store(out_ptr + offsets, out, mask=mask)


# --------------------------------------------------------------------------- #
#                            Custom GroupNorm Module                           #
# --------------------------------------------------------------------------- #

class ModelNew(nn.Module):
    """
    Group Normalization implemented with custom Triton kernels.
    """
    def __init__(self, num_features: int, num_groups: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps

        # Learnable scale (gamma) and bias (beta)
        self.weight = nn.Parameter(torch.ones(num_features, device="cuda"))
        self.bias = nn.Parameter(torch.zeros(num_features, device="cuda"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Group Normalization to input tensor x using Triton kernels.
        Args:
            x: Tensor of shape (B, C, H, W) on CUDA.
        Returns:
            Normalized tensor of the same shape.
        """
        assert x.is_cuda, "Input must be on CUDA."
        B, C, H, W = x.shape
        assert C % self.num_groups == 0, "num_features must be divisible by num_groups."
        group_size = C // self.num_groups

        # Ensure contiguous and float32 for kernel ops
        x_contig = x.contiguous()
        dtype_orig = x_contig.dtype
        x_fp32 = x_contig.to(torch.float32)

        # Allocate buffers for sum and sumsq
        sum_buf = torch.empty(B * self.num_groups, dtype=torch.float32, device=x_fp32.device)
        sumsq_buf = torch.empty_like(sum_buf)

        total_elements = B * C * H * W
        BLOCK_SIZE = 256  # tune as needed

        # 1st pass: compute per‑group sums
        grid1 = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        gn_sum_kernel[grid1](
            x_fp32, sum_buf, sumsq_buf,
            B, C, H, W, group_size, self.num_groups,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # 2nd pass: compute mean and variance
        mean_buf = torch.empty_like(sum_buf)
        var_buf = torch.empty_like(sum_buf)
        grid2 = lambda meta: ((B * self.num_groups + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        gn_mean_var_kernel[grid2](
            sum_buf, sumsq_buf, mean_buf, var_buf,
            B, self.num_groups, group_size, H, W,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # 3rd pass: normalize, scale, and shift
        out_fp32 = torch.empty_like(x_fp32)
        grid3 = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        gn_norm_kernel[grid3](
            x_fp32, out_fp32,
            self.weight.to(torch.float32), self.bias.to(torch.float32),
            mean_buf, var_buf,
            B, C, H, W, group_size, self.num_groups,
            eps=self.eps, BLOCK_SIZE=BLOCK_SIZE
        )

        # Cast back to original dtype if necessary
        if dtype_orig != torch.float32:
            out_fp32 = out_fp32.to(dtype_orig)
        return out_fp32