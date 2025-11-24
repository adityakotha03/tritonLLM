import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
# Triton kernel for fused Group Normalization (and optional ReLU)
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 512}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def group_norm_fused_kernel(
    X_ptr,          # input (N, C, D, H, W) flattened to (N*C, D*H*W)
    Y_ptr,          # output
    G_ptr,          # group indices (C,)
    mean_ptr,       # per group mean (N, G)
    var_ptr,        # per group variance (N, G)
    gamma_ptr,      # scale per channel (C,)
    beta_ptr,       # bias per channel (C,)
    N: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    G: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    do_relu: tl.constexpr,
):
    """
    X_ptr : (N*C, D*H*W)
    Y_ptr : (N*C, D*H*W)
    G_ptr : (C,) group assignment for each channel
    mean_ptr : (N, G)
    var_ptr : (N, G)
    gamma_ptr, beta_ptr : (C,)
    """
    # compute M = N*C, N = D*H*W
    M = N * C
    # each program handles a block of size BLOCK_SIZE_M x BLOCK_SIZE_N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < M
    mask_n = offs_n < (D * H * W)

    # load data
    x = tl.load(X_ptr + offs_m[:, None] * (D * H * W) + offs_n[None, :], mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    # compute channel index and spatial idx
    c = offs_m // (D * H * W)  # channel index
    g = tl.load(G_ptr + c)     # group index

    # load mean and var for this batch and group
    n_batch = offs_m // C  # batch index
    mean = tl.load(mean_ptr + n_batch * G + g)
    var  = tl.load(var_ptr  + n_batch * G + g)

    gamma = tl.load(gamma_ptr + c)
    beta  = tl.load(beta_ptr  + c)

    # normalize
    y = (x - mean) * tl.rsqrt(var + eps) * gamma + beta

    # optional ReLU
    if do_relu:
        y = tl.max(y, 0.0)

    # store result
    tl.store(Y_ptr + offs_m[:, None] * (D * H * W) + offs_n[None, :], y, mask=mask_m[:, None] & mask_n[None, :])

# --------------------------------------------------------------------------- #
# Helper to compute mean/var per group
# --------------------------------------------------------------------------- #
def compute_group_stats(x, G, eps=1e-5):
    """
    x : (N, C, D, H, W)
    G : tensor of group assignments per channel, shape (C,)
    returns mean, var of shape (N, num_groups)
    """
    N, C, D, H, W = x.shape
    G_num = G.max().item() + 1
    x_flat = x.reshape(N * C, -1)
    # sum per channel
    sum_c = torch.sum(x_flat, dim=1)  # (N*C,)
    sqsum_c = torch.sum(x_flat * x_flat, dim=1)  # (N*C,)
    # map to groups
    mean = torch.zeros(N, G_num, device=x.device)
    var  = torch.zeros(N, G_num, device=x.device)
    for g in range(G_num):
        idx = (G == g).nonzero(as_tuple=False).squeeze()
        if idx.numel() == 0:
            continue
        # select channels belonging to group g
        idx_exp = idx.unsqueeze(0).expand(N, -1).reshape(-1)
        cnt = idx_exp.shape[0] * D * H * W
        mean[:, g] = sum_c[idx_exp].reshape(N, -1).sum(dim=1) / cnt
        var[:, g]  = (sqsum_c[idx_exp].reshape(N, -1).sum(dim=1) / cnt) - mean[:, g]**2
    return mean, var

# --------------------------------------------------------------------------- #
# Optimized Model
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Model that performs a transposed 3D convolution, applies ReLU,
    and then applies group normalization with a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False, use_relu_in_fusion=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.use_relu_in_fusion = use_relu_in_fusion

        # precompute group assignment tensor
        self.register_buffer('group_ids', torch.arange(out_channels) // (out_channels // groups))

    def forward(self, x):
        # ConvTranspose3d + ReLU (PyTorch)
        x = self.conv_transpose(x)
        x = self.relu(x)

        # Prepare data for Triton
        N, C, D, H, W = x.shape
        x_flat = x.reshape(N * C, D * H * W).contiguous()

        # Compute mean and var per group
        mean, var = compute_group_stats(x, self.group_ids, eps=1e-5)

        # Prepare output tensor
        y = torch.empty_like(x_flat)

        # Triton kernel launch
        grid = lambda meta: (
            ( (N * C + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M'],
              (D * H * W + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N'] ),
        )
        group_norm_fused_kernel[grid](
            x_flat, y,
            self.group_ids,
            mean,
            var,
            self.group_norm.weight,
            self.group_norm.bias,
            N, C, D, H, W,
            self.group_ids.max().item() + 1,
            1e-5,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=128,
            do_relu=self.use_relu_in_fusion,
        )

        # Reshape back to original shape
        y = y.reshape(N, C, D, H, W)
        return y