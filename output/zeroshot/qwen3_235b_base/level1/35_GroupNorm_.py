import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _group_norm_kernel(
    X_ptr,  # pointer to input data
    W_ptr,  # pointer to weight (gamma)
    B_ptr,  # pointer to bias (beta)
    Y_ptr,  # pointer to output
    mean_ptr,  # pointer to mean buffer (for backward)
    rstd_ptr,  # pointer to rstd buffer (for backward)
    N: tl.constexpr,  # number of elements per group
    num_channels: tl.constexpr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # program id
    pid = tl.program_id(0)
    group_id = pid // num_channels * group_size + pid % group_size
    block_start = pid * N

    # offsets within block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (pid + 1) * N

    # load data
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # compute mean
    mean = tl.sum(x, axis=0) / N
    tl.store(mean_ptr + pid, mean)

    # compute variance (via E[X^2] - E[X]^2)
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(rstd_ptr + pid, rstd)

    # normalize and apply affine transform
    x_norm = x_centered * rstd
    weight = tl.load(W_ptr + pid % group_size, mask=pid % group_size < num_channels, other=1.0)
    bias = tl.load(B_ptr + pid % group_size, mask=pid % group_size < num_channels, other=0.0)
    output = x_norm * weight + bias

    # store result
    tl.store(Y_ptr + offsets, output, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized version of GroupNorm using a custom Triton kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        super(ModelNew, self).__init__()
        assert num_features % num_groups == 0, "num_features must be divisible by num_groups"
        self.num_features = num_features
        self.num_groups = num_groups
        self.group_size = num_features // num_groups
        self.eps = 1e-5

        # Learnable parameters: weight (gamma) and bias (beta)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_features, d1, d2, ...)
        assert x.size(1) == self.num_features, "Input channels must match num_features"
        x = x.contiguous()
        batch_size = x.size(0)
        remaining_dims = x.shape[2:]
        num_remaining = 1
        for d in remaining_dims:
            num_remaining *= d
        total_elements = x.numel()
        N = num_remaining  # number of elements per group (per channel)

        # reshape to (batch_size, num_groups, group_size, ...)
        x_reshaped = x.view(batch_size, self.num_groups, self.group_size, *remaining_dims)
        x_reshaped = x_reshaped.transpose(0, 1).contiguous()  # (num_groups, batch_size, group_size, ...)
        x_reshaped = x_reshaped.view(self.num_groups * self.group_size, -1)  # (num_features, batch_size * num_remaining)
        x_reshaped = x_reshaped.t().contiguous()  # (total_elements_per_feature, num_features) -> transpose to (..., num_features)

        # flatten input for kernel processing: each feature group is processed independently
        X_flat = x_reshaped.view(-1)  # shape: (batch_size * num_features * num_remaining,)

        # expand weight and bias to match per-element size
        W = self.weight.data.repeat_interleave(N).contiguous()
        B = self.bias.data.repeat_interleave(N).contiguous()

        # output buffer
        Y_flat = torch.empty_like(X_flat)

        # buffers for mean and rstd (needed for backward, but we don't support backward here)
        mean_buf = torch.empty((self.num_groups * batch_size,), dtype=torch.float32, device=x.device)
        rstd_buf = torch.empty((self.num_groups * batch_size,), dtype=torch.float32, device=x.device)

        # launch kernel
        def grid(meta):
            return (x_reshaped.size(0),)

        # BLOCK_SIZE must divide N; we choose the largest power of 2 <= min(N, 4096)
        BLOCK_SIZE = min(4096, 1 << (N.bit_length() - 1))
        if BLOCK_SIZE == 0:
            BLOCK_SIZE = N  # fallback for very small N

        _group_norm_kernel[grid](
            X_flat,
            W,
            B,
            Y_flat,
            mean_buf,
            rstd_buf,
            N=N,
            num_channels=self.num_features,
            num_groups=self.num_groups,
            group_size=self.group_size,
            eps=self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # reshape back to original shape
        y_reshaped = Y_flat.view(batch_size, self.num_features, *remaining_dims)
        return y_reshaped