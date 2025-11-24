import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)  # (B, C, D, H, W)
        x = self.custom_hardswish(x)  # Nonlinear activation
        x = self.custom_group_norm(x)  # Normalization over channels
        x = self.custom_mean_pool(x)  # Mean over spatial dims → (B, C)
        return x

    @triton.jit
    def custom_hardswish_kernel(
        x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = x + 3.0
        y = x * y * (1.0 / (x + 3.0))
        tl.store(y_ptr + offsets, y, mask=mask)

    def custom_hardswish(self, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        custom_hardswish_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return y

    @triton.jit
    def custom_group_norm_kernel(
        x_ptr, gamma_ptr, beta_ptr, mean_ptr, var_ptr, out_ptr,
        n_channels, n_elements, eps, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        mean = tl.load(mean_ptr + offsets, mask=mask, other=0.0)
        var = tl.load(var_ptr + offsets, mask=mask, other=0.0)
        gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
        beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
        inv_std = 1.0 / tl.sqrt(var + eps)
        out = gamma * (x - mean) * inv_std + beta
        tl.store(out_ptr + offsets, out, mask=mask)

    def custom_group_norm(self, x):
        B, C, D, H, W = x.shape
        x = x.contiguous()
        gamma = torch.ones(C, device=x.device)
        beta = torch.zeros(C, device=x.device)
        mean = torch.mean(x, dim=[2, 3, 4])
        var = torch.var(x, dim=[2, 3, 4], unbiased=False)
        mean = mean.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        var = var.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        y = torch.empty_like(x)
        n_channels = C
        n_elements = x.numel()
        eps = 1e-5
        BLOCK_SIZE = 1024
        grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        custom_group_norm_kernel[grid](
            x, gamma, beta, mean, var, y, n_channels, n_elements, eps, BLOCK_SIZE=BLOCK_SIZE
        )
        return y

    @triton.jit
    def custom_mean_pool_kernel(
        x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x, mask=mask)

    def custom_mean_pool(self, x):
        x = x.contiguous()
        y = torch.empty(x.size(0), x.size(1), device=x.device)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        custom_mean_pool_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return y