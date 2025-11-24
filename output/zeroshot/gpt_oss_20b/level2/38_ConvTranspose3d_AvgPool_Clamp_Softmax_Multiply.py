import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------- Custom Triton kernels ----------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    ],
    key=["n_elements"],
)
@triton.jit
def clamp_kernel(
    x_ptr,
    out_ptr,
    min_val,
    max_val,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = tl.where(x < min_val, min_val, x)
    x = tl.where(x > max_val, max_val, x)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # compute max for numerical stability
    max_val = tl.max(x, mask=mask)
    max_val = tl.broadcast_to(max_val, (BLOCK_SIZE,))
    x_exp = tl.exp(x - max_val)

    sum_exp = tl.sum(x_exp, mask=mask)
    sum_exp = tl.broadcast_to(sum_exp, (BLOCK_SIZE,))

    softmax = x_exp / sum_exp
    tl.store(out_ptr + offsets, softmax, mask=mask)


# ---------- Helper wrappers ----------

def triton_clamp(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: ( (n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], )
    clamp_kernel[grid](x, out, min_val, max_val, n, BLOCK_SIZE=meta := 256)
    return out


def triton_softmax(x: torch.Tensor, dim: int = 2) -> torch.Tensor:
    # flatten all dimensions except the target dim
    shape = list(x.shape)
    dim_size = shape[dim]
    # move dim to last
    x = x.permute([i for i in range(len(shape)) if i != dim] + [dim])
    flat = x.contiguous()
    out = torch.empty_like(flat)
    n = flat.numel()
    grid = lambda meta: ( (n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], )
    softmax_kernel[grid](flat, out, n, BLOCK_SIZE=meta := 128)
    out = out.view(x.shape)
    # permute back
    inv_perm = list(range(x.dim() - 1)) + [x.dim() - 1]
    for i, d in enumerate([i for i in range(len(shape)) if i != dim]):
        inv_perm[d] = i
    return out.permute(inv_perm)


# ---------- Optimized model ----------

class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for clamp and softmax.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        pool_kernel_size,
        clamp_min,
        clamp_max,
    ):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average pooling
        x = self.avg_pool(x)
        # Transposed convolution
        x = self.conv_transpose(x)
        # Clamp using Triton
        x = triton_clamp(x, self.clamp_min, self.clamp_max)
        # Softmax over flattened spatial dims
        b, c, d, h, w = x.shape
        x = x.view(b, c, -1)
        x = triton_softmax(x, dim=2)
        x = x.view(b, c, d, h, w)
        # Scale multiplication
        x = x * self.scale
        return x