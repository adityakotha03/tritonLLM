import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------------
# Triton kernel for global avg pooling
# ------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    ],
    key=["N"],
)
@triton.jit
def global_avg_pool3d_kernel(
    inp_ptr,          # [B, C, D, H, W] input
    out_ptr,          # [B, C, 1, 1, 1] output
    D, H, W,          # spatial dimensions
    B, C,             # batch and channel counts
    N,                # total number of (b,c) pairs
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE
    end = tl.minimum(idx + BLOCK_SIZE, N)

    # constants for indexing
    DWH = D * H * W

    for i in range(idx, end):
        bc = i
        b = bc // C
        c = bc % C

        base = ((b * C + c) * DWH)

        sum_val = tl.zeros([1], dtype=tl.float32)
        # iterate over spatial tiles
        for d_offset in range(0, DWH, BLOCK_SIZE):
            offsets = base + d_offset + tl.arange(0, BLOCK_SIZE)
            mask = offsets < base + DWH
            vals = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
            sum_val += tl.sum(vals, axis=0)

        mean_val = sum_val / tl.float32(DWH)
        out_ptr[bc] = mean_val


def global_avg_pool3d_torch(inp: torch.Tensor) -> torch.Tensor:
    """
    inp: [B, C, D, H, W] float32 tensor on CUDA
    returns: [B, C, 1, 1, 1] float32 tensor on CUDA
    """
    B, C, D, H, W = inp.shape
    out = torch.empty((B, C, 1, 1, 1), dtype=inp.dtype, device=inp.device)

    N = B * C
    BLOCK_SIZE = 128

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    global_avg_pool3d_kernel[grid](
        inp, out, D, H, W, B, C, N, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


# ------------------------------
# Triton kernel for scale + clamp
# ------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def scale_and_clamp_kernel(
    inp_ptr,          # [B, C, D, H, W]
    out_ptr,          # same shape
    scale: tl.constexpr,
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
    N,                # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    end = tl.minimum(start + BLOCK_SIZE, N)

    for i in range(start, end):
        val = inp_ptr[i] * scale
        val = tl.where(val < clamp_min, clamp_min, val)
        val = tl.where(val > clamp_max, clamp_max, val)
        out_ptr[i] = val


def scale_and_clamp_torch(inp: torch.Tensor, scale: float, clamp_min: float, clamp_max: float) -> torch.Tensor:
    """
    inp: tensor on CUDA
    returns: tensor of same shape
    """
    out = torch.empty_like(inp)
    N = inp.numel()
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    scale_and_clamp_kernel[grid](
        inp, out, scale, clamp_min, clamp_max, N, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


# ------------------------------
# Optimized Model
# ------------------------------
class ModelNew(nn.Module):
    """
    Optimized model with custom Triton kernels for scale+clamp and global average pooling.
    The transposed convolution and max pooling use the default PyTorch implementations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                stride=stride, padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.clamp_min = 0.0
        self.clamp_max = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        x = global_avg_pool3d_torch(x)
        x = scale_and_clamp_torch(x, self.scale, self.clamp_min, self.clamp_max)
        return x