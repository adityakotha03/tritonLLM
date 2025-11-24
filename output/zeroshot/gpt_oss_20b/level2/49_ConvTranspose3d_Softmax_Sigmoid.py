import torch
import torch.nn as nn
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton kernel that fuses softmax (dim=1) and sigmoid
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    ],
    key=["N", "C", "D", "H", "W"],
)
@triton.jit
def softmax_sigmoid_fused_kernel(
    x_ptr,          # (N, C, D, H, W)
    out_ptr,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one voxel (N, D, H, W)
    # block_start holds the index of the voxel in the flattened spatial grid
    block_idx = tl.program_id(0)
    max_voxels = N * D * H * W
    if block_idx >= max_voxels:
        return

    # Compute the multi-dimensional index of this voxel
    w = block_idx % W
    h = (block_idx // W) % H
    d = (block_idx // (W * H)) % D
    n = block_idx // (W * H * D)

    # Offsets for the start of this voxel in each dimension
    offset = (
        n * stride_n
        + d * stride_d
        + h * stride_h
        + w * stride_w
    )

    # Load the entire channel vector for this voxel
    offsets = offset + tl.arange(0, BLOCK_SIZE) * stride_c
    mask = tl.arange(0, BLOCK_SIZE) < C

    # Load values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # ------------------------------
    # Softmax computation
    # ------------------------------
    # 1) Find max for numerical stability
    max_val = tl.max(x, mask=mask)
    # 2) Subtract max and exponentiate
    x = tl.exp(x - max_val)
    # 3) Sum of exponentials
    sum_exp = tl.sum(x, mask=mask)
    # 4) Normalize
    softmax = x / sum_exp

    # ------------------------------
    # Sigmoid applied to softmax output
    # ------------------------------
    out = tl.sigmoid(softmax)

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_softmax_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    x: Tensor of shape (N, C, D, H, W)
    Returns: Tensor of same shape after applying softmax over channel
             and then sigmoid.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    N, C, D, H, W = x.shape
    stride_n, stride_c, stride_d, stride_h, stride_w = x.stride()

    grid = lambda meta: (N * D * H * W,)

    softmax_sigmoid_fused_kernel[grid](
        x,
        out,
        N, C, D, H, W,
        stride_n, stride_c, stride_d, stride_h, stride_w,
        BLOCK_SIZE=meta["BLOCK_SIZE"],
    )
    return out


# ----------------------------------------------------------------------
# New model with fused softmax+sigmoid kernel
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, then fuses
    Softmax (dim=1) and Sigmoid into a single Triton kernel.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias=True,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    def forward(self, x):
        x = self.conv_transpose(x)          # shape: (N, C_out, D, H, W)
        x = fused_softmax_sigmoid(x)       # fused softmax + sigmoid
        return x