import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# --------------------
# Triton kernel for 3D average pooling (kernel size 2x2x2, stride 2)
# --------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_D": 8, "BLOCK_SIZE_H": 8, "BLOCK_SIZE_W": 8}, num_warps=4),
        triton.Config({"BLOCK_SIZE_D": 4, "BLOCK_SIZE_H": 4, "BLOCK_SIZE_W": 4}, num_warps=2),
    ],
    key=["N", "C", "D_in", "H_in", "W_in"],
)
@triton.jit
def avg_pool3d_kernel(
    out_ptr,            # output pointer
    in_ptr,             # input pointer
    N, C, D_in, H_in, W_in,
    D_out, H_out, W_out,
    stride_d, stride_h, stride_w,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
):
    # compute program indices
    n = tl.program_id(0)
    c = tl.program_id(1)
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # compute global output offset
    out_offset = (
        n * (C * D_out * H_out * W_out)
        + c * (D_out * H_out * W_out)
        + d_out * (H_out * W_out)
        + h_out * W_out
        + w_out
    )

    # accumulate 2x2x2 window
    acc = tl.zeros([BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W], dtype=tl.float32)
    count = 0

    # iterate over the 8 voxels in the window
    for dd in range(2):
        for hh in range(2):
            for ww in range(2):
                d_in = d_out * stride_d + dd
                h_in = h_out * stride_h + hh
                w_in = w_out * stride_w + ww
                if (d_in < D_in) & (h_in < H_in) & (w_in < W_in):
                    in_offset = (
                        n * (C * D_in * H_in * W_in)
                        + c * (D_in * H_in * W_in)
                        + d_in * (H_in * W_in)
                        + h_in * W_in
                        + w_in
                    )
                    val = tl.load(in_ptr + in_offset, mask=True, other=0.0)
                    acc += val
                    count += 1

    # store mean
    mean = acc / count
    tl.store(out_ptr + out_offset, mean, mask=True)


def triton_avg_pool3d(x: torch.Tensor, kernel_size=2, stride=2) -> torch.Tensor:
    """
    Triton implementation of 3D average pooling with kernel_size=2 and stride=2.
    """
    assert x.is_cuda, "Input must be on CUDA"
    N, C, D_in, H_in, W_in = x.shape
    D_out = (D_in - kernel_size) // stride + 1
    H_out = (H_in - kernel_size) // stride + 1
    W_out = (W_in - kernel_size) // stride + 1

    out = torch.empty((N, C, D_out, H_out, W_out), dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        (N, C, D_out, H_out, W_out),
        meta["BLOCK_SIZE_D"],
        meta["BLOCK_SIZE_H"],
        meta["BLOCK_SIZE_W"],
    )

    avg_pool3d_kernel[grid](
        out,
        x,
        N, C, D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride, stride, stride,
        BLOCK_SIZE_D=64,
        BLOCK_SIZE_H=64,
        BLOCK_SIZE_W=64,
        stride_d=stride,
        stride_h=stride,
        stride_w=stride,
    )
    return out


# --------------------
# Triton kernel for 3D batch normalization (per channel)
# --------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=2),
    ],
    key=["N", "C", "D", "H", "W"],
)
@triton.jit
def batchnorm3d_kernel(
    out_ptr,
    in_ptr,
    gamma_ptr,
    beta_ptr,
    running_mean_ptr,
    running_var_ptr,
    eps,
    N, C, D, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * D * H * W
    mask = idx < total

    # compute channel index
    # flatten index to N, C, D, H, W
    n = idx // (C * D * H * W)
    rem = idx % (C * D * H * W)
    c = rem // (D * H * W)
    # load input
    x = tl.load(in_ptr + idx, mask=mask, other=0.0)
    # load gamma, beta
    gamma = tl.load(gamma_ptr + c, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + c, mask=mask, other=0.0)
    # load running mean & var
    mean = tl.load(running_mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(running_var_ptr + c, mask=mask, other=1.0)

    # normalize
    inv_std = 1.0 / tl.sqrt(var + eps)
    y = gamma * (x - mean) * inv_std + beta
    tl.store(out_ptr + idx, y, mask=mask)


def triton_batchnorm3d(x: torch.Tensor,
                      running_mean: torch.Tensor,
                      running_var: torch.Tensor,
                      weight: torch.Tensor,
                      bias: torch.Tensor,
                      eps: float = 1e-5) -> torch.Tensor:
    """
    Triton implementation of 3D batch normalization in evaluation mode.
    """
    assert x.is_cuda, "Input must be on CUDA"
    N, C, D, H, W = x.shape
    out = torch.empty_like(x)

    grid = lambda meta: (
        (N * D * H * W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    batchnorm3d_kernel[grid](
        out,
        x,
        weight,
        bias,
        running_mean,
        running_var,
        eps,
        N, C, D, H, W,
        BLOCK_SIZE=1024,
    )
    return out


# --------------------
# Optimized model using custom Triton kernels
# --------------------
class ModelNew(nn.Module):
    """
    A model that performs a 3D transposed convolution, followed by batch normalization,
    and two average pooling layers implemented with Triton kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.Identity()  # placeholder; will use Triton
        self.avg_pool2 = nn.Identity()

    def forward(self, x):
        # ConvTranspose3d (native PyTorch)
        x = self.conv_transpose(x)

        # BatchNorm3d (custom Triton)
        x = triton_batchnorm3d(
            x,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.eps
        )

        # AvgPool3d (custom Triton)
        x = triton_avg_pool3d(x, kernel_size=2, stride=2)
        x = triton_avg_pool3d(x, kernel_size=2, stride=2)
        return x