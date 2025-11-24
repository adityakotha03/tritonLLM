import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['B', 'C', 'D', 'H', 'W', 'k', 's', 'p', 'out_D', 'out_H', 'out_W'],
)
@triton.jit
def avgpool3d_kernel(
    in_ptr,          # pointer to input tensor data
    out_ptr,         # pointer to output tensor data
    B, C, D, H, W,   # input dimensions
    k, s, p,         # kernel, stride, padding
    out_D, out_H, out_W,  # output dimensions
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for 3D average pooling.
    Each thread processes one output element.
    """
    # linear index of the output element processed by this thread
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < (B * C * out_D * out_H * out_W)

    # unpack the 5D output index from the linear index
    # order: b, c, d_o, h_o, w_o
    b = idx // (C * out_D * out_H * out_W)
    rem1 = idx % (C * out_D * out_H * out_W)
    c = rem1 // (out_D * out_H * out_W)
    rem2 = rem1 % (out_D * out_H * out_W)
    d_o = rem2 // (out_H * out_W)
    rem3 = rem2 % (out_H * out_W)
    h_o = rem3 // out_W
    w_o = rem3 % out_W

    # compute the starting input indices for the kernel window
    d_start = d_o * s - p
    h_start = h_o * s - p
    w_start = w_o * s - p

    sum_val = tl.zeros([], dtype=tl.float32)
    count = 0

    # iterate over the kernel window
    for kd in range(k):
        d_in = d_start + kd
        if d_in < 0 or d_in >= D:
            continue
        for kh in range(k):
            h_in = h_start + kh
            if h_in < 0 or h_in >= H:
                continue
            for kw in range(k):
                w_in = w_start + kw
                if w_in < 0 or w_in >= W:
                    continue
                # linear index of the input element
                in_idx = ((b * C + c) * D + d_in) * H * W + h_in * W + w_in
                val = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
                sum_val += val
                count += 1

    # avoid division by zero (should never happen with valid pooling params)
    out_val = sum_val / tl.math.max(tl.cast(count, tl.float32), 1.0)

    # compute the linear index for the output element
    out_idx = ((b * C + c) * out_D + d_o) * out_H * out_W + h_o * out_W + w_o
    tl.store(out_ptr + out_idx, out_val, mask=mask)


def triton_avgpool3d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """
    Wrapper that calls the Triton average‑pooling kernel.
    """
    assert x.is_cuda and x.is_contiguous(), "Input must be a contiguous CUDA tensor"

    B, C, D, H, W = x.shape
    k = kernel_size
    s = stride if stride is not None else k
    p = padding

    out_D = (D + 2 * p - k) // s + 1
    out_H = (H + 2 * p - k) // s + 1
    out_W = (W + 2 * p - k) // s + 1

    out = torch.empty((B, C, out_D, out_H, out_W), dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        (B * C * out_D * out_H * out_W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    avgpool3d_kernel[grid](
        x,
        out,
        B, C, D, H, W,
        k, s, p,
        out_D, out_H, out_W,
        BLOCK_SIZE=256,
    )

    return out


class ModelNew(nn.Module):
    """
    Triton‑based implementation of 3D average pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avgpool3d(x, self.kernel_size, self.stride, self.padding)