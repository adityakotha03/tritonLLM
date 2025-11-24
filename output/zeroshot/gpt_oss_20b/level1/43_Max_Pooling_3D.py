import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool3d_kernel(
    x_ptr,
    y_ptr,
    n_out,
    stride_d, stride_h, stride_w,
    in_d, in_h, in_w,
    pad_d, pad_h, pad_w,
    dilation_d, dilation_h, dilation_w,
    kernel_d, kernel_h, kernel_w,
    N: tl.constexpr,
    C: tl.constexpr,
    od: tl.constexpr,
    oh: tl.constexpr,
    ow: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Starting index for this program (block)
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_out

    # Precompute constants for coordinate decoding
    per_output = od * oh * ow
    per_channel = C * per_output
    per_batch = N * per_channel

    # Decode linear index to (b, c, od_, oh_, ow_)
    b   = (offsets // per_channel) % N
    c   = (offsets // per_output) % C
    od_ = (offsets // (oh * ow)) % od
    oh_ = (offsets // ow) % oh
    ow_ = offsets % ow

    # Initialize maximum values
    max_val = tl.full([BLOCK_SIZE], -1e9, dtype=tl.float32)

    # Iterate over the kernel window
    for kd in range(kernel_d):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                d_in = od_ * stride_d - pad_d + kd * dilation_d
                h_in = oh_ * stride_h - pad_h + kh * dilation_h
                w_in = ow_ * stride_w - pad_w + kw * dilation_w

                inside = (d_in >= 0) & (d_in < in_d) & \
                         (h_in >= 0) & (h_in < in_h) & \
                         (w_in >= 0) & (w_in < in_w)

                # Compute linear input offset
                offset = ((b * C + c) * in_d + d_in) * in_h * in_w + h_in * in_w + w_in

                val = tl.load(x_ptr + offset, mask=mask & inside, other=-1e9)
                max_val = tl.maximum(max_val, val)

    tl.store(y_ptr + offsets, max_val, mask=mask)


def triton_maxpool3d(x: torch.Tensor,
                     kernel_size: int,
                     stride: int,
                     padding: int,
                     dilation: int) -> torch.Tensor:
    N, C, D, H, W = x.shape
    k = kernel_size
    s = stride
    p = padding
    d = dilation

    # Compute output dimensions (same formula as nn.MaxPool3d)
    od = (D + 2 * p - d * (k - 1) - 1) // s + 1
    oh = (H + 2 * p - d * (k - 1) - 1) // s + 1
    ow = (W + 2 * p - d * (k - 1) - 1) // s + 1

    out = torch.empty((N, C, od, oh, ow), device=x.device, dtype=x.dtype)

    n_out = out.numel()
    BLOCK_SIZE = 256

    grid = lambda meta: ((n_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    maxpool3d_kernel[grid](
        x,
        out,
        n_out,
        s, s, s,
        D, H, W,
        p, p, p,
        d, d, d,
        k, k, k,
        N, C, od, oh, ow,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Max Pooling 3D implemented with a custom Triton kernel.
    """

    def __init__(self,
                 kernel_size: int,
                 stride: int = None,
                 padding: int = 0,
                 dilation: int = 1,
                 return_indices: bool = False,
                 ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_maxpool3d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )