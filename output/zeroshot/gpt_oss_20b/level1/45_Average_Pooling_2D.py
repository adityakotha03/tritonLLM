import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool_kernel(
    x_ptr,        # pointer to input tensor
    out_ptr,      # pointer to output tensor
    B: tl.constexpr,      # batch size
    C: tl.constexpr,      # channel count
    H: tl.constexpr,      # input height
    W: tl.constexpr,      # input width
    H_out: tl.constexpr,  # output height
    W_out: tl.constexpr,  # output width
    stride: tl.constexpr,
    kernel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for 2‑D average pooling.
    Each program instance processes BLOCK_SIZE output elements.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    total_out = B * C * H_out * W_out
    mask = offsets < total_out

    # --- Compute coordinates of each output element --------------------
    out_idx = offsets
    bc = out_idx // (H_out * W_out)
    yx = out_idx % (H_out * W_out)

    c = bc % C
    b = bc // C
    y = yx // W_out
    x = yx % W_out

    # base pointer for the (b, c) slice in the input tensor
    base_bc = ((b * C + c) * H) * W

    # --- Accumulate the sum over the kernel window ---------------------
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for ky in range(kernel):
        in_y = y * stride + ky
        for kx in range(kernel):
            in_x = x * stride + kx
            in_offset = base_bc + in_y * W + in_x
            val = tl.load(x_ptr + in_offset, mask=mask, other=0.0)
            sum_val += val

    # --- Store the average ------------------------------------------------
    out_val = sum_val / (kernel * kernel)
    tl.store(out_ptr + offsets, out_val, mask=mask)


def triton_avg_pool2d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int | None = None,
) -> torch.Tensor:
    """
    2‑D average pooling implemented with a custom Triton kernel.
    """
    if stride is None:
        stride = kernel_size

    B, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    out = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 256  # tunable

    grid = lambda meta: ((B * C * H_out * W_out + meta["BLOCK_SIZE"] - 1)
                         // meta["BLOCK_SIZE"],)

    avg_pool_kernel[grid](
        x, out,
        B, C, H, W, H_out, W_out,
        stride=stride,
        kernel=kernel_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Model that replaces the standard AvgPool2d with a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        # padding is ignored in this custom implementation; only support zero padding
        assert padding == 0, "Custom Triton kernel only supports zero padding."

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool2d(x, self.kernel_size, self.stride)