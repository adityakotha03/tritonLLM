import torch
import torch.nn as nn
import triton
import triton.language as tl

# -------------------- Triton kernel --------------------
@triton.jit
def _maxpool1d_kernel(
    x_ptr,              # pointer to input (batch*channels, seq_len)
    y_ptr,              # pointer to output (batch*channels, output_len)
    seq_len: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    output_len: tl.constexpr,
    N: tl.constexpr,                 # number of channels * batch
    BLOCK_SIZE: tl.constexpr,        # total threads per program
):
    # Thread indices within the program
    idx = tl.arange(0, BLOCK_SIZE)

    # Map to (output position, channel index)
    out_idx = idx % output_len          # 0 .. output_len-1
    ch_idx  = idx // output_len        # 0 .. N-1

    # Valid mask for the current thread
    mask = (out_idx < output_len) & (ch_idx < N)

    # Base pointers for the current channel
    x_base = x_ptr + ch_idx * seq_len
    y_base = y_ptr + ch_idx * output_len

    # Starting position in the input for the current output element
    start = out_idx * stride - padding

    # Initialise maximum value to negative infinity
    max_val = tl.full([BLOCK_SIZE], -float("inf"))

    # Iterate over the kernel window
    for k in range(kernel_size):
        pos   = start + k * dilation
        valid = (pos >= 0) & (pos < seq_len)
        val   = tl.load(x_base + pos, mask=valid, other=-float("inf"))
        max_val = tl.maximum(max_val, val)

    # Store result (only for valid threads)
    tl.store(y_base + out_idx, max_val, mask=mask)


# -------------------- Triton wrapper --------------------
def triton_maxpool(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    """
    Performs 1‑D max pooling on a 3‑D tensor (batch, channels, seq_len)
    using a custom Triton kernel.
    """
    # Ensure input is contiguous and on GPU
    x = x.contiguous().float()          # Triton kernel operates in float32
    batch, channels, seq_len = x.shape
    N = batch * channels

    # Compute output length (same formula as torch.nn.functional.max_pool1d)
    output_len = (seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    # Allocate output tensor
    out = torch.empty((batch, channels, output_len), dtype=torch.float32, device=x.device)

    # Kernel launch parameters
    BLOCK_SIZE = 1024                      # number of threads per program
    num_programs = (N * output_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = lambda meta: (num_programs,)

    # Launch the kernel
    _maxpool1d_kernel[grid](
        x, out,
        seq_len,
        kernel_size,
        stride,
        padding,
        dilation,
        output_len,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


# -------------------- Optimized model --------------------
class ModelNew(nn.Module):
    """
    Max‑Pooling 1D implemented with a custom Triton kernel.
    """
    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        if self.return_indices:
            raise NotImplementedError("`return_indices` is not supported in the Triton implementation.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_maxpool(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )