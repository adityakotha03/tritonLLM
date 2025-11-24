import torch
import torch.nn as nn
import triton
import triton.language as tl

# ---------- Triton kernel for 1D convolution (groups=1) ----------
@triton.jit
def conv1d_kernel(
    x_ptr: tl.tensor,          # (B, C_in, L)
    w_ptr: tl.tensor,          # (C_out, C_in, K)
    b_ptr: tl.tensor,          # (C_out,)   (may be None)
    out_ptr: tl.tensor,        # (B, C_out, L_out)
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    L_in: tl.constexpr,
    L_out: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    K: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a contiguous block of output positions
    prog_id = tl.program_id(0)
    # Compute total number of output elements per batch per channel
    total_out = batch_size * C_out * L_out
    # Compute starting index for this program
    start = prog_id * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_out

    # Map linear offset to (b, oc, out_idx)
    # order: batch, channel, position
    out_idx = offsets
    out_b = out_idx // (C_out * L_out)
    out_tmp = out_idx % (C_out * L_out)
    out_oc = out_tmp // L_out
    out_pos = out_tmp % L_out

    # Compute corresponding input start index
    inp_start = out_pos * stride - padding
    # For each output element, compute convolution sum
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for ki in range(K):
        # Input positions for this kernel index
        inp_idx = inp_start + ki * dilation
        # Need to handle boundary: if inp_idx < 0 or inp_idx >= L_in, contribution is zero
        inp_valid = (inp_idx >= 0) & (inp_idx < L_in)
        # Load input values: shape (B, C_in, L_in) -> gather per batch and channel
        # We compute for all in_channels
        # Compute linear address: ((b * C_in + c_in) * L_in + inp_idx)
        base = (out_b * C_in + tl.arange(0, C_in)) * L_in + inp_idx
        # Mask per channel
        channel_mask = tl.arange(0, C_in) < C_in
        # Broadcast to BLOCK_SIZE
        inp_mask = tl.broadcast(inp_valid, [BLOCK_SIZE]) & channel_mask
        x_vals = tl.load(x_ptr + base, mask=inp_mask, other=0.0)
        # Weight for this kernel index: shape (C_out, C_in, K)
        w_base = (out_oc * C_in + tl.arange(0, C_in)) * K + ki
        w_vals = tl.load(w_ptr + w_base, mask=channel_mask, other=0.0)
        # Multiply and accumulate
        acc += tl.sum(x_vals * w_vals, axis=0)

    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + out_oc, mask=mask, other=0.0)
        acc += bias

    # Store output
    out_base = (out_b * C_out + out_oc) * L_out + out_pos
    tl.store(out_ptr + out_base, acc, mask=mask)

# ---------- Triton wrapper ----------
def triton_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
    dilation: int,
    groups: int = 1,
) -> torch.Tensor:
    assert groups == 1, "Only groups=1 is supported in this implementation."
    batch_size, C_in, L_in = x.shape
    C_out, _, K = weight.shape
    # Compute output length
    L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    out = torch.empty((batch_size, C_out, L_out), device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 128  # Tune if needed
    grid = lambda meta: ((batch_size * C_out * L_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    conv1d_kernel[grid](
        x_ptr=x,
        w_ptr=weight,
        b_ptr=bias,
        out_ptr=out,
        stride=stride,
        padding=padding,
        dilation=dilation,
        L_in=L_in,
        L_out=L_out,
        C_in=C_in,
        C_out=C_out,
        K=K,
        batch_size=batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

# ---------- Optimized model ----------
class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation using a custom Triton kernel.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

        # Weight parameter: (out_channels, in_channels, kernel_size)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, device="cuda")
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, device="cuda"))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            groups=self.groups,
        )