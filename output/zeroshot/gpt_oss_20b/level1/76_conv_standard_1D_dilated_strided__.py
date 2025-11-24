import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Triton kernel for a general 1‑D convolution (with stride, dilation, bias)
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    ],
    key=["batch", "out_channels", "length_out"],
)
@triton.jit
def conv1d_kernel(
    in_ptr,            # (batch, in_channels, length_in)
    weight_ptr,        # (out_channels, in_channels, kernel_size)
    bias_ptr,          # (out_channels,) or None
    out_ptr,           # (batch, out_channels, length_out)
    batch: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    length_in: tl.constexpr,
    length_out: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    dilation: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program processes a contiguous block of output elements.
    The block size is chosen to maximise occupancy.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_out = batch * out_channels * length_out
    mask = offsets < total_out

    # Decode global indices into (b, oc, o)
    # b * out_channels * length_out + oc * length_out + o
    bc = offsets // (out_channels * length_out)
    rem = offsets % (out_channels * length_out)
    oc = rem // length_out
    o = rem % length_out

    # Load weight slice for this output channel
    # Shape: (in_channels, kernel_size)
    w_offset = oc * in_channels * kernel_size
    w = tl.load(weight_ptr + w_offset + tl.arange(0, in_channels)[:, None] * kernel_size
               + tl.arange(0, kernel_size)[None, :])

    # Accumulate convolution result
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for ic in range(in_channels):
        # Input offset for channel ic
        in_offset_base = bc * in_channels * length_in + ic * length_in
        # Compute input indices for the kernel window
        in_offsets = in_offset_base + (o * stride + tl.arange(0, kernel_size) * dilation)
        # Load input values
        inp = tl.load(in_ptr + in_offsets[None, :], mask=mask[:, None], other=0.0)
        # Weight for channel ic (kernel elements)
        w_ic = w[ic, :]
        # Element‑wise multiplication and reduce over kernel dimension
        acc += tl.sum(inp * w_ic, axis=1)

    # Add bias if present
    if has_bias:
        acc += tl.load(bias_ptr + oc)

    # Store result
    out_base = bc * out_channels * length_out + oc * length_out
    tl.store(out_ptr + out_base + o, acc, mask=mask)


# --------------------------------------------------------------------------- #
# Wrapper that mimics nn.Conv1d forward
# --------------------------------------------------------------------------- #
def conv1d_torch(x: torch.Tensor,
                 weight: torch.Tensor,
                 bias: torch.Tensor | None = None,
                 stride: int = 1,
                 dilation: int = 1) -> torch.Tensor:
    """
    x: (B, C_in, L_in)      dtype bf16 or fp32
    weight: (C_out, C_in, K)
    bias: (C_out,) or None
    Returns:
        out: (B, C_out, L_out)
    """
    assert x.is_cuda and weight.is_cuda
    B, Cin, Lin = x.shape
    Cout, _, K = weight.shape

    Lout = (Lin - (K - 1) * dilation - 1) // stride + 1
    out = torch.empty((B, Cout, Lout), dtype=x.dtype, device=x.device)

    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Grid and launch
    grid = lambda meta: (
        (B * Cout * Lout + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    conv1d_kernel[grid](
        x,
        weight,
        bias,
        out,
        B, Cin, Cout, Lin, Lout,
        K, stride, dilation,
        bias is not None,
        BLOCK_SIZE=meta["BLOCK_SIZE"],
    )
    return out


# --------------------------------------------------------------------------- #
# Optimized model using the custom Triton kernel
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Performs a 1D convolution using a custom Triton kernel.
    The API matches the original nn.Conv1d interface.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(
            torch.empty((out_channels, in_channels, kernel_size),
                        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                        device="cuda")
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels,
                                                 dtype=self.weight.dtype,
                                                 device="cuda"))
        else:
            self.bias = None
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv1d_torch(x,
                            self.weight,
                            self.bias,
                            stride=self.stride,
                            dilation=self.dilation)