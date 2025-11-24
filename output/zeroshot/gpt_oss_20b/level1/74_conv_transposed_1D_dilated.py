import torch
import torch.nn as nn
import triton
import triton.language as tl


# ────────────────────────────────────────────────────────────────────────────────
# 1️⃣  Triton kernel for a 1‑D transposed convolution (stride=1, dilation=3)
# ────────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["batch", "out_channels", "out_len"],
)
@triton.jit
def conv_transpose_1d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    batch,
    in_channels,
    out_channels,
    in_len,
    out_len,
    k,          # kernel size
    stride,
    padding,
    dilation,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    lane = tl.arange(0, BLOCK_SIZE)

    # Compute the linear index of the output element to write
    out_idx = offset + lane
    mask = out_idx < (batch * out_channels * out_len)

    # Decode the multi‑dimensional indices
    b = out_idx // (out_channels * out_len)
    oc = (out_idx // out_len) % out_channels
    op = out_idx % out_len

    # Prepare accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    if mask:
        # Compute the effective input start index (including padding)
        in_start = op * stride - padding

        # Iterate over all input channels and kernel elements
        for ic in range(in_channels):
            # Load the kernel slice for this (oc, ic) pair
            w_offset = (oc * in_channels + ic) * k
            w = tl.load(
                w_ptr + w_offset + tl.arange(0, k),
                mask=None,
                other=0.0,
            )

            for ki in range(k):
                # Index in the input signal
                inp_idx = in_start + ki * dilation
                # Bounds check for the input
                inp_mask = (inp_idx >= 0) & (inp_idx < in_len)
                if inp_mask:
                    # Load input value (broadcast across lanes)
                    x_val = tl.load(
                        x_ptr + b * in_channels * in_len + ic * in_len + inp_idx,
                        mask=None,
                        other=0.0,
                    )
                    acc += x_val * w[ki]

    # Write the accumulated value
    tl.store(out_ptr + out_idx, acc, mask=mask)


# ────────────────────────────────────────────────────────────────────────────────
# 2️⃣  Wrapper that prepares tensors and launches the kernel
# ────────────────────────────────────────────────────────────────────────────────
def conv_transpose_1d_torch(x: torch.Tensor,
                            weight: torch.Tensor,
                            stride: int,
                            padding: int,
                            dilation: int,
                            bias: torch.Tensor | None = None) -> torch.Tensor:
    """
    x      : (B, C_in, L_in)
    weight : (C_out, C_in, K)
    bias   : (C_out,)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    B, C_in, L_in = x.shape
    C_out, _, K = weight.shape
    # output length for stride=1
    L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) + stride
    out = torch.empty((B, C_out, L_out), dtype=x.dtype, device=x.device)

    # Flatten pointers
    x_ptr = x.data_ptr()
    w_ptr = weight.data_ptr()
    out_ptr = out.data_ptr()

    # Grid size: one program per output element
    grid = lambda meta: ((B * C_out * L_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    conv_transpose_1d_kernel[grid](
        x_ptr,
        w_ptr,
        out_ptr,
        B,
        C_in,
        C_out,
        L_in,
        L_out,
        K,
        stride,
        padding,
        dilation,
        BLOCK_SIZE=128,
    )

    if bias is not None:
        out += bias.view(1, -1, 1)
    return out


# ────────────────────────────────────────────────────────────────────────────────
# 3️⃣  Optimized PyTorch model using the Triton kernel
# ────────────────────────────────────────────────────────────────────────────────
class ModelNew(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, device="cuda")
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, device="cuda"))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose_1d_torch(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=self.bias,
        )