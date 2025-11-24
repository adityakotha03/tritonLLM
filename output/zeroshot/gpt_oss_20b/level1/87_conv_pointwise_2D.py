import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for a pointwise (1×1) 2‑D convolution
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_OC": 32}, num_warps=2),
        triton.Config({"BLOCK_SIZE_OC": 64}, num_warps=4),
        triton.Config({"BLOCK_SIZE_OC": 128}, num_warps=8),
    ],
    key=["n_pixels", "oc_total"],
)
@triton.jit
def pointwise_conv2d_kernel(
    in_ptr,          # [n_pixels, in_channels]
    weight_ptr,      # [oc_total, in_channels]
    bias_ptr,        # [oc_total] or nullptr
    out_ptr,         # [n_pixels, oc_total]
    n_pixels: tl.constexpr,
    in_channels: tl.constexpr,
    oc_total: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
):
    """
    Each program handles a single pixel (row in the flattened input) and
    computes BLOCK_SIZE_OC output channels for that pixel.
    """
    pixel_idx = tl.program_id(0)                     # pixel index 0 .. n_pixels-1
    oc_start = tl.program_id(1) * BLOCK_SIZE_OC      # starting output channel index

    # Load the whole input vector for this pixel (broadcast across threads)
    in_offset = pixel_idx * in_channels
    in_vec = tl.load(in_ptr + in_offset + tl.arange(0, in_channels))

    # Accumulate partial dot products for this block of output channels
    acc = tl.zeros([BLOCK_SIZE_OC], dtype=tl.float32)

    # Loop over input channels in chunks to keep data in registers
    for ic in range(0, in_channels, BLOCK_SIZE_OC):
        # Load a chunk of the weight matrix: [BLOCK_SIZE_OC, BLOCK_SIZE_OC]
        weight_chunk = tl.load(
            weight_ptr + (oc_start + tl.arange(0, BLOCK_SIZE_OC)) * in_channels
            + ic
        )  # shape (BLOCK_SIZE_OC, BLOCK_SIZE_OC)

        # Broadcast the input vector chunk
        in_chunk = in_vec[ic : ic + BLOCK_SIZE_OC]  # shape (BLOCK_SIZE_OC)

        # Matrix–vector multiply: accumulate
        acc += tl.dot(weight_chunk, in_chunk)

    # Add bias if provided
    if bias_ptr is not None:
        bias_chunk = tl.load(bias_ptr + oc_start + tl.arange(0, BLOCK_SIZE_OC))
        acc += bias_chunk

    # Store the computed block of output channels
    out_offset = pixel_idx * oc_total + oc_start
    tl.store(out_ptr + out_offset + tl.arange(0, BLOCK_SIZE_OC), acc)


def triton_pointwise_conv2d(
    x: torch.Tensor,          # (B, C_in, H, W)
    weight: torch.Tensor,     # (C_out, C_in)
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Wrapper around the Triton kernel that performs a 1×1 convolution
    (pointwise conv) over the spatial dimensions.
    """
    assert x.is_cuda and weight.is_cuda
    if bias is not None:
        assert bias.is_cuda

    B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    n_pixels = B * H * W

    # Flatten input to shape (n_pixels, C_in)
    in_flat = x.permute(0, 2, 3, 1).reshape(n_pixels, C_in)
    out_flat = torch.empty((n_pixels, C_out), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        n_pixels,
        (C_out + meta["BLOCK_SIZE_OC"] - 1) // meta["BLOCK_SIZE_OC"],
    )

    pointwise_conv2d_kernel[grid](
        in_flat,
        weight,
        bias,
        out_flat,
        n_pixels=n_pixels,
        in_channels=C_in,
        oc_total=C_out,
        BLOCK_SIZE_OC=meta["BLOCK_SIZE_OC"],
    )

    # Reshape back to (B, C_out, H, W)
    return out_flat.view(B, H, W, C_out).permute(0, 3, 1, 2).contiguous()


class ModelNew(nn.Module):
    """
    Pointwise 2‑D convolution implemented with a custom Triton kernel.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, device="cuda")
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(out_channels, device="cuda")
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_pointwise_conv2d(x, self.weight, self.bias)