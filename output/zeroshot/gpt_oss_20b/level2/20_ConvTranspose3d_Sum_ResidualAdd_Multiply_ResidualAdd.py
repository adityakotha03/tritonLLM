import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def conv_out_fusion_kernel(
    conv_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    bias_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Elementwise fusion of:
        y = conv + bias + conv + conv * conv + conv
      -> y = conv * conv + 2 * conv + bias
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load conv output
    conv = tl.load(conv_ptr + offsets, mask=mask, other=0.0)

    # Load bias (broadcast along spatial dims)
    bias_offset = (offsets // (bias_dim * bias_dim * bias_dim)) % bias_dim
    bias = tl.load(bias_ptr + bias_offset, mask=mask, other=0.0)

    # Compute fused expression
    out = conv * conv + 2.0 * conv + bias

    tl.store(out_ptr + offsets, out, mask=mask)


def fuse_conv_output(conv: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Wrapper that launches the fusion kernel.
    """
    assert conv.is_cuda and bias.is_cuda
    n_elements = conv.numel()
    out = torch.empty_like(conv)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    conv_out_fusion_kernel[grid](
        conv,
        bias,
        out,
        n_elements,
        bias_dim=bias.shape[0],
        BLOCK_SIZE=256,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that fuses bias addition, residuals, and multiplication
    into a single Triton kernel after the ConvTranspose3d.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        # bias is shaped (C,1,1,1) and will be broadcast
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ConvTranspose3d produces shape [B, C, D, H, W]
        conv_out = self.conv_transpose(x)
        # Fuse bias, residuals, and multiplication in one kernel
        fused_out = fuse_conv_output(conv_out, self.bias)
        return fused_out