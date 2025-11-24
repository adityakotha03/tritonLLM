import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------------------------------------------------
# Triton kernel that fuses softmax + bias addition + scaling + sigmoid
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["C"],
)
@triton.jit
def _softmax_add_scale_sigmoid_kernel(
    input_ptr,          # [B, C, H, W] input
    bias_ptr,           # [C, 1, 1] bias
    output_ptr,         # [B, C, H, W] output
    B, C, H, W,         # dimensions
    scaling_factor: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute global index for (b, h, w)
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)

    # Each thread processes BLOCK_SIZE contiguous channels
    c_start = tl.arange(0, BLOCK_SIZE)
    c_end = c_start + B * H * W * C  # unused but keeps compiler happy
    mask = c_start < C

    # Load bias (broadcasted over spatial dimensions)
    bias = tl.load(bias_ptr + c_start, mask=mask, other=0.0)

    # Offsets for input and output
    base_idx = ((b * C + c_start) * H + h) * W + w
    inp = tl.load(input_ptr + base_idx, mask=mask, other=0.0)

    # ------------------------------------------------------------------
    # 1. Softmax along channel dimension
    # ------------------------------------------------------------------
    # Compute max for numerical stability
    max_val = tl.max(inp, axis=0)
    # Subtract max and exponentiate
    exp_val = tl.exp(inp - max_val)
    # Sum of exponentials
    sum_exp = tl.sum(exp_val, axis=0)
    # Softmax probabilities
    softmax = exp_val / sum_exp

    # ------------------------------------------------------------------
    # 2. Add bias
    # ------------------------------------------------------------------
    out = softmax + bias

    # ------------------------------------------------------------------
    # 3. Scale
    # ------------------------------------------------------------------
    out = out * scaling_factor

    # ------------------------------------------------------------------
    # 4. Sigmoid
    # ------------------------------------------------------------------
    out = 1.0 / (1.0 + tl.exp(-out))

    # Store result
    tl.store(output_ptr + base_idx, out, mask=mask)


# ------------------------------------------------------------------
# Helper function that launches the Triton kernel
# ------------------------------------------------------------------
def triton_softmax_add_scale_sigmoid(input: torch.Tensor,
                                     bias: torch.Tensor,
                                     scaling_factor: float) -> torch.Tensor:
    """
    input: [B, C, H, W] float32/float16/bfloat16 tensor on CUDA
    bias:  [C, 1, 1] broadcastable bias
    scaling_factor: scalar
    """
    assert input.is_cuda and bias.is_cuda, "Input and bias must be CUDA tensors."
    assert input.dtype == bias.dtype, "Data types must match."
    B, C, H, W = input.shape
    out = torch.empty_like(input)

    # Grid: one program per (b, h, w)
    grid = lambda meta: (B, H, W)

    _softmax_add_scale_sigmoid_kernel[grid](
        input,
        bias,
        out,
        B, C, H, W,
        scaling_factor=scaling_factor,
        BLOCK_SIZE=256,
    )
    return out


# ------------------------------------------------------------------
# Optimized model that uses the Triton fused kernel
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model that performs ConvTranspose2d followed by a fused
    softmax + bias addition + scaling + sigmoid using Triton.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 output_padding: int,
                 bias_shape: tuple,
                 scaling_factor: float):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        # Initialize bias with required shape
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Transposed convolution
        x = self.conv_transpose(x)
        # 2. Fused softmax + bias + scaling + sigmoid
        x = triton_softmax_add_scale_sigmoid(
            x, self.bias, self.scaling_factor
        )
        return x