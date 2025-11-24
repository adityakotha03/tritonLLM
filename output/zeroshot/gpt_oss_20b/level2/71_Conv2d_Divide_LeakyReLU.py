import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Triton kernel that performs:  (x * w + b) / divisor -> leaky_relu
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64},
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128},
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 256},
                      num_warps=8),
    ],
    key=['M', 'N', 'K', 'divisor', 'in_channels', 'kernel_h', 'kernel_w', 'stride_h', 'stride_w'],
)
@triton.jit
def conv_div_leaky_kernel(
    # Pointers to input, weight, bias and output
    input_ptr, weight_ptr, bias_ptr, out_ptr,
    # Shape parameters
    B, C_in, H_in, W_in,
    C_out, H_out, W_out,
    K_h, K_w,
    stride_h, stride_w,
    # Additional params
    divisor: tl.constexpr,
    in_channels: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h_const: tl.constexpr,
    stride_w_const: tl.constexpr,
    # Block size
    BLOCK_SIZE_M: tl.constexpr,   # H_out
    BLOCK_SIZE_N: tl.constexpr,   # W_out
    BLOCK_SIZE_K: tl.constexpr,   # C_in*K_h*K_w
):
    """
    A Triton kernel that computes
          out[b, c_out, h_out, w_out] =
              leaky_relu( ( Σ_{c_in, kh, kw} x[b, c_in, h_in+kh, w_in+kw] * w[c_out, c_in, kh, kw]
                           + bias[c_out] ) / divisor,
                         negative_slope=0.01 )
    The kernel is tiled over the output height (M) and width (N).
    """
    pid_m = tl.program_id(0)  # block row (output height)
    pid_n = tl.program_id(1)  # block column (output width)

    # Compute the output region this block will process
    h_start = pid_m * BLOCK_SIZE_M
    w_start = pid_n * BLOCK_SIZE_N

    # Prepare thread indices within the block
    tid_m = tl.arange(0, BLOCK_SIZE_M)
    tid_n = tl.arange(0, BLOCK_SIZE_N)

    # Compute global output coordinates for each thread
    h_out = h_start + tid_m[:, None]
    w_out = w_start + tid_n[None, :]

    # Mask to handle boundary conditions
    mask_m = h_out < H_out
    mask_n = w_out < W_out
    mask = mask_m[:, None] & mask_n[None, :]

    # Initialize accumulator for each output element
    # Shape: (BLOCK_SIZE_M, BLOCK_SIZE_N, C_out)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N, C_out), dtype=tl.float32)

    # Iterate over input channels and kernel spatial dimensions
    for c_in in range(in_channels):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Input coordinates (broadcasted over block)
                h_in = h_out * stride_h_const + kh
                w_in = w_out * stride_w_const + kw
                in_mask = (h_in < H_in) & (w_in < W_in)
                in_mask = in_mask[:, None] & in_mask[None, :]
                # Load input patch
                in_offset = (0 * C_in * H_in * W_in) + \
                            (c_in * H_in * W_in) + \
                            (h_in * W_in) + w_in
                inp = tl.load(input_ptr + in_offset, mask=in_mask, other=0.0)
                # Load weight
                w_offset = (0 * C_out * C_in * K_h * K_w) + \
                           (c_out * C_in * K_h * K_w) + \
                           (c_in * K_h * K_w) + (kh * K_w) + kw
                wgt = tl.load(weight_ptr + w_offset)
                # Accumulate
                acc += inp[:, :, None] * wgt

    # Add bias
    bias = tl.load(bias_ptr)
    acc += bias[None, None, :]

    # Divide by divisor and apply LeakyReLU
    acc = acc / divisor
    acc = tl.where(acc > 0, acc, acc * 0.01)

    # Store results
    out_offset = (0 * C_out * H_out * W_out) + \
                 (c_out * H_out * W_out) + \
                 (h_out * W_out) + w_out
    tl.store(out_ptr + out_offset, acc, mask=mask[..., None])


# ------------------------------------------------------------------
# Helper functions to launch the Triton kernel
# ------------------------------------------------------------------
def triton_conv_div_leaky(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    divisor: float,
    stride: int = 1,
    padding: int = 0,
):
    """
    Performs convolution with bias, division and leaky_relu in a single
    Triton kernel. The input is expected to be of shape
    (B, C_in, H_in, W_in). The weight is of shape
    (C_out, C_in, K_h, K_w). Bias is of shape (C_out,).
    """
    B, C_in, H_in, W_in = input.shape
    C_out, _, K_h, K_w = weight.shape
    stride_h = stride_w = stride

    # Compute output spatial size
    H_out = (H_in + 2 * padding - K_h) // stride_h + 1
    W_out = (W_in + 2 * padding - K_w) // stride_w + 1

    # Pad input if needed
    if padding > 0:
        padded = torch.nn.functional.pad(input, (padding, padding, padding, padding))
    else:
        padded = input

    # Allocate output tensor
    out = torch.empty((B, C_out, H_out, W_out), device=input.device, dtype=input.dtype)

    # Prepare grid: one program per output tile
    grid = lambda meta: (
        (H_out + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M'],
        (W_out + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N'],
    )

    conv_div_leaky_kernel[grid](
        padded, weight, bias, out,
        B, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        divisor,
        C_in, K_h, K_w, stride_h, stride_w,
        BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64,
    )

    return out


# ------------------------------------------------------------------
# Optimized model using Triton kernels
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model that replaces the standard conv/divide/LeakyReLU
    chain with a single Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)
        self.divisor = divisor

    def forward(self, x):
        # Use Triton fused kernel
        # Extract weight and bias from the conv layer
        weight = self.conv.weight  # shape (C_out, C_in, K_h, K_w)
        bias = self.conv.bias      # shape (C_out,)
        return triton_conv_div_leaky(x, weight, bias, self.divisor)

# ------------------------------------------------------------------
# Example usage (not part of the required code)
# ------------------------------------------------------------------
# if __name__ == "__main__":
#     model = ModelNew(8, 64, 3, 2.0).cuda()
#     inp = torch.rand(128, 8, 128, 128).cuda()
#     out = model(inp)