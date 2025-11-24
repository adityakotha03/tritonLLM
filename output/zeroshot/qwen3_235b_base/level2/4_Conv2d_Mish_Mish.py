import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def mish_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute tanh(softplus(x)) = tanh(ln(1 + exp(x)))
    # softplus = ln(1 + exp(x))
    softplus = tl.log(1.0 + tl.exp(x))
    # Avoid overflow by clamping x before exp, but let's assume input is reasonable
    # More stable: for x < -20, softplus ≈ 0; for x > 20, softplus ≈ x. Not implemented here for simplicity.
    tanh_sp = tl.tanh(softplus)
    result = x * tanh_sp

    tl.store(out_ptr + offsets, result, mask=mask)


def triton_mish(x):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    # Autotune block size
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    mish_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


@triton.jit
def conv2d_mish_mish_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch, out_ch, out_h, out_w, in_ch, in_h, in_w, k, s, p,
    stride_row, stride_col,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # 2D tiled GEMM using implicit Tensor Cores via dot_product
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Pointers are in row-major order
    # x: [batch, in_ch, in_h, in_w] -> view as [batch * in_h_padded * in_w_padded, in_ch * k * k]
    # weight: [out_ch, in_ch, k, k] -> [out_ch, in_ch * k * k]
    # Output: [batch, out_ch, out_h, out_w]

    # Handle output spatial dimensions
    total_output_elements = batch * out_ch * out_h * out_w
    out_row_stride = out_ch * out_h * out_w
    out_col_stride = out_h * out_w

    # We flatten output into [batch * out_ch, out_h * out_w]
    output_tiles_per_cta = BLOCK_SIZE_M * BLOCK_SIZE_N
    num_tiles = (total_output_elements + output_tiles_per_cta - 1) // output_tiles_per_cta
    if pid_m * BLOCK_SIZE_M >= total_output_elements:
        return

    # Each block handles a tile of [BLOCK_SIZE_M, BLOCK_SIZE_N]
    # We map linear index to (b*c, h*w)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Not used per se, but we need to index into im2col implicitly
    # Instead, we use a fused im2col + GEMM + activation approach

    # This is a simplified version: we do full im2col in shared memory is too complex
    # Instead, we use a tiling strategy over output pixels and input channels

    # Given complexity, we instead opt for fusing Conv + Mish + Mish in a naive outer loop?
    # But that won't be efficient.

    # Alternative: Use existing cuDNN for Conv, then fuse the two Mish activations
    # Since two consecutive Mish calls are redundant, we can fuse them into one.
    # However, Mish is idempotent? No: mish(mish(x)) ≠ mish(x)
    # So we must compute: y = mish(mish(x))

    # But note: two Mish in a row is equivalent to one Mish applied twice.
    # We can fuse them into a single kernel that does: mish(mish(x))

    # So we don't fuse conv, but we fuse the two Mish calls.
    # Let's change strategy: keep PyTorch Conv2d, but replace the two Mish calls with one fused Mish-Mish kernel.
    pass


# Instead, we keep Conv2d as-is (cuDNN is highly optimized), but fuse the two Mish calls
@triton.jit
def mish_mish_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # First mish: x1 = x * tanh(softplus(x))
    softplus1 = tl.log(1.0 + tl.exp(x))
    x1 = x * tl.tanh(softplus1)

    # Second mish: x2 = x1 * tanh(softplus(x1))
    softplus2 = tl.log(1.0 + tl.exp(x1))
    x2 = x1 * tl.tanh(softplus2)

    tl.store(out_ptr + offsets, x2, mask=mask)


def triton_mish_mish(x):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    mish_mish_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that uses fused Mish-Mish activation via Triton.
    The Conv2d is kept as-is because cuDNN implementation is already highly optimized.
    However, two consecutive Mish activations are fused into a single kernel to reduce memory bandwidth.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = triton_mish_mish(x)
        return x